"""Weekly sequence construction for the TCN tower.

Each user's 90-day activity is bucketed into 13 weekly time steps with 9 features
per step: trips, distance, duration, cancels, rating, earnings, spend, txns,
unique merchants. The full sequence tensor is z-score-normalized per feature
across all (user, week) pairs; the means and stds are returned so inference can
apply the identical transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import SEQ_FEATURES, T_WEEKS

SEQ_COLS: tuple[str, ...] = SEQ_FEATURES


@dataclass(frozen=True)
class SequenceScaler:
    """Per-feature z-score parameters (mean, std) used at training time."""

    means: dict[str, float]
    stds: dict[str, float]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, d: dict[str, dict[str, float]]) -> SequenceScaler:
        return cls(means=dict(d["means"]), stds=dict(d["stds"]))


def _week_index_func(anchor_end: pd.Timestamp, t_weeks: int = T_WEEKS):
    end_day = anchor_end.normalize()

    def _wi(ts: pd.Timestamp) -> int:
        delta = (end_day - ts.normalize()).days
        return (t_weeks - 1) - (delta // 7)

    return _wi


def _build_weekly(
    df: pd.DataFrame,
    time_col: str,
    agg_map: dict,
    anchor_end: pd.Timestamp,
    t_weeks: int,
    id_col: str = "user_id",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[id_col, "week_idx", *agg_map.keys()])
    df = df.copy()
    wi = _week_index_func(anchor_end, t_weeks)
    df["week_idx"] = df[time_col].dt.floor("D").map(wi)
    df = df[(df["week_idx"] >= 0) & (df["week_idx"] <= t_weeks - 1)]
    grouped = df.groupby([id_col, "week_idx"])
    sample = next(iter(agg_map.values()))
    if isinstance(sample, tuple):
        # Named aggregation: {"new_col": ("source_col", "func")} -> agg(**agg_map)
        return grouped.agg(**agg_map).reset_index()
    # Column-keyed: {"col": "func"} -> agg(agg_map)
    return grouped.agg(agg_map).reset_index()


def build_sequences(
    trips_w: pd.DataFrame,
    txns_w: pd.DataFrame,
    user_ids: pd.Series,
    anchor_end: pd.Timestamp,
    t_weeks: int = T_WEEKS,
) -> tuple[np.ndarray, SequenceScaler]:
    """Build the (n_users, t_weeks, len(SEQ_COLS)) z-scored sequence tensor."""
    trip_week = _build_weekly(
        trips_w,
        "trip_start_ts",
        dict(
            trips=("trip_id", "count"),
            dist=("trip_distance", "sum"),
            dur=("trip_duration", "sum"),
            cancels=("cancellation_flag", "sum"),
            rating=("trip_rating", "mean"),
            earnings=("fare_amount", "sum"),
        ),
        anchor_end,
        t_weeks,
    )
    txn_week = _build_weekly(
        txns_w,
        "transaction_dt",
        {
            "transaction_amount": "sum",
            "transaction_id": "count",
            "merchant_id": pd.Series.nunique,
        },
        anchor_end,
        t_weeks,
    ).rename(
        columns={
            "transaction_amount": "spend",
            "transaction_id": "txns",
            "merchant_id": "merchants",
        }
    )

    w = pd.merge(trip_week, txn_week, on=["user_id", "week_idx"], how="outer")
    for c in SEQ_COLS:
        if c not in w.columns:
            w[c] = 0.0
    w[list(SEQ_COLS)] = w[list(SEQ_COLS)].fillna(0.0)

    means = {}
    stds = {}
    for c in SEQ_COLS:
        col = w[c].astype(float).to_numpy()
        mu = float(np.nanmean(col))
        sd = float(np.nanstd(col)) + 1e-6
        means[c] = mu
        stds[c] = sd
        w[c] = (w[c] - mu) / sd

    uid2idx = {u: i for i, u in enumerate(user_ids.tolist())}
    x_seq = np.zeros((len(user_ids), t_weeks, len(SEQ_COLS)), dtype="float32")
    for _, row in w.iterrows():
        i = uid2idx.get(row["user_id"])
        t = int(row["week_idx"])
        if i is None or t < 0 or t > (t_weeks - 1):
            continue
        x_seq[i, t, :] = np.asarray([row[c] for c in SEQ_COLS], dtype="float32")
    return x_seq, SequenceScaler(means=means, stds=stds)
