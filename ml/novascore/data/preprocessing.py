"""Data ingestion, column normalization, windowing, labeling, and tabular features.

Mirrors the legacy Colab pipeline's preprocessing exactly so existing decisions
(window size, label rule, top-k cardinality cutoffs) are preserved. The only
substantive change is that everything is now a pure function and returns the
artifacts needed for downstream modules (anchor_end, dominant city, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import WINDOW_DAYS

BAD_STATUSES: frozenset[str] = frozenset(
    {
        "chargeback",
        "default",
        "failed",
        "fraud",
        "reversed",
        "disputed",
        "late",
        "bounced",
        "cancelled",
    }
)


def is_bad(status: object) -> bool:
    """True if a transaction status string contains any default-like keyword."""
    if pd.isna(status):
        return False
    s = str(status).strip().lower()
    return any(bad in s for bad in BAD_STATUSES)


def load_parquets(
    trips_path: str | Path, txns_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read trips + txns parquets and lower-case all column names."""
    trips = pd.read_parquet(trips_path)
    txns = pd.read_parquet(txns_path)
    trips.columns = [c.strip().lower() for c in trips.columns]
    txns.columns = [c.strip().lower() for c in txns.columns]
    return trips, txns


def parse_datetimes(trips: pd.DataFrame, txns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build `trip_start_ts`, `trip_end_ts`, `transaction_dt` on the input frames."""
    trips = trips.copy()
    txns = txns.copy()
    d = pd.to_datetime(trips["trip_date"], errors="coerce")
    trips["trip_start_ts"] = d + pd.to_timedelta(trips["trip_start_time"], errors="coerce")
    trips["trip_end_ts"] = d + pd.to_timedelta(trips["trip_end_time"], errors="coerce")
    txns["transaction_dt"] = pd.to_datetime(
        txns["transaction_date"].astype(str) + " " + txns["transaction_time"].astype(str),
        errors="coerce",
    )

    for col in (
        "trip_duration",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "trip_rating",
        "safety_score",
    ):
        if col in trips.columns:
            trips[col] = pd.to_numeric(trips[col], errors="coerce")
    if "cancellation_flag" in trips.columns:
        trips["cancellation_flag"] = (
            pd.to_numeric(trips["cancellation_flag"], errors="coerce").fillna(0).astype(int)
        )
    if "incident_flag" in trips.columns:
        trips["incident_flag"] = trips["incident_flag"].fillna(False).astype(bool)
    for col in ("transaction_amount", "balance_after_transaction"):
        if col in txns.columns:
            txns[col] = pd.to_numeric(txns[col], errors="coerce")
    return trips, txns


def apply_window(
    trips: pd.DataFrame,
    txns: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Restrict to the most recent `window_days` ending at the latest timestamp."""
    max_dt = pd.Series(
        [
            trips["trip_end_ts"].max(),
            trips["trip_start_ts"].max(),
            txns["transaction_dt"].max(),
        ]
    ).max()
    anchor_end = pd.to_datetime(max_dt)
    anchor_start = anchor_end - pd.Timedelta(days=window_days)
    trips_w = trips[
        (trips["trip_start_ts"] >= anchor_start) & (trips["trip_start_ts"] <= anchor_end)
    ].copy()
    txns_w = txns[
        (txns["transaction_dt"] >= anchor_start) & (txns["transaction_dt"] <= anchor_end)
    ].copy()
    return trips_w, txns_w, anchor_end


def compute_labels(txns_w: pd.DataFrame) -> pd.DataFrame:
    """User-level binary label: 1 if any transaction in window is bad-statused."""
    flagged = txns_w.assign(is_bad=txns_w["transaction_status"].map(is_bad))
    out = flagged.groupby("user_id", as_index=False)["is_bad"].max().rename(columns={"is_bad": "y"})
    out["y"] = out["y"].astype(int)
    return out


def topk_values(df: pd.DataFrame, col: str, k: int) -> list[str]:
    """Return the top-k most common string values for a categorical column."""
    if col not in df.columns:
        return []
    return df[col].astype(str).value_counts().nlargest(k).index.tolist()


def topk_ratio(
    df: pd.DataFrame,
    col: str,
    values: list[str],
    prefix: str,
) -> pd.DataFrame:
    """For each user, fraction of rows where `col` equals each of `values`."""
    outs: list[pd.Series] = []
    if not values or col not in df.columns:
        return pd.DataFrame(index=df["user_id"].unique() if "user_id" in df.columns else None)
    for v in values:
        name = f"{prefix}{v}".lower().replace(" ", "_")
        s = (
            df.assign(_t=(df[col].astype(str) == v).astype(int))
            .groupby("user_id")["_t"]
            .mean()
            .rename(name)
        )
        outs.append(s)
    return pd.concat(outs, axis=1) if outs else pd.DataFrame()


def compute_tabular_features(
    trips_w: pd.DataFrame, txns_w: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Per-user tabular feature matrix (trips + txns aggregations + top-k mixes).

    Returns: (features_df, learned_topk) where `learned_topk` maps each
    categorical column to the list of top-k values used at training time.
    Inference must apply the same `topk_ratio(df, col, learned_topk[col], ...)`
    to produce a column-identical feature vector.
    """
    trip_g = trips_w.groupby("user_id", dropna=False)
    incident_agg = (
        ("incident_flag", "mean")
        if "incident_flag" in trips_w.columns
        else ("cancellation_flag", "mean")
    )
    trip_agg = trip_g.agg(
        trips_count=("trip_id", "count"),
        trip_dur_mean=("trip_duration", "mean"),
        trip_dur_sum=("trip_duration", "sum"),
        trip_dist_sum=("trip_distance", "sum"),
        fare_sum=("fare_amount", "sum"),
        tip_sum=("tip_amount", "sum"),
        rating_mean=("trip_rating", "mean"),
        safety_mean=("safety_score", "mean"),
        cancel_rate=("cancellation_flag", "mean"),
        incident_rate=incident_agg,
    ).reset_index()

    learned: dict[str, list[str]] = {
        "payment_method": topk_values(trips_w, "payment_method", k=4),
        "route_type": topk_values(trips_w, "route_type", k=3),
        "device_channel": topk_values(txns_w, "device_channel", k=3),
        "merchant_category": topk_values(txns_w, "merchant_category", k=5),
    }
    pm = topk_ratio(trips_w, "payment_method", learned["payment_method"], prefix="pay_")
    rt = topk_ratio(trips_w, "route_type", learned["route_type"], prefix="route_")
    tab_trip = trip_agg.merge(pm, left_on="user_id", right_index=True, how="left").merge(
        rt, left_on="user_id", right_index=True, how="left"
    )

    txn_g = txns_w.groupby("user_id", dropna=False)
    txn_agg = txn_g.agg(
        txn_count=("transaction_id", "count"),
        txn_amt_sum=("transaction_amount", "sum"),
        txn_amt_mean=("transaction_amount", "mean"),
        txn_amt_std=("transaction_amount", "std"),
        bal_after_mean=("balance_after_transaction", "mean"),
    ).reset_index()
    dc = topk_ratio(txns_w, "device_channel", learned["device_channel"], prefix="devc_")
    mc = topk_ratio(txns_w, "merchant_category", learned["merchant_category"], prefix="mcat_")
    tab_txn = txn_agg.merge(dc, left_on="user_id", right_index=True, how="left").merge(
        mc, left_on="user_id", right_index=True, how="left"
    )

    tab = tab_trip.merge(tab_txn, on="user_id", how="outer").fillna(0.0)
    return tab, learned


def compute_user_city_group(
    txns_w: pd.DataFrame,
    user_ids: pd.Series,
    top_k: int = 10,
) -> pd.DataFrame:
    """Per-user dominant city, collapsed to {top-k cities} ∪ {"other"}."""
    if "city" not in txns_w.columns:
        return pd.DataFrame({"user_id": user_ids, "city_grp": "all"})
    dom = (
        txns_w.assign(city=txns_w["city"].astype(str))
        .groupby(["user_id", "city"])
        .size()
        .reset_index(name="n")
        .sort_values(["user_id", "n"], ascending=[True, False])
        .drop_duplicates("user_id")[["user_id", "city"]]
    )
    top_cities = txns_w["city"].astype(str).value_counts().nlargest(top_k).index.tolist()
    dom["city_grp"] = np.where(dom["city"].isin(top_cities), dom["city"], "other")
    out = pd.DataFrame({"user_id": user_ids}).merge(
        dom[["user_id", "city_grp"]], on="user_id", how="left"
    )
    out["city_grp"] = out["city_grp"].fillna("other")
    return out
