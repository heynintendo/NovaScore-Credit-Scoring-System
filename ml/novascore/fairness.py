"""Group-fairness metrics and threshold-based mitigation.

Metrics provided:
- demographic_parity_ratio: min/max ratio of P(prediction=1) across groups.
- disparate_impact_ratio: alias for demographic_parity_ratio (the 4/5 rule).
- delta_tpr / delta_fpr: max minus min rate across groups.
- equalized_odds_difference: max(|ΔTPR|, |ΔFPR|).
- compute_all_metrics: tabulates the above per protected attribute.

Mitigation:
- optimize_thresholds_per_group: grid-search per-group classification thresholds
  so all groups hit the same target TPR (within 2 percentage-point tolerance).
  These shifts translate into score adjustments at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _rate(y: np.ndarray) -> float:
    return float(np.mean(y)) if len(y) else float("nan")


def demographic_parity_ratio(y_pred: np.ndarray, group: np.ndarray) -> float:
    """min_g P(ŷ=1 | g) / max_g P(ŷ=1 | g). Closer to 1 is fairer."""
    y_pred = np.asarray(y_pred).astype(int)
    group = np.asarray(group)
    rates = []
    for g in np.unique(group):
        m = group == g
        if m.sum() == 0:
            continue
        rates.append(_rate(y_pred[m]))
    if not rates or max(rates) == 0:
        return float("nan")
    return float(min(rates) / max(rates))


def disparate_impact_ratio(y_pred: np.ndarray, group: np.ndarray) -> float:
    """Alias for demographic_parity_ratio (EEOC's 4/5 rule)."""
    return demographic_parity_ratio(y_pred, group)


def _per_group_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    target: int,
) -> dict[object, float]:
    """Per-group rate of y_pred=1 among examples where y_true == target."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    group = np.asarray(group)
    out: dict[object, float] = {}
    for g in np.unique(group):
        m = (group == g) & (y_true == target)
        if m.sum() == 0:
            continue
        out[g] = _rate(y_pred[m])
    return out


def per_group_tpr(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> dict[object, float]:
    """P(ŷ=1 | y=1, g) per group."""
    return _per_group_rate(y_true, y_pred, group, target=1)


def per_group_fpr(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> dict[object, float]:
    """P(ŷ=1 | y=0, g) per group."""
    return _per_group_rate(y_true, y_pred, group, target=0)


def delta_tpr(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> float:
    """max_g TPR_g − min_g TPR_g. 0 means perfect equal-opportunity."""
    rates = list(per_group_tpr(y_true, y_pred, group).values())
    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


def delta_fpr(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> float:
    """max_g FPR_g − min_g FPR_g."""
    rates = list(per_group_fpr(y_true, y_pred, group).values())
    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


def equalized_odds_difference(
    y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray
) -> float:
    """max(ΔTPR, ΔFPR). 0 means perfect equalized odds."""
    return float(max(delta_tpr(y_true, y_pred, group), delta_fpr(y_true, y_pred, group)))


def compute_all_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    groups_dict: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Tabulate fairness metrics across multiple protected attributes.

    Args:
        y_true: binary labels.
        p_pred: predicted probabilities (PD).
        groups_dict: {attribute_name: group_array}, e.g. {"gender": [...], "age_bucket": [...]}.
        threshold: cutoff at which p_pred becomes a hard prediction.

    Returns: DataFrame with one row per protected attribute, columns:
        attribute, dp_ratio, di_ratio, delta_tpr, delta_fpr, eq_odds_diff.
    """
    y_pred_hard = (np.asarray(p_pred) >= threshold).astype(int)
    rows = []
    for name, group in groups_dict.items():
        rows.append(
            {
                "attribute": name,
                "demographic_parity_ratio": demographic_parity_ratio(y_pred_hard, group),
                "disparate_impact_ratio": disparate_impact_ratio(y_pred_hard, group),
                "delta_tpr": delta_tpr(y_true, y_pred_hard, group),
                "delta_fpr": delta_fpr(y_true, y_pred_hard, group),
                "equalized_odds_difference": equalized_odds_difference(y_true, y_pred_hard, group),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ThresholdMap:
    """Per-group classification thresholds discovered by `optimize_thresholds_per_group`."""

    target_tpr: float
    thresholds: dict[object, float]
    achieved_tpr: dict[object, float]
    default_threshold: float = 0.5

    def threshold_for(self, group: object) -> float:
        return float(self.thresholds.get(group, self.default_threshold))

    def to_dict(self) -> dict[str, object]:
        return {
            "target_tpr": float(self.target_tpr),
            "default_threshold": float(self.default_threshold),
            "thresholds": {str(k): float(v) for k, v in self.thresholds.items()},
            "achieved_tpr": {str(k): float(v) for k, v in self.achieved_tpr.items()},
        }


def optimize_thresholds_per_group(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    group: np.ndarray,
    target_tpr: float = 0.8,
    n_grid: int = 201,
    tolerance: float = 0.02,
) -> ThresholdMap:
    """Find per-group thresholds achieving `target_tpr` to within `tolerance`.

    For each unique group with at least one positive example, grid-search
    thresholds in [0, 1] (n_grid points) and select the one closest to
    target_tpr. Groups without positives keep the default 0.5.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    group = np.asarray(group)
    grid = np.linspace(0.0, 1.0, n_grid)

    thresholds: dict[object, float] = {}
    achieved: dict[object, float] = {}
    for g in np.unique(group):
        m = group == g
        y_g = y_true[m]
        p_g = p_pred[m]
        pos_mask = y_g == 1
        n_pos = int(pos_mask.sum())
        if n_pos == 0:
            thresholds[g] = 0.5
            achieved[g] = float("nan")
            continue
        best_t = 0.5
        best_diff = float("inf")
        best_tpr = 0.0
        for t in grid:
            tpr = float(((p_g >= t) & pos_mask).sum()) / n_pos
            diff = abs(tpr - target_tpr)
            # Prefer the *highest* threshold that meets tolerance (least permissive).
            if diff < best_diff - 1e-12 or (
                diff <= best_diff + 1e-12 and t > best_t and diff <= tolerance
            ):
                best_t, best_diff, best_tpr = float(t), diff, tpr
        thresholds[g] = best_t
        achieved[g] = best_tpr
    return ThresholdMap(target_tpr=target_tpr, thresholds=thresholds, achieved_tpr=achieved)


def score_adjustment_from_threshold(
    group_threshold: float,
    A: float,
    B: float,
    default_threshold: float = 0.5,
) -> float:
    """Convert a per-group classification threshold to an additive score shift.

    For the score map `score = A - B * logit(PD)`, shifting the decision PD
    threshold from `default_threshold` (e.g. 0.5) up to `group_threshold` is
    equivalent to adding `B * (logit(group_threshold) - logit(default_threshold))`
    to every member of that group's score. Positive shifts mean the group's
    members get *higher* scores under fairness mitigation.
    """
    default_threshold = float(np.clip(default_threshold, 1e-6, 1 - 1e-6))
    group_threshold = float(np.clip(group_threshold, 1e-6, 1 - 1e-6))
    logit_def = float(np.log(default_threshold / (1 - default_threshold)))
    logit_grp = float(np.log(group_threshold / (1 - group_threshold)))
    return float(B * (logit_grp - logit_def))
