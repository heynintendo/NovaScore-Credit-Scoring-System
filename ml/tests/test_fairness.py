"""Fairness module tests: handcrafted groups with known TPR / FPR."""

from __future__ import annotations

import numpy as np
import pytest

from novascore.fairness import (
    ThresholdMap,
    compute_all_metrics,
    delta_fpr,
    delta_tpr,
    demographic_parity_ratio,
    disparate_impact_ratio,
    equalized_odds_difference,
    optimize_thresholds_per_group,
    per_group_fpr,
    per_group_tpr,
    score_adjustment_from_threshold,
)

# Two groups, both 10 examples. Hand-built so the per-group rates are obvious.
# Group A: 5 positives (4 TP, 1 FN), 5 negatives (1 FP, 4 TN). TPR=0.8, FPR=0.2.
# Group B: 5 positives (2 TP, 3 FN), 5 negatives (0 FP, 5 TN). TPR=0.4, FPR=0.0.
_GROUP_A_Y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
_GROUP_A_P = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]  # 4 TP, 1 FN, 1 FP
_GROUP_B_Y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
_GROUP_B_P = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 2 TP, 3 FN, 0 FP
Y_TRUE = np.array(_GROUP_A_Y + _GROUP_B_Y)
Y_PRED = np.array(_GROUP_A_P + _GROUP_B_P)
GROUP = np.array(["A"] * 10 + ["B"] * 10)


def test_per_group_tpr_known_rates():
    tpr = per_group_tpr(Y_TRUE, Y_PRED, GROUP)
    assert tpr["A"] == pytest.approx(0.8)
    assert tpr["B"] == pytest.approx(0.4)


def test_per_group_fpr_known_rates():
    fpr = per_group_fpr(Y_TRUE, Y_PRED, GROUP)
    assert fpr["A"] == pytest.approx(0.2)
    assert fpr["B"] == pytest.approx(0.0)


def test_delta_tpr_is_max_minus_min():
    assert delta_tpr(Y_TRUE, Y_PRED, GROUP) == pytest.approx(0.4)


def test_delta_fpr_is_max_minus_min():
    assert delta_fpr(Y_TRUE, Y_PRED, GROUP) == pytest.approx(0.2)


def test_equalized_odds_difference_is_max_of_deltas():
    assert equalized_odds_difference(Y_TRUE, Y_PRED, GROUP) == pytest.approx(0.4)


def test_demographic_parity_ratio_hand_example():
    # Group A predicted-positive rate = 5/10 = 0.5
    # Group B predicted-positive rate = 2/10 = 0.2
    # ratio = min / max = 0.2 / 0.5 = 0.4
    assert demographic_parity_ratio(Y_PRED, GROUP) == pytest.approx(0.4)
    assert disparate_impact_ratio(Y_PRED, GROUP) == pytest.approx(0.4)


def test_compute_all_metrics_returns_frame():
    df = compute_all_metrics(
        Y_TRUE,
        Y_PRED.astype(float),  # soft probs not used — threshold path triggers
        {"group": GROUP},
        threshold=0.5,
    )
    assert list(df.columns) == [
        "attribute",
        "demographic_parity_ratio",
        "disparate_impact_ratio",
        "delta_tpr",
        "delta_fpr",
        "equalized_odds_difference",
    ]
    row = df.iloc[0]
    assert row["delta_tpr"] == pytest.approx(0.4)
    assert row["delta_fpr"] == pytest.approx(0.2)


def test_delta_tpr_zero_when_one_group_has_no_positives():
    """Group with no positives is skipped; metric uses remaining groups only."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    group = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    # Only group A has positives — fewer than 2 groups for comparison → 0.
    assert delta_tpr(y_true, y_pred, group) == 0.0


def test_optimize_thresholds_per_group_equalizes_tpr():
    rng = np.random.default_rng(42)
    n = 600
    # Two groups whose label-vs-probability relationships diverge.
    # Group A: positives have prob ~ Beta(8, 4); negatives ~ Beta(2, 8).
    # Group B: positives have prob ~ Beta(4, 4); negatives ~ Beta(2, 8).
    # At a fixed global threshold, A would have much higher TPR than B.
    grp = np.repeat(["A", "B"], n // 2)
    y = np.tile(np.concatenate([np.ones(n // 4), np.zeros(n // 4)]), 2).astype(int)
    pos_probs_a = rng.beta(8, 4, n // 4)
    neg_probs_a = rng.beta(2, 8, n // 4)
    pos_probs_b = rng.beta(4, 4, n // 4)
    neg_probs_b = rng.beta(2, 8, n // 4)
    p = np.concatenate([pos_probs_a, neg_probs_a, pos_probs_b, neg_probs_b])

    tm = optimize_thresholds_per_group(y, p, grp, target_tpr=0.8, tolerance=0.03)
    assert isinstance(tm, ThresholdMap)
    # Every group's achieved TPR should be near the target.
    for grp_id, tpr in tm.achieved_tpr.items():
        if not np.isnan(tpr):
            assert abs(tpr - 0.8) <= 0.05, f"group {grp_id}: TPR {tpr:.3f} far from target 0.80"


def test_optimize_thresholds_handles_zero_positives():
    """Groups with no positives keep the default threshold and NaN TPR."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    grp = np.array(["x", "x", "x", "y", "y", "y"])
    tm = optimize_thresholds_per_group(y, p, grp, target_tpr=0.8)
    assert tm.thresholds["x"] == 0.5  # default kept (no positives in group x)
    assert np.isnan(tm.achieved_tpr["x"])


def test_score_adjustment_sign_follows_threshold_direction():
    """Stricter group threshold (lower PD) → negative score shift."""
    # B = 80 for easy arithmetic.
    # default = 0.5 (logit 0). Group threshold 0.3 (logit ≈ -0.847).
    adj = score_adjustment_from_threshold(0.3, A=540, B=80, default_threshold=0.5)
    assert adj < 0
    # And a group threshold ABOVE 0.5 should give a positive shift.
    adj2 = score_adjustment_from_threshold(0.7, A=540, B=80, default_threshold=0.5)
    assert adj2 > 0


def test_threshold_map_serialization():
    tm = ThresholdMap(
        target_tpr=0.8,
        thresholds={"x": 0.3, "y": 0.6},
        achieved_tpr={"x": 0.78, "y": 0.82},
        default_threshold=0.5,
    )
    d = tm.to_dict()
    assert d["target_tpr"] == 0.8
    assert d["thresholds"] == {"x": 0.3, "y": 0.6}
    assert d["default_threshold"] == 0.5
