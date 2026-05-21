"""Calibration tests: PD → score map, clipping, decision bands, empirical refinement."""

from __future__ import annotations

import numpy as np
import pytest

from novascore.calibration import (
    BAND_DESCRIPTIONS,
    SCORE_MAX,
    SCORE_MIN,
    CalibrationParams,
    apply_calibration,
    decision_band,
    empirical_refinement,
    pd_to_score,
    solve_score_params,
)


def test_solve_score_params_hits_default_anchors():
    A, B = solve_score_params(pd_anchors=(0.01, 0.20), score_anchors=(900, 650))
    # At PD=0.01 the score is exactly 900; at PD=0.20 exactly 650.
    s1 = pd_to_score(0.01, A, B)
    s2 = pd_to_score(0.20, A, B)
    assert s1 == pytest.approx(900, abs=1e-3)
    assert s2 == pytest.approx(650, abs=1e-3)


def test_pd_to_score_clips_to_range():
    A, B = solve_score_params()
    # PD = 0.0 should clip to SCORE_MAX (perfect applicant).
    assert float(pd_to_score(0.0, A, B)) == SCORE_MAX
    # PD = 1.0 should clip to SCORE_MIN (certain default).
    assert float(pd_to_score(1.0, A, B)) == SCORE_MIN


def test_pd_to_score_monotonic_decreasing():
    A, B = solve_score_params()
    pds = np.linspace(0.001, 0.999, 25)
    scores = pd_to_score(pds, A, B)
    # Diff should be non-positive everywhere (lower PD → higher score, after clip).
    diffs = np.diff(scores)
    assert np.all(diffs <= 1e-6)


def test_pd_to_score_round_trip_at_anchors():
    """Inverting the formula recovers the anchor PDs."""
    A, B = solve_score_params(pd_anchors=(0.05, 0.30), score_anchors=(880, 600))
    # Recover anchors from formula.
    s1 = pd_to_score(0.05, A, B)
    s2 = pd_to_score(0.30, A, B)
    assert s1 == pytest.approx(880, abs=1e-3)
    assert s2 == pytest.approx(600, abs=1e-3)


def test_decision_band_boundaries():
    assert decision_band(950) == "Platinum"
    assert decision_band(800) == "Platinum"
    assert decision_band(799.99) == "Gold"
    assert decision_band(700) == "Gold"
    assert decision_band(699.99) == "Silver"
    assert decision_band(600) == "Silver"
    assert decision_band(599.99) == "Bronze"
    assert decision_band(300) == "Bronze"


def test_decision_band_descriptions_present():
    for band in ("Platinum", "Gold", "Silver", "Bronze"):
        desc = BAND_DESCRIPTIONS[band]
        assert isinstance(desc, str) and len(desc) > 10


def test_calibration_params_roundtrip_dict():
    params = CalibrationParams(A=540.0, B=78.0, a=1.2, b=-1.0)
    d = params.to_dict()
    restored = CalibrationParams.from_dict(d)
    assert restored == params


def test_apply_calibration_orientation():
    """After empirical refinement, low PDs should yield high scores, not low ones.

    This is a regression test for the Phase-5 calibration-direction bug where
    `q_low` was anchored to `logit(p600)` instead of `logit(p800)`.
    """
    rng = np.random.default_rng(0)
    # Synthetic PD distribution skewed toward small values, similar to a
    # well-calibrated default-prediction model on a 10%-positive dataset.
    probs = rng.beta(2, 12, size=2000)
    A, B = solve_score_params()
    a, b = empirical_refinement(probs, A, B)
    calib = CalibrationParams(A=A, B=B, a=a, b=b)

    low_pds = np.array([0.01, 0.03, 0.05])
    high_pds = np.array([0.40, 0.60, 0.80])
    low_scores = apply_calibration(low_pds, calib)
    high_scores = apply_calibration(high_pds, calib)
    # Strict ordering: every low-PD score should beat every high-PD score.
    assert low_scores.min() > high_scores.max(), (
        f"Inverted calibration detected: low={low_scores}, high={high_scores}"
    )


def test_empirical_refinement_anchors_quantiles_to_score_targets():
    """The 20th-pct PD should rescale to ~p800; 80th-pct → ~p600."""
    rng = np.random.default_rng(1)
    probs = rng.beta(2, 12, size=5000)
    A, B = solve_score_params()
    a, b = empirical_refinement(probs, A, B, q_low=0.20, q_high=0.80)
    calib = CalibrationParams(A=A, B=B, a=a, b=b)

    q20 = float(np.quantile(probs, 0.20))
    q80 = float(np.quantile(probs, 0.80))
    score_at_q20 = float(apply_calibration(np.array([q20]), calib)[0])
    score_at_q80 = float(apply_calibration(np.array([q80]), calib)[0])

    # Within 2 score points of the targets (numerical wiggle from clip + refine).
    assert abs(score_at_q20 - 800) < 2.5
    assert abs(score_at_q80 - 600) < 2.5
