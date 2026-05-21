"""PD → NovaScore calibration and decision-band assignment.

The mapping is a logit-linear scoring function anchored at two PD points:

    score = A - B * logit(PD)

Anchors default to (PD=0.01 -> 900) and (PD=0.20 -> 650) per the NovaScore deck.
The final score is clipped to [SCORE_MIN, SCORE_MAX] = [300, 950] and bucketed
into the four decision bands: Platinum / Gold / Silver / Bronze.

An optional empirical refinement step rescales the predicted PD distribution
so its 20th and 80th percentiles align with the score-anchor PDs (`p600` and
`p800`). This keeps the score distribution well-spread even when the model's
raw probabilities are concentrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import PD_ANCHORS, SCORE_ANCHORS, SCORE_MAX, SCORE_MIN


@dataclass(frozen=True)
class CalibrationParams:
    """Coefficients for the PD -> score map and empirical rescaling."""

    A: float
    B: float
    a: float = 1.0
    b: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"A": self.A, "B": self.B, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "CalibrationParams":
        return cls(A=float(d["A"]), B=float(d["B"]), a=float(d.get("a", 1.0)), b=float(d.get("b", 0.0)))


def solve_score_params(
    pd_anchors: tuple[float, float] = PD_ANCHORS,
    score_anchors: tuple[float, float] = SCORE_ANCHORS,
) -> tuple[float, float]:
    """Return (A, B) such that score = A - B * logit(PD) hits both anchors."""
    p1, p2 = pd_anchors
    s1, s2 = score_anchors
    x1, x2 = math.log(p1 / (1 - p1)), math.log(p2 / (1 - p2))
    B = (s2 - s1) / (-x2 + x1)
    A = s1 + B * x1
    return float(A), float(B)


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def _inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def pd_to_score(pd: np.ndarray | float, A: float, B: float) -> np.ndarray:
    """Map predicted PD to NovaScore, clipped to [SCORE_MIN, SCORE_MAX]."""
    pd_arr = np.asarray(pd, dtype="float64")
    pd_arr = np.clip(pd_arr, 1e-6, 1 - 1e-6)
    score = A - B * np.log(pd_arr / (1 - pd_arr))
    return np.clip(score, SCORE_MIN, SCORE_MAX)


def empirical_refinement(
    probs: np.ndarray,
    A: float,
    B: float,
    q_low: float = 0.20,
    q_high: float = 0.80,
) -> tuple[float, float]:
    """Compute (a, b) so the rescaled-logit PD distribution spreads to [600, 800].

    Maps the empirical q_low quantile of `probs` to the PD at score=600 and the
    q_high quantile to the PD at score=800, in logit space.
    """
    p600 = float(_inv_logit(np.array((A - 600) / B)))
    p800 = float(_inv_logit(np.array((A - 800) / B)))
    q_lo = float(np.quantile(probs, q_low))
    q_hi = float(np.quantile(probs, q_high))
    u_lo, u_hi = _logit(np.array(q_lo)), _logit(np.array(q_hi))
    v_lo, v_hi = _logit(np.array(p600)), _logit(np.array(p800))
    den = float(u_hi - u_lo)
    if abs(den) < 1e-8:
        return 1.0, float(v_lo - u_lo)
    a = float((v_hi - v_lo) / den)
    b = float(v_lo - a * u_lo)
    return a, b


def apply_calibration(probs: np.ndarray, params: CalibrationParams) -> np.ndarray:
    """Apply empirical logit rescaling then map to score."""
    rescaled_logits = params.a * _logit(np.asarray(probs)) + params.b
    rescaled = np.clip(_inv_logit(rescaled_logits), 1e-6, 1 - 1e-6)
    return pd_to_score(rescaled, params.A, params.B)


def decision_band(score: float) -> str:
    """Return the score tier (Platinum / Gold / Silver / Bronze)."""
    if score >= 800:
        return "Platinum"
    if score >= 700:
        return "Gold"
    if score >= 600:
        return "Silver"
    return "Bronze"


BAND_DESCRIPTIONS: dict[str, str] = {
    "Platinum": "Auto-approve, large limit, lower interest, premium perks.",
    "Gold": "Medium limit, standard rates, small rewards, upgrade path.",
    "Silver": "Small limit, manual review, repayment coaching, gamification.",
    "Bronze": "Coaching and saving plans; not approved for credit yet.",
}
