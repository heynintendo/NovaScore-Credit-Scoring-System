"""Self-contained calibration + decision-band logic for the API.

Mirrors ml/novascore/calibration.py so the API has no cross-package dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SCORE_MIN: float = 300.0
SCORE_MAX: float = 950.0

BAND_POLICY: dict[str, str] = {
    "Platinum": "Auto-approve. Large credit limit, lower interest, premium benefits.",
    "Gold": "Standard approval. Medium limit, standard rates, upgrade path available.",
    "Silver": "Manual review recommended. Smaller limit, repayment coaching offered.",
    "Bronze": "Application declined. Coaching and savings plans available to rebuild standing.",
}


@dataclass(frozen=True)
class CalibrationParams:
    A: float
    B: float
    a: float = 1.0
    b: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "CalibrationParams":
        return cls(
            A=float(d["A"]),
            B=float(d["B"]),
            a=float(d.get("a", 1.0)),
            b=float(d.get("b", 0.0)),
        )


def _logit(x: float | np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def _inv_logit(z: float | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def apply_calibration(pd: float, params: CalibrationParams) -> float:
    """Map predicted PD to NovaScore (300–950)."""
    rescaled_logit = params.a * _logit(pd) + params.b
    rescaled_pd = float(np.clip(_inv_logit(rescaled_logit), 1e-6, 1 - 1e-6))
    score = params.A - params.B * math.log(rescaled_pd / (1 - rescaled_pd))
    return float(np.clip(score, SCORE_MIN, SCORE_MAX))


def decision_band(score: float) -> str:
    if score >= 800:
        return "Platinum"
    if score >= 700:
        return "Gold"
    if score >= 600:
        return "Silver"
    return "Bronze"


def bucket_age(age_years: float) -> str:
    if age_years <= 25:
        return "18-25"
    if age_years <= 40:
        return "26-40"
    if age_years <= 55:
        return "41-55"
    return "56+"
