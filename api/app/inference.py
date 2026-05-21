"""Model loading + applicant scoring for the NovaScore API.

Loads the four committed artifacts:
  models/lightgbm.txt        — LightGBM booster
  models/feature_columns.json — ordered feature names the booster expects
  models/scaler.json          — per-feature mean/scale used at training time
  models/calibration.json     — A, B, a, b for the PD → score map
  models/threshold_map.json   — per-age-bucket score adjustments

The model expects standardized inputs. We construct a length-N feature vector
from the user's request by:
  1. Initialising every position to the *training mean* (which standardizes to 0).
  2. Overriding the positions for fields the user supplies, computed in raw scale.
  3. Standardizing via (x − mean) / scale.
This is the same recipe the synthetic-data CLI uses (`novascore score`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .calibration import (
    BAND_POLICY,
    CalibrationParams,
    apply_calibration,
    bucket_age,
    decision_band,
)
from .schemas import ScoreRequest, ScoreResponse


@dataclass
class _Bundle:
    booster: lgb.Booster
    feature_columns: list[str]
    col_index: dict[str, int]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    calibration: CalibrationParams
    score_adjustments: dict[str, float]


_BUNDLE: _Bundle | None = None


def load_artifacts(models_dir: Path) -> _Bundle:
    """Idempotent loader; cached in module state for the lifetime of the process."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    models_dir = Path(models_dir)

    booster = lgb.Booster(model_file=str(models_dir / "lightgbm.txt"))
    feature_columns: list[str] = json.loads((models_dir / "feature_columns.json").read_text())
    scaler_d = json.loads((models_dir / "scaler.json").read_text())
    scaler_mean = np.asarray(scaler_d["mean"], dtype="float32")
    scaler_scale = np.asarray(scaler_d["scale"], dtype="float32")
    calib_d = json.loads((models_dir / "calibration.json").read_text())
    calibration = CalibrationParams.from_dict(calib_d)

    # threshold_map.json (Phase 4.6 fairness mitigation on age_bucket).
    threshold_path = models_dir / "threshold_map.json"
    score_adjustments: dict[str, float] = {}
    if threshold_path.exists():
        tm = json.loads(threshold_path.read_text())
        # The fairness module stored score adjustments inside the run's
        # metrics. Reconstruct from the threshold map + calibration B.
        # adjustment = B * (logit(group_thr) - logit(default_thr))
        thresholds = tm.get("thresholds", {})
        default_thr = float(tm.get("default_threshold", 0.5))
        if thresholds:
            from math import log

            def logit(p: float) -> float:
                p = min(max(p, 1e-6), 1 - 1e-6)
                return log(p / (1 - p))

            ldef = logit(default_thr)
            for bucket, thr in thresholds.items():
                score_adjustments[bucket] = float(calibration.B * (logit(float(thr)) - ldef))

    bundle = _Bundle(
        booster=booster,
        feature_columns=feature_columns,
        col_index={c: i for i, c in enumerate(feature_columns)},
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        calibration=calibration,
        score_adjustments=score_adjustments,
    )
    _BUNDLE = bundle
    return bundle


def _build_feature_vector(req: ScoreRequest, b: _Bundle) -> np.ndarray:
    """Map the 12-field demo request onto the model's full feature vector."""
    raw = b.scaler_mean.copy().astype("float64")

    def setf(name: str, value: float) -> None:
        i = b.col_index.get(name)
        if i is not None:
            raw[i] = float(value)

    # Application columns the model actually trained on.
    setf("AMT_CREDIT", req.loan_amount)
    setf("AMT_GOODS_PRICE", req.loan_amount * 0.95)  # close proxy for goods price
    setf("DAYS_BIRTH", -req.age_years * 365.0)
    setf("DAYS_EMPLOYED", -req.years_employed * 365.0)
    setf("EXT_SOURCE_1", req.ext_source_1)
    setf("EXT_SOURCE_2", req.ext_source_2)
    setf("EXT_SOURCE_3", req.ext_source_3)
    setf("FLAG_EMP_PHONE", 1.0 if req.years_employed > 0 else 0.0)
    # Derived columns from preprocessing.
    setf("age_years", req.age_years)
    setf("employed_years", req.years_employed if req.years_employed > 0 else float("nan"))
    setf("credit_to_goods", 1.0 / 0.95)  # AMT_CREDIT / AMT_GOODS_PRICE proxy
    # OWN_CAR_AGE is meaningful only when has_car; otherwise leave at mean.
    if not req.has_car:
        # Force OWN_CAR_AGE to 0 (no car) — this slightly nudges the standardised value.
        setf("OWN_CAR_AGE", 0.0)

    # Handle the NaN we just inserted for unemployed applicants.
    raw = np.where(np.isnan(raw), b.scaler_mean.astype("float64"), raw)

    # Standardize.
    standardized = ((raw - b.scaler_mean) / b.scaler_scale).astype("float32")
    return standardized.reshape(1, -1)


def score_applicant(req: ScoreRequest, b: _Bundle) -> ScoreResponse:
    """Run the LightGBM booster and apply calibration + fairness adjustment."""
    X = _build_feature_vector(req, b)
    pd_value = float(b.booster.predict(X, num_iteration=b.booster.best_iteration)[0])
    novascore = apply_calibration(pd_value, b.calibration)
    band = decision_band(novascore)

    age_bucket = bucket_age(req.age_years)
    adj = b.score_adjustments.get(age_bucket, 0.0)
    adjusted = max(300.0, min(950.0, novascore + adj))
    adjusted_band = decision_band(adjusted)
    reason: str | None = None
    if abs(adj) > 0.5:
        sign = "+" if adj > 0 else "−"
        reason = (
            f"Per-group equal-opportunity threshold for age {age_bucket} "
            f"({sign}{abs(adj):.1f} points)."
        )

    return ScoreResponse(
        pd=pd_value,
        novascore=round(novascore, 1),
        decision_band=band,
        policy=BAND_POLICY[band],
        fairness_adjusted_score=round(adjusted, 1),
        fairness_adjusted_band=adjusted_band,
        fairness_adjustment_reason=reason,
        age_bucket=age_bucket,
        raw_features_used=len(b.feature_columns),
    )
