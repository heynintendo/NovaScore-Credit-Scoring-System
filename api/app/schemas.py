"""Pydantic models for the NovaScore inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Gender = Literal["F", "M"]
FamilyStatus = Literal["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]


class ScoreRequest(BaseModel):
    """Applicant features supplied by the demo frontend.

    Fields the LightGBM model uses directly: age_years, loan_amount,
    years_employed, ext_source_1/2/3. The others are collected for
    fairness audit display (gender, family_status) or downstream UX (children,
    car) but standardize toward training mean in the underlying feature vector.
    """

    age_years: float = Field(..., ge=18, le=80, description="Applicant age in years")
    gender: Gender = "F"
    family_status: FamilyStatus = "Married"
    num_children: int = Field(0, ge=0, le=10)
    has_car: bool = True
    annual_income: float = Field(150_000, ge=10_000, le=10_000_000)
    loan_amount: float = Field(500_000, ge=10_000, le=10_000_000)
    annuity: float = Field(25_000, ge=1_000, le=1_000_000)
    years_employed: float = Field(8, ge=0, le=60)
    ext_source_1: float = Field(0.50, ge=0.0, le=1.0)
    ext_source_2: float = Field(0.50, ge=0.0, le=1.0)
    ext_source_3: float = Field(0.50, ge=0.0, le=1.0)


class ScoreResponse(BaseModel):
    """One applicant's calibrated score, band, and fairness-adjusted variants."""

    pd: float = Field(..., description="Predicted probability of default")
    novascore: float = Field(..., description="Calibrated score in [300, 950]")
    decision_band: Literal["Platinum", "Gold", "Silver", "Bronze"]
    policy: str
    fairness_adjusted_score: float
    fairness_adjusted_band: Literal["Platinum", "Gold", "Silver", "Bronze"]
    fairness_adjustment_reason: str | None = None
    age_bucket: Literal["18-25", "26-40", "41-55", "56+"]
    raw_features_used: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    feature_count: int
