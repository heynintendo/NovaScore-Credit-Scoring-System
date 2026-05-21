"""Smoke test for Home Credit preprocessing on a tiny mock DataFrame.

We exercise the feature-engineering functions individually (not the full
download/load path) using a minimal fake `application_train`-like table with
the columns the engineering code expects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from novascore.data.home_credit import (
    BUREAU_STATUS_CODES,
    _aggregate_bureau,
    _aggregate_installments,
    _aggregate_prev_app,
    _bucket_age,
    _engineer_application,
    build_bureau_balance_sequences,
)


def _mini_app() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "DAYS_BIRTH": [-9000, -15000, -22000],
            "DAYS_EMPLOYED": [-365, -2000, 365243],  # 365243 is the unemployed sentinel
            "AMT_INCOME_TOTAL": [200_000.0, 150_000.0, 80_000.0],
            "AMT_CREDIT": [500_000.0, 600_000.0, 300_000.0],
            "AMT_ANNUITY": [25_000.0, 30_000.0, 15_000.0],
            "AMT_GOODS_PRICE": [450_000.0, 550_000.0, 280_000.0],
            "CODE_GENDER": ["F", "M", "F"],
            "FLAG_OWN_CAR": ["N", "Y", "N"],
            "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Married"],
        }
    )


def test_engineer_application_adds_derived_columns():
    app = _engineer_application(_mini_app())
    assert "age_years" in app.columns
    assert "employed_years" in app.columns
    assert "credit_to_income" in app.columns
    assert "annuity_to_income" in app.columns
    assert "credit_to_goods" in app.columns
    # Unemployed sentinel must become NaN, then employed_years is NaN.
    assert pd.isna(app.loc[2, "employed_years"])
    # Ratios: row 0 has AMT_CREDIT=500k / income=200k = 2.5.
    assert app.loc[0, "credit_to_income"] == pytest.approx(2.5, rel=1e-3)


def test_age_bucketing_boundaries():
    s = pd.Series([20.0, 26.0, 41.0, 60.0])
    out = _bucket_age(s).tolist()
    assert out == ["18-25", "26-40", "41-55", "56+"]


def test_aggregate_bureau_produces_expected_columns():
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 12],
            "DAYS_CREDIT": [-300, -100, -200],
            "CREDIT_DAY_OVERDUE": [0, 5, 0],
            "DAYS_CREDIT_ENDDATE": [100, 200, 50],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 100.0, 0.0],
            "AMT_CREDIT_SUM": [10000.0, 5000.0, 8000.0],
            "AMT_CREDIT_SUM_DEBT": [0.0, 0.0, 200.0],
            "AMT_CREDIT_SUM_LIMIT": [10000.0, 5000.0, 8000.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 0.0],
            "CNT_CREDIT_PROLONG": [0, 0, 0],
            "CREDIT_ACTIVE": ["Closed", "Active", "Active"],
        }
    )
    out = _aggregate_bureau(bureau)
    assert set(["SK_ID_CURR", "bureau_n_loans", "bureau_n_active"]).issubset(out.columns)
    row1 = out[out["SK_ID_CURR"] == 1].iloc[0]
    assert int(row1["bureau_n_loans"]) == 2
    assert int(row1["bureau_n_active"]) == 1


def test_aggregate_prev_app_computes_refusal_rate():
    prev = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "AMT_ANNUITY": [100.0, 200.0, 50.0, 75.0],
            "AMT_APPLICATION": [1000.0, 2000.0, 500.0, 800.0],
            "AMT_CREDIT": [900.0, 1800.0, 0.0, 750.0],
            "DAYS_DECISION": [-100, -50, -200, -30],
            "CNT_PAYMENT": [12, 24, 6, 12],
            "NAME_CONTRACT_STATUS": ["Approved", "Approved", "Refused", "Approved"],
        }
    )
    out = _aggregate_prev_app(prev)
    row1 = out[out["SK_ID_CURR"] == 1].iloc[0]
    assert int(row1["prev_n_applications"]) == 3
    assert int(row1["prev_n_approved"]) == 2
    assert int(row1["prev_n_refused"]) == 1
    assert row1["prev_refusal_rate"] == pytest.approx(1.0 / 3.0, rel=1e-3)


def test_aggregate_installments_flags_late_payments():
    inst = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "AMT_INSTALMENT": [100.0, 200.0, 50.0],
            "AMT_PAYMENT": [100.0, 180.0, 50.0],  # row 2 underpaid by 20
            "DAYS_INSTALMENT": [-30, -60, -10],
            "DAYS_ENTRY_PAYMENT": [-25, -65, -15],  # row 1 was 5 days LATE
        }
    )
    out = _aggregate_installments(inst)
    row1 = out[out["SK_ID_CURR"] == 1].iloc[0]
    assert int(row1["inst_n_payments"]) == 2
    # Row 0: paid -25 vs scheduled -30 (5 days late). Row 1: -65 vs -60 (5 days early).
    assert row1["inst_was_late_sum"] == pytest.approx(1.0)


def test_build_bureau_balance_sequences_shapes():
    # Two applicants, two bureau loans each, three months of history.
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2],
            "SK_ID_BUREAU": [101, 102, 201, 202],
        }
    )
    bb = pd.DataFrame(
        {
            "SK_ID_BUREAU": [101, 101, 102, 201, 201, 202],
            "MONTHS_BALANCE": [0, -1, 0, 0, -2, -1],
            "STATUS": ["0", "1", "C", "0", "X", "0"],
        }
    )
    user_ids = np.array([1, 2], dtype=np.int64)
    x_seq, mean, std = build_bureau_balance_sequences(bureau, bb, user_ids, n_months=12)
    assert x_seq.shape == (2, 12, len(BUREAU_STATUS_CODES))
    assert mean.shape == (len(BUREAU_STATUS_CODES),)
    assert std.shape == (len(BUREAU_STATUS_CODES),)
    # Last time index (11) has rows from MONTHS_BALANCE=0.
    # Non-trivial: ensure not all zeros after normalization.
    assert np.any(x_seq != 0.0)
