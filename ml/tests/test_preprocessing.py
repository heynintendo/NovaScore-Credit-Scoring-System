"""Preprocessing tests for the synthetic-data path (legacy 90-day-window pipeline).

Builds a tiny synthetic DataFrame matching the schema the legacy code expects
and verifies column normalization, datetime parsing, 90-day windowing, label
computation, and the top-k ratio feature.
"""

from __future__ import annotations

import pandas as pd
import pytest

from novascore.data.preprocessing import (
    BAD_STATUSES,
    apply_window,
    compute_labels,
    compute_user_city_group,
    is_bad,
    load_parquets,
    parse_datetimes,
    topk_ratio,
    topk_values,
)


def _tiny_trips() -> pd.DataFrame:
    # Two users; one inside the 90-day window, one outside, plus mixed cases.
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "trip_id": [10, 11, 20, 21, 22],
            "trip_date": [
                "2025-01-01",
                "2025-02-15",
                "2025-03-10",
                "2025-03-20",
                "2024-01-01",  # Old — should be dropped by window.
            ],
            "trip_start_time": ["08:00:00"] * 5,
            "trip_end_time": ["08:30:00"] * 5,
            "trip_duration": [30, 25, 40, 28, 35],
            "trip_distance": [5.0, 4.5, 8.0, 6.5, 7.0],
            "fare_amount": [10.0, 9.0, 16.0, 13.0, 14.0],
            "tip_amount": [1.0, 0.5, 2.0, 1.5, 1.0],
            "trip_rating": [4.8, 5.0, 4.0, 3.5, 4.2],
            "safety_score": [0.95, 0.97, 0.85, 0.80, 0.90],
            "cancellation_flag": [0, 0, 1, 0, 0],
            "incident_flag": [False, False, True, False, False],
            "payment_method": ["card", "card", "cash", "gpay", "card"],
            "route_type": ["city", "highway", "city", "city", "suburb"],
            "city": ["Jakarta", "Jakarta", "Bangkok", "Bangkok", "Manila"],
        }
    )


def _tiny_txns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "transaction_id": [100, 101, 200, 201, 202],
            "transaction_date": [
                "2025-01-05",
                "2025-02-20",
                "2025-03-12",
                "2025-03-22",
                "2024-01-05",  # Old — dropped.
            ],
            "transaction_time": ["10:00:00"] * 5,
            "transaction_amount": [50.0, 30.0, 100.0, 80.0, 40.0],
            "balance_after_transaction": [500, 470, 200, 120, 80],
            "transaction_status": ["settled", "settled", "late", "settled", "settled"],
            "device_channel": ["android", "android", "ios", "ios", "android"],
            "merchant_category": ["grocery", "fuel", "grocery", "fuel", "grocery"],
            "merchant_id": [1, 2, 3, 3, 1],
            "city": ["Jakarta", "Jakarta", "Bangkok", "Bangkok", "Manila"],
        }
    )


def test_is_bad_classifies_keywords():
    assert is_bad("late") is True
    assert is_bad("Chargeback") is True
    assert is_bad("CANCELLED") is True
    assert is_bad("settled") is False
    assert is_bad("approved") is False
    assert is_bad(None) is False  # NaN-safe.
    assert is_bad(float("nan")) is False


def test_bad_statuses_set_includes_expected_keywords():
    for kw in ("chargeback", "default", "failed", "late", "bounced", "cancelled"):
        assert kw in BAD_STATUSES


def test_parse_datetimes_builds_timestamp_columns():
    trips, txns = parse_datetimes(_tiny_trips(), _tiny_txns())
    assert "trip_start_ts" in trips.columns
    assert "trip_end_ts" in trips.columns
    assert "transaction_dt" in txns.columns
    # cancellation_flag is coerced to int (no NaN here, so all 0/1).
    assert trips["cancellation_flag"].dtype.kind in "iu"


def test_apply_window_drops_old_rows():
    trips, txns = parse_datetimes(_tiny_trips(), _tiny_txns())
    trips_w, txns_w, anchor_end = apply_window(trips, txns, window_days=90)
    # The 2024-01-01 row in each table is far outside any 90-day window
    # anchored to the 2025 dates and must be dropped.
    assert len(trips_w) == 4
    assert len(txns_w) == 4
    # Anchor end equals the most recent timestamp across both tables.
    assert anchor_end == pd.Timestamp("2025-03-22 10:00:00")


def test_compute_labels_marks_any_bad_status():
    trips, txns = parse_datetimes(_tiny_trips(), _tiny_txns())
    _, txns_w, _ = apply_window(trips, txns)
    labels = compute_labels(txns_w)
    # User 2 has a "late" transaction in window → y=1. User 1 settled-only → y=0.
    assert labels.loc[labels.user_id == 1, "y"].iloc[0] == 0
    assert labels.loc[labels.user_id == 2, "y"].iloc[0] == 1


def test_topk_values_picks_most_common():
    df = pd.DataFrame({"col": ["a", "a", "b", "c", "a", "b"]})
    assert topk_values(df, "col", k=2) == ["a", "b"]


def test_topk_ratio_produces_user_fractions():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "payment_method": ["card", "card", "cash", "gpay", "gpay"],
        }
    )
    out = topk_ratio(df, "payment_method", ["card", "cash", "gpay"], prefix="pay_")
    # User 1: 2/3 card, 1/3 cash. User 2: 0 card, 0 cash, 1 gpay.
    user1 = out.loc[1]
    user2 = out.loc[2]
    assert user1["pay_card"] == pytest.approx(2 / 3)
    assert user1["pay_cash"] == pytest.approx(1 / 3)
    assert user2["pay_gpay"] == pytest.approx(1.0)


def test_compute_user_city_group_collapses_to_other():
    txns = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "city": ["Jakarta", "Jakarta", "Bandung", "Bangkok", "Bangkok", "TinyTown"],
        }
    )
    user_ids = pd.Series([1, 2, 3])
    out = compute_user_city_group(txns, user_ids, top_k=2)
    # Top 2 cities (Jakarta=2, Bangkok=2; ties broken by pandas value_counts).
    assert out.loc[out.user_id == 1, "city_grp"].iloc[0] in {"Jakarta", "Bangkok"}
    # TinyTown isn't in the top-2 → "other".
    assert out.loc[out.user_id == 3, "city_grp"].iloc[0] == "other"


def test_load_parquets_normalizes_column_case(tmp_path):
    """Mixed-case columns from upstream must be lower-cased on load."""
    trips_p = tmp_path / "trips.parquet"
    txns_p = tmp_path / "txns.parquet"
    pd.DataFrame({"User_ID": [1], "Trip_Date": ["2025-01-01"]}).to_parquet(trips_p)
    pd.DataFrame({"User_ID": [1], "Transaction_Date": ["2025-01-01"]}).to_parquet(txns_p)
    trips, txns = load_parquets(trips_p, txns_p)
    assert set(trips.columns) == {"user_id", "trip_date"}
    assert set(txns.columns) == {"user_id", "transaction_date"}
