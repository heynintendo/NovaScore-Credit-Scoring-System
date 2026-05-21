"""Synthetic data generator for NovaScore.

Produces three parquet files matching the schema the rest of the pipeline expects:

- trips.parquet — per-trip rows (engagement, fares, ratings, cancellations).
- txns.parquet — per-transaction rows (amount, balance, merchant, channel, status).
- users_demo.parquet — protected attributes per user (kept separate; never fed to model).

Bias injection (used as test case for the fairness module):
    Young + female + bicycle partners get a +15% base default rate via the
    `BIAS_MULT` coefficient. Legitimate default drivers are high cancel rate,
    low rating, high transaction-amount volatility, and late payment statuses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_CITIES = (
    "Jakarta",
    "Singapore",
    "Bangkok",
    "Kuala_Lumpur",
    "Manila",
    "Ho_Chi_Minh_City",
    "Hanoi",
    "Yangon",
    "Phnom_Penh",
    "Bandung",
    "other",
)
_VEHICLES = ("motorcycle", "car", "bicycle")
_AGE_BUCKETS = ("18-25", "26-40", "41-55", "56+")
_GENDERS = ("M", "F")
_PAYMENT_METHODS = ("gpay", "card", "cash", "wallet")
_ROUTE_TYPES = ("city", "highway", "suburb")
_DEVICE_CHANNELS = ("android", "ios", "web")
_MERCHANT_CATEGORIES = (
    "grocery",
    "fuel",
    "restaurant",
    "telco",
    "utility",
    "transport",
    "retail",
)
_GOOD_STATUS = ("settled", "completed", "approved")
_BAD_STATUS = ("late", "default", "chargeback", "failed", "bounced")

# Bias coefficient: how much extra default risk young+female+bicycle users carry.
BIAS_MULT: float = 0.15


def _draw_demographics(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": np.arange(n, dtype=np.int64),
            "gender": rng.choice(_GENDERS, size=n, p=[0.55, 0.45]),
            "age_bucket": rng.choice(_AGE_BUCKETS, size=n, p=[0.25, 0.45, 0.22, 0.08]),
            "vehicle_type": rng.choice(_VEHICLES, size=n, p=[0.60, 0.30, 0.10]),
            "city": rng.choice(
                _CITIES,
                size=n,
                p=[0.16, 0.10, 0.12, 0.10, 0.12, 0.10, 0.08, 0.05, 0.05, 0.07, 0.05],
            ),
        }
    )


def _draw_latent_default_prob(demo: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Per-user latent default probability, mixing legitimate signal + injected bias."""
    n = len(demo)
    base = rng.beta(2.0, 18.0, size=n)  # ~ mean 0.10
    is_biased = (
        (demo["gender"].to_numpy() == "F")
        & (demo["age_bucket"].to_numpy() == "18-25")
        & (demo["vehicle_type"].to_numpy() == "bicycle")
    )
    base = np.where(is_biased, np.clip(base + BIAS_MULT, 0.0, 0.95), base)
    return base.astype("float64")


def _draw_trips(
    demo: pd.DataFrame,
    latent_pd: np.ndarray,
    anchor_end: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    trip_id = 0
    for i, uid in enumerate(demo["user_id"].to_numpy()):
        # Higher-PD users have more cancellations and lower ratings.
        pd_u = float(latent_pd[i])
        n_trips = int(rng.poisson(lam=max(10.0, 80.0 - 40.0 * pd_u)))
        if n_trips == 0:
            continue
        # Spread across 90 days, biased toward recent activity.
        day_offsets = rng.integers(0, 90, size=n_trips)
        for d in day_offsets:
            trip_dt = anchor_end - pd.Timedelta(days=int(d))
            duration = float(rng.lognormal(mean=2.7, sigma=0.4))  # ~ 15 min mean
            distance = float(rng.lognormal(mean=1.5, sigma=0.6))  # km
            # Risky drivers tip less (lower customer satisfaction) and earn less per trip.
            fare = max(0.5, distance * rng.uniform(1.0 + 0.4 * (1 - pd_u), 2.0))
            tip = max(0.0, fare * rng.uniform(0.0, max(0.0, 0.15 - 0.30 * pd_u)))
            # Strong feature-PD coupling: rating drops fast with risk, cancel/incident climb.
            rating = float(np.clip(rng.normal(5.0 - 4.0 * pd_u, 0.10), 1.0, 5.0))
            safety = float(np.clip(rng.normal(0.99 - 1.5 * pd_u, 0.03), 0.0, 1.0))
            cancel = int(rng.random() < min(0.95, 0.01 + 0.80 * pd_u))
            incident = int(rng.random() < min(0.95, 0.005 + 0.45 * pd_u))
            start_time = pd.Timedelta(
                hours=int(rng.integers(6, 22)), minutes=int(rng.integers(0, 59))
            )
            end_time = start_time + pd.Timedelta(minutes=int(duration))
            rows.append(
                {
                    "user_id": int(uid),
                    "trip_id": trip_id,
                    "trip_date": trip_dt.normalize(),
                    "trip_start_time": str(start_time),
                    "trip_end_time": str(end_time),
                    "trip_duration": duration,
                    "trip_distance": distance,
                    "fare_amount": fare,
                    "tip_amount": tip,
                    "trip_rating": rating,
                    "safety_score": safety,
                    "cancellation_flag": cancel,
                    "incident_flag": bool(incident),
                    "payment_method": rng.choice(_PAYMENT_METHODS, p=[0.5, 0.25, 0.15, 0.10]),
                    "route_type": rng.choice(_ROUTE_TYPES, p=[0.6, 0.2, 0.2]),
                    "city": demo.iloc[i]["city"],
                }
            )
            trip_id += 1
    return pd.DataFrame(rows)


def _draw_txns(
    demo: pd.DataFrame,
    latent_pd: np.ndarray,
    anchor_end: pd.Timestamp,
    rng: np.random.Generator,
    n_merchants: int = 200,
) -> pd.DataFrame:
    rows = []
    txn_id = 0
    for i, uid in enumerate(demo["user_id"].to_numpy()):
        pd_u = float(latent_pd[i])
        n_txns = int(rng.poisson(lam=max(8.0, 60.0 - 25.0 * pd_u)))
        if n_txns == 0:
            continue
        # Volatility ∝ pd_u (high-risk users have wider spending swings).
        # High-risk users have much higher transaction-amount volatility.
        sigma_amt = 0.4 + 1.8 * pd_u
        day_offsets = rng.integers(0, 90, size=n_txns)
        balance = float(rng.uniform(50, 5000))
        for d in day_offsets:
            txn_dt = anchor_end - pd.Timedelta(days=int(d))
            amount = float(rng.lognormal(mean=2.5, sigma=sigma_amt))
            balance = max(0.0, balance + rng.choice([-1, 1]) * amount * 0.3)
            # Per-txn bad rate tuned so user-level "any bad txn" label sits near
            # the latent default rate (~10% overall, ~25% in biased subgroup)
            # at the ~57 txns/user the synthesizer produces. See REPORT.md.
            late_prob = 0.001 + 0.015 * pd_u
            status = (
                rng.choice(_BAD_STATUS) if rng.random() < late_prob else rng.choice(_GOOD_STATUS)
            )
            merchant_id = int(rng.integers(0, n_merchants))
            rows.append(
                {
                    "user_id": int(uid),
                    "transaction_id": txn_id,
                    "transaction_date": txn_dt.normalize(),
                    "transaction_time": f"{int(rng.integers(0, 23)):02d}:{int(rng.integers(0, 59)):02d}:00",
                    "transaction_amount": amount,
                    "balance_after_transaction": balance,
                    "transaction_status": status,
                    "device_channel": rng.choice(_DEVICE_CHANNELS, p=[0.65, 0.30, 0.05]),
                    "merchant_category": rng.choice(_MERCHANT_CATEGORIES),
                    "merchant_id": merchant_id,
                    "city": demo.iloc[i]["city"],
                }
            )
            txn_id += 1
    return pd.DataFrame(rows)


def generate(
    n_users: int = 10000,
    seed: int = 42,
    output_dir: str | Path = "ml/data/synthetic/",
    anchor_end: pd.Timestamp | None = None,
) -> dict[str, Path]:
    """Generate trips, txns, and demographic parquets.

    Args:
        n_users: number of synthetic users.
        seed: RNG seed for reproducibility.
        output_dir: directory to write trips.parquet, txns.parquet, users_demo.parquet.
        anchor_end: end of the 90-day window. Defaults to today (UTC, midnight).

    Returns: mapping {"trips": Path, "txns": Path, "users_demo": Path}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    anchor_end = anchor_end or pd.Timestamp.utcnow().normalize().tz_localize(None)

    demo = _draw_demographics(n_users, rng)
    latent_pd = _draw_latent_default_prob(demo, rng)
    trips = _draw_trips(demo, latent_pd, anchor_end, rng)
    txns = _draw_txns(demo, latent_pd, anchor_end, rng)

    # Keep demo separate so the model never sees protected attributes directly.
    demo_out = demo.assign(latent_default_pd=latent_pd)

    paths = {
        "trips": out / "trips.parquet",
        "txns": out / "txns.parquet",
        "users_demo": out / "users_demo.parquet",
    }
    trips.to_parquet(paths["trips"], index=False)
    txns.to_parquet(paths["txns"], index=False)
    demo_out.to_parquet(paths["users_demo"], index=False)
    return paths


if __name__ == "__main__":
    paths = generate(n_users=2000, seed=42, output_dir="ml/data/synthetic/")
    for k, v in paths.items():
        df = pd.read_parquet(v)
        print(f"{k}: {len(df):,} rows -> {v}")
