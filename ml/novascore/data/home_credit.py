"""Home Credit Default Risk dataset ingestion + feature engineering.

Public Kaggle dataset: 307K real anonymized loan applications. After Phase 4.5's
synth-data ceiling at ~0.61 AUROC, NovaScore pivoted to this dataset for
credible benchmarking. The same architecture (FT-Transformer + TCN + LightGBM
ensemble) applies; only the data wiring changes.

Pipeline:
1. `ensure_data(data_dir)` — downloads via Kaggle CLI if missing, then unzips.
2. `load_tables(data_dir)` — loads application_train + linked tables.
3. `build_tabular_features(...)` — application columns + aggregates from
    bureau / previous_application / installments_payments.
4. `build_bureau_balance_sequences(...)` — applicant-month STATUS one-hot tensor
    (60 months × ~8 channels) for the TCN tower.
5. `prepare_home_credit(...)` — runs everything, returns a `HomeCreditBundle`.
"""

from __future__ import annotations

import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# STATUS codes from bureau_balance.csv. C=closed, X=unknown, 0..5=days past due
# bucket. Used as one-hot channels for the TCN.
BUREAU_STATUS_CODES: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "C", "X")

# Bureau balance: 60-month history per applicant.
SEQ_T_MONTHS: int = 60

# Columns to drop from application_train if >MAX_MISSING_RATIO are NaN.
MAX_MISSING_RATIO: float = 0.80

# Cap final tabular feature count so the FT-Transformer's O(n²) attention stays
# tractable on commodity hardware. Selected by f_classif against TARGET.
MAX_TAB_FEATURES: int = 80


@dataclass
class HomeCreditBundle:
    """Everything needed to train: features, sequence, labels, protected attrs."""

    user_ids: np.ndarray  # SK_ID_CURR
    X_tab: np.ndarray  # standardized tabular
    X_seq: np.ndarray  # (n, SEQ_T_MONTHS, len(BUREAU_STATUS_CODES))
    y: np.ndarray
    feature_columns: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    seq_mean: np.ndarray
    seq_scale: np.ndarray
    categorical_maps: dict[str, list[str]]  # learned one-hot levels
    protected_attrs: pd.DataFrame  # SK_ID_CURR + gender/age/own_car/family_status


# ---------------------------------------------------------------------------
# Download / load


def ensure_data(data_dir: Path) -> None:
    """Download + unzip Home Credit Default Risk if not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    primary = data_dir / "application_train.csv"
    if primary.exists():
        return
    zip_path = data_dir / "home-credit-default-risk.zip"
    if not zip_path.exists():
        print(f"[home_credit] downloading via Kaggle CLI to {data_dir}")
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "home-credit-default-risk",
                "-p",
                str(data_dir),
            ],
            check=True,
        )
    print(f"[home_credit] unzipping {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    # Some files in the bundle are themselves zipped.
    for inner in data_dir.glob("*.csv.zip"):
        with zipfile.ZipFile(inner) as z:
            z.extractall(data_dir)
        inner.unlink()


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load primary + linked tables. Aggregations stay opt-in (lazy)."""
    data_dir = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name in (
        "application_train",
        "bureau",
        "bureau_balance",
        "previous_application",
        "installments_payments",
    ):
        p = data_dir / f"{name}.csv"
        if not p.exists():
            continue
        tables[name] = pd.read_csv(p)
        print(f"[home_credit] {name}: {len(tables[name]):,} rows")
    return tables


# ---------------------------------------------------------------------------
# Feature engineering — application + light aggregates


def _drop_high_missing(df: pd.DataFrame, max_ratio: float = MAX_MISSING_RATIO) -> pd.DataFrame:
    miss = df.isna().mean()
    keep = miss[miss <= max_ratio].index.tolist()
    return df[keep]


def _engineer_application(app: pd.DataFrame) -> pd.DataFrame:
    """Add ratio / age features to the application table."""
    df = app.copy()
    # DAYS_* columns are negative integers (days before application).
    df["age_years"] = (-df["DAYS_BIRTH"] / 365.0).astype("float32")
    if "DAYS_EMPLOYED" in df.columns:
        # 365243 = sentinel for "unemployed". Replace with NaN.
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        df["employed_years"] = (-df["DAYS_EMPLOYED"] / 365.0).astype("float32")
    if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
        df["credit_to_income"] = (
            df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        ).astype("float32")
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["annuity_to_income"] = (
            df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        ).astype("float32")
    if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
        df["credit_to_goods"] = (
            df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
        ).astype("float32")
    return df


def _aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Per-applicant aggregates from bureau.csv."""
    agg_funcs: dict[str, list[str]] = {
        "DAYS_CREDIT": ["count", "mean", "max", "min"],
        "CREDIT_DAY_OVERDUE": ["mean", "max", "sum"],
        "DAYS_CREDIT_ENDDATE": ["mean", "max"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
        "AMT_CREDIT_SUM": ["mean", "sum", "max"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "sum", "max"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "sum", "max"],
        "CNT_CREDIT_PROLONG": ["sum"],
    }
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in bureau.columns}
    g = bureau.groupby("SK_ID_CURR").agg(agg_funcs)
    g.columns = [f"bureau_{c[0]}_{c[1]}" for c in g.columns]
    g["bureau_n_loans"] = bureau.groupby("SK_ID_CURR").size()
    if "CREDIT_ACTIVE" in bureau.columns:
        active = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
        g["bureau_n_active"] = active.groupby(bureau["SK_ID_CURR"]).sum()
    return g.reset_index()


def _aggregate_prev_app(prev: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_APPLICATION": ["mean", "max", "sum"],
        "AMT_CREDIT": ["mean", "max"],
        "AMT_DOWN_PAYMENT": ["mean", "max"],
        "RATE_DOWN_PAYMENT": ["mean", "max"],
        "DAYS_DECISION": ["mean", "max", "min"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    cols = {k: v for k, v in cols.items() if k in prev.columns}
    g = prev.groupby("SK_ID_CURR").agg(cols)
    g.columns = [f"prev_{c[0]}_{c[1]}" for c in g.columns]
    g["prev_n_applications"] = prev.groupby("SK_ID_CURR").size()
    if "NAME_CONTRACT_STATUS" in prev.columns:
        approved = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
        refused = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
        g["prev_n_approved"] = approved.groupby(prev["SK_ID_CURR"]).sum()
        g["prev_n_refused"] = refused.groupby(prev["SK_ID_CURR"]).sum()
        g["prev_refusal_rate"] = g["prev_n_refused"] / (
            g["prev_n_approved"] + g["prev_n_refused"]
        ).replace(0, np.nan)
    return g.reset_index()


def _aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    df = inst.copy()
    if "AMT_INSTALMENT" in df.columns and "AMT_PAYMENT" in df.columns:
        df["payment_diff"] = df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]
        df["payment_ratio"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)
    if "DAYS_ENTRY_PAYMENT" in df.columns and "DAYS_INSTALMENT" in df.columns:
        df["days_late"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
        df["was_late"] = (df["days_late"] > 0).astype(int)
    cols = {
        "AMT_INSTALMENT": ["sum", "mean"],
        "AMT_PAYMENT": ["sum", "mean"],
        "payment_diff": ["sum", "mean"],
        "payment_ratio": ["mean"],
        "days_late": ["mean", "max"],
        "was_late": ["sum", "mean"],
    }
    cols = {k: v for k, v in cols.items() if k in df.columns}
    g = df.groupby("SK_ID_CURR").agg(cols)
    g.columns = [f"inst_{c[0]}_{c[1]}" for c in g.columns]
    g["inst_n_payments"] = df.groupby("SK_ID_CURR").size()
    return g.reset_index()


def build_tabular_features(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Stitch application + linked-table aggregates; one-hot low-cardinality cats.

    Returns (features_df indexed by SK_ID_CURR, categorical_maps for inference).
    """
    app = _engineer_application(tables["application_train"])
    if "bureau" in tables:
        app = app.merge(_aggregate_bureau(tables["bureau"]), on="SK_ID_CURR", how="left")
    if "previous_application" in tables:
        app = app.merge(
            _aggregate_prev_app(tables["previous_application"]), on="SK_ID_CURR", how="left"
        )
    if "installments_payments" in tables:
        app = app.merge(
            _aggregate_installments(tables["installments_payments"]), on="SK_ID_CURR", how="left"
        )

    app = _drop_high_missing(app, MAX_MISSING_RATIO)
    # One-hot low-cardinality categoricals; record the levels for inference.
    cat_cols = [c for c in app.columns if app[c].dtype == "object" and app[c].nunique() <= 30]
    cat_maps: dict[str, list[str]] = {}
    for c in cat_cols:
        levels = sorted(app[c].dropna().astype(str).unique().tolist())
        cat_maps[c] = levels
    app = pd.get_dummies(app, columns=cat_cols, drop_first=False, dummy_na=False).astype(
        {col: "float32" for col in cat_cols if col in app.columns}, errors="ignore"
    )
    # Drop any remaining object columns (high-cardinality strings we don't encode here).
    obj_cols = [c for c in app.columns if app[c].dtype == "object"]
    app = app.drop(columns=obj_cols)
    return app, cat_maps


# ---------------------------------------------------------------------------
# Sequence features — bureau_balance


def build_bureau_balance_sequences(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    user_ids: np.ndarray,
    n_months: int = SEQ_T_MONTHS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (n_users, n_months, n_channels) tensor of one-hot bureau STATUS counts.

    Each applicant may have multiple bureau loans; for each (applicant, month)
    cell we sum the one-hot vectors of STATUS across their bureau loans, then
    crop/pad to the last `n_months` months (MONTHS_BALANCE in [-n_months+1, 0]).
    Standardizes per-channel.
    """
    n_ch = len(BUREAU_STATUS_CODES)
    if "SK_ID_BUREAU" not in bureau.columns:
        return (
            np.zeros((len(user_ids), n_months, n_ch), dtype="float32"),
            np.zeros(n_ch, dtype="float32"),
            np.ones(n_ch, dtype="float32"),
        )
    # Restrict to the sampled user set before the big merge.
    user_set = set(int(u) for u in user_ids)
    bureau_lookup = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates("SK_ID_BUREAU")
    bureau_lookup = bureau_lookup[bureau_lookup["SK_ID_CURR"].isin(user_set)]
    bb = bureau_balance.merge(bureau_lookup, on="SK_ID_BUREAU", how="inner")
    bb = bb[(bb["MONTHS_BALANCE"] > -n_months) & (bb["MONTHS_BALANCE"] <= 0)]
    if len(bb) == 0:
        return (
            np.zeros((len(user_ids), n_months, n_ch), dtype="float32"),
            np.zeros(n_ch, dtype="float32"),
            np.ones(n_ch, dtype="float32"),
        )
    bb["t_idx"] = (bb["MONTHS_BALANCE"] + n_months - 1).astype("int32")
    bb["STATUS"] = bb["STATUS"].astype(str)
    status_cols = [f"s_{k}" for k in range(n_ch)]
    for k, code in enumerate(BUREAU_STATUS_CODES):
        bb[status_cols[k]] = (bb["STATUS"] == code).astype("int16")
    grouped = bb.groupby(["SK_ID_CURR", "t_idx"], as_index=False)[status_cols].sum()

    uid2idx = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids.astype(int))
    uid_idx = grouped["SK_ID_CURR"].map(uid2idx).to_numpy()
    t_idx = grouped["t_idx"].to_numpy()
    status_vals = grouped[status_cols].to_numpy(dtype="float32")

    x_seq = np.zeros((len(user_ids), n_months, n_ch), dtype="float32")
    valid = ~np.isnan(uid_idx)
    x_seq[uid_idx[valid].astype("int32"), t_idx[valid]] = status_vals[valid]

    flat = x_seq.reshape(-1, n_ch).astype("float64")
    mean = flat.mean(axis=0).astype("float32")
    std = flat.std(axis=0).astype("float32") + 1e-6
    x_seq = ((x_seq - mean) / std).astype("float32")
    return x_seq, mean, std


# ---------------------------------------------------------------------------
# Top-level orchestrator


def _bucket_age(age_years: pd.Series) -> pd.Series:
    bins = [-np.inf, 25, 40, 55, np.inf]
    labels = ["18-25", "26-40", "41-55", "56+"]
    return pd.cut(age_years, bins=bins, labels=labels, ordered=False).astype(str)


def prepare_home_credit(
    data_dir: Path,
    *,
    sample_n: int | None = None,
    seed: int = 42,
) -> HomeCreditBundle:
    """Run the full Home Credit pipeline; return a HomeCreditBundle.

    Args:
        data_dir: directory with the Kaggle CSVs (will download if missing).
        sample_n: if provided, subsample applicants for fast iteration.
        seed: RNG seed for the sample.
    """
    data_dir = Path(data_dir)
    ensure_data(data_dir)
    tables = load_tables(data_dir)
    app = tables["application_train"]
    if sample_n is not None and sample_n < len(app):
        app = app.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        tables["application_train"] = app
        # Restrict linked tables to the sampled applicants so the heavy
        # aggregations don't process the full ~13M-13M rows.
        sampled_ids = set(app["SK_ID_CURR"].astype(int).tolist())
        for name in ("bureau", "previous_application", "installments_payments"):
            if name in tables:
                t = tables[name]
                tables[name] = t[t["SK_ID_CURR"].isin(sampled_ids)].reset_index(drop=True)
                print(f"[home_credit] filtered {name} -> {len(tables[name]):,} rows")

    # Protected attributes for fairness analysis.
    age_years = (-app["DAYS_BIRTH"] / 365.0).astype(float)
    protected = pd.DataFrame(
        {
            "SK_ID_CURR": app["SK_ID_CURR"].to_numpy(),
            "gender": app["CODE_GENDER"].astype(str),
            "age_bucket": _bucket_age(age_years),
            "own_car": app["FLAG_OWN_CAR"].astype(str),
            "family_status": app["NAME_FAMILY_STATUS"].astype(str),
        }
    )

    # Tabular features (engineering + one-hot + aggregates from linked tables).
    feats_df, cat_maps = build_tabular_features(tables)
    # Drop the label + identifier from feature matrix.
    y = feats_df["TARGET"].astype(int).to_numpy()
    user_ids = feats_df["SK_ID_CURR"].astype(int).to_numpy()
    feats_df = feats_df.drop(columns=["SK_ID_CURR", "TARGET"])
    # Restrict to numeric / bool columns only (any leftover string columns get dropped).
    feats_df = feats_df.select_dtypes(include=[np.number, "bool"]).astype("float32")
    # Median-impute NaNs (StandardScaler can't handle NaN).
    medians = feats_df.median(numeric_only=True)
    feats_df = feats_df.fillna(medians).fillna(0.0)
    # Drop near-constant columns (variance threshold).
    var = feats_df.var(numeric_only=True)
    keep_var = var[var > 1e-8].index.tolist()
    feats_df = feats_df[keep_var]
    # Feature selection to cap the FT-Transformer sequence length.
    if feats_df.shape[1] > MAX_TAB_FEATURES:
        selector = SelectKBest(score_func=f_classif, k=MAX_TAB_FEATURES)
        selector.fit(feats_df.to_numpy(dtype="float32"), y)
        mask = selector.get_support()
        feats_df = feats_df.loc[:, mask]
        print(f"[home_credit] SelectKBest kept {feats_df.shape[1]} of {len(mask)} features")
    feature_columns = list(feats_df.columns)
    X_tab_raw = feats_df.to_numpy(dtype="float32")

    # StandardScaler (compute manually so we can persist mean/scale cleanly).
    mean = X_tab_raw.mean(axis=0).astype("float32")
    std = X_tab_raw.std(axis=0).astype("float32") + 1e-6
    X_tab = ((X_tab_raw - mean) / std).astype("float32")

    # Sequence features (bureau_balance one-hot history).
    if "bureau" in tables and "bureau_balance" in tables:
        X_seq, seq_mean, seq_std = build_bureau_balance_sequences(
            tables["bureau"], tables["bureau_balance"], user_ids, n_months=SEQ_T_MONTHS
        )
    else:
        X_seq = np.zeros((len(user_ids), SEQ_T_MONTHS, len(BUREAU_STATUS_CODES)), dtype="float32")
        seq_mean = np.zeros(len(BUREAU_STATUS_CODES), dtype="float32")
        seq_std = np.ones(len(BUREAU_STATUS_CODES), dtype="float32")

    return HomeCreditBundle(
        user_ids=user_ids,
        X_tab=X_tab,
        X_seq=X_seq,
        y=y,
        feature_columns=feature_columns,
        scaler_mean=mean,
        scaler_scale=std,
        seq_mean=seq_mean,
        seq_scale=seq_std,
        categorical_maps=cat_maps,
        protected_attrs=protected,
    )
