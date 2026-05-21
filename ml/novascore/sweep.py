"""Phase 4.5: HP sweep + LightGBM grid + ensemble training pipeline.

Top-level function `run_phase45_pipeline(cfg)`:
1. Loads and preprocesses the data once (shared across all trials).
2. Runs a 27-config grid search on LightGBM, picks best by val AUROC.
3. Runs a 15-trial random search on the hybrid model from a fixed seed (so the
   sweep is reproducible), picks best by val AUROC.
4. Builds two ensembles (simple-avg, val-AUROC-weighted), picks the one with
   higher val AUROC and reports its test AUROC.
5. Refreshes empirical calibration using the chosen ensemble's probability
   distribution.
6. Re-runs the per-group fairness mitigation against the ensemble predictions.
7. Persists all artifacts (sweep history, ensemble predictions, plots, etc.)
   to ml/results/.

Hard rule (from the Phase 4.5 spec): all reported metrics come from real
training runs. We never re-train a config to fish for a better number.
"""

from __future__ import annotations

import itertools
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from . import GRAPH_DIM, SEQ_FEATURES, T_WEEKS
from .calibration import (
    CalibrationParams,
    apply_calibration,
    empirical_refinement,
    solve_score_params,
)
from .data.preprocessing import (
    apply_window,
    compute_labels,
    compute_tabular_features,
    compute_user_city_group,
    load_parquets,
    parse_datetimes,
)
from .data.sequences import build_sequences
from .data.synthesize import generate as generate_synth
from .evaluate import (
    plot_calibration,
    plot_fairness_before_after,
    plot_feature_importance,
    plot_roc,
)
from .fairness import (
    compute_all_metrics,
    delta_fpr,
    delta_tpr,
    demographic_parity_ratio,
    equalized_odds_difference,
    optimize_thresholds_per_group,
    per_group_tpr,
    score_adjustment_from_threshold,
)
from .io import (
    save_calibration,
    save_checkpoint,
    save_json,
    save_scaler,
    save_sequence_scaler,
)
from .models.hybrid import HybridModel
from .models.lightgbm_baseline import (
    feature_importance as lgb_feature_importance,
)
from .train import NovaDS, _val_auroc, train_model

# ---------------------------------------------------------------------------
# Search spaces.

HP_SEARCH_SPACE: dict[str, list[Any]] = {
    "d_tab": [128, 256, 384],
    "d_seq": [64, 128],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4],
    "n_layers": [2, 3],
}

LGB_GRID: dict[str, list[Any]] = {
    "num_leaves": [31, 64, 128],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_samples": [20, 50, 100],
}


@dataclass
class SweepConfig:
    data_dir: Path = field(default_factory=lambda: Path("ml/data/synthetic"))
    results_dir: Path = field(default_factory=lambda: Path("ml/results"))
    n_users: int = 10000
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.1765
    n_hp_trials: int = 15
    epochs: int = 15
    patience: int = 5
    batch_size: int = 512
    use_graph: bool = False  # Phase 4.5 default: graph dropped from production.
    target_tpr: float = 0.8
    mitigation_attribute: str = "vehicle_type"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)


@dataclass
class PreparedData:
    user_ids: np.ndarray
    X_tab: np.ndarray
    X_seq: np.ndarray
    X_g: np.ndarray
    y: np.ndarray
    group_codes: np.ndarray
    group_categories: list[str]
    feature_columns: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    sequence_scaler: Any
    learned_topk: dict[str, list[str]]
    tr_idx: np.ndarray
    va_idx: np.ndarray
    te_idx: np.ndarray
    g_dim: int


def _ensure_synth(data_dir: Path, n_users: int, seed: int) -> None:
    trips = data_dir / "trips.parquet"
    txns = data_dir / "txns.parquet"
    if trips.exists() and txns.exists():
        return
    print(f"[data] missing parquets at {data_dir}; generating n_users={n_users}")
    generate_synth(n_users=n_users, seed=seed, output_dir=data_dir)


def _stratify(y: np.ndarray, min_per_class: int = 2):
    vals, cnts = np.unique(y, return_counts=True)
    if len(vals) < 2 or cnts.min() < min_per_class:
        return None
    return y


def prepare_data(cfg: SweepConfig) -> PreparedData:
    """End-to-end data preparation; runs once, shared across all sweep trials."""
    _ensure_synth(cfg.data_dir, cfg.n_users, cfg.seed)
    print(f"[data] loading parquets from {cfg.data_dir}")
    trips, txns = load_parquets(cfg.data_dir / "trips.parquet", cfg.data_dir / "txns.parquet")
    trips, txns = parse_datetimes(trips, txns)
    trips_w, txns_w, anchor_end = apply_window(trips, txns)
    print(f"[data] window rows -> trips: {len(trips_w):,}  txns: {len(txns_w):,}")

    labels = compute_labels(txns_w)
    print(f"[data] positive rate: {labels['y'].mean():.3f}")
    tab, learned_topk = compute_tabular_features(trips_w, txns_w)
    feature_columns = [c for c in tab.columns if c != "user_id"]

    users = pd.DataFrame({"user_id": sorted(set(tab["user_id"]) | set(labels["user_id"]))})
    users = users.merge(labels, on="user_id", how="left").fillna({"y": 0})
    users["y"] = users["y"].astype(int)
    users = users.merge(
        compute_user_city_group(txns_w, users["user_id"], top_k=10),
        on="user_id",
        how="left",
    )
    users["city_grp"] = users["city_grp"].fillna("other")

    tab_indexed = users[["user_id"]].merge(tab, on="user_id", how="left").fillna(0.0)
    X_tab_raw = tab_indexed[feature_columns].astype("float32").to_numpy()

    print(f"[seq] building weekly sequences ({T_WEEKS}w × {len(SEQ_FEATURES)}f)")
    X_seq, seq_scaler = build_sequences(trips_w, txns_w, users["user_id"], anchor_end, T_WEEKS)

    g_dim = GRAPH_DIM if cfg.use_graph else 0
    X_graph = np.zeros((len(users), GRAPH_DIM), dtype="float32")
    # Graph training is skipped per Phase 4.5 ablation; X_graph kept for shape
    # compatibility with NovaDS but the model with g_dim=0 ignores it.

    scaler = StandardScaler()
    X_tab = scaler.fit_transform(X_tab_raw).astype("float32")
    y = users["y"].to_numpy().astype("int64")
    grp_cat = users["city_grp"].astype("category")
    group_codes = grp_cat.cat.codes.to_numpy()
    group_categories = grp_cat.cat.categories.tolist()

    idx = np.arange(len(users))
    tr_idx, te_idx = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.seed, shuffle=True, stratify=_stratify(y)
    )
    tr_idx, va_idx = train_test_split(
        tr_idx,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        shuffle=True,
        stratify=_stratify(y[tr_idx]),
    )
    print(f"[split] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    return PreparedData(
        user_ids=users["user_id"].to_numpy(),
        X_tab=X_tab,
        X_seq=X_seq,
        X_g=X_graph,
        y=y,
        group_codes=group_codes,
        group_categories=group_categories,
        feature_columns=feature_columns,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        sequence_scaler=seq_scaler,
        learned_topk=learned_topk,
        tr_idx=tr_idx,
        va_idx=va_idx,
        te_idx=te_idx,
        g_dim=g_dim,
    )


# ---------------------------------------------------------------------------
# LightGBM grid search.


def lgb_grid_search(data: PreparedData) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    import lightgbm as lgb

    history: list[dict[str, Any]] = []
    best = {"val_auc": -1.0, "cfg": None, "booster": None, "test_auc": None}

    keys = list(LGB_GRID)
    grid = list(itertools.product(*(LGB_GRID[k] for k in keys)))
    print(f"[lgbm-grid] {len(grid)} configs")

    pos_rate = max(float(np.mean(data.y[data.tr_idx])), 1e-6)
    for i, combo in enumerate(grid, 1):
        params: dict[str, Any] = {keys[j]: combo[j] for j in range(len(keys))}
        params.update(
            dict(
                objective="binary",
                metric="auc",
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=1,
                verbosity=-1,
                scale_pos_weight=(1 - pos_rate) / pos_rate,
            )
        )
        dtr = lgb.Dataset(data.X_tab[data.tr_idx], label=data.y[data.tr_idx])
        dva = lgb.Dataset(
            data.X_tab[data.va_idx], label=data.y[data.va_idx], reference=dtr
        )
        cb = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
        b = lgb.train(
            params=params,
            train_set=dtr,
            valid_sets=[dva],
            num_boost_round=2000,
            callbacks=cb,
        )
        val_p = b.predict(data.X_tab[data.va_idx], num_iteration=b.best_iteration)
        te_p = b.predict(data.X_tab[data.te_idx], num_iteration=b.best_iteration)
        v_auc = float(roc_auc_score(data.y[data.va_idx], val_p))
        t_auc = float(roc_auc_score(data.y[data.te_idx], te_p))
        # Strip non-pickleable callbacks before saving.
        cfg_logged = {k: params[k] for k in keys}
        cfg_logged["best_iteration"] = int(b.best_iteration)
        history.append({"config": cfg_logged, "val_auc": v_auc, "test_auc": t_auc})
        print(f"[lgbm-grid] {i:2d}/{len(grid)}  {cfg_logged}  val={v_auc:.4f}  test={t_auc:.4f}")
        if v_auc > best["val_auc"]:
            best = {"val_auc": v_auc, "cfg": cfg_logged, "booster": b, "test_auc": t_auc}

    return best["booster"], {**best["cfg"], "val_auc": best["val_auc"], "test_auc": best["test_auc"]}, history


# ---------------------------------------------------------------------------
# Hybrid HP random search.


def _sample_hp(rng: random.Random) -> dict[str, Any]:
    return {k: rng.choice(v) for k, v in HP_SEARCH_SPACE.items()}


def _hybrid_predict_full(model: HybridModel, data: PreparedData, idx: np.ndarray, device: str) -> np.ndarray:
    ds = NovaDS(data.X_tab[idx], data.X_seq[idx], data.X_g[idx], data.y[idx], data.group_codes[idx])
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    _, _, p = _val_auroc(model, loader, device)
    return p


def hp_sweep_hybrid(
    data: PreparedData,
    cfg: SweepConfig,
    device: str,
) -> tuple[HybridModel, dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(cfg.seed)
    # Pre-sample distinct configs to avoid duplicate trials.
    all_configs = list(itertools.product(*HP_SEARCH_SPACE.values()))
    rng.shuffle(all_configs)
    trial_configs = [dict(zip(HP_SEARCH_SPACE.keys(), c)) for c in all_configs[: cfg.n_hp_trials]]

    history: list[dict[str, Any]] = []
    best = {"val_auc": -1.0, "cfg": None, "model": None, "test_auc": None, "val_p": None, "test_p": None}

    for i, hp in enumerate(trial_configs, 1):
        t0 = time.time()
        tr_loader = DataLoader(
            NovaDS(
                data.X_tab[data.tr_idx],
                data.X_seq[data.tr_idx],
                data.X_g[data.tr_idx],
                data.y[data.tr_idx],
                data.group_codes[data.tr_idx],
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        va_loader = DataLoader(
            NovaDS(
                data.X_tab[data.va_idx],
                data.X_seq[data.va_idx],
                data.X_g[data.va_idx],
                data.y[data.va_idx],
                data.group_codes[data.va_idx],
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        # Seed torch per trial for reproducibility, with the trial index folded in.
        torch.manual_seed(cfg.seed * 100 + i)
        np.random.seed(cfg.seed * 100 + i)

        model = _train_hybrid_with_hp(
            data=data,
            cfg=cfg,
            hp=hp,
            tr_loader=tr_loader,
            va_loader=va_loader,
            device=device,
        )

        val_p = _hybrid_predict_full(model, data, data.va_idx, device)
        test_p = _hybrid_predict_full(model, data, data.te_idx, device)
        v_auc = float(roc_auc_score(data.y[data.va_idx], val_p))
        t_auc = float(roc_auc_score(data.y[data.te_idx], test_p))
        n_params = int(sum(p.numel() for p in model.parameters()))
        history.append(
            {
                "trial": i,
                "config": hp,
                "val_auc": v_auc,
                "test_auc": t_auc,
                "n_params": n_params,
                "wall_s": round(time.time() - t0, 2),
            }
        )
        print(
            f"[hp-sweep] {i:2d}/{cfg.n_hp_trials}  {hp}  "
            f"val={v_auc:.4f}  test={t_auc:.4f}  params={n_params/1e6:.2f}M  "
            f"t={time.time() - t0:.1f}s"
        )
        if v_auc > best["val_auc"]:
            best = {
                "val_auc": v_auc,
                "cfg": hp,
                "model": model,
                "test_auc": t_auc,
                "val_p": val_p,
                "test_p": test_p,
            }

    best_summary = {
        **best["cfg"],
        "val_auc": best["val_auc"],
        "test_auc": best["test_auc"],
    }
    return best["model"], best_summary, history


def _train_hybrid_with_hp(
    *,
    data: PreparedData,
    cfg: SweepConfig,
    hp: dict[str, Any],
    tr_loader: DataLoader,
    va_loader: DataLoader,
    device: str,
) -> HybridModel:
    """Inline training loop honoring per-trial HP (d_tab, d_seq, n_layers, etc.)."""
    import copy

    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = HybridModel(
        n_tab=data.X_tab.shape[1],
        d_tab=hp["d_tab"],
        d_seq=hp["d_seq"],
        g_dim=data.g_dim,
        n_seq_features=len(SEQ_FEATURES),
        n_layers=hp["n_layers"],
        dropout=hp["dropout"],
    ).to(device)
    opt = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for x_tab, x_seq, x_g, yy, _gg in tr_loader:
            x_tab = x_tab.to(device)
            x_seq = x_seq.to(device)
            x_g = x_g.to(device)
            yy = yy.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x_tab, x_seq, x_g)
                loss = loss_fn(logits, yy)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        val_auc, _, _ = _val_auroc(model, va_loader, device)
        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Ensemble + final pipeline.


def _build_ensemble(
    p_hybrid_val: np.ndarray,
    p_hybrid_test: np.ndarray,
    p_lgb_val: np.ndarray,
    p_lgb_test: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Build simple-avg + val-AUROC-weighted ensembles; pick best on val."""
    auc_h = float(roc_auc_score(y_val, p_hybrid_val))
    auc_l = float(roc_auc_score(y_val, p_lgb_val))

    # Simple average.
    simple_val = 0.5 * p_hybrid_val + 0.5 * p_lgb_val
    simple_test = 0.5 * p_hybrid_test + 0.5 * p_lgb_test
    simple_val_auc = float(roc_auc_score(y_val, simple_val))
    simple_test_auc = float(roc_auc_score(y_test, simple_test))

    # AUROC-weighted average (subtract 0.5 so random-guessers get zero weight).
    wh = max(auc_h - 0.5, 1e-6)
    wl = max(auc_l - 0.5, 1e-6)
    wsum = wh + wl
    wh /= wsum
    wl /= wsum
    weighted_val = wh * p_hybrid_val + wl * p_lgb_val
    weighted_test = wh * p_hybrid_test + wl * p_lgb_test
    weighted_val_auc = float(roc_auc_score(y_val, weighted_val))
    weighted_test_auc = float(roc_auc_score(y_test, weighted_test))

    # Pick by val AUROC.
    if weighted_val_auc >= simple_val_auc:
        chosen = "weighted"
        val_p = weighted_val
        test_p = weighted_test
        val_auc = weighted_val_auc
        test_auc = weighted_test_auc
        weights = {"hybrid": wh, "lightgbm": wl}
    else:
        chosen = "simple"
        val_p = simple_val
        test_p = simple_test
        val_auc = simple_val_auc
        test_auc = simple_test_auc
        weights = {"hybrid": 0.5, "lightgbm": 0.5}

    return {
        "chosen": chosen,
        "weights": weights,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "val_p": val_p,
        "test_p": test_p,
        "simple_val_auc": simple_val_auc,
        "simple_test_auc": simple_test_auc,
        "weighted_val_auc": weighted_val_auc,
        "weighted_test_auc": weighted_test_auc,
        "hybrid_val_auc": auc_h,
        "lightgbm_val_auc": auc_l,
    }


def _ensemble_probs_full(
    hybrid_model: HybridModel,
    lgb_booster,
    data: PreparedData,
    weights: dict[str, float],
    device: str,
) -> np.ndarray:
    """Compute the chosen ensemble's probabilities over the full population."""
    full_idx = np.arange(len(data.y))
    p_h = _hybrid_predict_full(hybrid_model, data, full_idx, device)
    p_l = lgb_booster.predict(data.X_tab, num_iteration=lgb_booster.best_iteration)
    return weights["hybrid"] * p_h + weights["lightgbm"] * p_l


def _global_threshold_for_tpr(p_pred: np.ndarray, y_true: np.ndarray, target_tpr: float) -> float:
    pos = p_pred[y_true == 1]
    if len(pos) == 0:
        return 0.5
    sorted_pos = np.sort(pos)
    k = max(1, int(np.floor(len(sorted_pos) * (1 - target_tpr))))
    return float(sorted_pos[k - 1])


def _run_fairness(
    data: PreparedData,
    p_pred_full: np.ndarray,
    A: float,
    B: float,
    cfg: SweepConfig,
    results_dir: Path,
) -> dict[str, Any] | None:
    demo_path = cfg.data_dir / "users_demo.parquet"
    if not demo_path.exists():
        return None
    demo = pd.read_parquet(demo_path)[["user_id", "gender", "age_bucket", "vehicle_type", "city"]]
    df = pd.DataFrame({"user_id": data.user_ids, "p": p_pred_full, "y": data.y}).merge(
        demo, on="user_id", how="left"
    )
    for col in ("gender", "age_bucket", "vehicle_type", "city"):
        df[col] = df[col].fillna("unknown").astype(str)
    groups_dict = {col: df[col].to_numpy() for col in ("gender", "age_bucket", "vehicle_type", "city")}
    threshold = _global_threshold_for_tpr(df.p.to_numpy(), df.y.to_numpy(), cfg.target_tpr)
    print(f"[fairness] global threshold for TPR={cfg.target_tpr}: {threshold:.4f}")

    before = compute_all_metrics(df.y.to_numpy(), df.p.to_numpy(), groups_dict, threshold=threshold)
    grp = df[cfg.mitigation_attribute].to_numpy()
    thr_map = optimize_thresholds_per_group(
        df.y.to_numpy(), df.p.to_numpy(), grp, target_tpr=cfg.target_tpr
    )
    y_pred_after = np.fromiter(
        (int(p >= thr_map.threshold_for(g)) for p, g in zip(df.p.to_numpy(), grp, strict=True)),
        dtype=int,
        count=len(df),
    )
    after_rows: list[dict[str, Any]] = []
    for name, g_arr in groups_dict.items():
        if name == cfg.mitigation_attribute:
            after_rows.append(
                {
                    "attribute": name,
                    "demographic_parity_ratio": demographic_parity_ratio(y_pred_after, g_arr),
                    "disparate_impact_ratio": demographic_parity_ratio(y_pred_after, g_arr),
                    "delta_tpr": delta_tpr(df.y.to_numpy(), y_pred_after, g_arr),
                    "delta_fpr": delta_fpr(df.y.to_numpy(), y_pred_after, g_arr),
                    "equalized_odds_difference": equalized_odds_difference(
                        df.y.to_numpy(), y_pred_after, g_arr
                    ),
                }
            )
        else:
            after_rows.append(before.loc[before.attribute == name].iloc[0].to_dict())
    after = pd.DataFrame(after_rows)

    y_pred_before = (df.p.to_numpy() >= threshold).astype(int)
    tpr_before = per_group_tpr(df.y.to_numpy(), y_pred_before, grp)
    tpr_after = per_group_tpr(df.y.to_numpy(), y_pred_after, grp)
    plot_fairness_before_after(
        {str(k): v for k, v in tpr_before.items()},
        {str(k): v for k, v in tpr_after.items()},
        results_dir / "fairness_before_after.png",
        attribute=cfg.mitigation_attribute,
    )
    summary = {
        "mitigation_attribute": cfg.mitigation_attribute,
        "target_tpr": float(cfg.target_tpr),
        "global_threshold_before": float(threshold),
        "thresholds": thr_map.to_dict(),
        "before": before.to_dict(orient="records"),
        "after": after.to_dict(orient="records"),
        "tpr_before": {str(k): float(v) for k, v in tpr_before.items()},
        "tpr_after": {str(k): float(v) for k, v in tpr_after.items()},
        "score_adjustment_per_group": {
            str(k): score_adjustment_from_threshold(v, A, B, default_threshold=threshold)
            for k, v in thr_map.thresholds.items()
        },
    }
    save_json(results_dir / "fairness_before_after.json", summary)
    save_json(results_dir / "threshold_map.json", thr_map.to_dict())
    return summary


def run_phase45_pipeline(cfg: SweepConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SweepConfig()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()

    print("=" * 70)
    print("PHASE 4.5 — HP sweep + ensemble + recalibrate")
    print("=" * 70)
    data = prepare_data(cfg)

    # --- LightGBM grid ------------------------------------------------------
    print("\n--- LightGBM grid search ---")
    t = time.time()
    best_lgb, best_lgb_cfg, lgb_history = lgb_grid_search(data)
    print(f"[lgbm-grid] best val_auc={best_lgb_cfg['val_auc']:.4f}  test_auc={best_lgb_cfg['test_auc']:.4f}")
    print(f"[lgbm-grid] took {time.time() - t:.1f}s")
    p_lgb_val = best_lgb.predict(data.X_tab[data.va_idx], num_iteration=best_lgb.best_iteration)
    p_lgb_test = best_lgb.predict(data.X_tab[data.te_idx], num_iteration=best_lgb.best_iteration)

    # --- Hybrid HP sweep ----------------------------------------------------
    print("\n--- Hybrid HP random search ---")
    t = time.time()
    best_hybrid, best_hybrid_cfg, hp_history = hp_sweep_hybrid(data, cfg, device)
    print(
        f"[hp-sweep] best val_auc={best_hybrid_cfg['val_auc']:.4f}  "
        f"test_auc={best_hybrid_cfg['test_auc']:.4f}"
    )
    print(f"[hp-sweep] took {time.time() - t:.1f}s")
    p_hyb_val = _hybrid_predict_full(best_hybrid, data, data.va_idx, device)
    p_hyb_test = _hybrid_predict_full(best_hybrid, data, data.te_idx, device)

    # --- Ensemble -----------------------------------------------------------
    print("\n--- Ensemble ---")
    ens = _build_ensemble(
        p_hyb_val, p_hyb_test, p_lgb_val, p_lgb_test, data.y[data.va_idx], data.y[data.te_idx]
    )
    print(
        f"[ensemble] chosen={ens['chosen']}  weights={ens['weights']}  "
        f"val={ens['val_auc']:.4f}  test={ens['test_auc']:.4f}"
    )
    p_ens_full = _ensemble_probs_full(best_hybrid, best_lgb, data, ens["weights"], device)

    # --- Calibration refresh -----------------------------------------------
    A, B = solve_score_params()
    a, b = empirical_refinement(p_ens_full, A, B, q_low=0.20, q_high=0.80)
    calib = CalibrationParams(A=A, B=B, a=a, b=b)
    all_scores = apply_calibration(p_ens_full, calib)
    score_dist = {
        "Bronze": float((all_scores < 600).mean()),
        "Silver": float(((all_scores >= 600) & (all_scores < 700)).mean()),
        "Gold": float(((all_scores >= 700) & (all_scores < 800)).mean()),
        "Platinum": float((all_scores >= 800).mean()),
    }
    print(f"[calib] score distribution: {score_dist}")

    # --- Fairness re-run ---------------------------------------------------
    print("\n--- Fairness re-run on ensemble predictions ---")
    fairness_summary = _run_fairness(data, p_ens_full, A, B, cfg, cfg.results_dir)
    if fairness_summary:
        before_dtpr = next(
            r["delta_tpr"] for r in fairness_summary["before"] if r["attribute"] == cfg.mitigation_attribute
        )
        after_dtpr = next(
            r["delta_tpr"] for r in fairness_summary["after"] if r["attribute"] == cfg.mitigation_attribute
        )
        print(f"[fairness] {cfg.mitigation_attribute} ΔTPR {before_dtpr:.4f} → {after_dtpr:.4f}")

    # --- Persist all artifacts ---------------------------------------------
    rd = cfg.results_dir
    n_params = int(sum(p.numel() for p in best_hybrid.parameters()))
    save_checkpoint(
        best_hybrid,
        rd / "checkpoint.pt",
        hparams={
            "n_tab": int(data.X_tab.shape[1]),
            "d_tab": int(best_hybrid_cfg["d_tab"]),
            "d_seq": int(best_hybrid_cfg["d_seq"]),
            "g_dim": int(data.g_dim),
            "n_seq_features": len(SEQ_FEATURES),
            "n_layers": int(best_hybrid_cfg["n_layers"]),
            "dropout": float(best_hybrid_cfg["dropout"]),
        },
    )
    best_lgb.save_model(str(rd / "lightgbm.txt"))
    save_json(rd / "feature_columns.json", data.feature_columns)
    save_json(rd / "topk.json", data.learned_topk)
    save_json(rd / "group_categories.json", data.group_categories)
    save_sequence_scaler(rd / "sequence_scaler.json", data.sequence_scaler)
    save_scaler(rd / "scaler.json", data.scaler_mean, data.scaler_scale)
    save_calibration(rd / "calibration.json", calib)
    save_json(
        rd / "feature_importance.json",
        lgb_feature_importance(best_lgb, data.feature_columns),
    )
    save_json(
        rd / "hp_sweep.json",
        {
            "hybrid_sweep": hp_history,
            "lightgbm_grid": lgb_history,
            "best_hybrid": best_hybrid_cfg,
            "best_lightgbm": best_lgb_cfg,
            "ensemble": {k: v for k, v in ens.items() if k not in ("val_p", "test_p")},
        },
    )
    ens_weights = ens["weights"]
    save_json(rd / "ensemble.json", {"weights": ens_weights, "chosen": ens["chosen"]})

    pred_df = pd.DataFrame(
        {
            "user_id": data.user_ids[data.te_idx],
            "y_true": data.y[data.te_idx],
            "pd90_hybrid": p_hyb_test,
            "pd90_lightgbm": p_lgb_test,
            "pd90_ensemble": ens["test_p"],
            "novascore": np.round(apply_calibration(ens["test_p"], calib), 1),
            "group": pd.Categorical.from_codes(
                data.group_codes[data.te_idx], categories=data.group_categories
            ).astype(str),
        }
    )
    pred_df.to_csv(rd / "test_predictions.csv", index=False)
    np.save(rd / "all_probs.npy", p_ens_full)
    np.save(rd / "all_scores.npy", all_scores)
    np.save(rd / "test_probs_lightgbm.npy", p_lgb_test)
    np.save(rd / "test_probs_hybrid.npy", p_hyb_test)
    np.save(rd / "test_probs_ensemble.npy", ens["test_p"])

    # Plots — three-way ROC overlay, refreshed calibration, importance.
    _plot_roc_three(
        data.y[data.te_idx],
        p_hyb_test,
        p_lgb_test,
        ens["test_p"],
        rd / "roc_curve.png",
    )
    plot_calibration(data.y[data.te_idx], ens["test_p"], rd / "calibration_plot.png")
    plot_feature_importance(
        lgb_feature_importance(best_lgb, data.feature_columns),
        rd / "feature_importance.png",
    )

    metrics = {
        "headline_test_auroc": float(ens["test_auc"]),
        "ensemble_strategy": ens["chosen"],
        "ensemble_weights": ens_weights,
        "hybrid_best_test_auroc": float(best_hybrid_cfg["test_auc"]),
        "hybrid_best_val_auroc": float(best_hybrid_cfg["val_auc"]),
        "hybrid_best_config": {k: best_hybrid_cfg[k] for k in HP_SEARCH_SPACE},
        "lightgbm_best_test_auroc": float(best_lgb_cfg["test_auc"]),
        "lightgbm_best_val_auroc": float(best_lgb_cfg["val_auc"]),
        "lightgbm_best_config": {k: best_lgb_cfg[k] for k in LGB_GRID},
        "n_users": int(len(data.user_ids)),
        "n_train": int(len(data.tr_idx)),
        "n_val": int(len(data.va_idx)),
        "n_test": int(len(data.te_idx)),
        "positive_rate_full": float(np.mean(data.y)),
        "score_distribution": score_dist,
        "hybrid_parameter_count": n_params,
        "feature_count": int(data.X_tab.shape[1]),
        "seed": int(cfg.seed),
        "fairness": fairness_summary,
        "phase": "4.5",
        "graph_disabled": True,
        "total_wall_seconds": round(time.time() - t_start, 1),
    }
    save_json(rd / "metrics.json", metrics)
    print("\n" + "=" * 70)
    print("PHASE 4.5 SUMMARY")
    print("=" * 70)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("fairness",)}, indent=2))
    print(f"\nTotal wall time: {metrics['total_wall_seconds']}s")
    return metrics


def _plot_roc_three(
    y_true: np.ndarray,
    p_hybrid: np.ndarray,
    p_lgb: np.ndarray,
    p_ens: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    for name, p, color in (
        ("Hybrid", p_hybrid, "#0B1628"),
        ("LightGBM", p_lgb, "#C9A26F"),
        ("Ensemble", p_ens, "#1A8F3B"),
    ):
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = float(roc_auc_score(y_true, p))
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Hybrid vs LightGBM vs Ensemble")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    run_phase45_pipeline()
