"""Training loop, orchestration, and group-fairness helpers used during training.

The inner training loop (`train_model`) follows the NovaScore deck:
- AdamW(lr=3e-4, weight_decay=1e-4)
- Cosine annealing over `epochs`
- BCEWithLogitsLoss
- Early stopping on validation AUROC (patience=5)
- Mixed precision (AMP) on CUDA, fp32 on CPU
- Best validation checkpoint is restored before returning.

The outer orchestrator (`run_training`) wires synthesizer → preprocessing →
feature engineering → LightGBM baseline → hybrid training → calibration →
artifact persistence. See `ml/results/` after a run for everything it produces.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

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
from .evaluate import plot_fairness_before_after
from .fairness import (
    compute_all_metrics,
    optimize_thresholds_per_group,
    per_group_tpr,
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
from .models.lightgbm_baseline import (
    score_lightgbm,
    train_lightgbm,
)
from .models.lightgbm_baseline import (
    test_auroc as lgb_test_auroc,
)
from .models.node2vec_embed import compute_user_embeddings


class NovaDS(Dataset):
    """Tensor dataset bundling tabular, weekly sequence, graph embedding, label, group."""

    def __init__(
        self,
        X_tab: np.ndarray,
        X_seq: np.ndarray,
        X_g: np.ndarray,
        y: np.ndarray,
        grp: np.ndarray,
    ) -> None:
        self.X_tab = X_tab.astype("float32")
        self.X_seq = X_seq.astype("float32")
        self.X_g = X_g.astype("float32")
        self.y = y.astype("float32")
        self.grp = grp.astype("int64")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.X_tab[i]),
            torch.from_numpy(self.X_seq[i]),
            torch.from_numpy(self.X_g[i]),
            torch.tensor(self.y[i]),
            torch.tensor(self.grp[i]),
        )


def _val_auroc(
    model: HybridModel, loader: DataLoader, device: str
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for x_tab, x_seq, x_g, yy, _gg in loader:
            x_tab = x_tab.to(device)
            x_seq = x_seq.to(device)
            x_g = x_g.to(device)
            p = torch.sigmoid(model(x_tab, x_seq, x_g)).cpu().numpy()
            ps.append(p)
            ys.append(yy.numpy())
    y_arr = np.concatenate(ys)
    p_arr = np.concatenate(ps)
    auc = roc_auc_score(y_arr, p_arr) if len(np.unique(y_arr)) > 1 else float("nan")
    return float(auc), y_arr, p_arr


def train_model(
    tr_loader: DataLoader,
    va_loader: DataLoader,
    n_tab: int,
    n_seq_features: int = 9,
    g_dim: int = 64,
    epochs: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device: str | None = None,
    verbose: bool = True,
) -> HybridModel:
    """Train HybridModel and return the model with best-val weights loaded.

    Hyperparameters match the NovaScore deck specification:
    AdamW(lr=3e-4, wd=1e-4), cosine LR over `epochs`, early-stop on val AUROC with
    patience=5, mixed precision on CUDA.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(n_tab=n_tab, n_seq_features=n_seq_features, g_dim=g_dim).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        seen = 0
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
            tot += float(loss.item()) * yy.size(0)
            seen += yy.size(0)
        sched.step()
        train_loss = tot / max(seen, 1)

        val_auc, _, _ = _val_auroc(model, va_loader, device)
        if verbose:
            print(f"epoch {ep:02d}  train_loss={train_loss:.4f}  val_auroc={val_auc:.4f}")

        if not np.isnan(val_auc) and val_auc > best_auc + 1e-6:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"early stop at epoch {ep} (best val_auroc={best_auc:.4f})")
                break

    model.load_state_dict(best_state)
    return model


def delta_tpr_at_threshold(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    group_codes: np.ndarray,
    thr: float = 0.5,
) -> float:
    """Mean absolute pairwise difference in TPR across groups at a given threshold.

    Groups with no positive examples have undefined TPR and are skipped. Returns
    0.0 when fewer than two groups have computable TPR (nothing meaningful to
    compare).
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    group_codes = np.asarray(group_codes).astype(int)
    tprs: list[float] = []
    for g in np.unique(group_codes):
        m = group_codes == g
        pos_mask = m & (y_true == 1)
        n_pos = int(pos_mask.sum())
        if n_pos == 0:
            continue
        tp = int(((p_pred >= thr) & pos_mask).sum())
        tprs.append(tp / n_pos)
    if len(tprs) < 2:
        return 0.0
    diffs = [abs(a - b) for i, a in enumerate(tprs) for b in tprs[i + 1 :]]
    return float(np.mean(diffs))


@dataclass
class TrainingConfig:
    """Top-level config for `run_training`. Defaults match the deck spec."""

    data_dir: Path = field(default_factory=lambda: Path("ml/data/synthetic"))
    results_dir: Path = field(default_factory=lambda: Path("ml/results"))
    n_users: int = 10000
    seed: int = 42
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    batch_size: int = 512
    test_size: float = 0.15
    val_size: float = 0.1765  # fraction of train+val pool used as val
    use_graph: bool = True
    device: str | None = None
    skip_lightgbm: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)


def _stratify(y: np.ndarray, min_per_class: int = 2) -> np.ndarray | None:
    vals, cnts = np.unique(y, return_counts=True)
    if len(vals) < 2 or cnts.min() < min_per_class:
        return None
    return y


def _ensure_synth(data_dir: Path, n_users: int, seed: int) -> None:
    """Generate synthetic parquets at data_dir if either trips or txns is missing."""
    trips = data_dir / "trips.parquet"
    txns = data_dir / "txns.parquet"
    if trips.exists() and txns.exists():
        return
    print(f"[data] synthetic parquets missing at {data_dir} — generating n_users={n_users}")
    generate_synth(n_users=n_users, seed=seed, output_dir=data_dir)


def _global_threshold_for_tpr(p_pred: np.ndarray, y_true: np.ndarray, target_tpr: float) -> float:
    """Find the threshold at which overall TPR equals target_tpr (best-effort)."""
    pos = p_pred[y_true == 1]
    if len(pos) == 0:
        return 0.5
    sorted_pos = np.sort(pos)
    # take the threshold so that the top fraction of positives (target_tpr) are above it
    k = max(1, int(np.floor(len(sorted_pos) * (1 - target_tpr))))
    return float(sorted_pos[k - 1])


def _maybe_run_fairness(
    data_dir: Path,
    results_dir: Path,
    user_ids: np.ndarray,
    p_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    A: float,
    B: float,
    target_tpr: float = 0.8,
    mitigation_attribute: str = "vehicle_type",
) -> dict[str, Any] | None:
    """Compute fairness metrics and threshold-based mitigation. Saves JSON + plot.

    No-op (returns None) if users_demo.parquet is not in `data_dir` — the
    pipeline can run on real data that lacks protected-attribute labels.

    The "before" threshold is the *global* cutoff that yields `target_tpr`
    overall; the "after" threshold is computed per group to equalize TPR
    within each group to the same target.
    """
    demo_path = data_dir / "users_demo.parquet"
    if not demo_path.exists():
        print("[fairness] no users_demo.parquet — skipping fairness analysis")
        return None
    demo = pd.read_parquet(demo_path)[["user_id", "gender", "age_bucket", "vehicle_type", "city"]]
    df = pd.DataFrame({"user_id": user_ids, "p": p_pred, "y": y_true}).merge(
        demo, on="user_id", how="left"
    )
    for col in ("gender", "age_bucket", "vehicle_type", "city"):
        df[col] = df[col].fillna("unknown").astype(str)
    groups_dict = {
        col: df[col].to_numpy() for col in ("gender", "age_bucket", "vehicle_type", "city")
    }
    threshold = _global_threshold_for_tpr(df.p.to_numpy(), df.y.to_numpy(), target_tpr)
    print(f"[fairness] global threshold for target_tpr={target_tpr}: {threshold:.4f}")
    before = compute_all_metrics(df.y.to_numpy(), df.p.to_numpy(), groups_dict, threshold=threshold)

    grp = df[mitigation_attribute].to_numpy()
    thr_map = optimize_thresholds_per_group(
        df.y.to_numpy(), df.p.to_numpy(), grp, target_tpr=target_tpr
    )
    # Apply per-group threshold to the mitigated attribute; other attributes keep
    # the global threshold so the comparison isolates the mitigation effect.
    y_pred_after = np.fromiter(
        (int(p >= thr_map.threshold_for(g)) for p, g in zip(df.p.to_numpy(), grp, strict=True)),
        dtype=int,
        count=len(df),
    )

    from .fairness import (
        delta_fpr,
    )
    from .fairness import (
        delta_tpr as _dt,
    )
    from .fairness import (
        demographic_parity_ratio as _dpr,
    )
    from .fairness import (
        equalized_odds_difference as _eod,
    )

    # `after` mirrors `before` for every attribute except the mitigated one, where
    # we recompute metrics from the per-group-thresholded predictions.
    after_rows: list[dict[str, Any]] = []
    for name, g_arr in groups_dict.items():
        if name == mitigation_attribute:
            after_rows.append(
                {
                    "attribute": name,
                    "demographic_parity_ratio": _dpr(y_pred_after, g_arr),
                    "disparate_impact_ratio": _dpr(y_pred_after, g_arr),
                    "delta_tpr": _dt(df.y.to_numpy(), y_pred_after, g_arr),
                    "delta_fpr": delta_fpr(df.y.to_numpy(), y_pred_after, g_arr),
                    "equalized_odds_difference": _eod(df.y.to_numpy(), y_pred_after, g_arr),
                }
            )
        else:
            after_rows.append(before.loc[before.attribute == name].iloc[0].to_dict())
    after = pd.DataFrame(after_rows)

    # Per-group TPR for the plot (before vs after on the mitigated attribute).
    y_pred_before = (df.p.to_numpy() >= threshold).astype(int)
    tpr_before = per_group_tpr(df.y.to_numpy(), y_pred_before, grp)
    tpr_after = per_group_tpr(df.y.to_numpy(), y_pred_after, grp)
    plot_fairness_before_after(
        {str(k): v for k, v in tpr_before.items()},
        {str(k): v for k, v in tpr_after.items()},
        results_dir / "fairness_before_after.png",
        attribute=mitigation_attribute,
    )

    summary = {
        "mitigation_attribute": mitigation_attribute,
        "target_tpr": float(target_tpr),
        "global_threshold_before": float(threshold),
        "thresholds": thr_map.to_dict(),
        "before": before.to_dict(orient="records"),
        "after": after.to_dict(orient="records"),
        "tpr_before": {str(k): float(v) for k, v in tpr_before.items()},
        "tpr_after": {str(k): float(v) for k, v in tpr_after.items()},
        "score_adjustment_per_group": _score_adjustments_dict(thr_map, A, B, threshold),
    }
    save_json(results_dir / "fairness_before_after.json", summary)
    save_json(results_dir / "threshold_map.json", thr_map.to_dict())
    delta_tpr_before = float(before.set_index("attribute").loc[mitigation_attribute, "delta_tpr"])
    delta_tpr_after = float(after.set_index("attribute").loc[mitigation_attribute, "delta_tpr"])
    print(
        f"[fairness] mitigated on {mitigation_attribute}: "
        f"ΔTPR {delta_tpr_before:.3f} -> {delta_tpr_after:.3f}"
    )
    return summary


def _score_adjustments_dict(thr_map, A: float, B: float, baseline: float) -> dict[str, float]:
    """Per-group additive score shift relative to a baseline threshold."""
    from .fairness import score_adjustment_from_threshold

    return {
        str(k): score_adjustment_from_threshold(v, A, B, default_threshold=baseline)
        for k, v in thr_map.thresholds.items()
    }


def run_training(cfg: TrainingConfig) -> dict[str, Any]:
    """End-to-end training pipeline. Persists all artifacts to cfg.results_dir."""
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

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

    # Align tabular features to user_index order; fill missing rows with zeros.
    tab_indexed = users[["user_id"]].merge(tab, on="user_id", how="left").fillna(0.0)
    X_tab_raw = tab_indexed[feature_columns].astype("float32").to_numpy()

    print(f"[seq] building weekly sequences ({T_WEEKS} weeks x {len(SEQ_FEATURES)} features)")
    X_seq, seq_scaler = build_sequences(trips_w, txns_w, users["user_id"], anchor_end, T_WEEKS)

    if cfg.use_graph:
        print("[graph] computing Node2Vec embeddings on user-merchant graph")
        X_graph = compute_user_embeddings(txns_w, users["user_id"], dim=GRAPH_DIM, seed=cfg.seed)
    else:
        X_graph = np.zeros((len(users), GRAPH_DIM), dtype="float32")

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

    # LightGBM baseline on tabular features only.
    booster = None
    p_te_lgb: np.ndarray | None = None
    lgb_info: dict[str, Any] = {}
    if not cfg.skip_lightgbm:
        print("[lgbm] training baseline")
        booster, lgb_info = train_lightgbm(X_tab[tr_idx], y[tr_idx], X_tab[va_idx], y[va_idx])
        p_te_lgb = score_lightgbm(booster, X_tab[te_idx])
        lgb_info["test_auc"] = lgb_test_auroc(booster, X_tab[te_idx], y[te_idx])
        print(
            f"[lgbm] best_iter={lgb_info['best_iteration']} "
            f"val_auc={lgb_info['val_auc']:.4f} test_auc={lgb_info['test_auc']:.4f}"
        )

    # Hybrid model training.
    print("[hybrid] training")
    tr_ds = NovaDS(X_tab[tr_idx], X_seq[tr_idx], X_graph[tr_idx], y[tr_idx], group_codes[tr_idx])
    va_ds = NovaDS(X_tab[va_idx], X_seq[va_idx], X_graph[va_idx], y[va_idx], group_codes[va_idx])
    te_ds = NovaDS(X_tab[te_idx], X_seq[te_idx], X_graph[te_idx], y[te_idx], group_codes[te_idx])
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)
    model = train_model(
        tr_loader,
        va_loader,
        n_tab=X_tab.shape[1],
        n_seq_features=len(SEQ_FEATURES),
        g_dim=GRAPH_DIM,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        device=device,
    )

    # Test predictions (hybrid).
    test_auc, y_te, p_te = _val_auroc(model, te_loader, device)
    dtpr = delta_tpr_at_threshold(y_te, p_te, group_codes[te_idx], thr=0.5)
    print(f"[hybrid] test AUROC={test_auc:.4f}  ΔTPR={dtpr:.4f}")

    # Calibration (analytical + empirical refinement against full-population PDs).
    A, B = solve_score_params()
    full_ds = NovaDS(X_tab, X_seq, X_graph, y, group_codes)
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=False)
    _, _, all_p = _val_auroc(model, full_loader, device)
    a, b = empirical_refinement(all_p, A, B, q_low=0.20, q_high=0.80)
    calib = CalibrationParams(A=A, B=B, a=a, b=b)
    all_scores = apply_calibration(all_p, calib)

    # Persist artifacts.
    rd = cfg.results_dir
    n_params = int(sum(p.numel() for p in model.parameters()))
    save_checkpoint(
        model,
        rd / "checkpoint.pt",
        hparams={
            "n_tab": int(X_tab.shape[1]),
            "d_tab": 256,
            "d_seq": 128,
            "g_dim": GRAPH_DIM,
            "n_seq_features": len(SEQ_FEATURES),
        },
    )
    save_json(rd / "feature_columns.json", feature_columns)
    save_json(rd / "topk.json", learned_topk)
    save_json(rd / "group_categories.json", group_categories)
    save_sequence_scaler(rd / "sequence_scaler.json", seq_scaler)
    save_scaler(rd / "scaler.json", scaler.mean_, scaler.scale_)
    save_calibration(rd / "calibration.json", calib)
    if booster is not None:
        booster.save_model(str(rd / "lightgbm.txt"))
        save_json(
            rd / "feature_importance.json",
            lgb_feature_importance(booster, feature_columns),
        )

    test_pred_df = pd.DataFrame(
        {
            "user_id": users["user_id"].to_numpy()[te_idx],
            "y_true": y_te.astype(int),
            "pd90": p_te,
            "novascore": np.round(apply_calibration(p_te, calib), 1),
            "group": pd.Categorical.from_codes(
                group_codes[te_idx], categories=group_categories
            ).astype(str),
        }
    )
    test_pred_df.to_csv(rd / "test_predictions.csv", index=False)
    np.save(rd / "all_probs.npy", all_p)
    np.save(rd / "all_scores.npy", all_scores)
    if p_te_lgb is not None:
        np.save(rd / "test_probs_lightgbm.npy", p_te_lgb)

    fairness_summary = _maybe_run_fairness(
        cfg.data_dir,
        rd,
        users["user_id"].to_numpy(),
        all_p,
        y,
        A=A,
        B=B,
    )

    metrics = {
        "hybrid_test_auroc": float(test_auc),
        "hybrid_delta_tpr_at_0.5": float(dtpr),
        "lightgbm_test_auroc": float(lgb_info.get("test_auc", float("nan"))),
        "n_users": int(len(users)),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "positive_rate_full": float(np.mean(y)),
        "positive_rate_test": float(np.mean(y_te)),
        "score_distribution": {
            "Bronze": float((all_scores < 600).mean()),
            "Silver": float(((all_scores >= 600) & (all_scores < 700)).mean()),
            "Gold": float(((all_scores >= 700) & (all_scores < 800)).mean()),
            "Platinum": float((all_scores >= 800).mean()),
        },
        "hybrid_parameter_count": n_params,
        "feature_count": int(X_tab.shape[1]),
        "seed": int(cfg.seed),
        "epochs_config": int(cfg.epochs),
        "fairness": fairness_summary,
    }
    save_json(rd / "metrics.json", metrics)
    print(f"[done] artifacts written to {rd}")
    print(json.dumps(metrics, indent=2))
    return metrics
