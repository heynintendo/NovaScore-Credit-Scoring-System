"""End-to-end Home Credit Default Risk training pipeline.

This is the post-Phase-4.5 production path. Replaces the synthetic-data
orchestration in sweep.py for the headline NovaScore results. The architecture
is unchanged from earlier phases (FT-Transformer + TCN fusion model + LightGBM
baseline + ensemble); only the data wiring + fairness attributes changed.

CLI: `novascore train --dataset home_credit` invokes `run_home_credit_pipeline`.
"""

from __future__ import annotations

import copy
import gc
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import CosineAnnealingLR  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .calibration import (  # noqa: E402
    CalibrationParams,
    apply_calibration,
    empirical_refinement,
    solve_score_params,
)
from .data.home_credit import HomeCreditBundle, prepare_home_credit  # noqa: E402
from .evaluate import (  # noqa: E402
    plot_calibration,
    plot_fairness_before_after,
    plot_feature_importance,
)
from .fairness import (  # noqa: E402
    compute_all_metrics,
    delta_fpr,
    delta_tpr,
    demographic_parity_ratio,
    equalized_odds_difference,
    optimize_thresholds_per_group,
    per_group_tpr,
    score_adjustment_from_threshold,
)
from .io import (  # noqa: E402
    save_calibration,
    save_checkpoint,
    save_json,
    save_scaler,
)
from .models.hybrid import HybridModel  # noqa: E402
from .models.lightgbm_baseline import feature_importance as lgb_feature_importance  # noqa: E402
from .train import NovaDS, _val_auroc  # noqa: E402

HP_SEARCH_SPACE: dict[str, list[Any]] = {
    "d_tab": [128, 256, 384],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-4, 3e-4, 1e-3],
}


@dataclass
class HomeCreditConfig:
    data_dir: Path = field(default_factory=lambda: Path("ml/data/home_credit"))
    results_dir: Path = field(default_factory=lambda: Path("ml/results"))
    seed: int = 42
    sample_n: int | None = None  # set for quick smoke runs
    test_size: float = 0.15
    val_size: float = 0.1765  # → ~70/15/15
    n_hp_trials: int = 4
    epochs: int = 12
    patience: int = 4
    batch_size: int = 128  # Phase 4.6 guardrail: 512 OOM'd on macOS MPS at 60K sample.
    d_seq: int = 128
    weight_decay: float = 1e-4
    target_tpr: float = 0.8

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)


@dataclass
class _SplitData:
    bundle: HomeCreditBundle
    tr_idx: np.ndarray
    va_idx: np.ndarray
    te_idx: np.ndarray


def _split(bundle: HomeCreditBundle, cfg: HomeCreditConfig) -> _SplitData:
    idx = np.arange(len(bundle.user_ids))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        shuffle=True,
        stratify=bundle.y,
    )
    tr_idx, va_idx = train_test_split(
        tr_idx,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        shuffle=True,
        stratify=bundle.y[tr_idx],
    )
    print(f"[split] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    return _SplitData(bundle=bundle, tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx)


# ---------------------------------------------------------------------------
# LightGBM baseline


def _train_lightgbm_baseline(d: _SplitData) -> tuple[lgb.Booster, dict[str, Any]]:
    """Single-config LightGBM baseline with strong defaults and early stopping."""
    pos_rate = max(float(np.mean(d.bundle.y[d.tr_idx])), 1e-6)
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbosity=-1,
        scale_pos_weight=(1 - pos_rate) / pos_rate,
    )
    dtr = lgb.Dataset(d.bundle.X_tab[d.tr_idx], label=d.bundle.y[d.tr_idx])
    dva = lgb.Dataset(d.bundle.X_tab[d.va_idx], label=d.bundle.y[d.va_idx], reference=dtr)
    cb = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
    b = lgb.train(
        params=params,
        train_set=dtr,
        valid_sets=[dva],
        num_boost_round=3000,
        callbacks=cb,
    )
    val_p = b.predict(d.bundle.X_tab[d.va_idx], num_iteration=b.best_iteration)
    te_p = b.predict(d.bundle.X_tab[d.te_idx], num_iteration=b.best_iteration)
    info = {
        "best_iteration": int(b.best_iteration),
        "val_auc": float(roc_auc_score(d.bundle.y[d.va_idx], val_p)),
        "test_auc": float(roc_auc_score(d.bundle.y[d.te_idx], te_p)),
        "config": {k: params[k] for k in ("learning_rate", "num_leaves", "min_child_samples")},
    }
    print(
        f"[lgbm] best_iter={info['best_iteration']} val={info['val_auc']:.4f} test={info['test_auc']:.4f}"
    )
    return b, info


# ---------------------------------------------------------------------------
# Hybrid HP sweep


def _empty_device_cache(device: str) -> None:
    """Free unused MPS / CUDA buffers between epochs and trials."""
    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device == "cuda":
        torch.cuda.empty_cache()


def _train_hybrid_with_hp(
    d: _SplitData,
    cfg: HomeCreditConfig,
    hp: dict[str, Any],
    device: str,
) -> HybridModel:
    tr_loader = DataLoader(
        NovaDS(
            d.bundle.X_tab[d.tr_idx],
            d.bundle.X_seq[d.tr_idx],
            np.zeros((len(d.tr_idx), 0), dtype="float32"),  # no graph tower
            d.bundle.y[d.tr_idx],
            np.zeros(len(d.tr_idx), dtype="int64"),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    va_loader = DataLoader(
        NovaDS(
            d.bundle.X_tab[d.va_idx],
            d.bundle.X_seq[d.va_idx],
            np.zeros((len(d.va_idx), 0), dtype="float32"),
            d.bundle.y[d.va_idx],
            np.zeros(len(d.va_idx), dtype="int64"),
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    n_seq = d.bundle.X_seq.shape[-1]
    model = HybridModel(
        n_tab=d.bundle.X_tab.shape[1],
        d_tab=hp["d_tab"],
        d_seq=cfg.d_seq,
        g_dim=0,
        n_seq_features=n_seq,
        n_layers=2,
        dropout=hp["dropout"],
    ).to(device)
    pos_weight = torch.tensor(
        [(1 - d.bundle.y[d.tr_idx].mean()) / max(d.bundle.y[d.tr_idx].mean(), 1e-6)],
        device=device,
        dtype=torch.float32,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = AdamW(model.parameters(), lr=hp["lr"], weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    # AMP only on CUDA; CPU / MPS run fp32. (MPS autocast is fp16 but unstable
    # for some ops in current PyTorch; fp32 on MPS is already ~40x faster than CPU.)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    for ep in range(1, cfg.epochs + 1):
        ep_start = time.time()
        _empty_device_cache(device)
        model.train()
        for x_tab, x_seq, _x_g, yy, _gg in tr_loader:
            x_tab = x_tab.to(device)
            x_seq = x_seq.to(device)
            yy = yy.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(x_tab, x_seq, torch.zeros(yy.size(0), 0, device=device))
                    loss = loss_fn(logits, yy)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x_tab, x_seq, torch.zeros(yy.size(0), 0, device=device))
                loss = loss_fn(logits, yy)
                loss.backward()
                opt.step()
        sched.step()
        val_auc, _, _ = _val_auroc(model, va_loader, device)
        print(
            f"      ep {ep:02d}  val_auc={val_auc:.4f}  ({time.time() - ep_start:.1f}s)", flush=True
        )
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


def _hybrid_predict(
    model: HybridModel, X_tab: np.ndarray, X_seq: np.ndarray, device: str, batch_size: int = 1024
) -> np.ndarray:
    model.eval()
    n = len(X_tab)
    out = np.zeros(n, dtype="float64")
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            xt = torch.from_numpy(X_tab[sl].astype("float32")).to(device)
            xs = torch.from_numpy(X_seq[sl].astype("float32")).to(device)
            xg = torch.zeros(xt.size(0), 0, device=device)
            out[sl] = torch.sigmoid(model(xt, xs, xg)).cpu().numpy()
    return out


def _hybrid_hp_sweep(
    d: _SplitData,
    cfg: HomeCreditConfig,
    device: str,
) -> tuple[HybridModel, dict[str, Any], list[dict[str, Any]]]:
    import random

    rng = random.Random(cfg.seed)
    all_configs = list(itertools.product(*HP_SEARCH_SPACE.values()))
    rng.shuffle(all_configs)
    trial_configs = [
        dict(zip(HP_SEARCH_SPACE.keys(), c, strict=True)) for c in all_configs[: cfg.n_hp_trials]
    ]

    history: list[dict[str, Any]] = []
    best = {"val_auc": -1.0, "cfg": None, "model": None, "test_auc": None}

    for i, hp in enumerate(trial_configs, 1):
        t0 = time.time()
        torch.manual_seed(cfg.seed * 100 + i)
        np.random.seed(cfg.seed * 100 + i)
        _empty_device_cache(device)
        gc.collect()
        model = _train_hybrid_with_hp(d, cfg, hp, device)
        val_p = _hybrid_predict(model, d.bundle.X_tab[d.va_idx], d.bundle.X_seq[d.va_idx], device)
        test_p = _hybrid_predict(model, d.bundle.X_tab[d.te_idx], d.bundle.X_seq[d.te_idx], device)
        v_auc = float(roc_auc_score(d.bundle.y[d.va_idx], val_p))
        t_auc = float(roc_auc_score(d.bundle.y[d.te_idx], test_p))
        n_params = int(sum(p.numel() for p in model.parameters()))
        wall = time.time() - t0
        history.append(
            {
                "trial": i,
                "config": hp,
                "val_auc": v_auc,
                "test_auc": t_auc,
                "n_params": n_params,
                "wall_s": round(wall, 2),
            }
        )
        print(
            f"[hp-sweep] {i:2d}/{cfg.n_hp_trials}  {hp}  "
            f"val={v_auc:.4f}  test={t_auc:.4f}  params={n_params / 1e6:.2f}M  t={wall:.1f}s",
            flush=True,
        )
        # Persist sweep progress incrementally so a crash mid-sweep is recoverable.
        try:
            (cfg.results_dir / "hp_sweep_partial.json").write_text(json.dumps(history, indent=2))
        except Exception:
            pass
        if v_auc > best["val_auc"]:
            best = {"val_auc": v_auc, "cfg": hp, "model": model, "test_auc": t_auc}
        else:
            # Free the loser's parameters before the next trial.
            del model
        gc.collect()
        _empty_device_cache(device)

    return (
        best["model"],
        {**best["cfg"], "val_auc": best["val_auc"], "test_auc": best["test_auc"]},
        history,
    )


# ---------------------------------------------------------------------------
# Ensemble + calibration + fairness


def _ensemble(p_h_val, p_h_test, p_l_val, p_l_test, y_val, y_test):
    auc_h = float(roc_auc_score(y_val, p_h_val))
    auc_l = float(roc_auc_score(y_val, p_l_val))
    wh = max(auc_h - 0.5, 1e-6)
    wl = max(auc_l - 0.5, 1e-6)
    s = wh + wl
    wh /= s
    wl /= s
    val_p = wh * p_h_val + wl * p_l_val
    test_p = wh * p_h_test + wl * p_l_test
    return {
        "weights": {"hybrid": float(wh), "lightgbm": float(wl)},
        "val_auc": float(roc_auc_score(y_val, val_p)),
        "test_auc": float(roc_auc_score(y_test, test_p)),
        "val_p": val_p,
        "test_p": test_p,
    }


def _ensemble_probs_full(
    model: HybridModel, booster: lgb.Booster, bundle: HomeCreditBundle, weights, device: str
) -> np.ndarray:
    p_h = _hybrid_predict(model, bundle.X_tab, bundle.X_seq, device)
    p_l = booster.predict(bundle.X_tab, num_iteration=booster.best_iteration)
    return weights["hybrid"] * p_h + weights["lightgbm"] * p_l


def _global_threshold_for_tpr(p_pred: np.ndarray, y_true: np.ndarray, target_tpr: float) -> float:
    pos = p_pred[y_true == 1]
    if len(pos) == 0:
        return 0.5
    sorted_pos = np.sort(pos)
    k = max(1, int(np.floor(len(sorted_pos) * (1 - target_tpr))))
    return float(sorted_pos[k - 1])


def _run_fairness(
    bundle: HomeCreditBundle,
    p_pred_full: np.ndarray,
    A: float,
    B: float,
    target_tpr: float,
    results_dir: Path,
    mitigation_attribute: str = "age_bucket",
) -> dict[str, Any]:
    df = bundle.protected_attrs.copy()
    df["p"] = p_pred_full
    df["y"] = bundle.y
    # Drop CODE_GENDER == "XNA" for the gender axis only (handful of rows).
    valid_gender = df["gender"] != "XNA"
    groups_dict = {
        "gender": df["gender"].where(valid_gender, "unknown").to_numpy(),
        "age_bucket": df["age_bucket"].to_numpy(),
        "own_car": df["own_car"].to_numpy(),
        "family_status": df["family_status"].to_numpy(),
    }
    threshold = _global_threshold_for_tpr(df.p.to_numpy(), df.y.to_numpy(), target_tpr)
    print(f"[fairness] global threshold for TPR={target_tpr}: {threshold:.4f}")
    before = compute_all_metrics(df.y.to_numpy(), df.p.to_numpy(), groups_dict, threshold=threshold)

    grp = groups_dict[mitigation_attribute]
    thr_map = optimize_thresholds_per_group(
        df.y.to_numpy(), df.p.to_numpy(), grp, target_tpr=target_tpr
    )
    y_pred_after = np.fromiter(
        (int(p >= thr_map.threshold_for(g)) for p, g in zip(df.p.to_numpy(), grp, strict=True)),
        dtype=int,
        count=len(df),
    )
    after_rows: list[dict[str, Any]] = []
    for name, g_arr in groups_dict.items():
        if name == mitigation_attribute:
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
        "score_adjustment_per_group": {
            str(k): score_adjustment_from_threshold(v, A, B, default_threshold=threshold)
            for k, v in thr_map.thresholds.items()
        },
    }
    save_json(results_dir / "fairness_before_after.json", summary)
    save_json(results_dir / "threshold_map.json", thr_map.to_dict())
    return summary


# ---------------------------------------------------------------------------
# Plot helpers


def _plot_roc(y_true, p_h, p_l, p_ens, out_path):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    for name, p, color in (
        ("Hybrid", p_h, "#0B1628"),
        ("LightGBM", p_l, "#C9A26F"),
        ("Ensemble", p_ens, "#1A8F3B"),
    ):
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = float(roc_auc_score(y_true, p))
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Home Credit: Hybrid vs LightGBM vs Ensemble")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_score_distribution(scores: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.hist(scores, bins=40, color="#0B1628", edgecolor="white")
    for x, label, color in (
        (600, "Silver", "#C9A26F"),
        (700, "Gold", "#1A8F3B"),
        (800, "Platinum", "#1565C0"),
    ):
        ax.axvline(x, color=color, linestyle="--", linewidth=1, alpha=0.75)
        ax.text(x + 4, ax.get_ylim()[1] * 0.92, label, color=color, fontsize=9)
    ax.set_xlabel("NovaScore")
    ax.set_ylabel("Applicants")
    ax.set_title("Score distribution across decision bands")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level orchestrator


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_home_credit_pipeline(cfg: HomeCreditConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HomeCreditConfig()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    device = _best_device()
    print(f"[device] {device}")
    t_start = time.time()

    print("=" * 70)
    print("PHASE 4.6 — Home Credit Default Risk pipeline")
    print("=" * 70)
    bundle = prepare_home_credit(cfg.data_dir, sample_n=cfg.sample_n, seed=cfg.seed)
    print(
        f"[data] features={bundle.X_tab.shape[1]}  "
        f"sequence={bundle.X_seq.shape[1]}x{bundle.X_seq.shape[2]}  "
        f"y_pos_rate={bundle.y.mean():.4f}"
    )
    d = _split(bundle, cfg)

    print("\n--- LightGBM baseline ---")
    t = time.time()
    booster, lgb_info = _train_lightgbm_baseline(d)
    print(f"[lgbm] took {time.time() - t:.1f}s")
    # Interim save: if hybrid sweep later crashes (OOM, SIGKILL),
    # we still have a defensible LightGBM-only headline number.
    try:
        booster.save_model(str(cfg.results_dir / "lightgbm.txt"))
        save_json(
            cfg.results_dir / "lgbm_partial.json",
            {
                "stage": "phase_4.6_lightgbm_baseline_committed",
                "dataset": "home_credit_default_risk",
                "sample_n": cfg.sample_n,
                "n_train": int(len(d.tr_idx)),
                "n_val": int(len(d.va_idx)),
                "n_test": int(len(d.te_idx)),
                **lgb_info,
            },
        )
    except Exception as e:
        print(f"[lgbm] interim save failed: {e}", flush=True)

    print("\n--- Hybrid HP sweep ---")
    t = time.time()
    best_hybrid, hybrid_cfg, hp_history = _hybrid_hp_sweep(d, cfg, device)
    print(
        f"[hp-sweep] took {time.time() - t:.1f}s  best val={hybrid_cfg['val_auc']:.4f}  test={hybrid_cfg['test_auc']:.4f}"
    )

    # Component predictions on val + test for ensemble.
    p_h_val = _hybrid_predict(
        best_hybrid, d.bundle.X_tab[d.va_idx], d.bundle.X_seq[d.va_idx], device
    )
    p_h_test = _hybrid_predict(
        best_hybrid, d.bundle.X_tab[d.te_idx], d.bundle.X_seq[d.te_idx], device
    )
    p_l_val = booster.predict(d.bundle.X_tab[d.va_idx], num_iteration=booster.best_iteration)
    p_l_test = booster.predict(d.bundle.X_tab[d.te_idx], num_iteration=booster.best_iteration)

    print("\n--- Ensemble ---")
    ens = _ensemble(
        p_h_val, p_h_test, p_l_val, p_l_test, d.bundle.y[d.va_idx], d.bundle.y[d.te_idx]
    )
    print(
        f"[ensemble] weights={ens['weights']}  val={ens['val_auc']:.4f}  test={ens['test_auc']:.4f}"
    )
    p_ens_full = _ensemble_probs_full(best_hybrid, booster, bundle, ens["weights"], device)

    # Calibration.
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
    print(f"[calib] score_dist: {score_dist}")

    print("\n--- Fairness ---")
    fairness_summary = _run_fairness(
        bundle, p_ens_full, A, B, cfg.target_tpr, cfg.results_dir, mitigation_attribute="age_bucket"
    )
    for r in fairness_summary["before"]:
        print(
            f"[fair] {r['attribute']:14s}  before ΔTPR={r['delta_tpr']:.4f}  "
            f"DPR={r['demographic_parity_ratio']:.3f}"
        )
    for r in fairness_summary["after"]:
        if r["attribute"] == fairness_summary["mitigation_attribute"]:
            print(
                f"[fair] {r['attribute']:14s}  AFTER  ΔTPR={r['delta_tpr']:.4f}  "
                f"DPR={r['demographic_parity_ratio']:.3f}"
            )

    # Persist artifacts.
    rd = cfg.results_dir
    n_params = int(sum(p.numel() for p in best_hybrid.parameters()))
    save_checkpoint(
        best_hybrid,
        rd / "checkpoint.pt",
        hparams={
            "n_tab": int(bundle.X_tab.shape[1]),
            "d_tab": int(hybrid_cfg["d_tab"]),
            "d_seq": int(cfg.d_seq),
            "g_dim": 0,
            "n_seq_features": int(bundle.X_seq.shape[-1]),
            "n_layers": 2,
            "dropout": float(hybrid_cfg["dropout"]),
        },
    )
    booster.save_model(str(rd / "lightgbm.txt"))
    save_json(rd / "feature_columns.json", bundle.feature_columns)
    save_json(rd / "categorical_maps.json", bundle.categorical_maps)
    save_scaler(rd / "scaler.json", bundle.scaler_mean, bundle.scaler_scale)
    save_json(
        rd / "sequence_scaler.json",
        {"means": bundle.seq_mean.tolist(), "stds": bundle.seq_scale.tolist()},
    )
    save_calibration(rd / "calibration.json", calib)
    save_json(
        rd / "feature_importance.json", lgb_feature_importance(booster, bundle.feature_columns)
    )
    save_json(
        rd / "hp_sweep.json",
        {"hybrid_sweep": hp_history, "lightgbm": lgb_info, "ensemble_weights": ens["weights"]},
    )
    save_json(rd / "ensemble.json", {"weights": ens["weights"]})

    # Test predictions with protected-attribute audit columns.
    test_protected = bundle.protected_attrs.iloc[d.te_idx].reset_index(drop=True)
    pred_df = pd.DataFrame(
        {
            "user_id": bundle.user_ids[d.te_idx],
            "y_true": bundle.y[d.te_idx],
            "pd_hybrid": p_h_test,
            "pd_lightgbm": p_l_test,
            "pd_ensemble": ens["test_p"],
            "novascore": np.round(apply_calibration(ens["test_p"], calib), 1),
        }
    )
    pred_df = pd.concat([pred_df, test_protected.drop(columns=["SK_ID_CURR"])], axis=1)
    pred_df.to_csv(rd / "test_predictions.csv", index=False)
    np.save(rd / "all_probs.npy", p_ens_full)
    np.save(rd / "all_scores.npy", all_scores)
    np.save(rd / "test_probs_hybrid.npy", p_h_test)
    np.save(rd / "test_probs_lightgbm.npy", p_l_test)
    np.save(rd / "test_probs_ensemble.npy", ens["test_p"])

    # Plots.
    _plot_roc(bundle.y[d.te_idx], p_h_test, p_l_test, ens["test_p"], rd / "roc_curve.png")
    plot_calibration(bundle.y[d.te_idx], ens["test_p"], rd / "calibration_plot.png")
    plot_feature_importance(
        lgb_feature_importance(booster, bundle.feature_columns),
        rd / "feature_importance.png",
        top_k=30,
    )
    _plot_score_distribution(all_scores, rd / "score_distribution.png")

    metrics = {
        "dataset": "home_credit_default_risk",
        "headline_test_auroc": float(ens["test_auc"]),
        "ensemble_weights": ens["weights"],
        "hybrid_best_test_auroc": float(hybrid_cfg["test_auc"]),
        "hybrid_best_val_auroc": float(hybrid_cfg["val_auc"]),
        "hybrid_best_config": {k: hybrid_cfg[k] for k in HP_SEARCH_SPACE},
        "lightgbm_test_auroc": float(lgb_info["test_auc"]),
        "lightgbm_val_auroc": float(lgb_info["val_auc"]),
        "lightgbm_config": lgb_info["config"],
        "n_users": int(len(bundle.user_ids)),
        "n_train": int(len(d.tr_idx)),
        "n_val": int(len(d.va_idx)),
        "n_test": int(len(d.te_idx)),
        "positive_rate_full": float(np.mean(bundle.y)),
        "score_distribution": score_dist,
        "hybrid_parameter_count": n_params,
        "feature_count": int(bundle.X_tab.shape[1]),
        "sequence_shape": list(bundle.X_seq.shape[1:]),
        "seed": int(cfg.seed),
        "fairness": fairness_summary,
        "phase": "4.6",
        "total_wall_seconds": round(time.time() - t_start, 1),
    }
    save_json(rd / "metrics.json", metrics)
    print("\n" + "=" * 70)
    print("PHASE 4.6 SUMMARY")
    print("=" * 70)
    summary_for_print = {k: v for k, v in metrics.items() if k != "fairness"}
    print(json.dumps(summary_for_print, indent=2))
    print(f"\nTotal wall time: {metrics['total_wall_seconds']}s")
    return metrics


if __name__ == "__main__":  # pragma: no cover
    run_home_credit_pipeline()
