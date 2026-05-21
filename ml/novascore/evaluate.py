"""Evaluation utilities: test metrics, plots, predictions DataFrame.

Used both by the training orchestrator (to emit ml/results/*) and by the
`novascore evaluate` CLI subcommand (to rebuild plots from a saved checkpoint
without retraining).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .calibration import CalibrationParams, apply_calibration, decision_band  # noqa: E402


def predict_probs(
    model: torch.nn.Module,
    X_tab: np.ndarray,
    X_seq: np.ndarray,
    X_g: np.ndarray,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """Run sigmoid(model) over the dataset in batches, return PD predictions."""
    model.eval()
    n = len(X_tab)
    out = np.zeros(n, dtype="float64")
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            xt = torch.from_numpy(X_tab[sl].astype("float32")).to(device)
            xs = torch.from_numpy(X_seq[sl].astype("float32")).to(device)
            xg = torch.from_numpy(X_g[sl].astype("float32")).to(device)
            out[sl] = torch.sigmoid(model(xt, xs, xg)).cpu().numpy()
    return out


def build_predictions_df(
    user_ids: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    calibration: CalibrationParams,
    groups: np.ndarray | None = None,
    group_categories: list[str] | None = None,
) -> pd.DataFrame:
    """Per-user DataFrame with PD, calibrated NovaScore, and decision band."""
    scores = apply_calibration(probs, calibration)
    bands = np.array([decision_band(float(s)) for s in scores])
    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "y_true": np.asarray(y_true).astype(int),
            "pd90": probs,
            "novascore": np.round(scores, 1),
            "decision_band": bands,
        }
    )
    if groups is not None and group_categories is not None:
        df["group"] = pd.Categorical.from_codes(
            groups, categories=group_categories
        ).astype(str)
    return df


def plot_roc(
    y_te: np.ndarray,
    p_hybrid: np.ndarray,
    p_lgb: np.ndarray | None,
    out_path: Path,
    title: str = "ROC — NovaScore Hybrid vs LightGBM",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    fpr, tpr, _ = roc_curve(y_te, p_hybrid)
    auc_h = float(np.trapz(tpr, fpr))
    ax.plot(fpr, tpr, label=f"Hybrid (AUC={auc_h:.3f})", color="#0B1628", linewidth=2)
    if p_lgb is not None:
        fpr_l, tpr_l, _ = roc_curve(y_te, p_lgb)
        auc_l = float(np.trapz(tpr_l, fpr_l))
        ax.plot(fpr_l, tpr_l, label=f"LightGBM (AUC={auc_l:.3f})", color="#C9A26F", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_calibration(
    y_te: np.ndarray, p_te: np.ndarray, out_path: Path, n_bins: int = 10
) -> None:
    """Reliability diagram comparing predicted PD to observed default rate."""
    frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", color="#0B1628", linewidth=2, label="hybrid model")
    ax.set_xlabel("Mean predicted PD (bin)")
    ax.set_ylabel("Fraction of observed positives")
    ax.set_title("Calibration — predicted vs observed default rate")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(
    pairs: list[tuple[str, float]],
    out_path: Path,
    top_k: int = 20,
    title: str = "LightGBM feature importance (gain)",
) -> None:
    top = pairs[:top_k][::-1]
    names = [p[0] for p in top]
    gains = [p[1] for p in top]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(top))), dpi=120)
    ax.barh(names, gains, color="#C9A26F")
    ax.set_xlabel("gain")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def score_distribution_summary(scores: np.ndarray) -> dict[str, float]:
    """Decision-band shares; useful as a sanity check after calibration."""
    return {
        "Bronze": float((scores < 600).mean()),
        "Silver": float(((scores >= 600) & (scores < 700)).mean()),
        "Gold": float(((scores >= 700) & (scores < 800)).mean()),
        "Platinum": float((scores >= 800).mean()),
    }


def plot_fairness_before_after(
    tpr_before: dict[str, float],
    tpr_after: dict[str, float],
    out_path: Path,
    attribute: str = "age_bucket",
) -> None:
    """Grouped bar chart: per-group TPR before vs after threshold mitigation."""
    groups = sorted(set(tpr_before) | set(tpr_after))
    before = [tpr_before.get(g, 0.0) for g in groups]
    after = [tpr_after.get(g, 0.0) for g in groups]
    x = np.arange(len(groups))
    w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.bar(x - w / 2, before, w, label="before mitigation", color="#0B1628")
    ax.bar(x + w / 2, after, w, label="after threshold optimization", color="#C9A26F")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Equal-opportunity check by {attribute}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
