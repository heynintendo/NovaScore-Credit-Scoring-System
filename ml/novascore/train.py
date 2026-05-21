"""Training loop, dataset wrapper, and group-fairness helpers used during training.

The training loop follows the original NovaScore deck:
- AdamW(lr=3e-4, weight_decay=1e-4)
- Cosine annealing over `epochs`
- BCEWithLogitsLoss
- Early stopping on validation AUROC (patience=5)
- Mixed precision (AMP) on CUDA, fp32 on CPU
- Best validation checkpoint is restored before returning.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .models.hybrid import HybridModel


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


def _val_auroc(model: HybridModel, loader: DataLoader, device: str) -> tuple[float, np.ndarray, np.ndarray]:
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
    device: Optional[str] = None,
    verbose: bool = True,
) -> HybridModel:
    """Train HybridModel and return the model with best-val weights loaded.

    Hyperparameters match the NovaScore deck specification:
    AdamW(lr=3e-4, wd=1e-4), cosine LR over `epochs`, early-stop on val AUROC with
    patience=5, mixed precision on CUDA.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(
        n_tab=n_tab, n_seq_features=n_seq_features, g_dim=g_dim
    ).to(device)
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
