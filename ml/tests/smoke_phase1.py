"""Phase 1 smoke test — verify HybridModel trains end-to-end on random tensors.

Not part of the pytest suite; this is a one-shot integration check to confirm
the broken pieces (`HybridModel`, `train_model`, `delta_tpr_at_threshold`) are
wired up correctly and gradients flow.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from novascore.models.hybrid import HybridModel
from novascore.train import NovaDS, delta_tpr_at_threshold, train_model


def main() -> None:
    rng = np.random.default_rng(0)
    n, n_tab, t_weeks, n_seq, g_dim = 256, 12, 13, 9, 64
    X_tab = rng.standard_normal((n, n_tab)).astype("float32")
    X_seq = rng.standard_normal((n, t_weeks, n_seq)).astype("float32")
    X_g = rng.standard_normal((n, g_dim)).astype("float32")
    # Signal: label correlates with the sum of two tabular features.
    logits = 0.7 * X_tab[:, 0] - 0.5 * X_tab[:, 1] + 0.3 * rng.standard_normal(n).astype("float32")
    y = (logits > 0).astype("int64")
    grp = rng.integers(0, 3, size=n)

    split = int(n * 0.75)
    tr = NovaDS(X_tab[:split], X_seq[:split], X_g[:split], y[:split], grp[:split])
    va = NovaDS(X_tab[split:], X_seq[split:], X_g[split:], y[split:], grp[split:])
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True)
    va_loader = DataLoader(va, batch_size=64, shuffle=False)

    model = train_model(
        tr_loader,
        va_loader,
        n_tab=n_tab,
        n_seq_features=n_seq,
        g_dim=g_dim,
        epochs=5,
        patience=3,
        device="cpu",
        verbose=True,
    )
    assert isinstance(model, HybridModel)

    # Parameter count (reported honestly, not the deck's "8M" claim).
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HybridModel parameter count: {n_params:,}")

    # Forward pass on a single batch.
    model.eval()
    with torch.no_grad():
        x_tab = torch.from_numpy(X_tab[split:])
        x_seq = torch.from_numpy(X_seq[split:])
        x_g = torch.from_numpy(X_g[split:])
        p = torch.sigmoid(model(x_tab, x_seq, x_g)).numpy()

    dtpr = delta_tpr_at_threshold(y[split:], p, grp[split:], thr=0.5)
    print(f"delta_tpr_at_threshold on smoke holdout: {dtpr:.4f}")

    # delta_tpr is in [0, 1].
    assert 0.0 <= dtpr <= 1.0
    print("smoke OK")


if __name__ == "__main__":
    main()
