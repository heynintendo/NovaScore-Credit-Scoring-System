"""Re-apply the corrected empirical calibration to saved Phase 4.6 artifacts.

The original Phase 4.6 calibration had its empirical-refinement targets swapped
(low-PD quantile mapped to score-600 PD instead of score-800 PD), which inverted
the PD → score direction in committed test_predictions.csv. The model itself
is unaffected — only the post-hoc PD → score transform was wrong.

This script:
  1. Loads ml/results/all_probs.npy and ml/results/test_predictions.csv.
  2. Re-derives the corrected (a, b) via empirical_refinement.
  3. Rewrites ml/results/calibration.json, all_scores.npy, test_predictions.csv.
  4. Recomputes score_distribution in metrics.json.
  5. Re-renders score_distribution.png and calibration_plot.png.

No retraining; no PD values change. Only the calibrated score column moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from novascore.calibration import (
    CalibrationParams,
    apply_calibration,
    decision_band,
    empirical_refinement,
    solve_score_params,
)
from novascore.evaluate import plot_calibration

RESULTS = Path("ml/results")


def main() -> None:
    all_probs = np.load(RESULTS / "all_probs.npy")
    test_preds = pd.read_csv(RESULTS / "test_predictions.csv")
    metrics = json.loads((RESULTS / "metrics.json").read_text())

    A, B = solve_score_params()
    a, b = empirical_refinement(all_probs, A, B, q_low=0.20, q_high=0.80)
    calib = CalibrationParams(A=A, B=B, a=a, b=b)
    print(f"corrected calibration: A={A:.2f}  B={B:.2f}  a={a:.4f}  b={b:.4f}")

    # 1. Recompute full-population scores.
    all_scores = apply_calibration(all_probs, calib)
    np.save(RESULTS / "all_scores.npy", all_scores)

    # 2. Recompute test predictions' novascore column from pd_ensemble.
    p_te = test_preds["pd_ensemble"].to_numpy()
    new_scores = apply_calibration(p_te, calib)
    test_preds["novascore"] = np.round(new_scores, 1)
    test_preds.to_csv(RESULTS / "test_predictions.csv", index=False)

    # 3. Update calibration.json + metrics.json.
    (RESULTS / "calibration.json").write_text(json.dumps(calib.to_dict(), indent=2))
    new_dist = {
        "Bronze": float((all_scores < 600).mean()),
        "Silver": float(((all_scores >= 600) & (all_scores < 700)).mean()),
        "Gold": float(((all_scores >= 700) & (all_scores < 800)).mean()),
        "Platinum": float((all_scores >= 800).mean()),
    }
    metrics["score_distribution"] = new_dist
    metrics["recalibrated"] = True
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"corrected score_distribution: {new_dist}")

    # 4. Re-render plots.
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.hist(all_scores, bins=40, color="#0B1628", edgecolor="white")
    for x, label, color in (
        (600, "Silver", "#C9A26F"),
        (700, "Gold", "#1A8F3B"),
        (800, "Platinum", "#1565C0"),
    ):
        ax.axvline(x, color=color, linestyle="--", linewidth=1, alpha=0.75)
        ax.text(x + 4, ax.get_ylim()[1] * 0.92, label, color=color, fontsize=9)
    ax.set_xlabel("NovaScore")
    ax.set_ylabel("Applicants")
    ax.set_title("Score distribution across decision bands (recalibrated)")
    fig.tight_layout()
    fig.savefig(RESULTS / "score_distribution.png")
    plt.close(fig)

    # Calibration reliability diagram uses raw PDs vs observed, so it's
    # unaffected. Regenerate anyway for consistency.
    y_te = test_preds["y_true"].to_numpy()
    plot_calibration(y_te, p_te, RESULTS / "calibration_plot.png")

    print("recalibration complete.")


if __name__ == "__main__":
    main()
