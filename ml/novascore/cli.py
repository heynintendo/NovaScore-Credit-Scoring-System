"""`novascore` command-line entry point.

Subcommands:
    train       Generate (if needed) synthetic data and run the full pipeline.
    evaluate    Rebuild plots and metrics from a saved checkpoint.
    score       Score a single partner from a JSON file containing raw features.

The CLI is the canonical way to reproduce the project from a fresh clone:

    pip install -e ml/
    novascore train --epochs 20

writes everything to `ml/results/` for the API and frontend to consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import SEQ_FEATURES, T_WEEKS
from .calibration import BAND_DESCRIPTIONS, apply_calibration, decision_band
from .evaluate import (
    plot_calibration,
    plot_feature_importance,
    plot_roc,
    predict_probs,
)
from .home_credit_pipeline import HomeCreditConfig, run_home_credit_pipeline
from .io import (
    load_calibration,
    load_checkpoint,
    load_json,
    load_scaler,
)
from .models.hybrid import HybridModel
from .sweep import SweepConfig, run_phase45_pipeline
from .train import TrainingConfig, run_training


def _cmd_train(args: argparse.Namespace) -> int:
    if args.dataset == "home_credit":
        cfg = HomeCreditConfig(
            data_dir=Path(args.data_dir)
            if args.data_dir != "ml/data/synthetic"
            else Path("ml/data/home_credit"),
            results_dir=Path(args.results_dir),
            seed=args.seed,
            epochs=args.epochs,
            sample_n=args.sample_n,
            n_hp_trials=args.n_trials,
        )
        run_home_credit_pipeline(cfg)
        return 0
    cfg = TrainingConfig(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        n_users=args.n_users,
        seed=args.seed,
        epochs=args.epochs,
        skip_lightgbm=args.skip_lightgbm,
        use_graph=not args.skip_graph,
    )
    run_training(cfg)
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    cfg = SweepConfig(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        n_users=args.n_users,
        n_hp_trials=args.n_trials,
        epochs=args.epochs,
        seed=args.seed,
    )
    run_phase45_pipeline(cfg)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    rd = Path(args.results_dir)
    metrics = load_json(rd / "metrics.json")
    print(json.dumps(metrics, indent=2))

    # Rebuild plots from saved arrays + LightGBM model. Just verify existence;
    # the actual probabilities for plotting come from test_predictions.csv below.
    if not (rd / "all_probs.npy").exists():
        print("missing all_probs.npy — re-run `novascore train` first")
        return 1

    test_pred_path = rd / "test_predictions.csv"
    if not test_pred_path.exists():
        print(f"missing {test_pred_path}")
        return 1
    import pandas as pd

    df = pd.read_csv(test_pred_path)
    y_te = df["y_true"].to_numpy()
    # Schema differs between paths: synth path emits `pd90`; Home Credit path
    # emits `pd_ensemble` (with separate `pd_hybrid` / `pd_lightgbm` columns).
    for pd_col in ("pd_ensemble", "pd90", "pd_lightgbm"):
        if pd_col in df.columns:
            p_te = df[pd_col].to_numpy()
            break
    else:
        print(f"no PD column found in {test_pred_path}. Columns: {list(df.columns)}")
        return 1

    p_te_lgb = None
    lgb_probs_path = rd / "test_probs_lightgbm.npy"
    if lgb_probs_path.exists():
        p_te_lgb = np.load(lgb_probs_path)

    plot_roc(y_te, p_te, p_te_lgb, rd / "roc_curve.png")
    plot_calibration(y_te, p_te, rd / "calibration_plot.png")

    fi_path = rd / "feature_importance.json"
    if fi_path.exists():
        pairs = load_json(fi_path)
        plot_feature_importance([(p[0], float(p[1])) for p in pairs], rd / "feature_importance.png")

    print(f"plots rebuilt under {rd}")
    return 0


def _build_input_vec(
    inputs: dict[str, float],
    feature_columns: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    """Map a partial JSON input dict to a standardized feature vector.

    Unspecified features use the training mean so they become 0 after scaling.
    """
    raw = scaler_mean.copy().astype("float64")
    for i, col in enumerate(feature_columns):
        if col in inputs:
            raw[i] = float(inputs[col])
    return ((raw - scaler_mean) / scaler_scale).astype("float32")


def _cmd_score(args: argparse.Namespace) -> int:
    rd = Path(args.results_dir)
    feature_columns: list[str] = load_json(rd / "feature_columns.json")
    scaler_mean, scaler_scale = load_scaler(rd / "scaler.json")
    calib = load_calibration(rd / "calibration.json")

    inputs: dict[str, Any] = json.loads(Path(args.input).read_text())
    x_tab = _build_input_vec(inputs, feature_columns, scaler_mean, scaler_scale).reshape(1, -1)
    model, hparams = load_checkpoint(rd / "checkpoint.pt", HybridModel)
    # Sequence + graph dims come from the saved checkpoint (synth used 13×9 +
    # 64-d graph, Home Credit uses 60×8 + no graph). Zeros = "no history".
    n_seq_features = int(hparams.get("n_seq_features", len(SEQ_FEATURES)))
    seq_len = T_WEEKS if n_seq_features == len(SEQ_FEATURES) else 60
    g_dim = int(hparams.get("g_dim", 0))
    x_seq = np.zeros((1, seq_len, n_seq_features), dtype="float32")
    x_g = np.zeros((1, g_dim), dtype="float32")
    probs = predict_probs(model, x_tab, x_seq, x_g, device="cpu")
    score = float(apply_calibration(probs, calib)[0])
    band = decision_band(score)
    out = {
        "pd": float(probs[0]),
        "novascore": round(score, 1),
        "decision_band": band,
        "policy": BAND_DESCRIPTIONS[band],
    }
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="novascore", description="NovaScore CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train end-to-end")
    p_train.add_argument(
        "--dataset",
        choices=("synthetic", "home_credit"),
        default="synthetic",
        help="which dataset to train on (default: synthetic)",
    )
    p_train.add_argument("--data-dir", default="ml/data/synthetic")
    p_train.add_argument("--results-dir", default="ml/results")
    p_train.add_argument("--n-users", type=int, default=10000)
    p_train.add_argument(
        "--sample-n", type=int, default=None, help="sample N applicants (home_credit only)"
    )
    p_train.add_argument(
        "--n-trials", type=int, default=8, help="hybrid HP trials (home_credit only)"
    )
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--skip-lightgbm", action="store_true")
    p_train.add_argument("--skip-graph", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="rebuild plots and print metrics")
    p_eval.add_argument("--results-dir", default="ml/results")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_sweep = sub.add_parser(
        "sweep",
        help="Phase 4.5 pipeline: HP sweep + LightGBM grid + ensemble + recalibrate",
    )
    p_sweep.add_argument("--data-dir", default="ml/data/synthetic")
    p_sweep.add_argument("--results-dir", default="ml/results")
    p_sweep.add_argument("--n-users", type=int, default=10000)
    p_sweep.add_argument("--n-trials", type=int, default=15)
    p_sweep.add_argument("--epochs", type=int, default=15)
    p_sweep.add_argument("--seed", type=int, default=42)
    p_sweep.set_defaults(func=_cmd_sweep)

    p_score = sub.add_parser("score", help="score one partner from a JSON file")
    p_score.add_argument("--input", required=True, help="path to JSON with raw features")
    p_score.add_argument("--results-dir", default="ml/results")
    p_score.set_defaults(func=_cmd_score)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
