"""Save/load helpers for the artifacts produced by training.

A single training run writes ~10 files to `results_dir`:
- checkpoint.pt — model state dict + hyperparameters
- feature_columns.json — tabular feature names in column order
- scaler.json — StandardScaler mean and scale arrays
- topk.json — learned top-k categorical values per categorical column
- sequence_scaler.json — per-feature z-score parameters for the sequence tower
- calibration.json — A, B, a, b for the PD → score map
- metrics.json — AUROC, ΔTPR, score distribution
- test_predictions.csv — per-user test scores and decision bands

The API loads these artifacts at startup; the `novascore` CLI rebuilds plots
from them without retraining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration import CalibrationParams
from .data.sequences import SequenceScaler


def save_checkpoint(model: torch.nn.Module, path: Path, hparams: dict[str, Any]) -> None:
    torch.save({"state_dict": model.state_dict(), "hparams": hparams}, path)


def load_checkpoint(path: Path, model_cls, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = model_cls(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["hparams"]


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def save_scaler(path: Path, mean: np.ndarray, scale: np.ndarray) -> None:
    save_json(path, {"mean": mean.tolist(), "scale": scale.tolist()})


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = load_json(path)
    return np.asarray(d["mean"], dtype="float32"), np.asarray(d["scale"], dtype="float32")


def save_calibration(path: Path, params: CalibrationParams) -> None:
    save_json(path, params.to_dict())


def load_calibration(path: Path) -> CalibrationParams:
    return CalibrationParams.from_dict(load_json(path))


def save_sequence_scaler(path: Path, scaler: SequenceScaler) -> None:
    save_json(path, scaler.to_dict())


def load_sequence_scaler(path: Path) -> SequenceScaler:
    return SequenceScaler.from_dict(load_json(path))
