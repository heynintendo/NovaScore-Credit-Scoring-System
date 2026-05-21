"""LightGBM baseline on tabular features only — interpretability sanity check."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score


def train_lightgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    num_boost_round: int = 2000,
    stopping_rounds: int = 200,
    log_period: int = 0,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Train a binary-classification LightGBM and return (booster, info)."""
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    pos_rate = max(float(np.mean(y_tr)), 1e-6)
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbosity=-1,
        scale_pos_weight=(1 - pos_rate) / pos_rate,
    )
    cb: list = [lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)]
    if log_period > 0:
        cb.append(lgb.log_evaluation(period=log_period))
    booster = lgb.train(
        params=params,
        train_set=dtr,
        valid_sets=[dva],
        num_boost_round=num_boost_round,
        callbacks=cb,
    )
    info = {
        "best_iteration": int(booster.best_iteration),
        "val_auc": float(booster.best_score["valid_0"]["auc"]),
    }
    return booster, info


def score_lightgbm(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return np.asarray(booster.predict(X, num_iteration=booster.best_iteration))


def test_auroc(booster: lgb.Booster, X_te: np.ndarray, y_te: np.ndarray) -> float:
    p = score_lightgbm(booster, X_te)
    if len(np.unique(y_te)) < 2:
        return float("nan")
    return float(roc_auc_score(y_te, p))


def feature_importance(booster: lgb.Booster, feature_names: list[str]) -> list[tuple[str, float]]:
    """Return [(feature, gain)] sorted descending."""
    gains = booster.feature_importance(importance_type="gain").tolist()
    pairs = sorted(zip(feature_names, gains, strict=True), key=lambda kv: kv[1], reverse=True)
    return pairs
