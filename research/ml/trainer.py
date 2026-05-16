"""
ML Model Training Pipeline for QuantumEdge.

Trains a LightGBM classifier on engineered features to predict
short-term price direction. Generates trading signals from
predicted probabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[dict] = None,
    model_dir: Optional[Path] = None,
    name: str = "lgb_model",
) -> lgb.Booster:
    """
    Train a LightGBM classifier with early stopping.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices for training and validation.
    y_train, y_val : pd.Series
        Binary targets (0/1).
    params : dict, optional
        LightGBM parameters. Uses sensible defaults if not provided.
    model_dir : Path, optional
        Directory to save the model file.
    name : str
        Base name for the saved model file.

    Returns
    -------
    lgb.Booster
        Trained model.
    """
    if params is None:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "seed": 42,
            "num_threads": 32,
        }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{name}.txt"
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

    return model


def predict_probabilities(
    model: lgb.Booster,
    features: pd.DataFrame,
) -> pd.Series:
    """
    Predict probability of upward move.

    Returns
    -------
    pd.Series
        Probability of class 1 (up) aligned with features index.
    """
    probs = model.predict(features)
    return pd.Series(probs, index=features.index)


def signals_from_probabilities(
    probabilities: pd.Series,
    threshold: float = 0.55,
    neutral_zone: float = 0.05,
    continuous: bool = False,
) -> pd.Series:
    """
    Convert model probabilities to trading signals.

    Parameters
    ----------
    probabilities : pd.Series
        Predicted probability of upward move (0-1).
    threshold : float
        For discrete mode: above threshold → long (1), below (1-threshold) → short (-1).
    neutral_zone : float
        Width of neutral zone around 0.5. Signals within this zone are 0.
    continuous : bool
        If True, position size scales linearly with confidence:
        position = 2 * (prob - 0.5), clipped to [-1, 1].
        Useful for weak models where discrete thresholding is too aggressive.

    Returns
    -------
    pd.Series
        Signals: -1 (short), 0 (neutral), +1 (long), or float in [-1, 1] if continuous.
    """
    if continuous:
        # Scale confidence: 0.5 → 0, 0.75 → 0.5, 0.0 → -1, 1.0 → 1
        signals = 2 * (probabilities - 0.5)
        return signals.clip(-1, 1)

    low = 0.5 - neutral_zone / 2
    high = 0.5 + neutral_zone / 2

    signals = pd.Series(0, index=probabilities.index)
    signals[probabilities > max(threshold, high)] = 1
    signals[probabilities < min(1 - threshold, low)] = -1
    return signals


def run_ml_pipeline(
    data: pd.DataFrame,
    model_params: Optional[dict] = None,
    prob_threshold: float = 0.55,
    horizon: int = 6,
    continuous: bool = False,
    model_dir: Optional[Path] = None,
) -> dict:
    """
    End-to-end ML pipeline: features → train → predict → signals.

    Parameters
    ----------
    data : pd.DataFrame
        Full OHLCV dataset.
    model_params : dict, optional
    prob_threshold : float
    horizon : int
        Prediction horizon in 5-min periods.
    model_dir : Path, optional

    Returns
    -------
    dict with keys: model, signals, feature_importance, metrics
    """
    from research.features.ml_features import (
        compute_features,
        compute_target,
        train_val_test_split,
    )

    logger.info("Computing features...")
    features = compute_features(data)

    logger.info("Computing target...")
    target = compute_target(data, horizon=horizon, method="binary")

    # Drop rows with NaN in features or target
    valid = features.dropna().index.intersection(target.dropna().index)
    features = features.loc[valid]
    target = target.loc[valid]

    logger.info(f"Valid rows: {len(features)} ({len(features)/len(data)*100:.1f}% of total)")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        features, target, val_split=0.15, test_split=0.15,
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    model = train_model(X_train, y_train, X_val, y_val, model_params, model_dir)

    # Feature importance
    importance = pd.DataFrame({
        "feature": features.columns,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    # Predict on test set
    test_probs = predict_probabilities(model, X_test)
    test_signals = signals_from_probabilities(test_probs, threshold=prob_threshold, continuous=continuous)

    # Backtest
    from research.backtest.engine import VectorizedBacktest
    from research.backtest.metrics import compute_metrics, format_metrics_report

    # Align test data with signal index
    test_data = data.loc[X_test.index]
    bt = VectorizedBacktest(test_data)
    result = bt.run(test_signals)
    metrics = compute_metrics(result.equity_curve, result.trades)

    logger.info(f"\n{format_metrics_report(metrics)}")

    return {
        "model": model,
        "signals": test_signals,
        "feature_importance": importance,
        "metrics": metrics,
        "result": result,
    }
