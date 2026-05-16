"""
Feature Importance Analysis for QuantumEdge ML models.

Generates reports on feature importance, ablation, and correlation.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def load_model(path: Path) -> lgb.Booster:
    """Load a trained LightGBM model."""
    return lgb.Booster(model_file=str(path))


def compute_importance(model: lgb.Booster, feature_names: list[str]) -> pd.DataFrame:
    """
    Compute feature importance from a trained model.

    Returns
    -------
    pd.DataFrame with columns: feature, gain, split, cover
    """
    df = pd.DataFrame({
        "feature": feature_names,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    })
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    df["gain_pct"] = df["gain"] / df["gain"].sum() * 100
    df["cumulative_pct"] = df["gain_pct"].cumsum()
    return df


def generate_importance_report(
    importance: pd.DataFrame,
    output_dir: Path,
    name: str = "feature_importance",
) -> Path:
    """
    Generate and save a feature importance report.

    Returns path to the saved report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.txt"

    lines = []
    lines.append("=" * 70)
    lines.append("QUANTUMEDGE — FEATURE IMPORTANCE REPORT")
    lines.append("=" * 70)
    lines.append(f"\nTotal features: {len(importance)}")
    lines.append(f"\n{'Rank':>5}  {'Feature':<25}  {'Gain':>10}  {'Gain%':>8}  {'Cum%':>8}  {'Splits':>8}")
    lines.append("-" * 70)

    for i, row in importance.iterrows():
        lines.append(
            f"{i + 1:>5}  {row['feature']:<25}  {row['gain']:>10.1f}  "
            f"{row['gain_pct']:>7.2f}%  {row['cumulative_pct']:>7.2f}%  {row['split']:>8}"
        )

    # Top N features that explain 80% of gain
    top_n = (importance["cumulative_pct"] <= 80).sum()
    lines.append(f"\nTop {top_n} features explain 80% of total gain.")

    # Bottom features (negligible)
    negligible = importance[importance["gain_pct"] < 0.5]
    lines.append(f"Features with <0.5% gain: {len(negligible)}")

    lines.append("\n" + "=" * 70)
    content = "\n".join(lines)

    path.write_text(content)
    return path


def compute_feature_correlation(features: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation of features."""
    return features.corr()


def ablation_analysis(
    model: lgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    n_top: int = 5,
) -> pd.DataFrame:
    """
    Simple ablation: drop top N features one by one and measure logloss impact.

    Returns DataFrame with columns: dropped_feature, logloss_before, logloss_after, delta.
    """
    import lightgbm as lgb

    base_logloss = _eval_logloss(model, X_val, y_val)
    results = []

    for i in range(min(n_top, len(feature_names))):
        feat = feature_names[i]
        X_dropped = X_val.drop(columns=[feat])
        # Retrain without this feature
        train_data = lgb.Dataset(X_val.drop(columns=[feat]), label=y_val)
        m = lgb.train(
            {"objective": "binary", "verbosity": -1, "num_threads": 4},
            train_data,
            num_boost_round=50,
        )
        logloss = _eval_logloss(m, X_dropped, y_val)
        results.append({
            "dropped_feature": feat,
            "logloss_before": base_logloss,
            "logloss_after": logloss,
            "delta": logloss - base_logloss,
        })

    return pd.DataFrame(results)


def _eval_logloss(model, X, y) -> float:
    """Compute binary logloss."""
    from sklearn.metrics import log_loss
    probs = model.predict(X)
    return log_loss(y, probs)