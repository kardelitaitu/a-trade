"""Tests for research/features/ml_features.py."""

import numpy as np
import pandas as pd
import pytest

from research.features.ml_features import compute_features, compute_target, train_val_test_split


@pytest.fixture
def sample_data():
    """Small OHLCV dataset for testing features."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 50000 + np.cumsum(np.random.randn(500) * 10)
    df = pd.DataFrame({
        "open": close - 5,
        "high": close + 10,
        "low": close - 10,
        "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)
    return df


class TestComputeFeatures:

    def test_feature_shape(self, sample_data):
        """Feature matrix should have same index as input."""
        feat = compute_features(sample_data)
        assert feat.index.equals(sample_data.index)
        assert feat.shape[1] >= 20, "Expected at least 20 features"

    def test_no_lookahead(self, sample_data):
        """Features should not use future data (all shifted/rolling)."""
        feat = compute_features(sample_data)
        # All features should have NaN at the start (warmup), not at the end
        head_nan = feat.iloc[:10].isna().sum().sum()
        tail_nan = feat.iloc[-10:].isna().sum().sum()
        assert head_nan > tail_nan, "Features may have look-ahead bias"

    def test_feature_names(self, sample_data):
        """Feature columns should have descriptive names."""
        feat = compute_features(sample_data)
        prefixes = ["ret_", "log_ret_", "rsi_", "macd", "bb_", "atr_", "vol_", "hour_"]
        found = any(any(col.startswith(p) for col in feat.columns) for p in prefixes)
        assert found, "No expected feature prefixes found"

    def test_time_features(self, sample_data):
        """Time features should be present when requested."""
        feat = compute_features(sample_data, include_time_features=True)
        assert "hour_sin" in feat.columns
        assert "hour_cos" in feat.columns
        assert "is_weekend" in feat.columns

    def test_can_disable_time(self, sample_data):
        """Time features should be absent when disabled."""
        feat = compute_features(sample_data, include_time_features=False)
        assert "hour_sin" not in feat.columns

    def test_no_inf_values(self, sample_data):
        """No infinite values in features."""
        feat = compute_features(sample_data)
        assert not np.isinf(feat.values).any()


class TestComputeTarget:

    def test_binary_target_shape(self, sample_data):
        """Binary target should match input length."""
        target = compute_target(sample_data, horizon=6, method="binary")
        assert len(target) == len(sample_data)
        assert set(target.dropna().unique()).issubset({0, 1})

    def test_binary_has_nan_at_end(self, sample_data):
        """Last horizon rows should be NaN (no future data)."""
        target = compute_target(sample_data, horizon=6, method="binary")
        assert target.iloc[-6:].isna().all()
        assert target.iloc[-7].item() in (0, 1) or np.isnan(target.iloc[-7])

    def test_direction_target(self, sample_data):
        """Direction target should have -1, 0, 1 values."""
        target = compute_target(sample_data, horizon=6, method="direction")
        valid = target.dropna().unique()
        assert set(valid).issubset({-1, 0, 1})


class TestTrainValTestSplit:

    def test_split_shapes(self, sample_data):
        """Split should preserve chronological order and correct ratios."""
        feat = compute_features(sample_data)
        target = compute_target(sample_data, horizon=6)

        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            feat, target, val_split=0.15, test_split=0.15,
        )

        total = len(feat)
        assert len(X_train) > len(X_val) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_val) + len(X_test) == total
        # Chronological: train < val < test
        assert X_train.index[-1] < X_val.index[0]
        assert X_val.index[-1] < X_test.index[0]
