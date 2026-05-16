"""Tests for research/strategies/ensemble.py."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.ensemble import (
    compute_rolling_sharpe,
    ensemble_signals,
)


@pytest.fixture
def signal_data():
    """Two simple signals + close price."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close_arr = 100 + np.cumsum(np.random.randn(500) * 0.5)
    close = pd.Series(close_arr, index=idx)
    sig1 = pd.Series(np.random.choice([-1, 0, 1], 500), index=idx)
    sig2 = pd.Series(np.random.choice([-1, 0, 1], 500), index=idx)
    return {"close": close, "sig1": sig1, "sig2": sig2}


class TestComputeRollingSharpe:

    def test_output_shape(self, signal_data):
        """Rolling Sharpe should match input length."""
        sh = compute_rolling_sharpe(signal_data["sig1"], signal_data["close"], window=50)
        assert len(sh) == len(signal_data["close"])
        assert sh.dtype == float

    def test_no_nan(self, signal_data):
        """After warmup, should have no NaN."""
        sh = compute_rolling_sharpe(signal_data["sig1"], signal_data["close"], window=50)
        assert not sh.iloc[100:].isna().any()


class TestEnsembleSignals:

    def test_equal_weight(self, signal_data):
        """Equal weight should average signals."""
        combined = ensemble_signals(
            {"a": signal_data["sig1"], "b": signal_data["sig2"]},
            signal_data["close"],
            method="equal",
        )
        expected = (signal_data["sig1"] + signal_data["sig2"]) / 2
        pd.testing.assert_series_equal(combined, expected)

    def test_single_signal(self, signal_data):
        """Single signal should pass through."""
        combined = ensemble_signals(
            {"a": signal_data["sig1"]},
            signal_data["close"],
            method="equal",
        )
        # equal weight of single signal = signal itself, converted to float
        pd.testing.assert_series_equal(combined, signal_data["sig1"].astype(float))

    def test_output_range(self, signal_data):
        """Combined signal should be in [-1, 1]."""
        combined = ensemble_signals(
            {"a": signal_data["sig1"], "b": signal_data["sig2"]},
            signal_data["close"],
            method="sharpe_weight",
            window=50,
        )
        assert combined.min() >= -1
        assert combined.max() <= 1

    def test_sharpe_rank_output(self, signal_data):
        """Sharpe rank method should produce valid output."""
        combined = ensemble_signals(
            {"a": signal_data["sig1"], "b": signal_data["sig2"]},
            signal_data["close"],
            method="sharpe_rank",
            window=50,
        )
        assert combined.min() >= -1
        assert combined.max() <= 1