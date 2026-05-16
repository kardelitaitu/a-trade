"""Tests for research/optimization/robustness.py."""

import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from research.backtest.engine import Trade, VectorizedBacktest
from research.optimization.robustness import (
    monte_carlo_shuffle,
    regime_breakdown,
    parameter_stability,
)


@pytest.fixture
def fake_trades():
    """Create fake trade objects for Monte Carlo testing."""
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    return [
        Trade(idx[i], idx[i+1], 1, 100, 101, 1.0, np.random.randn() * 10, 0, 0, 0.1, "5min")
        for i in range(50)
    ]


class TestMonteCarlo:

    def test_monte_carlo_runs(self, fake_trades):
        """Monte Carlo should complete without error."""
        result = monte_carlo_shuffle(fake_trades, n_simulations=100)
        assert "actual_sharpe" in result
        assert "p_value" in result
        assert result["n_simulations"] == 100

    def test_few_trades(self):
        """Fewer than 30 trades should give a warning."""
        result = monte_carlo_shuffle([], n_simulations=10)
        assert result["is_significant"] is False
        assert "Too few trades" in result["message"]


class TestParameterStability:

    def test_stability_runs(self):
        """Parameter stability test should complete."""
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=2000, freq="5min")
        close = 100 + np.cumsum(np.random.randn(2000) * 0.3)
        data = pd.DataFrame({
            "open": close - 0.2, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": np.random.uniform(10, 100, 2000),
        }, index=idx)

        from research.strategies.mean_reversion import MeanReversion

        base_params = {"mode": "rsi", "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}
        df = parameter_stability(MeanReversion, base_params, data, variations={
            "rsi_period": [10, 14, 20],
        })
        assert len(df) > 1
        assert "sharpe" in df.columns
