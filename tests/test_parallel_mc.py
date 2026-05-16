"""Tests for research/optimization/parallel_mc.py."""

import numpy as np
import pandas as pd
import pytest

from research.optimization.parallel_mc import (
    parallel_monte_carlo,
    format_parallel_results,
)


@pytest.fixture
def small_data():
    """Small dataset for testing parallel MC."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.3)
    return pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5,
        "low": close - 0.5, "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)


class TestParallelMonteCarlo:

    def test_basic_run(self, small_data):
        """Parallel MC should complete with 2 combos."""
        param_grid = [
            {"fast_period": 10, "slow_period": 30},
            {"fast_period": 20, "slow_period": 40},
        ]
        results = parallel_monte_carlo(
            "ma_crossover", param_grid, small_data,
            capital=10_000, fee_rate=0.0001, n_jobs=2,
        )
        assert len(results) == 2
        assert "sharpe" in results[0]
        assert results[0]["sharpe"] >= results[1]["sharpe"]  # sorted

    def test_format_output(self, small_data):
        """Format function should produce string output."""
        param_grid = [{"fast_period": 10, "slow_period": 30}]
        results = parallel_monte_carlo(
            "ma_crossover", param_grid, small_data, n_jobs=1,
        )
        output = format_parallel_results(results)
        assert isinstance(output, str)
        assert "BEST PARAMETERS" in output
        assert "Sharpe" in output
