"""Tests for research/optimization/search.py."""

import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from research.optimization.search import (
    random_parameter_search,
    generate_optimization_report,
)


@pytest.fixture
def small_data():
    """Small dataset for testing."""
    np.random.seed(42)
    idx = pd.date_range("2023-01-01", periods=2000, freq="5min")
    close = 100 + np.cumsum(np.random.randn(2000) * 0.3)
    return pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.random.uniform(10, 100, 2000),
    }, index=idx)


class TestRandomParameterSearch:

    def test_search_runs(self, small_data):
        """Search should complete without errors."""
        from research.strategies.ma_crossover import MACrossover

        param_ranges = {
            "fast_period": [5, 10, 20],
            "slow_period": [20, 50],
            "ma_type": ["sma", "ema"],
        }
        splits = [("2023-01-01", "2023-03-01"), ("2023-03-01", "2023-06-01")]
        results = random_parameter_search(
            MACrossover, param_ranges, small_data,
            n_samples=5, walk_forward_splits=splits, seed=42,
        )
        assert len(results) == 5
        assert "params" in results[0]
        assert "score" in results[0]

    def test_sorted_by_score(self, small_data):
        """Results should be sorted descending by score."""
        from research.strategies.ma_crossover import MACrossover

        results = random_parameter_search(
            MACrossover, {"fast_period": [5], "slow_period": [20]},
            small_data, n_samples=2,
            walk_forward_splits=[("2023-01-01", "2023-06-01"), ("2023-06-01", "2023-12-01")],
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_report_generated(self, small_data, tmp_path):
        """Report file should be created."""
        from research.strategies.ma_crossover import MACrossover

        results = random_parameter_search(
            MACrossover, {"fast_period": [5], "slow_period": [20]},
            small_data, n_samples=2,
            walk_forward_splits=[("2023-01-01", "2023-06-01"), ("2023-06-01", "2023-12-01")],
        )
        path = generate_optimization_report(results, "Test Strategy", tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "Best Parameters" in content
        assert "Walk-Forward Metrics" in content
