"""Tests for research/optimization/parallel_mc.py."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from research.optimization.parallel_mc import (
    parallel_monte_carlo,
    format_parallel_results,
    save_parallel_results,
)


@pytest.fixture
def small_data():
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
        assert results[0]["sharpe"] >= results[1]["sharpe"]

    def test_format_output(self, small_data):
        param_grid = [{"fast_period": 10, "slow_period": 30}]
        results = parallel_monte_carlo(
            "ma_crossover", param_grid, small_data, n_jobs=1,
        )
        output = format_parallel_results(results)
        assert isinstance(output, str)
        assert "BEST PARAMETERS" in output
        assert "Sharpe" in output

    def test_single_combo(self, small_data):
        results = parallel_monte_carlo(
            "ma_crossover", [{"fast_period": 10, "slow_period": 30}],
            small_data, n_jobs=1,
        )
        assert len(results) == 1

    def test_empty_grid(self, small_data):
        results = parallel_monte_carlo("ma_crossover", [], small_data, n_jobs=1)
        assert results == []

    def test_multiple_combos(self, small_data):
        pg = [{"fast_period": i, "slow_period": i+20} for i in [5, 10, 15]]
        results = parallel_monte_carlo("ma_crossover", pg, small_data, n_jobs=2)
        assert len(results) == 3

    def test_results_have_keys(self, small_data):
        pg = [{"fast_period": i, "slow_period": i+20} for i in [5, 10]]
        results = parallel_monte_carlo("ma_crossover", pg, small_data, n_jobs=2)
        for k in ["sharpe", "profit_factor", "max_dd", "trades", "params"]:
            assert k in results[0]

    def test_save_report_creates_file(self, small_data, tmp_path):
        pg = [{"fast_period": 10, "slow_period": 30}]
        results = parallel_monte_carlo("ma_crossover", pg, small_data, n_jobs=1)
        path = save_parallel_results(results, "test", output_dir=str(tmp_path))
        assert Path(path).exists()

    def test_format_strategy_name(self, small_data):
        pg = [{"fast_period": 10, "slow_period": 30}]
        results = parallel_monte_carlo("ma_crossover", pg, small_data, n_jobs=1)
        rep = format_parallel_results(results, strategy_name="MA Crossover")
        assert "MA Crossover" in rep

    def test_format_top_n(self, small_data):
        pg = [{"fast_period": i, "slow_period": i+20} for i in [5, 10, 15]]
        results = parallel_monte_carlo("ma_crossover", pg, small_data, n_jobs=2)
        rep = format_parallel_results(results, top_n=1)
        assert "TOP 1 BY SHARPE" in rep

    def test_n_jobs_capped(self, small_data):
        pg = [{"fast_period": 5, "slow_period": 20}]
        results = parallel_monte_carlo("ma_crossover", pg, small_data, n_jobs=100)
        assert len(results) == 1
