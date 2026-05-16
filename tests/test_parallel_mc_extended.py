"""Extended tests for research/optimization/parallel_mc.py."""

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


class TestParallelMCExtended:

    def test_empty_param_grid(self, small_data):
        """Empty grid should return empty list."""
        results = parallel_monte_carlo("ma_crossover", [], small_data, n_jobs=1)
        assert len(results) == 0

    def test_single_combo(self, small_data):
        """Single combo should return 1 result."""
        results = parallel_monte_carlo(
            "ma_crossover", [{"fast_period": 10, "slow_period": 30}],
            small_data, n_jobs=1,
        )
        assert len(results) == 1
        assert "sharpe" in results[0]
        assert "max_dd" in results[0]
        assert "trades" in results[0]

    def test_results_have_all_keys(self, small_data):
        """Each result should have all expected keys."""
        param_grid = [
            {"fast_period": 10, "slow_period": 30},
            {"fast_period": 20, "slow_period": 40},
        ]
        results = parallel_monte_carlo("ma_crossover", param_grid, small_data, n_jobs=2)
        required = {"sharpe", "profit_factor", "max_dd", "trades", "final_equity", "total_return", "params"}
        for r in results:
            assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"

    def test_save_report_creates_file(self, small_data, tmp_path):
        """save_parallel_results should create a file."""
        param_grid = [{"fast_period": 10, "slow_period": 30}]
        results = parallel_monte_carlo("ma_crossover", param_grid, small_data, n_jobs=1)
        path = save_parallel_results(results, "test_strategy", output_dir=str(tmp_path))
        assert Path(path).exists()
        content = open(path).read()
        assert "MONTE CARLO OPTIMIZATION" in content
        assert "TOP" in content
        assert "BEST PARAMETERS" in content
        assert "REPRODUCIBILITY" in content

    def test_format_has_top_by_drawdown(self, small_data):
        """Report should contain TOP BY DRAWDOWN section."""
        param_grid = [
            {"fast_period": 10, "slow_period": 30},
            {"fast_period": 20, "slow_period": 40},
        ]
        results = parallel_monte_carlo("ma_crossover", param_grid, small_data, n_jobs=2)
        report = format_parallel_results(results, top_n=2)
        assert "TOP 2 BY SHARPE" in report
        assert "TOP 2 BY DRAWDOWN" in report

    def test_format_has_swapped_columns(self, small_data):
        """Report should have Sharpe before Parameters."""
        param_grid = [{"fast_period": 10, "slow_period": 30}]
        results = parallel_monte_carlo("ma_crossover", param_grid, small_data, n_jobs=1)
        report = format_parallel_results(results)
        # The header should have Sharpe before Parameters
        sh_idx = report.find("Sharpe")
        pf_idx = report.find("PF")
        param_idx = report.find("Parameters")
        assert sh_idx < param_idx  # Sharpe comes before Parameters
        assert pf_idx < param_idx  # PF comes before Parameters

    def test_n_jobs_respected(self, small_data):
        """Should use at most n_jobs workers."""
        import time
        param_grid = [{"fast_period": i, "slow_period": i + 20} for i in range(1, 10)]
        t0 = time.perf_counter()
        results = parallel_monte_carlo("ma_crossover", param_grid, small_data, n_jobs=4)
        elapsed = time.perf_counter() - t0
        assert len(results) == 9
        assert elapsed < 30  # should complete quickly

    def test_reproducibility_in_report(self, small_data):
        """Report should include the command."""
        param_grid = [{"fast_period": 10, "slow_period": 30}]
        results = parallel_monte_carlo("ma_crossover", param_grid, small_data, n_jobs=1)
        report = format_parallel_results(
            results, strategy_name="MA Crossover",
            command="sweep v1", data_range="2024",
        )
        assert "sweep v1" in report
        assert "MA Crossover" in report
        assert "2024" in report
