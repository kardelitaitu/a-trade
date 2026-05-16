"""Extended tests for research/optimization/monte_carlo.py — batched engine."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from research.optimization.monte_carlo import (
    monte_carlo_ma,
    monte_carlo_rsi,
    monte_carlo_ma_sltp,
    save_mc_report,
    MCResult,
    MCMetrics,
)


@pytest.fixture
def close_array():
    np.random.seed(42)
    return np.array(100 + np.cumsum(np.random.randn(1000)), dtype=float)


class TestMCResult:

    def test_best_metrics(self):
        metrics = np.array([[100, -10, 0.5, 10, 0.5, 1.2],
                            [200, -5, 1.5, 20, 0.6, 2.0]], dtype=float)
        r = MCResult(
            strategy="test", n_combos=2, duration=1.0, combos_per_second=2.0,
            params_list=[{"a": 1}, {"a": 2}], metrics=metrics,
            best_idx=1, sorted_indices=np.array([1, 0]),
            initial_capital=10000,
        )
        best = r.best_metrics()
        assert best.final_equity == 200
        assert best.sharpe == 1.5

    def test_summary_string(self, close_array):
        r = monte_carlo_ma(close_array, (5, 15), (20, 40), capital=10000, fee_rate=0.0001)
        s = r.summary()
        assert isinstance(s, str)
        assert "MA Crossover" in s or "monte_carlo" in s

    def test_save_report_creates_file(self, close_array, tmp_path):
        r = monte_carlo_ma(close_array, (5, 15), (20, 40), capital=10000, fee_rate=0.0001)
        path = save_mc_report(r, output_dir=Path(tmp_path))
        assert Path(path).exists()


class TestMonteCarloBatch:

    def test_ma_crossover_returns_mcresult(self, close_array):
        r = monte_carlo_ma(close_array, (5, 15), (20, 40), capital=10000, fee_rate=0.0001)
        assert isinstance(r, MCResult)
        assert r.n_combos > 0
        assert r.n_combos <= 11 * 21

    def test_ma_crossover_metrics_shape(self, close_array):
        r = monte_carlo_ma(close_array, (5, 15), (20, 40), capital=10000, fee_rate=0.0001)
        assert r.metrics.shape[1] == 6

    def test_ma_crossover_sorted(self, close_array):
        r = monte_carlo_ma(close_array, (5, 15), (20, 40), capital=10000, fee_rate=0.0001)
        # First sorted should be best Sharpe
        assert r.sorted_indices[0] == r.best_idx

    def test_rsi_monte_carlo(self, close_array):
        r = monte_carlo_rsi(close_array, [7, 14], [25, 30], [70, 75], capital=10000, fee_rate=0.0001)
        assert isinstance(r, MCResult)
        assert r.n_combos > 0

    def test_ma_sltp_monte_carlo(self, close_array):
        """SL/TP Monte Carlo should work."""
        high = close_array + 5
        low = close_array - 5
        r = monte_carlo_ma_sltp(
            close_array, high, low,
            (5, 15), (20, 40),
            [0.01], [0.03],
            capital=10000, fee_rate=0.0001,
        )
        assert isinstance(r, MCResult)
        assert r.n_combos > 0

    def test_report_has_sensitivity(self, close_array, tmp_path):
        """Report should include sensitivity analysis."""
        r = monte_carlo_ma(close_array, (5, 15), (20, 40), capital=10000, fee_rate=0.0001)
        path = save_mc_report(r, output_dir=Path(tmp_path))
        content = open(path).read()
        assert "Sensitivity" in content or "How to read" in content

    def test_report_has_top_by_drawdown(self, close_array, tmp_path):
        """Report should include TOP BY DRAWDOWN."""
        r = monte_carlo_ma(close_array, (5, 15), (20, 40), capital=10000, fee_rate=0.0001)
        path = save_mc_report(r, output_dir=Path(tmp_path))
        content = open(path).read()
        assert "TOP " in content
