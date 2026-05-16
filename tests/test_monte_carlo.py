"""Tests for research/backtest/_batch.py and research/optimization/monte_carlo.py."""

import numpy as np
import pandas as pd
import pytest

from research.backtest._batch import compute_sma_bank, single_backtest
from research.backtest.engine import VectorizedBacktest
from research.backtest.metrics import compute_metrics
from research.strategies.ma_crossover import MACrossover
from research.optimization.monte_carlo import monte_carlo_ma, monte_carlo_rsi, save_mc_report


@pytest.fixture
def sample_data():
    """Small dataset for correctness checking."""
    np.random.seed(42)
    n = 2000
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5,
        "low": close - 0.5, "close": close,
        "volume": np.random.uniform(10, 100, n),
    }, index=idx)


class TestSingleBacktest:

    def test_flat_signals(self):
        """All-flat signals should return initial capital."""
        close = np.array([100.0, 101.0, 102.0, 101.0, 100.0], dtype=np.float64)
        signals = np.array([0, 0, 0, 0, 0], dtype=np.float64)
        eq, dd, sh, tr, wr, pf = single_backtest(close, signals, 10_000.0, 0.000011)
        assert eq == pytest.approx(10_000.0, abs=0.01)
        assert tr == 0
        assert sh == 0.0

    def test_vs_pandas_engine(self, sample_data):
        """Numba single_backtest should match pandas VectorizedBacktest closely."""
        close_arr = sample_data["close"].values.astype(np.float64)
        strat = MACrossover({"fast_period": 10, "slow_period": 30})
        signals_pd = strat.generate_signals(sample_data)

        # Numba result
        eq_nb, dd_nb, sh_nb, tr_nb, wr_nb, pf_nb = single_backtest(
            close_arr, signals_pd.values.astype(np.float64), 10_000.0, 0.000011,
        )

        # Pandas result
        bt = VectorizedBacktest(sample_data)
        result = bt.run(signals_pd)
        m = compute_metrics(result.equity_curve, result.trades)

        # Compare — allow small tolerance for fee rounding differences
        assert eq_nb == pytest.approx(m["final_equity"], rel=0.05), \
            f"Equity mismatch: nb={eq_nb:.2f} vs pd={m['final_equity']:.2f}"
        assert tr_nb == pytest.approx(m["total_trades"], rel=0.1), \
            f"Trades mismatch: nb={tr_nb} vs pd={m['total_trades']}"


class TestComputeSmaBank:

    def test_sma_bank_shape(self, sample_data):
        """SMA bank should have correct shape."""
        close = sample_data["close"].values.astype(np.float64)
        periods = np.array([5, 10, 20], dtype=np.int32)
        bank = compute_sma_bank(close, periods)
        assert bank.shape == (3, len(close))
        # SMA(5) should not be NaN after index 4
        assert not np.isnan(bank[0, 5])
        assert np.isnan(bank[0, 3])

    def test_sma_correctness(self, sample_data):
        """SMA values should match pandas rolling mean."""
        close_pd = sample_data["close"]
        close_np = close_pd.values.astype(np.float64)
        periods = np.array([5, 20], dtype=np.int32)
        bank = compute_sma_bank(close_np, periods)

        pd_5 = close_pd.rolling(5).mean().values
        np.testing.assert_array_almost_equal(bank[0, 4:], pd_5[4:], decimal=4)


class TestMonteCarloMA:

    def test_exhaustive_ma(self, sample_data):
        """Exhaustive MA search should complete quickly."""
        close = sample_data["close"].values.astype(np.float64)
        result = monte_carlo_ma(close, (5, 10), (20, 30), step=5)
        assert result.n_combos > 0
        assert result.n_combos == 6  # 2 fast × 3 slow = 6 (filtered by fast < slow)
        assert result.duration < 15.0  # includes JIT compilation

    def test_best_sharpe(self, sample_data):
        """Best result should have highest Sharpe."""
        close = sample_data["close"].values.astype(np.float64)
        result = monte_carlo_ma(close, (5, 15), (20, 40))
        best_sharpe = result.metrics[result.best_idx, 2]
        assert best_sharpe == max(result.metrics[:, 2])


class TestMonteCarloRSI:

    def test_exhaustive_rsi(self, sample_data):
        """Exhaustive RSI search should complete quickly."""
        close = sample_data["close"].values.astype(np.float64)
        result = monte_carlo_rsi(close, (7, 14), [30], [70])
        assert result.n_combos > 0
        assert result.duration < 10.0

    def test_report_generated(self, sample_data, tmp_path):
        """Report should be written as a text file."""
        close = sample_data["close"].values.astype(np.float64)
        result = monte_carlo_ma(close, (5, 10), (20, 30), step=5)
        path = save_mc_report(result, output_dir=tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "BEST PARAMETERS" in content
        assert "Initial Equity" in content
        assert "REPRODUCIBILITY" in content
