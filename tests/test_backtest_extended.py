"""Extended tests for research/backtest/engine.py — edge cases, numba ops, batch kernels."""

import numpy as np
import pandas as pd
import pytest

from research.backtest.engine import VectorizedBacktest, Trade
from research.backtest.metrics import compute_metrics, format_metrics_report
from research.backtest._numba_ops import extract_trades_numba
from research.backtest._batch import (
    single_backtest,
    single_backtest_sltp,
    compute_sma_bank,
    batch_ma_crossover,
    batch_ma_crossover_sltp,
    batch_mean_reversion_rsi,
)


@pytest.fixture
def volatile_data():
    """Data with high volatility spikes."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.5)
    # Add a flash crash
    close[200:210] = close[199] - np.arange(10) * 2
    close[210:220] = close[209] + np.arange(10) * 1.5
    return pd.DataFrame({
        "open": close - 0.2, "high": close + np.abs(np.random.randn(500)) * 2,
        "low": close - np.abs(np.random.randn(500)) * 2, "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)


# ═══════════════════════════════════════════════════════════════════
# Extended Backtest Engine Tests
# ═══════════════════════════════════════════════════════════════════

class TestBacktestExtended:

    def test_zero_signals(self, volatile_data):
        """Zero signals should keep equity flat (minus fees)."""
        bt = VectorizedBacktest(volatile_data, {"fee": 0.0})
        result = bt.run(pd.Series(0, index=volatile_data.index))
        assert abs(result.equity_curve.iloc[-1] - 10000) < 1

    def test_always_long(self, volatile_data):
        """Always long should track price (minus fees)."""
        bt = VectorizedBacktest(volatile_data, {"fee": 0.0, "initial_capital": 1000})
        sig = pd.Series(1, index=volatile_data.index)
        result = bt.run(sig)
        price_change = volatile_data["close"].iloc[-1] / volatile_data["close"].iloc[0]
        expected = 1000 * price_change
        assert abs(result.equity_curve.iloc[-1] - expected) / expected < 0.1

    def test_custom_capital(self):
        """Backtest should respect initial capital."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        data = pd.DataFrame({"close": 100 + np.sin(np.arange(100) * 0.1), "open": 100, "high": 105, "low": 95, "volume": 50}, index=idx)
        bt = VectorizedBacktest(data, {"initial_capital": 5000, "fee": 0.0})
        result = bt.run(pd.Series(0, index=idx))
        assert result.equity_curve.iloc[-1] == 5000

    def test_high_fee_drains_equity(self, volatile_data):
        """High fees should clearly reduce returns."""
        bt_low = VectorizedBacktest(volatile_data, {"fee": 0.0001, "initial_capital": 10000})
        bt_high = VectorizedBacktest(volatile_data, {"fee": 0.05, "initial_capital": 10000})
        sig = pd.Series(1, index=volatile_data.index)
        r_low = bt_low.run(sig)
        r_high = bt_high.run(sig)
        assert r_low.equity_curve.iloc[-1] > r_high.equity_curve.iloc[-1]


# ═══════════════════════════════════════════════════════════════════
# Numa Batch Kernel Tests
# ═══════════════════════════════════════════════════════════════════

class TestNumbaBatch:

    def test_compute_sma_bank(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        periods = np.array([2, 3])
        bank = compute_sma_bank(close, periods)
        assert bank.shape == (2, 6)
        # SMA(2) at index 1
        assert bank[0, 1] == pytest.approx(1.5)
        # SMA(3) at index 2
        assert bank[1, 2] == pytest.approx(2.0)

    def test_sma_bank_nan_handling(self):
        close = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        periods = np.array([2])
        bank = compute_sma_bank(close, periods)
        # After forward-fill, close[2] becomes 2.0
        # SMA(2): [nan, 1.5, 1.5, 2.0, 3.0, 4.5, 5.5]
        assert not np.isnan(bank[0]).all()

    def test_single_backtest_basic(self):
        close = np.array([100.0, 102.0, 101.0, 103.0, 100.0])
        sig = np.array([0.0, 1.0, 1.0, 0.0, -1.0])
        eq, dd, sh, tr, wr, pf = single_backtest(close, sig, 10000, 0.0001)
        assert eq > 0
        assert tr >= 0

    def test_single_backtest_sltp_triggers(self):
        """SL/TP should trigger on extreme moves."""
        close = np.array([100.0, 105.0, 95.0, 100.0, 102.0])
        high = np.array([102.0, 107.0, 97.0, 102.0, 104.0])
        low = np.array([98.0, 103.0, 93.0, 98.0, 100.0])
        sig = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        eq, dd, sh, tr, wr, pf = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.02, 0.04, 0.0, high, low,
        )
        # Should have at least some trades
        assert tr > 0

    def test_batch_ma_crossover_output_shape(self):
        np.random.seed(42)
        close = np.array(100 + np.cumsum(np.random.randn(200)), dtype=float)
        periods = np.array([5, 10, 20], dtype=np.int32)
        sma_bank = compute_sma_bank(close, periods)
        fast_idxs = np.array([0, 1], dtype=np.int32)  # sma(5), sma(10)
        slow_idxs = np.array([1, 2], dtype=np.int32)  # sma(10), sma(20)
        result = batch_ma_crossover(close, sma_bank, fast_idxs, slow_idxs, 10000, 0.0001)
        assert result.shape == (2, 6)

    def test_batch_rsi_output_shape(self):
        np.random.seed(42)
        close = np.array(100 + np.cumsum(np.random.randn(200)), dtype=float)
        from research.backtest._batch import _rsi_bank
        rsi_periods = np.array([7, 14], dtype=np.int32)
        rsi_bank = _rsi_bank(close, rsi_periods)
        oversold = np.array([30.0, 25.0], dtype=np.float64)
        overbought = np.array([70.0, 75.0], dtype=np.float64)
        period_idxs = np.array([0, 1], dtype=np.int32)
        result = batch_mean_reversion_rsi(
            close, rsi_bank, period_idxs, oversold, overbought, 10000, 0.0001,
        )
        assert result.shape == (2, 6)


# ═══════════════════════════════════════════════════════════════════
# Metrics Extended Tests
# ═══════════════════════════════════════════════════════════════════

class TestMetricsExtended:

    def test_format_report_has_init_equity(self):
        """Report should mention initial equity."""
        idx = pd.date_range("2024-01-01", periods=4, freq="5min")
        eq = pd.Series([10000, 10200, 10100, 10300], index=idx)
        m = compute_metrics(eq)
        m["initial_capital"] = 10000.0
        report = format_metrics_report(m)
        assert "Initial" in report
        assert "10,000" in report

    def test_sharpe_positive_on_uptrend(self):
        """Rising equity should give positive Sharpe."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        eq = pd.Series(10000 * (1 + np.arange(100) / 10000), index=idx)
        m = compute_metrics(eq)
        assert m["sharpe_ratio"] > 0

    def test_sharpe_negative_on_downtrend(self):
        """Falling equity should give negative Sharpe."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        eq = pd.Series(10000 * (1 - np.arange(100) / 10000), index=idx)
        m = compute_metrics(eq)
        assert m["sharpe_ratio"] < 0

    def test_drawdown_calculation(self):
        """Max drawdown should be accurate."""
        idx = pd.date_range("2024-01-01", periods=6, freq="5min")
        eq = pd.Series([10000, 11000, 9000, 9500, 8000, 8500], index=idx)
        m = compute_metrics(eq)
        # Peak was 11000, trough 8000, DD% = (11000-8000)/11000 * 100 = 27.27%
        assert m["max_drawdown_pct"] == pytest.approx(-27.27, abs=0.5)
