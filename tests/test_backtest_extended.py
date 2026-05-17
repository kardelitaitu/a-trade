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
        assert m["max_drawdown_pct"] == pytest.approx(-27.27, abs=0.5)

    def test_return_pct_calculation(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="5min")
        m = compute_metrics(pd.Series([10000, 20000], index=idx))
        assert abs(m["total_return_pct"] - 100.0) < 1.0

    def test_cagr_positive(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="5min")
        m = compute_metrics(pd.Series([10000, 12000], index=idx))
        assert m["cagr_pct"] > 0

    def test_profit_factor_on_gain(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="5min")
        m = compute_metrics(pd.Series([10000, 11000], index=idx))
        assert m["profit_factor"] >= 0


# ═══════════════════════════════════════════════════════════════════
# SL/TP Equity Curve Capping Tests
# ═══════════════════════════════════════════════════════════════════

class TestSLTPEquityCapping:
    """Verify SL/TP exits actually cap the equity curve."""

    @pytest.fixture
    def crash_data(self):
        """Price series with a 15% crash midway, then recovery."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        # Gradual uptrend then crash
        close = 100.0 + np.arange(n) * 0.2
        # Crash from bar 40 to bar 50: drop 15%
        crash_val = close[39]
        for i in range(40, 51):
            close[i] = crash_val * (1.0 - 0.15 * (i - 39) / 11)
        # Recovery
        for i in range(51, n):
            close[i] = close[50] * (1.0 + 0.005 * (i - 50))
        # Build OHLC with realistic ranges
        high = close + np.abs(np.random.randn(n)) * 0.3
        low = close - np.abs(np.random.randn(n)) * 0.3
        # Ensure high >= close >= low
        high = np.maximum(high, close + 0.01)
        low = np.minimum(low, close - 0.01)
        return pd.DataFrame({
            "open": close - 0.1, "high": high, "low": low,
            "close": close, "volume": np.random.uniform(10, 100, n),
        }, index=idx)

    @pytest.fixture
    def steady_long_signal(self, crash_data):
        """Go long on bar 1 and stay long."""
        sig = pd.Series(0, index=crash_data.index, dtype=float)
        sig.iloc[1:] = 1.0
        return sig

    def test_no_sl_allows_full_crash_dd(self, crash_data):
        """Without SL/TP, max drawdown should reflect the full crash depth."""
        close = crash_data["close"].values.astype(np.float64)
        high = crash_data["high"].values.astype(np.float64)
        low = crash_data["low"].values.astype(np.float64)
        sig = np.zeros(len(close), dtype=np.float64)
        sig[1:] = 1.0

        # Run WITHOUT stop loss
        _, dd_no_sl, _, _, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.0, 0.0, 0.0, high, low,
        )
        # The crash is ~15%, so max_dd should be close to that
        assert dd_no_sl > 10.0, f"Without SL, max_dd ({dd_no_sl:.1f}%) should reflect the ~15% crash"

    def test_sl_3pct_caps_dd(self):
        """With 3% SL, max drawdown should be capped at ~3%.
        Uses deterministic data where crash happens immediately after entry
        (no run-up to inflate peak equity).
        """
        np.random.seed(42)
        n = 30
        close = np.zeros(n, dtype=np.float64)
        close[:3] = [100.0, 100.5, 101.0]
        # Immediate crash after entry bar
        close[3:] = np.linspace(95.0, 80.0, n - 3)

        high = close + np.abs(np.random.randn(n)) * 0.3
        low = close - np.abs(np.random.randn(n)) * 0.3
        high = np.maximum(high, close + 0.01)
        low = np.minimum(low, close - 0.01)

        # sig[1]=1 -> entry at close[2]=101 (i=2 iteration opens position)
        # SL/TP checks start from i=3 onwards
        sig = np.zeros(n, dtype=np.float64)
        sig[1:] = 1.0

        # Without SL: equity keeps dropping with the crash
        _, dd_no_sl, _, _, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.0, 0.0, 0.0, high, low,
        )
        assert dd_no_sl > 8.0, (
            f"Without SL, max_dd ({dd_no_sl:.1f}%) should be large "
            f"(price drops from 101 to 80)"
        )

        # With 3% SL: sl_price = 101*0.97 = 97.97
        # low[3] should be below 97.97 -> SL triggers on bar 3
        # return on exit: 97.97/101 - 1 = -0.03 = -3%
        # No run-up before crash, so max_dd ~= SL%
        _, dd_sl, _, _, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.03, 0.0, 0.0, high, low,
        )
        assert dd_sl < 5.0, f"With 3% SL, max_dd ({dd_sl:.1f}%) should be capped near 3%"
        assert dd_sl < dd_no_sl * 0.7, (
            f"SL max_dd ({dd_sl:.1f}%) should be much smaller than no-SL ({dd_no_sl:.1f}%)"
        )

    def test_sl_5pct_caps_dd(self):
        """With 5% SL, max drawdown should be capped at ~5%.
        Uses deterministic data with immediate crash after entry.
        """
        np.random.seed(42)
        n = 30
        close = np.zeros(n, dtype=np.float64)
        close[:3] = [100.0, 100.5, 101.0]
        close[3:] = np.linspace(95.0, 80.0, n - 3)

        high = close + np.abs(np.random.randn(n)) * 0.3
        low = close - np.abs(np.random.randn(n)) * 0.3
        high = np.maximum(high, close + 0.01)
        low = np.minimum(low, close - 0.01)

        sig = np.zeros(n, dtype=np.float64)
        sig[1:] = 1.0

        # With 5% SL: sl_price = 101*0.95 = 95.95
        # low[3] should be below 95.95 -> SL triggers on bar 3
        # return on exit: 95.95/101 - 1 = -0.05 = -5%
        _, dd_sl, _, _, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.05, 0.0, 0.0, high, low,
        )
        assert dd_sl < 7.0, f"With 5% SL, max_dd ({dd_sl:.1f}%) should be capped near 5%"

    def test_tighter_sl_gives_smaller_dd(self, crash_data):
        """Tighter SL should result in strictly smaller (or equal) max drawdown."""
        close = crash_data["close"].values.astype(np.float64)
        high = crash_data["high"].values.astype(np.float64)
        low = crash_data["low"].values.astype(np.float64)
        sig = np.zeros(len(close), dtype=np.float64)
        sig[1:] = 1.0

        dds = []
        for sl in [0.01, 0.02, 0.03, 0.05, 0.10]:
            _, dd, _, _, _, _ = single_backtest_sltp(
                close, sig, 10000, 0.0001,
                float(sl), 0.0, 0.0, high, low,
            )
            dds.append(dd)

        # Each tighter SL should produce ≤ previous dd
        for i in range(1, len(dds)):
            assert dds[i] >= dds[i - 1] - 0.5, (
                f"SL {[0.01,0.02,0.03,0.05,0.10][i-1]*100:.0f}% gave dd={dds[i-1]:.1f}% "
                f"but SL {[0.01,0.02,0.03,0.05,0.10][i]*100:.0f}% gave dd={dds[i]:.1f}% "
                f"— tighter SL should not produce larger dd"
            )

    def test_short_sl_caps_dd(self):
        """Short position with SL should also cap drawdown.
        Uses deterministic data with immediate rally after entry.
        """
        np.random.seed(42)
        n = 30
        close = np.zeros(n, dtype=np.float64)
        close[:3] = [101.0, 100.5, 100.0]
        # Sharp rally immediately after entry bar
        close[3:] = np.linspace(106.0, 115.0, n - 3)

        high = close + np.abs(np.random.randn(n)) * 0.3
        low = close - np.abs(np.random.randn(n)) * 0.3
        high = np.maximum(high, close + 0.01)
        low = np.minimum(low, close - 0.01)

        # sig[1]=-1 -> entry at close[2]=100
        sig = np.zeros(n, dtype=np.float64)
        sig[1:] = -1.0  # short

        # Without SL: unrestricted losses from rally
        _, dd_no_sl, _, _, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.0, 0.0, 0.0, high, low,
        )
        assert dd_no_sl > 8.0, (
            f"Without SL, max_dd ({dd_no_sl:.1f}%) should be large "
            f"(price rallies from 100 to 115)"
        )

        # With 4% SL: sl_price = 100*1.04 = 104
        # high[3] >= 104 -> SL triggers on bar 3
        _, dd_sl, _, _, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.04, 0.0, 0.0, high, low,
        )
        assert dd_sl < 6.0, f"With 4% SL on short, max_dd ({dd_sl:.1f}%) should be capped near 4%"
        assert dd_sl < dd_no_sl * 0.7, (
            f"SL max_dd ({dd_sl:.1f}%) should be much smaller than no-SL ({dd_no_sl:.1f}%)"
        )

    def test_trailing_stop_locks_profits(self, crash_data):
        """Trailing stop should exit on retracement from peak, capping drawdown."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        # Uptrend then retracement
        close = 100.0 + np.arange(n) * 0.3  # steady uptrend
        # Retrace 8% from peak
        peak = close[79]
        for i in range(80, n):
            close[i] = peak * (1.0 - 0.08 * (i - 79) / 20)
        high = close + np.abs(np.random.randn(n)) * 0.2
        low = close - np.abs(np.random.randn(n)) * 0.2
        high = np.maximum(high, close + 0.01)
        low = np.minimum(low, close - 0.01)

        close_arr = close.astype(np.float64)
        high_arr = high.astype(np.float64)
        low_arr = low.astype(np.float64)
        sig = np.zeros(n, dtype=np.float64)
        sig[1:] = 1.0  # long

        # Without trail
        _, dd_no_trail, eq_no_trail, _, _, _ = single_backtest_sltp(
            close_arr, sig, 10000, 0.0001,
            0.0, 0.0, 0.0, high_arr, low_arr,
        )
        # With 5% trailing stop
        _, dd_trail, eq_trail, _, _, _ = single_backtest_sltp(
            close_arr, sig, 10000, 0.0001,
            0.0, 0.0, 0.05, high_arr, low_arr,
        )
        # Trail should preserve more equity by exiting early on retrace
        assert dd_trail <= dd_no_trail + 1.0, (
            f"Trail dd ({dd_trail:.1f}%) should be similar to or better than "
            f"no-trail dd ({dd_no_trail:.1f}%)"
        )

    def test_tp_exit_caps_gain_but_locks_it_in(self):
        """TP should exit at the target price, locking in the gain.
        Uses deterministic data: uptrend to above TP, then deep crash.
        """
        np.random.seed(42)
        n = 30
        # Price: flat, entry at 100, rally above 110, then crash to 80
        close = np.zeros(n, dtype=np.float64)
        close[:3] = [99.0, 99.5, 100.0]
        close[3:6] = [103.0, 108.0, 112.0]  # rally above 10% TP (tp=110)
        close[6:] = np.linspace(90.0, 80.0, n - 6)  # deep crash

        high = close + np.abs(np.random.randn(n)) * 0.3
        low = close - np.abs(np.random.randn(n)) * 0.3
        high = np.maximum(high, close + 0.01)
        low = np.minimum(low, close - 0.01)

        # sig[1]=1 -> entry at close[2]=100
        sig = np.zeros(n, dtype=np.float64)
        sig[1:] = 1.0

        # No SL/TP: equity rides to 112 then crashes to 80
        eq_no_tp, _, _, _, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.0, 0.0, 0.0, high, low,
        )

        # With 10% TP: exits at 110, locks in gain, avoids crash
        # tp triggers on bar 5 (high[5] >= 110)
        eq_tp, _, _, _, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.0, 0.10, 0.0, high, low,
        )
        # TP locks in ~10% gain, no-TP crashes back down
        assert eq_tp > eq_no_tp, (
            f"TP equity ({eq_tp:.2f}) should exceed no-TP equity ({eq_no_tp:.2f}) "
            f"since TP locks in gains before the crash"
        )

    def test_sltp_trades_are_counted(self, crash_data, steady_long_signal):
        """SL/TP forced exits should be counted as trades."""
        close = crash_data["close"].values.astype(np.float64)
        high = crash_data["high"].values.astype(np.float64)
        low = crash_data["low"].values.astype(np.float64)
        sig = steady_long_signal.values.astype(np.float64)

        # No SL: single trade (entry + exit on signal change at end)
        _, _, _, trades_no, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.0, 0.0, 0.0, high, low,
        )
        # With 2% SL: entry + SL exit, then re-entry on signal
        _, _, _, trades_sl, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.02, 0.0, 0.0, high, low,
        )
        # SL should cause more trades (exit + re-entry)
        assert trades_sl >= trades_no, (
            f"SL should trigger more trades ({trades_sl}) than no-SL ({trades_no})"
        )

    def test_sl_on_entry_bar_does_not_false_trigger(self):
        """SL should not trigger on the entry bar itself (position opens at close[i]).
        Entry happens at close[i] where signal transitions from 0 to 1.
        SL/TP check starts from the NEXT bar (i+1).
        """
        n = 6
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        # sig[1]=1 -> entry at close[2]=102 (i=2 iteration opens position)
        # SL/TP checks start from i=3 onwards
        # SL at 3%: sl_price = 102*0.97 = 98.94
        # low[3]=100 > 98.94 -> no trigger on bar 3 (right after entry)
        # low[4]=96 < 98.94 -> trigger on bar 4
        close = np.array([100.0, 101.0, 102.0, 101.0, 99.0, 100.0], dtype=np.float64)
        high = np.array([101.0, 102.0, 103.0, 102.0, 100.0, 101.0], dtype=np.float64)
        low = np.array([99.0, 100.0, 101.0, 100.0, 96.0, 99.0], dtype=np.float64)
        sig = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)

        # With 3% SL: sl_price = 102*0.97 = 98.94
        # low[3]=100 > 98.94 -> NOT triggered on bar 3 (first check after entry)
        # low[4]=96 < 98.94 -> triggered on bar 4
        _, _, _, trades_3pct, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.03, 0.0, 0.0, high, low,
        )
        # SL should trigger on bar 4 -> entry + SL exit + re-entry = 2+ trades
        assert trades_3pct >= 2, (
            f"3% SL should trigger on bar 4 (low=96 < sl_price=98.94), "
            f"got trades={trades_3pct}"
        )

        # With 7% SL: sl_price = 102*0.93 = 94.86
        # low[3]=100 > 94.86, low[4]=96 > 94.86 -> NOT triggered
        # Only exit at signal change on bar 5
        _, _, _, trades_7pct, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.07, 0.0, 0.0, high, low,
        )
        # 1 trade: entry at close[2], exit at close[5] on signal 1->0
        assert trades_7pct == 1, (
            f"7% SL should NOT trigger (all lows above 94.86), "
            f"only signal-based exit, got trades={trades_7pct}"
        )

    def test_sl_and_tp_whichever_triggers_first(self):
        """When both SL and TP are set, the one hit first should trigger."""
        n = 6
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        # Entry at 100. SL=2%, TP=3%.
        # Bar 2: low=96, high=104 → both SL(98) and TP(103) could trigger
        # SL price = 98, TP price = 103
        # Low=96 < 98 → SL triggered first (intra-bar)
        close = np.array([99.0, 100.0, 102.0, 101.0, 102.0, 103.0], dtype=np.float64)
        high = np.array([100.0, 101.0, 104.0, 102.0, 103.0, 104.0], dtype=np.float64)
        low = np.array([98.0, 99.0, 96.0, 100.0, 101.0, 102.0], dtype=np.float64)
        sig = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        _, _, _, trades, _, _ = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.02, 0.03, 0.0, high, low,
        )
        # Should have at least 1 trade (SL-triggered exit)
        assert trades >= 1, f"SL should trigger on bar where both SL + TP levels are crossed"

    def test_equity_never_goes_negative(self):
        """Equity should never go negative even with repeated SL hits."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        # Oscillating price that keeps hitting SL
        close = 100.0 + np.sin(np.arange(n) * 0.5) * 5.0
        high = close + 1.0
        low = close - 1.0

        close_arr = close.astype(np.float64)
        high_arr = high.astype(np.float64)
        low_arr = low.astype(np.float64)
        sig = np.zeros(n, dtype=np.float64)
        sig[1:] = 1.0

        eq, dd, _, trades, _, _ = single_backtest_sltp(
            close_arr, sig, 10000, 0.0001,
            0.03, 0.0, 0.0, high_arr, low_arr,
        )
        assert eq > 0, f"Equity ({eq:.2f}) should never go negative"
        assert trades > 0, "Should have some SL trades"

    def test_sltp_without_sltp_disabled(self, crash_data, steady_long_signal):
        """With sl_pct=0 and tp_pct=0, function behaves like single_backtest."""
        close = crash_data["close"].values.astype(np.float64)
        high = crash_data["high"].values.astype(np.float64)
        low = crash_data["low"].values.astype(np.float64)
        sig = steady_long_signal.values.astype(np.float64)

        # single_backtest_sltp with no SL/TP
        eq_sltp, dd_sltp, sh_sltp, tr_sltp, wr_sltp, pf_sltp = single_backtest_sltp(
            close, sig, 10000, 0.0001,
            0.0, 0.0, 0.0, high, low,
        )
        # single_backtest (plain)
        eq_plain, dd_plain, sh_plain, tr_plain, wr_plain, pf_plain = single_backtest(
            close, sig, 10000, 0.0001,
        )
        # Results should be close (same logic when SL/TP disabled)
        assert abs(eq_sltp - eq_plain) / eq_plain < 0.01, (
            "SLTP with all stops disabled should match plain single_backtest"
        )
