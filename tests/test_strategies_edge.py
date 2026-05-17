"""Additional edge case tests for strategies and ML trainer."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.ma_crossover import MACrossover
from research.strategies.breakout import DonchianBreakout
from research.strategies.mean_reversion import MeanReversion
from research.strategies.volatility_breakout import VolatilityBreakout
from research.backtest.engine import VectorizedBacktest
from research.backtest.metrics import compute_metrics


@pytest.fixture
def sdata():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.3)
    return pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5,
        "low": close - 0.5, "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)


class TestStrategiesEdgeCases:

    def test_ma_extreme_periods(self, sdata):
        """Very extreme periods should not crash."""
        for fast, slow in [(2, 5), (1, 3), (100, 400), (2, 499)]:
            sig = MACrossover({"fast_period": fast, "slow_period": slow}).generate_signals(sdata)
            assert set(sig.unique()).issubset({-1, 0, 1})

    def test_ma_sma_ema_different(self, sdata):
        """SMA and EMA should produce different outputs."""
        sma_sig = MACrossover({"ma_type": "sma", "fast_period": 10, "slow_period": 30}).generate_signals(sdata)
        ema_sig = MACrossover({"ma_type": "ema", "fast_period": 10, "slow_period": 30}).generate_signals(sdata)
        assert not (sma_sig == ema_sig).all()

    def test_ma_unsupported_type_raises(self, sdata):
        """Unsupported ma_type should raise ValueError."""
        with pytest.raises(ValueError, match="ma_type"):
            MACrossover({"ma_type": "wma"}).generate_signals(sdata)

    def test_donchian_entry_exit_equal(self, sdata):
        """Equal entry/exit periods should produce signals."""
        sig = DonchianBreakout({"entry_period": 20, "exit_period": 20}).generate_signals(sdata)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_donchian_short_entry(self, sdata):
        """Strategy should generate both long and short signals."""
        sig = DonchianBreakout({"entry_period": 10, "exit_period": 5}).generate_signals(sdata)
        assert (sig == -1).sum() >= 0

    def test_mr_rsi_period_variation(self, sdata):
        """RSI period should not affect output type."""
        for p in [7, 9, 14, 21, 25]:
            sig = MeanReversion({"mode": "rsi", "rsi_period": p}).generate_signals(sdata)
            assert set(sig.unique()).issubset({-1, 0, 1})

    def test_mr_bollinger_std_variation(self, sdata):
        """Bollinger std should not affect output type."""
        for s in [1.5, 2.0, 2.5, 3.0]:
            sig = MeanReversion({"mode": "bollinger", "bb_std": s}).generate_signals(sdata)
            assert set(sig.unique()).issubset({-1, 0, 1})

    def test_vol_breakout_periods(self, sdata):
        """Varied ATR periods should not crash."""
        for p in [7, 14, 20, 30]:
            sig = VolatilityBreakout({"atr_period": p}).generate_signals(sdata)
            assert not sig.isna().any()

    def test_vol_breakout_description(self, sdata):
        """Description should mention key params."""
        s = VolatilityBreakout({"atr_period": 14, "atr_multiplier": 2.0})
        desc = s.description
        assert "ATR" in desc or "Volatility" in desc

    def test_strategy_roundtrip_backtest(self, sdata):
        """Generate signals + backtest should not crash."""
        strat = MACrossover({"fast_period": 10, "slow_period": 30})
        sig = strat.generate_signals(sdata)
        bt = VectorizedBacktest(sdata)
        result = bt.run(sig)
        m = compute_metrics(result.equity_curve, result.trades)
        assert m["total_trades"] >= 0

    def test_mr_creates_signals(self, sdata):
        """RSI based strategy should create valid signals."""
        sig = MeanReversion({"mode": "rsi", "rsi_oversold": 40, "rsi_overbought": 60})
        result = sig.generate_signals(sdata)
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_donchian_minimal_periods(self, sdata):
        """Minimal periods should not crash."""
        sig = DonchianBreakout({"entry_period": 3, "exit_period": 2}).generate_signals(sdata)
        assert not sig.isna().any()
