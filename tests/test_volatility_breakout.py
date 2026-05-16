"""Tests for research/strategies/volatility_breakout.py."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.volatility_breakout import VolatilityBreakout


@pytest.fixture
def vol_data():
    """Data with clear volatility expansion — long quiet then loud periods."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")

    close = np.zeros(500)
    # Quiet period (0-150)
    close[:150] = 100 + np.cumsum(np.random.randn(150) * 0.05)
    # Vol expansion up (150-200) — big bars
    close[150:200] = close[149] + np.cumsum(np.abs(np.random.randn(50)) * 0.8 + 0.1)
    # Quiet period (200-350)
    close[200:350] = close[199] + np.cumsum(np.random.randn(150) * 0.05)
    # Vol expansion down (350-400)
    close[350:400] = close[349] - np.cumsum(np.abs(np.random.randn(50)) * 0.8 + 0.1)
    # Quiet period
    close[400:] = close[399] + np.cumsum(np.random.randn(100) * 0.05)

    # Build high/low with expanded ranges during vol periods
    vol_mask = np.zeros(500, dtype=bool)
    vol_mask[150:200] = True
    vol_mask[350:400] = True

    high = close.copy()
    low = close.copy()
    high[~vol_mask] += 0.3
    low[~vol_mask] -= 0.3
    high[vol_mask] += np.abs(np.random.randn(vol_mask.sum())) * 1.5 + 0.5
    low[vol_mask] -= np.abs(np.random.randn(vol_mask.sum())) * 1.5 + 0.5

    df = pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)
    return df


class TestVolatilityBreakout:

    def test_default_config(self):
        s = VolatilityBreakout()
        assert s.config["atr_period"] == 14
        assert s.config["atr_multiplier"] == 2.0
        assert s.config["min_hold"] == 6

    def test_name(self):
        assert "Volatility" in VolatilityBreakout().name

    def test_signals_generated(self, vol_data):
        """Should produce signals without errors."""
        s = VolatilityBreakout()
        signals = s.generate_signals(vol_data)
        assert not signals.isna().any()
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_long_on_vol_spike_up(self, vol_data):
        """Volatility spike upward should trigger long signals."""
        s = VolatilityBreakout({"atr_period": 10, "atr_multiplier": 1.5})
        signals = s.generate_signals(vol_data)
        # After vol expansion at index 150, should have some longs
        post_spike = signals.iloc[150:200]
        assert (post_spike == 1).sum() > 0, "No long signals after vol spike up"

    def test_no_nan(self, vol_data):
        """No NaN in signals."""
        s = VolatilityBreakout()
        signals = s.generate_signals(vol_data)
        assert not signals.isna().any()
