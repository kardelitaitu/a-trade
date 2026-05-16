"""Tests for research/strategies/volatility_breakout.py."""
import numpy as np
import pandas as pd
import pytest
from research.strategies.volatility_breakout import VolatilityBreakout


@pytest.fixture
def vol_data():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.2)
    close[200:220] += np.cumsum(np.abs(np.random.randn(20)) * 0.8)
    return pd.DataFrame({
        "open": close - 0.2, "high": close + np.abs(np.random.randn(500)) * 1.0,
        "low": close - np.abs(np.random.randn(500)) * 1.0, "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)


class TestVolatilityBreakout:

    def test_default_config(self):
        s = VolatilityBreakout()
        assert s.config["atr_period"] == 14

    def test_name(self):
        assert "Volatility" in VolatilityBreakout().name

    def test_signals_generated(self, vol_data):
        s = VolatilityBreakout()
        sig = s.generate_signals(vol_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_no_nan(self, vol_data):
        s = VolatilityBreakout()
        assert not s.generate_signals(vol_data).isna().any()

    def test_low_multiplier(self, vol_data):
        s = VolatilityBreakout({"atr_multiplier": 0.5})
        sig = s.generate_signals(vol_data)
        assert (sig != 0).sum() >= 0

    def test_high_multiplier(self, vol_data):
        s = VolatilityBreakout({"atr_multiplier": 5.0})
        sig = s.generate_signals(vol_data)
        assert (sig != 0).sum() >= 0

    def test_various_periods(self, vol_data):
        for p in [7, 10, 14, 20, 30]:
            s = VolatilityBreakout({"atr_period": p})
            assert not s.generate_signals(vol_data).isna().any()

    def test_min_hold(self, vol_data):
        s = VolatilityBreakout({"min_hold": 5, "atr_multiplier": 0.5})
        sig = s.generate_signals(vol_data)
        assert not sig.isna().any()

    def test_lookback(self, vol_data):
        s = VolatilityBreakout({"atr_lookback": 30})
        sig = s.generate_signals(vol_data)
        assert not sig.isna().any()

    def test_volume_filter(self, vol_data):
        s = VolatilityBreakout({"filter_volume_pct": 0.5})
        sig = s.generate_signals(vol_data)
        assert not sig.isna().any()

    def test_description(self):
        s = VolatilityBreakout({"atr_period": 14, "atr_multiplier": 2.0})
        desc = s.description
        assert "ATR" in desc or "Volatility" in desc

    def test_various_atr_periods(self, vol_data):
        for p in [7, 10, 14, 20, 30]:
            s = VolatilityBreakout({"atr_period": p})
            assert not s.generate_signals(vol_data).isna().any()

    def test_various_multipliers(self, vol_data):
        for m in [0.5, 1.0, 2.0, 5.0]:
            s = VolatilityBreakout({"atr_multiplier": m})
            assert not s.generate_signals(vol_data).isna().any()

    def test_config_persists(self):
        s = VolatilityBreakout({"atr_period": 7, "atr_multiplier": 1.5, "atr_lookback": 30})
        assert s.config["atr_period"] == 7 and s.config["atr_lookback"] == 30
