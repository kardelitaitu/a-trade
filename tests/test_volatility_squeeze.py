"""Tests for research/strategies/volatility_squeeze.py."""
import numpy as np
import pandas as pd
import pytest
from research.strategies.volatility_squeeze import VolatilitySqueeze


@pytest.fixture
def squeeze_data():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.3)
    close[150:250] = 102 + np.cumsum(np.abs(np.random.randn(100)) * 0.5 + 0.05)
    close[250:350] = 105 + np.random.randn(100) * 0.3
    return pd.DataFrame({
        "open": close - 0.2, "high": close + np.abs(np.random.randn(500)) * 1.0,
        "low": close - np.abs(np.random.randn(500)) * 1.0, "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)


class TestVolatilitySqueeze:

    def test_default_config(self):
        s = VolatilitySqueeze()
        assert s.config["bb_period"] == 20

    def test_name(self):
        assert "Squeeze" in VolatilitySqueeze().name

    def test_signals_generated(self, squeeze_data):
        s = VolatilitySqueeze()
        sig = s.generate_signals(squeeze_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_no_nan(self, squeeze_data):
        assert not VolatilitySqueeze().generate_signals(squeeze_data).isna().any()

    def test_custom_bb(self, squeeze_data):
        s = VolatilitySqueeze({"bb_period": 30, "bb_std": 3.0})
        assert not s.generate_signals(squeeze_data).isna().any()

    def test_custom_keltner(self, squeeze_data):
        s = VolatilitySqueeze({"keltner_period": 30, "keltner_atr_mult": 2.0})
        assert not s.generate_signals(squeeze_data).isna().any()

    def test_lookback(self, squeeze_data):
        s = VolatilitySqueeze({"squeeze_lookback": 100})
        sig = s.generate_signals(squeeze_data)
        assert not sig.isna().any()

    def test_min_hold(self, squeeze_data):
        s = VolatilitySqueeze({"min_hold": 5})
        assert not s.generate_signals(squeeze_data).isna().any()

    def test_various_bb_std(self, squeeze_data):
        for bb in [1.5, 2.0, 2.5, 3.0]:
            s = VolatilitySqueeze({"bb_std": bb})
            assert not s.generate_signals(squeeze_data).isna().any()

    def test_various_keltner_mult(self, squeeze_data):
        for km in [0.5, 1.0, 1.5, 2.0]:
            s = VolatilitySqueeze({"keltner_atr_mult": km})
            assert not s.generate_signals(squeeze_data).isna().any()

    def test_description(self):
        assert "Squeeze" in VolatilitySqueeze().description

    def test_volume_filter(self, squeeze_data):
        s = VolatilitySqueeze({"filter_volume_pct": 0.5})
        assert not s.generate_signals(squeeze_data).isna().any()

    def test_config_params(self):
        s = VolatilitySqueeze({"bb_period": 25, "bb_std": 2.5, "keltner_period": 15, "keltner_atr_mult": 1.2})
        assert s.config["bb_period"] == 25
        assert s.config["keltner_period"] == 15
