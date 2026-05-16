"""Tests for research/strategies/mean_reversion.py."""
import numpy as np
import pandas as pd
import pytest
from research.strategies.mean_reversion import MeanReversion


@pytest.fixture
def meanrev_data():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.3)
    return pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.random.uniform(10, 100, 500),
    }, index=idx)


class TestMeanReversion:

    def test_default_config(self):
        s = MeanReversion()
        assert s.config["mode"] == "rsi"
        assert s.config["rsi_period"] == 14

    def test_name(self):
        assert "Mean Reversion" in MeanReversion().name

    def test_rsi_signals_generated(self, meanrev_data):
        s = MeanReversion({"mode": "rsi"})
        sig = s.generate_signals(meanrev_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_bollinger_signals_generated(self, meanrev_data):
        s = MeanReversion({"mode": "bollinger"})
        sig = s.generate_signals(meanrev_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_no_nan(self, meanrev_data):
        s = MeanReversion({"mode": "rsi"})
        assert not s.generate_signals(meanrev_data).isna().any()

    def test_volume_filter(self, meanrev_data):
        raw = MeanReversion({"mode": "rsi"}).generate_signals(meanrev_data)
        fil = MeanReversion({"mode": "rsi", "filter_volume_pct": 0.5}).generate_signals(meanrev_data)
        assert (fil == 0).sum() >= (raw == 0).sum()

    # --- ADDITIONAL TESTS ---

    def test_rsi_wide_thresholds(self, meanrev_data):
        s = MeanReversion({"mode": "rsi", "rsi_oversold": 10, "rsi_overbought": 90})
        sig = s.generate_signals(meanrev_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_rsi_narrow_thresholds(self, meanrev_data):
        s = MeanReversion({"mode": "rsi", "rsi_oversold": 40, "rsi_overbought": 60})
        sig = s.generate_signals(meanrev_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_rsi_various_periods(self, meanrev_data):
        for p in [5, 7, 9, 14, 21, 25]:
            s = MeanReversion({"mode": "rsi", "rsi_period": p})
            assert not s.generate_signals(meanrev_data).isna().any()

    def test_bollinger_various_stds(self, meanrev_data):
        for std in [1.0, 1.5, 2.0, 2.5, 3.0]:
            s = MeanReversion({"mode": "bollinger", "bb_std": std})
            assert not s.generate_signals(meanrev_data).isna().any()

    def test_bollinger_various_periods(self, meanrev_data):
        for p in [10, 20, 30, 50]:
            s = MeanReversion({"mode": "bollinger", "bb_period": p})
            assert not s.generate_signals(meanrev_data).isna().any()

    def test_description(self):
        s = MeanReversion({"mode": "rsi"})
        assert "RSI" in s.description or "mean" in s.description.lower()

    def test_custom_rsi_params(self, meanrev_data):
        s = MeanReversion({"mode": "rsi", "rsi_period": 7, "rsi_oversold": 25, "rsi_overbought": 75})
        sig = s.generate_signals(meanrev_data)
        assert not sig.isna().any()

    def test_custom_bb_params(self, meanrev_data):
        s = MeanReversion({"mode": "bollinger", "bb_period": 15, "bb_std": 2.5})
        sig = s.generate_signals(meanrev_data)
        assert not sig.isna().any()
