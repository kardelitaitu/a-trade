"""Tests for research/strategies/ma_crossover.py."""
import numpy as np
import pandas as pd
import pytest
from research.strategies.ma_crossover import MACrossover


@pytest.fixture
def trend_data():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.3 + 0.05)
    return pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.random.uniform(10, 100, 500),
    }, index=idx)


class TestMACrossover:

    def test_default_config(self):
        s = MACrossover()
        assert s.config["fast_period"] == 12
        assert s.config["slow_period"] == 26
        assert s.config["ma_type"] == "ema"

    def test_name(self):
        assert "MA Crossover" in MACrossover().name

    def test_long_signals_in_uptrend(self, trend_data):
        s = MACrossover()
        signals = s.generate_signals(trend_data)
        warmup = signals.iloc[50:]
        assert (warmup == 1).sum() > (warmup == -1).sum()

    def test_no_nan_signals(self, trend_data):
        s = MACrossover()
        assert not s.generate_signals(trend_data).isna().any()

    def test_volume_filter(self, trend_data):
        raw = MACrossover().generate_signals(trend_data)
        fil = MACrossover({"filter_volume_pct": 0.5}).generate_signals(trend_data)
        assert (fil == 0).sum() >= (raw == 0).sum()

    def test_ma_type_sma(self, trend_data):
        s = MACrossover({"ma_type": "sma"})
        sig = s.generate_signals(trend_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    # --- ADDITIONAL TESTS ---

    def test_ma_type_wma_raises(self, trend_data):
        with pytest.raises(ValueError, match="ma_type"):
            MACrossover({"ma_type": "wma"}).generate_signals(trend_data)

    def test_ma_type_hma_raises(self, trend_data):
        with pytest.raises(ValueError, match="ma_type"):
            MACrossover({"ma_type": "hma"}).generate_signals(trend_data)

    def test_fast_equals_slow_produces_zeros(self, trend_data):
        s = MACrossover({"fast_period": 30, "slow_period": 30})
        sig = s.generate_signals(trend_data)
        assert (sig == 0).all()

    def test_fast_greater_than_slow_swaps(self, trend_data):
        s = MACrossover({"fast_period": 50, "slow_period": 10})
        sig = s.generate_signals(trend_data)
        assert not sig.isna().any()

    def test_custom_indicator_names(self, trend_data):
        s = MACrossover({"fast_indicator": "ema", "slow_indicator": "sma"})
        sig = s.generate_signals(trend_data)
        assert not sig.isna().any()

    def test_minimal_periods(self, trend_data):
        s = MACrossover({"fast_period": 2, "slow_period": 5})
        sig = s.generate_signals(trend_data)
        assert not sig.isna().any()

    def test_maximal_periods(self, trend_data):
        s = MACrossover({"fast_period": 100, "slow_period": 200})
        sig = s.generate_signals(trend_data)
        assert not sig.isna().any()

    def test_description_contains_params(self):
        s = MACrossover({"fast_period": 10, "slow_period": 30})
        assert "10" in s.description and "30" in s.description

    def test_atr_filter(self, trend_data):
        s = MACrossover({"filter_atr_mult": 2.0})
        sig = s.generate_signals(trend_data)
        assert not sig.isna().any()

    def test_sma_ema_different(self, trend_data):
        sma = MACrossover({"ma_type": "sma", "fast_period": 10, "slow_period": 30}).generate_signals(trend_data)
        ema = MACrossover({"ma_type": "ema", "fast_period": 10, "slow_period": 30}).generate_signals(trend_data)
        assert not (sma == ema).all()

    def test_all_ma_types(self, trend_data):
        for t in ["sma", "ema"]:
            sig = MACrossover({"ma_type": t, "fast_period": 10, "slow_period": 30}).generate_signals(trend_data)
            assert set(sig.unique()).issubset({-1, 0, 1})

    def test_config_override_all(self, trend_data):
        s = MACrossover({"fast_period": 5, "slow_period": 15, "ma_type": "sma"})
        assert s.config["fast_period"] == 5 and s.config["slow_period"] == 15

    def test_ema_alternate(self, trend_data):
        for f, s_ in [(5, 20), (8, 24), (12, 26), (15, 30)]:
            sig = MACrossover({"fast_period": f, "slow_period": s_}).generate_signals(trend_data)
            assert not sig.isna().any()

    def test_sma_alternate(self, trend_data):
        for f, s_ in [(5, 20), (8, 24), (12, 26), (15, 30)]:
            sig = MACrossover({"ma_type": "sma", "fast_period": f, "slow_period": s_}).generate_signals(trend_data)
            assert not sig.isna().any()

    def test_ma_unsupported_types_raise(self):
        for t in ["wma", "hma"]:
            with pytest.raises(ValueError, match="ma_type"):
                MACrossover({"ma_type": t, "fast_period": 5, "slow_period": 20}).generate_signals(pd.DataFrame())
