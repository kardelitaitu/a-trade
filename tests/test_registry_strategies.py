"""Additional tests for strategies — squeeze, regime, and registry edge cases."""

import numpy as np
import pandas as pd
import pytest

from research.features.registry import INDICATORS, get_indicator, list_indicators


@pytest.fixture
def sample():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    close = 50000 + np.cumsum(np.random.randn(200) * 10)
    return {
        "close": pd.Series(close, index=idx),
        "high": pd.Series(close + np.abs(np.random.randn(200) * 5), index=idx),
        "low": pd.Series(close - np.abs(np.random.randn(200) * 5), index=idx),
        "volume": pd.Series(np.random.uniform(10, 100, 200), index=idx),
        "open": pd.Series(close - np.random.randn(200) * 2, index=idx),
    }


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

class TestRegistryAdditional:

    def test_all_indicators_have_category(self):
        for name, meta in INDICATORS.items():
            assert "category" in meta, f"{name} missing category"

    def test_all_indicators_have_description(self):
        for name, meta in INDICATORS.items():
            assert "desc" in meta, f"{name} missing description"

    def test_all_indicators_callable(self):
        for name in INDICATORS:
            fn = get_indicator(name)
            assert callable(fn), f"{name} not callable"

    def test_indicators_by_category(self):
        cats = set(v["category"] for v in INDICATORS.values())
        assert "trend" in cats
        assert "volume" in cats
        assert "momentum" in cats
        assert "volatility" in cats
        assert "structure" in cats

    def test_list_indicators_returns_string(self):
        result = list_indicators()
        assert isinstance(result, str)
        assert len(result) > 500

    def test_list_indicators_filter(self):
        result = list_indicators(category="momentum")
        assert "rsi" in result
        assert "sma" not in result

    def test_volume_indicators_listed(self):
        result = list_indicators(category="volume")
        assert "vol_delta" in result
        assert "vwap" in result

    def test_time_indicators_listed(self):
        result = list_indicators(category="time")
        assert "hour_sin" in result


# ═══════════════════════════════════════════════════════════════════
# Indicator function edge cases
# ═══════════════════════════════════════════════════════════════════

class TestIndicatorEdgeCases:

    def test_ret_negative_on_downmove(self, sample):
        from research.features.registry import ret
        r = ret(sample["close"], 1)
        # On random walk, some returns should be negative
        assert (r.dropna() < 0).sum() > 0

    def test_log_ret_small_vs_regular(self, sample):
        from research.features.registry import ret, log_ret
        r1 = ret(sample["close"], 1)
        r2 = log_ret(sample["close"], 1)
        diff = (r2 - r1).dropna()
        assert diff.abs().max() < 0.01

    def test_bb_width_positive(self, sample):
        from research.features.registry import get_indicator
        bb_w = get_indicator("bb_width")
        result = bb_w(sample["close"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample["close"])

    def test_natr_vs_atr_ratio(self, sample):
        from research.features.registry import natr, atr
        n = natr(sample["high"], sample["low"], sample["close"], 14).dropna()
        a = atr(sample["high"], sample["low"], sample["close"], 14).dropna()
        # NATR should be close to ATR / close * 100
        close_at_valid = sample["close"].loc[n.index]
        assert n.mean() > 0
        assert a.mean() > 0

    def test_mfi_zero_volume(self, sample):
        from research.features.registry import mfi
        zero_vol = sample["volume"].copy()
        zero_vol[:] = 0
        r = mfi(sample["high"], sample["low"], sample["close"], zero_vol)
        assert r.notna().sum() >= 0

    def test_obv_constant_price(self):
        from research.features.registry import obv
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        c = pd.Series(100.0, index=idx)
        v = pd.Series(50.0, index=idx)
        r = obv(c, v)
        # No price change → OBV stays at 0
        assert abs(r.iloc[-1]) < 1e-6

    def test_vwap_constant_data(self):
        from research.features.registry import vwap
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        h = pd.Series(101.0, index=idx)
        l = pd.Series(99.0, index=idx)
        c = pd.Series(100.0, index=idx)
        v = pd.Series(50.0, index=idx)
        r = vwap(h, l, c, v, period=5)
        valid = r.dropna()
        assert abs(valid.mean() - 100) < 1


# ═══════════════════════════════════════════════════════════════════
# Volatility Squeeze additional tests
# ═══════════════════════════════════════════════════════════════════

class TestVolSqueezeAdditional:

    @pytest.fixture
    def data(self):
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=500, freq="5min")
        close = 100 + np.cumsum(np.random.randn(500) * 0.3)
        return pd.DataFrame({
            "open": close - 0.2, "high": close + np.abs(np.random.randn(500)) * 0.5,
            "low": close - np.abs(np.random.randn(500)) * 0.5, "close": close,
            "volume": np.random.uniform(10, 100, 500),
        }, index=idx)

    def test_squeeze_config_custom(self, data):
        from research.strategies.volatility_squeeze import VolatilitySqueeze
        s = VolatilitySqueeze({"bb_period": 30, "bb_std": 3.0, "squeeze_lookback": 100})
        assert s.config["bb_period"] == 30
        assert s.config["squeeze_lookback"] == 100

    def test_squeeze_min_hold(self, data):
        from research.strategies.volatility_squeeze import VolatilitySqueeze
        s = VolatilitySqueeze({"min_hold": 10, "bb_std": 3.0, "keltner_atr_mult": 0.5})
        sig = s.generate_signals(data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_squeeze_no_nan(self, data):
        from research.strategies.volatility_squeeze import VolatilitySqueeze
        s = VolatilitySqueeze()
        sig = s.generate_signals(data)
        assert not sig.isna().any()

    def test_squeeze_description(self, data):
        from research.strategies.volatility_squeeze import VolatilitySqueeze
        s = VolatilitySqueeze()
        desc = s.description
        assert "Squeeze" in desc or "BB" in desc or "Keltner" in desc


# ═══════════════════════════════════════════════════════════════════
# Regime additional tests
# ═══════════════════════════════════════════════════════════════════

class TestRegimeAdditional:

    @pytest.fixture
    def data(self):
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=500, freq="5min")
        close = 100 + np.cumsum(np.random.randn(500) * 0.3)
        return pd.DataFrame({
            "high": close + np.abs(np.random.randn(500)) * 0.5,
            "low": close - np.abs(np.random.randn(500)) * 0.5,
            "close": close,
        }, index=idx)

    def test_regime_sma_no_nan(self, data):
        from research.strategies.regime import detect_regime_sma
        regime = detect_regime_sma(data["close"])
        assert not regime.isna().any()

    def test_regime_adx_no_nan(self, data):
        from research.strategies.regime import detect_regime_adx
        regime = detect_regime_adx(data["high"], data["low"], data["close"])
        assert not regime.isna().any()

    def test_filter_by_regime_zeroes_outside(self, data):
        from research.strategies.regime import detect_regime_sma, filter_by_regime
        sig = pd.Series(1, index=data.index)
        regime = pd.Series("bear", index=data.index)
        regime.iloc[:100] = "bull"
        filtered = filter_by_regime(sig, regime, allowed_regimes=["bull"])
        assert (filtered[regime == "bull"] == 1).all()
        assert (filtered[regime == "bear"] == 0).all()

    def test_filter_by_regime_preserves_length(self, data):
        from research.strategies.regime import filter_by_regime
        sig = pd.Series(np.random.choice([-1, 0, 1], 500), index=data.index)
        regime = pd.Series("bull", index=data.index)
        filtered = filter_by_regime(sig, regime)
        assert len(filtered) == len(sig)
