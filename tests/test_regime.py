"""Tests for research/strategies/regime.py."""
import numpy as np
import pandas as pd
import pytest
from research.strategies.regime import detect_regime_sma, detect_regime_adx, filter_by_regime


@pytest.fixture
def trend_data():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.3)
    return pd.DataFrame({"high": close + 0.5, "low": close - 0.5, "close": close}, index=idx)


class TestRegime:

    def test_sma_returns_series(self, trend_data):
        r = detect_regime_sma(trend_data["close"])
        assert isinstance(r, pd.Series)

    def test_sma_valid_values(self, trend_data):
        r = detect_regime_sma(trend_data["close"])
        assert set(r.dropna().unique()).issubset({"bull", "bear", "sideways"})

    def test_sma_no_nan_after_warmup(self, trend_data):
        r = detect_regime_sma(trend_data["close"])
        assert not r.iloc[100:].isna().any()

    def test_adx_returns_series(self, trend_data):
        r = detect_regime_adx(trend_data["high"], trend_data["low"], trend_data["close"])
        assert isinstance(r, pd.Series)

    def test_adx_valid_values(self, trend_data):
        r = detect_regime_adx(trend_data["high"], trend_data["low"], trend_data["close"])
        assert set(r.dropna().unique()).issubset({"bull", "bear", "sideways"})

    def test_adx_no_nan_after_warmup(self, trend_data):
        r = detect_regime_adx(trend_data["high"], trend_data["low"], trend_data["close"])
        assert not r.iloc[100:].isna().any()

    def test_filter_by_regime_bull(self, trend_data):
        sig = pd.Series(np.random.choice([-1, 0, 1], 500), index=trend_data.index)
        regime = pd.Series("bear", index=trend_data.index)
        regime.iloc[100:300] = "bull"
        filtered = filter_by_regime(sig, regime, allowed_regimes=["bull"])
        assert (filtered[regime == "bull"] == sig[regime == "bull"]).all()
        assert (filtered[regime == "bear"] == 0).all()

    def test_filter_by_regime_allows_all(self, trend_data):
        sig = pd.Series(np.random.choice([-1, 0, 1], 500), index=trend_data.index)
        regime = pd.Series("bull", index=trend_data.index)
        filtered = filter_by_regime(sig, regime, allowed_regimes=["bull"])
        assert (filtered == sig).all()

    def test_filter_by_regime_empty_allowed(self, trend_data):
        sig = pd.Series(1, index=trend_data.index)
        regime = pd.Series("bull", index=trend_data.index)
        filtered = filter_by_regime(sig, regime, allowed_regimes=[])
        assert (filtered == 0).all()

    def test_sma_custom_threshold(self, trend_data):
        r = detect_regime_sma(trend_data["close"], period=50, bull_threshold=1.01, bear_threshold=0.99)
        assert set(r.dropna().unique()).issubset({"bull", "bear", "sideways"})

    def test_adx_default_params(self, trend_data):
        r = detect_regime_adx(trend_data["high"], trend_data["low"], trend_data["close"])
        assert set(r.dropna().unique()).issubset({"bull", "bear", "sideways"})
