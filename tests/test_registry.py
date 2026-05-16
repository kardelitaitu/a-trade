"""Tests for research/features/registry.py — all 29 registered indicators."""

import numpy as np
import pandas as pd
import pytest

from research.features.registry import (
    INDICATORS, get_indicator, list_indicators,
    sma, ema, wma, hma, rsi, atr, stoch, williams_r, cci, roc, momentum,
    macd_line, macd_signal, macd_hist,
    keltner, natr,
    obv, vwap, mfi, vol_delta, cmf,
    donchian, pivot_points,
    bollinger_bands,
)


@pytest.fixture
def sample():
    """Standard test fixture — 200 rows of simulated BTC data."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    close = 50000 + np.cumsum(np.random.randn(200) * 10)
    high = close + np.abs(np.random.randn(200) * 5)
    low = close - np.abs(np.random.randn(200) * 5)
    volume = np.random.uniform(10, 100, 200)
    return {
        "close": pd.Series(close, index=idx),
        "high": pd.Series(high, index=idx),
        "low": pd.Series(low, index=idx),
        "volume": pd.Series(volume, index=idx),
        "open": pd.Series(close - np.random.randn(200) * 2, index=idx),
    }


# ═══════════════════════════════════════════════════════════════════════
# Registry tests
# ═══════════════════════════════════════════════════════════════════════

class TestRegistry:

    def test_all_indicators_listed(self):
        """list_indicators should return all 29 names."""
        result = list_indicators()
        for name in INDICATORS:
            assert name in result, f"{name} missing from list_indicators()"

    def test_get_indicator(self):
        """get_indicator should return callable."""
        fn = get_indicator("sma")
        assert callable(fn)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown indicator"):
            get_indicator("nonexistent")

    def test_category_filter(self):
        result = list_indicators(category="trend")
        assert "sma" in result
        assert "rsi" not in result


# ═══════════════════════════════════════════════════════════════════════
# Price Trend
# ═══════════════════════════════════════════════════════════════════════

class TestSMA:
    def test_sma_basic(self, sample):
        r = sma(sample["close"], 5)
        assert r.iloc[0] != r.iloc[0]  # NaN
        assert not np.isnan(r.iloc[5])  # valid after warmup

class TestEMA:
    def test_ema_basic(self, sample):
        r = ema(sample["close"], 5)
        assert r.notna().sum() > 0
        assert r.iloc[-1] > 0

class TestWMA:
    def test_wma_basic(self, sample):
        r = wma(sample["close"], 5)
        assert r.notna().sum() > 0
        assert r.iloc[-1] > 0

    def test_wma_weighted(self, sample):
        """WMAs should react differently from SMA."""
        wma_r = wma(sample["close"], 5)
        sma_r = sma(sample["close"], 5)
        # At least some values should differ (WMA ≠ SMA)
        assert not (wma_r == sma_r).all()

class TestHMA:
    def test_hma_basic(self, sample):
        r = hma(sample["close"], 10)
        assert r.notna().sum() > 0

    def test_hma_output_type(self, sample):
        r = hma(sample["close"], 10)
        assert isinstance(r, pd.Series)

class TestMACD:
    def test_macd_line(self, sample):
        r = macd_line(sample["close"])
        assert r.notna().sum() > 0

    def test_macd_signal(self, sample):
        r = macd_signal(sample["close"])
        assert r.notna().sum() > 0

    def test_macd_hist(self, sample):
        r = macd_hist(sample["close"])
        assert r.notna().sum() > 0
        # hist = line - signal
        line = macd_line(sample["close"])
        sig = macd_signal(sample["close"])
        pd.testing.assert_series_equal(r, line - sig, check_dtype=False)


# ═══════════════════════════════════════════════════════════════════════
# Momentum
# ═══════════════════════════════════════════════════════════════════════

class TestRSI:
    def test_rsi_range(self, sample):
        r = rsi(sample["close"], 14)
        valid = r.dropna()
        assert valid.between(0, 100).all()

class TestStoch:
    def test_stoch_output(self, sample):
        k, d = stoch(sample["high"], sample["low"], sample["close"])
        assert k.notna().sum() > 0
        assert d.notna().sum() > 0
        valid = k.dropna()
        assert valid.between(0, 100).all()

class TestWilliamsR:
    def test_williams_r_output(self, sample):
        r = williams_r(sample["high"], sample["low"], sample["close"])
        valid = r.dropna()
        assert valid.between(-100, 0).all()

class TestCCI:
    def test_cci_output(self, sample):
        r = cci(sample["high"], sample["low"], sample["close"])
        assert r.notna().sum() > 0
        # CCI is unbounded, but typical range is -300 to +300
        assert not r.dropna().empty

class TestROC:
    def test_roc_basic(self, sample):
        r = roc(sample["close"], 10)
        assert r.notna().sum() > 0
        assert abs(r.iloc[15]) < 100  # reasonable %

class TestMomentum:
    def test_momentum_basic(self, sample):
        r = momentum(sample["close"], 10)
        assert r.notna().sum() > 0


# ═══════════════════════════════════════════════════════════════════════
# Volatility
# ═══════════════════════════════════════════════════════════════════════

class TestATR:
    def test_atr_positive(self, sample):
        r = atr(sample["high"], sample["low"], sample["close"], 14)
        valid = r.dropna()
        assert (valid > 0).all()

class TestNATR:
    def test_natr_positive(self, sample):
        r = natr(sample["high"], sample["low"], sample["close"], 14)
        valid = r.dropna()
        assert (valid > 0).all()
        # NATR is ATR/close*100, so should be small % (<50%)
        assert valid.max() < 50

class TestBollinger:
    def test_bb_order(self, sample):
        mid, upper, lower = bollinger_bands(sample["close"])
        valid = mid.notna()
        assert (upper[valid] >= mid[valid]).all()
        assert (lower[valid] <= mid[valid]).all()

class TestKeltner:
    def test_keltner_order(self, sample):
        mid, upper, lower = keltner(sample["high"], sample["low"], sample["close"])
        valid = upper.notna()
        assert valid.sum() > 0
        assert (upper[valid] >= mid[valid]).all()
        assert (lower[valid] <= mid[valid]).all()


# ═══════════════════════════════════════════════════════════════════════
# Volume
# ═══════════════════════════════════════════════════════════════════════

class TestOBV:
    def test_obv_basic(self, sample):
        r = obv(sample["close"], sample["volume"])
        assert r.notna().all()
        # OBV should vary as price moves
        assert r.iloc[0] != r.iloc[-1] or abs(r).sum() > 0

class TestVWAP:
    def test_vwap_basic(self, sample):
        r = vwap(sample["high"], sample["low"], sample["close"], sample["volume"])
        assert r.notna().sum() > 0
        # VWAP should be near close price
        assert abs(r.iloc[-1] / sample["close"].iloc[-1] - 1) < 0.1

class TestMFI:
    def test_mfi_range(self, sample):
        r = mfi(sample["high"], sample["low"], sample["close"], sample["volume"])
        valid = r.dropna()
        assert valid.between(0, 100).all()

class TestVolDelta:
    def test_vol_delta_basic(self, sample):
        r = vol_delta(sample["close"], sample["open"], sample["volume"])
        assert r.notna().all()
        # delta can be positive or negative
        assert not (r == 0).all()

class TestCMF:
    def test_cmf_range(self, sample):
        r = cmf(sample["high"], sample["low"], sample["close"], sample["volume"])
        valid = r.dropna()
        # CMF ranges from -1 to +1 typically
        assert valid.between(-2, 2).all()


# ═══════════════════════════════════════════════════════════════════════
# Price Structure
# ═══════════════════════════════════════════════════════════════════════

class TestDonchian:
    def test_donchian_order(self, sample):
        u, m, l = donchian(sample["high"], sample["low"])
        valid = u.notna()
        assert (u[valid] >= m[valid]).all()
        assert (m[valid] >= l[valid]).all()

class TestPivot:
    def test_pivot_output(self, sample):
        p, r1, r2, s1, s2 = pivot_points(sample["high"], sample["low"], sample["close"])
        assert p.notna().all()
        assert (r2 >= r1).all()
        assert (s1 >= s2).all()
        assert (r1 >= p).all()
        assert (p >= s1).all()
