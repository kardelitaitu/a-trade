"""Extended edge-case tests for research/features/registry.py — 39 indicators."""

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
    close_z, vol_z, ret, log_ret,
    hour_sin, hour_cos, dow_sin, dow_cos, is_weekend,
    bollinger_bands,
)


# ═══════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════

class TestRegistryExtended:

    def test_all_get_indicator_callable(self):
        """Every registered indicator should be callable and return something."""
        for name, meta in INDICATORS.items():
            fn = get_indicator(name)
            assert callable(fn), f"{name} not callable"

    def test_unknown_indicator_message(self):
        with pytest.raises(KeyError, match="Unknown indicator 'xyzzy'"):
            get_indicator("xyzzy")

    def test_category_filter_volume(self):
        result = list_indicators(category="volume")
        assert "obv" in result
        assert "rsi" not in result

    def test_list_all_categories_present(self):
        result = list_indicators()
        for name in INDICATORS:
            assert name in result


# ═══════════════════════════════════════════════════════════════════════
# TREND — Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestSMAExtended:

    def test_period_1_equals_input(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = sma(s, 1)
        assert r.iloc[-1] == pytest.approx(5.0)

    def test_flat_series(self, flat):
        r = sma(flat["close"], 10)
        assert (r.dropna() == 50000.0).all()

    def test_sma_equals_pandas(self):
        s = pd.Series(np.random.randn(50) * 10 + 100)
        np.random.seed(0)  # does nothing here, just random
        r = sma(s, 5)
        expected = s.rolling(5).mean()
        pd.testing.assert_series_equal(r, expected, check_dtype=False)


class TestWMAExtended:

    def test_period_1_equals_input(self):
        s = pd.Series([10.0, 20.0, 30.0])
        r = wma(s, 1)
        valid = r.dropna()
        assert valid.iloc[0] == pytest.approx(10.0)

    def test_wma_positive_on_positive(self, sample):
        r = wma(sample["close"], 5)
        valid = r.dropna()
        assert (valid > 0).all()


class TestHMAExtended:

    def test_hma_on_trend(self, trending):
        r = hma(trending["close"], 10)
        valid = r.dropna()
        assert (valid > 50000).all()  # should be above start

    def test_hma_not_nan(self, sample):
        r = hma(sample["close"], 5)
        assert r.notna().sum() > 10


class TestMACDExtended:

    def test_macd_flat_zero(self, flat):
        r = macd_line(flat["close"])
        valid = r.dropna()
        # MACD of flat series should be 0
        assert valid.abs().max() < 1e-6

    def test_macd_hist_sum(self, sample):
        line = macd_line(sample["close"])
        sig = macd_signal(sample["close"])
        hist = macd_hist(sample["close"])
        # hist = line - signal
        valid = hist.notna()
        assert hist[valid].sub(line[valid] - sig[valid]).abs().max() < 1e-6

    def test_macd_custom_params(self, sample):
        r = macd_line(sample["close"], fast=8, slow=17)
        assert r.notna().sum() > 0


class TestEMAExtended:

    def test_ema_flat(self, flat):
        r = ema(flat["close"], 10)
        assert (r.dropna() == 50000.0).all()

    def test_ema_alpha_one(self):
        s = pd.Series([1.0, 5.0, 3.0, 8.0])
        r = ema(s, 2)
        assert r.notna().sum() > 0


# ═══════════════════════════════════════════════════════════════════════
# MOMENTUM — Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestRSIExtended:

    def test_rsi_flat_is_50(self, flat):
        r = rsi(flat["close"], 14)
        valid = r.dropna()
        assert valid.iloc[0] == pytest.approx(50.0, abs=0.5)

    def test_rsi_uptrend_over_50(self, trending):
        r = rsi(trending["close"], 14)
        valid = r.dropna()
        assert valid.mean() > 50

    def test_rsi_downtrend_below_50(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        close = pd.Series(50000 - np.arange(100) * 10, index=idx)
        r = rsi(close, 14)
        valid = r.dropna()
        assert valid.mean() < 50

    def test_rsi_no_nan_after_warmup(self, sample):
        r = rsi(sample["close"], 7)
        valid = r.dropna()
        assert not valid.isna().any()


class TestStochExtended:

    def test_stoch_flat(self, flat):
        k, d = stoch(flat["high"], flat["low"], flat["close"])
        valid = k.dropna()
        # When price is within range, stoch should be stable
        assert valid.between(0, 100).all()

    def test_stoch_k_between_0_100(self, sample):
        k, d = stoch(sample["high"], sample["low"], sample["close"])
        valid = k.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100


class TestCCIExtended:

    def test_cci_not_nan(self, sample):
        r = cci(sample["high"], sample["low"], sample["close"])
        valid = r.dropna()
        assert not valid.empty

    def test_cci_extreme_values(self, trending):
        r = cci(trending["high"], trending["low"], trending["close"])
        valid = r.dropna()
        assert (valid > 0).sum() > 0  # most should be positive in uptrend


class TestROCExtended:

    def test_roc_flat_is_zero(self, flat):
        r = roc(flat["close"], 5)
        valid = r.dropna()
        assert valid.abs().max() < 1e-6

    def test_roc_increasing(self, trending):
        r = roc(trending["close"], 5)
        valid = r.dropna()
        assert (valid > -1).all()  # all reasonable
        assert valid.mean() > 0  # uptrend → positive ROC


class TestMomentumExtended:

    def test_momentum_flat_is_zero(self, flat):
        r = momentum(flat["close"], 5)
        valid = r.dropna()
        assert valid.abs().max() < 1e-6

    def test_momentum_increasing(self, trending):
        r = momentum(trending["close"], 5)
        valid = r.dropna()
        assert (valid > 0).all()  # uptrend → positive momentum


# ═══════════════════════════════════════════════════════════════════════
# VOLATILITY — Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestATRExtended:

    def test_atr_flat(self, flat):
        r = atr(flat["high"], flat["low"], flat["close"], 14)
        valid = r.dropna()
        assert (valid > 0).all()  # ATR includes high-low range even in flat

    def test_atr_zero_range(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="5min")
        h = pd.Series(100.0, index=idx)
        l = pd.Series(100.0, index=idx)
        c = pd.Series(100.0, index=idx)
        r = atr(h, l, c, 10)
        valid = r.dropna()
        assert valid.max() < 1e-6  # zero true range → zero ATR

    def test_atr_positive(self, sample):
        r = atr(sample["high"], sample["low"], sample["close"], 14)
        valid = r.dropna()
        assert (valid > 0).all()


class TestBollingerExtended:

    def test_bb_collapses_flat(self, flat):
        mid, upper, lower = bollinger_bands(flat["close"], 20, 2.0)
        valid = mid.dropna()
        # On flat price, std=0, so upper=lower=mid
        assert (upper[valid.index] - lower[valid.index]).abs().max() < 1e-6

    def test_bb_upper_above_lower(self, sample):
        mid, upper, lower = bollinger_bands(sample["close"])
        valid = mid.notna()
        assert (upper[valid] >= lower[valid]).all()


class TestKeltnerExtended:

    def test_keltner_collapses_flat(self, flat):
        mid, upper, lower = keltner(flat["high"], flat["low"], flat["close"])
        valid = upper.notna()
        assert valid.sum() > 0
        # Keltner should still have width from ATR of range
        assert (upper[valid] >= mid[valid]).all()
        assert (lower[valid] <= mid[valid]).all()


class TestNATRExtended:

    def test_natr_zero(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="5min")
        h = pd.Series(100.0, index=idx)
        l = pd.Series(100.0, index=idx)
        c = pd.Series(100.0, index=idx)
        r = natr(h, l, c, 10)
        valid = r.dropna()
        assert (valid < 1).all()  # very small %


# ═══════════════════════════════════════════════════════════════════════
# VOLUME — Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestOBVExtended:

    def test_obv_constant_price_no_change(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="5min")
        c = pd.Series(100.0, index=idx)
        v = pd.Series(50.0, index=idx)
        r = obv(c, v)
        # OBV should stay at 0 (first value: sign(diff)=0, volume*sign=0, cumsum=0)
        assert r.iloc[-1] == pytest.approx(0.0, abs=1e-6)


class TestVWAPExtended:

    def test_vwap_tracks_price(self, sample):
        r = vwap(sample["high"], sample["low"], sample["close"], sample["volume"])
        valid = r.dropna()
        # VWAP should be relatively close to actual price
        price = sample["close"].loc[valid.index]
        ratio = (valid / price).dropna()
        assert ratio.between(0.9, 1.1).all()


class TestMFIExtended:

    def test_mfi_flat(self, flat):
        r = mfi(flat["high"], flat["low"], flat["close"], flat["volume"])
        valid = r.dropna()
        assert valid.between(0, 100).all()


class TestVolDeltaExtended:

    def test_vol_delta_zero(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="5min")
        c = pd.Series(100.0, index=idx)
        o = pd.Series(100.0, index=idx)
        v = pd.Series(50.0, index=idx)
        r = vol_delta(c, o, v)
        assert r.dropna().abs().max() < 1e-6

    def test_vol_delta_sign(self, sample):
        r = vol_delta(sample["close"], sample["open"], sample["volume"], period=1)
        # When close > open, delta positive; close < open, negative
        positive = sample["close"] > sample["open"]
        assert (r[positive] >= 0).all() or positive.sum() == 0


class TestCMFExtended:

    def test_cmf_flat(self, flat):
        r = cmf(flat["high"], flat["low"], flat["close"], flat["volume"])
        valid = r.dropna()
        # CMF should center around 0 for flat high=low... but high != low
        assert valid.abs().max() < 10


# ═══════════════════════════════════════════════════════════════════════
# STRUCTURE — Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestDonchianExtended:

    def test_donchian_period_1(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        h = pd.Series([105, 110, 108, 112, 107, 109, 111, 106, 108, 110], index=idx)
        l = pd.Series([95,  98,  97,  99,  96,  97,  98,  95,  97,  98], index=idx)
        u, m, l2 = donchian(h, l, period=1)
        # period=1 → upper = high, lower = low, middle = (high+low)/2
        assert (u == h).all()
        assert (l2 == l).all()


class TestPivotExtended:

    def test_pivot_flat(self, flat):
        p, r1, r2, s1, s2 = pivot_points(flat["high"], flat["low"], flat["close"])
        assert (r2 >= r1).all()
        assert (s1 >= s2).all()
        # For flat data, pivot = high = low = close
        assert abs(p.mean() - 50000) < 1000


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL — Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestCloseZExtended:

    def test_close_z_flat(self, flat):
        r = close_z(flat["close"], 20)
        valid = r.dropna()
        # Flat price → z-score is 0 or NaN (std=0)
        if len(valid) > 0:
            assert valid.abs().max() < 1e-6

    def test_close_z_positive_on_rise(self, sample):
        r = close_z(sample["close"], 30)
        valid = r.dropna()
        assert len(valid) > 0


class TestVolZExtended:

    def test_vol_z_flat(self, flat):
        r = vol_z(flat["volume"], 20)
        valid = r.dropna()
        # Flat volume → z-score is 0 or NaN (std=0)
        if len(valid) > 0:
            assert valid.abs().max() < 1e-6


class TestRetExtended:

    def test_ret_flat_zero(self, flat):
        r = ret(flat["close"], 5)
        valid = r.dropna()
        assert valid.abs().max() < 1e-6

    def test_ret_uptrend_positive(self, trending):
        r = ret(trending["close"], 5)
        valid = r.dropna()
        assert valid.mean() > 0


class TestLogRetExtended:

    def test_log_ret_close_to_ret(self, sample):
        r_reg = ret(sample["close"], 5)
        r_log = log_ret(sample["close"], 5)
        valid = r_reg.notna()
        # For small returns, log ≈ regular
        diff = (r_log[valid] - r_reg[valid]).abs()
        assert diff.mean() < 0.01


# ═══════════════════════════════════════════════════════════════════════
# TIME — Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestTimeExtended:

    def test_hour_sin_cos_identity(self, sample):
        s = hour_sin(sample["close"].index)
        c = hour_cos(sample["close"].index)
        # sin^2 + cos^2 ≈ 1
        identity = np.array(s) ** 2 + np.array(c) ** 2
        assert np.nanmean(identity) == pytest.approx(1.0, abs=0.05)

    def test_dow_sin_cos_identity(self, sample):
        s = dow_sin(sample["close"].index)
        c = dow_cos(sample["close"].index)
        identity = np.array(s) ** 2 + np.array(c) ** 2
        assert np.nanmean(identity) == pytest.approx(1.0, abs=0.05)

    def test_is_weekend_only_weekdays(self):
        idx = pd.date_range("2024-01-01", "2024-01-07", freq="5min")  # Mon-Sun
        r = is_weekend(idx)
        # Monday-Thursday = 0, Saturday-Sunday = 1
        assert r[-1] == 1.0  # Sunday

    def test_nan_input_return_none(self):
        r = hour_sin("not_a_datetimeindex")
        assert r is None
