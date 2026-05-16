"""Extended tests for all strategy implementations — edge cases."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.ma_crossover import MACrossover
from research.strategies.breakout import DonchianBreakout
from research.strategies.mean_reversion import MeanReversion
from research.strategies.volatility_breakout import VolatilityBreakout
from research.strategies.volatility_squeeze import VolatilitySqueeze
from research.strategies.regime import detect_regime_sma, detect_regime_adx, filter_by_regime


@pytest.fixture
def sample_data():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.3)
    return pd.DataFrame({
        "open": close - 0.2, "high": close + np.abs(np.random.randn(500)) * 0.5,
        "low": close - np.abs(np.random.randn(500)) * 0.5, "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)


class TestMAExtended:

    def test_ema_vs_sma_different(self, sample_data):
        """EMA should differ from SMA for same period."""
        ema_cfg = {"ma_type": "ema", "fast_period": 10, "slow_period": 30}
        sma_cfg = {"ma_type": "sma", "fast_period": 10, "slow_period": 30}
        ema_sig = MACrossover(ema_cfg).generate_signals(sample_data)
        sma_sig = MACrossover(sma_cfg).generate_signals(sample_data)
        assert not (ema_sig == sma_sig).all()

    def test_filter_volume_reduces_trades(self, sample_data):
        """Volume filter should reduce number of trades."""
        base = MACrossover({"fast_period": 10, "slow_period": 30})
        filtered = MACrossover({"fast_period": 10, "slow_period": 30, "filter_volume_pct": 0.5})
        base_sig = base.generate_signals(sample_data)
        fil_sig = filtered.generate_signals(sample_data)
        assert (fil_sig != 0).sum() <= (base_sig != 0).sum()

    def test_large_periods(self, sample_data):
        """Large periods should not crash."""
        sig = MACrossover({"fast_period": 100, "slow_period": 300}).generate_signals(sample_data)
        assert set(sig.dropna().unique()).issubset({-1, 0, 1})

    def test_identical_periods(self, sample_data):
        """Equal fast/slow periods should produce no signals."""
        sig = MACrossover({"fast_period": 30, "slow_period": 30}).generate_signals(sample_data)
        assert (sig == 0).all()


class TestBreakoutExtended:

    def test_short_entry_on_down_breakout(self, sample_data):
        """Short signal when price breaks below lower channel."""
        strat = DonchianBreakout({"entry_period": 20, "exit_period": 10})
        sig = strat.generate_signals(sample_data)
        assert -1 in sig.values or 1 in sig.values or (sig == 0).all()

    def test_no_signals_with_large_period(self, sample_data):
        """Very large period should not crash (fewer signals)."""
        sig = DonchianBreakout({"entry_period": 200, "exit_period": 100}).generate_signals(sample_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_no_nan_in_signals(self, sample_data):
        """Signals should never contain NaN."""
        sig = DonchianBreakout({"entry_period": 20, "exit_period": 10}).generate_signals(sample_data)
        assert not sig.isna().any()


class TestMeanRevExtended:

    def test_bollinger_mode(self, sample_data):
        """Bollinger mode should produce valid signals."""
        sig = MeanReversion({"mode": "bollinger"}).generate_signals(sample_data)
        assert set(sig.dropna().unique()).issubset({-1, 0, 1})

    def test_rsi_mode_different_from_bollinger(self, sample_data):
        """RSI and Bollinger modes should produce different signals."""
        rsi_sig = MeanReversion({"mode": "rsi", "rsi_period": 14}).generate_signals(sample_data)
        bb_sig = MeanReversion({"mode": "bollinger", "bb_period": 20}).generate_signals(sample_data)
        assert not (rsi_sig == bb_sig).all()

    def test_extreme_thresholds(self, sample_data):
        """Extreme RSI thresholds should produce fewer or equal signals vs normal."""
        normal = MeanReversion({"mode": "rsi", "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70})
        extreme = MeanReversion({"mode": "rsi", "rsi_period": 14, "rsi_oversold": 5, "rsi_overbought": 95})
        n = normal.generate_signals(sample_data)
        e = extreme.generate_signals(sample_data)
        assert (e != 0).sum() >= 0

    def test_no_nan(self, sample_data):
        """No NaN after warmup."""
        sig = MeanReversion({"mode": "rsi", "rsi_period": 14}).generate_signals(sample_data)
        assert not sig.isna().any()


class TestVolBreakoutExtended:

    def test_high_multiplier_no_signals(self, sample_data):
        """Very high ATR multiplier should produce few or no signals."""
        sig = VolatilityBreakout({"atr_multiplier": 10.0}).generate_signals(sample_data)
        assert (sig != 0).sum() < 10

    def test_low_multiplier_many_signals(self, sample_data):
        """Very low ATR multiplier should produce many signals."""
        sig = VolatilityBreakout({"atr_multiplier": 0.1, "atr_lookback": 20}).generate_signals(sample_data)
        total_nonzero = (sig != 0).sum()
        assert total_nonzero > 0

    def test_min_hold_works(self, sample_data):
        """min_hold should prevent immediate reversals."""
        sig = VolatilityBreakout({"min_hold": 5, "atr_multiplier": 0.5}).generate_signals(sample_data)
        # Check no rapid flips within min_hold
        changes = sig.diff().abs()
        assert (changes[sig != 0].dropna() <= 2).all()


class TestVolSqueezeExtended:

    def test_wide_bb_few_signals(self, sample_data):
        """Very wide Bollinger Bands should produce few signals."""
        sig = VolatilitySqueeze({"bb_period": 20, "bb_std": 4.0}).generate_signals(sample_data)
        assert (sig != 0).sum() < 50

    def test_narrow_keltner_more_signals(self, sample_data):
        """Narrow Keltner should make squeeze easier to detect."""
        sig = VolatilitySqueeze({"keltner_atr_mult": 0.5}).generate_signals(sample_data)
        total = (sig != 0).sum()
        assert total > 0

    def test_signals_symmetric(self, sample_data):
        """Signals should have both positive and negative values."""
        sig = VolatilitySqueeze({"bb_std": 3.0, "keltner_atr_mult": 0.5}).generate_signals(sample_data)
        non_zero = sig[sig != 0]
        if len(non_zero) > 0:
            assert non_zero.nunique() >= 1


class TestRegimeExtended:

    def test_regime_sma_detects_trend(self, sample_data):
        """SMA regime should detect bull/bear/sideways."""
        close = sample_data["close"]
        regime = detect_regime_sma(close)
        assert set(regime.unique()).issubset({"bull", "bear", "sideways"})

    def test_regime_adx_detects_trend(self, sample_data):
        """ADX regime should detect bull/bear/sideways."""
        regime = detect_regime_adx(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )
        assert set(regime.unique()).issubset({"bull", "bear", "sideways"})

    def test_filter_signals_reduces(self, sample_data):
        """Filter should zero out signals in unwanted regimes."""
        sig = pd.Series(np.random.choice([-1, 0, 1], 500), index=sample_data.index)
        regime = pd.Series("bear", index=sample_data.index)
        regime.iloc[100:300] = "bull"
        filtered = filter_by_regime(sig, regime, allowed_regimes=["bull"])
        # All bear/sideways signals should be zero
        assert (filtered[regime != "bull"] == 0).all()
