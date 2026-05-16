"""Tests for research/features/indicators.py."""

import numpy as np
import pandas as pd
import pytest

from research.features.indicators import sma, ema, rsi, atr, bollinger_bands, true_range


@pytest.fixture
def sample_series():
    """Basic price series for testing."""
    return pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0])


class TestSMA:
    def test_sma_basic(self, sample_series):
        result = sma(sample_series, 3)
        assert result.iloc[0] != result.iloc[0]  # NaN
        assert result.iloc[2] == pytest.approx(101.0)  # (100+102+101)/3
        assert result.iloc[4] == pytest.approx(103.0)  # (101+103+105)/3


class TestEMA:
    def test_ema_basic(self, sample_series):
        result = ema(sample_series, 3)
        # EMA starts at period 2 (first value is SMA, then EWMA)
        assert result.notna().sum() > 0
        assert result.iloc[-1] > 0


class TestRSI:
    def test_rsi_range(self, sample_series):
        result = rsi(sample_series, 3)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.between(0, 100).all()

    def test_rsi_constant(self):
        const = pd.Series([100.0] * 20)
        result = rsi(const, 14)
        assert result.dropna().iloc[0] == pytest.approx(50.0, abs=1)


class TestATR:
    def test_atr_basic(self):
        df = pd.DataFrame({
            "high": [102, 104, 103, 106],
            "low": [98, 99, 98, 100],
            "close": [101, 103, 102, 105],
        })
        result = atr(df["high"], df["low"], df["close"], 2)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()


class TestBollinger:
    def test_bollinger_order(self, sample_series):
        mid, upper, lower = bollinger_bands(sample_series, 3)
        valid_mask = mid.notna()
        assert (upper[valid_mask] >= mid[valid_mask]).all()
        assert (lower[valid_mask] <= mid[valid_mask]).all()
