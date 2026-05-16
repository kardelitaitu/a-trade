"""pytest configuration for QuantumEdge."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Shared fixtures (available to all test files)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample():
    """Standard OHLCV fixture — 200 rows, random walk."""
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


@pytest.fixture
def flat():
    """Completely flat price — no movement at all."""
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    return {
        "close": pd.Series(50000.0, index=idx),
        "high": pd.Series(50100.0, index=idx),
        "low": pd.Series(49900.0, index=idx),
        "volume": pd.Series(50.0, index=idx),
        "open": pd.Series(50000.0, index=idx),
    }


@pytest.fixture
def trending():
    """Steady uptrend — price goes up every period."""
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    close = pd.Series(50000 + np.arange(100) * 10, index=idx)
    return {
        "close": close,
        "high": close + 20,
        "low": close - 20,
        "volume": pd.Series(50.0, index=idx),
        "open": close - 5,
    }


@pytest.fixture
def nan_data():
    """Data with NaN gaps."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    close = pd.Series(50000 + np.cumsum(np.random.randn(100) * 10), index=idx)
    close.iloc[30:35] = np.nan
    return {
        "close": close,
        "high": close + 20,
        "low": close - 20,
        "volume": pd.Series(50.0, index=idx),
        "open": close - 5,
    }
