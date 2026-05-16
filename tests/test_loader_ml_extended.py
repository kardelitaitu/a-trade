"""Extended tests for research/data/loader.py and research/ml modules."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Data Loader
# ═══════════════════════════════════════════════════════════════════

class TestLoader:

    def test_load_parquet_exists(self):
        from research.data.loader import load_parquet
        df = load_parquet()
        assert df is not None
        assert len(df) > 0
        assert "close" in df.columns

    def test_load_parquet_columns(self):
        from research.data.loader import load_parquet
        df = load_parquet()
        assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)

    def test_load_parquet_datetime_index(self):
        from research.data.loader import load_parquet
        df = load_parquet()
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_ohlcv_columns_constant(self):
        from research.data.loader import OHLCV_COLUMNS
        assert len(OHLCV_COLUMNS) == 5
        assert OHLCV_COLUMNS[3] == "close"

    def test_freq_constant(self):
        from research.data.loader import FREQ
        assert FREQ == "5min"

    def test_load_parquet_range_bounds(self):
        from research.data.loader import load_parquet
        df = load_parquet()
        assert df.index[0] < pd.Timestamp("2018-01-01")
        assert df.index[-1] >= pd.Timestamp("2024-01-01")

    def test_load_parquet_no_duplicate_timestamps(self):
        from research.data.loader import load_parquet
        df = load_parquet()
        assert df.index.is_unique

    def test_fetch_function_exists(self):
        from research.data.loader import fetch_binance_klines
        assert callable(fetch_binance_klines)

    def test_save_parquet_creates_file(self, tmp_path):
        from research.data.loader import save_parquet
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df = pd.DataFrame({
            "open": 100, "high": 105, "low": 95, "close": 102, "volume": 50
        }, index=idx)
        path = save_parquet(df, str(tmp_path / "test.parquet"))
        assert Path(path).exists()

    def test_load_all_json_exists(self):
        """load_all should work with default JSON path."""
        from research.data.loader import load_all
        df = load_all()
        assert len(df) > 1000
        assert "close" in df.columns


# ═══════════════════════════════════════════════════════════════════
# ML Trainer
# ═══════════════════════════════════════════════════════════════════

class TestMLTrainer:

    def test_signals_from_probabilities_basic(self):
        from research.ml.trainer import signals_from_probabilities
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        probs = pd.Series([0.3, 0.4, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5, 0.6, 0.7], index=idx)
        signals = signals_from_probabilities(probs)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_from_prob_continuous(self):
        from research.ml.trainer import signals_from_probabilities
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        probs = pd.Series([0.3, 0.55, 0.7, 0.5, 0.45], index=idx[:5])
        signals = signals_from_probabilities(probs, continuous=True)
        # Continuous should have float values, not just -1, 0, 1
        assert not set(signals.dropna().unique()).issubset({-1, 0, 1})

    def test_signals_prob_edge_cases(self):
        from research.ml.trainer import signals_from_probabilities
        idx = pd.date_range("2024-01-01", periods=5, freq="5min")
        # Test extremes
        probs = pd.Series([0.0, 0.5, 1.0, 0.51, 0.49], index=idx)
        signals = signals_from_probabilities(probs)
        assert signals.iloc[0] == -1  # 0.0 → extreme short
        assert signals.iloc[2] == 1   # 1.0 → extreme long

    def test_signals_neutral_zone(self):
        from research.ml.trainer import signals_from_probabilities
        idx = pd.date_range("2024-01-01", periods=3, freq="5min")
        probs = pd.Series([0.48, 0.50, 0.52], index=idx)
        signals = signals_from_probabilities(probs, threshold=0.55, neutral_zone=0.1)
        # All should be 0 (inside neutral zone 0.45-0.65? no, 0.55 ± 0.05 = 0.50-0.60)
        # Actually neutral_zone=0.1 means 0.45-0.65
        assert (signals == 0).all()

    def test_predict_probabilities_exists(self):
        from research.ml.trainer import predict_probabilities
        assert callable(predict_probabilities)
