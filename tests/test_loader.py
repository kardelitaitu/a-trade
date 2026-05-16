"""Tests for research/data/loader.py."""

import pandas as pd
import pytest
from research.data.loader import (
    load_all,
    OHLCV_COLUMNS,
    FREQ,
)


class TestLoader:
    """Basic loader smoke tests."""

    def test_load_all_basic(self):
        """Load all data and check basic shape."""
        df = load_all(validate_freq=False, fill_gaps=False, remove_outliers=False)

        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
        assert len(df) > 0, "Should have data"
        assert isinstance(df.index, pd.DatetimeIndex), "Index should be datetime"

        for col in OHLCV_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

        # Year range: should cover at least 2017-2025
        assert df.index.year.min() <= 2017, "Should start 2017 or earlier"
        assert df.index.year.max() >= 2025, "Should end 2025 or later"

        # No duplicate timestamps
        assert df.index.is_unique, "Duplicate timestamps found"

    def test_load_and_save_parquet(self, tmp_path):
        """Load and save to parquet, then reload."""
        # Load a single year
        df = load_all(validate_freq=False, fill_gaps=False, remove_outliers=False)
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=True)
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == len(df)
        assert list(reloaded.columns) == list(df.columns)

    def test_small_file_count(self):
        """Should have at least 8 yearly files (2017-2025)."""
        from research.data.loader import _discover_files

        files = _discover_files()
        assert len(files) >= 8, f"Expected >=8 files, got {len(files)}"


class TestFrequencyValidation:
    """Frequency validation and gap handling."""

    def test_freq_validation_runs(self):
        """Frequency validation should not crash and should produce expected row count."""
        df = load_all(validate_freq=True, fill_gaps=True)

        # Every row should be on a 5-min boundary
        assert all(df.index.minute % 5 == 0), "Not all timestamps on 5-min grid"

        # Data range (2017-09-01 to 2025-12-31)
        total_mins = (df.index[-1] - df.index[0]).total_seconds() / 60
        expected_rows = int(total_mins // 5) + 1
        # Allow some tolerance for edge alignment
        assert abs(len(df) - expected_rows) <= 10, \
            f"Expected ~{expected_rows} rows, got {len(df)}"

    def test_missing_timestamps_logged(self):
        """Loading should not crash; missing timestamps should be handled."""
        # Just ensure it doesn't crash
        df = load_all(validate_freq=True, fill_gaps=True)
        assert len(df) > 0


class TestDataQuality:
    """Data quality checks."""

    def test_no_excessive_nan(self):
        """After cleaning, NaN should be rare (<0.5% of price cells)."""
        df = load_all(validate_freq=True, fill_gaps=True)

        price_cols = [c for c in OHLCV_COLUMNS if c != "volume"]
        total_cells = len(df) * len(price_cols)
        nan_count = df[price_cols].isna().sum().sum()
        nan_pct = 100 * nan_count / total_cells
        assert nan_pct < 0.5, f"Too many NaN in prices: {nan_pct:.2f}%"

    def test_no_negative_prices(self):
        """All prices should be strictly positive."""
        df = load_all(validate_freq=False, fill_gaps=False, remove_outliers=False)
        price_cols = [c for c in OHLCV_COLUMNS if c != "volume"]
        assert (df[price_cols] > 0).all().all(), "Negative or zero prices found"

    def test_no_negative_volume(self):
        """Volume should be non-negative."""
        df = load_all(validate_freq=False, fill_gaps=False, remove_outliers=False)
        assert (df["volume"] >= 0).all(), "Negative volume found"

    def test_ohlc_consistency(self):
        """High should be >= max(open, close), low <= min(open, close)."""
        df = load_all(validate_freq=False, fill_gaps=False, remove_outliers=False)
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), \
            "High < max(open, close)"
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), \
            "Low > min(open, close)"


class TestDateRange:
    """Check specific historical events are covered."""

    @pytest.mark.parametrize("check_date", [
        "2017-09-01",  # Early data start
        "2020-03-12",  # COVID crash
        "2021-11-10",  # BTC all-time high ~$68k
        "2022-11-09",  # FTX crash
        "2025-12-31",  # End of data
    ])
    def test_key_dates_present(self, check_date):
        """Key historical dates should be in the dataset."""
        df = load_all(validate_freq=False, fill_gaps=False, remove_outliers=False)

        dt = pd.Timestamp(check_date)
        assert dt >= df.index[0] and dt <= df.index[-1], \
            f"{check_date} not in date range ({df.index[0]} to {df.index[-1]})"

        # At least some data on that date
        day_data = df.loc[check_date]
        assert len(day_data) > 0, f"No data for {check_date}"
