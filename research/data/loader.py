"""
Data loader for QuantumEdge.

Reads yearly JSON files from data/btc-yearly/, concatenates them
into a single DataFrame, validates timestamps, and exports as Parquet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "btc-yearly"
PROCESSED_DIR = DATA_DIR / "processed"

# Expected columns for a standard OHLCV dataset
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# 5-minute frequency as a pandas offset string
FREQ = "5min"


def _discover_files(pattern: str = "BTCUSDT-5m-*.json") -> list[Path]:
    """Return sorted list of raw JSON files."""
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {RAW_DIR}")
    return files


def _parse_single_file(path: Path) -> pd.DataFrame:
    """Load a single yearly JSON file into a DataFrame with a datetime index."""
    with open(path) as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    # Convert timestamp from milliseconds to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Set as index and sort
    df = df.set_index("timestamp").sort_index()

    # Ensure numeric types
    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Loaded {path.name}: {len(df)} rows, "
                f"{df.index[0]} -> {df.index[-1]}")
    return df


def load_all(
    files: Optional[list[Path]] = None,
    validate_freq: bool = True,
    fill_gaps: bool = True,
    remove_outliers: bool = True,
) -> pd.DataFrame:
    """
    Load all yearly JSON files and return a single clean DataFrame.

    Parameters
    ----------
    files : list[Path], optional
        Subset of files to load. Defaults to all BTCUSDT-5m-*.json files.
    validate_freq : bool
        If True, resample to a regular 5-min grid and flag dropped/missing rows.
    fill_gaps : bool
        If True, forward-fill gaps of up to 2 consecutive candles.
    remove_outliers : bool
        If True, flag extreme price moves (>50σ) as NaN.

    Returns
    -------
    pd.DataFrame
        Clean OHLCV data with datetime index, continuous 5-min frequency.
    """
    if files is None:
        files = _discover_files()

    parts = [_parse_single_file(f) for f in files]
    df = pd.concat(parts)
    df = df.sort_index()
    # Drop any duplicate timestamps (keep last)
    df = df[~df.index.duplicated(keep="last")]

    n_raw = len(df)

    # ----- Frequency validation -----
    if validate_freq:
        expected = pd.date_range(
            start=df.index[0].floor("5min"),
            end=df.index[-1].ceil("5min"),
            freq=FREQ,
        )
        missing = expected.difference(df.index)
        if len(missing) > 0:
            logger.warning(f"Missing {len(missing)} / {len(expected)} "
                           f"timestamps ({100 * len(missing) / len(expected):.2f}%)")
            # Reindex to the full expected grid
            df = df.reindex(expected)
            # Fill gaps up to 2 candles
            if fill_gaps:
                # Mark gaps > 2 candles as NaN instead of filling
                gap_mask = _find_gap_mask(df.index, limit=2)
                df.loc[gap_mask, OHLCV_COLUMNS] = np.nan
                df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].ffill(limit=2)

    # ----- Outlier detection -----
    if remove_outliers:
        for col in OHLCV_COLUMNS:
            returns = df[col].pct_change(fill_method=None)
            extreme = returns.abs() > 50 * returns.std(skipna=True)
            n_extreme = extreme.sum()
            if n_extreme > 0:
                logger.warning(f"Removing {n_extreme} extreme {col} "
                               f"values (>50σ) as NaN")
                df.loc[extreme, col] = np.nan

    logger.info(f"Total: {len(df)} rows ({n_raw} raw, "
                f"{len(df) - n_raw} inserted from reindex)")
    return df


def _find_gap_mask(index: pd.DatetimeIndex, limit: int = 2) -> pd.Series:
    """Return boolean series where gaps exceed `limit` candles."""
    diff = index.to_series().diff().dt.total_seconds()
    return diff > (limit + 1) * 300  # 300 s = 5 min


def save_parquet(df: pd.DataFrame, name: str = "btcusdt_5m.parquet") -> Path:
    """Save DataFrame to parquet under data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_parquet(path, index=True)
    logger.info(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_parquet(name: str = "btcusdt_5m.parquet") -> pd.DataFrame:
    """Load a pre-processed Parquet file."""
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No processed file at {path}. Run load_all() first.")
    return pd.read_parquet(path)
