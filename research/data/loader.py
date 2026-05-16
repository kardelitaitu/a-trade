"""
Data loader for QuantumEdge.

Reads yearly JSON files from data/btc-yearly/, concatenates them
into a single DataFrame, validates timestamps, and exports as Parquet.
Also supports fetching historical data from Binance API with proxy rotation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "btc-yearly"
PROCESSED_DIR = DATA_DIR / "processed"

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()

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
    """
    if files is None:
        files = _discover_files()

    parts = [_parse_single_file(f) for f in files]
    df = pd.concat(parts)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    n_raw = len(df)

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
            df = df.reindex(expected)
            if fill_gaps:
                gap_mask = _find_gap_mask(df.index, limit=2)
                df.loc[gap_mask, OHLCV_COLUMNS] = np.nan
                df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].ffill(limit=2)

    if remove_outliers:
        for col in OHLCV_COLUMNS:
            returns = df[col].pct_change(fill_method=None)
            extreme = returns.abs() > 50 * returns.std(skipna=True)
            n_extreme = extreme.sum()
            if n_extreme > 0:
                logger.warning(f"Removing {n_extreme} extreme {col} values (>50sigma) as NaN")
                df.loc[extreme, col] = np.nan

    logger.info(f"Total: {len(df)} rows ({n_raw} raw, "
                f"{len(df) - n_raw} inserted from reindex)")
    return df


def _find_gap_mask(index: pd.DatetimeIndex, limit: int = 2) -> pd.Series:
    """Return boolean series where gaps exceed `limit` candles."""
    diff = index.to_series().diff().dt.total_seconds()
    return diff > (limit + 1) * 300


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


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start_str: str = "2017-01-01",
    end_str: Optional[str] = None,
    save: bool = True,
    use_proxies: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical klines from Binance API with optional proxy rotation.

    Uses ProxyManager for round-robin proxy rotation when use_proxies=True.
    Handles pagination internally (max 1000 candles per request).
    """
    import datetime
    import time

    from research.data.proxies import ProxyManager

    # Map interval to milliseconds
    interval_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
        "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
        "12h": 43_200_000, "1d": 86_400_000,
    }.get(interval, 300_000)

    start_dt = pd.Timestamp(start_str)
    end_dt = pd.Timestamp(end_str or datetime.datetime.now())
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    pm = ProxyManager() if use_proxies else None

    all_klines = []
    current_start = start_ms
    max_candles = (end_ms - start_ms) // interval_ms + 1

    logger.info(f"Fetching {symbol} {interval} from {start_dt} to {end_dt} "
                f"(~{max_candles:,} candles)")

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }

        if pm:
            resp = pm.fetch(base_url, params=params)
        else:
            resp = requests.get(base_url, params=params, timeout=30)

        klines = resp.json()

        if not klines or not isinstance(klines, list):
            logger.warning(f"Empty response at {current_start}, stopping")
            break

        all_klines.extend(klines)

        # Advance to next batch
        last_open = klines[-1][0]
        current_start = last_open + interval_ms
        time.sleep(0.1)

    logger.info(f"Fetched {len(all_klines):,} klines")

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    if save:
        safe_name = symbol.lower().replace("usdt", "_usdt")
        path = DATA_DIR / "processed" / f"{safe_name}_{interval}.parquet"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=True)
        logger.info(f"Saved {len(df):,} rows to {path}")

    return df
