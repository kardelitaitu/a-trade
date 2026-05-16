"""
Numba-accelerated operations for the backtest engine.
"""

from __future__ import annotations

import numpy as np
from numba import jit, int64, float64


@jit(nopython=True)
def extract_trades_numba(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract trade entry/exit indices from position series.

    Parameters
    ----------
    positions : np.ndarray (float64)
        Position values (-1, 0, +1 or continuous).

    Returns
    -------
    entry_idxs : np.ndarray (int64)
    exit_idxs : np.ndarray (int64)
    sides : np.ndarray (int64)
    """
    n = len(positions)
    # Pre-allocate with max possible trades (n // 2)
    max_trades = n // 2 + 1
    entry_idxs = np.zeros(max_trades, dtype=np.int64)
    exit_idxs = np.zeros(max_trades, dtype=np.int64)
    sides = np.zeros(max_trades, dtype=np.int64)

    trade_count = 0
    entry_idx: int = -1
    entry_side: int = 0
    prev_pos: float = 0.0

    for i in range(n):
        curr_pos = positions[i]

        # Convert continuous float to direction: sign with zero threshold
        if curr_pos > 0:
            curr_dir = 1
        elif curr_pos < 0:
            curr_dir = -1
        else:
            curr_dir = 0

        if prev_pos > 0:
            prev_dir = 1
        elif prev_pos < 0:
            prev_dir = -1
        else:
            prev_dir = 0

        if curr_dir == prev_dir:
            continue

        # Direction changed
        went_to_zero = curr_dir == 0
        came_from_zero = prev_dir == 0

        if came_from_zero and not went_to_zero:
            # Opening a NEW position (0 → +1 or 0 → -1)
            entry_idx = i
            entry_side = curr_dir

        elif went_to_zero and not came_from_zero:
            # Closing an existing position (+1 → 0 or -1 → 0)
            if entry_idx >= 0:
                entry_idxs[trade_count] = entry_idx
                exit_idxs[trade_count] = i
                sides[trade_count] = entry_side
                trade_count += 1
            entry_idx = -1
            entry_side = 0

        else:
            # Position FLIP (+1 → -1 or -1 → +1)
            if entry_idx >= 0:
                entry_idxs[trade_count] = entry_idx
                exit_idxs[trade_count] = i
                sides[trade_count] = entry_side
                trade_count += 1
            # Open new flipped position
            entry_idx = i
            entry_side = curr_dir

        prev_pos = curr_pos

    # Trim arrays to actual trade count
    return (
        entry_idxs[:trade_count],
        exit_idxs[:trade_count],
        sides[:trade_count],
    )
