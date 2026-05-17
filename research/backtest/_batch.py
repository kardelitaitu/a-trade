"""
Batched backtest kernels for QuantumEdge.

Single numba entry point that processes thousands of parameter
combinations in parallel using prange across all CPU cores.
"""

from __future__ import annotations

import numpy as np
from numba import jit, prange, float64, int32


@jit(nopython=True)
def single_backtest(
    close: np.ndarray,
    signals: np.ndarray,
    capital: float,
    fee_rate: float,
    periods_per_year: int = 105120,
) -> tuple[float, float, float, int, float, float]:
    """Backtest without SL/TP. Delegates to single_backtest_sltp with zeros."""
    return single_backtest_sltp(close, signals, capital, fee_rate,
                                0.0, 0.0, 0.0,
                                np.empty(0), np.empty(0),
                                periods_per_year)


@jit(nopython=True)
def single_backtest_sltp(
    close: np.ndarray,
    signals: np.ndarray,
    capital: float,
    fee_rate: float,
    sl_pct: float,       # stop loss as decimal (0.02 = 2%)
    tp_pct: float,       # take profit as decimal (0.04 = 4%)
    trail_pct: float,    # trailing stop as decimal (0 = off)
    high: np.ndarray,    # high prices, same length as close
    low: np.ndarray,     # low prices, same length as close
    periods_per_year: int = 105120,
) -> tuple[float, float, float, int, float, float]:
    """
    Single backtest with SL/TP/trailing stop support.

    SL/TP logic:
    - Long: SL if low <= entry * (1 - sl_pct), TP if high >= entry * (1 + tp_pct)
    - Short: SL if high >= entry * (1 + sl_pct), TP if low <= entry * (1 - tp_pct)
    - Whichever triggers first wins.
    - Trailing: tracks extreme price since entry, exits on retracement.

    Equity computation uses ACTUAL position state (not raw signals),
    so SL/TP exits properly cap the equity curve.

    Returns: final_equity, max_dd_pct, sharpe, n_trades, win_rate, profit_factor
    """
    n = len(close)
    use_sltp = (sl_pct > 0.0 or tp_pct > 0.0 or trail_pct > 0.0) and len(high) == n and len(low) == n
    if n < 2:
        return capital, 0.0, 0.0, 0, 0.0, 0.0

    equity = capital
    peak = capital
    max_dd = 0.0
    total_ret_sum = 0.0
    total_ret_sq = 0.0
    n_returns = 0

    n_trades = 0
    n_wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    in_position = False
    entry_price = 0.0
    entry_side = 0

    # Trailing stop tracking
    trail_extreme = 0.0   # highest high (long) or lowest low (short) since entry
    trail_stop = 0.0      # current trailing stop level

    last_valid_pos = 0.0
    exit_forced = False
    exit_price = 0.0

    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]):
            continue

        pos = signals[i - 1]
        if np.isnan(pos):
            pos = last_valid_pos

        # ── SL/TP check BEFORE position logic ──
        if in_position and use_sltp:
            hit_sl = False
            hit_tp = False
            hit_trail = False
            sl_price = 0.0
            tp_price = 0.0

            if entry_side > 0:  # LONG
                if sl_pct > 0.0:
                    sl_price = entry_price * (1.0 - sl_pct)
                    hit_sl = low[i] <= sl_price
                if tp_pct > 0.0:
                    tp_price = entry_price * (1.0 + tp_pct)
                    hit_tp = high[i] >= tp_price
                if trail_pct > 0.0:
                    if high[i] > trail_extreme:
                        trail_extreme = high[i]
                        trail_stop = trail_extreme * (1.0 - trail_pct)
                    hit_trail = low[i] <= trail_stop
            else:  # SHORT
                if sl_pct > 0.0:
                    sl_price = entry_price * (1.0 + sl_pct)
                    hit_sl = high[i] >= sl_price
                if tp_pct > 0.0:
                    tp_price = entry_price * (1.0 - tp_pct)
                    hit_tp = low[i] <= tp_price
                if trail_pct > 0.0:
                    if low[i] < trail_extreme:
                        trail_extreme = low[i]
                        trail_stop = trail_extreme * (1.0 + trail_pct)
                    hit_trail = high[i] >= trail_stop

            if hit_sl:
                exit_forced = True
                exit_price = sl_price
            elif hit_tp:
                exit_forced = True
                exit_price = tp_price
            elif hit_trail:
                exit_forced = True
                exit_price = trail_stop

        # ── Record SL/TP trade BEFORE clearing in_position ──
        if exit_forced and in_position:
            ep = exit_price
            pnl = (ep - entry_price) * entry_side
            roundtrip_fee = (entry_price + ep) * fee_rate
            net_pnl = pnl - roundtrip_fee
            n_trades += 1
            if net_pnl > 0:
                n_wins += 1
                gross_profit += net_pnl
            else:
                gross_loss += abs(net_pnl)
            in_position = False

        # ── Equity computation using ACTUAL position state ──
        if exit_forced:
            # SL/TP exit: use exit_price for this bar's return, then go flat
            effective_pos = entry_side
            price_ret = exit_price / close[i - 1] - 1.0
            pos_change = abs(entry_side)  # exit fee (entry fee was paid when position opened)
            was_forced = True
            exit_forced = False
            exit_price = 0.0
        else:
            effective_pos = entry_side if in_position else 0.0
            price_ret = close[i] / close[i - 1] - 1.0
            pos_change = abs(effective_pos - last_valid_pos)
            was_forced = False

        fee_pct = pos_change * fee_rate
        period_return = effective_pos * price_ret - fee_pct
        equity *= (1.0 + period_return)

        total_ret_sum += period_return
        total_ret_sq += period_return * period_return
        n_returns += 1

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

        # ── Signal change handling (not SL/TP) ──
        pos_changed = pos != last_valid_pos
        if pos_changed and not was_forced:
            # Close on signal change
            if in_position:
                ep = close[i]
                pnl = (ep - entry_price) * entry_side
                roundtrip_fee = (entry_price + ep) * fee_rate
                net_pnl = pnl - roundtrip_fee
                n_trades += 1
                if net_pnl > 0:
                    n_wins += 1
                    gross_profit += net_pnl
                else:
                    gross_loss += abs(net_pnl)
                in_position = False

            # Open new position on signal change
            if pos != 0:
                in_position = True
                entry_price = close[i]
                entry_side = pos
                if trail_pct > 0.0:
                    trail_extreme = close[i]
                    trail_stop = close[i] * (1.0 - trail_pct) if pos > 0 else close[i] * (1.0 + trail_pct)

        last_valid_pos = pos

    # ── Final metrics ──
    if n_returns > 1:
        mean_ret = total_ret_sum / n_returns
        variance = (total_ret_sq / n_returns) - mean_ret * mean_ret
        std_ret = max(variance, 0.0) ** 0.5
        sharpe = (mean_ret / std_ret) * (periods_per_year ** 0.5) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    max_dd_pct = max_dd * 100.0
    win_rate = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    return equity, max_dd_pct, sharpe, n_trades, win_rate, profit_factor


# ---------------------------------------------------------------------------
# Batched: MA Crossover with SL/TP
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True)
def batch_ma_crossover_sltp(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sma_bank: np.ndarray,
    fast_idxs: np.ndarray,
    slow_idxs: np.ndarray,
    sl_pcts: np.ndarray,
    tp_pcts: np.ndarray,
    capital: float,
    fee_rate: float,
) -> np.ndarray:
    """
    Batched MA Crossover with per-combo SL/TP.

    sl_pcts, tp_pcts: (n_combos,) float64 — stop loss and take profit percentages.
    """
    n_combos = len(fast_idxs)
    metrics = np.zeros((n_combos, 6), dtype=np.float64)

    for c in prange(n_combos):
        fast = sma_bank[fast_idxs[c]]
        slow = sma_bank[slow_idxs[c]]
        signals = np.where(fast > slow, 1.0, np.where(fast < slow, -1.0, 0.0))
        eq, dd, sh, tr, wr, pf = single_backtest_sltp(
            close, signals, capital, fee_rate,
            sl_pcts[c], tp_pcts[c], 0.0,
            high, low,
        )
        metrics[c, 0] = eq
        metrics[c, 1] = dd
        metrics[c, 2] = sh
        metrics[c, 3] = tr
        metrics[c, 4] = wr
        metrics[c, 5] = pf

    return metrics


# ---------------------------------------------------------------------------
# Batched: MA Crossover
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True)
def batch_ma_crossover(
    close: np.ndarray,
    sma_bank: np.ndarray,
    fast_idxs: np.ndarray,
    slow_idxs: np.ndarray,
    capital: float,
    fee_rate: float,
) -> np.ndarray:
    """
    Batched MA Crossover — all combos in parallel.

    Parameters
    ----------
    close : (n,) float64
    sma_bank : (n_periods, n) float64 — Each row is SMA(period)
    fast_idxs : (n_combos,) int32 — Index into sma_bank for fast MA
    slow_idxs : (n_combos,) int32 — Index into sma_bank for slow MA
    capital, fee_rate : float

    Returns
    -------
    metrics : (n_combos, 6) float64 — final_eq, max_dd, sharpe, trades, win_rate, pf
    """
    n_combos = len(fast_idxs)
    metrics = np.zeros((n_combos, 6), dtype=np.float64)

    for c in prange(n_combos):
        fast = sma_bank[fast_idxs[c]]
        slow = sma_bank[slow_idxs[c]]
        signals = np.where(fast > slow, 1.0, np.where(fast < slow, -1.0, 0.0))
        eq, dd, sh, tr, wr, pf = single_backtest(close, signals, capital, fee_rate)
        metrics[c, 0] = eq
        metrics[c, 1] = dd
        metrics[c, 2] = sh
        metrics[c, 3] = tr
        metrics[c, 4] = wr
        metrics[c, 5] = pf

    return metrics


# ---------------------------------------------------------------------------
# Batched: Mean Reversion (RSI)
# ---------------------------------------------------------------------------

@jit(nopython=True)
def _rsi_bank(close: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Compute RSI for multiple periods. Returns (n_periods, n)."""
    n = len(close)
    m = len(periods)
    bank = np.zeros((m, n))
    for p in range(m):
        period = periods[p]
        # RSI computation
        gain_sum = 0.0
        loss_sum = 0.0
        for i in range(1, min(period + 1, n)):
            diff = close[i] - close[i - 1]
            if diff > 0:
                gain_sum += diff
            else:
                loss_sum -= diff
        if loss_sum > 0:
            rs = gain_sum / loss_sum
            bank[p, period] = 100.0 - 100.0 / (1.0 + rs)
        else:
            bank[p, period] = 50.0 if gain_sum == 0 else 100.0

        for i in range(period + 1, n):
            diff = close[i] - close[i - 1]
            gain = diff if diff > 0 else 0.0
            loss = -diff if diff < 0 else 0.0
            gain_sum = (gain_sum * (period - 1) + gain) / period
            loss_sum = (loss_sum * (period - 1) + loss) / period
            if loss_sum > 0:
                rs = gain_sum / loss_sum
                bank[p, i] = 100.0 - 100.0 / (1.0 + rs)
            else:
                bank[p, i] = 50.0 if gain_sum == 0 else 100.0
    return bank


@jit(nopython=True, parallel=True)
def batch_mean_reversion_rsi(
    close: np.ndarray,
    rsi_bank: np.ndarray,
    period_idxs: np.ndarray,
    oversold_values: np.ndarray,
    overbought_values: np.ndarray,
    capital: float,
    fee_rate: float,
) -> np.ndarray:
    """
    Batched RSI Mean Reversion — all combos in parallel.
    """
    n_combos = len(period_idxs)
    metrics = np.zeros((n_combos, 6), dtype=np.float64)

    for c in prange(n_combos):
        pi = period_idxs[c]
        os = oversold_values[c]
        ob = overbought_values[c]
        rsi_vals = rsi_bank[pi]

        signals = np.zeros(len(close), dtype=np.float64)
        in_long = False
        in_short = False

        for i in range(len(close)):
            r = rsi_vals[i]
            if np.isnan(r):
                continue
            if in_long:
                if r >= 50.0:
                    in_long = False
                else:
                    signals[i] = 1.0
            elif in_short:
                if r <= 50.0:
                    in_short = False
                else:
                    signals[i] = -1.0
            if not (in_long or in_short):
                if r < os:
                    in_long = True
                    signals[i] = 1.0
                elif r > ob:
                    in_short = True
                    signals[i] = -1.0

        eq, dd, sh, tr, wr, pf = single_backtest(close, signals, capital, fee_rate)
        metrics[c, 0] = eq
        metrics[c, 1] = dd
        metrics[c, 2] = sh
        metrics[c, 3] = tr
        metrics[c, 4] = wr
        metrics[c, 5] = pf

    return metrics


# ---------------------------------------------------------------------------
# Pre-compute: SMA bank
# ---------------------------------------------------------------------------

@jit(nopython=True)
def _ffill_numba(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values in-place."""
    last_valid = arr[0]
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = last_valid
        else:
            last_valid = arr[i]
    return arr


@jit(nopython=True)
def compute_sma_bank(close: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Compute SMA for all periods. Returns (n_periods, n)."""
    close = _ffill_numba(close.copy())
    n = len(close)
    m = len(periods)
    bank = np.zeros((m, n))
    for p in range(m):
        period = periods[p]
        cum = 0.0
        for i in range(n):
            cum += close[i]
            if i >= period:
                cum -= close[i - period]
            if i >= period - 1:
                bank[p, i] = cum / period
            else:
                bank[p, i] = np.nan
    return bank
