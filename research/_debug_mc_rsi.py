"""Debug: verify RSI MC PF correlates with equity after trade-accounting fix."""
import numpy as np
from research.data.loader import load_parquet
from research.optimization.monte_carlo import monte_carlo_rsi

print("Loading data...")
data = load_parquet()
close = data["close"].values.astype(np.float64)
print(f"Data loaded: {len(close)} rows\n")

print("=== RSI Mean Reversion MC (FIXED fee=0.011%) ===")
r = monte_carlo_rsi(close, (7, 14), [25, 30, 35], [60, 65, 70, 75], fee_rate=0.00011)
print(r.summary())
print()

# Show top 10 by Sharpe with key metrics
print("Top 10 by Sharpe:")
print(f"  {'#':>3}  {'Sharpe':>7}  {'PF':>7}  {'DD%':>7}  {'Trades':>7}  {'Final $':>10}  {'WR%':>7}  Params")
print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*30}")
for rank in range(min(10, r.n_combos)):
    ci = int(r.sorted_indices[rank])
    p = r.params_list[ci]
    m = r.metrics[ci]
    params_str = f"period={p['period']}, os={p['oversold']}, ob={p['overbought']}"
    print(f"  {rank+1:>3}  {m[2]:>7.2f}  {m[5]:>7.2f}  {m[1]:>7.2f}%  {int(m[3]):>7,}  ${m[0]:>8,.0f}  {m[4]:>6.1f}%  {params_str}")

# Key check: top combos should now have PF that makes sense relative to equity growth
print("\n--- Sanity Checks ---")
for rank in range(min(5, r.n_combos)):
    ci = int(r.sorted_indices[rank])
    m = r.metrics[ci]
    pf = m[5]
    final_eq = m[0]
    initial = r.initial_capital
    total_return_pct = (final_eq / initial - 1) * 100
    print(f"  Rank {rank+1}: PF={pf:.2f}  Final=${final_eq:>8,.0f}  Return={total_return_pct:+.1f}%  "
          f"PF>1 = {pf > 1.0}  (PF should be >1 if strategy is profitable)")
