# HL Trend Lab

A research backtesting stack for crypto trend-following strategies. It takes public Binance spot
klines, runs them through a no-lookahead Python backtester and an in-sample/out-of-sample
survivor filter, then exports Pine Script v6. The test suite checks for lookahead, recomputes fees
by hand, checks stop fills, and confirms Python↔Pine parity: the exported Pine text is parsed and
run by an independent simulator and has to reproduce the Python trade list exactly.

> **Research/educational use only. Not financial advice.** There is no live trading, no wallets,
> no keys and no authenticated exchange APIs. Backtest results are not a prediction of future
> returns.

## Layout
```
engine/     data.py (fetch/cache/resample) indicators.py (Pine-identical ta.*) strategies.py
            backtest.py (fill engine) metrics.py search.py (grid + pass rules)
            pine.py (Pine v6 exporter) pine_sim.py (Pine-subset executor used for parity tests)
            walkforward.py robustness.py research.py overlays.py honest.py frozen*.py bootstrap.py
scripts/    fetch_data.py fetch_funding.py run_search.py export_pine.py walkforward.py ...
tests/      unit tests + tests/fixtures/ (small public-data sample used by the tests)
exports/    example Pine scripts
data/ results/ cache/ pine/   created locally by the scripts (git-ignored, not published)
```

## Install
```bash
git clone https://github.com/Jtrago/hl-trend-lab.git
cd hl-trend-lab
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"        # or: pip install -r requirements.txt
```

## Run the tests
```bash
pytest -q
```
The tests run offline. `tests/conftest.py` points the engine at `tests/fixtures/`, which holds
BTC/ETH Binance spot 1h/4h klines and Hyperliquid funding from 2026-02 onward (about 1.5 MB of
public market data), plus the frozen SMA200 rule spec.

## Try it on the sample data
```bash
HL_TREND_LAB_DATA=tests/fixtures/data python scripts/run_search.py --smoke ma_cross   # tiny grid, BTC 4h
python scripts/export_pine.py                                                          # writes exports/example_ema_cross_12_26.pine
```
On the short sample, the smoke grid runs 9 backtests and none usually pass the in-sample gate.
That is expected, because the gate needs years of bars. Fetch the full data for real searches.

## Full data
```bash
python scripts/fetch_data.py      # public Binance klines -> data/ (no key)
python scripts/fetch_funding.py   # public Hyperliquid funding history -> data/
python scripts/run_search.py ma_cross trend_pullback
```
Environment overrides: `HL_TREND_LAB_DATA` (kline/funding directory, default `./data`) and
`HL_TREND_LAB_RESULTS` (frozen-spec directory, default `./results`).

## Data
* Source: `https://data-api.binance.vision/api/v3/klines` (Binance **spot**, public market-data
  mirror). Funding comes from Hyperliquid's public info endpoint.
* Bars are never invented. Exchange outages are left as gaps, and the engine treats rows as
  consecutive bars.

## Fill rule (no lookahead)
* Indicators/signals at bar *t* use bars `0..t` only (closed bars). Indicators reproduce Pine's
  `ta.ema/ta.sma/ta.rsi` exactly (SMA-seeded EMA/RMA, `na` before warmup).
* A decision made at the close of bar *t* fills at the **open of bar t+1**. A signal on the final
  bar is never filled.
* Per-bar decision order (identical in Python and Pine):
  1. long and (longExit or shortEntry) → close long; short and (shortExit or longEntry) → close short
  2. then if flat: longEntry and not longExit → open long; elif shortEntry and not shortExit → open short
  (a reversal fills both legs at the same next open).
* One position at a time, 100% of equity, 1x, compounding.

### Intrabar stop rule (ATR families: `donchian_atr`, `ma_cross_atr`)
* Stop level for bar *t* = level computed at the **close of bar t−1** (never bar t's own data).
* Active from the fill bar onward, **including the entry bar** (pessimistic: the entry fills at
  the open, then that same bar's low/high can stop it out).
* Long: stopped if `low[t] <= stop`; raw fill = `min(open[t], stop)` (gap through → worse open).
  Short: stopped if `high[t] >= stop`; raw fill = `max(open[t], stop)`.
* Stop fills pay the same adverse slip (0.02%) + fee (0.045%) as every fill.
* Stops are checked after bar t's open fills and before bar t's close decision; a stopped-out
  position is flat at that close and can re-enter only on a new signal, at the next open.
* No take-profit exists, so there is no stop-vs-target same-bar ambiguity.

### Ratcheting stop (`donchian_ratchet`, engine `stop_mode="ratchet"`)
* Base level = Chandelier level at each close. Level for the next bar = `max(trail, base[t])` if
  long at this close, else `base[t]` (short: `min`, mirrored). So within a trade the stop only
  tightens; a new trade starts from its signal bar's base. Touch/gap/entry-bar rules as above.
* Pine: `var float trailL` updated with `strategy.position_size` at each close (`@@stateful` block),
  then `strategy.exit(stop=trailL)`. `pine_sim` executes that block bar-by-bar for parity.

### Higher-timeframe filter (`htf_donchian`, 1h only)
* 4h bars are built from the 1h bars (UTC buckets 00/04/…/20). A 1h bar in bucket B sees ONLY the
  4h values of bucket B−1 (the previous fully closed 4h bar) — even on the last 1h bar of B, when
  B has technically just closed. Conservative by design; equals the Pine non-repainting idiom
  `request.security(syminfo.tickerid, "240", expr[1], lookahead=barmerge.lookahead_on)`.
* Pine: `strategy.exit("LX","L",stop=longStop)` re-placed every bar (guarded by `not na`), which
  is TradingView's stop-order semantics (fill at stop, or at the open on a gap). Entry-bar stop-outs
  are assumed to behave the same in TradingView's broker emulator; verified only via `pine_sim`.

## Costs (Hyperliquid-realistic defaults)
* **Fee 0.045% per side** = Hyperliquid perps Tier-0 *taker* (HL fee docs, checked 2026-10-30;
  maker is 0.015%). Next-open market fills are taker fills, so taker is the honest default.
* **Slippage 0.02% (2 bp) per side**, adverse (buy at open×1.0002, sell at open×0.9998).
* Round trip ≈ 0.13% of notional. Applied on every entry and exit, including reversals and the
  forced end-of-segment close. Change via `engine.backtest.Costs(fee=..., slip=...)`.
* Pine: `slippage` is in ticks, so the exporter folds fee+slip into `commission_value=0.065` (%).
  Python vs Pine net trade returns agree to O(cost × |trade return|); trade timing is identical.

## IS / OOS split
* Chronological: first ⌊2/3·N⌋ bars = **IS (optimise)**, last 1/3 = **OOS (validate)**.
* Each segment is simulated starting flat; an IS position still open at the split is closed at the
  last IS bar close (`forced`, costs applied). Indicators are causal, so OOS warmup uses only past
  (IS) prices — no OOS information ever reaches IS.
* **Selection uses IS only.** A point is eligible only if it passes the IS gate: IS closed trades ≥
  min trades for the tf, IS PF ≥ 1.10, IS expectancy > 0. OOS is then used only to accept/reject.
  The run summary also reports the single IS-best point (max IS Sharpe) per asset/tf and its OOS
  result, as a one-shot holdout check. Grid sizes are reported so multiple-testing risk is visible.

## Metrics (IS and OOS separately)
trades / closed_trades (excl. forced), win rate, expectancy (mean net % per trade), avg win/loss,
profit factor (Σwins/|Σlosses| of net trade returns), total return, CAGR, Sharpe (mark-to-market
bar returns, annualised by √8760 for 1h and √2190 for 4h, rf=0), max drawdown (mark-to-market, %).

## Pass rules (survivor = IS gate + ALL of these on OOS for that asset/tf)
1. OOS closed trades ≥ **30 (1h)** or ≥ **15 (4h)**.
2. OOS profit factor ≥ **1.20** and OOS expectancy **> 0** after fees+slip.
3. OOS max DD ≤ **min(35%, 2 × IS max DD)** (the stricter of the two).
4. IS expectancy and OOS expectancy both **> 0**.
5. OOS Sharpe (bar-based, annualised for tf) ≥ **0.5**.
6. Plateau: ≥ **2 neighbouring grid points** (±1 step on one numeric parameter, same categorical
   settings) with OOS PF ≥ 1.0.

Survivors are appended (deduped by id) to `survivors.json` / `survivors.md`, each with a Pine v6
file in `pine/`. Every batch also writes `results/<family>_<timestamp>.csv` (all points, all
metrics, per-rule booleans) and `_summary.json`.

### survivors.json schema (one object per survivor)
`id, family, symbol, tf, params{}, batch, found_at_utc, split_time, is{metrics}, oos{metrics},
checks{r1_trades,r2_pf_exp,r3_dd,r4_sign,r5_sharpe,r6_plateau}, plateau_neighbours_ok, pine_file`

## Strategy families
* `ma_cross` — fast/slow EMA or SMA; long on crossover, exit on crossunder; `side=both` adds the
  mirrored short (always-in-market reversal). Grid: fast {5..50}, slow {20..250}, fast<slow.
* `trend_pullback` — trend MA filter (EMA/SMA 50–200) + RSI pullback: long when close>MA and
  RSI<entry; exit when RSI>exit or close<MA. Shorts mirrored (RSI>100−entry, exit RSI<100−exit or
  close>MA). Grid: rsi_len {2,3,5,7,14}, entry {20..45}, exit {50..75}.

* `donchian` — long when close > highest(high, n_entry) of the PRIOR n_entry bars; exit when
  close < lowest(low, n_exit) of the prior n_exit bars; shorts mirrored. Grid: n_entry {10..200},
  n_exit {5..100}, n_exit < n_entry (104 pts).
* `donchian_atr` — same Donchian entry; exit ONLY via Chandelier ATR trailing stop
  `longStop = highest(high, atr_len) − atr_mult·ATR(atr_len)` (short: `lowest(low) + mult·ATR`),
  or a reversal when side=both. Classic Chandelier: position-independent, so it can loosen if the
  rolling high drops out of the window or ATR expands. Grid: n_entry 9 × atr_len {10,14,22,30,44}
  × mult {1.5..5} × side (630 pts).
* `ma_cross_atr` — `ma_cross` entries/exits plus the Chandelier stop (whichever first). 1,152 pts.

* `donchian_ratchet` — `donchian_atr` with the ratcheting stop (630 pts).
* `bollinger_breakout` — long when close > SMA(n) + k·stdev(n) (Pine population stdev); exit when
  close < SMA(n); shorts mirrored. n {10..200} × k {1.0..3.0} × side (80 pts).
* `keltner_breakout` — long when close > EMA(n) + k·ATR(n); exit when close < EMA(n); shorts
  mirrored. n {10..200} × k {1.0..4.0} × side (96 pts).
* `htf_donchian` — 1h Donchian breakout, only long when previous closed 4h close > 4h EMA(len)
  (short when below); also exits when the 4h filter flips. n_entry × n_exit × htf_len {20,50,100,200}
  × side (144 pts, 1h markets only).

## Walk-forward + regime (`scripts/walkforward.py`, tiers A and B by default)
* Anchored expanding folds per market: IS = data start → fold start (first fold after 2 years),
  OOS = next 1 year; a final remainder < 6 months merges into the last fold. BTC/ETH 7 folds,
  XRP 6, SOL 4.
* **reopt** (clean): per fold, pick the max-IS-Sharpe IS-qualified point of the cluster's sub-grid
  using only that fold's IS window; score on the fold OOS. **fixed**: the medoid on every fold —
  stability only, since the medoid was selected on this same history.
* Fold rules = the six survivor rules, trade minimum pro-rated (MIN_TRADES × fold_years / 3).
* Regime: fold is bull if asset buy-and-hold > 0; also strategy vs B&H return/Sharpe/DD, exposure
  and bar-return correlation on the original OOS window (`results/regime_original_oos.csv`).
* Outputs: `results/walkforward_folds.csv`, `walkforward_summary.{csv,json}`, `walkforward.md`
  (also embedded in survivors.md).

## Overlays (`engine/overlays.py`) — batch 4
* **BTC trend-phase filter**: `btcBull` = BTC previous closed UTC daily close > daily SMA200
  (daily bars built from cached BTCUSDT 1h; a bar in day D sees only day D−1 and earlier; False until
  warm). `gate`: no new longs when not bull. `flat`: gate + exit longs (signal exit → next open) when
  not bull. Registered as derived families `<family>__btcgate` / `<family>__btcflat` (same grid).
  Pine: `request.security("BINANCE:BTCUSDT", "D", close[1] / ta.sma(close, 200)[1], lookahead_on)`.
* **Vol-target sizing**: f = min(cap, 0.40 / realised vol), realised vol = annualised stdev of the
  last 30 days of bar log returns at the signal bar, fixed for the trade (`sig["size"]`); undeployed
  equity is idle cash at 0%. Research comparison only (not exported to Pine).
* `htf_ma_cross_atr` / `htf_keltner`: 1h `ma_cross_atr` / `keltner_breakout` gated by the closed-4h
  EMA filter (same rule as `htf_donchian`). Grids 432 / 144 pts, 1h markets only.

## Batch-4 research (`scripts/batch4_research.py`)
* Deflated Sharpe (Bailey & López de Prado 2014) on stitched reopt walk-forward bar returns per unique
  sub-grid; N = all grid points on that market (also all backtests, and N_eff = plateau clusters on
  that market as a correlation-aware sensitivity); PSR vs 0 and Bonferroni p also reported.
  → `results/deflated_sharpe.{csv,json,md}`.
* Phase-filter walk-forward on the exact baseline cluster set → `walkforward_*_btc{gate,flat}.*`,
  `phase_filter_compare.json`. Vol targeting → `voltarget_compare.csv`. Portfolio →
  `portfolio.json`, `portfolio_corr_{IS,OOS}.csv`. Summary in `results/research_section.md` and survivors.md.

## Robustness layer (`scripts/robustness.py`, runs automatically after each full batch)
* **Clusters**: survivors of the same family/market/categorical settings that are grid neighbours
  form one cluster = one *unique* survivor. Medoid = most in-cluster neighbours, tie-break IS Sharpe
  (IS only). Raw survivor counts are inflated; clusters are the honest count.
* **Cross-market**: whether the medoid's exact params also survive on other symbol×tf markets.
* **Best-IS one-shot holdout**: per family × market, the max-IS-Sharpe point and its OOS verdict.
* **Cost stress**: medoid + grid neighbours re-run at 2× fee and 2× slip; all rules re-checked.
* **Tiers**: A = cluster ≥3 + passes at 2× cost + medoid params survive on another market;
  B = ≥3 + passes at 2×; C = one of those; D = neither.
* Outputs: `results/robustness.json`, `robustness_clusters.csv`, `best_is_table.csv`,
  `cross_market_param_sets.json`, `robustness_section.md` (also embedded at top of survivors.md);
  survivors.json rows gain `cluster_id`, `cluster_size`, `cluster_tier`, `is_cluster_medoid`.

## Tests
* `test_lookahead.py` — indicators prefix-invariant; signals and fills before bar k unchanged when
  bars ≥ k are perturbed; every fill is exactly next-bar open; last-bar signal never fills; OOS
  starts flat.
* `test_fees.py` — exact hand-computed long/short round trips; flat market loses 2×(fee+slip);
  zero-cost = gross; costs monotone; reversal charges both legs; PF/win-rate math.
* `test_stops.py` — stop touch fills at stop−slip; gap-through fills at worse open; stop uses only
  the previous close's level; entry-bar stop-out; short touch/gap; NaN stop never fires; fee on stop
  fill exact; re-entry after stop waits for a signal + next open.
* `test_ratchet_htf.py` — ratchet holds a higher level where the plain stop loosens; initial level =
  signal-bar base and live on the entry bar; resets per trade; short mirror; HTF value = previous
  bucket's value; changing the current 4h bucket (incl. its last 1h bar) never changes it; HTF
  family signals prefix-invariant mid-bucket.
* `test_pine_parity.py` — the **exported Pine text itself** is parsed and executed by
  `engine/pine_sim.py` (independent bar-by-bar Pine-reference `ta.*` + Pine broker semantics) and
  must reproduce the Python trade list (direction, entry bar, exit bar, prices) on a synthetic
  fixture and on real BTC 1h/4h slices (shipped in `tests/fixtures/data/`); indicator arrays match Pine reference implementations.
  (TradingView itself is not run here; to spot-check, paste a file from `exports/`.)

## License
MIT, see [LICENSE](LICENSE).
