#!/usr/bin/env python3
"""Deflated Sharpe, BTC phase-filter walk-forward, vol-target sizing, BTC+ETH portfolio.
Writes results/deflated_*, walkforward_*_btc*, voltarget_compare.csv, portfolio*.{json,csv},
results/research_section.md (embedded in survivors.md).  Usage: .venv/bin/python scripts/batch4_research.py [--skip-wf]
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.research import (RES, ROOT, deflated_table, portfolio, regime_pass_rates,  # noqa: E402
                             vol_target_compare)
from engine.search import _clean, write_md  # noqa: E402
from engine.walkforward import run_walkforward  # noqa: E402

base = pd.read_csv(RES / "walkforward_summary.csv")          # frozen baseline cluster set (tiers A+B)
ids = sorted(base.cluster_id.unique())
tiers = pd.read_csv(RES / "robustness_clusters.csv").set_index("cluster_id")["tier"]
surv = json.loads((ROOT / "survivors.json").read_text())
med = {s["cluster_id"]: s for s in surv if s.get("is_cluster_medoid")}

# 1) deflated Sharpe on the clean (reopt) walk-forward
dsr = deflated_table("") if "--reuse-deflated" not in sys.argv else None
if dsr is None:
    dsr = deflated_table("")
print("deflated done", int(dsr.non_null_dsr_market.sum()), "/", len(dsr), flush=True)

# 2) BTC phase filter overlays, same cluster set
if "--skip-wf" not in sys.argv:
    for mode in ("gate", "flat"):
        run_walkforward(cluster_ids=ids, suffix=f"__btc{mode}", tag=f"_btc{mode}")
        print("wf", mode, "done", flush=True)
rates = {k or "baseline": regime_pass_rates(k) for k in ("", "_btcgate", "_btcflat")}
(RES / "phase_filter_compare.json").write_text(json.dumps(_clean(rates), indent=2))

# 3) vol targeting on tier-A medoids (baseline set)
tierA = [c for c in ids if tiers.get(c) == "A"]
rows = [{"cluster_id": c, "family": med[c]["family"], "symbol": med[c]["symbol"], "tf": med[c]["tf"],
         "params": med[c]["params"]} for c in tierA]
vt = vol_target_compare(rows)

# 4) BTC+ETH portfolio: fixed list = tier-A medoids on BTC/ETH, top-3 DISTINCT families per symbol by IS Sharpe
cand = sorted([med[c] for c in tierA if med[c]["symbol"] in ("BTCUSDT", "ETHUSDT")],
              key=lambda s: -s["is"]["sharpe"])
picked, seen = [], set()
for s in cand:
    k = (s["symbol"], s["family"])
    if k in seen or sum(p["symbol"] == s["symbol"] for p in picked) >= 3:
        continue
    seen.add(k)
    picked.append(s)
systems = [{"label": f"{s['symbol'][:3]} {s['tf']} {s['family']}", "family": s["family"], "symbol": s["symbol"],
            "tf": s["tf"], "params": s["params"], "params_str": ",".join(f"{k}={v}" for k, v in s["params"].items())}
           for s in picked]
pf = portfolio(systems, pd.Timestamp("2023-10-07", tz="UTC"))

# ---- markdown ----
f2 = lambda v: f"{v:.2f}"
L = ["## Batch 4 research: deflation, BTC phase filter, vol targeting, portfolio", ""]
L += ["### Deflated Sharpe on the clean (reopt) walk-forward", "",
      "Method: Bailey & López de Prado (2014) deflated Sharpe ratio on the stitched re-optimised walk-forward "
      "OOS bar returns of each unique sub-grid (tiers A+B). SR0 = expected max Sharpe of N iid trials with the "
      "cross-trial variance of OOS Sharpe actually observed; N_market = every grid point backtested on that "
      "symbol×tf across all families; N_total = every backtest. DSR ≥ 0.95 = 'non-null'. Also PSR vs 0 and a "
      "Bonferroni p (p × N_market). Files: `results/deflated_sharpe.{csv,json,md}`.", "",
      f"**Non-null after deflation (DSR_market ≥ 0.95): {int(dsr.non_null_dsr_market.sum())} / {len(dsr)}; "
      f"Bonferroni p ≤ 0.05: {int(dsr.non_null_bonferroni.sum())} / {len(dsr)}; "
      f"DSR_total ≥ 0.95: {int((dsr.dsr_total >= 0.95).sum())} / {len(dsr)}.** "
      f"Sensitivity (trials are highly correlated): with N_eff = number of plateau clusters on that market, "
      f"{int(dsr.non_null_dsr_n_eff.sum())} / {len(dsr)} clear 0.95; PSR vs 0 alone ≥ 0.95: "
      f"{int((dsr.psr_vs_0 >= 0.95).sum())} / {len(dsr)}. Break-even N = largest trial count at which DSR ≥ 0.95 still holds.", "",
      "| sub-grid | WF folds pass | WF Sharpe | PSR>0 | DSR (N mkt) | DSR (N_eff) | N_eff | break-even N | Bonf. p |",
      "|---|---|---|---|---|---|---|---|---|"]
for r in dsr.sort_values(["dsr_n_eff", "psr_vs_0"], ascending=False).head(25).itertuples():
    L.append(f"| {r.combo} | {r.folds_pass}/{r.folds} | {f2(r.sharpe_ann)} | {f2(r.psr_vs_0)} | {f2(r.dsr_market)} | "
             f"**{f2(r.dsr_n_eff)}** | {r.n_eff_clusters} | {r.breakeven_n_trials} | {r.bonferroni_p_market:.3g} |")
L += ["", "### BTC trend-phase filter (walk-forward, same cluster set)", "",
      "Filter: BTC previous closed UTC daily close vs daily SMA200 (closed bars only). gate = no new longs "
      "when BTC bearish; flat = gate + exit longs when bearish. Files: `results/walkforward_*_btc{gate,flat}.csv`, "
      "`results/phase_filter_compare.json`.", "",
      "| variant | mode | units ≥50% folds | non-bull fold pass | bull fold pass | mean non-bull fold ret % | mean bull fold ret % | median stitched Sharpe |",
      "|---|---|---|---|---|---|---|---|"]
for v, d in rates.items():
    for m in ("reopt", "fixed"):
        x = d[m]
        L.append(f"| {v} | {m} | {x['units_pass_half']}/{x['units']} | {x['nonbull_pass']}/{x['nonbull_folds']} "
                 f"({x['nonbull_rate']:.0%}) | {x['bull_pass']}/{x['bull_folds']} ({x['bull_rate']:.0%}) | "
                 f"{x['nonbull_mean_ret_pct']:.1f} | {x['bull_mean_ret_pct']:.1f} | {x['median_stitched_sharpe']:.2f} |")
agg = vt.groupby("sizing")[["oos_sharpe", "oos_cagr_pct", "oos_max_dd_pct", "is_sharpe", "is_max_dd_pct"]].median()
L += ["", "### Vol-targeted sizing (tier-A medoids, original IS/OOS split)", "",
      "f = min(cap, 40% / realised vol), realised vol = annualised stdev of the last 30 days of bar log returns at "
      "the signal bar; fixed for the trade; idle cash earns 0. Medians across "
      f"{vt.cluster_id.nunique()} systems. Per-system table: `results/voltarget_compare.csv`.", "",
      "| sizing | OOS Sharpe | OOS CAGR % | OOS max DD % | IS Sharpe | IS max DD % |", "|---|---|---|---|---|---|"]
for k, r in agg.iterrows():
    L.append(f"| {k} | {r.oos_sharpe:.2f} | {r.oos_cagr_pct:.1f} | {r.oos_max_dd_pct:.1f} | {r.is_sharpe:.2f} | {r.is_max_dd_pct:.1f} |")
L += ["", "### BTC+ETH paper portfolio (daily rebalanced)", "",
      "Selection: fixed list = the current tier-A medoids on BTC/ETH, top 3 distinct families per symbol ranked by "
      "IS Sharpe (note: tier-A membership itself used the original OOS, so OOS portfolio numbers are optimistic). "
      "Inverse-vol weights estimated on IS only. Split 2023-10-07. Files: `results/portfolio.json`, "
      "`results/portfolio_corr_{IS,OOS}.csv`.", "",
      "Systems: " + "; ".join(f"{s['label']} ({s['params_str']})" for s in systems), "",
      "| period | portfolio | total % | CAGR % | Sharpe | max DD % |", "|---|---|---|---|---|---|"]
for per in ("IS", "OOS"):
    for k in ("equal_weight", "inv_vol", "bh_50_50_btc_eth"):
        x = pf[per][k]
        L.append(f"| {per} | {k} | {x['total_return_pct']:.0f} | {x['cagr_pct']:.1f} | {x['sharpe']:.2f} | {x['max_dd_pct']:.1f} |")
L += ["", f"Average pairwise daily-return correlation: IS {pf['IS']['avg_pairwise_corr']:.2f}, OOS {pf['OOS']['avg_pairwise_corr']:.2f}.", ""]
(RES / "research_section.md").write_text("\n".join(L))
(RES / "deflated_sharpe.md").write_text("\n".join(L[:L.index("### BTC trend-phase filter (walk-forward, same cluster set)")]))
write_md(surv)
print("\n".join(L))
