#!/usr/bin/env python3
"""Walk-forward + regime context for robustness tiers (default A and B); refreshes survivors.md.

Usage: .venv/bin/python scripts/walkforward.py [--tiers A,B]
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.search import _clean, write_md  # noqa: E402
from engine.walkforward import RES, ROOT, regime_original_oos, run_walkforward  # noqa: E402

tiers = tuple(sys.argv[sys.argv.index("--tiers") + 1].split(",")) if "--tiers" in sys.argv else ("A", "B")
if "--from-csv" in sys.argv:      # re-render markdown from the last run without recomputing
    sm = pd.read_csv(RES / "walkforward_summary.csv")
else:
    fd, sm = run_walkforward(tiers)
cl = pd.read_csv(RES / "robustness_clusters.csv")[["cluster_id", "tier"]]
sm = sm.merge(cl, on="cluster_id")
rg = (pd.read_csv(RES / "regime_original_oos.csv") if "--from-csv" in sys.argv
      else regime_original_oos(tiers)).merge(cl, on="cluster_id")


def f(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


L = ["## Walk-forward & regime (tiers " + "/".join(tiers) + ")", "",
     "Anchored expanding folds: IS = data start → fold start (≥2 y), OOS = next 1 y (last fold takes any "
     "remainder ≥ 6 mo). Fold rules = the six survivor rules, trade minimum pro-rated to fold length (≈10 on 1h, ≈5 on 4h per year). "
     "**reopt** = re-pick the best-IS-Sharpe point of the cluster's sub-grid on each fold's IS only (clean test). "
     "**fixed** = the medoid on every fold (stability only; the medoid was chosen using this same history). "
     "Bull fold = asset buy-and-hold > 0 over the fold. Files: `results/walkforward_{folds,summary}.csv`, "
     "`results/walkforward_summary.json`, `results/regime_original_oos.csv`, `results/walkforward.md`.", ""]
for mode in ("reopt", "fixed"):
    s = sm[sm["mode"] == mode]
    hold = s[s.pass_frac >= 0.5]
    L += [f"**{mode}: {len(hold)} / {len(s)} clusters pass ≥50% of folds** "
          f"(tier A: {len(hold[hold.tier == 'A'])} / {len(s[s.tier == 'A'])}); "
          f"{int((s.stitched_oos_return_pct > s.stitched_bh_return_pct).sum())} / {len(s)} beat stitched buy-and-hold return; "
          f"median stitched OOS Sharpe {s.stitched_oos_sharpe.median():.2f}.", ""]
sm["combo"] = sm.cluster_id.str.rsplit("|", n=1).str[0]
u = sm[sm["mode"] == "reopt"].drop_duplicates("combo")
fx_ = sm[sm["mode"] == "fixed"]
L += [f"Reopt depends only on (family, market, categorical settings), so clusters sharing that sub-grid share one "
      f"reopt result: **unique reopt sub-grids {len(u)}; pass ≥50% of folds: {int((u.pass_frac >= 0.5).sum())}; "
      f"beat stitched B&H: {int((u.stitched_oos_return_pct > u.stitched_bh_return_pct).sum())}.**", "",
      f"Fold pass rate by regime — reopt (unique sub-grids): bull {int(u.bull_pass.sum())}/{int(u.bull_folds.sum())} "
      f"({u.bull_pass.sum() / max(1, u.bull_folds.sum()):.0%}), non-bull {int(u.nonbull_pass.sum())}/{int(u.nonbull_folds.sum())} "
      f"({u.nonbull_pass.sum() / max(1, u.nonbull_folds.sum()):.0%}); fixed: bull {int(fx_.bull_pass.sum())}/{int(fx_.bull_folds.sum())}, "
      f"non-bull {int(fx_.nonbull_pass.sum())}/{int(fx_.nonbull_folds.sum())}. Fixed medoids in non-bull folds: mean fold "
      f"return {fx_.nonbull_mean_ret_pct.mean():.1f}% vs B&H {fx_.nonbull_mean_bh_pct.mean():.1f}%; bull folds "
      f"{fx_.bull_mean_ret_pct.mean():.1f}% vs B&H {fx_.bull_mean_bh_pct.mean():.1f}%. "
      "→ these systems mostly **lose less in down-years and capture part of up-years**; they rarely pass the full rule set "
      "in non-bull folds.", ""]
L += ["| tier | family | mkt | medoid | folds | reopt pass | reopt PF≥1 | reopt pick in cluster | reopt stitched % | fixed pass | fixed stitched % | B&H stitched % | bull folds pass (reopt) | non-bull folds pass (reopt) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
piv = sm.pivot_table(index="cluster_id", columns="mode", values=["folds_pass", "stitched_oos_return_pct"], aggfunc="first")
for r in sm[sm["mode"] == "reopt"].sort_values(["tier", "pass_frac"], ascending=[True, False]).itertuples():
    fx = sm[(sm.cluster_id == r.cluster_id) & (sm["mode"] == "fixed")].iloc[0]
    L.append(f"| {r.tier} | {r.family} | {r.symbol} {r.tf} | {r.medoid} | {r.folds} | {r.folds_pass} | {r.folds_pf_ge_1} | "
             f"{r.reopt_pick_in_cluster} | {f(r.stitched_oos_return_pct, 0)} | {fx.folds_pass} | {f(fx.stitched_oos_return_pct, 0)} | "
             f"{f(r.stitched_bh_return_pct, 0)} | {r.bull_pass}/{r.bull_folds} | {r.nonbull_pass}/{r.nonbull_folds} |")
L += ["", "### Regime: original OOS window (last 1/3) — medoid vs buy-and-hold", "",
      f"Beats B&H total return: **{int(rg.beats_bh_return.sum())} / {len(rg)}**; beats B&H Sharpe: "
      f"**{int(rg.beats_bh_sharpe.sum())} / {len(rg)}**; median exposure {rg.exposure.median():.0%}; "
      f"median correlation of bar returns to the asset {rg.corr_to_asset.median():.2f}. "
      "Most edge here is lower drawdown / partial exposure to the same up-trends, not independent alpha.", "",
      "| tier | family | mkt | strat % | B&H % | strat Sharpe | B&H Sharpe | strat DD% | B&H DD% | exposure | corr |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rg.sort_values(["tier", "strat_sharpe"], ascending=[True, False]).itertuples():
    L.append(f"| {r.tier} | {r.family} | {r.symbol} {r.tf} | {r.strat_return_pct:.0f} | {r.bh_return_pct:.0f} | "
             f"{r.strat_sharpe:.2f} | {r.bh_sharpe:.2f} | {r.strat_dd_pct:.1f} | {r.bh_dd_pct:.1f} | {r.exposure:.0%} | {r.corr_to_asset:.2f} |")
L.append("")
(RES / "walkforward.md").write_text("\n".join(L))
(RES / "walkforward_section.md").write_text("\n".join(L))
write_md(json.loads((ROOT / "survivors.json").read_text()))
print("\n".join(L[:12]))
