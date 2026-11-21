"""Survivor robustness: plateau clustering, cross-market hold-up, best-IS table, cost stress.

* Clusters: survivors of the same (family, symbol, tf, categorical settings) that are grid
  neighbours (±1 step on one numeric axis) are connected; each connected component is ONE
  unique survivor. Representative (medoid) = member with most in-cluster survivor neighbours,
  tie-break by IS Sharpe (IS only — OOS never used to pick the representative).
* Cross-market: for every exact param set, the markets (symbol×tf) where it survives; and for
  every cluster, which OTHER markets have a survivor with any of the cluster's param sets.
* Best-IS table: per family × market, the single IS-qualified point with max IS Sharpe and its
  OOS result (the honest one-shot holdout).
* Cost stress: each cluster medoid (and its grid neighbours, for the plateau rule) re-run with
  costs × k (default 2× fee and 2× slip); re-check IS gate + all six OOS rules.
"""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .backtest import Costs, DEFAULT_FEE, DEFAULT_SLIP
from .search import (CATEGORICAL, FAMILY_TFS, GRIDS, MIN_TRADES, RULES, _clean, _job, is_gate, neighbours,
                     oos_checks, pid)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _pkey(p):
    return tuple(sorted(p.items()))


def _cats(p):
    return ",".join(f"{k}={p[k]}" for k in sorted(p) if k in CATEGORICAL)


def cluster(survivors):
    groups = defaultdict(list)
    for s in survivors:
        groups[(s["family"], s["symbol"], s["tf"], _cats(s["params"]))].append(s)
    clusters = []
    for (fam, sym, tf, cats), members in groups.items():
        by = {_pkey(m["params"]): m for m in members}
        adj = {k: [_pkey(q) for q in neighbours(fam, dict(k)) if _pkey(q) in by] for k in by}
        seen = set()
        for k in by:
            if k in seen:
                continue
            comp, stack = [], [k]
            seen.add(k)
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            med = max(comp, key=lambda x: (len(adj[x]), by[x]["is"]["sharpe"]))
            ms = [by[x] for x in comp]
            oos_pf = pd.Series([m["oos"]["profit_factor"] for m in ms])
            oos_sh = pd.Series([m["oos"]["sharpe"] for m in ms])
            clusters.append({
                "family": fam, "symbol": sym, "tf": tf, "cats": cats, "size": len(comp),
                "medoid": by[med], "member_keys": comp,
                "oos_pf_median": float(oos_pf.median()), "oos_pf_min": float(oos_pf.min()),
                "oos_sharpe_median": float(oos_sh.median()), "oos_sharpe_min": float(oos_sh.min()),
            })
    for fam in {c["family"] for c in clusters}:
        pass
    clusters.sort(key=lambda c: (c["family"], c["symbol"], c["tf"], -c["size"]))
    counters = defaultdict(int)
    for c in clusters:
        g = (c["family"], c["symbol"], c["tf"])
        counters[g] += 1
        c["cluster_id"] = f"{c['family']}|{c['symbol']}|{c['tf']}|{c['cats']}|c{counters[g]}"
    return clusters


def cross_market(survivors, clusters):
    markets = defaultdict(set)
    for s in survivors:
        markets[(s["family"], _pkey(s["params"]))].add((s["symbol"], s["tf"]))
    multi = []
    for (fam, k), mk in markets.items():
        syms = {m[0] for m in mk}
        both_tf = any(((sy, "1h") in mk and (sy, "4h") in mk) for sy in syms)
        if len(mk) >= 2:
            multi.append({"family": fam, "params": dict(k), "markets": sorted(f"{a} {b}" for a, b in mk),
                          "n_markets": len(mk), "n_symbols": len(syms), "both_tfs_same_symbol": both_tf})
    multi.sort(key=lambda r: (-r["n_markets"], -r["n_symbols"]))
    for c in clusters:
        other = set()
        for k in c["member_keys"]:
            other |= markets[(c["family"], k)]
        other.discard((c["symbol"], c["tf"]))
        c["other_markets_any_member"] = sorted(f"{a} {b}" for a, b in other)
        med = set(markets[(c["family"], _pkey(c["medoid"]["params"]))])
        med.discard((c["symbol"], c["tf"]))
        c["other_markets"] = sorted(f"{a} {b}" for a, b in med)   # medoid's EXACT params elsewhere
    return multi


def latest_results(family):
    files = sorted(f for f in RES.glob(f"{family}_2*.csv"))
    return pd.read_csv(files[-1]) if files else None


def best_is_table(families):
    rows = []
    for fam in families:
        df = latest_results(fam)
        if df is None:
            continue
        pcols = list(GRIDS[fam])
        q = df[df["is_gate"]]
        for (sym, tf), g in df.groupby(["symbol", "tf"]):
            gq = q[(q.symbol == sym) & (q.tf == tf)]
            if gq.empty:
                rows.append({"family": fam, "symbol": sym, "tf": tf, "params": "(no IS-qualified point)",
                             "survivor": False})
                continue
            r = gq.loc[gq["is_sharpe"].idxmax()]
            fails = [k for k in ["r1_trades", "r2_pf_exp", "r3_dd", "r4_sign", "r5_sharpe", "r6_plateau"] if not r[k]]
            rows.append({"family": fam, "symbol": sym, "tf": tf,
                         "params": ", ".join(f"{k}={r[k]}" for k in pcols),
                         "is_sharpe": r.is_sharpe, "is_pf": r.is_profit_factor, "is_trades": r.is_closed_trades,
                         "oos_trades": r.oos_closed_trades, "oos_pf": r.oos_profit_factor,
                         "oos_exp_pct": r.oos_expectancy_pct, "oos_sharpe": r.oos_sharpe,
                         "oos_dd_pct": r.oos_max_dd_pct, "oos_cagr_pct": r.oos_cagr_pct,
                         "survivor": bool(r.survivor), "failed_rules": ",".join(fails)})
    return pd.DataFrame(rows)


def cost_stress(clusters, mult=2.0, workers=8):
    costs = Costs(DEFAULT_FEE * mult, DEFAULT_SLIP * mult)
    jobs, want = [], {}
    for c in clusters:
        m = c["medoid"]
        pts = [m["params"]] + list(neighbours(c["family"], m["params"]))
        for p in pts:
            key = pid(c["family"], m["symbol"], m["tf"], p)
            if key not in want:
                want[key] = len(jobs)
                jobs.append((c["family"], m["symbol"], m["tf"], p, costs))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, jobs, chunksize=4))
    by = {r["id"]: r for r in res}
    for c in clusters:
        m = c["medoid"]
        r = by[m["id"]]
        chk = oos_checks(r["is"], r["oos"], m["tf"])
        nb = [by[pid(c["family"], m["symbol"], m["tf"], q)] for q in neighbours(c["family"], m["params"])]
        chk["r6_plateau"] = sum(n["oos"]["profit_factor"] >= RULES["plateau_pf"] for n in nb) >= RULES["plateau_min"]
        c["stress"] = {"mult": mult, "passes": bool(is_gate(r["is"], m["tf"]) and all(chk.values())),
                       "is_gate": bool(is_gate(r["is"], m["tf"])),
                       "failed": [k for k, v in chk.items() if not v],
                       "oos_pf": r["oos"]["profit_factor"], "oos_sharpe": r["oos"]["sharpe"],
                       "oos_dd_pct": r["oos"]["max_dd_pct"], "oos_exp_pct": r["oos"]["expectancy_pct"]}


def tier(c):
    s = c.get("stress", {}).get("passes", False)
    if c["size"] >= 3 and s and c["other_markets"]:
        return "A"   # plateau + cost-robust + holds on another market
    if c["size"] >= 3 and s:
        return "B"   # plateau + cost-robust
    if s or c["size"] >= 3:
        return "C"
    return "D"       # isolated, cost-fragile


def run_all(stress_mult=2.0):
    jpath = ROOT / "survivors.json"
    surv = json.loads(jpath.read_text())
    from .search import ensure_family
    for f in {s["family"] for s in surv} | {p.name.split("_2026")[0] for p in RES.glob("*__btc*_2*.csv")}:
        ensure_family(f)
    clusters = cluster(surv)
    multi = cross_market(surv, clusters)
    if clusters:
        cost_stress(clusters, stress_mult)
    for c in clusters:
        c["tier"] = tier(c)
    # annotate survivors.json with cluster membership
    cid = {}
    for c in clusters:
        for k in c["member_keys"]:
            cid[(c["family"], c["symbol"], c["tf"], k)] = (c["cluster_id"], c["size"], c["tier"],
                                                          c["medoid"]["id"])
    for s in surv:
        a = cid[(s["family"], s["symbol"], s["tf"], _pkey(s["params"]))]
        s["cluster_id"], s["cluster_size"], s["cluster_tier"] = a[0], a[1], a[2]
        s["is_cluster_medoid"] = s["id"] == a[3]
    jpath.write_text(json.dumps(surv, indent=2))

    fams = sorted({s["family"] for s in surv} | set(f for f in GRIDS if latest_results(f) is not None))
    best = best_is_table(fams)
    RES.mkdir(exist_ok=True)
    best.to_csv(RES / "best_is_table.csv", index=False)
    crows = []
    for c in clusters:
        m = c["medoid"]
        crows.append({"cluster_id": c["cluster_id"], "tier": c["tier"], "family": c["family"],
                      "symbol": c["symbol"], "tf": c["tf"], "size": c["size"],
                      "medoid_params": ", ".join(f"{k}={v}" for k, v in m["params"].items()),
                      "oos_trades": m["oos"]["closed_trades"], "oos_pf": m["oos"]["profit_factor"],
                      "oos_exp_pct": m["oos"]["expectancy_pct"], "oos_sharpe": m["oos"]["sharpe"],
                      "oos_dd_pct": m["oos"]["max_dd_pct"], "oos_cagr_pct": m["oos"]["cagr_pct"],
                      "is_pf": m["is"]["profit_factor"], "is_sharpe": m["is"]["sharpe"],
                      "cluster_oos_pf_median": c["oos_pf_median"], "cluster_oos_pf_min": c["oos_pf_min"],
                      "stress2x_pass": c["stress"]["passes"], "stress2x_failed": ",".join(c["stress"]["failed"]),
                      "stress2x_oos_pf": c["stress"]["oos_pf"], "stress2x_oos_sharpe": c["stress"]["oos_sharpe"],
                      "medoid_also_survives_on": "; ".join(c["other_markets"]),
                      "any_member_also_survives_on": "; ".join(c["other_markets_any_member"])})
    cdf = pd.DataFrame(crows)
    cdf.to_csv(RES / "robustness_clusters.csv", index=False)
    pd.DataFrame(multi).to_json(RES / "cross_market_param_sets.json", orient="records", indent=2)
    summary = {"raw_survivors": len(surv), "unique_clusters": len(clusters),
               "by_family": {}, "tiers": cdf["tier"].value_counts().to_dict() if len(cdf) else {},
               "param_sets_on_2plus_markets": len(multi),
               "param_sets_on_2plus_symbols": sum(r["n_symbols"] >= 2 for r in multi),
               "param_sets_on_both_tfs": sum(r["both_tfs_same_symbol"] for r in multi),
               "stress_mult": stress_mult}
    for fam in fams:
        fs = [s for s in surv if s["family"] == fam]
        fc = [c for c in clusters if c["family"] == fam]
        summary["by_family"][fam] = {
            "raw": len(fs), "clusters": len(fc),
            "clusters_size_ge3": sum(c["size"] >= 3 for c in fc),
            "stress2x_pass": sum(c["stress"]["passes"] for c in fc),
            "tier_A": sum(c["tier"] == "A" for c in fc), "tier_B": sum(c["tier"] == "B" for c in fc),
            "markets_with_survivors": sorted({f"{c['symbol']} {c['tf']}" for c in fc}),
            "zero_survivor_markets": sorted({f"{a} {b}" for a in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
                                             for b in FAMILY_TFS.get(fam, ["1h", "4h"])} - {f"{c['symbol']} {c['tf']}" for c in fc})}
    (RES / "robustness.json").write_text(json.dumps(_clean(summary), indent=2))
    write_section(summary, cdf, best, multi)
    from .search import write_md
    write_md(surv)
    return summary, cdf, best, multi


def _fmt(v, nd=2):
    return "∞" if isinstance(v, float) and v > 1e8 else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_section(summary, cdf, best, multi):
    L = ["## Robustness (read this before the raw table)", "",
         f"**Raw survivor rows: {summary['raw_survivors']} — this count is inflated**: neighbouring grid "
         f"points of one plateau each count separately, and grids are large (multiple-testing). "
         f"**Unique plateau clusters: {summary['unique_clusters']}.** "
         f"Cost stress = {summary['stress_mult']:.0f}× fee and 2× slip (0.09% + 0.04% per side).", "",
         "Tiers: **A** = cluster ≥3 points + medoid passes all rules at 2× cost + the medoid's exact params "
         "also survive on another market; **B** = ≥3 points + passes at 2× cost; **C** = one of the two; **D** = isolated & cost-fragile.", "",
         "| family | raw rows | clusters | clusters ≥3 pts | pass 2× cost | tier A | tier B | zero-survivor markets |",
         "|---|---|---|---|---|---|---|---|"]
    for fam, d in summary["by_family"].items():
        L.append(f"| {fam} | {d['raw']} | {d['clusters']} | {d['clusters_size_ge3']} | {d['stress2x_pass']} | "
                 f"{d['tier_A']} | {d['tier_B']} | {', '.join(d['zero_survivor_markets']) or 'none'} |")
    L += ["", f"Exact param sets surviving on ≥2 markets: {summary['param_sets_on_2plus_markets']} "
          f"(≥2 symbols: {summary['param_sets_on_2plus_symbols']}; both 1h and 4h of one symbol: "
          f"{summary['param_sets_on_both_tfs']}). Full list: `results/cross_market_param_sets.json`.", ""]
    if len(cdf):
        top = cdf[cdf.tier.isin(["A", "B"])].sort_values(["tier", "size", "oos_sharpe"], ascending=[True, False, False])
        L += ["### Tier A/B clusters (medoid shown; medoid picked by IS only)", "",
              "| tier | family | mkt | size | medoid params | OOS n | OOS PF | OOS Sharpe | OOS DD% | OOS CAGR% | 2× PF | 2× Sharpe | medoid params also survive on |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in top.itertuples():
            L.append(f"| {r.tier} | {r.family} | {r.symbol} {r.tf} | {r.size} | {r.medoid_params} | {r.oos_trades} | "
                     f"{_fmt(r.oos_pf)} | {_fmt(r.oos_sharpe)} | {_fmt(r.oos_dd_pct, 1)} | {_fmt(r.oos_cagr_pct, 1)} | "
                     f"{_fmt(r.stress2x_oos_pf)} | {_fmt(r.stress2x_oos_sharpe)} | {r.medoid_also_survives_on or '—'} |")
        if top.empty:
            L.append("| — | none | | | | | | | | | | | |")
        L += ["", "All clusters: `results/robustness_clusters.csv`.", ""]
    if len(best):
        L += ["### Best-IS one-shot holdout (max IS Sharpe per family × market → OOS)", "",
              f"Survives: **{int(best['survivor'].sum())} / {len(best)}**. Table: `results/best_is_table.csv`.", "",
              "| family | mkt | params | IS Sharpe | OOS n | OOS PF | OOS Sharpe | OOS DD% | survivor | failed rules |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in best.itertuples():
            if pd.isna(getattr(r, "is_sharpe", float("nan"))):
                L.append(f"| {r.family} | {r.symbol} {r.tf} | {r.params} | | | | | | no | |")
                continue
            L.append(f"| {r.family} | {r.symbol} {r.tf} | {r.params} | {_fmt(r.is_sharpe)} | {r.oos_trades} | "
                     f"{_fmt(r.oos_pf)} | {_fmt(r.oos_sharpe)} | {_fmt(r.oos_dd_pct, 1)} | "
                     f"{'**yes**' if r.survivor else 'no'} | {r.failed_rules} |")
    L.append("")
    (RES / "robustness_section.md").write_text("\n".join(L))
