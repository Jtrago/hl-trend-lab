"""Walk-forward validation + regime context for robustness clusters (paper research only).

Folds (per symbol × tf): ANCHORED, EXPANDING.  IS = [data start, e_k);  OOS = [e_k, e_k + 1y).
  e_0 = data start + 2 years; step = 1 year; a final remainder < 6 months is merged into the last
  fold. Every fold starts flat; indicators are causal so warmup uses only earlier bars.

Two modes per cluster:
  * reopt (the CLEAN test): within the cluster's sub-grid (same family, market, categorical
    settings, all numeric grid values) pick the point with max IS Sharpe among those passing the IS
    gate, using ONLY that fold's IS window; score it on the fold's OOS. Nothing from the fold OOS
    (or later) is used for the pick.
  * fixed: the cluster medoid's params on every fold. This is a STABILITY check, not a clean
    test: the medoid was itself chosen with the original 2/3-1/3 split (IS gate on early data,
    pass rules on the last third), so every fold overlaps data used to select it.

Fold pass rules = the six survivor rules with the trade minimum pro-rated to fold length
(MIN_TRADES × oos_years / 3, i.e. ≈10 for 1h, ≈5 for 4h for a 1-year fold); plateau = ≥2 grid
neighbours with fold-OOS PF ≥ 1.0. Regime: fold is "bull" if asset buy-and-hold over the fold > 0.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import Costs, run
from .data import load
from .metrics import BARS_PER_YEAR, compute
from .search import CATEGORICAL, GRIDS, MIN_TRADES, RULES, _clean, grid_points, is_gate, neighbours, pid
from .strategies import FAMILIES

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MIN_IS_YEARS, OOS_YEARS = 2.0, 1.0


def folds(n_bars: int, tf: str):
    bpy = BARS_PER_YEAR[tf]
    first, step = int(MIN_IS_YEARS * bpy), int(OOS_YEARS * bpy)
    out, e = [], first
    while e + step <= n_bars:
        out.append([e, e + step])
        e += step
    if out and n_bars - out[-1][1] > 0:
        if n_bars - out[-1][1] < step // 2:
            out[-1][1] = n_bars
        else:
            out.append([out[-1][1], n_bars])
    return [tuple(f) for f in out]


_DF = {}


def _df(sym, tf):
    if (sym, tf) not in _DF:
        _DF[(sym, tf)] = load(sym, tf)
    return _DF[(sym, tf)]


def _point_job(args):
    family, sym, tf, p = args
    from .search import ensure_family
    ensure_family(family)
    df = _df(sym, tf)
    sig = FAMILIES[family](df, **p)
    out = []
    for k, (a, b) in enumerate(folds(len(df), tf)):
        is_r = run(df, sig, Costs(), end=a)
        oos_r = run(df, sig, Costs(), start=a, end=b)
        out.append({"fold": k, "is": compute(is_r, tf), "oos": compute(oos_r, tf),
                    "oos_bar_ret": np.diff(oos_r.equity) / oos_r.equity[:-1]})
    return pid(family, sym, tf, p), out


def fold_checks(is_m, oos_m, tf, oos_years):
    mt = max(3, math.ceil(MIN_TRADES[tf] * oos_years / 3))
    dd_cap = min(RULES["oos_dd_cap_pct"], 2 * is_m["max_dd_pct"])
    return {"r1_trades": oos_m["closed_trades"] >= mt,
            "r2_pf_exp": oos_m["profit_factor"] >= RULES["oos_pf"] and oos_m["expectancy_pct"] > 0,
            "r3_dd": oos_m["max_dd_pct"] <= dd_cap,
            "r4_sign": is_m["expectancy_pct"] > 0 and oos_m["expectancy_pct"] > 0,
            "r5_sharpe": oos_m["sharpe"] >= RULES["oos_sharpe"]}


def subgrid(family, cats_str):
    cats = dict(kv.split("=") for kv in cats_str.split(",")) if cats_str else {}
    return [p for p in grid_points(family) if all(str(p[k]) == v for k, v in cats.items())]


def run_walkforward(tiers=("A",), workers=8, suffix="", tag="", cluster_ids=None):
    """suffix: overlay applied to every cluster's family (e.g. '__btcgate'); the sub-grid and
    medoid are unchanged, only signals change. tag: output file suffix. cluster_ids: explicit list
    (overrides tiers) so overlay runs use exactly the baseline cluster set."""
    from .search import ensure_family
    cl = pd.read_csv(RES / "robustness_clusters.csv")
    if cluster_ids is not None:
        cl = cl[cl.cluster_id.isin(list(cluster_ids))].reset_index(drop=True)
    else:
        cl = cl[cl.tier.isin(tiers)].reset_index(drop=True)
    surv = {s["id"]: s for s in json.loads((ROOT / "survivors.json").read_text())}
    medoid_params, members = {}, {}
    for s in surv.values():
        if s.get("is_cluster_medoid"):
            medoid_params[s["cluster_id"]] = s["params"]
        members.setdefault(s.get("cluster_id"), set()).add(s["id"])
    combos = {}
    for r in cl.itertuples():
        cats = r.cluster_id.split("|")[3]
        combos.setdefault((r.family + suffix, r.symbol, r.tf, cats), []).append(r.cluster_id)
        if suffix:
            ensure_family(r.family + suffix)
            members[r.cluster_id] = {i.replace(r.family + "|", r.family + suffix + "|", 1)
                                     for i in members.get(r.cluster_id, set())}
    jobs = []
    for (fam, sym, tf, cats) in combos:
        for p in subgrid(fam, cats):
            jobs.append((fam, sym, tf, p))
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(_point_job, jobs, chunksize=4))

    rows, summ = [], []
    for (fam, sym, tf, cats), cids in combos.items():
        df = _df(sym, tf)
        fl = folds(len(df), tf)
        pts = subgrid(fam, cats)
        o, c = df["open"].to_numpy(), df["close"].to_numpy()
        split = len(df) * 2 // 3
        for cid in cids:
            med = medoid_params[cid]
            med_id = pid(fam, sym, tf, med)
            for mode in ("reopt", "fixed"):
                fold_rets, bh_rets, passes, pf_ok, in_cluster, bull_flags, bar_rets = [], [], [], [], [], [], []
                for k, (a, b) in enumerate(fl):
                    if mode == "fixed":
                        pick = med_id
                    else:
                        cand = [pid(fam, sym, tf, p) for p in pts
                                if is_gate(res[pid(fam, sym, tf, p)][k]["is"], tf)]
                        if not cand:
                            pick = None
                        else:
                            pick = max(cand, key=lambda q: res[q][k]["is"]["sharpe"])
                    bh = c[b - 1] / o[a] - 1
                    yrs = (b - a) / BARS_PER_YEAR[tf]
                    row = {"cluster_id": cid, "mode": mode, "fold": k, "oos_start": str(df.index[a]),
                           "oos_end": str(df.index[b - 1]), "oos_years": round(yrs, 2),
                           "overlaps_original_oos": b > split, "bh_return_pct": bh * 100, "bull": bh > 0}
                    if pick is None:
                        row.update({"params": None, "pass": False, "note": "no IS-qualified point"})
                        rows.append(row); passes.append(False); fold_rets.append(0.0); bh_rets.append(bh)
                        bull_flags.append(bh > 0); pf_ok.append(False); in_cluster.append(False)
                        continue
                    f = res[pick][k]
                    chk = fold_checks(f["is"], f["oos"], tf, yrs)
                    p = dict(x.split("=") for x in pick.split("|")[3].split(","))
                    pp = {kk: type(GRIDS[fam][kk][0])(v) if kk not in CATEGORICAL else v for kk, v in p.items()}
                    nb = [pid(fam, sym, tf, q) for q in neighbours(fam, pp)]
                    chk["r6_plateau"] = sum(res[q][k]["oos"]["profit_factor"] >= 1.0 for q in nb if q in res) >= 2
                    ok = is_gate(f["is"], tf) and all(chk.values())
                    row.update({"params": pick.split("|")[3], "pick_in_cluster": pick in members.get(cid, set()),
                                "is_sharpe": f["is"]["sharpe"], "oos_trades": f["oos"]["closed_trades"],
                                "oos_pf": f["oos"]["profit_factor"], "oos_exp_pct": f["oos"]["expectancy_pct"],
                                "oos_sharpe": f["oos"]["sharpe"], "oos_dd_pct": f["oos"]["max_dd_pct"],
                                "oos_return_pct": f["oos"]["total_return_pct"], "oos_exposure": f["oos"]["exposure"],
                                "pass": ok, "failed": ",".join(kk for kk, v in chk.items() if not v)})
                    rows.append(row)
                    passes.append(ok); fold_rets.append(f["oos"]["total_return_pct"] / 100); bh_rets.append(bh)
                    bull_flags.append(bh > 0); pf_ok.append(f["oos"]["profit_factor"] >= 1.0)
                    in_cluster.append(pick in members.get(cid, set()))
                    bar_rets.append(f["oos_bar_ret"])
                br = np.concatenate(bar_rets) if bar_rets else np.array([])
                st_sh = float(br.mean() / br.std(ddof=1) * math.sqrt(BARS_PER_YEAR[tf])) if len(br) > 2 and br.std() > 0 else 0.0
                bulls = [i for i, x in enumerate(bull_flags) if x]
                bears = [i for i, x in enumerate(bull_flags) if not x]
                summ.append({
                    "cluster_id": cid, "family": fam, "symbol": sym, "tf": tf, "mode": mode,
                    "medoid": ",".join(f"{k}={v}" for k, v in med.items()),
                    "folds": len(fl), "folds_pass": int(sum(passes)), "pass_frac": sum(passes) / len(fl),
                    "folds_pf_ge_1": int(sum(pf_ok)),
                    "reopt_pick_in_cluster": int(sum(in_cluster)) if mode == "reopt" else None,
                    "stitched_oos_return_pct": (np.prod([1 + r for r in fold_rets]) - 1) * 100,
                    "stitched_bh_return_pct": (np.prod([1 + r for r in bh_rets]) - 1) * 100,
                    "stitched_oos_sharpe": st_sh,
                    "bull_folds": len(bulls), "bull_pass": int(sum(passes[i] for i in bulls)),
                    "bull_mean_ret_pct": float(np.mean([fold_rets[i] for i in bulls]) * 100) if bulls else None,
                    "bull_mean_bh_pct": float(np.mean([bh_rets[i] for i in bulls]) * 100) if bulls else None,
                    "nonbull_folds": len(bears), "nonbull_pass": int(sum(passes[i] for i in bears)),
                    "nonbull_mean_ret_pct": float(np.mean([fold_rets[i] for i in bears]) * 100) if bears else None,
                    "nonbull_mean_bh_pct": float(np.mean([bh_rets[i] for i in bears]) * 100) if bears else None,
                })
    RES.mkdir(exist_ok=True)
    fd = pd.DataFrame(rows)
    sm = pd.DataFrame(summ)
    fd.to_csv(RES / f"walkforward_folds{tag}.csv", index=False)
    sm.to_csv(RES / f"walkforward_summary{tag}.csv", index=False)
    (RES / f"walkforward_summary{tag}.json").write_text(json.dumps(_clean(sm.to_dict("records")), indent=2))
    return fd, sm


def regime_original_oos(tiers=("A",)):
    """Strategy vs buy-and-hold on the ORIGINAL OOS window (last 1/3) for each cluster medoid."""
    cl = pd.read_csv(RES / "robustness_clusters.csv")
    cl = cl[cl.tier.isin(tiers)]
    surv = [s for s in json.loads((ROOT / "survivors.json").read_text()) if s.get("is_cluster_medoid")]
    by = {s["cluster_id"]: s for s in surv}
    rows = []
    for r in cl.itertuples():
        s = by[r.cluster_id]
        df = _df(r.symbol, r.tf)
        split = len(df) * 2 // 3
        sig = FAMILIES[r.family](df, **s["params"])
        res = run(df, sig, Costs(), start=split)
        m = compute(res, r.tf)
        eq = res.equity
        sret = np.diff(eq) / eq[:-1]
        c = df["close"].to_numpy()[split:]
        aret = np.diff(c) / c[:-1]
        corr = float(np.corrcoef(sret, aret)[0, 1]) if sret.std() > 0 else 0.0
        bh_eq = c / df["open"].to_numpy()[split]
        bh_dd = float(((np.maximum.accumulate(bh_eq) - bh_eq) / np.maximum.accumulate(bh_eq)).max() * 100)
        bh_sh = float(aret.mean() / aret.std(ddof=1) * math.sqrt(BARS_PER_YEAR[r.tf]))
        rows.append({"cluster_id": r.cluster_id, "family": r.family, "symbol": r.symbol, "tf": r.tf,
                     "medoid": r.medoid_params, "oos_start": str(df.index[split]),
                     "strat_return_pct": m["total_return_pct"], "bh_return_pct": (bh_eq[-1] - 1) * 100,
                     "strat_minus_bh_pct": m["total_return_pct"] - (bh_eq[-1] - 1) * 100,
                     "strat_sharpe": m["sharpe"], "bh_sharpe": bh_sh,
                     "strat_dd_pct": m["max_dd_pct"], "bh_dd_pct": bh_dd, "exposure": m["exposure"],
                     "corr_to_asset": corr, "beats_bh_return": m["total_return_pct"] > (bh_eq[-1] - 1) * 100,
                     "beats_bh_sharpe": m["sharpe"] > bh_sh})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "regime_original_oos.csv", index=False)
    return out
