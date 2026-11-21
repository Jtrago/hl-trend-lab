"""Grid search harness with IS/OOS split + survivor pass rules (paper research only).

Protocol (no tuning on OOS):
  * split = floor(2/3 * bars): IS = [0, split), OOS = [split, N). Each segment is simulated
    starting flat; indicators are causal so OOS warmup uses only earlier (IS) prices.
  * IS gate (selection uses IS only): IS closed trades >= min_trades(tf), IS PF >= 1.10,
    IS expectancy > 0. Only IS-qualified points are eligible to become survivors.
  * Survivor rules (all on OOS for that asset/tf) — see README "Pass rules".
  * We also record the single IS-best point (max IS Sharpe among IS-qualified) per
    family/asset/tf and its OOS outcome, as an honest one-shot holdout test.
"""
from __future__ import annotations

import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import Costs, run
from .data import load
from .metrics import compute
from .strategies import FAMILIES

ROOT = Path(__file__).resolve().parent.parent
MIN_TRADES = {"1h": 30, "4h": 15}
IS_GATE = {"pf": 1.10}
RULES = {"oos_pf": 1.20, "oos_dd_cap_pct": 35.0, "oos_sharpe": 0.5, "plateau_pf": 1.0, "plateau_min": 2}

# ---- grids (ordered lists => neighbours = +-1 index step in exactly one numeric axis) ----
GRIDS: dict = {
    "ma_cross": {
        "ma_type": ["ema", "sma"],
        "side": ["long", "both"],
        "fast": [5, 8, 10, 13, 16, 20, 25, 30, 40, 50],
        "slow": [20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250],
    },
    "trend_pullback": {
        "trend_type": ["ema", "sma"],
        "side": ["long", "both"],
        "trend_len": [50, 100, 150, 200],
        "rsi_len": [2, 3, 5, 7, 14],
        "rsi_entry": [20, 25, 30, 35, 40, 45],
        "rsi_exit": [50, 55, 60, 65, 70, 75],
    },
    "donchian": {
        "side": ["long", "both"],
        "n_entry": [10, 20, 30, 40, 55, 70, 100, 150, 200],
        "n_exit": [5, 10, 15, 20, 30, 40, 55, 70, 100],
    },
    "donchian_atr": {
        "side": ["long", "both"],
        "n_entry": [10, 20, 30, 40, 55, 70, 100, 150, 200],
        "atr_len": [10, 14, 22, 30, 44],
        "atr_mult": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    },
    "ma_cross_atr": {
        "ma_type": ["ema", "sma"],
        "side": ["long", "both"],
        "fast": [5, 10, 16, 25, 40],
        "slow": [30, 50, 80, 120, 200],
        "atr_len": [14, 22, 44],
        "atr_mult": [2.0, 3.0, 4.0, 5.0],
    },
    "donchian_ratchet": {
        "side": ["long", "both"],
        "n_entry": [10, 20, 30, 40, 55, 70, 100, 150, 200],
        "atr_len": [10, 14, 22, 30, 44],
        "atr_mult": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    },
    "bollinger_breakout": {
        "side": ["long", "both"],
        "bb_len": [10, 20, 30, 50, 80, 100, 150, 200],
        "bb_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "keltner_breakout": {
        "side": ["long", "both"],
        "kc_len": [10, 20, 30, 50, 80, 100, 150, 200],
        "kc_mult": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    },
    "htf_donchian": {
        "side": ["long", "both"],
        "n_entry": [20, 40, 55, 100, 150, 200],
        "n_exit": [10, 20, 40, 55],
        "htf_len": [20, 50, 100, 200],
    },
    "htf_ma_cross_atr": {
        "ma_type": ["ema", "sma"],
        "side": ["long", "both"],
        "fast": [5, 10, 25, 40],
        "slow": [50, 120, 200],
        "atr_len": [22],
        "atr_mult": [3.0, 4.0, 5.0],
        "htf_len": [50, 100, 200],
    },
    "htf_keltner": {
        "side": ["long", "both"],
        "kc_len": [10, 20, 30, 50, 100, 150],
        "kc_mult": [1.5, 2.0, 2.5, 3.0],
        "htf_len": [50, 100, 200],
    },
}
CATEGORICAL = {"ma_type", "trend_type", "side"}
FAMILY_TFS = {"htf_donchian": ["1h"], "htf_ma_cross_atr": ["1h"], "htf_keltner": ["1h"]}


def ensure_family(family):
    """Lazily register derived overlay families (e.g. 'donchian__btcgate') in GRIDS/FAMILIES."""
    if family not in GRIDS and "__btc" in family:
        from .overlays import register_btc_families
        base, mode = family.split("__btc")
        register_btc_families([base], (mode,))     # 1h entries + 4h filter; all others run on 1h and 4h


def valid(family, p):
    family = family.split("__btc")[0]
    if family in ("ma_cross", "ma_cross_atr", "htf_ma_cross_atr"):
        return p["fast"] < p["slow"]
    if family in ("donchian", "htf_donchian"):
        return p["n_exit"] < p["n_entry"]
    if family == "trend_pullback":
        return p["rsi_entry"] < p["rsi_exit"]
    return True


def grid_points(family, grid=None):
    ensure_family(family)
    g = grid or GRIDS[family]
    keys = list(g)
    for vals in itertools.product(*[g[k] for k in keys]):
        p = dict(zip(keys, vals))
        if valid(family, p):
            yield p


def neighbours(family, p, grid=None):
    ensure_family(family)
    g = grid or GRIDS[family]
    for k, vals in g.items():
        if k in CATEGORICAL:
            continue
        i = vals.index(p[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                q = dict(p, **{k: vals[j]})
                if valid(family, q):
                    yield q


def pid(family, sym, tf, p):
    return f"{family}|{sym}|{tf}|" + ",".join(f"{k}={p[k]}" for k in sorted(p))


def evaluate(df, family, p, tf, costs=Costs()):
    sig = FAMILIES[family](df, **p)
    split = len(df) * 2 // 3
    is_m = compute(run(df, sig, costs, end=split), tf)
    oos_m = compute(run(df, sig, costs, start=split), tf)
    return is_m, oos_m, split


def is_gate(is_m, tf):
    return (is_m["closed_trades"] >= MIN_TRADES[tf] and is_m["profit_factor"] >= IS_GATE["pf"]
            and is_m["expectancy_pct"] > 0)


def oos_checks(is_m, oos_m, tf):
    dd_cap = min(RULES["oos_dd_cap_pct"], 2 * is_m["max_dd_pct"])
    return {
        "r1_trades": oos_m["closed_trades"] >= MIN_TRADES[tf],
        "r2_pf_exp": oos_m["profit_factor"] >= RULES["oos_pf"] and oos_m["expectancy_pct"] > 0,
        "r3_dd": oos_m["max_dd_pct"] <= dd_cap,
        "r4_sign": is_m["expectancy_pct"] > 0 and oos_m["expectancy_pct"] > 0,
        "r5_sharpe": oos_m["sharpe"] >= RULES["oos_sharpe"],
    }


_DF_CACHE: dict = {}


def _job(args):
    family, sym, tf, p = args[:4]
    ensure_family(family)
    costs = args[4] if len(args) > 4 else Costs()
    key = (sym, tf)
    if key not in _DF_CACHE:
        _DF_CACHE[key] = load(sym, tf)
    df = _DF_CACHE[key]
    is_m, oos_m, split = evaluate(df, family, p, tf, costs)
    return {"id": pid(family, sym, tf, p), "family": family, "symbol": sym, "tf": tf, "params": p,
            "split_time": str(df.index[split]), "is": is_m, "oos": oos_m}


def run_family(family, symbols, tfs, grid=None, workers=8, log=print, costs=None):
    pts = list(grid_points(family, grid))
    jobs = [(family, s, tf, p) + ((costs,) if costs else ()) for s in symbols for tf in tfs for p in pts]
    log(f"[{family}] {len(pts)} grid points x {len(symbols)*len(tfs)} datasets = {len(jobs)} backtests")
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_job, jobs, chunksize=8))
    by_id = {r["id"]: r for r in rows}
    survivors, best_is = [], {}
    for r in rows:
        tf = r["tf"]
        r["is_gate"] = is_gate(r["is"], tf)
        r["checks"] = oos_checks(r["is"], r["oos"], tf)
        nb = [by_id[pid(family, r["symbol"], tf, q)] for q in neighbours(family, r["params"], grid)]
        good = [n for n in nb if n["oos"]["profit_factor"] >= RULES["plateau_pf"]]
        r["plateau_neighbours_ok"] = len(good)
        r["plateau_neighbours_total"] = len(nb)
        r["checks"]["r6_plateau"] = len(good) >= RULES["plateau_min"]
        r["survivor"] = r["is_gate"] and all(r["checks"].values())
        if r["survivor"]:
            survivors.append(r)
        k = (r["symbol"], tf)
        if r["is_gate"] and (k not in best_is or r["is"]["sharpe"] > best_is[k]["is"]["sharpe"]):
            best_is[k] = r
    return rows, survivors, best_is


def _clean(o):
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return None if math.isnan(o) else (1e9 if o > 0 else -1e9)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def save_rows(rows, name):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    flat = []
    for r in rows:
        d = {"id": r["id"], "family": r["family"], "symbol": r["symbol"], "tf": r["tf"], **r["params"],
             "is_gate": r["is_gate"], "survivor": r["survivor"],
             "plateau_ok": r["plateau_neighbours_ok"], "plateau_total": r["plateau_neighbours_total"]}
        d.update({f"is_{k}": v for k, v in r["is"].items()})
        d.update({f"oos_{k}": v for k, v in r["oos"].items()})
        d.update(r["checks"])
        flat.append(d)
    pd.DataFrame(flat).to_csv(out / f"{name}.csv", index=False)
    return out / f"{name}.csv"


def update_survivors(new, batch_label):
    """Merge survivors into survivors.json (dedupe by id) and regenerate survivors.md."""
    from .pine import export
    jpath, mpath = ROOT / "survivors.json", ROOT / "survivors.md"
    cur = json.loads(jpath.read_text() or "[]") if jpath.exists() else []
    ids = {s["id"] for s in cur}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pine_dir = ROOT / "pine"
    pine_dir.mkdir(exist_ok=True)
    for r in new:
        if r["id"] in ids:
            continue
        fname = r["id"].replace("|", "__").replace(",", "_").replace("=", "-") + ".pine"
        (pine_dir / fname).write_text(export(r["family"], r["params"],
                                             title=f"HLTL {r['family']} {r['symbol']} {r['tf']}"))
        cur.append(_clean({"id": r["id"], "family": r["family"], "symbol": r["symbol"], "tf": r["tf"],
                           "params": r["params"], "batch": batch_label, "found_at_utc": now,
                           "split_time": r["split_time"], "is": r["is"], "oos": r["oos"],
                           "checks": r["checks"], "plateau_neighbours_ok": r["plateau_neighbours_ok"],
                           "pine_file": f"pine/{fname}"}))
    jpath.write_text(json.dumps(cur, indent=2))
    write_md(cur)
    return cur


def write_md(cur):
    lines = ["# Survivors", "",
             "Paper research only. A survivor passed the IS gate and ALL six OOS pass rules (see README).",
             "Costs: fee 0.045%/side + slip 0.02%/side; fill at next bar open. Funding NOT modelled in the survivor "
             "tables below (see the honest-evaluation section for Hyperliquid funding).", ""]
    hs = ROOT / "results" / "honest_section.md"
    if hs.exists():
        lines += [hs.read_text()]
    sec = ROOT / "results" / "robustness_section.md"
    wf = ROOT / "results" / "walkforward_section.md"
    if sec.exists():
        lines += [sec.read_text()]
    if wf.exists():
        lines += [wf.read_text()]
    rs = ROOT / "results" / "research_section.md"
    if rs.exists():
        lines += [rs.read_text()]
    if sec.exists():
        lines += ["## Raw survivors (inflated — every grid point counted)", ""]
    if not cur:
        lines.append("_None yet._")
    else:
        lines.append("| # | family | symbol | tf | params | OOS trades | OOS PF | OOS exp % | OOS Sharpe | OOS maxDD % | OOS CAGR % | IS PF | IS Sharpe | plateau | pine | tier (*=medoid) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        cur = sorted(cur, key=lambda s: (s.get("cluster_id", ""), not s.get("is_cluster_medoid", False)))
        for i, s in enumerate(cur, 1):
            o, i_ = s["oos"], s["is"]
            ps = ", ".join(f"{k}={v}" for k, v in s["params"].items())
            lines.append(f"| {i} | {s['family']} | {s['symbol']} | {s['tf']} | {ps} | {o['closed_trades']} | "
                         f"{o['profit_factor']:.2f} | {o['expectancy_pct']:.3f} | {o['sharpe']:.2f} | "
                         f"{o['max_dd_pct']:.1f} | {o['cagr_pct']:.1f} | {i_['profit_factor']:.2f} | "
                         f"{i_['sharpe']:.2f} | {s['plateau_neighbours_ok']} | `{s['pine_file']}` |"
                         + f" {s.get('cluster_tier', '')}{'*' if s.get('is_cluster_medoid') else ''} |")
    (ROOT / "survivors.md").write_text("\n".join(lines) + "\n")
