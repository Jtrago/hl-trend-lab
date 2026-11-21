"""Honest evaluation (batch 5): pre-registered shortlist on a late holdout, portfolio walk-forward,
portfolio vol targeting, Hyperliquid funding. Paper research only.

Pre-registration protocol
  * Cutoff C = 2025-10-30 00:00 UTC. Holdout H = [C, last bar] (~12 months).
  * register(): uses ONLY (a) walk-forward reopt folds whose OOS ended before C and (b) price bars
    before C (data frames are TRUNCATED to < C before signals are computed). It writes
    results/preregistered_list.{md,json} with a UTC timestamp + sha256 of the frozen list.
  * evaluate(): verifies the hash, then scores each frozen setting once on H.
  Contamination (stated, not hidden): H lies inside the ORIGINAL 2/3-1/3 OOS window and inside the
  last walk-forward fold, both of which shaped which families were built and which clusters got
  tier A/B (= the candidate menu). Params and the shortlist ranking do not use H, the menu does.
"""
from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import Costs, run
from .data import load
from .funding import hl_bar_funding, constant_band
from .metrics import BARS_PER_YEAR, compute
from .overlays import vol_target_size
from .research import expected_max_sr, psr, _clean
from .search import is_gate, neighbours, pid, ensure_family
from .strategies import FAMILIES
from .walkforward import fold_checks, subgrid

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CUTOFF = pd.Timestamp("2025-10-30", tz="UTC")
SHORTLIST_N = 8
MAX_PER_FAMILY, MAX_PER_SYMBOL = 2, 3
_DF = {}


def df_(sym, tf):
    if (sym, tf) not in _DF:
        _DF[(sym, tf)] = load(sym, tf)
    return _DF[(sym, tf)]


def pstr(p):
    return ",".join(f"{k}={p[k]}" for k in sorted(p))


def daily(res, init=1.0):
    eq = pd.Series(res.equity, index=res.index).resample("1D").last().ffill()
    return eq.pct_change().fillna(eq.iloc[0] / init - 1)


def dstats(r):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    sd = r.std(ddof=1)
    return {"total_return_pct": float((eq.iloc[-1] - 1) * 100),
            "cagr_pct": float((eq.iloc[-1] ** (1 / yrs) - 1) * 100) if eq.iloc[-1] > 0 else -100.0,
            "sharpe": float(r.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0,
            "max_dd_pct": float(((eq.cummax() - eq) / eq.cummax()).max() * 100), "days": int(len(r)),
            "skew": float(r.skew()), "kurt": float(r.kurt() + 3)}


# ------------------------------------------------------------------ pre-registration
def _pick_job(args):
    fam, sym, tf, p = args
    ensure_family(fam)
    df = df_(sym, tf)
    dfc = df[df.index < CUTOFF]                       # TRUNCATED: nothing at/after C exists here
    m = compute(run(dfc, FAMILIES[fam](dfc, **p), Costs()), tf)
    return pid(fam, sym, tf, p), m


def register(workers=8):
    fd = pd.read_csv(RES / "walkforward_folds.csv")
    fd = fd[fd["mode"] == "reopt"].copy()
    fd["combo"] = fd.cluster_id.str.rsplit("|", n=1).str[0]
    fd = fd[pd.to_datetime(fd.oos_end) < CUTOFF].drop_duplicates(["combo", "fold"])
    g = fd.groupby("combo").agg(pre_folds=("fold", "size"), pre_pass=("pass", "sum"),
                                pre_mean_oos_sharpe=("oos_sharpe", "mean")).reset_index()
    g["pre_pass_frac"] = g.pre_pass / g.pre_folds
    g = g[g.pre_folds >= 3].sort_values(["pre_pass_frac", "pre_mean_oos_sharpe"], ascending=False)
    picked, fam_n, sym_n, seen = [], {}, {}, set()
    for r in g.itertuples():
        fam, sym, tf, cats = r.combo.split("|")
        if (fam, sym) in seen or fam_n.get(fam, 0) >= MAX_PER_FAMILY or sym_n.get(sym, 0) >= MAX_PER_SYMBOL:
            continue
        seen.add((fam, sym)); fam_n[fam] = fam_n.get(fam, 0) + 1; sym_n[sym] = sym_n.get(sym, 0) + 1
        picked.append(r)
        if len(picked) == SHORTLIST_N:
            break
    jobs = [(r.combo.split("|")[0], r.combo.split("|")[1], r.combo.split("|")[2], p)
            for r in picked for p in subgrid(r.combo.split("|")[0], r.combo.split("|")[3])]
    with ProcessPoolExecutor(workers) as ex:
        ism = dict(ex.map(_pick_job, jobs, chunksize=4))
    out = []
    for r in picked:
        fam, sym, tf, cats = r.combo.split("|")
        cand = [(pid(fam, sym, tf, p), p) for p in subgrid(fam, cats) if is_gate(ism[pid(fam, sym, tf, p)], tf)]
        best_id, best_p = max(cand, key=lambda x: ism[x[0]]["sharpe"])
        m = ism[best_id]
        out.append({"rank": len(out) + 1, "family": fam, "symbol": sym, "tf": tf, "subgrid": cats,
                    "params": best_p, "pre_cutoff_wf_folds": int(r.pre_folds), "pre_cutoff_wf_pass": int(r.pre_pass),
                    "pre_cutoff_wf_mean_oos_sharpe": float(r.pre_mean_oos_sharpe),
                    "is_window": f"data start .. {CUTOFF.date()} (exclusive)", "is_sharpe": m["sharpe"],
                    "is_pf": m["profit_factor"], "is_trades": m["closed_trades"], "is_max_dd_pct": m["max_dd_pct"],
                    "is_cagr_pct": m["cagr_pct"], "n_gated_candidates": len(cand)})
    body = json.dumps(_clean(out), indent=2, sort_keys=True)
    sha = hashlib.sha256(body.encode()).hexdigest()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc = {"registered_at_utc": ts, "cutoff_utc": str(CUTOFF), "sha256_of_shortlist": sha,
           "rules": {"menu": "66 reopt sub-grids from walkforward_summary.csv (tier A/B clusters, baseline, no overlays)",
                     "eligibility": ">=3 walk-forward folds whose OOS ended before cutoff",
                     "rank": "pre-cutoff fold pass fraction desc, then mean pre-cutoff fold OOS Sharpe desc",
                     "diversity": f"<=1 per (family,symbol), <={MAX_PER_FAMILY} per family, <={MAX_PER_SYMBOL} per symbol; take {SHORTLIST_N}",
                     "params": "within the sub-grid: max IS Sharpe among IS-gated points, IS = all bars < cutoff (frames truncated)"},
           "shortlist": json.loads(body)}
    (RES / "preregistered_list.json").write_text(json.dumps(doc, indent=2))
    L = ["# Pre-registered shortlist (frozen BEFORE holdout evaluation)", "",
         f"- Registered at: **{ts} UTC** (holdout evaluation runs strictly after this; see preregistered_eval.md)",
         f"- Cutoff C = **{CUTOFF.date()} 00:00 UTC**; holdout H = [C, last bar ≈ 2026-10-30] (~1 year), never used for any pick below.",
         f"- sha256(shortlist JSON) = `{sha}`", "",
         "## Selection rules (fixed in code: engine/honest.py::register)", ""]
    L += [f"- **{k}**: {v}" for k, v in doc["rules"].items()]
    L += ["", "## Contamination risk (honest label)", "",
          "- H is inside the original 2/3–1/3 OOS window and the last walk-forward fold. Those windows shaped which "
          "families were built and which clusters became tier A/B — i.e. the *candidate menu* has seen H. "
          "The **ranking** (pre-C folds only) and the **params** (bars < C only, frames truncated) have not.",
          "- So a good H result is weak-to-moderate evidence, a bad one is strong evidence against.", "",
          "## Shortlist", "",
          "| # | family | symbol | tf | params (frozen) | pre-C WF pass | pre-C mean fold OOS Sharpe | IS Sharpe | IS PF | IS trades | IS maxDD% |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in out:
        L.append(f"| {s['rank']} | {s['family']} | {s['symbol']} | {s['tf']} | `{pstr(s['params'])}` | "
                 f"{s['pre_cutoff_wf_pass']}/{s['pre_cutoff_wf_folds']} | {s['pre_cutoff_wf_mean_oos_sharpe']:.2f} | "
                 f"{s['is_sharpe']:.2f} | {s['is_pf']:.2f} | {s['is_trades']} | {s['is_max_dd_pct']:.1f} |")
    L += ["", "Evaluation plan (fixed now): run each frozen setting once on H with standard costs; report trades, PF, "
          "expectancy, Sharpe, CAGR, maxDD, pass rules (trade minimum pro-rated to H length, plateau = ≥2 grid "
          "neighbours PF≥1 on H), vs buy-and-hold, PSR vs 0 and DSR with N = 8 (daily returns); then repeat with "
          "Hyperliquid funding. An equal-weight daily-rebalanced basket of all 8 is also reported."]
    (RES / "preregistered_list.md").write_text("\n".join(L) + "\n")
    return doc


# ------------------------------------------------------------------ holdout evaluation (run once)
def _bh_daily(df, a):
    c = df["close"][a:].resample("1D").last().ffill()
    r = c.pct_change()
    r.iloc[0] = c.iloc[0] / df["open"].iloc[a] - 1
    return r


def evaluate():
    doc = json.loads((RES / "preregistered_list.json").read_text())
    body = json.dumps(doc["shortlist"], indent=2, sort_keys=True)
    assert hashlib.sha256(body.encode()).hexdigest() == doc["sha256_of_shortlist"], "shortlist changed after registration"
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, D, DF, BH = [], {}, {}, {}
    for s in doc["shortlist"]:
        fam, sym, tf, p = s["family"], s["symbol"], s["tf"], s["params"]
        ensure_family(fam)
        df = df_(sym, tf)
        a = int(df.index.searchsorted(CUTOFF))
        yrs = (len(df) - a) / BARS_PER_YEAR[tf]
        sig = FAMILIES[fam](df, **p)
        is_m = compute(run(df, sig, Costs(), end=a), tf)
        res = run(df, sig, Costs(), start=a)
        m = compute(res, tf)
        fr = hl_bar_funding(df, sym)
        cov = float(np.isfinite(fr[a:]).mean())
        resf = run(df, dict(sig, funding=np.nan_to_num(fr)), Costs(), start=a)
        mf = compute(resf, tf)
        chk = fold_checks(is_m, m, tf, yrs)
        nb = list(neighbours(fam, p))
        nb_pf = [compute(run(df, FAMILIES[fam](df, **q), Costs(), start=a), tf)["profit_factor"] for q in nb]
        chk["r6_plateau"] = sum(x >= 1.0 for x in nb_pf) >= 2
        chkf = fold_checks(is_m, mf, tf, yrs); chkf["r6_plateau"] = chk["r6_plateau"]
        lab = f"{s['rank']}:{fam}|{sym}|{tf}"
        D[lab], DF[lab], BH[lab] = daily(res), daily(resf), _bh_daily(df, a)
        ds, dsf, bh = dstats(D[lab]), dstats(DF[lab]), dstats(BH[lab])
        rows.append({"rank": s["rank"], "label": lab, "family": fam, "symbol": sym, "tf": tf, "params": pstr(p),
                     "holdout_start": str(df.index[a]), "holdout_end": str(df.index[-1]), "holdout_years": round(yrs, 3),
                     "trades": m["closed_trades"], "win_rate": m["win_rate"], "pf": m["profit_factor"],
                     "expectancy_pct": m["expectancy_pct"], "bar_sharpe": m["sharpe"], "cagr_pct": m["cagr_pct"],
                     "return_pct": m["total_return_pct"], "max_dd_pct": m["max_dd_pct"], "exposure": m["exposure"],
                     "daily_sharpe": ds["sharpe"], "bh_return_pct": bh["total_return_pct"], "bh_sharpe": bh["sharpe"],
                     "bh_max_dd_pct": bh["max_dd_pct"], "beats_bh_return": ds["total_return_pct"] > bh["total_return_pct"],
                     "beats_bh_sharpe": ds["sharpe"] > bh["sharpe"],
                     "neighbours": len(nb), "neighbours_pf_ge_1": int(sum(x >= 1 for x in nb_pf)),
                     "pass_all_rules": all(chk.values()) and is_gate(is_m, tf),
                     "failed_rules": ",".join(k for k, v in chk.items() if not v),
                     "funding_coverage": cov, "funding_paid_pct_of_equity": float(sum(t.funding_paid for t in resf.trades) * 100),
                     "f_return_pct": mf["total_return_pct"], "f_pf": mf["profit_factor"], "f_bar_sharpe": mf["sharpe"],
                     "f_max_dd_pct": mf["max_dd_pct"], "f_daily_sharpe": dsf["sharpe"],
                     "f_pass_all_rules": all(chkf.values()) and is_gate(is_m, tf),
                     "skew": ds["skew"], "kurt": ds["kurt"], "days": ds["days"]})
    out = pd.DataFrame(rows)
    # deflation across the N = 8 pre-registered trials (daily units)
    N = len(out)
    for pre in ("", "f_"):
        srd = out[f"{pre}daily_sharpe"] / math.sqrt(365)
        var = float(srd.var(ddof=1))
        sr0 = expected_max_sr(N, var)
        out[f"{pre}psr_vs_0"] = [psr(x, 0.0, T, sk, ku) for x, T, sk, ku in zip(srd, out["days"], out["skew"], out["kurt"])]
        out[f"{pre}dsr_N{N}"] = [psr(x, sr0, T, sk, ku) for x, T, sk, ku in zip(srd, out["days"], out["skew"], out["kurt"])]
        out[f"{pre}sr0_ann"] = sr0 * math.sqrt(365)
    basket = pd.concat(D, axis=1).dropna().mean(axis=1)
    basketf = pd.concat(DF, axis=1).dropna().mean(axis=1)
    bhb = pd.concat(BH, axis=1).dropna().mean(axis=1)
    bs, bsf, bb = dstats(basket), dstats(basketf), dstats(bhb)
    for d in (bs, bsf, bb):
        d["psr_vs_0"] = psr(d["sharpe"] / math.sqrt(365), 0.0, d["days"], d["skew"], d["kurt"])
    summary = {"evaluated_at_utc": ts, "registered_at_utc": doc["registered_at_utc"],
               "sha256_verified": doc["sha256_of_shortlist"], "n": N,
               "pass_all_rules": int(out.pass_all_rules.sum()), "pass_all_rules_with_funding": int(out.f_pass_all_rules.sum()),
               "positive_return": int((out.return_pct > 0).sum()), "pf_ge_1": int((out.pf >= 1).sum()),
               "beats_bh_return": int(out.beats_bh_return.sum()), "beats_bh_sharpe": int(out.beats_bh_sharpe.sum()),
               "psr_ge_0.95": int((out.psr_vs_0 >= 0.95).sum()), f"dsr_N{N}_ge_0.95": int((out[f"dsr_N{N}"] >= 0.95).sum()),
               "basket_ew": bs, "basket_ew_with_funding": bsf, "basket_bh_same_assets": bb}
    out.to_csv(RES / "preregistered_eval.csv", index=False)
    (RES / "preregistered_eval.json").write_text(json.dumps(_clean({"summary": summary, "rows": out.to_dict("records")}), indent=2))
    return out, summary


def write_eval_md():
    d = json.loads((RES / "preregistered_eval.json").read_text())
    s, rows = d["summary"], d["rows"]
    L = ["# Pre-registered shortlist — holdout evaluation (run ONCE)", "",
         f"- Shortlist registered {s['registered_at_utc']} UTC; evaluated {s['evaluated_at_utc']} UTC; sha256 verified `{s['sha256_verified'][:16]}…`.",
         f"- Holdout H = {rows[0]['holdout_start'][:10]} → {rows[0]['holdout_end'][:16]} UTC (~1y). Costs: 0.045% fee + 2 bp slip per side. "
         "Funding columns (f_*) add actual Hyperliquid hourly funding (coverage 100% of H).",
         "- Contamination: H sat inside the original OOS / last WF fold that shaped the candidate *menu* (see preregistered_list.md). "
         "Params and ranking did not use H.",
         "- Context: H was a down year for every asset in the list (B&H −29% BTC, −43% ETH, −50% SOL).", "",
         "## Per setting", "",
         "| # | system | params | trades | PF | exp% | ret% | maxDD% | daily Sharpe | B&H ret% | B&H Sharpe | nbrs PF≥1 | pass rules | failed | ret% w/ funding | funding paid (% eq) | PSR>0 | DSR N=8 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['rank']} | {r['family']} {r['symbol']} {r['tf']} | `{r['params']}` | {r['trades']} | {r['pf']:.2f} | "
                 f"{r['expectancy_pct']:.2f} | {r['return_pct']:.1f} | {r['max_dd_pct']:.1f} | {r['daily_sharpe']:.2f} | "
                 f"{r['bh_return_pct']:.1f} | {r['bh_sharpe']:.2f} | {r['neighbours_pf_ge_1']}/{r['neighbours']} | "
                 f"{'✅' if r['pass_all_rules'] else '❌'} | {r['failed_rules'] or '—'} | {r['f_return_pct']:.1f} | "
                 f"{r['funding_paid_pct_of_equity']:.2f} | {r['psr_vs_0']:.2f} | {r['dsr_N8']:.2f} |")
    b, bf, bh = s["basket_ew"], s["basket_ew_with_funding"], s["basket_bh_same_assets"]
    L += ["", f"DSR benchmark: expected max annualised Sharpe of 8 trials given their cross-sectional dispersion = {rows[0]['sr0_ann']:.2f}.", "",
          "## Scorecard", "",
          f"- Pass all six rules on H: **{s['pass_all_rules']}/8** (with funding {s['pass_all_rules_with_funding']}/8).",
          f"- Positive return: {s['positive_return']}/8; PF ≥ 1: {s['pf_ge_1']}/8.",
          f"- Beat own-asset B&H on return: {s['beats_bh_return']}/8, on Sharpe: {s['beats_bh_sharpe']}/8 — *only because B&H fell hard; being flat beats a −30…−50% year.*",
          f"- PSR > 0 at 95%: {s['psr_ge_0.95']}/8; DSR(N=8) ≥ 0.95: {s['dsr_N8_ge_0.95']}/8.",
          f"- Equal-weight basket of all 8: return {b['total_return_pct']:.1f}%, Sharpe {b['sharpe']:.2f}, maxDD {b['max_dd_pct']:.1f}% "
          f"(with funding {bf['total_return_pct']:.1f}% / {bf['sharpe']:.2f}); same-asset B&H basket {bh['total_return_pct']:.1f}% / Sharpe {bh['sharpe']:.2f} / DD {bh['max_dd_pct']:.1f}%.", "",
          "## Verdict", "",
          "The pre-registered settings did **not** show a positive edge on the held-out year: 6/8 lost money, 1/8 "
          "(htf_donchian SOL 1h) passed the rules, and nothing is significant even before deflation. What they did do is the "
          "expected trend-follower behaviour in a bear year — mostly flat, losing ~6% as a basket while B&H lost ~39%. "
          "That is drawdown avoidance, not demonstrated alpha. One year with ~7–32 trades per system is also low power: "
          "this result cannot confirm an edge, and it is consistent with the DSR 0/66 finding."]
    (RES / "preregistered_eval.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ portfolio walk-forward
PF_ASSETS = ["BTCUSDT", "ETHUSDT"]
PF_MENU = [("donchian", "4h", "side=long"), ("donchian_atr", "4h", "side=long"),
           ("ma_cross_atr", "4h", "ma_type=sma,side=long"), ("keltner_breakout", "4h", "side=long"),
           ("bollinger_breakout", "4h", "side=long"), ("htf_donchian", "1h", "side=long")]
PF_K = 3                       # systems per asset per fold (primary); K=2 is a reported sensitivity
PF_FOLD_YEARS = list(range(2020, 2027))   # OOS = calendar year; 2026 runs to the last bar
VT = dict(target_vol=0.40, vol_days=30, cap=1.0)   # same formula as batch 4


def pf_folds(df):
    out = []
    for y in PF_FOLD_YEARS:
        a = int(df.index.searchsorted(pd.Timestamp(f"{y}-01-01", tz="UTC")))
        b = int(df.index.searchsorted(pd.Timestamp(f"{y + 1}-01-01", tz="UTC")))
        out.append((y, a, b))
    return out


def _pf_is_job(args):
    fam, sym, tf, p = args[:4]
    pick_funding = len(args) > 4 and args[4]
    df = df_(sym, tf)
    sig = FAMILIES[fam](df, **p)
    if pick_funding:                       # IS Sharpe net of funding (HL where exists, flat band before)
        from .funding import blended
        sig = dict(sig, funding=blended(df, sym, tf))
    return pid(fam, sym, tf, p), [compute(run(df, sig, Costs(), end=a), tf) for (_, a, _) in pf_folds(df)]


def portfolio_walkforward(workers=8, pick_funding=False):
    jobs = [(fam, sym, tf, p, pick_funding) for sym in PF_ASSETS for (fam, tf, cats) in PF_MENU for p in subgrid(fam, cats)]
    with ProcessPoolExecutor(workers) as ex:
        ism = dict(ex.map(_pf_is_job, jobs, chunksize=4))
    from .funding import blended
    picks = []
    for k, y in enumerate(PF_FOLD_YEARS):
        for sym in PF_ASSETS:
            items = []
            for fam, tf, cats in PF_MENU:
                cand = [(pid(fam, sym, tf, p), p) for p in subgrid(fam, cats) if is_gate(ism[pid(fam, sym, tf, p)][k], tf)]
                if cand:
                    i, p = max(cand, key=lambda x: ism[x[0]][k]["sharpe"])
                    items.append({"fold": y, "symbol": sym, "family": fam, "tf": tf, "params": p,
                                  "is_sharpe": ism[i][k]["sharpe"], "is_pf": ism[i][k]["profit_factor"]})
            items.sort(key=lambda d: -d["is_sharpe"])
            for r, it in enumerate(items):
                it["is_rank"] = r + 1
                picks.append(it)
    # OOS + trailing-IS daily returns for every pick that could be used (rank <= 3)
    fund = {s: {tf: np.nan_to_num(hl_bar_funding(df_(s, tf), s)) for tf in ("1h", "4h")} for s in PF_ASSETS}
    for it in picks:
        if it["is_rank"] > PF_K:
            continue
        df = df_(it["symbol"], it["tf"])
        y, a, b = [f for f in pf_folds(df) if f[0] == it["fold"]][0]
        sig = FAMILIES[it["family"]](df, **it["params"])
        size = vol_target_size(df, it["tf"], **VT)
        it["is_daily"] = daily(run(df, sig, Costs(), end=a)).iloc[-365:]
        bl = blended(df, it["symbol"], it["tf"])
        it["oos"] = {"plain": daily(run(df, sig, Costs(), start=a, end=b)),
                     "vt": daily(run(df, dict(sig, size=size), Costs(), start=a, end=b)),
                     "plain_hlfund": daily(run(df, dict(sig, funding=fund[it["symbol"]][it["tf"]]), Costs(), start=a, end=b)),
                     "vt_hlfund": daily(run(df, dict(sig, size=size, funding=fund[it["symbol"]][it["tf"]]), Costs(), start=a, end=b)),
                     "plain_blend": daily(run(df, dict(sig, funding=bl), Costs(), start=a, end=b)),
                     "vt_blend": daily(run(df, dict(sig, size=size, funding=bl), Costs(), start=a, end=b))}
        for band in (0.05, 0.10, 0.20):
            it["oos"][f"plain_band{int(band * 100)}"] = daily(run(df, dict(sig, funding=constant_band(df, it["tf"], band)), Costs(), start=a, end=b))
            it["oos"][f"vt_band{int(band * 100)}"] = daily(run(df, dict(sig, size=size, funding=constant_band(df, it["tf"], band)), Costs(), start=a, end=b))
    return picks


def chain(picks, K, weight, variant):
    """weight: 'ew' or 'iv' (inverse of trailing-365d IS daily vol of each picked system)."""
    parts = []
    for y in PF_FOLD_YEARS:
        sel = [p for p in picks if p["fold"] == y and p["is_rank"] <= K]
        if not sel:
            continue
        R = pd.concat({f"{p['symbol']}|{p['family']}": p["oos"][variant] for p in sel}, axis=1).fillna(0.0)
        if weight == "ew":
            w = pd.Series(1 / len(sel), index=R.columns)
        else:
            iv = pd.Series({f"{p['symbol']}|{p['family']}": 1 / max(p["is_daily"].std(), 1e-6) for p in sel})
            w = iv / iv.sum()
        parts.append(R @ w[R.columns])
    return pd.concat(parts)


def bh_chain():
    r = {}
    for s in PF_ASSETS:
        c = df_(s, "1h")["close"].resample("1D").last()
        r[s] = c.pct_change()
    bh = pd.concat(r, axis=1).mean(axis=1)
    return bh[(bh.index >= f"{PF_FOLD_YEARS[0]}-01-01")]


def portfolio_report(picks):
    variants = {}
    for K in (3, 2):
        for w in ("ew", "iv"):
            for sz in ("plain", "vt"):
                variants[f"K{K}_{w}_{sz}"] = chain(picks, K, w, sz)
    bh = bh_chain()
    idx = variants["K3_ew_plain"].index
    bh = bh.reindex(idx).fillna(0.0)
    st = {k: dstats(v) for k, v in variants.items()}
    st["bh_50_50"] = dstats(bh)
    srs = {k: st[k]["sharpe"] / math.sqrt(365) for k in variants}
    var8 = float(np.var(list(srs.values()), ddof=1))
    var4 = float(np.var([srs[k] for k in variants if k.startswith("K3")], ddof=1))
    for k in list(variants) + ["bh_50_50"]:
        s = st[k]
        sr = s["sharpe"] / math.sqrt(365)
        s["psr_vs_0"] = s["dsr_N1"] = psr(sr, 0.0, s["days"], s["skew"], s["kurt"])
        s["dsr_N4"] = psr(sr, expected_max_sr(4, var4), s["days"], s["skew"], s["kurt"])
        s["dsr_N8"] = psr(sr, expected_max_sr(8, var8), s["days"], s["skew"], s["kurt"])
        s["psr_vs_bh_sharpe"] = psr(sr, st["bh_50_50"]["sharpe"] / math.sqrt(365), s["days"], s["skew"], s["kurt"])
    # funding
    fund = {}
    seg = pd.Timestamp("2023-05-12", tz="UTC")
    for sz in ("plain", "vt"):
        for fv in ("", "_hlfund", "_band5", "_band10", "_band20"):
            r = chain(picks, 3, "ew", sz + fv)
            fund[f"K3_ew_{sz}{fv}"] = {"full_chain": dstats(r), "hl_covered_2023-05-12_on": dstats(r[r.index >= seg])}
    fund["bh_50_50"] = {"full_chain": st["bh_50_50"], "hl_covered_2023-05-12_on": dstats(bh[bh.index >= seg])}
    # per fold
    frows = []
    for y in PF_FOLD_YEARS:
        sel = [p for p in picks if p["fold"] == y and p["is_rank"] <= 3]
        m = (idx.year == y)
        row = {"fold": y, "systems": "; ".join(f"{p['symbol'][:3]}:{p['family']}({pstr(p['params'])})" for p in sel)}
        for k in ("K3_ew_plain", "K3_ew_vt", "K3_iv_plain"):
            d = dstats(variants[k][m]); row[f"{k}_ret_pct"] = d["total_return_pct"]; row[f"{k}_sharpe"] = d["sharpe"]; row[f"{k}_dd_pct"] = d["max_dd_pct"]
        d = dstats(bh[m]); row["bh_ret_pct"] = d["total_return_pct"]; row["bh_sharpe"] = d["sharpe"]; row["bh_dd_pct"] = d["max_dd_pct"]
        frows.append(row)
    fdf = pd.DataFrame(frows)
    pdf = pd.DataFrame([{k: (pstr(v) if k == "params" else v) for k, v in p.items() if k not in ("oos", "is_daily")} for p in picks])
    fdf.to_csv(RES / "portfolio_walkforward_folds.csv", index=False)
    pdf.to_csv(RES / "portfolio_walkforward_picks.csv", index=False)
    pd.DataFrame({**variants, "bh_50_50": bh}).to_csv(RES / "portfolio_walkforward_daily_returns.csv")
    summ = {"design": "results/portfolio_walkforward_design.md", "variants": st, "funding": fund,
            "n_design_variants": 8, "primary": "K3_ew_plain"}
    (RES / "portfolio_walkforward_summary.json").write_text(json.dumps(_clean(summ), indent=2))
    return summ, fdf, pdf


def null_breakeven_n(sr_ann, T, skew, kurt):
    """Largest N such that DSR >= 0.95 when every trial is pure noise (Var[SR_hat] = 1/T, daily units)."""
    sr = sr_ann / math.sqrt(365)
    be = 0
    for n in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 5000, 20000):
        if psr(sr, expected_max_sr(n, 1.0 / T), T, skew, kurt) >= 0.95:
            be = n
        else:
            break
    return be


def alpha_vs_bh(r, bh):
    import numpy.linalg as la
    X = np.column_stack([np.ones(len(bh)), bh.to_numpy()])
    y = r.to_numpy()
    beta, *_ = la.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res.var(ddof=2)
    cov = s2 * la.inv(X.T @ X)
    return {"alpha_ann_pct": float(beta[0] * 365 * 100), "alpha_t": float(beta[0] / math.sqrt(cov[0, 0])),
            "beta": float(beta[1]), "corr": float(np.corrcoef(y, bh)[0, 1])}


def write_portfolio_md(picks):
    s, fdf, pdf = portfolio_report(picks)
    d = pd.read_csv(RES / "portfolio_walkforward_daily_returns.csv", index_col=0, parse_dates=True)
    V = s["variants"]
    for k in V:
        v = V[k]
        v["null_breakeven_N"] = null_breakeven_n(v["sharpe"], v["days"], v["skew"], v["kurt"])
        if k != "bh_50_50":
            v.update(alpha_vs_bh(d[k], d["bh_50_50"]))
    (RES / "portfolio_walkforward_summary.json").write_text(json.dumps(_clean(s), indent=2))
    L = ["# Portfolio walk-forward (BTC + ETH, IS-only picks per fold, chained OOS)", "",
         "Design fixed beforehand: `portfolio_walkforward_design.md`. Picks per fold: `portfolio_walkforward_picks.csv`. "
         "Daily returns: `portfolio_walkforward_daily_returns.csv`.", "",
         "OOS chain 2020-01-01 → 2026-10-30 (2,495 days). Costs 0.045% + 2 bp per side; funding excluded here (see below).", "",
         "## All 8 design variants (primary = K3_ew_plain)", "",
         "| variant | CAGR% | Sharpe | maxDD% | total% | PSR>0 | DSR N=4 | DSR N=8 | null break-even N | P(SR > B&H SR) | α/yr% vs B&H | α t | β |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in V.items():
        a = f"{v['alpha_ann_pct']:.1f} | {v['alpha_t']:.2f} | {v['beta']:.2f}" if "alpha_t" in v else "— | — | 1"
        L.append(f"| {k} | {v['cagr_pct']:.1f} | {v['sharpe']:.2f} | {v['max_dd_pct']:.1f} | {v['total_return_pct']:.0f} | "
                 f"{v['psr_vs_0']:.3f} | {v['dsr_N4']:.3f} | {v['dsr_N8']:.3f} | {v['null_breakeven_N']} | {v['psr_vs_bh_sharpe']:.2f} | {a} |")
    L += ["", "## Per fold (OOS calendar year)", "",
          "| year | EW ret% | EW Sharpe | EW DD% | EW+VT ret% | EW+VT Sharpe | EW+VT DD% | B&H ret% | B&H DD% |", "|---|---|---|---|---|---|---|---|---|"]
    for r in fdf.itertuples():
        L.append(f"| {r.fold} | {r.K3_ew_plain_ret_pct:.1f} | {r.K3_ew_plain_sharpe:.2f} | {r.K3_ew_plain_dd_pct:.1f} | "
                 f"{r.K3_ew_vt_ret_pct:.1f} | {r.K3_ew_vt_sharpe:.2f} | {r.K3_ew_vt_dd_pct:.1f} | {r.bh_ret_pct:.1f} | {r.bh_dd_pct:.1f} |")
    L += ["", "Systems picked per fold (IS only):", ""] + [f"- {r.fold}: {r.systems}" for r in fdf.itertuples()]
    F = s["funding"]
    L += ["", "## Vol targeting on the portfolio (same formula: f = min(1, 0.40 / 30-day realised vol))", "",
          f"- EW K=3: CAGR {V['K3_ew_plain']['cagr_pct']:.1f}% → {V['K3_ew_vt']['cagr_pct']:.1f}%, Sharpe {V['K3_ew_plain']['sharpe']:.2f} → "
          f"{V['K3_ew_vt']['sharpe']:.2f}, maxDD {V['K3_ew_plain']['max_dd_pct']:.1f}% → {V['K3_ew_vt']['max_dd_pct']:.1f}%.",
          "- Same direction for every variant: lower CAGR, higher Sharpe, ~6–7 pp lower drawdown.", "",
          "## Funding (Hyperliquid perp carry; spot signal assumed traded as HL perp)", "",
          "| variant | full chain CAGR / Sharpe / DD | 2023-05-12 → end CAGR / Sharpe / DD |", "|---|---|---|"]
    for k, x in F.items():
        a, b = x["full_chain"], x["hl_covered_2023-05-12_on"]
        L.append(f"| {k} | {a['cagr_pct']:.1f}% / {a['sharpe']:.2f} / {a['max_dd_pct']:.1f}% | {b['cagr_pct']:.1f}% / {b['sharpe']:.2f} / {b['max_dd_pct']:.1f}% |")
    L += ["", "- `_hlfund` = actual HL hourly funding; HL has no series before 2023-05-12, so the full-chain `_hlfund` row only "
          "charges funding from then on (not invented). The right-hand column is the like-for-like covered window.",
          "- `_bandX` = longs pay a flat X%/yr while in position over the whole chain (sensitivity).", ""]
    (RES / "portfolio_walkforward.md").write_text("\n".join(L) + "\n")
    return s
