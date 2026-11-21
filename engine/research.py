"""Batch-4 research: deflated Sharpe on walk-forward results, BTC phase-filter WF comparison,
vol-target sizing comparison, BTC+ETH paper portfolio. Paper research only."""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import Costs, run
from .data import load
from .metrics import BARS_PER_YEAR, compute
from .overlays import vol_target_size
from .search import CATEGORICAL, GRIDS, _clean, ensure_family
from .strategies import FAMILIES
from .walkforward import folds

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EULER = 0.5772156649015329
_DF = {}


def df_(sym, tf):
    if (sym, tf) not in _DF:
        _DF[(sym, tf)] = load(sym, tf)
    return _DF[(sym, tf)]


def parse_params(fam, s):
    ensure_family(fam)
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k] = v if k in CATEGORICAL else type(GRIDS[fam][k][0])(v)
    return out


# ---------------- deflated Sharpe (Bailey & López de Prado 2014) ----------------
def expected_max_sr(n_trials, var_sr):
    """E[max SR] of N iid trials with cross-trial SR variance var_sr (per-bar units)."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                                + EULER * norm.ppf(1 - 1 / (n_trials * math.e)))


def psr(sr, sr0, T, skew, kurt):
    """Probabilistic Sharpe ratio P(true SR > sr0); sr per bar, kurt NON-excess."""
    den = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(den, 1e-12))))


def trial_universe():
    """All grid points ever backtested (latest results CSV per family) -> per-market OOS SR list."""
    by_fam = {}
    for f in sorted(glob.glob(str(RES / "*_2026*.csv"))):
        name = Path(f).name
        if "smoke" in name or name.startswith(("walkforward", "best_is", "robustness", "regime", "deflated")):
            continue
        fam = name.rsplit("_2026", 1)[0]
        by_fam[fam] = f                                     # latest per family
    frames = [pd.read_csv(f, usecols=["family", "symbol", "tf", "oos_sharpe"]) for f in by_fam.values()]
    u = pd.concat(frames, ignore_index=True)
    u["sr_bar"] = u.oos_sharpe / np.sqrt(u.tf.map(BARS_PER_YEAR))
    return u


def stitched_reopt_returns(folds_csv, combo_row):
    """Bar returns of the re-optimised walk-forward: each fold's IS-picked params on that fold's OOS."""
    fam, sym, tf = combo_row["family"], combo_row["symbol"], combo_row["tf"]
    df = df_(sym, tf)
    fl = folds(len(df), tf)
    rets = []
    for r in folds_csv.itertuples():
        a, b = fl[r.fold]
        if not isinstance(r.params, str):
            rets.append(np.zeros(b - a - 1))                  # no IS-qualified pick -> flat
            continue
        p = parse_params(fam, r.params)
        res = run(df, FAMILIES[fam](df, **p), Costs(), start=a, end=b)
        e = res.equity
        rets.append(np.diff(e) / e[:-1])
    return np.concatenate(rets)


def deflated_table(tag=""):
    fd = pd.read_csv(RES / f"walkforward_folds{tag}.csv")
    sm = pd.read_csv(RES / f"walkforward_summary{tag}.csv")
    cl = pd.read_csv(RES / "robustness_clusters.csv")[["cluster_id", "tier"]]
    sm = sm.merge(cl, on="cluster_id", how="left")
    sm["combo"] = sm.cluster_id.str.rsplit("|", n=1).str[0]
    u = trial_universe()
    n_total = len(u)
    rows = []
    for combo, g in sm[sm["mode"] == "reopt"].groupby("combo"):
        r0 = g.iloc[0]
        cid = r0.cluster_id
        f = fd[(fd.cluster_id == cid) & (fd["mode"] == "reopt")].sort_values("fold")
        br = stitched_reopt_returns(f, r0)
        T = len(br)
        sr = br.mean() / br.std(ddof=1) if br.std() > 0 else 0.0
        sk = float(pd.Series(br).skew())
        ku = float(pd.Series(br).kurt()) + 3
        mk = u[(u.symbol == r0.symbol) & (u.tf == r0.tf)]
        n_mkt, var_mkt = len(mk), float(mk.sr_bar.var())
        sr0_mkt = expected_max_sr(n_mkt, var_mkt)
        sr0_tot = expected_max_sr(n_total, float(u.sr_bar.var()))
        p0 = psr(sr, 0.0, T, sk, ku)
        rows.append({"combo": combo, "tier": g.tier.min(), "family": r0.family, "symbol": r0.symbol, "tf": r0.tf,
                     "folds_pass": int(r0.folds_pass), "folds": int(r0.folds), "bars": T,
                     "sharpe_ann": sr * math.sqrt(BARS_PER_YEAR[r0.tf]), "skew": sk, "kurtosis": ku,
                     "psr_vs_0": p0, "p_value": 1 - p0,
                     "bonferroni_p_market": min(1.0, (1 - p0) * n_mkt),
                     "n_trials_market": n_mkt, "sr0_market_ann": sr0_mkt * math.sqrt(BARS_PER_YEAR[r0.tf]),
                     "dsr_market": psr(sr, sr0_mkt, T, sk, ku),
                     "n_trials_total": n_total, "sr0_total_ann": sr0_tot * math.sqrt(BARS_PER_YEAR[r0.tf]),
                     "dsr_total": psr(sr, sr0_tot, T, sk, ku)})
    # correlation-aware sensitivity: grid points are highly correlated, so also deflate by an
    # "effective" trial count = number of distinct plateau clusters (all tiers) on that market, and
    # report the largest N at which the result would still clear DSR >= 0.95 (break-even N).
    allc = pd.read_csv(RES / "robustness_clusters.csv")
    for r in rows:
        mk = u[(u.symbol == r["symbol"]) & (u.tf == r["tf"])]
        var = float(mk.sr_bar.var())
        n_eff = int(((allc.symbol == r["symbol"]) & (allc.tf == r["tf"])).sum())
        bpy = BARS_PER_YEAR[r["tf"]]
        sr = r["sharpe_ann"] / math.sqrt(bpy)
        r["n_eff_clusters"] = n_eff
        r["dsr_n_eff"] = psr(sr, expected_max_sr(n_eff, var), r["bars"], r["skew"], r["kurtosis"])
        be = 1
        for n in (2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
            if psr(sr, expected_max_sr(n, var), r["bars"], r["skew"], r["kurtosis"]) >= 0.95:
                be = n
            else:
                break
        r["breakeven_n_trials"] = be if r["psr_vs_0"] >= 0.95 else 0
    out = pd.DataFrame(rows).sort_values("dsr_market", ascending=False)
    out["non_null_dsr_n_eff"] = out.dsr_n_eff >= 0.95
    out["non_null_dsr_market"] = out.dsr_market >= 0.95
    out["non_null_bonferroni"] = out.bonferroni_p_market <= 0.05
    out.to_csv(RES / "deflated_sharpe.csv", index=False)
    (RES / "deflated_sharpe.json").write_text(json.dumps(_clean(out.to_dict("records")), indent=2))
    return out


# ---------------- phase filter comparison ----------------
def regime_pass_rates(tag):
    sm = pd.read_csv(RES / f"walkforward_summary{tag}.csv")
    sm["combo"] = sm.cluster_id.str.rsplit("|", n=1).str[0]
    out = {}
    for mode in ("reopt", "fixed"):
        s = sm[sm["mode"] == mode]
        if mode == "reopt":
            s = s.drop_duplicates("combo")
        out[mode] = {"bull_pass": int(s.bull_pass.sum()), "bull_folds": int(s.bull_folds.sum()),
                     "nonbull_pass": int(s.nonbull_pass.sum()), "nonbull_folds": int(s.nonbull_folds.sum()),
                     "nonbull_rate": s.nonbull_pass.sum() / max(1, s.nonbull_folds.sum()),
                     "bull_rate": s.bull_pass.sum() / max(1, s.bull_folds.sum()),
                     "nonbull_mean_ret_pct": float(s.nonbull_mean_ret_pct.mean()),
                     "bull_mean_ret_pct": float(s.bull_mean_ret_pct.mean()),
                     "units_pass_half": int((s.pass_frac >= 0.5).sum()), "units": len(s),
                     "median_stitched_sharpe": float(s.stitched_oos_sharpe.median()),
                     "median_stitched_ret_pct": float(s.stitched_oos_return_pct.median())}
    return out


# ---------------- vol targeting ----------------
def vol_target_compare(cluster_rows, target=0.40, vol_days=30, caps=(1.0, 2.0)):
    rows = []
    for r in cluster_rows:
        fam, sym, tf, p = r["family"], r["symbol"], r["tf"], r["params"]
        df = df_(sym, tf)
        split = len(df) * 2 // 3
        sig = FAMILIES[fam](df, **p)
        variants = {"full": sig}
        for cap in caps:
            variants[f"vt{int(target * 100)}_cap{cap:g}"] = dict(sig, size=vol_target_size(df, tf, target, vol_days, cap))
        for name, sg in variants.items():
            is_m = compute(run(df, sg, Costs(), end=split), tf)
            oos_m = compute(run(df, sg, Costs(), start=split), tf)
            rows.append({"cluster_id": r["cluster_id"], "family": fam, "symbol": sym, "tf": tf,
                         "params": ",".join(f"{k}={v}" for k, v in p.items()), "sizing": name,
                         **{f"is_{k}": is_m[k] for k in ("sharpe", "cagr_pct", "max_dd_pct", "profit_factor")},
                         **{f"oos_{k}": oos_m[k] for k in ("sharpe", "cagr_pct", "max_dd_pct", "profit_factor",
                                                           "closed_trades", "exposure")}})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "voltarget_compare.csv", index=False)
    return out


# ---------------- portfolio ----------------
def system_daily_returns(fam, sym, tf, p):
    df = df_(sym, tf)
    res = run(df, FAMILIES[fam](df, **p), Costs())
    eq = pd.Series(res.equity, index=res.index)
    return eq.resample("1D").last().ffill().pct_change().dropna()


def portfolio(systems, split_time):
    """systems: list of dicts (label, family, symbol, tf, params). Daily-rebalanced paper portfolio."""
    R = pd.concat({s["label"]: system_daily_returns(s["family"], s["symbol"], s["tf"], s["params"])
                   for s in systems}, axis=1).dropna()
    is_r, oos_r = R[R.index < split_time], R[R.index >= split_time]
    w_ew = pd.Series(1 / R.shape[1], index=R.columns)
    iv = 1 / is_r.std()                                     # inverse vol from IS ONLY
    w_rp = iv / iv.sum()
    btc = load("BTCUSDT", "1h")["close"].resample("1D").last().pct_change()
    eth = load("ETHUSDT", "1h")["close"].resample("1D").last().pct_change()
    bh = (0.5 * btc + 0.5 * eth).reindex(R.index)

    def stats(r):
        eq = (1 + r).cumprod()
        yrs = len(r) / 365
        dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
        return {"total_return_pct": float((eq.iloc[-1] - 1) * 100), "cagr_pct": float((eq.iloc[-1] ** (1 / yrs) - 1) * 100),
                "sharpe": float(r.mean() / r.std() * math.sqrt(365)), "max_dd_pct": dd, "days": len(r)}
    out = {"systems": [s["label"] for s in systems], "split_time": str(split_time),
           "weights_ew": w_ew.round(4).to_dict(), "weights_invvol_IS": w_rp.round(4).to_dict()}
    for name, part in (("IS", is_r), ("OOS", oos_r)):
        out[name] = {"equal_weight": stats(part @ w_ew), "inv_vol": stats(part @ w_rp),
                     "bh_50_50_btc_eth": stats(bh.loc[part.index].fillna(0)),
                     "components": {c: stats(part[c]) for c in part.columns},
                     "avg_pairwise_corr": float(part.corr().values[np.triu_indices(part.shape[1], 1)].mean())}
        part.corr().round(2).to_csv(RES / f"portfolio_corr_{name}.csv")
    (RES / "portfolio.json").write_text(json.dumps(_clean(out), indent=2))
    return out
