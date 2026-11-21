"""No-grid time-series momentum baseline (fixed classic rule, stated before running):

  SMA200 daily trend rule, long-only, traded on 4h bars:
    long  while the asset's last COMPLETED UTC daily close > SMA(200) of completed daily closes
    flat  otherwise. Daily bars are built from the asset's own 1h bars; a 4h bar inside day D sees
    day D-1 (never day D) = Pine request.security(sym, "D", close[1] > ta.sma(close, 200)[1],
    lookahead_on). Decision at the 4h close, fill at the next 4h open, same fees/slip as the lab.
  No parameters are searched (200 is the textbook value). BTC, ETH, and EW 50/50 daily-rebalanced.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import honest as H
from . import indicators as ta
from .backtest import Costs, run
from .funding import blended, constant_band, hl_bar_funding
from .research import _clean, expected_max_sr, psr

SMA_LEN = 200


def sma200_signal(df, src1h, n: int = SMA_LEN):
    c = ta.htf_prev_closed_external(df.index, src1h, "1D", lambda x: x["close"].to_numpy(float))
    s = ta.htf_prev_closed_external(df.index, src1h, "1D", lambda x: ta.sma(x["close"].to_numpy(float), n))
    with np.errstate(invalid="ignore"):
        bull = c > s
    z = np.zeros(len(df), bool)
    return {"long_entry": bull, "long_exit": ~bull, "short_entry": z, "short_exit": z.copy()}


def baseline_daily(sym, variant, start=None, end=None):
    """Chained calendar-fold daily returns (each fold starts flat, same as the lab portfolio WF)."""
    df = H.df_(sym, "4h")
    sig = sma200_signal(df, H.df_(sym, "1h"))
    fund = {"plain": None, "hlfund": np.nan_to_num(hl_bar_funding(df, sym)), "blend": blended(df, sym, "4h"),
            "band10": constant_band(df, "4h", 0.10), "band20": constant_band(df, "4h", 0.20)}[variant]
    s = sig if fund is None else dict(sig, funding=fund)
    parts = []
    for y, a, b in H.pf_folds(df):
        if b > a:
            parts.append(H.daily(run(df, s, Costs(), start=a, end=b)))
    return pd.concat(parts)


def compare(port_picks):
    """port_picks: lab portfolio WF picks (pickled, primary pick rule)."""
    out, rows = {}, []
    seg = pd.Timestamp("2023-05-12", tz="UTC")
    bh = H.bh_chain()
    for v, pv in (("plain", "plain"), ("hlfund", "plain_hlfund"), ("blend", "plain_blend"), ("band10", "plain_band10"), ("band20", "plain_band20")):
        b = {s: baseline_daily(s, v) for s in H.PF_ASSETS}
        ew = pd.concat(b, axis=1).fillna(0).mean(axis=1)
        lab = H.chain(port_picks, 3, "ew", pv)
        labvt = H.chain(port_picks, 3, "ew", pv.replace("plain", "vt"))
        idx = lab.index.intersection(ew.index)
        series = {"baseline_BTC": b["BTCUSDT"].reindex(idx).fillna(0), "baseline_ETH": b["ETHUSDT"].reindex(idx).fillna(0),
                  "baseline_EW": ew.reindex(idx), "lab_EW_K3": lab.reindex(idx), "lab_EW_K3_vt": labvt.reindex(idx),
                  "bh_50_50": bh.reindex(idx).fillna(0)}
        for name, r in series.items():
            for win, rr in (("full_2020_on", r), ("hl_covered_2023-05-12_on", r[r.index >= seg])):
                d = H.dstats(rr)
                rows.append({"funding": v, "series": name, "window": win, **{k: d[k] for k in ("cagr_pct", "sharpe", "max_dd_pct", "total_return_pct", "days")}})
        # lab vs baseline: P(lab SR > baseline SR), regression alpha of lab on baseline
        L, B = series["lab_EW_K3"], series["baseline_EW"]
        dL, dB = H.dstats(L), H.dstats(B)
        a = H.alpha_vs_bh(L, B)
        out[v] = {"p_lab_sr_gt_baseline_sr": psr(dL["sharpe"] / math.sqrt(365), dB["sharpe"] / math.sqrt(365), dL["days"], dL["skew"], dL["kurt"]),
                  "lab_alpha_vs_baseline_ann_pct": a["alpha_ann_pct"], "lab_alpha_t": a["alpha_t"], "lab_beta_to_baseline": a["beta"],
                  "corr_lab_baseline": a["corr"], "baseline_psr_vs_0": psr(dB["sharpe"] / math.sqrt(365), 0, dB["days"], dB["skew"], dB["kurt"]),
                  "baseline_null_breakeven_N": H.null_breakeven_n(dB["sharpe"], dB["days"], dB["skew"], dB["kurt"])}
        if v == "plain":
            pd.DataFrame(series).to_csv(H.RES / "momentum_baseline_daily_returns.csv")
            yr = []
            for y in H.PF_FOLD_YEARS:
                m = idx.year == y
                yr.append({"year": y, **{f"{k}_ret_pct": H.dstats(s_[m])["total_return_pct"] for k, s_ in series.items()}})
            pd.DataFrame(yr).to_csv(H.RES / "momentum_baseline_by_year.csv", index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(H.RES / "momentum_baseline_compare.csv", index=False)
    (H.RES / "momentum_baseline.json").write_text(json.dumps(_clean({"rule": __doc__.strip(), "tests": out, "table": tab.to_dict("records")}), indent=2))
    return tab, out


def funding_in_picks_compare(picks_nf, picks_f):
    """Same portfolio WF, picks by IS Sharpe without vs with funding (HL where exists, flat 10%/yr before)."""
    seg = pd.Timestamp("2023-05-12", tz="UTC")
    rows, fold_rows = [], []
    for mode, P in (("pick_no_funding", picks_nf), ("pick_with_funding", picks_f)):
        for v in ("plain", "plain_hlfund", "plain_blend", "vt_blend"):
            r = H.chain(P, 3, "ew", v)
            for win, rr in (("full_2020_on", r), ("hl_covered_2023-05-12_on", r[r.index >= seg])):
                d = H.dstats(rr)
                rows.append({"pick_mode": mode, "oos_costs": v, "window": win,
                             **{k: d[k] for k in ("cagr_pct", "sharpe", "max_dd_pct", "total_return_pct", "days")}})
    key = lambda p: (p["symbol"], p["family"], H.pstr(p["params"]))  # noqa: E731
    for y in H.PF_FOLD_YEARS:
        a = {key(p) for p in picks_nf if p["fold"] == y and p["is_rank"] <= 3}
        b = {key(p) for p in picks_f if p["fold"] == y and p["is_rank"] <= 3}
        fa = {(k[0], k[1]) for k in a}
        fb = {(k[0], k[1]) for k in b}
        fold_rows.append({"fold": y, "same_systems": len(a & b), "of": len(a), "same_families": len(fa & fb),
                          "only_no_funding": "; ".join(sorted(f"{s[:3]}:{f}({p})" for s, f, p in a - b)),
                          "only_with_funding": "; ".join(sorted(f"{s[:3]}:{f}({p})" for s, f, p in b - a))})
    t, f = pd.DataFrame(rows), pd.DataFrame(fold_rows)
    t.to_csv(H.RES / "funding_in_picks_compare.csv", index=False)
    f.to_csv(H.RES / "funding_in_picks_fold_diffs.csv", index=False)
    (H.RES / "funding_in_picks.json").write_text(json.dumps(_clean({"table": t.to_dict("records"), "folds": f.to_dict("records"),
        "pre_hl_band_per_year": 0.10}), indent=2))
    return t, f
