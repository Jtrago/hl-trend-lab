"""Paired circular block bootstrap for lab-minus-SMA200 performance, and 50/50 blend sensitivity.
Research only; no optimisation (blend weight fixed at 50/50 a priori)."""
from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pandas as pd

from . import honest as H
from .baseline import baseline_daily
from .research import _clean

BLOCK = 20          # days; ~1 trading month, covers the autocorrelation of trend-system daily returns
REPS = 10_000
SEED = 20261030
SEG = pd.Timestamp("2023-05-12", tz="UTC")


def block_indices(n, block, rng):
    """Circular block bootstrap: ceil(n/block) blocks with uniform random starts, wrapped, cut to n."""
    k = math.ceil(n / block)
    starts = rng.integers(0, n, k)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def sharpe(x):
    sd = x.std(ddof=1, axis=-1)
    return np.where(sd > 0, x.mean(axis=-1) / sd * math.sqrt(365), 0.0)


def cagr(x):
    return np.exp(np.log1p(x).sum(axis=-1) * 365 / x.shape[-1]) - 1


def paired_bootstrap(a, b, block=BLOCK, reps=REPS, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = np.stack([block_indices(len(a), block, rng) for _ in range(reps)])
    A, B = a[idx], b[idx]
    ds, dc = sharpe(A) - sharpe(B), cagr(A) - cagr(B)
    pt_s = float(sharpe(a[None])[0] - sharpe(b[None])[0])
    pt_c = float(cagr(a[None])[0] - cagr(b[None])[0])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"block_days": block, "reps": reps, "n_days": len(a),
            "sharpe_diff": pt_s, "sharpe_diff_ci95": q(ds), "sharpe_diff_p_le_0": float((ds <= 0).mean()),
            "zero_in_sharpe_ci": bool(q(ds)[0] <= 0 <= q(ds)[1]),
            "cagr_diff_pct": pt_c * 100, "cagr_diff_ci95_pct": [x * 100 for x in q(dc)], "cagr_diff_p_le_0": float((dc <= 0).mean()),
            "zero_in_cagr_ci": bool(q(dc)[0] <= 0 <= q(dc)[1])}


def series(picks):
    out = {}
    for fv, pv in (("nofund", "plain"), ("hlfund", "plain_hlfund"), ("blend_fund", "plain_blend")):
        bl = pd.concat({s: baseline_daily(s, {"nofund": "plain", "hlfund": "hlfund", "blend_fund": "blend"}[fv])
                        for s in H.PF_ASSETS}, axis=1).fillna(0).mean(axis=1)
        lab = H.chain(picks, 3, "ew", pv)
        idx = lab.index.intersection(bl.index)
        out[fv] = pd.DataFrame({"lab": lab.reindex(idx), "sma200": bl.reindex(idx)})
    return out


def run_all():
    picks = pickle.load(open(H.ROOT / "cache/portfolio_wf_picks.pkl", "rb"))
    S = series(picks)
    rows, js = [], {}
    for fv, df in S.items():
        for win, d in (("full_2020_on", df), ("hl_covered_2023-05-12_on", df[df.index >= SEG])):
            for blk in (BLOCK, 5, 60):
                r = paired_bootstrap(d.lab.to_numpy(), d.sma200.to_numpy(), block=blk)
                rows.append({"funding": fv, "window": win, **{k: v for k, v in r.items() if not isinstance(v, list)},
                             "sharpe_ci_lo": r["sharpe_diff_ci95"][0], "sharpe_ci_hi": r["sharpe_diff_ci95"][1],
                             "cagr_ci_lo_pct": r["cagr_diff_ci95_pct"][0], "cagr_ci_hi_pct": r["cagr_diff_ci95_pct"][1]})
    t = pd.DataFrame(rows)
    t.to_csv(H.RES / "bootstrap_lab_vs_sma200.csv", index=False)
    (H.RES / "bootstrap_lab_vs_sma200.json").write_text(json.dumps(_clean({
        "method": f"paired circular block bootstrap, primary block {BLOCK} days (sensitivity 5, 60), {REPS} reps, seed {SEED}; "
                  "same resampled days for lab and SMA200; percentile 95% CI", "rows": t.to_dict("records")}), indent=2))
    # 50/50 blend
    brows = []
    bh = H.bh_chain()
    for fv, df in S.items():
        blend = 0.5 * df.lab + 0.5 * df.sma200
        for win, m in (("full_2020_on", df.index >= df.index[0]), ("hl_covered_2023-05-12_on", df.index >= SEG)):
            for name, r in (("lab_EW_K3", df.lab), ("sma200_EW", df.sma200), ("blend_50_50", blend),
                            ("bh_50_50_spot", bh.reindex(df.index).fillna(0))):
                d = H.dstats(r[m])
                brows.append({"funding": fv, "window": win, "series": name,
                              **{k: d[k] for k in ("cagr_pct", "sharpe", "max_dd_pct", "total_return_pct", "days")}})
    corr = {fv: float(df.lab.corr(df.sma200)) for fv, df in S.items()}
    bt = pd.DataFrame(brows)
    bt.to_csv(H.RES / "blend_50_50.csv", index=False)
    (H.RES / "blend_50_50.json").write_text(json.dumps(_clean({"weights": "0.5 lab EW K3 + 0.5 SMA200 EW, daily rebalanced, fixed a priori",
                                                               "corr_lab_sma200": corr, "rows": bt.to_dict("records")}), indent=2))
    return t, bt, corr
