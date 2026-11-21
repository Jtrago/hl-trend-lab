"""Frozen portfolio research protocol (NOT a live or paper trade): the lab portfolio rules are fixed
at FREEZE time and can only be scored on bars that open at/after DATA_CUTOFF.

Rule (identical to the batch-5 portfolio walk-forward primary, PF_* constants in engine/honest.py):
  assets BTCUSDT+ETHUSDT; menu = 6 long-only family sub-grids; per asset pick param = max IS Sharpe among
  IS-gated points of each menu item, keep top K=3 items by that IS Sharpe; equal weight, daily rebalanced;
  costs 0.045% + 2 bp per side; HL funding charged in evaluation; picks use IS Sharpe WITHOUT funding.
  Re-pick each Jan 1 (UTC) with IS = all bars before that date. Segment 0 = [DATA_CUTOFF, next Jan 1)
  uses the systems frozen in the spec (picked with IS = all bars before DATA_CUTOFF).
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import honest as H
from .backtest import Costs, run
from .funding import hl_bar_funding
from .metrics import compute
from .search import is_gate, pid
from .strategies import FAMILIES
from .walkforward import subgrid

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SPEC = RES / "frozen_portfolio_spec.json"


def _is_job(args):
    fam, sym, tf, p, end_time = args
    df = H.df_(sym, tf)
    dfc = df[df.index < end_time]                      # truncated: nothing at/after end_time exists
    return pid(fam, sym, tf, p), compute(run(dfc, FAMILIES[fam](dfc, **p), Costs()), tf)


def pick_systems(end_time, menu, assets, K, workers=8):
    jobs = [(fam, sym, tf, p, end_time) for sym in assets for (fam, tf, cats) in menu for p in subgrid(fam, cats)]
    with ProcessPoolExecutor(workers) as ex:
        ism = dict(ex.map(_is_job, jobs, chunksize=4))
    out = []
    for sym in assets:
        items = []
        for fam, tf, cats in menu:
            cand = [(pid(fam, sym, tf, p), p) for p in subgrid(fam, cats) if is_gate(ism[pid(fam, sym, tf, p)], tf)]
            if cand:
                i, p = max(cand, key=lambda x: ism[x[0]]["sharpe"])
                items.append({"symbol": sym, "family": fam, "tf": tf, "params": p,
                              "is_sharpe": ism[i]["sharpe"], "is_pf": ism[i]["profit_factor"],
                              "is_trades": ism[i]["closed_trades"]})
        items.sort(key=lambda d: -d["is_sharpe"])
        out += items[:K]
    return out


def last_closed_bar_end(assets):
    ends = []
    for s in assets:
        for tf, h in (("1h", 1), ("4h", 4)):
            ends.append(H.df_(s, tf).index[-1] + pd.Timedelta(hours=h))
    return min(ends)


def freeze():
    if SPEC.exists():
        raise SystemExit("frozen spec exists; refusing to overwrite")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    cutoff = last_closed_bar_end(H.PF_ASSETS)
    systems = pick_systems(cutoff, H.PF_MENU, H.PF_ASSETS, H.PF_K)
    rule = {"assets": H.PF_ASSETS,
            "menu": [{"family": f, "tf": tf, "subgrid": c, "n_points": len(subgrid(f, c))} for f, tf, c in H.PF_MENU],
            "pick": "per asset: for each menu item take the IS-gated grid point with max IS Sharpe (no funding in IS); keep top K by that IS Sharpe",
            "K_per_asset": H.PF_K, "is_gate": "closed trades >= MIN_TRADES[tf], PF >= IS gate, expectancy > 0 (engine/search.py)",
            "weighting": "equal weight across picked systems, daily rebalanced",
            "position": "long-only, 100% notional per system when in a trade, 1 position per system, no leverage",
            "costs": {"fee_per_side": 0.00045, "slip_per_side": 0.0002, "fill": "next bar open; ATR stops intrabar per README"},
            "funding": "evaluation charges actual Hyperliquid hourly funding (scripts/fetch_funding.py); spot signal assumed traded as HL perp",
            "calendar": "segment 0 = [data_cutoff, next Jan 1 UTC) with frozen systems; then re-pick every Jan 1 UTC with IS = all bars before it",
            "primary_metric": "EW K=3 with HL funding: CAGR, Sharpe (daily, 365), max DD, vs 50/50 BTC+ETH B&H and the SMA200 momentum baseline",
            "secondary_declared": ["same without funding", "vol-targeted f=min(1, 0.40/30d realised vol)"],
            "code_version": "engine/honest.py + engine/frozen.py as of freeze; tests 111+"}
    body = json.dumps(H._clean({"rule": rule, "frozen_systems": systems}), indent=2, sort_keys=True)
    sha = hashlib.sha256(body.encode()).hexdigest()
    et = now.astimezone(pd.Timestamp.now(tz="America/New_York").tz)
    doc = {"freeze_time_utc": now.isoformat(), "freeze_time_et": et.isoformat(),
           "data_cutoff_utc": cutoff.isoformat(), "data_cutoff_et": cutoff.tz_convert("America/New_York").isoformat(),
           "evaluate_only_bars_opening_at_or_after": cutoff.isoformat(), "sha256": sha, "status": "frozen research protocol — NOT a live or paper trade",
           **json.loads(body)}
    SPEC.write_text(json.dumps(doc, indent=2))
    L = ["# Frozen portfolio spec (research protocol — NOT a live or paper trade)", "",
         f"- Freeze time: **{now:%Y-%m-%d %H:%M:%S} UTC = {et:%Y-%m-%d %H:%M:%S} ET**",
         f"- Data cutoff: **{cutoff:%Y-%m-%d %H:%M} UTC = {cutoff.tz_convert('America/New_York'):%Y-%m-%d %H:%M} ET** "
         "(end of the last closed cached 1h/4h bar). Only bars opening at/after this are ever scored.",
         f"- sha256(rule+systems) = `{sha}` — `scripts/eval_frozen_portfolio.py` refuses to run if it changes.", "",
         "## Rule", ""] + [f"- **{k}**: {v}" for k, v in rule.items() if k != "menu"]
    L += ["- **menu**: " + "; ".join(f"{m['family']} {m['tf']} [{m['subgrid']}] ({m['n_points']} pts)" for m in rule["menu"]), "",
          "## Frozen systems for segment 0 (IS = all bars before the cutoff)", "",
          "| asset | family | tf | params | IS Sharpe | IS PF | IS trades |", "|---|---|---|---|---|---|---|"]
    for s in systems:
        L.append(f"| {s['symbol']} | {s['family']} | {s['tf']} | `{H.pstr(s['params'])}` | {s['is_sharpe']:.2f} | {s['is_pf']:.2f} | {s['is_trades']} |")
    L += ["", "## How to evaluate later", "",
          "```", ".venv/bin/python scripts/fetch_data.py      # refresh closed Binance bars", ".venv/bin/python scripts/fetch_funding.py   # refresh HL funding",
          ".venv/bin/python scripts/eval_frozen_portfolio.py", "```",
          "Writes `results/frozen_eval_latest.{md,json}`. Nothing in the rule may be changed; a changed rule = a new protocol with a new freeze time.",
          "Statistical note: one year of post-freeze data at Sharpe ~1 is ~1σ; expect several years before this can confirm or reject anything."]
    (RES / "frozen_portfolio_spec.md").write_text("\n".join(L) + "\n")
    return doc
