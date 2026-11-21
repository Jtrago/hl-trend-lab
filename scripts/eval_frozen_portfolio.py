#!/usr/bin/env python3
"""Score the FROZEN portfolio protocol (results/frozen_portfolio_spec.json) on bars that opened at/after
its data cutoff. Never re-designs anything: systems for segment 0 come from the spec; later calendar
years are re-picked by the frozen rule only. Research protocol — not a live or paper trade.

  .venv/bin/python scripts/fetch_data.py && .venv/bin/python scripts/fetch_funding.py   # refresh first
  .venv/bin/python scripts/eval_frozen_portfolio.py
"""
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import honest as H  # noqa: E402
from engine.backtest import Costs, run  # noqa: E402
from engine.baseline import sma200_signal  # noqa: E402
from engine.frozen import SPEC, pick_systems  # noqa: E402
from engine.funding import hl_bar_funding  # noqa: E402
from engine.overlays import vol_target_size  # noqa: E402
from engine.research import _clean  # noqa: E402
from engine.strategies import FAMILIES  # noqa: E402

OUT = ROOT / "results"


def load_spec():
    doc = json.loads(SPEC.read_text())
    body = json.dumps({"rule": doc["rule"], "frozen_systems": doc["frozen_systems"]}, indent=2, sort_keys=True)
    if hashlib.sha256(body.encode()).hexdigest() != doc["sha256"]:
        sys.exit("frozen spec was modified after freeze — refusing to evaluate")
    return doc


def segments(cutoff, last_end):
    """[(start, end, repick_time|None)]: segment 0 uses frozen systems; then each Jan 1 re-pick."""
    segs, s = [], cutoff
    while s < last_end:
        nxt = pd.Timestamp(f"{s.year + 1}-01-01", tz="UTC")
        segs.append((s, min(nxt, last_end), None if s == cutoff else s))
        s = nxt
    return segs


def sys_daily(sym, tf, sig, a_t, b_t, variant):
    df = H.df_(sym, tf)
    a, b = int(df.index.searchsorted(a_t)), int(df.index.searchsorted(b_t))
    if b <= a:
        return None
    s = dict(sig)
    if "fund" in variant:
        s["funding"] = np.nan_to_num(hl_bar_funding(df, sym))
    if variant.startswith("vt"):
        s["size"] = vol_target_size(df, tf, **H.VT)
    return H.daily(run(df, s, Costs(), start=a, end=b))


def main():
    doc = load_spec()
    rule = doc["rule"]
    cutoff = pd.Timestamp(doc["data_cutoff_utc"])
    menu = [(m["family"], m["tf"], m["subgrid"]) for m in rule["menu"]]
    assets, K = rule["assets"], rule["K_per_asset"]
    last_end = min(H.df_(s, tf).index[-1] + pd.Timedelta(hours=int(tf[0])) for s in assets for tf in ("1h", "4h"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res = {"evaluated_at_utc": now, "spec_sha256": doc["sha256"], "data_cutoff_utc": str(cutoff),
           "data_last_closed_utc": str(last_end)}
    if last_end <= cutoff + pd.Timedelta(days=1):
        res["status"] = "no (or <1 day of) post-freeze data yet — nothing to score"
        print(res["status"])
    else:
        series = {k: [] for k in ("lab_hlfund", "lab_plain", "lab_vt_hlfund", "baseline_hlfund", "bh_50_50")}
        seg_info = []
        for a_t, b_t, repick in segments(cutoff, last_end):
            systems = doc["frozen_systems"] if repick is None else pick_systems(repick, menu, assets, K)
            seg_info.append({"start": str(a_t), "end": str(b_t), "systems": systems})
            for key, var in (("lab_hlfund", "plain_fund"), ("lab_plain", "plain"), ("lab_vt_hlfund", "vt_fund")):
                rs = [sys_daily(s["symbol"], s["tf"], FAMILIES[s["family"]](H.df_(s["symbol"], s["tf"]), **s["params"]), a_t, b_t, var)
                      for s in systems]
                series[key].append(pd.concat([r for r in rs if r is not None], axis=1).fillna(0).mean(axis=1))
            bl = [sys_daily(s, "4h", sma200_signal(H.df_(s, "4h"), H.df_(s, "1h")), a_t, b_t, "plain_fund") for s in assets]
            series["baseline_hlfund"].append(pd.concat(bl, axis=1).fillna(0).mean(axis=1))
            bh = []
            for s in assets:
                c = H.df_(s, "1h")["close"]
                c = c[(c.index >= a_t) & (c.index < b_t)].resample("1D").last()
                o0 = H.df_(s, "1h")["open"][H.df_(s, "1h").index >= a_t].iloc[0]
                r = c.pct_change(); r.iloc[0] = c.iloc[0] / o0 - 1
                bh.append(r)
            series["bh_50_50"].append(pd.concat(bh, axis=1).fillna(0).mean(axis=1))
        stats = {}
        for k, parts in series.items():
            r = pd.concat(parts)
            d = H.dstats(r) if len(r) > 2 else {"days": len(r)}
            stats[k] = d
        res.update({"status": "scored", "segments": seg_info, "stats": stats,
                    "note": "short windows are noise: ~1 year at Sharpe 1 is ~1 standard error"})
        L = ["# Frozen portfolio — post-freeze evaluation", "", f"- Evaluated {now} UTC on data {cutoff} → {last_end}.",
             f"- Spec sha256 `{doc['sha256'][:16]}…` verified. Research protocol, not a live or paper trade.", "",
             "| series | days | return % | CAGR % | Sharpe | maxDD % |", "|---|---|---|---|---|---|"]
        for k, d in stats.items():
            if "sharpe" in d:
                L.append(f"| {k} | {d['days']} | {d['total_return_pct']:.1f} | {d['cagr_pct']:.1f} | {d['sharpe']:.2f} | {d['max_dd_pct']:.1f} |")
        (OUT / "frozen_eval_latest.md").write_text("\n".join(L) + "\n")
    (OUT / "frozen_eval_latest.json").write_text(json.dumps(_clean(res), indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
