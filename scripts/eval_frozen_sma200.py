#!/usr/bin/env python3
"""Score the FROZEN SMA200 baseline (results/frozen_sma200_spec.json) on bars opening at/after its cutoff.
Checksum-protected; no parameters. Research protocol — not a live or paper trade."""
import json
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
from engine.frozen_sma200 import load_verified  # noqa: E402
from engine.funding import hl_bar_funding  # noqa: E402
from engine.research import _clean  # noqa: E402

OUT = ROOT / "results"


def segments(cutoff, last_end):
    segs, s = [], cutoff
    while s < last_end:
        nxt = pd.Timestamp(f"{s.year + 1}-01-01", tz="UTC")
        segs.append((s, min(nxt, last_end)))
        s = nxt
    return segs


def main(spec_path=None, out=None):
    out = out or OUT
    doc = load_verified(spec_path)
    rule = doc["rule"]
    cutoff = pd.Timestamp(doc["data_cutoff_utc"])
    assets = rule["assets"]
    last_end = min(H.df_(s, tf).index[-1] + pd.Timedelta(hours=int(tf[0])) for s in assets for tf in ("1h", "4h"))
    res = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "spec_sha256": doc["sha256"],
           "data_cutoff_utc": str(cutoff), "data_last_closed_utc": str(last_end)}
    if last_end <= cutoff + pd.Timedelta(days=1):
        res["status"] = "no (or <1 day of) post-freeze data yet — nothing to score"
        print("SMA200:", res["status"])
    else:
        ser = {"sma200_hlfund": [], "sma200_plain": []}
        for a_t, b_t in segments(cutoff, last_end):
            for key in ser:
                rs = []
                for s in assets:
                    df = H.df_(s, "4h")
                    a, b = int(df.index.searchsorted(a_t)), int(df.index.searchsorted(b_t))
                    if b <= a:
                        continue
                    sig = sma200_signal(df, H.df_(s, "1h"))
                    if key.endswith("hlfund"):
                        sig = dict(sig, funding=np.nan_to_num(hl_bar_funding(df, s)))
                    rs.append(H.daily(run(df, sig, Costs(), start=a, end=b)))
                if rs:
                    ser[key].append(pd.concat(rs, axis=1).fillna(0).mean(axis=1))
        res["status"] = "scored"
        res["stats"] = {k: H.dstats(pd.concat(v)) for k, v in ser.items() if v and len(pd.concat(v)) > 2}
        res["daily"] = {k: {str(i.date()): float(x) for i, x in pd.concat(v).items()} for k, v in ser.items() if v}
    (out / "frozen_sma200_eval_latest.json").write_text(json.dumps(_clean(res), indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
