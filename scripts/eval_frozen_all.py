#!/usr/bin/env python3
"""Score BOTH frozen protocols (lab portfolio + SMA200 baseline) on the same post-freeze window and
write results/frozen_eval_combined.md. Refresh data first (fetch_data.py, fetch_funding.py)."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def main():
    p = _load("eval_frozen_portfolio").main()
    s = _load("eval_frozen_sma200").main()
    assert p["data_cutoff_utc"] == s["data_cutoff_utc"], "frozen specs have different cutoffs"
    L = ["# Frozen protocols — combined post-freeze scorecard (research only, not live/paper trading)", "",
         f"- Cutoff {p['data_cutoff_utc']}; data through {p['data_last_closed_utc']}.",
         f"- Lab spec sha `{p['spec_sha256'][:16]}…`, SMA200 spec sha `{s['spec_sha256'][:16]}…` (both verified).", ""]
    if p["status"] != "scored":
        L.append(f"Status: {p['status']}")
    else:
        L += ["| series | days | return % | CAGR % | Sharpe | maxDD % |", "|---|---|---|---|---|---|"]
        for k, d in {**p["stats"], **s.get("stats", {})}.items():
            if "sharpe" in d:
                L.append(f"| {k} | {d['days']} | {d['total_return_pct']:.1f} | {d['cagr_pct']:.1f} | {d['sharpe']:.2f} | {d['max_dd_pct']:.1f} |")
        L += ["", "Short windows are noise (~1 yr at Sharpe 1 ≈ 1 standard error). Do not change either rule based on this."]
    (ROOT / "results" / "frozen_eval_combined.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return p, s


if __name__ == "__main__":
    main()
