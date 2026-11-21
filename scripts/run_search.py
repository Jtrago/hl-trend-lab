#!/usr/bin/env python3
"""Run a strategy-family grid over cached data and update survivors.

Usage: .venv/bin/python scripts/run_search.py ma_cross [trend_pullback ...]
       .venv/bin/python scripts/run_search.py --smoke   # tiny grid on BTC 4h only
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.data import SYMBOLS  # noqa: E402
from engine.search import _clean, run_family, save_rows, update_survivors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def summarize(family, rows, survivors, best_is):
    out = {"family": family, "backtests": len(rows),
           "is_qualified": sum(r["is_gate"] for r in rows), "survivors": len(survivors),
           "rule_pass_counts_among_is_qualified": {}, "is_best_oos": []}
    q = [r for r in rows if r["is_gate"]]
    if q:
        for k in q[0]["checks"]:
            out["rule_pass_counts_among_is_qualified"][k] = sum(r["checks"][k] for r in q)
    for (sym, tf), r in sorted(best_is.items()):
        out["is_best_oos"].append({"symbol": sym, "tf": tf, "params": r["params"],
                                   "is_sharpe": round(r["is"]["sharpe"], 2), "is_pf": round(r["is"]["profit_factor"], 2),
                                   "oos_pf": round(r["oos"]["profit_factor"], 2), "oos_sharpe": round(r["oos"]["sharpe"], 2),
                                   "oos_trades": r["oos"]["closed_trades"], "oos_dd": round(r["oos"]["max_dd_pct"], 1),
                                   "survivor": r["survivor"]})
    # buy-and-hold context for the same segments (no costs) — trend results must be read against this
    from engine.data import load
    bh = {}
    for (sym, tf) in sorted(best_is):
        df = load(sym, tf); sp = len(df) * 2 // 3; c = df["close"]
        bh[f"{sym}_{tf}"] = {"is_bh_pct": round((c.iloc[sp - 1] / df["open"].iloc[0] - 1) * 100, 1),
                             "oos_bh_pct": round((c.iloc[-1] / df["open"].iloc[sp] - 1) * 100, 1)}
    out["buy_and_hold"] = bh
    return out


def main(argv):
    smoke = "--smoke" in argv
    fams = [a for a in argv if not a.startswith("--")] or ["ma_cross"]
    for fam in fams:
        t0 = time.time()
        if smoke:
            grid = {"ma_type": ["ema"], "side": ["long"], "fast": [10, 20, 30], "slow": [50, 100, 200]} \
                if fam == "ma_cross" else None
            rows, surv, best = run_family(fam, ["BTCUSDT"], ["4h"], grid=grid)
        else:
            from engine.search import FAMILY_TFS
            rows, surv, best = run_family(fam, SYMBOLS, FAMILY_TFS.get(fam, ["1h", "4h"]))
        label = f"{fam}{'_smoke' if smoke else ''}_{time.strftime('%Y%m%d_%H%M')}"
        path = save_rows(rows, label)
        if not smoke:
            update_survivors(surv, label)
        s = summarize(fam, rows, surv, best)
        s["seconds"] = round(time.time() - t0, 1)
        s["results_csv"] = str(path.relative_to(ROOT))
        (ROOT / "results" / f"{label}_summary.json").write_text(json.dumps(_clean(s), indent=2))
        print(json.dumps(_clean(s), indent=2), flush=True)
        if not smoke and "--no-robustness" not in argv:
            from engine.robustness import run_all
            rs, *_ = run_all()
            print("robustness:", json.dumps(_clean(rs["by_family"].get(fam, {}))), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
