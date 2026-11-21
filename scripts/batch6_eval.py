#!/usr/bin/env python3
"""Batch 6: SMA200 momentum baseline vs lab portfolio WF, and funding-inside-picks re-run (no new grids)."""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.baseline import compare, funding_in_picks_compare  # noqa: E402
from engine.honest import portfolio_walkforward  # noqa: E402

nf = pickle.load(open(ROOT / "cache/portfolio_wf_picks.pkl", "rb"))       # from scripts/honest_eval.py portfolio
f = portfolio_walkforward(pick_funding=True)
pickle.dump(f, open(ROOT / "cache/portfolio_wf_picks_fundpick.pkl", "wb"))
compare(nf)
funding_in_picks_compare(nf, f)
print("ok")
