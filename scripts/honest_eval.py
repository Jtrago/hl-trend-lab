#!/usr/bin/env python3
"""Batch-5 honest evaluation.
  python scripts/honest_eval.py register    # freeze shortlist (refuses if one is already registered)
  python scripts/honest_eval.py evaluate    # one-shot holdout evaluation (verifies sha256)
  python scripts/honest_eval.py portfolio   # portfolio walk-forward + vol target + funding
Requires data/funding_*.csv from scripts/fetch_funding.py (public Hyperliquid API, no keys)."""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import honest  # noqa: E402

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "register":
    if (ROOT / "results/preregistered_list.json").exists() and "--force" not in sys.argv:
        sys.exit("already registered; re-registering after seeing the holdout would defeat the point (use --force knowingly)")
    print(honest.register()["sha256_of_shortlist"])
elif cmd == "evaluate":
    honest.evaluate(); honest.write_eval_md()
elif cmd == "portfolio":
    picks = honest.portfolio_walkforward()
    (ROOT / "cache").mkdir(exist_ok=True)
    pickle.dump(picks, open(ROOT / "cache/portfolio_wf_picks.pkl", "wb"))
    honest.write_portfolio_md(picks)
else:
    sys.exit(__doc__)
