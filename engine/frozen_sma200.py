"""Frozen SMA200 baseline protocol (research only — NOT a live or paper trade).
Same data cutoff as the frozen lab portfolio so both are scored on the identical post-freeze window."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Spec files live in results/ (git-ignored); HL_TREND_LAB_RESULTS overrides (tests use tests/fixtures/results).
RES = Path(os.environ.get("HL_TREND_LAB_RESULTS") or ROOT / "results")
SPEC = RES / "frozen_sma200_spec.json"
PORTFOLIO_SPEC = RES / "frozen_portfolio_spec.json"

RULE = {
    "name": "SMA200 daily trend, long-only, BTC+ETH equal weight",
    "assets": ["BTCUSDT", "ETHUSDT"],
    "signal": "long while the asset's last COMPLETED UTC daily close (daily bars built from its own Binance 1h bars) > SMA(200) of completed daily closes; flat otherwise. A 4h bar inside day D sees day D-1 only.",
    "trade_tf": "4h", "decision": "at 4h bar close", "fill": "next 4h bar open",
    "sma_len": 200, "parameters_searched": "none (textbook value)",
    "costs": {"fee_per_side": 0.00045, "slip_per_side": 0.0002},
    "position": "100% notional per asset when long, no leverage, no shorts",
    "weighting": "equal weight BTC/ETH, daily rebalanced",
    "funding": "evaluation charges actual Hyperliquid hourly funding (spot signal assumed traded as HL perp); also reported without funding",
    "calendar": "post-freeze window starts flat at the data cutoff; same calendar segments (Jan 1 UTC splits) as the frozen lab portfolio",
    "code": "engine/baseline.py::sma200_signal",
}


def body_sha(rule):
    body = json.dumps({"rule": rule}, indent=2, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def freeze():
    if SPEC.exists():
        raise SystemExit("frozen SMA200 spec exists; refusing to overwrite")
    port = json.loads(PORTFOLIO_SPEC.read_text())
    now = datetime.now(timezone.utc).replace(microsecond=0)
    cutoff = pd.Timestamp(port["data_cutoff_utc"])
    et = pd.Timestamp(now).tz_convert("America/New_York")
    doc = {"freeze_time_utc": now.isoformat(), "freeze_time_et": et.isoformat(),
           "data_cutoff_utc": cutoff.isoformat(), "data_cutoff_et": cutoff.tz_convert("America/New_York").isoformat(),
           "relation_to_portfolio_freeze": f"same data cutoff as frozen_portfolio_spec (sha {port['sha256'][:16]}…, frozen {port['freeze_time_utc']}); "
                                           "no new bars arrived between the two freezes",
           "status": "frozen research protocol — NOT a live or paper trade",
           "sha256": body_sha(RULE), "rule": RULE}
    SPEC.write_text(json.dumps(doc, indent=2))
    L = ["# Frozen SMA200 baseline spec (research protocol — NOT a live or paper trade)", "",
         f"- Freeze time: **{now:%Y-%m-%d %H:%M:%S} UTC = {et:%Y-%m-%d %H:%M:%S} ET**",
         f"- Data cutoff: **{cutoff:%Y-%m-%d %H:%M} UTC = {cutoff.tz_convert('America/New_York'):%Y-%m-%d %H:%M} ET**, identical to the frozen lab "
         f"portfolio (frozen {port['freeze_time_et'][:19]} ET). No new bars arrived between the two freezes, so both score the same window.",
         f"- sha256(rule) = `{doc['sha256']}`; the runner refuses to score if it changes.", "", "## Rule", ""]
    L += [f"- **{k}**: {v}" for k, v in RULE.items()]
    L += ["", "## Evaluate", "", "```", ".venv/bin/python scripts/fetch_data.py && .venv/bin/python scripts/fetch_funding.py",
          ".venv/bin/python scripts/eval_frozen_all.py      # scores frozen lab portfolio + frozen SMA200 on the same window", "```"]
    (RES / "frozen_sma200_spec.md").write_text("\n".join(L) + "\n")
    return doc


def load_verified(path=None):
    doc = json.loads((path or SPEC).read_text())
    if body_sha(doc["rule"]) != doc["sha256"]:
        raise SystemExit("frozen SMA200 spec was modified after freeze — refusing to evaluate")
    return doc
