#!/usr/bin/env python3
"""Fetch Hyperliquid PUBLIC funding history (no keys): POST https://api.hyperliquid.xyz/info
{"type":"fundingHistory","coin":C,"startTime":ms}. Paginates 500 rows/request.
Writes data/funding_{COIN}.csv (time_ms, funding_rate, premium) + data/funding_manifest.json.
Rates are per funding event (signed fraction of notional; positive => longs pay shorts)."""
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

URL = "https://api.hyperliquid.xyz/info"
DATA = Path(__file__).resolve().parent.parent / "data"


def post(body, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "User-Agent": "hl-trend-lab/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def fetch(coin):
    rows, start = [], 0
    while True:
        batch = post({"type": "fundingHistory", "coin": coin, "startTime": start})
        if not batch:
            break
        rows += batch
        last = batch[-1]["time"]
        if len(batch) < 500:
            break
        start = last + 1
        time.sleep(0.25)
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    return pd.DataFrame({"time_ms": df["time"].astype("int64"), "funding_rate": df["fundingRate"].astype(float),
                         "premium": df["premium"].astype(float)})


def main():
    man = {"source": URL + ' {"type":"fundingHistory"} (public, no auth)',
           "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "coins": {}}
    for coin in ["BTC", "ETH", "SOL", "XRP"]:
        df = fetch(coin)
        df.to_csv(DATA / f"funding_{coin}.csv", index=False)
        t = pd.to_datetime(df.time_ms, unit="ms", utc=True)
        gaps = t.diff().dt.total_seconds().div(3600)
        man["coins"][coin] = {"events": len(df), "first": str(t.iloc[0]), "last": str(t.iloc[-1]),
                              "median_interval_h": float(gaps.median()),
                              "mean_rate": float(df.funding_rate.mean()),
                              "annualised_mean_pct_hourly_basis": float(df.funding_rate.mean() * 24 * 365 * 100)}
        print(coin, man["coins"][coin], flush=True)
    (DATA / "funding_manifest.json").write_text(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
