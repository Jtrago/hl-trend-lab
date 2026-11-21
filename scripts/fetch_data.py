#!/usr/bin/env python3
"""Fetch + cache public Binance spot klines (1h and native 4h) for the 4 symbols.

Usage:  .venv/bin/python scripts/fetch_data.py
Writes data/{SYMBOL}_{tf}.csv and data/manifest.json. Public endpoint, no keys.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.data import (DATA_DIR, SYMBOLS, cache_path, fetch_klines, gap_report,  # noqa: E402
                         load, resample_1h_to_4h)


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    manifest = {"source": "https://data-api.binance.vision/api/v3/klines (Binance spot, public)",
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "series": {}}
    for sym in SYMBOLS:
        for tf in ["1h", "4h"]:
            df = fetch_klines(sym, tf)
            df.to_csv(cache_path(sym, tf), index=False)
            g = gap_report(df, tf)
            manifest["series"][f"{sym}_{tf}"] = {
                "bars": len(df), "first_open": iso(df.open_time.iloc[0]),
                "last_open": iso(df.open_time.iloc[-1]), **g}
            print(sym, tf, len(df), iso(df.open_time.iloc[0]), "->", iso(df.open_time.iloc[-1]), g, flush=True)
        # cross-check native 4h vs 1h resampled
        n4 = load(sym, "4h")
        r4 = resample_1h_to_4h(load(sym, "1h"))
        common = n4.index.intersection(r4.index)
        diff = (n4.loc[common, ["open", "high", "low", "close"]] - r4.loc[common, ["open", "high", "low", "close"]]).abs()
        mism = int((diff.max(axis=1) > 1e-9).sum())
        manifest["series"][f"{sym}_4h"]["resample_check"] = {"common_bars": len(common), "ohlc_mismatch_bars": mism}
        print(sym, "4h native vs resampled-1h: common", len(common), "OHLC mismatches", mism, flush=True)
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
