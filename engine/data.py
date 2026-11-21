"""Public Binance spot kline fetch + cache. Paper/research only — no API keys used.

Endpoint: https://data-api.binance.vision/api/v3/klines (public market-data mirror).
Bars are never invented: gaps from exchange outages are left as gaps and reported.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
# Override with HL_TREND_LAB_DATA (the test suite points this at tests/fixtures/data).
DATA_DIR = Path(os.environ.get("HL_TREND_LAB_DATA") or Path(__file__).resolve().parent.parent / "data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000}
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _get(url: str, retries: int = 5) -> list:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hl-trend-lab/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    return []


def fetch_klines(symbol: str, interval: str, start_ms: int = 0, end_ms: int | None = None) -> pd.DataFrame:
    """Page through public klines; drop the still-forming bar (close_time >= now)."""
    now_ms = int(time.time() * 1000)
    end_ms = end_ms or now_ms
    rows: list = []
    cur = start_ms
    while True:
        url = f"{BASE_URL}?symbol={symbol}&interval={interval}&limit=1000&startTime={cur}&endTime={end_ms}"
        batch = _get(url)
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if len(batch) < 1000:
            break
        cur = last_open + INTERVAL_MS[interval]
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return df
    df = df.drop(columns=["ignore"]).drop_duplicates("open_time").sort_values("open_time")
    df = df[df["close_time"] < now_ms]  # closed bars only
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df.reset_index(drop=True)


def cache_path(symbol: str, interval: str) -> Path:
    return DATA_DIR / f"{symbol}_{interval}.csv"


def load(symbol: str, interval: str) -> pd.DataFrame:
    """Load cached bars with a UTC DatetimeIndex (bar open time)."""
    df = pd.read_csv(cache_path(symbol, interval))
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.index.name = "time"
    return df


def resample_1h_to_4h(df1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h -> 4h (UTC-aligned 00/04/08/...); only complete 4-bar buckets kept."""
    agg = df1h[["open", "high", "low", "close", "volume"]].resample("4h", label="left", closed="left")
    out = pd.DataFrame({
        "open": agg["open"].first(), "high": agg["high"].max(), "low": agg["low"].min(),
        "close": agg["close"].last(), "volume": agg["volume"].sum(),
        "n": df1h["close"].resample("4h", label="left", closed="left").count(),
    })
    return out[out["n"] == 4].drop(columns="n")


def gap_report(df: pd.DataFrame, interval: str) -> dict:
    d = df["open_time"].diff().dropna()
    step = INTERVAL_MS[interval]
    gaps = d[d != step]
    return {"gaps": int(len(gaps)), "missing_bars": int(((gaps / step) - 1).sum())}
