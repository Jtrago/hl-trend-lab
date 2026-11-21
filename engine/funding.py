"""Hyperliquid funding -> per-bar funding rates for the engine (sig["funding"]).

Assumption: signals come from Binance SPOT bars but would be traded as the Hyperliquid PERP of the
same coin. A funding event at time tau charges the position held at that instant; we map it to the
LAST bar whose open < floor_hour(tau) (for 1h: the bar ending at tau; for 4h: the 4h bar spanning
tau). The engine charges a bar's mapped funding if a position is open at that bar's close.
Before the first HL event (HL perps start 2023-05-12) there is NO data: those bars get NaN and are
reported as uncovered — no series is invented. `constant_band` gives a flat annualised-rate
sensitivity (e.g. +/-10%/yr paid by longs) for uncovered periods.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("HL_TREND_LAB_DATA") or Path(__file__).resolve().parent.parent / "data")
COIN = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "XRPUSDT": "XRP"}
BAR_HOURS = {"1h": 1, "4h": 4}


def load_events(symbol):
    f = pd.read_csv(DATA / f"funding_{COIN[symbol]}.csv")
    return f


def map_events_to_bars(bar_open_ms: np.ndarray, event_ms: np.ndarray, rates: np.ndarray):
    """Sum event rates into bars: bar = last bar with open < floor_hour(event). NaN before coverage."""
    hour = (event_ms // 3_600_000) * 3_600_000
    idx = np.searchsorted(bar_open_ms, hour, side="left") - 1
    out = np.zeros(len(bar_open_ms))
    ok = idx >= 0
    np.add.at(out, idx[ok], rates[ok])
    first = int(idx[ok].min()) if ok.any() else len(out)
    out[:first] = np.nan          # bars before the first mapped event: no data (not zero)
    return out


def hl_bar_funding(df, symbol):
    ev = load_events(symbol)
    return map_events_to_bars(df["open_time"].to_numpy("int64") if "open_time" in df else
                              df.index.asi8 // 1_000_000, ev.time_ms.to_numpy("int64"), ev.funding_rate.to_numpy(float))


def constant_band(df, tf, annual_rate: float):
    """Flat sensitivity: longs pay annual_rate per year of holding (shorts receive)."""
    return np.full(len(df), annual_rate * BAR_HOURS[tf] / (24 * 365))


def coverage(f, start, end):
    seg = f[start:end]
    return float(np.isfinite(seg).mean()) if len(seg) else 0.0


def hl_or_zero(df, symbol):
    f = hl_bar_funding(df, symbol)
    return np.nan_to_num(f, nan=0.0), f


PRE_HL_BAND = 0.10   # flat longs-pay rate (per year) used ONLY where no HL series exists (pre 2023-05-12)


def blended(df, symbol, tf, band: float = PRE_HL_BAND):
    """Actual HL funding where it exists; a flat `band`/yr (longs pay) before HL coverage starts."""
    f = hl_bar_funding(df, symbol)
    return np.where(np.isfinite(f), f, constant_band(df, tf, band))
