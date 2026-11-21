"""Freeze cutoff, SMA200 baseline timing, blended funding (minimal)."""
import numpy as np
import pandas as pd

import engine.frozen as F
import engine.honest as H
from engine.baseline import sma200_signal
from engine.funding import blended, constant_band
from tests.conftest import make_fixture


def _fx(start="2024-01-01", years=2):
    df = make_fixture(n=24 * 365 * years, seed=5)
    df.index = pd.date_range(start, periods=len(df), freq="1h", tz="UTC")
    return df


def test_freeze_pick_ignores_bars_at_or_after_cutoff():
    df = _fx()
    cut = pd.Timestamp("2025-06-01", tz="UTC")
    H._DF[("FAKEUSDT", "1h")] = df
    args = ("donchian", "FAKEUSDT", "1h", {"side": "long", "n_entry": 20, "n_exit": 10}, cut)
    _, m1 = F._is_job(args)
    df2 = df.copy()
    late = df2.index >= cut
    df2.loc[late, ["open", "high", "low", "close"]] *= 2.0
    H._DF[("FAKEUSDT", "1h")] = df2
    _, m2 = F._is_job(args)
    del H._DF[("FAKEUSDT", "1h")]
    assert m1 == m2


def test_runner_segments_calendar():
    import importlib.util
    import sys
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / "eval_frozen_portfolio.py"
    spec = importlib.util.spec_from_file_location("efp", p)
    m = importlib.util.module_from_spec(spec); sys.modules["efp"] = m; spec.loader.exec_module(m)
    segs = m.segments(pd.Timestamp("2026-10-30 08:00", tz="UTC"), pd.Timestamp("2028-03-01", tz="UTC"))
    assert [(str(a.date()), str(b.date()), r is None) for a, b, r in segs] == [
        ("2026-10-30", "2027-01-01", True), ("2027-01-01", "2028-01-01", False), ("2028-01-01", "2028-03-01", False)]


def test_sma200_uses_previous_completed_day_only():
    src = _fx("2023-01-01", 2)
    df4 = src.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    s1 = sma200_signal(df4, src)["long_entry"]
    t = df4.index.get_loc(pd.Timestamp("2024-06-10 16:00", tz="UTC"))
    src2 = src.copy()
    day = (src2.index >= "2024-06-10") & (src2.index < "2024-06-10 16:00")   # same-day, already-seen hours
    src2.loc[day, "close"] *= 10
    future = src2.index >= "2024-06-10 16:00"
    src2.loc[future, "close"] *= 0.01
    s2 = sma200_signal(df4, src2)["long_entry"]
    assert s1[t] == s2[t]                       # current day and future cannot change the signal
    prev = (src2.index >= "2024-06-09") & (src2.index < "2024-06-10")
    src3 = src.copy(); src3.loc[prev, "close"] *= 10          # previous completed day DOES matter
    assert sma200_signal(df4, src3)["long_entry"][t]


def test_blended_funding_fills_only_missing(monkeypatch):
    df = _fx("2023-05-01", 1)
    hl = np.full(len(df), np.nan); hl[500:] = 3e-5
    monkeypatch.setattr("engine.funding.hl_bar_funding", lambda d, s: hl)
    b = blended(df, "BTCUSDT", "1h", 0.10)
    assert np.allclose(b[500:], 3e-5)
    assert np.allclose(b[:500], constant_band(df, "1h", 0.10)[:500])
