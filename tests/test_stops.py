"""Intrabar ATR/stop fill rule (see engine/backtest.py docstring)."""
import numpy as np
import pandas as pd
import pytest

from engine.backtest import Costs, run

KEYS = ["long_entry", "long_exit", "short_entry", "short_exit"]


def _df(o, h, l, c):
    idx = pd.date_range("2024-01-01", periods=len(o), freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx, dtype=float)


def _sig(n, long_stop=None, short_stop=None, **marks):
    s = {k: np.zeros(n, bool) for k in KEYS}
    for k, bars in marks.items():
        for b in bars:
            s[k][b] = True
    s["long_stop"] = np.full(n, np.nan) if long_stop is None else np.asarray(long_stop, float)
    s["short_stop"] = np.full(n, np.nan) if short_stop is None else np.asarray(short_stop, float)
    return s


NAN = np.nan


def test_long_stop_touch_fills_at_stop_minus_slip():
    df = _df([100, 100, 102, 101, 101], [101, 103, 104, 102, 102], [99, 99, 101, 95, 100], [100, 102, 103, 100, 101])
    # stop level known at close of bar 2 = 96 -> active on bar 3, low 95 touches
    stops = [NAN, 90, 96, 96, 96]
    r = run(df, _sig(5, long_stop=stops, long_entry=[0]), Costs(0.001, 0.002))
    t = r.trades[0]
    assert t.exit_reason == "stop" and t.exit_bar == 3 and t.exit_signal_bar == 2
    assert t.exit_raw == 96 and t.exit_px == pytest.approx(96 * (1 - 0.002))


def test_long_stop_gap_fills_at_worse_open():
    df = _df([100, 100, 102, 94, 95], [101, 103, 104, 96, 96], [99, 99, 101, 93, 94], [100, 102, 103, 95, 95])
    r = run(df, _sig(5, long_stop=[NAN, 90, 96, 96, 96], long_entry=[0]), Costs(0, 0))
    t = r.trades[0]
    assert t.exit_bar == 3 and t.exit_raw == 94      # open 94 < stop 96 -> fill at open


def test_stop_uses_previous_close_level_only():
    df = _df([100, 100, 102, 101, 101], [101, 103, 104, 102, 102], [99, 99, 101, 97, 100], [100, 102, 103, 100, 101])
    # level at bar 3 itself (98) must NOT be used on bar 3; previous level 96 not touched (low 97)
    r = run(df, _sig(5, long_stop=[NAN, 90, 96, 98, 98], long_entry=[0]), Costs(0, 0), close_at_end=False)
    assert r.trades[0].exit_bar is None
    # changing the stop level on the current/future bars does not change bar-3 behaviour
    r2 = run(df, _sig(5, long_stop=[NAN, 90, 96, 1e9, 1e9], long_entry=[0]), Costs(0, 0), close_at_end=False)
    assert r2.trades[0].exit_bar == 4    # first bar that can see the new level is bar 4


def test_stop_active_on_entry_bar():
    # signal at bar 0 close with stop 98; entry fills bar 1 open 100; bar 1 low 97 -> stopped same bar
    df = _df([100, 100, 100, 100], [101, 101, 101, 101], [99, 97, 99, 99], [100, 100, 100, 100])
    r = run(df, _sig(4, long_stop=[98, 98, 98, 98], long_entry=[0]), Costs(0, 0))
    t = r.trades[0]
    assert t.entry_bar == 1 and t.exit_bar == 1 and t.exit_raw == 98 and t.exit_reason == "stop"


def test_short_stop_touch_and_gap():
    df = _df([100, 100, 98, 99, 106], [101, 101, 99, 105, 107], [99, 97, 97, 98, 105], [100, 98, 98, 99, 106])
    r = run(df, _sig(5, short_stop=[NAN, 104, 104, 104, 104], short_entry=[0]), Costs(0, 0.001))
    t = r.trades[0]
    assert t.exit_bar == 3 and t.exit_raw == 104 and t.exit_px == pytest.approx(104 * 1.001)
    df2 = _df([100, 100, 98, 106, 106], [101, 101, 99, 107, 107], [99, 97, 97, 105, 105], [100, 98, 98, 106, 106])
    r2 = run(df2, _sig(5, short_stop=[NAN, 104, 104, 104, 104], short_entry=[0]), Costs(0, 0))
    assert r2.trades[0].exit_bar == 3 and r2.trades[0].exit_raw == 106   # gap above stop -> open


def test_nan_stop_never_triggers_and_fees_on_stop_fill():
    df = _df([100] * 5, [101] * 5, [1] * 5, [100] * 5)
    r = run(df, _sig(5, long_entry=[0]), Costs(0, 0), close_at_end=False)
    assert r.trades[0].exit_bar is None
    fee = 0.001
    df2 = _df([100, 100, 100], [101, 101, 101], [99, 99, 90], [100, 100, 100])
    r2 = run(df2, _sig(3, long_stop=[95, 95, 95], long_entry=[0]), Costs(fee, 0))
    t = r2.trades[0]
    notional = 1 / (1 + fee)
    units = notional / 100
    assert t.ret == pytest.approx(notional + units * (95 - 100) - units * 95 * fee - 1, rel=1e-12)


def test_reentry_after_stop_waits_for_signal_and_next_open():
    df = _df([100, 100, 100, 100, 100], [101] * 5, [99, 99, 90, 99, 99], [100] * 5)
    r = run(df, _sig(5, long_stop=[95] * 5, long_entry=[0, 2]), Costs(0, 0))
    assert [(t.entry_bar, t.exit_bar, t.exit_reason) for t in r.trades][:2] == [(1, 2, "stop"), (3, 4, "forced")]
