"""Ratcheting ATR stop + higher-timeframe (closed 4h bar) timing rules."""
import numpy as np
import pandas as pd
import pytest

from engine import indicators as ta
from engine.backtest import Costs, run
from engine.strategies import htf_donchian

KEYS = ["long_entry", "long_exit", "short_entry", "short_exit"]
NAN = np.nan


def _df(o, h, l, c):
    idx = pd.date_range("2024-01-01", periods=len(o), freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx, dtype=float)


def _sig(n, mode=None, long_stop=None, short_stop=None, **marks):
    s = {k: np.zeros(n, bool) for k in KEYS}
    for k, bars in marks.items():
        for b in bars:
            s[k][b] = True
    s["long_stop"] = np.full(n, NAN) if long_stop is None else np.asarray(long_stop, float)
    s["short_stop"] = np.full(n, NAN) if short_stop is None else np.asarray(short_stop, float)
    if mode:
        s["stop_mode"] = mode
    return s


# base levels: 90 at signal bar, rises to 95, then FALLS to 92; bar 4 low 94 sits between 92 and 95
O = [100, 100, 100, 100, 100, 100]
H = [101] * 6
L = [99, 99, 99, 99, 94, 99]
C = [100] * 6
BASE = [90, 90, 95, 92, 92, 92]


def test_ratchet_never_loosens_and_stops_where_plain_does_not():
    df = _df(O, H, L, C)
    plain = run(df, _sig(6, None, BASE, long_entry=[0]), Costs(0, 0), close_at_end=False)
    ratch = run(df, _sig(6, "ratchet", BASE, long_entry=[0]), Costs(0, 0), close_at_end=False)
    assert plain.trades[0].exit_bar is None                 # plain level fell back to 92 -> not hit
    t = ratch.trades[0]
    assert t.exit_bar == 4 and t.exit_raw == 95 and t.exit_reason == "stop"   # ratchet held 95


def test_ratchet_initial_level_is_signal_bar_base_and_active_on_entry_bar():
    df = _df([100, 100, 100], [101] * 3, [99, 89.5, 99], [100] * 3)
    r = run(df, _sig(3, "ratchet", [90, 80, 80], long_entry=[0]), Costs(0, 0))
    assert r.trades[0].entry_bar == 1 and r.trades[0].exit_bar == 1 and r.trades[0].exit_raw == 90


def test_ratchet_resets_for_new_trade():
    # trade 1 trails up to 97, stopped; trade 2 must start from ITS signal-bar base (85), not 97
    o = [100, 100, 100, 100, 100, 100, 100, 100]
    h = [101] * 8
    l = [99, 99, 99, 96, 99, 99, 86, 99]
    base = [90, 97, 97, 97, 85, 85, 85, 85]
    r = run(_df(o, h, l, [100] * 8), _sig(8, "ratchet", base, long_entry=[0, 4]), Costs(0, 0), close_at_end=False)
    assert [(t.entry_bar, t.exit_bar, t.exit_raw) for t in r.trades] == [(1, 3, 97), (5, None, None)]


def test_short_ratchet_only_tightens_down():
    o = [100] * 6
    h = [101, 101, 101, 101, 106, 101]
    base = [110, 110, 105, 108, 108, 108]
    r = run(_df(o, h, [99] * 6, [100] * 6), _sig(6, "ratchet", None, base, short_entry=[0]), Costs(0, 0))
    t = r.trades[0]
    assert t.exit_bar == 4 and t.exit_raw == 105


def test_htf_value_comes_from_previous_closed_bucket():
    n = 48
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = np.arange(n, dtype=float)
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c}, index=idx)
    v = ta.htf_prev_closed(df, "4h", lambda x: x["close"].to_numpy(float))
    assert np.all(np.isnan(v[:4]))
    for t in range(4, n):
        prev_bucket_last = (t // 4) * 4 - 1           # last 1h bar of the previous 4h bucket
        assert v[t] == c[prev_bucket_last]


def test_htf_ignores_current_bucket_even_on_its_last_bar():
    from tests.conftest import make_fixture
    fx = make_fixture()
    fn = lambda x: ta.ema(x["close"].to_numpy(float), 10)
    base = ta.htf_prev_closed(fx, "4h", fn)
    for t in (402, 403, 407):                         # inside / last bar of a bucket
        d = fx.copy()
        bstart = (t // 4) * 4
        d.iloc[bstart:, d.columns.get_loc("close")] *= 1.5   # change current bucket + future
        v = ta.htf_prev_closed(d, "4h", fn)
        np.testing.assert_allclose(v[:t + 1][bstart:], base[:t + 1][bstart:])
    d = fx.copy()
    d.iloc[399, d.columns.get_loc("close")] *= 1.5     # previous bucket's last bar DOES matter
    assert ta.htf_prev_closed(d, "4h", fn)[400] != base[400]


def test_htf_donchian_signals_prefix_invariant_mid_bucket():
    from tests.conftest import make_fixture
    fx = make_fixture()
    p = dict(n_entry=20, n_exit=10, htf_len=10, side="both")
    a = htf_donchian(fx, **p)
    for k in (401, 402, 403):
        d = fx.copy()
        for col in ["open", "high", "low", "close"]:
            d.iloc[k:, d.columns.get_loc(col)] *= 0.7
        b = htf_donchian(d, **p)
        for key in KEYS:
            np.testing.assert_array_equal(a[key][:k], b[key][:k])
