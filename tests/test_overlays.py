"""BTC trend-phase filter timing, vol-target sizing math, and their Pine parity."""
import numpy as np
import pandas as pd
import pytest

from engine import indicators as ta
from engine import overlays as ov
from engine.backtest import Costs, run
from engine.pine import export
from engine.pine_sim import run as pine_run
from engine.strategies import FAMILIES, ma_cross
from tests.conftest import make_fixture

ov.register_btc_families(["donchian", "ma_cross_atr", "keltner_breakout"])


def _pair():
    fx = make_fixture(n=2400, seed=3)
    btc = make_fixture(n=2700, seed=11)
    btc.index = btc.index - pd.Timedelta(hours=300)       # BTC history starts earlier
    fx.attrs["btc"] = btc
    return fx, btc


def test_btc_daily_value_is_previous_closed_day():
    fx, btc = _pair()
    v = ta.htf_prev_closed_external(fx.index, btc, "1D", lambda x: x["close"].to_numpy(float))
    daily_close = btc["close"].groupby(btc.index.floor("1D")).last()
    for t in (50, 500, 1234, 2399):
        day = fx.index[t].floor("1D")
        prev = daily_close[daily_close.index < day]
        assert v[t] == prev.iloc[-1]


def test_btc_filter_ignores_current_day_and_future():
    fx, btc = _pair()
    base = ov.btc_bull(fx, 10)
    t = 1000
    day = fx.index[t].floor("1D")
    b2 = btc.copy()
    m = b2.index >= day                                   # change BTC from the start of t's day on
    b2.loc[m, ["open", "high", "low", "close"]] *= 3.0
    fx2 = fx.copy()
    fx2.attrs["btc"] = b2
    np.testing.assert_array_equal(ov.btc_bull(fx2, 10)[:t + 1], base[:t + 1])
    b3 = btc.copy()
    prev_day = b3.index[(b3.index < day)].floor("1D").max()
    b3.loc[b3.index.floor("1D") == prev_day, "close"] *= 0.3   # the previous closed day DOES matter
    fn = lambda x: x["close"].to_numpy(float)
    v0 = ta.htf_prev_closed_external(fx.index, btc, "1D", fn)
    v3 = ta.htf_prev_closed_external(fx.index, b3, "1D", fn)
    assert v3[t] == pytest.approx(v0[t] * 0.3)


def test_gate_and_flat_semantics():
    fx, _ = _pair()
    sig = ma_cross(fx, 5, 20, "ema", "both")
    bull = ov.btc_bull(fx, 10)
    g = ov.apply_btc_filter(sig, fx, "gate", 10)
    f = ov.apply_btc_filter(sig, fx, "flat", 10)
    assert not np.any(g["long_entry"] & ~bull)
    np.testing.assert_array_equal(g["long_exit"], sig["long_exit"])
    np.testing.assert_array_equal(g["short_entry"], sig["short_entry"])
    assert np.all(f["long_exit"][~bull])
    r = run(fx, f, Costs(), close_at_end=False)
    for t in r.trades:
        if t.direction == 1:
            assert bull[t.signal_bar]


@pytest.mark.parametrize("family,params", [
    ("donchian__btcgate", dict(n_entry=20, n_exit=10, side="both", _btc_len=10)),
    ("ma_cross_atr__btcflat", dict(fast=5, slow=20, atr_len=14, atr_mult=2.0, ma_type="ema", side="long", _btc_len=10)),
    ("keltner_breakout__btcgate", dict(kc_len=20, kc_mult=1.5, side="long", _btc_len=10)),
])
def test_btc_filter_pine_parity(family, params):
    fx, btc = _pair()
    py = run(fx, FAMILIES[family](fx, **params), Costs(), close_at_end=False).trades
    pine = pine_run(export(family, params), fx, {"BINANCE:BTCUSDT": btc})["trades"]
    assert len(py) >= 5 and len(py) == len(pine)
    assert [(t.direction, t.entry_bar, t.exit_bar) for t in py] == \
        [(b["direction"], b["entry_bar"], b["exit_bar"]) for b in pine]


# ---- vol-target sizing ----
def test_size_one_reproduces_default_exactly(fx):
    sig = ma_cross(fx, 5, 20, "ema", "both")
    a = run(fx, sig, Costs())
    s2 = dict(sig, size=np.ones(len(fx)))
    b = run(fx, s2, Costs())
    np.testing.assert_allclose(a.equity, b.equity, rtol=0, atol=0)


def test_fractional_size_math_exact():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": [100, 100, 120, 120, 120], "high": 121, "low": 99,
                       "close": [100, 110, 120, 120, 120]}, index=idx, dtype=float)
    z = np.zeros(5, bool)
    sig = {"long_entry": z.copy(), "long_exit": z.copy(), "short_entry": z.copy(), "short_exit": z.copy(),
           "size": np.full(5, 0.25)}
    sig["long_entry"][0] = True
    sig["long_exit"][1] = True
    fee = 0.001
    r = run(df, sig, Costs(fee, 0))
    n0 = 0.25 / (1 + fee)
    units = n0 / 100
    expected = 0.75 + n0 + units * 20 - units * 120 * fee
    assert r.equity[-1] == pytest.approx(expected, rel=1e-12)
    assert r.equity[1] == pytest.approx(0.75 + n0 + units * 10, rel=1e-12)   # MTM includes idle cash
    assert r.trades[0].size == 0.25


def test_vol_target_formula_cap_and_causality(fx):
    f = ov.vol_target_size(fx, "1h", target_vol=0.4, vol_days=5, cap=1.0)
    n = 5 * 24
    lr = np.diff(np.log(fx["close"].to_numpy()))
    t = 300
    rv = lr[t - n:t].std(ddof=1) * np.sqrt(8760)
    assert f[t] == pytest.approx(min(1.0, 0.4 / rv))
    assert np.nanmax(f) <= 1.0 and np.all(np.isnan(f[:n - 1]))
    f2 = ov.vol_target_size(fx, "1h", target_vol=5.0, vol_days=5, cap=2.0)
    assert np.nanmax(f2) == 2.0
    d = fx.copy()
    d.iloc[t + 1:, d.columns.get_loc("close")] *= 2
    np.testing.assert_allclose(ov.vol_target_size(d, "1h", 0.4, 5, 1.0)[:t + 1], f[:t + 1], equal_nan=True)


def test_size_read_at_signal_bar(fx):
    sig = ma_cross(fx, 5, 20, "ema", "long")
    size = np.linspace(0.1, 1.0, len(fx))
    r = run(fx, dict(sig, size=size), Costs())
    for t in r.trades:
        assert t.size == size[t.signal_bar]
