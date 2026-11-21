"""(2) Fee + slippage applied on every entry and exit, long and short."""
import numpy as np
import pandas as pd
import pytest

from engine.backtest import Costs, run
from engine.metrics import compute


def _df(opens, closes=None):
    closes = closes or opens
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="1h", tz="UTC")
    return pd.DataFrame({"open": opens, "high": opens, "low": opens, "close": closes}, index=idx)


def _sig(n, **marks):
    s = {k: np.zeros(n, bool) for k in ["long_entry", "long_exit", "short_entry", "short_exit"]}
    for k, bars in marks.items():
        for b in bars:
            s[k][b] = True
    return s


def test_long_round_trip_exact():
    df = _df([100, 100, 110, 120, 120])
    fee, slip = 0.001, 0.002
    r = run(df, _sig(5, long_entry=[0], long_exit=[2]), Costs(fee, slip))
    t = r.trades[0]
    assert t.entry_px == pytest.approx(100 * 1.002)
    assert t.exit_px == pytest.approx(120 * 0.998)
    notional = 1 / (1 + fee)
    units = notional / t.entry_px
    expected = notional + units * (t.exit_px - t.entry_px) - units * t.exit_px * fee
    assert r.equity[-1] == pytest.approx(expected, rel=1e-12)
    assert t.ret == pytest.approx(expected - 1, rel=1e-12)
    assert t.gross_ret == pytest.approx(0.2)


def test_short_round_trip_exact():
    df = _df([100, 100, 90, 80, 80])
    fee, slip = 0.00045, 0.0002
    r = run(df, _sig(5, short_entry=[0], short_exit=[2]), Costs(fee, slip))
    t = r.trades[0]
    assert t.entry_px == pytest.approx(100 * (1 - slip))
    assert t.exit_px == pytest.approx(80 * (1 + slip))
    notional = 1 / (1 + fee)
    units = notional / t.entry_px
    expected = notional + units * (t.entry_px - t.exit_px) - units * t.exit_px * fee
    assert t.ret == pytest.approx(expected - 1, rel=1e-12)


def test_flat_price_round_trip_costs_two_sides():
    df = _df([100.0] * 6)
    fee, slip = 0.00045, 0.0002
    r = run(df, _sig(6, long_entry=[0], long_exit=[2]), Costs(fee, slip))
    # lose ~ 2*fee + 2*slip on a flat market
    assert r.trades[0].ret == pytest.approx(-(2 * fee + 2 * slip), abs=2e-6)
    assert r.trades[0].ret < 0


def test_zero_cost_equals_gross_and_costs_monotone(fx):
    from engine.strategies import ma_cross
    sig = ma_cross(fx, 5, 20, "ema", "both")
    r0 = run(fx, sig, Costs(0, 0))
    for t in r0.closed:
        assert t.ret == pytest.approx(t.gross_ret, abs=1e-12)
    finals = [run(fx, sig, Costs(f, s)).equity[-1] for f, s in [(0, 0), (0.00045, 0), (0.00045, 0.0002), (0.001, 0.001)]]
    assert all(a > b for a, b in zip(finals, finals[1:]))
    # each round trip costs ~2*(fee+slip) of the exit value vs gross
    r1 = run(fx, sig, Costs(0.00045, 0.0002))
    assert all(t.ret < t.gross_ret for t in r1.closed)
    d = np.array([(t.gross_ret - t.ret) / (1 + t.gross_ret) for t in r1.closed if t.direction == 1])
    assert len(d) and np.all(d > 0.0012) and np.all(d < 0.0014)


def test_reversal_charges_both_legs():
    df = _df([100, 100, 100, 100, 100, 100])
    fee = 0.001
    r = run(df, _sig(6, long_entry=[0], short_entry=[2], short_exit=[4]), Costs(fee, 0))
    assert [t.direction for t in r.trades] == [1, -1]
    assert r.trades[0].exit_bar == r.trades[1].entry_bar == 3
    assert r.equity[-1] == pytest.approx((1 / (1 + fee) * (1 - fee)) ** 2, rel=1e-12)


def test_metrics_basic():
    df = _df([100, 100, 110, 110, 110, 100, 100])
    r = run(df, _sig(7, long_entry=[0, 3], long_exit=[1, 4]), Costs(0, 0))
    m = compute(r, "1h")
    assert m["trades"] == 2 and m["win_rate"] == 0.5
    assert m["profit_factor"] == pytest.approx(0.1 / (1 - 100 / 110))
