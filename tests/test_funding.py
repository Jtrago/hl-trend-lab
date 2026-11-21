"""Funding accrual (perp carry) math + event->bar mapping timing."""
import numpy as np
import pandas as pd

from engine.backtest import Costs, run
from engine.funding import constant_band, map_events_to_bars


def _sig(n, entry=5, exit_=50, d=1):
    z = np.zeros(n, bool)
    le, lx, se, sx = z.copy(), z.copy(), z.copy(), z.copy()
    if d == 1:
        le[entry], lx[exit_] = True, True
    else:
        se[entry], sx[exit_] = True, True
    return {"long_entry": le, "long_exit": lx, "short_entry": se, "short_exit": sx}


def test_zero_funding_is_identity(fx):
    s = _sig(len(fx))
    a = run(fx, s, Costs())
    b = run(fx, dict(s, funding=np.zeros(len(fx))), Costs())
    assert np.allclose(a.equity, b.equity)


def test_long_pays_exact_sum_short_receives(fx):
    n, rate = len(fx), 1e-4
    c = fx["close"].to_numpy()
    for d in (1, -1):
        s = _sig(n, d=d)
        a = run(fx, s, Costs())
        b = run(fx, dict(s, funding=np.full(n, rate)), Costs())
        tr = a.trades[0]
        units = b.trades[0].__dict__.get("units")  # not stored; recompute below
        held = range(tr.entry_bar, tr.exit_bar)     # position open at close of these bars
        eq0 = 1.0
        notional0 = eq0 / (1 + Costs().fee)
        units = notional0 / tr.entry_px
        expected = sum(units * c[t] * rate * d for t in held)
        assert np.isclose(b.trades[0].funding_paid, expected)
        assert np.isclose(a.equity[-1] - b.equity[-1], expected, rtol=1e-9, atol=1e-12)
        assert (expected > 0) == (d == 1)
        # bars before entry unaffected (no lookahead / no charge while flat)
        assert np.allclose(a.equity[:tr.entry_bar], b.equity[:tr.entry_bar])


def test_event_to_bar_mapping():
    H = 3_600_000
    t0 = int(pd.Timestamp("2025-01-01", tz="UTC").value // 1_000_000)
    bars_1h = np.array([t0 + k * H for k in range(24)])
    bars_4h = np.array([t0 + k * 4 * H for k in range(6)])
    ev = np.array([t0 + 9 * H + 48, t0 + 12 * H + 81])   # 09:00:00.048 and 12:00:00.081
    r = np.array([1.0, 2.0])
    f1 = map_events_to_bars(bars_1h, ev, r)
    assert f1[8] == 1.0 and f1[11] == 2.0 and np.isnan(f1[:8]).all()   # bar ending at the event hour
    f4 = map_events_to_bars(bars_4h, ev, r)
    assert f4[2] == 3.0 and np.isnan(f4[:2]).all()   # 08-12 bar holds both (09:00 inside, 12:00 = its close)


def test_constant_band_scale():
    df = pd.DataFrame(index=range(10))
    assert np.isclose(constant_band(df, "4h", 0.10)[0] * 6 * 365, 0.10)
