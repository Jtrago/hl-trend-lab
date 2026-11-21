"""(1) No-lookahead: values/decisions at bar t never depend on bars > t."""
import numpy as np
import pytest

from engine import indicators as ta
from engine.backtest import Costs, run
from engine.strategies import (bollinger_breakout, donchian, donchian_atr, donchian_ratchet, htf_donchian,
                               keltner_breakout, ma_cross, ma_cross_atr, trend_pullback,
                               htf_ma_cross_atr, htf_keltner)

STRATS = [
    (ma_cross, dict(fast=5, slow=20, ma_type="ema", side="both")),
    (ma_cross, dict(fast=8, slow=30, ma_type="sma", side="long")),
    (trend_pullback, dict(trend_len=50, rsi_len=7, rsi_entry=40, rsi_exit=60, trend_type="ema", side="both")),
    (donchian, dict(n_entry=20, n_exit=10, side="both")),
    (donchian_atr, dict(n_entry=20, atr_len=14, atr_mult=2.5, side="both")),
    (ma_cross_atr, dict(fast=5, slow=20, atr_len=10, atr_mult=2.0, ma_type="ema", side="both")),
    (donchian_ratchet, dict(n_entry=20, atr_len=14, atr_mult=2.5, side="both")),
    (bollinger_breakout, dict(bb_len=20, bb_mult=2.0, side="both")),
    (keltner_breakout, dict(kc_len=20, kc_mult=1.5, side="both")),
    (htf_donchian, dict(n_entry=20, n_exit=10, htf_len=10, side="both")),
    (htf_ma_cross_atr, dict(fast=5, slow=20, atr_len=14, atr_mult=2.0, htf_len=10, ma_type="ema", side="both")),
    (htf_keltner, dict(kc_len=20, kc_mult=1.5, htf_len=10, side="both")),
]


@pytest.mark.parametrize("fn", [lambda d: ta.ema(d.close.to_numpy(), 10), lambda d: ta.sma(d.close.to_numpy(), 10),
                                lambda d: ta.rsi(d.close.to_numpy(), 14),
                                lambda d: ta.highest(d.high.to_numpy(), 20), lambda d: ta.lowest(d.low.to_numpy(), 20),
                                lambda d: ta.atr(d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(), 14)])
def test_indicators_prefix_invariant(fx, fn):
    full = fn(fx)
    for k in (30, 200, 555):
        np.testing.assert_allclose(fn(fx.iloc[:k]), full[:k], equal_nan=True)


def _perturb_future(df, k, seed=1):
    d = df.copy()
    rng = np.random.default_rng(seed)
    m = 1 + rng.normal(0, 0.2, len(d) - k)
    for col in ["open", "high", "low", "close"]:
        d.iloc[k:, d.columns.get_loc(col)] = d[col].to_numpy()[k:] * m
    return d


@pytest.mark.parametrize("strat,params", STRATS)
def test_signals_unchanged_when_future_changes(fx, strat, params):
    k = 400
    a, b = strat(fx, **params), strat(_perturb_future(fx, k), **params)
    for key in a:
        np.testing.assert_array_equal(a[key][:k], b[key][:k])   # includes stop levels


@pytest.mark.parametrize("strat,params", STRATS)
def test_trades_unchanged_when_future_changes(fx, strat, params):
    k = 400
    ta_ = run(fx, strat(fx, **params)).trades
    d2 = _perturb_future(fx, k)
    tb = run(d2, strat(d2, **params)).trades
    # every fill strictly before bar k uses only data < k and must be identical
    ea = [(t.direction, t.entry_bar, t.entry_px) for t in ta_ if t.entry_bar < k]
    eb = [(t.direction, t.entry_bar, t.entry_px) for t in tb if t.entry_bar < k]
    assert ea == eb and len(ea) > 0
    xa = [(t.exit_bar, t.exit_px) for t in ta_ if t.exit_bar is not None and t.exit_bar < k]
    xb = [(t.exit_bar, t.exit_px) for t in tb if t.exit_bar is not None and t.exit_bar < k]
    assert xa == xb


@pytest.mark.parametrize("strat,params", STRATS)
def test_fill_is_next_bar_open(fx, strat, params):
    costs = Costs(fee=0.0, slip=0.0)
    res = run(fx, strat(fx, **params), costs, close_at_end=False)
    o = fx["open"].to_numpy()
    assert res.trades
    for t in res.trades:
        assert t.entry_bar == t.signal_bar + 1
        assert t.entry_px == o[t.entry_bar]
        if t.exit_bar is not None:
            assert t.exit_bar == t.exit_signal_bar + 1
            if t.exit_reason == "stop":   # intrabar: stop level from exit_signal_bar close, or worse open
                sig = strat(fx, **params)
                lvl = (sig["long_stop"] if t.direction == 1 else sig["short_stop"])[t.exit_signal_bar]
                worse = min(o[t.exit_bar], lvl) if t.direction == 1 else max(o[t.exit_bar], lvl)
                if sig.get("stop_mode") == "ratchet":   # trail >= base (long) / <= base (short)
                    assert (t.exit_px >= worse - 1e-12) if t.direction == 1 else (t.exit_px <= worse + 1e-12)
                else:
                    assert t.exit_px == worse
            else:
                assert t.exit_px == o[t.exit_bar]


def test_signal_on_last_bar_never_fills(fx):
    n = len(fx)
    sig = {k: np.zeros(n, bool) for k in ["long_entry", "long_exit", "short_entry", "short_exit"]}
    sig["long_entry"][n - 1] = True
    assert run(fx, sig).trades == []


def test_oos_segment_starts_flat_and_ignores_is_positions(fx):
    n = len(fx)
    sig = {k: np.zeros(n, bool) for k in ["long_entry", "long_exit", "short_entry", "short_exit"]}
    sig["long_entry"][10] = True        # IS entry, never exited
    split = n * 2 // 3
    oos = run(fx, sig, start=split)
    assert oos.trades == []
    is_ = run(fx, sig, end=split)
    assert len(is_.trades) == 1 and is_.trades[0].forced and is_.trades[0].exit_bar == split - 1
