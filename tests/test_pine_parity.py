"""(3) Python <-> Pine parity: the EXPORTED Pine text is executed by engine.pine_sim
(independent Pine-semantics ta.* + Pine broker) and must reproduce the engine's trades."""
import re

import numpy as np
import pytest

from engine import indicators as ta
from engine.backtest import Costs, run
from engine.pine import export
from engine.pine_sim import run as pine_run
from engine.strategies import FAMILIES

CASES = [
    ("ma_cross", dict(fast=5, slow=20, ma_type="ema", side="long")),
    ("ma_cross", dict(fast=9, slow=21, ma_type="ema", side="both")),
    ("ma_cross", dict(fast=10, slow=40, ma_type="sma", side="both")),
    ("trend_pullback", dict(trend_len=50, rsi_len=7, rsi_entry=40, rsi_exit=60, trend_type="ema", side="long")),
    ("trend_pullback", dict(trend_len=30, rsi_len=5, rsi_entry=35.0, rsi_exit=55.0, trend_type="sma", side="both")),
    ("donchian", dict(n_entry=20, n_exit=10, side="long")),
    ("donchian", dict(n_entry=30, n_exit=15, side="both")),
    ("donchian_atr", dict(n_entry=20, atr_len=14, atr_mult=2.5, side="long")),
    ("donchian_atr", dict(n_entry=20, atr_len=10, atr_mult=2.0, side="both")),
    ("ma_cross_atr", dict(fast=5, slow=20, atr_len=14, atr_mult=2.0, ma_type="ema", side="both")),
    ("donchian_ratchet", dict(n_entry=20, atr_len=14, atr_mult=2.5, side="both")),
    ("donchian_ratchet", dict(n_entry=30, atr_len=10, atr_mult=2.0, side="long")),
    ("bollinger_breakout", dict(bb_len=20, bb_mult=2.0, side="both")),
    ("keltner_breakout", dict(kc_len=20, kc_mult=1.5, side="both")),
    ("htf_donchian", dict(n_entry=20, n_exit=10, htf_len=10, side="both")),
    ("htf_ma_cross_atr", dict(fast=5, slow=20, atr_len=14, atr_mult=2.0, htf_len=10, ma_type="ema", side="both")),
    ("htf_keltner", dict(kc_len=20, kc_mult=1.5, htf_len=10, side="both")),
]


@pytest.mark.parametrize("family,params", CASES)
def test_trades_match(fx, family, params):
    costs = Costs()
    script = export(family, params, costs)
    py = run(fx, FAMILIES[family](fx, **params), costs, close_at_end=False)
    pine = pine_run(script, fx)["trades"]
    assert len(py.trades) >= 5, "fixture should produce trades"
    if "atr" in family or "ratchet" in family:
        assert any(t.exit_reason == "stop" for t in py.trades), "fixture must exercise the stop path"
    assert len(py.trades) == len(pine)
    for a, b in zip(py.trades, pine):
        assert (a.direction, a.entry_bar, a.exit_bar) == (b["direction"], b["entry_bar"], b["exit_bar"])
        assert a.entry_raw == pytest.approx(b["entry_price"])
        if a.exit_bar is not None:
            assert a.exit_raw == pytest.approx(b["exit_price"]) and a.exit_reason == b["exit_reason"]
        if a.exit_bar is not None:
            # slip-on-price (Python) vs slip-folded-into-commission (Pine): agree to O(cost * |ret|)
            assert a.ret == pytest.approx(b["ret"], abs=1e-6 + 0.002 * abs(a.ret))


def test_indicators_match_pine_reference(fx):
    from engine.pine_sim import p_ema, p_highest, p_lowest, p_rma, p_rsi, p_sma, p_stdev, p_tr
    c = fx["close"].to_numpy()
    for n in (3, 14, 50):
        np.testing.assert_allclose(ta.ema(c, n), p_ema(list(c), n), rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(ta.sma(c, n), p_sma(list(c), n), rtol=1e-9, equal_nan=True)
        np.testing.assert_allclose(ta.rsi(c, n), p_rsi(list(c), n), rtol=1e-9, atol=1e-9, equal_nan=True)
        h, l = fx["high"].to_numpy(), fx["low"].to_numpy()
        np.testing.assert_allclose(ta.highest(h, n), p_highest(list(h), n), equal_nan=True)
        np.testing.assert_allclose(ta.lowest(l, n), p_lowest(list(l), n), equal_nan=True)
        np.testing.assert_allclose(ta.atr(h, l, c, n), p_rma(p_tr(list(h), list(l), list(c)), n), rtol=1e-9, equal_nan=True)
        np.testing.assert_allclose(ta.stdev(c, n), p_stdev(list(c), n), rtol=1e-9, atol=1e-12, equal_nan=True)


def test_pine_script_shape():
    s = export("ma_cross", dict(fast=12, slow=26, ma_type="ema", side="long"))
    assert s.startswith("//@version=6\n")
    assert "strategy(" in s and "process_orders_on_close=false" in s
    assert "commission_value=0.065" in s
    assert re.search(r"fastLen = input.int\(12,", s) and re.search(r"slowLen = input.int\(26,", s)
    assert "ta.ema(close, fastLen)" in s and "ta.crossover(fastMa, slowMa)" in s
    assert 'strategy.entry("L", strategy.long)' in s and 'strategy.close("L")' in s
    # v6: no deprecated v4/v5-only names
    assert "study(" not in s and "security(" not in s and "transp=" not in s


@pytest.mark.parametrize("family,params", [
    ("ma_cross", dict(fast=10, slow=50, ma_type="ema", side="both")),
    ("trend_pullback", dict(trend_len=100, rsi_len=3, rsi_entry=30, rsi_exit=60, trend_type="sma", side="both")),
    ("donchian", dict(n_entry=55, n_exit=20, side="both")),
    ("donchian_atr", dict(n_entry=40, atr_len=22, atr_mult=3.0, side="both")),
    ("ma_cross_atr", dict(fast=10, slow=50, atr_len=22, atr_mult=3.0, ma_type="ema", side="both")),
    ("donchian_ratchet", dict(n_entry=40, atr_len=22, atr_mult=3.0, side="both")),
    ("bollinger_breakout", dict(bb_len=50, bb_mult=2.0, side="both")),
    ("keltner_breakout", dict(kc_len=50, kc_mult=2.0, side="both")),
    ("htf_donchian@1h", dict(n_entry=55, n_exit=20, htf_len=50, side="both")),
    ("htf_ma_cross_atr@1h", dict(fast=10, slow=50, atr_len=22, atr_mult=3.0, htf_len=50, ma_type="ema", side="both")),
    ("htf_keltner@1h", dict(kc_len=20, kc_mult=2.0, htf_len=50, side="both")),
])
def test_trades_match_real_btc_slice(family, params):
    from engine.data import cache_path, load
    tf = "1h" if family.endswith("@1h") else "4h"
    family = family.split("@")[0]
    if not cache_path("BTCUSDT", tf).exists():  # fixture ships with the repo
        pytest.skip("cached data not present; run scripts/fetch_data.py")
    df = load("BTCUSDT", tf).iloc[-2500:]
    py = run(df, FAMILIES[family](df, **params), Costs(), close_at_end=False)
    pine = pine_run(export(family, params), df)["trades"]
    assert len(py.trades) == len(pine) and len(pine) > 10
    assert [(t.direction, t.entry_bar, t.exit_bar) for t in py.trades] == \
        [(b["direction"], b["entry_bar"], b["exit_bar"]) for b in pine]
    for a, b in zip(py.trades, pine):
        if a.exit_bar is not None:
            assert a.exit_raw == pytest.approx(b["exit_price"])
