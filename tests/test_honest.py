"""Timing guarantees for the honest-evaluation harness: calendar folds and pre-cutoff truncation."""
import numpy as np
import pandas as pd

import engine.honest as H
from tests.conftest import make_fixture


def _long_fixture():
    df = make_fixture(n=24 * 365 * 3, seed=3)
    df.index = pd.date_range("2024-06-01", periods=len(df), freq="1h", tz="UTC")
    return df


def test_pf_folds_calendar_and_disjoint():
    df = _long_fixture()
    for y, a, b in H.pf_folds(df):
        if a < len(df):
            assert df.index[a] >= pd.Timestamp(f"{y}-01-01", tz="UTC")
            assert a == 0 or df.index[a - 1] < pd.Timestamp(f"{y}-01-01", tz="UTC")   # IS ends before OOS year
        if 0 < b < len(df):
            assert df.index[b - 1] < pd.Timestamp(f"{y + 1}-01-01", tz="UTC") <= df.index[b]


def test_registration_pick_ignores_data_after_cutoff():
    df = _long_fixture()
    H._DF[("FAKEUSDT", "1h")] = df
    p = {"side": "long", "n_entry": 20, "n_exit": 10}
    _, m1 = H._pick_job(("donchian", "FAKEUSDT", "1h", p))
    df2 = df.copy()
    late = df2.index >= H.CUTOFF
    assert late.any()
    df2.loc[late, ["open", "high", "low", "close"]] *= np.linspace(0.5, 3.0, late.sum())[:, None]
    H._DF[("FAKEUSDT", "1h")] = df2
    _, m2 = H._pick_job(("donchian", "FAKEUSDT", "1h", p))
    assert m1 == m2
    del H._DF[("FAKEUSDT", "1h")]


def test_null_breakeven_monotone_in_sharpe():
    a = H.null_breakeven_n(0.5, 2500, 0.0, 3.0)
    b = H.null_breakeven_n(1.5, 2500, 0.0, 3.0)
    assert b >= a and b > 0
