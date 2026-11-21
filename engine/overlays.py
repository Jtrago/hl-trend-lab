"""Overlays on any strategy family (paper research only).

BTC trend-phase filter
  btcBull[t] = BTC daily close > BTC daily SMA(btc_len), both from the PREVIOUS fully closed UTC
  day relative to bar t's open (engine.indicators.htf_prev_closed_external). Daily bars are built
  from cached BTCUSDT 1h bars. Until the SMA is warm (first btc_len days of BTC history) btcBull is
  False.
  mode "gate": long entries only when btcBull (shorts and exits unchanged).
  mode "flat": gate + longs are also exited (signal exit -> next open) when btcBull is False.

Volatility-targeted sizing
  f[t] = min(cap, target_vol / realized_vol[t]), realized_vol = annualised stdev of the last
  `vol_days` days of bar log returns (bars <= t). f is read at the SIGNAL bar and fixed for the trade
  (engine sig["size"]). No leverage above `cap`. Not exported to Pine (Pine qty would be computed
  from the signal-bar close, not the next open).
"""
from __future__ import annotations

import numpy as np

from . import indicators as ta
from .metrics import BARS_PER_YEAR
from .strategies import FAMILIES

_BTC = {}


def btc_source(df):
    if "btc" in df.attrs:
        return df.attrs["btc"]
    if "btc" not in _BTC:
        from .data import load
        _BTC["btc"] = load("BTCUSDT", "1h")
    return _BTC["btc"]


def btc_bull(df, btc_len: int = 200):
    src = btc_source(df)
    bc = ta.htf_prev_closed_external(df.index, src, "1D", lambda x: x["close"].to_numpy(float))
    bs = ta.htf_prev_closed_external(df.index, src, "1D", lambda x: ta.sma(x["close"].to_numpy(float), btc_len))
    with np.errstate(invalid="ignore"):
        return bc > bs


def apply_btc_filter(sig: dict, df, mode: str = "gate", btc_len: int = 200) -> dict:
    bull = btc_bull(df, btc_len)
    s = dict(sig)
    s["long_entry"] = sig["long_entry"] & bull
    if mode == "flat":
        s["long_exit"] = sig["long_exit"] | ~bull
    return s


def vol_target_size(df, tf: str, target_vol: float = 0.40, vol_days: int = 30, cap: float = 1.0):
    bpy = BARS_PER_YEAR[tf]
    n = int(vol_days * bpy / 365)
    rv = ta.realized_vol(df["close"].to_numpy(float), n, bpy)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.minimum(cap, target_vol / rv)
    f[~np.isfinite(f)] = np.nan
    return f


def make_btc_family(base: str, mode: str, btc_len: int = 200):
    fn = FAMILIES[base]

    def fam(df, _btc_len: int = btc_len, **p):
        return apply_btc_filter(fn(df, **p), df, mode, _btc_len)
    fam.__name__ = f"{base}__btc{mode}"
    return fam


def register_btc_families(bases, modes=("gate", "flat")):
    from .search import FAMILY_TFS, GRIDS
    names = []
    for b in bases:
        for m in modes:
            name = f"{b}__btc{m}"
            FAMILIES[name] = make_btc_family(b, m)
            GRIDS[name] = GRIDS[b]
            if b in FAMILY_TFS:
                FAMILY_TFS[name] = FAMILY_TFS[b]
            names.append(name)
    return names
