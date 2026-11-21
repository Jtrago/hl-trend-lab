"""Strategy families. Each returns boolean signal arrays evaluated at bar CLOSE.

Signals: long_entry, long_exit, short_entry, short_exit (np.bool_ arrays, len = bars).
The engine acts on them at the NEXT bar's open. NaN indicator => False.
Each family also has a Pine v6 body in engine/pine.py that must mirror this logic.
"""
from __future__ import annotations

import numpy as np

from . import indicators as ta


def _nz(a):
    return np.nan_to_num(np.asarray(a, float), nan=np.nan)


def ma_cross(df, fast: int, slow: int, ma_type: str = "ema", side: str = "long") -> dict:
    """Fast/slow MA cross. Long on crossover, exit on crossunder (and mirrored short)."""
    c = df["close"].to_numpy(float)
    f, s = ta.ma(c, fast, ma_type), ta.ma(c, slow, ma_type)
    up, dn = ta.crossover(f, s), ta.crossunder(f, s)
    z = np.zeros(len(c), bool)
    return {"long_entry": up, "long_exit": dn,
            "short_entry": dn if side == "both" else z, "short_exit": up if side == "both" else z}


def trend_pullback(df, trend_len: int, rsi_len: int, rsi_entry: float, rsi_exit: float,
                   trend_type: str = "ema", side: str = "long") -> dict:
    """Trend filter + RSI pullback.

    Long entry: close > trendMA and RSI < rsi_entry.  Long exit: RSI > rsi_exit or close < trendMA.
    Short (side='both') mirrored: close < trendMA and RSI > 100-rsi_entry; exit RSI < 100-rsi_exit or close > trendMA.
    """
    c = df["close"].to_numpy(float)
    t = ta.ma(c, trend_len, trend_type)
    r = ta.rsi(c, rsi_len)
    with np.errstate(invalid="ignore"):
        ok = ~np.isnan(t) & ~np.isnan(r)
        le = ok & (c > t) & (r < rsi_entry)
        lx = ok & ((r > rsi_exit) | (c < t))
        se = ok & (c < t) & (r > 100 - rsi_entry)
        sx = ok & ((r < 100 - rsi_exit) | (c > t))
    z = np.zeros(len(c), bool)
    return {"long_entry": le, "long_exit": lx,
            "short_entry": se if side == "both" else z, "short_exit": sx if side == "both" else z}


def _chandelier(df, atr_len: int, atr_mult: float):
    """Chandelier ATR trailing stop levels, computed at bar CLOSE, used on the NEXT bar.

    longStop  = ta.highest(high, atr_len) - atr_mult * ta.atr(atr_len)
    shortStop = ta.lowest(low, atr_len)  + atr_mult * ta.atr(atr_len)
    Position-independent (classic Chandelier exit): the level trails the rolling extreme and can
    loosen if the extreme rolls out of the window or ATR expands.
    """
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    a = ta.atr(h, l, c, atr_len)
    return ta.highest(h, atr_len) - atr_mult * a, ta.lowest(l, atr_len) + atr_mult * a


def donchian(df, n_entry: int, n_exit: int, side: str = "long") -> dict:
    """Donchian channel breakout (closed-bar signals).

    Long entry: close > highest(high, n_entry)[1]  (breaks the PRIOR n_entry-bar high)
    Long exit:  close < lowest(low, n_exit)[1]
    Short (side='both') mirrored: entry close < lowest(low, n_entry)[1]; exit close > highest(high, n_exit)[1].
    """
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    up_e, dn_e = ta.shift(ta.highest(h, n_entry), 1), ta.shift(ta.lowest(l, n_entry), 1)
    up_x, dn_x = ta.shift(ta.highest(h, n_exit), 1), ta.shift(ta.lowest(l, n_exit), 1)
    with np.errstate(invalid="ignore"):
        le, lx = c > up_e, c < dn_x
        se, sx = c < dn_e, c > up_x
    z = np.zeros(len(c), bool)
    return {"long_entry": le, "long_exit": lx,
            "short_entry": se if side == "both" else z, "short_exit": sx if side == "both" else z}


def donchian_atr(df, n_entry: int, atr_len: int, atr_mult: float, side: str = "long") -> dict:
    """Donchian breakout entry; exit ONLY via Chandelier ATR trailing stop (intrabar, see backtest)
    or a reversal when side='both'. Entries require a valid stop level."""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    ls, ss = _chandelier(df, atr_len, atr_mult)
    up_e, dn_e = ta.shift(ta.highest(h, n_entry), 1), ta.shift(ta.lowest(l, n_entry), 1)
    with np.errstate(invalid="ignore"):
        le = (c > up_e) & ~np.isnan(ls)
        se = (c < dn_e) & ~np.isnan(ss)
    z = np.zeros(len(c), bool)
    return {"long_entry": le, "long_exit": z.copy(),
            "short_entry": se if side == "both" else z.copy(), "short_exit": z.copy(),
            "long_stop": ls, "short_stop": ss}


def ma_cross_atr(df, fast: int, slow: int, atr_len: int, atr_mult: float, ma_type: str = "ema",
                 side: str = "long") -> dict:
    """ma_cross entries/exits plus a Chandelier ATR trailing stop (whichever exits first)."""
    s = ma_cross(df, fast, slow, ma_type, side)
    ls, ss = _chandelier(df, atr_len, atr_mult)
    s["long_entry"] = s["long_entry"] & ~np.isnan(ls)
    s["short_entry"] = s["short_entry"] & ~np.isnan(ss)
    s["long_stop"], s["short_stop"] = ls, ss
    return s


def donchian_ratchet(df, n_entry: int, atr_len: int, atr_mult: float, side: str = "long") -> dict:
    """Donchian breakout entry + RATCHETING Chandelier stop: the base level is the Chandelier
    level; while in a trade the stop only tightens (engine stop_mode='ratchet')."""
    s = donchian_atr(df, n_entry, atr_len, atr_mult, side)
    s["stop_mode"] = "ratchet"
    return s


def bollinger_breakout(df, bb_len: int, bb_mult: float, side: str = "long") -> dict:
    """Bollinger breakout: long when close > SMA(n) + k*stdev(n); exit when close < SMA(n).
    Short mirrored (close < lower band; exit close > basis). Pine ta.stdev = population stdev."""
    c = df["close"].to_numpy(float)
    basis, sd = ta.sma(c, bb_len), ta.stdev(c, bb_len)
    up, dn = basis + bb_mult * sd, basis - bb_mult * sd
    with np.errstate(invalid="ignore"):
        le, lx, se, sx = c > up, c < basis, c < dn, c > basis
    z = np.zeros(len(c), bool)
    return {"long_entry": le, "long_exit": lx,
            "short_entry": se if side == "both" else z, "short_exit": sx if side == "both" else z}


def keltner_breakout(df, kc_len: int, kc_mult: float, side: str = "long") -> dict:
    """Keltner breakout: long when close > EMA(n) + k*ATR(n); exit when close < EMA(n). Shorts mirrored."""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    mid, a = ta.ema(c, kc_len), ta.atr(h, l, c, kc_len)
    up, dn = mid + kc_mult * a, mid - kc_mult * a
    with np.errstate(invalid="ignore"):
        le, lx, se, sx = c > up, c < mid, c < dn, c > mid
    z = np.zeros(len(c), bool)
    return {"long_entry": le, "long_exit": lx,
            "short_entry": se if side == "both" else z, "short_exit": sx if side == "both" else z}


def htf_donchian(df, n_entry: int, n_exit: int, htf_len: int, side: str = "long", htf: str = "4h") -> dict:
    """1h Donchian breakout filtered by a 4h trend from CLOSED 4h bars only.

    htfClose / htfEma = close and EMA(htf_len) of the previous fully closed 4h bar
    (engine.indicators.htf_prev_closed == Pine request.security(..., expr[1], lookahead_on)).
    Long entry: Donchian long entry and htfClose > htfEma. Long exit: Donchian exit or htfClose < htfEma.
    Short mirrored.
    """
    hc = ta.htf_prev_closed(df, htf, lambda x: x["close"].to_numpy(float))
    he = ta.htf_prev_closed(df, htf, lambda x: ta.ema(x["close"].to_numpy(float), htf_len))
    s = donchian(df, n_entry, n_exit, "both")
    with np.errstate(invalid="ignore"):
        bull, bear = hc > he, hc < he
    z = np.zeros(len(df), bool)
    out = {"long_entry": s["long_entry"] & bull, "long_exit": s["long_exit"] | bear,
           "short_entry": s["short_entry"] & bear, "short_exit": s["short_exit"] | bull}
    if side != "both":
        out["short_entry"], out["short_exit"] = z, z.copy()
    return out


def _htf_filter(df, sig, htf_len: int, side: str, htf: str = "4h") -> dict:
    """Gate a 1h signal set with the previous CLOSED 4h bar's close vs EMA(htf_len) (as htf_donchian):
    longs need bull and also exit on bear; shorts mirrored."""
    hc = ta.htf_prev_closed(df, htf, lambda x: x["close"].to_numpy(float))
    he = ta.htf_prev_closed(df, htf, lambda x: ta.ema(x["close"].to_numpy(float), htf_len))
    with np.errstate(invalid="ignore"):
        bull, bear = hc > he, hc < he
    out = dict(sig)
    out["long_entry"] = sig["long_entry"] & bull
    out["long_exit"] = sig["long_exit"] | bear
    if side == "both":
        out["short_entry"] = sig["short_entry"] & bear
        out["short_exit"] = sig["short_exit"] | bull
    return out


def htf_ma_cross_atr(df, fast: int, slow: int, atr_len: int, atr_mult: float, htf_len: int,
                     ma_type: str = "ema", side: str = "long") -> dict:
    """ma_cross_atr on 1h, gated by the closed-4h EMA trend filter."""
    return _htf_filter(df, ma_cross_atr(df, fast, slow, atr_len, atr_mult, ma_type, side), htf_len, side)


def htf_keltner(df, kc_len: int, kc_mult: float, htf_len: int, side: str = "long") -> dict:
    """keltner_breakout on 1h, gated by the closed-4h EMA trend filter."""
    return _htf_filter(df, keltner_breakout(df, kc_len, kc_mult, side), htf_len, side)


FAMILIES = {"ma_cross": ma_cross, "trend_pullback": trend_pullback, "donchian": donchian,
            "donchian_atr": donchian_atr, "ma_cross_atr": ma_cross_atr,
            "donchian_ratchet": donchian_ratchet, "bollinger_breakout": bollinger_breakout,
            "keltner_breakout": keltner_breakout, "htf_donchian": htf_donchian,
            "htf_ma_cross_atr": htf_ma_cross_atr, "htf_keltner": htf_keltner}
