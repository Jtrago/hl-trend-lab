"""Causal indicators that match Pine v6 `ta.*` semantics exactly.

Pine seeds ta.ema / ta.rma with the SMA of the first `length` values and returns na
before that; ta.rsi is built on ta.rma of gains/losses. Value at index t uses only
bars 0..t (no lookahead).
"""
from __future__ import annotations

import numpy as np


def sma(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if n <= len(x):
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def _seeded_recursive(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    # first index where a full window of non-nan values exists
    valid = ~np.isnan(x)
    start = None
    run = 0
    for i, v in enumerate(valid):
        run = run + 1 if v else 0
        if run >= n:
            start = i
            break
    if start is None:
        return out
    prev = x[start - n + 1:start + 1].mean()
    out[start] = prev
    for i in range(start + 1, len(x)):
        prev = alpha * x[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def ema(x, n: int):
    return _seeded_recursive(x, n, 2.0 / (n + 1))


def rma(x, n: int):
    return _seeded_recursive(x, n, 1.0 / n)


def ma(x, n: int, kind: str):
    return {"ema": ema, "sma": sma}[kind](x, n)


def rsi(close, n: int):
    close = np.asarray(close, float)
    ch = np.diff(close, prepend=np.nan)
    up = rma(np.where(np.isnan(ch), np.nan, np.maximum(ch, 0.0)), n)
    dn = rma(np.where(np.isnan(ch), np.nan, -np.minimum(ch, 0.0)), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + up / dn)
    r = np.where(dn == 0, 100.0, r)
    r = np.where((up == 0) & (dn != 0), 0.0, r)
    r[np.isnan(up) | np.isnan(dn)] = np.nan
    return r


def crossover(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, pb = np.roll(a, 1), np.roll(b, 1)
    out = (a > b) & (pa <= pb)
    out[0] = False
    return out & ~np.isnan(pa) & ~np.isnan(pb)


def crossunder(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, pb = np.roll(a, 1), np.roll(b, 1)
    out = (a < b) & (pa >= pb)
    out[0] = False
    return out & ~np.isnan(pa) & ~np.isnan(pb)


def shift(x, k: int = 1):
    """Pine history operator x[k]: value k bars ago (na for the first k bars)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def highest(x, n: int):
    """Pine ta.highest(x, n): max of the last n values INCLUDING the current bar."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if n <= len(x):
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = w.max(axis=1)
    return out


def lowest(x, n: int):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if n <= len(x):
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = w.min(axis=1)
    return out


def true_range(high, low, close):
    """Pine ta.tr(true): high-low on the first bar, else max(h-l, |h-c[1]|, |l-c[1]|)."""
    h, l, c = (np.asarray(a, float) for a in (high, low, close))
    pc = shift(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def atr(high, low, close, n: int):
    """Pine ta.atr(n) = ta.rma(ta.tr(true), n)."""
    return rma(true_range(high, low, close), n)


def stdev(x, n: int):
    """Pine ta.stdev(x, n) (biased / population stdev over the last n values)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if n <= len(x):
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = w.std(axis=1, ddof=0)
    return out


def htf_bucket_ids(index, rule: str = "4h"):
    """Integer bucket id per LTF bar (UTC-aligned HTF bars, e.g. 00/04/08... for 4h)."""
    import pandas as pd
    b = pd.DatetimeIndex(index).floor(rule)
    return b


def htf_prev_closed(df, rule: str, fn):
    """Value of an HTF indicator from the PREVIOUS fully closed HTF bar, aligned to LTF bars.

    HTF OHLC is resampled from the LTF bars themselves (all buckets that have data). `fn(htf_df)`
    returns an array on HTF bars. LTF bar t in bucket B receives fn(...)[B-1] — the HTF bar
    before the one containing t, i.e. the Pine idiom
        request.security(sym, tf, expr[1], lookahead=barmerge.lookahead_on)
    It never uses the bucket t belongs to, even on that bucket's last LTF bar (conservative).
    """
    import pandas as pd
    b = htf_bucket_ids(df.index, rule)
    g = df.groupby(b)
    htf = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last()})
    vals = pd.Series(np.asarray(fn(htf), float), index=htf.index).shift(1)
    return vals.reindex(b).to_numpy(float)


def htf_prev_closed_external(target_index, src_df, rule: str, fn):
    """Like htf_prev_closed, but the HTF series comes from ANOTHER symbol's bars (e.g. BTC daily
    for an ETH 4h chart). Target bar in bucket B (floor of its OPEN time) gets fn(...) of the
    previous existing source bucket — never bucket B itself (conservative, = Pine
    request.security("BINANCE:BTCUSDT", "D", expr[1], lookahead=barmerge.lookahead_on))."""
    import pandas as pd
    b = pd.DatetimeIndex(src_df.index).floor(rule)
    g = src_df.groupby(b)
    htf = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last()})
    vals = pd.Series(np.asarray(fn(htf), float), index=htf.index).shift(1)
    return vals.reindex(pd.DatetimeIndex(target_index).floor(rule)).to_numpy(float)


def realized_vol(close, n: int, bars_per_year: int):
    """Annualised stdev of the last n log returns, using bars <= t only."""
    c = np.asarray(close, float)
    lr = np.diff(np.log(c), prepend=np.nan)
    out = np.full(len(c), np.nan)
    if n < len(c):
        w = np.lib.stride_tricks.sliding_window_view(lr, n)
        out[n - 1:] = w.std(axis=1, ddof=1) * np.sqrt(bars_per_year)
    return out
