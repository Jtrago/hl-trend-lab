"""No-lookahead single-position backtester (paper research only).

Fill rule (documented in README):
  * All signals are computed from bar t's CLOSE (indicators use bars <= t only).
  * Orders decided at bar t close are filled at bar t+1 OPEN. A signal on the final
    bar has no next open and is never filled.
  * Per bar-close decision (mirrors the Pine export):
      1. if long  and (long_exit  or short_entry): close long
      2. if short and (short_exit or long_entry):  close short
      3. if (flat after step 1-2) and long_entry and not long_exit:  open long
         elif ... short_entry and not short_exit: open short
    So a reversal (close + open) fills both legs at the same next open.
Intrabar stop rule (families that emit `long_stop` / `short_stop` level arrays):
  * The stop level for bar t is the level computed at the CLOSE of bar t-1 (no lookahead).
  * It is active from the fill bar onward, including the entry bar itself (pessimistic:
    after the entry fills at the open, the same bar's low/high can stop it out).
  * Long: stopped if low[t] <= level; fill raw price = min(open[t], level) (a gap through the
    stop fills at the worse open). Short: stopped if high[t] >= level; fill = max(open[t], level).
  * Stops are evaluated after the open fills of bar t and before bar t's close decision; a
    stopped-out position is flat at that close and may re-enter (at the next open) if signalled.
  * Stop fills pay the same adverse slippage + fee as every other fill. This mirrors Pine's
    strategy.exit(stop=...) semantics (stop orders fill at the stop or at a worse gap open).
Ratcheting stop (sig["stop_mode"] == "ratchet"): long_stop/short_stop are BASE levels. At each
bar close the level for the next bar is
    long:  max(previous trail, base[t]) if long at this close, else base[t]  (fresh trail)
    short: min(previous trail, base[t]) if short at this close, else base[t]
so while a position is open its stop only tightens; a new position starts from the base level
of its signal bar. Touch/gap/entry-bar rules are identical to the plain stop above.
Position sizing: by default 100% of equity. If sig["size"] (array of equity fractions) is given,
the fraction f = size[signal_bar] (known at the signal bar's close) is fixed for the whole trade:
deployed D = equity*f, notional N = D/(1+fee) (entry fee out of D), cash = equity - D stays idle
(0% interest). Exit equity = cash + N + pnl - exit_fee. f = 1 reproduces the default exactly.
Funding (perp carry): if sig["funding"] (per-bar signed rate, fraction of notional; positive =>
longs pay) is given, at each bar close while a position is open the engine charges
units * close[t] * funding[t] * direction to idle cash (shorts receive when positive). Exactly the
bar to which a funding event is mapped is defined in engine/funding.py.
Costs on EVERY fill: price slippage (buy at open*(1+slip), sell at open*(1-slip))
plus fee_rate * notional. Sizing: 100% of equity, 1x, compounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_FEE = 0.00045   # Hyperliquid perps Tier-0 taker 0.045%/side
DEFAULT_SLIP = 0.0002   # 2 bp adverse slippage per fill


@dataclass
class Costs:
    fee: float = DEFAULT_FEE
    slip: float = DEFAULT_SLIP


@dataclass
class Trade:
    direction: int          # +1 long, -1 short
    signal_bar: int         # bar whose close generated the entry
    entry_bar: int          # fill bar (signal_bar + 1)
    entry_raw: float        # bar open before slippage
    entry_px: float         # filled after slippage
    exit_signal_bar: int | None = None
    exit_bar: int | None = None
    exit_raw: float | None = None
    exit_px: float | None = None
    ret: float | None = None          # net return on equity for the trade (fees+slip)
    gross_ret: float | None = None    # raw open-to-open return, no costs
    forced: bool = False              # closed at segment end (not a strategy exit)
    exit_reason: str | None = None    # "signal" | "stop" | "forced"
    size: float = 1.0                 # equity fraction deployed
    funding_paid: float = 0.0         # funding paid (equity units, + = cost)


@dataclass
class Result:
    trades: list = field(default_factory=list)
    equity: np.ndarray | None = None      # mark-to-market at each bar close
    index: pd.Index | None = None
    start: int = 0

    @property
    def closed(self):
        return [t for t in self.trades if t.exit_bar is not None]


def run(df: pd.DataFrame, sig: dict, costs: Costs = Costs(), start: int = 0, end: int | None = None,
        close_at_end: bool = True, init_equity: float = 1.0) -> Result:
    """Simulate bars [start, end). Indicators/signals may use earlier bars (causal warmup).

    Position starts flat at `start`. Orders decided at the close of bar end-1 cannot fill
    inside the segment. If close_at_end, any open position is closed at the last bar's
    close (with costs) and flagged forced=True.
    """
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df) if end is None else end
    le, lx = sig["long_entry"], sig["long_exit"]
    se, sx = sig["short_entry"], sig["short_exit"]
    fee, slip = costs.fee, costs.slip
    ls, ss = sig.get("long_stop"), sig.get("short_stop")
    h = df["high"].to_numpy(float) if ls is not None or ss is not None else None
    lo = df["low"].to_numpy(float) if h is not None else None
    ratchet = sig.get("stop_mode") == "ratchet"
    act_l = act_s = np.nan          # stop levels active for the current bar (set at previous close)

    eq = init_equity
    pos = 0
    units = 0.0
    notional0 = 0.0
    cash = 0.0
    size = sig.get("size")
    funding = sig.get("funding")
    cur: Trade | None = None
    trades: list[Trade] = []
    equity = np.full(n - start, np.nan)
    pending: list[tuple] = []   # orders for next open: ("close",) / ("open", dir, signal_bar)

    def close_pos(bar, raw_px, signal_bar, forced=False, reason="signal"):
        nonlocal eq, pos, units, cur
        px = raw_px * (1 - slip) if pos == 1 else raw_px * (1 + slip)
        pnl = units * (px - cur.entry_px) * pos
        exit_fee = units * px * fee
        eq_before = cur._eq_before
        eq = cash + notional0 + pnl - exit_fee
        cur.exit_signal_bar, cur.exit_bar, cur.exit_raw, cur.exit_px = signal_bar, bar, raw_px, px
        cur.ret = eq / eq_before - 1
        cur.gross_ret = (raw_px / cur.entry_raw - 1) * pos
        cur.forced = forced
        cur.exit_reason = "forced" if forced else reason
        pos, units, cur = 0, 0.0, None

    def open_pos(bar, raw_px, d, signal_bar):
        nonlocal eq, pos, units, cur, notional0, cash
        px = raw_px * (1 + slip) if d == 1 else raw_px * (1 - slip)
        f = 1.0 if size is None else float(size[signal_bar])
        if not np.isfinite(f) or f <= 0:
            return                           # no size info -> no trade
        cash = eq * (1 - f)
        notional0 = eq * f / (1 + fee)      # entry fee paid out of the deployed equity
        units = notional0 / px
        cur = Trade(d, signal_bar, bar, raw_px, px)
        cur._eq_before = eq
        cur.size = f
        trades.append(cur)
        pos = d

    for t in range(start, n):
        # 1) fill orders decided at previous close, at this bar's open
        for order in pending:
            if order[0] == "close" and pos != 0:
                close_pos(t, o[t], order[1])
            elif order[0] == "open" and pos == 0:
                open_pos(t, o[t], order[1], order[2])
        pending = []
        # 1b) intrabar stop, level fixed at previous close
        if pos == 1 and ls is not None and not np.isnan(act_l) and lo[t] <= act_l:
            close_pos(t, min(o[t], act_l), t - 1, reason="stop")
        elif pos == -1 and ss is not None and not np.isnan(act_s) and h[t] >= act_s:
            close_pos(t, max(o[t], act_s), t - 1, reason="stop")
        # 1c) stop levels for the NEXT bar, fixed at this bar's close
        if ls is not None:
            act_l = (ls[t] if np.isnan(act_l) else max(act_l, ls[t])) if (ratchet and pos == 1) else ls[t]
        if ss is not None:
            act_s = (ss[t] if np.isnan(act_s) else min(act_s, ss[t])) if (ratchet and pos == -1) else ss[t]
        # 1d) funding accrual for holding through this bar
        if funding is not None and pos != 0 and funding[t] != 0 and np.isfinite(funding[t]):
            fc = units * c[t] * funding[t] * pos
            cash -= fc
            cur.funding_paid += fc
        # 2) mark to market at close
        equity[t - start] = eq if pos == 0 else cash + notional0 + units * (c[t] - cur.entry_px) * pos
        # 3) decide at close of bar t (only bar <= t information)
        if t == n - 1:
            break
        p = pos
        if p == 1 and (lx[t] or se[t]):
            pending.append(("close", t)); p = 0
        elif p == -1 and (sx[t] or le[t]):
            pending.append(("close", t)); p = 0
        if p == 0:
            if le[t] and not lx[t]:
                pending.append(("open", 1, t))
            elif se[t] and not sx[t]:
                pending.append(("open", -1, t))
    if close_at_end and pos != 0:
        close_pos(n - 1, c[n - 1], n - 1, forced=True)
        equity[-1] = eq
    for tr in trades:
        tr.__dict__.pop("_eq_before", None)
    return Result(trades=trades, equity=equity, index=df.index[start:n], start=start)
