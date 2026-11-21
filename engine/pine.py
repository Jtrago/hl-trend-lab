"""Pine Script v6 exporter. Emitted scripts mirror engine.strategies + engine.backtest:

* same indicator definitions (ta.ema / ta.sma / ta.rsi, Pine seeding == engine.indicators)
* signals evaluated on bar close (calc_on_every_tick=false, process_orders_on_close=false)
  -> market orders fill at the NEXT bar's open, exactly like the Python fill rule
* same per-bar decision order (closeLong/closeShort, then goLong/goShort)
* costs: Pine `slippage` is in ticks, so fee+slip are folded into commission percent
  (commission_value = (fee+slip)*100). Net returns match Python to O(cost^2).
* margin_long/short = 0 disables TradingView margin calls (Python models 1x, no liquidation).

Everything above the `// @@display` marker is the trading logic and is parsed and executed
by engine.pine_sim in the parity test, so keep that section to the supported subset:
`x = input.*(...)`, `x = <expr>`, and `if <cond>` + one indented strategy.* call.
"""
from __future__ import annotations

from .backtest import Costs


def _header(title: str, costs: Costs) -> str:
    comm = round((costs.fee + costs.slip) * 100, 6)
    return (
        "//@version=6\n"
        f"// HL Trend Lab export — PAPER RESEARCH ONLY. Not financial advice.\n"
        f"// Fill rule: signals on bar close, market fill at next bar open.\n"
        f"// Costs: fee {costs.fee*100:.4f}% + slip {costs.slip*100:.4f}% per side folded into commission.\n"
        f'strategy("{title}", overlay=true, initial_capital=10000, '
        "default_qty_type=strategy.percent_of_equity, default_qty_value=100, "
        f"commission_type=strategy.commission.percent, commission_value={comm}, slippage=0, "
        "pyramiding=1, margin_long=0, margin_short=0, process_orders_on_close=false, "
        "calc_on_every_tick=false, calc_on_order_fills=false, fill_orders_on_standard_ohlc=true)\n"
    )


_ORDERS = """
// --- orders (same decision order as engine/backtest.py) ---
closeLong = longExit or shortEntry
closeShort = shortExit or longEntry
goLong = longEntry and not longExit
goShort = shortEntry and not shortExit and not goLong
if closeLong
    strategy.close("L")
if closeShort
    strategy.close("S")
if goLong
    strategy.entry("L", strategy.long)
if goShort
    strategy.entry("S", strategy.short)
"""


def _b(v: bool) -> str:
    return "true" if v else "false"


def export_ma_cross(fast: int, slow: int, ma_type: str = "ema", side: str = "long",
                    costs: Costs = Costs(), title: str | None = None) -> str:
    title = title or f"HL Trend Lab ma_cross {ma_type.upper()} {fast}/{slow} {side}"
    f = {"ema": "ta.ema", "sma": "ta.sma"}[ma_type]
    body = f"""
fastLen = input.int({fast}, "Fast length", minval=1)
slowLen = input.int({slow}, "Slow length", minval=1)
allowShort = input.bool({_b(side == "both")}, "Allow shorts")
fastMa = {f}(close, fastLen)
slowMa = {f}(close, slowLen)
crossUp = ta.crossover(fastMa, slowMa)
crossDn = ta.crossunder(fastMa, slowMa)
longEntry = crossUp
longExit = crossDn
shortEntry = allowShort and crossDn
shortExit = allowShort and crossUp
"""
    disp = """
// @@display (not part of the trading logic)
plot(fastMa, "Fast MA", color=color.teal)
plot(slowMa, "Slow MA", color=color.orange)
"""
    return _header(title, costs) + body + _ORDERS + disp


def export_trend_pullback(trend_len: int, rsi_len: int, rsi_entry: float, rsi_exit: float,
                          trend_type: str = "ema", side: str = "long", costs: Costs = Costs(),
                          title: str | None = None) -> str:
    title = title or (f"HL Trend Lab trend_pullback {trend_type.upper()}{trend_len} "
                      f"RSI{rsi_len} {rsi_entry}/{rsi_exit} {side}")
    f = {"ema": "ta.ema", "sma": "ta.sma"}[trend_type]
    body = f"""
trendLen = input.int({trend_len}, "Trend MA length", minval=1)
rsiLen = input.int({rsi_len}, "RSI length", minval=1)
rsiEntry = input.float({float(rsi_entry)}, "RSI pullback entry (long below)")
rsiExit = input.float({float(rsi_exit)}, "RSI exit (long above)")
allowShort = input.bool({_b(side == "both")}, "Allow shorts")
trendMa = {f}(close, trendLen)
r = ta.rsi(close, rsiLen)
longEntry = close > trendMa and r < rsiEntry
longExit = r > rsiExit or close < trendMa
shortEntry = allowShort and close < trendMa and r > 100 - rsiEntry
shortExit = allowShort and (r < 100 - rsiExit or close > trendMa)
"""
    disp = """
// @@display (not part of the trading logic)
plot(trendMa, "Trend MA", color=color.orange)
"""
    return _header(title, costs) + body + _ORDERS + disp


_STOPS = """
// --- Chandelier ATR trailing stop: level from this bar's close, active from the next bar
// (incl. the entry bar), fills at the stop or a worse gap open (= engine intrabar rule) ---
if not na(longStop)
    strategy.exit("LX", "L", stop=longStop)
if not na(shortStop)
    strategy.exit("SX", "S", stop=shortStop)
"""


def _stop_lines(atr_len, atr_mult):
    return f"""atrLen = input.int({atr_len}, "ATR / Chandelier length", minval=1)
atrMult = input.float({float(atr_mult)}, "ATR multiple")
atrV = ta.atr(atrLen)
longStop = ta.highest(high, atrLen) - atrMult * atrV
shortStop = ta.lowest(low, atrLen) + atrMult * atrV
"""


def export_donchian(n_entry: int, n_exit: int, side: str = "long", costs: Costs = Costs(),
                    title: str | None = None) -> str:
    title = title or f"HL Trend Lab donchian {n_entry}/{n_exit} {side}"
    body = f"""
nEntry = input.int({n_entry}, "Entry channel length", minval=1)
nExit = input.int({n_exit}, "Exit channel length", minval=1)
allowShort = input.bool({_b(side == "both")}, "Allow shorts")
upE = ta.highest(high, nEntry)[1]
dnE = ta.lowest(low, nEntry)[1]
upX = ta.highest(high, nExit)[1]
dnX = ta.lowest(low, nExit)[1]
longEntry = close > upE
longExit = close < dnX
shortEntry = allowShort and close < dnE
shortExit = allowShort and close > upX
"""
    disp = """
// @@display (not part of the trading logic)
plot(upE, "Entry high", color=color.green)
plot(dnE, "Entry low", color=color.red)
"""
    return _header(title, costs) + body + _ORDERS + disp


def export_donchian_atr(n_entry: int, atr_len: int, atr_mult: float, side: str = "long",
                        costs: Costs = Costs(), title: str | None = None) -> str:
    title = title or f"HL Trend Lab donchian_atr {n_entry} ATR{atr_len}x{atr_mult} {side}"
    body = f"""
nEntry = input.int({n_entry}, "Entry channel length", minval=1)
allowShort = input.bool({_b(side == "both")}, "Allow shorts")
""" + _stop_lines(atr_len, atr_mult) + """upE = ta.highest(high, nEntry)[1]
dnE = ta.lowest(low, nEntry)[1]
longEntry = close > upE and not na(longStop)
longExit = false
shortEntry = allowShort and close < dnE and not na(shortStop)
shortExit = false
"""
    disp = """
// @@display (not part of the trading logic)
plot(strategy.position_size > 0 ? longStop : na, "Long stop", color=color.red, style=plot.style_linebr)
plot(strategy.position_size < 0 ? shortStop : na, "Short stop", color=color.green, style=plot.style_linebr)
"""
    return _header(title, costs) + body + _ORDERS + _STOPS + disp


def export_ma_cross_atr(fast: int, slow: int, atr_len: int, atr_mult: float, ma_type: str = "ema",
                        side: str = "long", costs: Costs = Costs(), title: str | None = None) -> str:
    title = title or f"HL Trend Lab ma_cross_atr {ma_type.upper()} {fast}/{slow} ATR{atr_len}x{atr_mult} {side}"
    f = {"ema": "ta.ema", "sma": "ta.sma"}[ma_type]
    body = f"""
fastLen = input.int({fast}, "Fast length", minval=1)
slowLen = input.int({slow}, "Slow length", minval=1)
allowShort = input.bool({_b(side == "both")}, "Allow shorts")
""" + _stop_lines(atr_len, atr_mult) + f"""fastMa = {f}(close, fastLen)
slowMa = {f}(close, slowLen)
crossUp = ta.crossover(fastMa, slowMa)
crossDn = ta.crossunder(fastMa, slowMa)
longEntry = crossUp and not na(longStop)
longExit = crossDn
shortEntry = allowShort and crossDn and not na(shortStop)
shortExit = allowShort and crossUp
"""
    disp = """
// @@display (not part of the trading logic)
plot(fastMa, "Fast MA", color=color.teal)
plot(slowMa, "Slow MA", color=color.orange)
plot(strategy.position_size > 0 ? longStop : na, "Long stop", color=color.red, style=plot.style_linebr)
"""
    return _header(title, costs) + body + _ORDERS + _STOPS + disp


_RATCHET = """
// @@stateful — ratcheting trail: while in a trade the stop only tightens (= engine stop_mode='ratchet')
var float trailL = na
var float trailS = na
if strategy.position_size <= 0
    trailL := longStop
else
    trailL := math.max(trailL, longStop)
if strategy.position_size >= 0
    trailS := shortStop
else
    trailS := math.min(trailS, shortStop)
if not na(trailL)
    strategy.exit("LX", "L", stop=trailL)
if not na(trailS)
    strategy.exit("SX", "S", stop=trailS)
"""


def export_donchian_ratchet(n_entry: int, atr_len: int, atr_mult: float, side: str = "long",
                            costs: Costs = Costs(), title: str | None = None) -> str:
    title = title or f"HL Trend Lab donchian_ratchet {n_entry} ATR{atr_len}x{atr_mult} {side}"
    body = f"""
nEntry = input.int({n_entry}, "Entry channel length", minval=1)
allowShort = input.bool({_b(side == "both")}, "Allow shorts")
""" + _stop_lines(atr_len, atr_mult) + """upE = ta.highest(high, nEntry)[1]
dnE = ta.lowest(low, nEntry)[1]
longEntry = close > upE and not na(longStop)
longExit = false
shortEntry = allowShort and close < dnE and not na(shortStop)
shortExit = false
"""
    disp = """
// @@display (not part of the trading logic)
plot(strategy.position_size > 0 ? trailL : na, "Long trail", color=color.red, style=plot.style_linebr)
plot(strategy.position_size < 0 ? trailS : na, "Short trail", color=color.green, style=plot.style_linebr)
"""
    return _header(title, costs) + body + _ORDERS + _RATCHET + disp


def export_bollinger_breakout(bb_len: int, bb_mult: float, side: str = "long", costs: Costs = Costs(),
                              title: str | None = None) -> str:
    title = title or f"HL Trend Lab bollinger_breakout {bb_len}x{bb_mult} {side}"
    body = f"""
bbLen = input.int({bb_len}, "BB length", minval=1)
bbMult = input.float({float(bb_mult)}, "BB stdev multiple")
allowShort = input.bool({_b(side == "both")}, "Allow shorts")
basis = ta.sma(close, bbLen)
dev = bbMult * ta.stdev(close, bbLen)
upper = basis + dev
lower = basis - dev
longEntry = close > upper
longExit = close < basis
shortEntry = allowShort and close < lower
shortExit = allowShort and close > basis
"""
    disp = """
// @@display (not part of the trading logic)
plot(basis, "Basis", color=color.orange)
plot(upper, "Upper", color=color.teal)
plot(lower, "Lower", color=color.teal)
"""
    return _header(title, costs) + body + _ORDERS + disp


def export_keltner_breakout(kc_len: int, kc_mult: float, side: str = "long", costs: Costs = Costs(),
                            title: str | None = None) -> str:
    title = title or f"HL Trend Lab keltner_breakout {kc_len}x{kc_mult} {side}"
    body = f"""
kcLen = input.int({kc_len}, "Keltner length", minval=1)
kcMult = input.float({float(kc_mult)}, "ATR multiple")
allowShort = input.bool({_b(side == "both")}, "Allow shorts")
mid = ta.ema(close, kcLen)
rng = kcMult * ta.atr(kcLen)
upper = mid + rng
lower = mid - rng
longEntry = close > upper
longExit = close < mid
shortEntry = allowShort and close < lower
shortExit = allowShort and close > mid
"""
    disp = """
// @@display (not part of the trading logic)
plot(mid, "Mid", color=color.orange)
plot(upper, "Upper", color=color.teal)
plot(lower, "Lower", color=color.teal)
"""
    return _header(title, costs) + body + _ORDERS + disp


def export_htf_donchian(n_entry: int, n_exit: int, htf_len: int, side: str = "long", htf: str = "4h",
                        costs: Costs = Costs(), title: str | None = None) -> str:
    assert htf == "4h"
    title = title or f"HL Trend Lab htf_donchian {n_entry}/{n_exit} 4hEMA{htf_len} {side}"
    body = f"""
// Run on a 1h chart. 4h values come from the PREVIOUS closed 4h bar only (expr[1] + lookahead_on:
// the standard non-repainting idiom) = engine.indicators.htf_prev_closed.
nEntry = input.int({n_entry}, "Entry channel length", minval=1)
nExit = input.int({n_exit}, "Exit channel length", minval=1)
htfLen = input.int({htf_len}, "4h EMA length", minval=1)
allowShort = input.bool({_b(side == "both")}, "Allow shorts")
htfClose = request.security(syminfo.tickerid, "240", close[1], lookahead=barmerge.lookahead_on)
htfEma = request.security(syminfo.tickerid, "240", ta.ema(close, htfLen)[1], lookahead=barmerge.lookahead_on)
bull = htfClose > htfEma
bear = htfClose < htfEma
upE = ta.highest(high, nEntry)[1]
dnE = ta.lowest(low, nEntry)[1]
upX = ta.highest(high, nExit)[1]
dnX = ta.lowest(low, nExit)[1]
longEntry = close > upE and bull
longExit = close < dnX or bear
shortEntry = allowShort and close < dnE and bear
shortExit = allowShort and (close > upX or bull)
"""
    disp = """
// @@display (not part of the trading logic)
plot(htfEma, "4h EMA (prev closed bar)", color=color.orange)
"""
    return _header(title, costs) + body + _ORDERS + disp


_HTF_LINES = """htfLen = input.int({htf_len}, "4h EMA length", minval=1)
// 4h values from the PREVIOUS closed 4h bar only (non-repainting idiom) = engine.indicators.htf_prev_closed
htfClose = request.security(syminfo.tickerid, "240", close[1], lookahead=barmerge.lookahead_on)
htfEma = request.security(syminfo.tickerid, "240", ta.ema(close, htfLen)[1], lookahead=barmerge.lookahead_on)
bull = htfClose > htfEma
bear = htfClose < htfEma
longEntry := longEntry and bull
longExit := longExit or bear
"""
_HTF_SHORT = """shortEntry := shortEntry and bear
shortExit := shortExit or bull
"""


def _htf_overlay(script, htf_len, side):
    lines = "\n// --- closed-4h trend filter (run on a 1h chart) ---\n" + _HTF_LINES.format(htf_len=htf_len)
    if side == "both":
        lines += _HTF_SHORT
    return script.replace("\n// --- orders", lines + "\n// --- orders", 1)


def export_htf_ma_cross_atr(fast, slow, atr_len, atr_mult, htf_len, ma_type="ema", side="long",
                            costs: Costs = Costs(), title=None):
    title = title or f"HL Trend Lab htf_ma_cross_atr {ma_type.upper()} {fast}/{slow} ATR{atr_len}x{atr_mult} 4hEMA{htf_len} {side}"
    return _htf_overlay(export_ma_cross_atr(fast, slow, atr_len, atr_mult, ma_type, side, costs, title), htf_len, side)


def export_htf_keltner(kc_len, kc_mult, htf_len, side="long", costs: Costs = Costs(), title=None):
    title = title or f"HL Trend Lab htf_keltner {kc_len}x{kc_mult} 4hEMA{htf_len} {side}"
    return _htf_overlay(export_keltner_breakout(kc_len, kc_mult, side, costs, title), htf_len, side)


EXPORTERS = {"htf_ma_cross_atr": export_htf_ma_cross_atr, "htf_keltner": export_htf_keltner,
             "donchian_ratchet": export_donchian_ratchet, "bollinger_breakout": export_bollinger_breakout,
             "keltner_breakout": export_keltner_breakout, "htf_donchian": export_htf_donchian,
             "ma_cross": export_ma_cross, "trend_pullback": export_trend_pullback,
             "donchian": export_donchian, "donchian_atr": export_donchian_atr,
             "ma_cross_atr": export_ma_cross_atr}


def _btc_overlay(script: str, mode: str, btc_len: int = 200) -> str:
    """Inject the BTC trend-phase filter (engine.overlays) before the orders section."""
    lines = f"""
// --- BTC trend-phase filter ({mode}): previous CLOSED UTC daily bar of BINANCE:BTCUSDT only ---
btcLen = input.int({btc_len}, "BTC daily SMA length", minval=1)
btcClose = request.security("BINANCE:BTCUSDT", "D", close[1], lookahead=barmerge.lookahead_on)
btcSma = request.security("BINANCE:BTCUSDT", "D", ta.sma(close, btcLen)[1], lookahead=barmerge.lookahead_on)
btcBull = btcClose > btcSma
longEntry := longEntry and btcBull
"""
    if mode == "flat":
        lines += "longExit := longExit or not btcBull\n"
    return script.replace("\n// --- orders", lines + "\n// --- orders", 1)


def export(family: str, params: dict, costs: Costs = Costs(), title: str | None = None) -> str:
    if "__btc" in family:
        base, mode = family.split("__btc")
        p = dict(params)
        bl = p.pop("_btc_len", 200)
        t = title or f"HL Trend Lab {base} +BTC{mode}"
        return _btc_overlay(EXPORTERS[base](**p, costs=costs, title=t), mode, bl)
    return EXPORTERS[family](**params, costs=costs, title=title)
