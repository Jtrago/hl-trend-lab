"""Minimal executor for the Pine v6 subset emitted by engine.pine (parity testing only).

Deliberately independent of engine.indicators/backtest: ta.* are written in the
bar-by-bar style of the Pine reference docs and the broker follows Pine strategy
semantics (orders queued at bar close, filled at next open, commission % per side,
pyramiding=1, strategy.entry reverses an opposite position).
Pine `na` is NaN; comparisons with na are false, `and`/`or` treat na as false.
"""
from __future__ import annotations

import ast
import math
import re

import numpy as np


# ---- Pine-reference ta.* (bar-by-bar) ----
def p_sma(src, n):
    out = [math.nan] * len(src)
    for i in range(n - 1, len(src)):
        w = src[i - n + 1:i + 1]
        out[i] = math.nan if any(math.isnan(v) for v in w) else sum(w) / n
    return out


def _p_recursive(src, n, alpha):
    sm = p_sma(src, n)
    out = [math.nan] * len(src)
    for i in range(len(src)):
        prev = out[i - 1] if i > 0 else math.nan
        out[i] = sm[i] if math.isnan(prev) else alpha * src[i] + (1 - alpha) * prev
    return out


def p_ema(src, n):
    return _p_recursive(src, n, 2 / (n + 1))


def p_rma(src, n):
    return _p_recursive(src, n, 1 / n)


def p_rsi(src, n):
    ch = [math.nan] + [src[i] - src[i - 1] for i in range(1, len(src))]
    up = p_rma([max(c, 0.0) if not math.isnan(c) else math.nan for c in ch], n)
    dn = p_rma([-min(c, 0.0) if not math.isnan(c) else math.nan for c in ch], n)
    out = []
    for u, d in zip(up, dn):
        if math.isnan(u) or math.isnan(d):
            out.append(math.nan)
        elif d == 0:
            out.append(100.0)
        elif u == 0:
            out.append(0.0)
        else:
            out.append(100 - 100 / (1 + u / d))
    return out


def p_cross(a, b, up=True):
    out = [False] * len(a)
    for i in range(1, len(a)):
        if any(math.isnan(v) for v in (a[i], b[i], a[i - 1], b[i - 1])):
            continue
        out[i] = (a[i] > b[i] and a[i - 1] <= b[i - 1]) if up else (a[i] < b[i] and a[i - 1] >= b[i - 1])
    return out


def p_highest(src, n):
    out = [math.nan] * len(src)
    for i in range(n - 1, len(src)):
        out[i] = max(src[i - n + 1:i + 1])
    return out


def p_lowest(src, n):
    out = [math.nan] * len(src)
    for i in range(n - 1, len(src)):
        out[i] = min(src[i - n + 1:i + 1])
    return out


def p_tr(h, l, c):
    out = []
    for i in range(len(h)):
        if i == 0:
            out.append(h[i] - l[i])          # ta.tr(true): na(close[1]) -> high - low
        else:
            out.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return out


def p_stdev(src, n):
    out = [math.nan] * len(src)
    for i in range(n - 1, len(src)):
        w = src[i - n + 1:i + 1]
        mu = sum(w) / n
        out[i] = math.sqrt(sum((v - mu) ** 2 for v in w) / n)   # Pine ta.stdev (biased)
    return out


TA = {"stdev": p_stdev, "highest": p_highest, "lowest": p_lowest, "ema": p_ema, "sma": p_sma, "rsi": p_rsi, "rma": p_rma,
      "crossover": lambda a, b: p_cross(a, b, True), "crossunder": lambda a, b: p_cross(a, b, False)}


# ---- expression evaluation (vectorised over bars) ----
class _Eval:
    def __init__(self, env, n):
        self.env, self.n = env, n

    def series(self, v):
        if isinstance(v, (list, np.ndarray)):
            return np.asarray(v, dtype=float if not isinstance(v[0], (bool, np.bool_)) else bool)
        return np.full(self.n, v)

    def ev(self, node):
        if isinstance(node, ast.Expression):
            return self.ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.env[node.id]
        if isinstance(node, ast.BoolOp):
            vals = [np.asarray(self.series(self.ev(v)), bool) for v in node.values]
            acc = vals[0]
            for v in vals[1:]:
                acc = (acc & v) if isinstance(node.op, ast.And) else (acc | v)
            return acc
        if isinstance(node, ast.UnaryOp):
            v = self.ev(node.operand)
            if isinstance(node.op, ast.Not):
                return ~np.asarray(self.series(v), bool)
            if isinstance(node.op, ast.USub):
                return -v
        if isinstance(node, ast.BinOp):
            a, b = self.ev(node.left), self.ev(node.right)
            a = np.asarray(a, float) if isinstance(a, (list, np.ndarray)) else a
            b = np.asarray(b, float) if isinstance(b, (list, np.ndarray)) else b
            ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}
            return ops[type(node.op)](a, b)
        if isinstance(node, ast.Compare):
            a = np.asarray(self.series(self.ev(node.left)), float)
            b = np.asarray(self.series(self.ev(node.comparators[0])), float)
            ops = {ast.Gt: np.greater, ast.Lt: np.less, ast.GtE: np.greater_equal,
                   ast.LtE: np.less_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}
            with np.errstate(invalid="ignore"):
                r = ops[type(node.ops[0])](a, b)
            return r & ~np.isnan(a) & ~np.isnan(b)   # na comparisons are false
        if isinstance(node, ast.Subscript):          # history operator x[k]
            v = np.asarray(self.series(self.ev(node.value)), float)
            k = node.slice.value
            out = np.full(self.n, np.nan)
            out[k:] = v[:self.n - k]
            return out
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "na":
            return np.isnan(np.asarray(self.series(self.ev(node.args[0])), float))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.value.id == "request" and node.func.attr == "security"):
            return self._security(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            mod, fn = node.func.value.id, node.func.attr
            if mod == "ta" and fn == "atr":
                n_ = self.ev(node.args[0])
                e = self.env
                tr = p_tr(list(e["high"]), list(e["low"]), list(e["close"]))
                return np.asarray(p_rma(tr, n_))
            if mod == "ta":
                args = [self.ev(a) for a in node.args]
                args = [list(np.asarray(a, float)) if isinstance(a, (list, np.ndarray)) else a for a in args]
                return np.asarray(TA[fn](*args))
        raise ValueError(f"unsupported Pine construct: {ast.dump(node)}")


    def _security(self, node):
        """request.security(syminfo.tickerid, "240", expr, lookahead=barmerge.lookahead_on).

        Builds HTF bars from the chart bars (UTC-aligned buckets), evaluates expr bar-by-bar on
        them, and with lookahead_on gives every chart bar the value of the HTF bar it belongs to
        (so expr must use [1] to stay non-repainting; asserted below)."""
        import pandas as pd
        tf = node.args[1].value
        rule = {"240": "4h", "D": "1D"}[tf]
        kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
        assert kw.get("lookahead") == "barmerge.lookahead_on"
        assert isinstance(node.args[2], ast.Subscript), "HTF expr must be offset with [1]"
        sym = node.args[0]
        if isinstance(sym, ast.Constant):                     # another symbol
            src = self.env["_externals"][sym.value]
            src_idx = pd.DatetimeIndex(src.index)
            cols = {k: src[k].to_numpy(float) for k in ("open", "high", "low", "close")}
        else:                                                  # syminfo.tickerid = chart symbol
            src_idx = self.env["_index"]
            cols = {k: self.env[k] for k in ("open", "high", "low", "close")}
        sb = src_idx.floor(rule)
        keys, inv = np.unique(sb.asi8, return_inverse=True)
        htf = {}
        for col, agg in (("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")):
            htf[col] = getattr(pd.Series(cols[col]).groupby(inv), agg)().to_numpy(float)
        sub_env = dict(self.env)
        sub_env.update(htf)
        vals = np.asarray(_Eval(sub_env, len(keys)).series(_Eval(sub_env, len(keys)).ev(node.args[2])), float)
        # chart bar in bucket B gets the source bucket-B value (expr[1] already refers to B-1)
        lookup = dict(zip(keys.tolist(), vals.tolist()))
        cb = self.env["_index"].floor(rule).asi8
        return np.array([lookup.get(k, np.nan) for k in cb], float)


def _pyexpr(s: str) -> str:
    return re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", s))


def parse_header(script: str) -> dict:
    m = re.search(r"commission_value=([0-9.]+)", script)
    return {"commission_pct": float(m.group(1)) if m else 0.0,
            "process_orders_on_close": "process_orders_on_close=true" in script}


def run(script: str, df, externals: dict | None = None) -> dict:
    """Execute the logic section of an exported script on OHLC `df`. Returns trades.
    `externals` maps a ticker string (e.g. "BINANCE:BTCUSDT") to that symbol's OHLC bars for
    request.security calls on other symbols."""
    hdr = parse_header(script)
    assert not hdr["process_orders_on_close"], "parity assumes next-open fills"
    assert script.startswith("//@version=6")
    n = len(df)
    env = {k: df[k].to_numpy(float) for k in ["open", "high", "low", "close"]}
    env["_index"] = df.index
    env["_externals"] = externals or {}
    ev = _Eval(env, n)
    logic_text = script.split("// @@display")[0]
    stateful_text = ""
    if "// @@stateful" in logic_text:
        logic_text, stateful_text = logic_text.split("// @@stateful", 1)
        stateful_text = stateful_text.split("\n", 1)[1]      # drop the rest of the marker line
    logic = logic_text.splitlines()
    blocks = []   # (cond_name, action)
    i = 0
    while i < len(logic):
        line = logic[i].rstrip()
        s = line.strip()
        i += 1
        if not s or s.startswith("//") or s.startswith("strategy("):
            continue
        if s.startswith("if "):
            act = logic[i].strip(); i += 1
            blocks.append((ev.series(ev.ev(ast.parse(_pyexpr(s[3:]), mode="eval"))).astype(bool), act))
            continue
        if ":=" in s:
            name, expr = [x.strip() for x in s.split(":=", 1)]
            env[name] = ev.ev(ast.parse(_pyexpr(expr), mode="eval"))
            continue
        name, expr = [x.strip() for x in s.split("=", 1)]
        m = re.match(r"input\.(int|float|bool|string)\((.+?)(,|\))", expr)
        if m:
            env[name] = ast.literal_eval(_pyexpr(m.group(2)))
            continue
        env[name] = ev.ev(ast.parse(_pyexpr(expr), mode="eval"))

    stateful = _parse_stateful(stateful_text)
    state = {}

    # ---- Pine-style broker ----
    comm = hdr["commission_pct"] / 100
    o, c = env["open"], env["close"]
    equity, pos, qty, entry_px, entry_bar = 10000.0, 0, 0.0, 0.0, None
    trades = []
    pending = []
    hi, lo = env["high"], env["low"]
    stop_order = {1: None, -1: None}    # active strategy.exit stop per side (persists, Pine-style)

    def close_position(bar, px=None, reason="signal"):
        nonlocal equity, pos, qty
        px = o[bar] if px is None else px
        pnl = qty * (px - entry_px) * pos
        fee = qty * px * comm
        eq_before = trades[-1]["equity_before"]
        equity += pnl - fee
        trades[-1].update(exit_bar=bar, exit_price=float(px), ret=equity / eq_before - 1, exit_reason=reason)
        pos, qty = 0, 0.0

    for t in range(n):
        for kind, arg in pending:
            if kind == "close" and pos == arg:
                close_position(t)
            elif kind == "entry" and pos != arg:
                if pos != 0:
                    close_position(t)
                eq_before = equity
                qty = equity / o[t]
                equity -= qty * o[t] * comm
                pos, entry_px = arg, o[t]
                trades.append({"direction": arg, "entry_bar": t, "entry_price": float(o[t]),
                               "exit_bar": None, "exit_price": None, "ret": None,
                               "equity_before": eq_before})
        pending = []
        # stop orders placed at previous closes trigger intrabar (after the open fills)
        if pos == 1 and stop_order[1] is not None and lo[t] <= stop_order[1]:
            close_position(t, min(o[t], stop_order[1]), "stop")
        elif pos == -1 and stop_order[-1] is not None and hi[t] >= stop_order[-1]:
            close_position(t, max(o[t], stop_order[-1]), "stop")
        for cond, act in blocks:
            if not cond[t]:
                continue
            if act.startswith("strategy.exit("):
                m = re.match(r'strategy\.exit\("(\w+)", "(\w)", stop=(\w+)\)', act)
                stop_order[1 if m.group(2) == "L" else -1] = float(env[m.group(3)][t])
            elif act.startswith('strategy.close("L")'):
                pending.append(("close", 1))
            elif act.startswith('strategy.close("S")'):
                pending.append(("close", -1))
            elif act.startswith('strategy.entry("L"'):
                pending.append(("entry", 1))
            elif act.startswith('strategy.entry("S"'):
                pending.append(("entry", -1))
            else:
                raise ValueError(f"unsupported action: {act}")
        # stateful (`var`) section, executed bar-by-bar at this bar's close
        for st in stateful:
            _exec(st, t, env, state, pos, stop_order)
    for tr in trades:
        tr.pop("equity_before")
    return {"trades": trades, "env": env}


# ---- bar-by-bar executor for the `// @@stateful` section ----
def _parse_stmt(line):
    line = line.strip()
    m = re.match(r'strategy\.exit\("(\w+)", "(\w)", stop=(\w+)\)$', line)
    if m:
        return ("exit", 1 if m.group(2) == "L" else -1, m.group(3))
    m = re.match(r"(\w+)\s*:=\s*(.+)$", line)
    if m:
        return ("assign", m.group(1), ast.parse(_pyexpr(m.group(2)), mode="eval").body)
    raise ValueError(f"unsupported stateful statement: {line}")


def _parse_stateful(text):
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("//")]
    out, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        m = re.match(r"var\s+float\s+(\w+)\s*=\s*na$", s)
        if m:
            out.append(("var", m.group(1)))
            continue
        if s.startswith("if "):
            cond = ast.parse(_pyexpr(s[3:]), mode="eval").body
            then = _parse_stmt(lines[i]); i += 1
            other = None
            if i < len(lines) and lines[i].strip() == "else":
                other = _parse_stmt(lines[i + 1]); i += 2
            out.append(("if", cond, then, other))
            continue
        out.append(_parse_stmt(s))
    return out


def _sev(node, t, env, state, pos):
    """Scalar Pine evaluation at bar t (na = NaN; math.max/min propagate na)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in state:
            return state[node.id]
        v = env[node.id]
        return v[t] if isinstance(v, np.ndarray) else v
    if isinstance(node, ast.Attribute) and node.value.id == "strategy" and node.attr == "position_size":
        return float(pos)
    if isinstance(node, ast.Compare):
        a, b = _sev(node.left, t, env, state, pos), _sev(node.comparators[0], t, env, state, pos)
        if math.isnan(a) or math.isnan(b):
            return False
        return {ast.Gt: a > b, ast.Lt: a < b, ast.GtE: a >= b, ast.LtE: a <= b,
                ast.Eq: a == b, ast.NotEq: a != b}[type(node.ops[0])]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _sev(node.operand, t, env, state, pos)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "na":
            v = _sev(node.args[0], t, env, state, pos)
            return v is None or (isinstance(v, float) and math.isnan(v))
        if isinstance(node.func, ast.Attribute) and node.func.value.id == "math":
            a, b = (_sev(x, t, env, state, pos) for x in node.args)
            if math.isnan(a) or math.isnan(b):
                return math.nan
            return max(a, b) if node.func.attr == "max" else min(a, b)
    raise ValueError(f"unsupported stateful expr: {ast.dump(node)}")


def _exec(st, t, env, state, pos, stop_order):
    kind = st[0]
    if kind == "var":
        state.setdefault(st[1], math.nan)
    elif kind == "assign":
        state[st[1]] = float(_sev(st[2], t, env, state, pos))
    elif kind == "exit":
        stop_order[st[1]] = float(_sev(ast.Name(id=st[2]), t, env, state, pos))
    elif kind == "if":
        if _sev(st[1], t, env, state, pos):
            _exec(st[2], t, env, state, pos, stop_order)
        elif st[3] is not None:
            _exec(st[3], t, env, state, pos, stop_order)
