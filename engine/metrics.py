"""Performance metrics. Expectancy is reported as mean net % return per closed trade."""
from __future__ import annotations

import numpy as np

BARS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365}


def compute(res, tf: str, include_forced: bool = True) -> dict:
    tr = [t for t in res.closed if include_forced or not t.forced]
    rets = np.array([t.ret for t in tr], float)
    eq = np.asarray(res.equity, float)
    eq = eq[~np.isnan(eq)]
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gp, gl = wins.sum(), -losses.sum()
    pf = float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    br = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
    sd = br.std(ddof=1) if len(br) > 1 else 0.0
    sharpe = float(br.mean() / sd * np.sqrt(BARS_PER_YEAR[tf])) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq) if len(eq) else eq
    mdd = float(((peak - eq) / peak).max()) if len(eq) else 0.0
    total = float(eq[-1] / eq[0] - 1) if len(eq) else 0.0
    years = len(eq) / BARS_PER_YEAR[tf]
    cagr = float((eq[-1] / eq[0]) ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else -1.0
    held = 0
    for t in res.trades:
        last = t.exit_bar if t.exit_bar is not None else res.start + len(eq) - 1
        held += max(0, last - t.entry_bar + 1)
    exposure = held / len(eq) if len(eq) else 0.0
    return {
        "exposure": float(exposure),
        "trades": int(len(rets)), "closed_trades": int(sum(not t.forced for t in tr)), "win_rate": float((rets > 0).mean()) if len(rets) else 0.0,
        "expectancy_pct": float(rets.mean() * 100) if len(rets) else 0.0,
        "avg_win_pct": float(wins.mean() * 100) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean() * 100) if len(losses) else 0.0,
        "profit_factor": pf, "total_return_pct": total * 100, "cagr_pct": cagr * 100,
        "sharpe": sharpe, "max_dd_pct": mdd * 100, "bars": int(len(eq)), "years": years,
    }
