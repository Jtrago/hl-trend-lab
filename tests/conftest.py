import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the engine at the small public-data fixtures shipped with the tests (BTC/ETH Binance
# spot klines and Hyperliquid funding from 2026-02 on, plus the frozen SMA200 spec). Must run
# before any engine module is imported.
_FIX = Path(__file__).resolve().parent / "fixtures"
os.environ["HL_TREND_LAB_DATA"] = str(_FIX / "data")
os.environ["HL_TREND_LAB_RESULTS"] = str(_FIX / "results")


def make_fixture(n=800, seed=7):
    """Tiny deterministic OHLC fixture (synthetic, used only for unit tests)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n) + 0.002 * np.sin(np.arange(n) / 40)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


@pytest.fixture
def fx():
    return make_fixture()
