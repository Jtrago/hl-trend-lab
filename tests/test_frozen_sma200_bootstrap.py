"""Frozen SMA200 spec checksum/cutoff and bootstrap sanity."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.bootstrap import block_indices, paired_bootstrap
from engine.frozen_sma200 import SPEC, PORTFOLIO_SPEC, load_verified

ROOT = Path(__file__).resolve().parent.parent


def _runner():
    spec = importlib.util.spec_from_file_location("efs", ROOT / "scripts" / "eval_frozen_sma200.py")
    m = importlib.util.module_from_spec(spec); sys.modules["efs"] = m; spec.loader.exec_module(m)
    return m


def test_spec_cutoff_matches_portfolio_and_checksum_ok():
    d = load_verified()
    assert d["data_cutoff_utc"] == json.loads(PORTFOLIO_SPEC.read_text())["data_cutoff_utc"]


def test_tampered_spec_refused(tmp_path):
    d = json.loads(SPEC.read_text())
    d["rule"]["sma_len"] = 150
    p = tmp_path / "s.json"; p.write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        load_verified(p)


def test_runner_scores_only_after_cutoff(tmp_path):
    d = json.loads(SPEC.read_text())
    d["data_cutoff_utc"] = "2026-09-01T00:00:00+00:00"          # fake earlier cutoff (rule untouched -> sha ok)
    p = tmp_path / "s.json"; p.write_text(json.dumps(d))
    r = _runner().main(spec_path=p, out=tmp_path)
    days = sorted(r["daily"]["sma200_plain"])
    assert r["status"] == "scored" and days[0] >= "2026-09-01"


def test_block_indices_circular_and_length():
    rng = np.random.default_rng(0)
    idx = block_indices(103, 20, rng)
    assert len(idx) == 103 and idx.min() >= 0 and idx.max() < 103
    blk = idx[:20]
    assert all((blk[i + 1] - blk[i]) % 103 == 1 for i in range(19))


def test_identical_series_zero_diff():
    x = np.random.default_rng(1).normal(0.001, 0.02, 500)
    r = paired_bootstrap(x, x, reps=200)
    assert r["sharpe_diff"] == 0 and r["sharpe_diff_ci95"] == [0.0, 0.0]
