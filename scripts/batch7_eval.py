#!/usr/bin/env python3
"""Batch 7: block bootstrap lab-vs-SMA200 + 50/50 blend sensitivity (needs cache/portfolio_wf_picks.pkl)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.bootstrap import run_all  # noqa: E402

run_all()
print("ok")
