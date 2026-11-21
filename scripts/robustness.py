#!/usr/bin/env python3
"""Cluster survivors, cross-market check, best-IS table, 2x cost stress; refresh survivors.md.

Usage: .venv/bin/python scripts/robustness.py [--stress 2.0]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.robustness import run_all  # noqa: E402
from engine.search import _clean  # noqa: E402

mult = float(sys.argv[sys.argv.index("--stress") + 1]) if "--stress" in sys.argv else 2.0
summary, *_ = run_all(mult)
print(json.dumps(_clean(summary), indent=2))
