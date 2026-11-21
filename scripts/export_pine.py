#!/usr/bin/env python3
"""Export a Pine v6 script. Example (default): EMA 12/26 cross, long-only.

Usage: .venv/bin/python scripts/export_pine.py [family] [json-params] [outfile]
  e.g. scripts/export_pine.py ma_cross '{"fast":12,"slow":26,"ma_type":"ema","side":"long"}' exports/ema_cross_12_26.pine
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.pine import export  # noqa: E402

fam = sys.argv[1] if len(sys.argv) > 1 else "ma_cross"
params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {"fast": 12, "slow": 26, "ma_type": "ema", "side": "long"}
out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parent.parent / "exports" / "example_ema_cross_12_26.pine"
out.parent.mkdir(exist_ok=True)
out.write_text(export(fam, params))
print(out)
