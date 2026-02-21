from __future__ import annotations

import json
from pathlib import Path

run_dir = Path("outputs/runs/latest")
segments = run_dir / "segments.json"
metrics = run_dir / "metrics.json"
rankings = run_dir / "rankings.json"

for path in [segments, metrics, rankings]:
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {path}")

metrics_data = json.loads(metrics.read_text())
if metrics_data:
    sample = metrics_data[0]
    for key in ["idle_blockers", "idle_blocker_summary"]:
        if key not in sample:
            raise SystemExit(f"metrics.json missing key: {key}")

print("Validation OK: core artifacts and blocker keys are present.")
