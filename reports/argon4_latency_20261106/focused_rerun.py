"""Authorized precise rechecks; no business, fixture or registry edits."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "workspace_checks/final"
NODES = [
    "tests/test_research_pipeline_v1.py::test_real_core4_bnb_run_preserves_native_funding_and_uses_simulated_transition_fills",
    "tests/test_research_pipeline_v1.py::test_real_core4_ridge_recipe_fits_only_mature_2021_and_scores_exposed_2025",
    "tests/test_research_pipeline_v1.py::test_real_ridge_debug_overlay_runs_shared_account_pipeline_with_causal_forecasts",
    "tests/test_research_pipeline_v1.py::test_explicit_terminal_flat_is_declared_as_simulated_and_run_window_is_half_open",
]
AUTHORITY = ["archive/catalog/dataset_registry.json", "archive/catalog/mark_candle_registry.json", "source_data/MANIFEST.json",
    "source_data/staging/core4_p0_20260925/partition_bindings_v1.json"]


def now():
    return datetime.now(timezone.utc).isoformat()


def snapshot():
    return {name: {"sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), "mtime_ns": (ROOT / name).stat().st_mtime_ns} for name in AUTHORITY}


def main():
    results = []
    env = dict(os.environ)
    env.update({"PYTHONPATH": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    for index, node in enumerate(NODES, 1):
        log = OUT / f"focused_{index:02d}.log"
        temp = OUT / ("focused_temp_" + uuid4().hex)
        command = [str(ROOT / ".venv/Scripts/python.exe"), "-u", "-B", "-m", "pytest", node, "-q", "--basetemp", str(temp)]
        record = {"node": node, "started_at_utc": now(), "before_authority": snapshot(), "command": command, "log": str(log),
            "environment_overrides": {key: env[key] for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        print(json.dumps({"starting": node, "at": record["started_at_utc"]}), flush=True)
        started = time.monotonic()
        with log.open("wb") as handle:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        record.update({"exit_code": process.returncode, "ended_at_utc": now(), "elapsed_seconds": time.monotonic() - started,
            "after_authority": snapshot(), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        record["authority_changes"] = [name for name in AUTHORITY if record["before_authority"][name]["sha256"] != record["after_authority"][name]["sha256"]]
        results.append(record)
        (OUT / "focused_rerun_summary.json").write_text(json.dumps({"status": "RUNNING" if index < len(NODES) else "COMPLETED",
            "completed_at_utc": now(), "results": results, "scope": "Four exact original nodes, each fresh process; running paper service and guards unchanged."}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"completed": node, "exit_code": process.returncode, "elapsed_seconds": record["elapsed_seconds"], "authority_changes": record["authority_changes"]}), flush=True)


if __name__ == "__main__":
    main()
