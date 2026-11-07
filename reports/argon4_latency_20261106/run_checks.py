"""Authorized ordered verification, retaining original logs and exit codes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
PREFLIGHT = [
    ("01_fetch_verify", ["scripts/fetch_local_source_copies.py", "--verify"]),
    ("02_import", ["scripts/import_perp_data.py"]),
    ("03_inventory", ["scripts/build_inventory.py"]),
    ("04_reports", ["scripts/build_reports.py"]),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    selected = sys.argv[1]
    output_dir = OUT / "workspace_checks/final" if selected.startswith("final_") else OUT
    selected = selected.removeprefix("final_")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"PYTHONPATH": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    if selected == "preflight":
        stages = PREFLIGHT
    elif selected == "pytest":
        temp = output_dir / ("pytest_temp_" + uuid4().hex)
        if any(word in str(temp).lower() for word in ("mark", "funding")):
            raise RuntimeError("temporary path contains a known legacy fixture-sensitive token")
        stages = [("05_pytest_full", ["-m", "pytest", "tests", "-q", "--basetemp", str(temp)])]
    else:
        raise ValueError("unsupported stage")
    for name, arguments in stages:
        log = output_dir / (name + ".log")
        if log.exists():
            raise RuntimeError("do not overwrite an existing verification log")
        command = [str(PYTHON), "-u", "-B", *arguments]
        record = {"started_at_utc": now(), "command": command, "pythonpath": "", "threads": 2,
            "purpose": "Authorized local verification only; active paper service is not paused or controlled."}
        print(json.dumps({"starting": name, "at": record["started_at_utc"]}), flush=True)
        started = time.monotonic()
        with log.open("wb") as output:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
        record.update({"finished_at_utc": now(), "elapsed_seconds": time.monotonic()-started, "exit_code": process.returncode,
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        (output_dir / (name + ".exit.txt")).write_text(str(process.returncode) + "\n", encoding="utf-8")
        (output_dir / (name + ".execution.json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"finished": name, "exit_code": process.returncode, "elapsed_seconds": record["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
