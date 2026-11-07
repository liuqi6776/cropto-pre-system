"""Ordered local verification. Run only after explicit root freeze approval."""
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

ROOT = Path(__file__).resolve().parents[3]
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
    if not (OUT / "chain_start_snapshot.json").exists():
        raise RuntimeError("root-authorized final freeze snapshot required before checks")
    selected = sys.argv[1]
    env = dict(os.environ)
    env.update({"PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    if selected == "preflight":
        stages = PREFLIGHT
    elif selected == "pytest":
        temp = OUT / ("pytest_temp_" + uuid4().hex)
        if any(word in str(temp).lower() for word in ("mark", "funding")):
            raise RuntimeError("fixture-sensitive temporary path")
        stages = [("05_pytest_full", ["-m", "pytest", "tests", "-q", "--basetemp", str(temp)])]
    else:
        raise ValueError("unsupported stage")
    for name, arguments in stages:
        log = OUT / (name + ".log")
        if log.exists():
            raise RuntimeError("do not overwrite existing verification log")
        command = [str(PYTHON), "-u", "-B", *arguments]
        record = {"started_at_utc": now(), "command": command, "pythonpath": "", "threads": 2,
                  "purpose": "Authorized local verification; active paper service remains running."}
        print(json.dumps({"starting": name, "at": record["started_at_utc"]}), flush=True)
        start = time.monotonic()
        with log.open("wb") as output:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
        record.update({"finished_at_utc": now(), "elapsed_seconds": time.monotonic() - start,
                       "exit_code": process.returncode, "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
        (OUT / (name + ".exit.txt")).write_text(str(process.returncode) + "\n", encoding="utf-8")
        (OUT / (name + ".execution.json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"finished": name, "exit_code": process.returncode, "elapsed_seconds": record["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
