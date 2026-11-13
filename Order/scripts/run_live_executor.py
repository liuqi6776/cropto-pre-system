"""Main CLI Entrypoint for the ARGON4 External Live Executor.

Usage:
    # Run in Shadow Mode (real-time listener + local V2 signing + isolated shadow audit logs):
    c:\\Users\\liuqi\\Order\\.venv\\Scripts\\python.exe scripts/run_live_executor.py --mode shadow

    # Run in Live Mode (Week 1 trial 0.10x scale = $1,000 capital basis, $20 max order):
    c:\\Users\\liuqi\\Order\\.venv\\Scripts\\python.exe scripts/run_live_executor.py --mode live --scale 0.10

    # Run in Live Mode (Full deployment 1.00x scale = $10,000 capital basis, $200 max order):
    c:\\Users\\liuqi\\Order\\.venv\\Scripts\\python.exe scripts/run_live_executor.py --mode live --scale 1.00
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
_VENV_SITE = ROOT / ".venv/Lib/site-packages"
if _VENV_SITE.is_dir() and str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from argon4_live.engine import LiveExecutionEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGON4 External Live Executor")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/live_config.json",
        help="Path to live_config.json",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="Path to .env credentials file",
    )
    parser.add_argument(
        "--mode",
        choices=("shadow", "live"),
        default=None,
        help="Execution mode ('shadow' or 'live'; defaults to ARGON4_EXECUTION_MODE in .env)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override live_scale_factor (e.g. 0.10 for Week 1 trial, 1.00 for full deployment)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single non-blocking poll pass and exit",
    )
    args = parser.parse_args()

    root_runtime_dir = (ROOT / "runtime").resolve()
    root_runtime_dir.mkdir(parents=True, exist_ok=True)
    log_handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        log_handlers.append(logging.FileHandler(root_runtime_dir / "live_executor.stderr.log", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=log_handlers,
    )

    engine = LiveExecutionEngine.from_config_and_env(
        config_path=args.config,
        env_path=args.env_file,
        mode_override=args.mode,
        scale_override=args.scale,
        acquire_instance_lock=True,
    )

    runtime_dir = Path(engine.config["audit"]["runtime_dir"]).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    root_runtime_dir = (ROOT / "runtime").resolve()
    root_runtime_dir.mkdir(parents=True, exist_ok=True)
    service_file = runtime_dir / "live_service.json"
    root_service_file = root_runtime_dir / "live_service.json"
    service_files = [service_file]
    if root_service_file != service_file:
        service_files.append(root_service_file)

    if args.once:
        try:
            res = engine.run_once(force_state_poll=True)
            print(json.dumps(res, indent=2, ensure_ascii=False))
        finally:
            engine.close()
        return 0

    svc_payload = json.dumps(
        {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "mode": engine.clob.mode,
            "wallet_type": engine.clob.wallet_type,
            "signature_type": engine.clob.signature_type,
            "funder_address": engine.clob.funder_address,
            "live_scale_factor": engine.live_scale_factor,
            "runtime_dir": str(runtime_dir),
            "withdrawal_permission": False,
            "order_type": "FAK",
        },
        indent=2,
    )
    for sf in service_files:
        sf.write_text(svc_payload, encoding="utf-8")

    def _handle_stop(signum: int, _frame: object) -> None:
        logging.getLogger("argon4_live").info("Received signal %s, stopping...", signum)
        engine.stop_event.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        engine.run_forever()
    finally:
        engine.close()
        for sf in service_files:
            if sf.is_file():
                try:
                    info = json.loads(sf.read_text(encoding="utf-8"))
                    if info.get("pid") == os.getpid():
                        sf.unlink(missing_ok=True)
                except Exception:
                    pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
