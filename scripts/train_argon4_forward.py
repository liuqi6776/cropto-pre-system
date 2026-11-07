"""Train and persist locally reproducible paper-only seven-stream estimators."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--phase", choices=["both", "oos_2026", "deployment_q4"], default="both")
    parser.add_argument("--streams", nargs="+")
    args = parser.parse_args()
    from tools.argon4_forward.model import train_all, REPORT_ROOT, _write_json, _now
    phases = ("oos_2026", "deployment_q4") if args.phase == "both" else (args.phase,)
    try:
        train_all(threads=args.threads, phases=phases, selected_streams=args.streams)
    except Exception as exc:
        import json
        path = REPORT_ROOT / "progress.json"
        progress = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        progress.update(status="FAILED", error=type(exc).__name__, updated_at=_now())
        _write_json(path, progress)
        raise


if __name__ == "__main__":
    main()
