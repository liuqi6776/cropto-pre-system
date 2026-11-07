"""Inspect disabled live preparation; never signs or transmits orders."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.argon4_forward.live_preflight import verify_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "config/argon4_forward_policy.json")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config/argon4_live_preflight.json").read_text(encoding="utf-8"))
    report = verify_artifacts(ROOT, policy, config)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["preparation_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
