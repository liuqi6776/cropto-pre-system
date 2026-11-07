"""Verify an interrupted, fully applied input revision; apply only while stopped.

Default is read-only verification. --apply completes audit publication at a
fresh calibrated time, never rewrites inputs, orders, positions or cash.
"""
import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.argon4_forward.revisions import recover_prepared_price_revision, abort_unapplied_price_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "paper_runtime/argon4_forward_20261105")
    parser.add_argument("--apply", action="store_true", help="requires verification pause and stopped local service")
    parser.add_argument("--abort-unapplied", action="store_true",
        help="verify an entirely unapplied transaction and its prior committed view; never roll back raw data")
    args = parser.parse_args()
    runtime = args.runtime_dir.resolve()
    if not runtime.is_relative_to((ROOT / "paper_runtime").resolve()):
        parser.error("runtime must remain under this workspace's paper_runtime")
    transport = None
    listener_guard = None
    callback = None
    if args.apply:
        if not (runtime / "verification_pause.json").is_file():
            parser.error("create a verification pause and stop the managed service before applying")
        # Reserve the address exclusively. A connect timeout cannot distinguish
        # a stopped listener from filtering; a successful exclusive bind can.
        # Keep the reservation until completion to prevent a concurrent restart.
        listener_guard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener_guard.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener_guard.bind(("127.0.0.1", 8897))
        except OSError:
            listener_guard.close()
            parser.error("cannot reserve stopped local service; custom-port writers must also be stopped")
        from tools.argon4_forward.clock_sync import calibrate, calibrated_now
        from tools.argon4_forward.polymarket_transport import PublicPolymarketTransport
        from tools.argon4_forward.monitor import ObservationStore
        transport = PublicPolymarketTransport(pool_size=1)
        def callback():
            clock = calibrate(transport=transport)
            if clock.get("status") != "CALIBRATED":
                raise ValueError("fresh revision recovery clock unavailable")
            ObservationStore(runtime / "observations.sqlite3").save("revision_recovery_clock", clock)
            return calibrated_now(clock)
    try:
        if args.abort_unapplied:
            result = abort_unapplied_price_revision(runtime, abort_now=callback, apply=args.apply)
        else:
            result = recover_prepared_price_revision(runtime,
                mark_registry_path=ROOT / "archive/catalog/mark_candle_registry.json",
                cutover_now=callback, apply=args.apply)
    finally:
        if transport is not None:
            transport.close()
        if listener_guard is not None:
            listener_guard.close()
    report_name = ("unapplied_revision_" if args.abort_unapplied else "prepared_revision_")+(
        "apply.json" if args.apply else "check.json")
    report = ROOT / "reports/argon4_latency_v11_20261107" / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "reason": result.get("reason"),
        "view_effective_at": result.get("view_effective_at"), "report": str(report)}, ensure_ascii=False))
    return 0 if result.get("status") in {"VERIFIED_PREPARED_CUTOVER", "COMMITTED",
                                         "VERIFIED_UNAPPLIED_REVISION", "ABORTED_UNAPPLIED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
