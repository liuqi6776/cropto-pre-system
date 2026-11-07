"""Fail-closed, offline comparison of frozen reference and cached feature paths.

Copies the same already-observed input bytes to a report-local audit snapshot,
uses actual decision timestamps from the read-only observation journal, and
loads all three deployed model bundles against policy manifest pins. This is
an implementation parity/timing check, not a return or execution backtest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys
import time
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.argon4_forward import feature_cache, model

STREAMS = ("ETH_5m", "BTC_15m", "BTC_5m")
SYMBOLS = ("ETHUSDT", "BTCUSDT")
REPORT_ROOT = ROOT / "reports/argon4_latency_v11_20261107"


def load_bundles(policy, phase):
    pins = policy.get("deployment_manifest_sha256", {}) if phase == "deployment_q4" else None
    if pins is not None and any(not pins.get(stream) for stream in STREAMS):
        raise ValueError("all three deployment model manifest pins are required")
    # An explicitly selected research phase uses its own adjacent checksums;
    # load_model_bundle also enforces phase identity and artifact hashes.
    return {stream: model.load_model_bundle(stream, phase,
                expected_manifest_sha256=pins[stream] if pins is not None else None)
            for stream in STREAMS}


def select_cohorts(database, phase, count):
    if count < 1:
        raise ValueError("cohort count must be positive")
    fold = model.FOLDS[phase]
    start, end = (model._timestamp(fold[key]) for key in ("valid_from", "valid_until"))
    result, seen, reasons = [], set(), {}
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as db:
        records = db.execute("SELECT id, saved_at, payload FROM snapshots WHERE kind = ? ORDER BY id DESC",
                             ("prediction_decisions",))
        scanned = 0
        for row_id, saved_at, raw in records:
            scanned += 1
            payload = json.loads(raw)
            cohort = model._timestamp(payload["cohort"])
            due = [s for s in STREAMS if cohort.minute % (model.STREAMS[s][1] * 5) == 0]
            predictions = {s: payload.get("decisions", {}).get(s, {}).get("prediction") for s in due}
            reason = None
            if cohort in seen:
                reason = "duplicate_cohort"
            elif any(not p or not p.get("decision_time") for p in predictions.values()):
                reason = "no_complete_recorded_prediction"
            elif not start <= cohort < end:
                reason = "outside_explicit_model_phase"
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            cutoff = max(model._timestamp(p["decision_time"]) for p in predictions.values())
            if not (start <= cutoff < end and pd.Timedelta(0) <= cutoff - cohort <= pd.Timedelta(seconds=120)):
                raise ValueError("recorded decision timestamp is invalid for cohort/model phase")
            seen.add(cohort)
            result.append({"observation_id": row_id, "saved_at": saved_at,
                           "cohort": cohort.isoformat(), "as_of": cutoff.isoformat(), "due_streams": due})
            if len(result) == count:
                break
    if len(result) != count:
        raise ValueError(f"requested {count} observed cohorts; found only {len(result)} matching phase")
    if any(not any(s in item["due_streams"] for item in result) for s in STREAMS):
        raise ValueError("selected cohorts do not exercise all three enabled models")
    return list(reversed(result)), {"journal_rows_scanned": scanned, "excluded_reasons": reasons}


def frozen_inputs(source_root, destination, *, expected_hashes=None):
    # No live input/ledger mutation. Capture fails if a collector replaces any
    # file during capture; all copied/hash-checked bytes belong to this snapshot.
    with feature_cache.FeatureSnapshot(source_root, as_of=datetime.now(timezone.utc)) as snapshot:
        manifest = []
        for path, (raw, digest) in snapshot._raw.items():
            relative = path.relative_to(snapshot.root)
            if expected_hashes is not None and expected_hashes.get(relative.as_posix()) != digest:
                raise ValueError("snapshot bytes do not match the recorded collector hash: " + relative.as_posix())
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("audit snapshot copy verification failed")
            manifest.append({"source_path": str(path), "snapshot_path": str(target),
                             "sha256": digest, "bytes": len(raw)})
    return manifest


def latest_input_cohort(database, phase):
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT id, saved_at, payload FROM snapshots WHERE kind = ? ORDER BY id DESC LIMIT 1",
                         ("model_inputs",)).fetchone()
    if row is None:
        raise ValueError("no observed collector snapshot")
    payload = json.loads(row[2])
    cohort, cutoff = (model._timestamp(payload[k]) for k in ("latest_closed_5m", "as_of"))
    fold = model.FOLDS[phase]
    if not (model._timestamp(fold["valid_from"]) <= cutoff < model._timestamp(fold["valid_until"])):
        raise ValueError("latest observed snapshot outside declared model phase")
    if not pd.Timedelta(0) <= cutoff - cohort <= pd.Timedelta(seconds=120):
        raise ValueError("latest observed timestamp outside model inference window")
    return [{"observation_id": row[0], "saved_at": row[1], "cohort": cohort.isoformat(),
             "as_of": cutoff.isoformat(), "due_streams": [s for s in STREAMS if cohort.minute % (model.STREAMS[s][1]*5) == 0],
             "collector_status": payload["status"], "collector_gaps": payload.get("gaps", []),
             "collector_errors": payload.get("errors", []),
             "verification_only": "Recorded input snapshot, not a replayed trade or readiness approval"}], {
                 "journal_rows_scanned": 1,
                 "expected_source_hashes": {Path(v["path"]).relative_to(Path(payload["data_root"])).as_posix(): v["sha256"]
                     for v in payload["sources"].values() if v.get("path") and v.get("sha256")}}


def assert_frames(left, right):
    for name in ("base", "expert", "raw_target"):
        pd.testing.assert_frame_equal(left[name], right[name], check_exact=True)
        if left[name].to_numpy().tobytes() != right[name].to_numpy().tobytes():
            raise AssertionError(f"numeric bit pattern mismatch: {name}")
    for name in ("availability", "provenance", "classification", "source_sha256"):
        if left[name] != right[name]:
            raise AssertionError(f"feature evidence mismatch: {name}")
    if left["base"].shape[1] != 311 or left["expert"].shape[1] != 65:
        raise AssertionError("376-feature schema required")


def assert_prediction(left, right):
    if left != right:
        raise AssertionError("prediction/selection/direction metadata mismatch")
    for name in ("z_base", "z_expert", "z_v4", "threshold_frozen", "threshold_reset", "w"):
        if struct.pack("!d", left[name]) != struct.pack("!d", right[name]):
            raise AssertionError(f"prediction bit pattern mismatch: {name}")


def compare_cohort(root, cohort, bundles, phase, *, candidate_first=False):
    cutoff = model._timestamp(cohort["as_of"])
    results, elapsed, counts, errors, cache_stats = {}, {}, {}, {}, None
    order = ("candidate", "reference") if candidate_first else ("reference", "candidate")
    for implementation in order:
        original_read = pd.read_parquet
        started = time.perf_counter()
        try:
            with patch.object(pd, "read_parquet", wraps=original_read) as read:
                if implementation == "reference":
                    frames = {s: model.build_feature_frames(root, s, as_of=cutoff, tail_rows=1) for s in SYMBOLS}
                else:
                    with feature_cache.FeatureSnapshot(root, as_of=cutoff) as snapshot:
                        frames = {s: snapshot.build(s, tail_rows=1) for s in SYMBOLS}
                        cache_stats = snapshot.stats
                predictions, scores = {}, {}
                for stream in STREAMS:
                    symbol = model.STREAMS[stream][0]
                    frame = frames[symbol]
                    if frame["base"].index[-1] + pd.Timedelta(minutes=5) != model._timestamp(cohort["cohort"]):
                        raise model.ModelInputError("observed inputs do not reach the recorded decision cohort")
                    scored = model.predict_scores(bundles[stream], frame["base"], frame["expert"])
                    zv = float(scored["LOCAL_V4_FROZEN"][0])
                    scores[stream] = {"score_hex": {k: [float(v).hex() for v in values] for k, values in scored.items()},
                                      "selected": abs(zv) >= bundles[stream]["package"]["threshold_frozen"],
                                      "side": "UP" if zv > 0 else "DOWN", "due": stream in cohort["due_streams"]}
                    if stream in cohort["due_streams"]:
                        predictions[stream] = model.predict_stream(stream, frame["base"], frame["expert"],
                                                                  cutoff, bundle=bundles[stream], phase=phase)
                counts[implementation] = read.call_count
        except model.ModelInputError as exc:
            errors[implementation] = str(exc)
            continue
        elapsed[implementation] = time.perf_counter() - started
        results[implementation] = (frames, predictions, scores)
    if errors:
        if len(errors) != 2 or errors["reference"] != errors["candidate"]:
            raise AssertionError(f"reference/candidate input rejection mismatch: {errors}")
        return {**cohort, "status": "INPUT_UNAVAILABLE", "errors": errors,
                "fail_closed_parity": True, "exact_features_scores_selection": False}
    for symbol in SYMBOLS:
        assert_frames(results["reference"][0][symbol], results["candidate"][0][symbol])
    for stream in cohort["due_streams"]:
        assert_prediction(results["reference"][1][stream], results["candidate"][1][stream])
    if results["reference"][2] != results["candidate"][2]:
        raise AssertionError("three-model raw score/selection/direction mismatch")
    return {**cohort, "status": "PASS", "seconds": elapsed, "parquet_decodes": counts, "cache_stats": cache_stats,
            "history_rows": {s: next(meta["rows"] for name, meta in results["candidate"][0][s]["availability"].items()
                                     if name.startswith(f"{s}_5m_futures"))
                             for s in SYMBOLS},
            "predictions": results["candidate"][1], "model_scores": results["candidate"][2], "exact_features_scores_selection": True}


def run(args, report):
    policy = json.loads(args.policy.read_text(encoding="utf-8-sig"))
    bundles = load_bundles(policy, args.phase)  # Missing models must never become a successful no-op.
    cohorts, selection = (latest_input_cohort(args.observations_db, args.phase) if args.latest_inputs_only else
                          select_cohorts(args.observations_db, args.phase, args.cohorts))
    run_root = args.output.parent / ("feature_snapshot_" + uuid.uuid4().hex)
    snapshot_root = run_root / "raw_data"
    report.update({"phase": args.phase, "models": {s: {"manifest_sha256": b["manifest_sha256"],
        "model_sha256": b["manifest"]["model_sha256"]} for s, b in bundles.items()},
        "selection": selection, "input_snapshot": frozen_inputs(args.data_root, snapshot_root,
                                        expected_hashes=selection.get("expected_source_hashes")),
        "recorded_collector_hashes_required": args.latest_inputs_only,
        "historical_limit": "historical mode compares implementations on one frozen current byte snapshot with original receipt cutoffs; it does not restore every original historical file version",
        "snapshot_root": str(snapshot_root), "as_of_policy": "actual recorded collector as_of or maximum recorded decision time for due streams; observed publication cutoffs retained",
        "timing_scope": "two full-history symbol feature builds, all three raw model scores and all due predictions; includes snapshot capture/validation; alternating order; local offline wall time"})
    # Let the first measurement include real cold compilation, rather than hide
    # it in the audit snapshot's capture. Subsequent cohorts reuse only code.
    feature_cache._CODE_CACHE.clear()
    report["cohorts"] = []
    for n, cohort in enumerate(cohorts):
        report["cohorts"].append(compare_cohort(snapshot_root, cohort, bundles, args.phase, candidate_first=bool(n % 2)))
        report["verified_cohorts"] = sum(c["status"] == "PASS" for c in report["cohorts"])
    passed = [c for c in report["cohorts"] if c["status"] == "PASS"]
    report["input_unavailable_cohorts"] = len(cohorts) - len(passed)
    report["stream_prediction_counts"] = {s: sum(s in c["due_streams"] for c in passed) for s in STREAMS}
    report["three_model_score_counts"] = {s: len(passed) for s in STREAMS}
    if not passed:
        raise ValueError("no real observed cohort was feature/score verified")
    report["timing_seconds"] = {kind: {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                                       "min": min(values), "max": max(values)}
        for kind in ("reference", "candidate")
        for values in ([c["seconds"][kind] for c in passed],)}
    report["median_speedup"] = report["timing_seconds"]["reference"]["median"] / report["timing_seconds"]["candidate"]["median"]
    report.update({"status": "PASS" if len(passed) == len(cohorts) else "PARTIAL",
                   "exact_features_scores_selection": len(passed) == len(cohorts),
                   "profitability_tested": False, "online_execution_speed_measured": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "paper_runtime/argon4_forward_20261105/features/raw_data")
    parser.add_argument("--observations-db", type=Path, default=ROOT / "paper_runtime/argon4_forward_20261105/observations.sqlite3")
    parser.add_argument("--policy", type=Path, default=ROOT / "config/argon4_forward_policy.json")
    parser.add_argument("--phase", choices=tuple(model.FOLDS), default="deployment_q4")
    parser.add_argument("--cohorts", type=int, default=100)
    parser.add_argument("--latest-inputs-only", action="store_true", help="one latest actually recorded collector timestamp, all three raw model scores")
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "feature_cache_verification.json")
    args = parser.parse_args(argv)
    if args.latest_inputs_only:
        args.cohorts = 1
    args.output = args.output.resolve()
    if not args.output.is_relative_to(REPORT_ROOT.resolve()):
        parser.error(f"output must remain under {REPORT_ROOT}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAIL", "verified_cohorts": 0, "requested_cohorts": args.cohorts,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "reference_implementation_sha256": model.sha256(Path(model.__file__)),
              "candidate_implementation_sha256": model.sha256(Path(feature_cache.__file__)),
              "frozen_formula_sha256": dict(model.SOURCE_HASHES)}
    try:
        run(args, report)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "verified_cohorts", "requested_cohorts")} |
                     {"report": str(args.output)}, ensure_ascii=False))
    if "error" in report:
        print(json.dumps(report["error"], ensure_ascii=False), file=sys.stderr)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
