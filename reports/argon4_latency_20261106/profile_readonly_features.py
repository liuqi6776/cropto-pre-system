"""Offline timings at a saved decision's as-of time; never a fresh decision."""
from datetime import datetime, timezone
from pathlib import Path
import cProfile
import hashlib
import io
import json
import os
import pstats
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tools.argon4_forward import model, revisions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*.parquet"))}


def measured(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter()-start


def main():
    runtime = ROOT/"paper_runtime/argon4_forward_20261105"
    folder = Path(__file__).parent
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex
    report = {"classification": "OFFLINE_PROFILE_OF_SAVED_DECISION_NOT_REEXECUTION",
              "writes_limited_to": str(folder), "network_used": False, "orders_created": False,
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    state = json.loads((runtime/"pipeline_state.json").read_text(encoding="utf-8"))
    policy_path = ROOT/"config/argon4_forward_policy.json"
    policy_before = sha(policy_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    predictions = [d["prediction"] for d in state.get("decisions", {}).values() if "prediction" in d]
    if not predictions:
        raise ValueError("No existing saved decision available for a causal profile")
    cutoff = min(pd.Timestamp(p["decision_time"]) for p in predictions)
    view, view_seconds = measured(lambda: revisions.ensure_view_committed(runtime))
    if view.get("status") == "COMMITTED" and cutoff < pd.Timestamp(view["view_effective_at"]):
        raise ValueError("Current working view cannot profile a pre-cutover decision")
    raw_root = Path(state["inputs"]["data_root"])
    before = inventory(raw_root)
    report.update(saved_decision_as_of=cutoff.isoformat(), raw_root=str(raw_root), raw_files_before=before,
                  revision_view=view, ensure_view_committed_seconds=view_seconds,
                  model_code_sha256=sha(model.__file__), policy_sha256=policy_before,
                  original_saved_decision_times=[p["decision_time"] for p in predictions])
    builds, frames, timings = [], {}, []
    with threadpool_limits(limits=2):
        for repeat in range(3):
            for symbol in ("ETHUSDT", "BTCUSDT"):
                profiler = cProfile.Profile()
                start = time.perf_counter()
                if repeat == 0:
                    profiler.enable()
                result = model.build_feature_frames(raw_root, symbol, as_of=cutoff, tail_rows=1)
                if repeat == 0:
                    profiler.disable()
                elapsed = time.perf_counter()-start
                builds.append({"symbol": symbol, "repeat": repeat, "profiled": repeat == 0, "seconds": elapsed,
                               "input_5m_rows": {k: v["rows"] for k, v in result["availability"].items() if "5m" in k},
                               "source_count": len(result["provenance"]), "output_shape": [result["base"].shape, result["expert"].shape],
                               "feature_bar_open": result["index"][-1].isoformat()})
                if symbol in frames:
                    pd.testing.assert_frame_equal(frames[symbol]["base"], result["base"])
                    pd.testing.assert_frame_equal(frames[symbol]["expert"], result["expert"])
                frames[symbol] = result
                if repeat == 0:
                    dest = folder/(identity+"_"+symbol+"_feature_profile.txt")
                    stream = io.StringIO()
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(45)
                    dest.write_text(stream.getvalue(), encoding="utf-8")
                    builds[-1]["profile_path"] = str(dest.relative_to(ROOT))
        for stream in ("ETH_5m", "BTC_5m", "BTC_15m"):
            bundle, load_seconds = measured(lambda: model.load_model_bundle(stream,
                expected_manifest_sha256=policy["deployment_manifest_sha256"][stream]))
            f = frames[stream.split("_")[0]+"USDT"]
            duration = model.STREAMS[stream][1]*5
            if (pd.Timestamp(f["index"][-1])+pd.Timedelta(minutes=5)).minute % duration:
                timings.append({"stream": stream, "status": "NOT_DUE_AT_SAVED_COHORT", "load_seconds": load_seconds})
                continue
            scores, seconds = [], []
            for _ in range(20):
                prediction, elapsed = measured(lambda: model.predict_stream(stream, f["base"], f["expert"], cutoff, bundle=bundle))
                scores.append((prediction["z_v4"], prediction["selected"], prediction["side"]))
                seconds.append(elapsed)
            if len(set(scores)) != 1:
                raise ValueError("Repeated saved-row inference changed")
            timings.append({"stream": stream, "load_seconds": load_seconds, "samples": len(seconds),
                            "median_seconds": statistics.median(seconds), "p95_seconds": float(np.quantile(seconds,.95)),
                            "maximum_seconds": max(seconds), "repeated_scores_exact": True})
    after = inventory(raw_root)
    view_after = revisions.ensure_view_committed(runtime)
    stable = before == after and view == view_after and policy_before == sha(policy_path)
    report.update(status="READ_ONLY_PROFILE_VALIDATED" if stable else "SOURCE_CHANGED_DURING_PROFILE_DO_NOT_USE_FOR_EQUIVALENCE",
                  raw_files_after=after, source_bytes_unchanged=before == after,
                  revision_view_unchanged=view == view_after, policy_unchanged=policy_before == sha(policy_path),
                  feature_builds=builds, inference=timings, feature_rows_repeat_exact=True,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    destination = folder/(identity+"_profile.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(destination.relative_to(ROOT)), "status": report["status"],
                      "saved_decision_as_of": cutoff.isoformat(), "ensure_view_committed_seconds": view_seconds,
                      "feature_builds": builds, "inference": timings}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
