"""Independent local tail-weight research candidate; never a production loader.

The author's incompletely specified TabNet is intentionally not implemented.
Existing 311+65 feature code, baseline artifacts and live paper state are read
only. Every candidate output lives under a separate fixed research directory.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import model as frozen

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "research/argon4_gen4_candidate_20261106"
REPORT_ROOT = ROOT / "reports/argon4_gen4_candidate_20261106"
BASELINE_REPORT_ROOT = ROOT / "reports/argon4_local_v4_20261106"
BASE_CODE_SHA256 = "1d9629abf05c27eeec0ed6e63ae4b4fc65c4f2b904dca4de9549698cf1a009f7"
DATA_MANIFEST_SHA256 = "72d9062e3add00c611605a8bc1a66caf7ab10c3cc13a2e93a80fc96adaab34df"
CLASSIFICATION = "LOCAL_311_PLUS_65_TAIL_WEIGHT_CANDIDATE_RESEARCH_ONLY"
VARIANTS = ("CURRENT_FROZEN", "CURRENT_RESET", "TAIL_CURRENT_RESET", "DOC_BASE_RESET", "TAIL_BASE_RESET", "TAIL_CURRENT_FROZEN")
CONFIG = {
    "study_id": "argon4_gen4_candidate_20261106", "schema_version": 1,
    "tail_features": "ORDERED_BASE311_PLUS_INDEPENDENT_EXPERT65",
    "tail_quantiles": [0.32, 0.68], "quantiles_fit": "TRAIN_RAW_UNCLIPPED_RETURN_ONLY",
    "positive_sample_weight": "N_NEGATIVE_DIVIDED_BY_N_POSITIVE", "negative_sample_weight": 1.0,
    "hgb": {"max_iter": 40, "max_leaf_nodes": 11, "l2_regularization": 30.0,
            "learning_rate": 0.04, "early_stopping": False, "random_state": 42, "class_weight": None},
    "min_samples_leaf": "max(50,n_train//100)", "scaler_fit": "TRAIN_ONLY_FLOAT64",
    "tail_score": "P_UP_TAIL_MINUS_P_DOWN_TAIL", "normalization_fit": "VALIDATION_ONLY",
    "current_weight": 0.85, "tail_weight": 0.15,
    "threshold_quantile": "FROZEN_STREAM_QUANTILE_ON_VALIDATION_ABSOLUTE_SCORE",
    "primary_variant": "TAIL_CURRENT_RESET", "primary_comparator": "CURRENT_RESET",
    "also_must_beat": "CURRENT_FROZEN", "variants": list(VARIANTS),
    "tabnet": "NOT_IMPLEMENTED_MISSING_ORIGINAL_SPECIFICATION",
    "routing": "NONE_NO_TEST_SELECTION", "baseline_score_atol": 1e-12, "baseline_score_rtol": 1e-12,
    "maximum_cpu_threads": 2,
}


class CandidateError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_sha256():
    return hashlib.sha256(canonical(CONFIG).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def verify_frozen_code():
    if frozen.sha256(frozen.__file__) != BASE_CODE_SHA256:
        raise CandidateError("frozen baseline code changed")


def verify_historical_data():
    path = frozen.DATA_ROOT.parent / "data_manifest.json"
    if frozen.sha256(path) != DATA_MANIFEST_SHA256:
        raise CandidateError("historical data manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest["datasets"]:
        source = (ROOT / entry["path"]).resolve()
        if not source.is_relative_to(ROOT.resolve()) or frozen.sha256(source) != entry["sha256"]:
            raise CandidateError("historical source changed")
    return manifest


def validate_preregistration(path):
    path = Path(path).resolve()
    expected = (REPORT_ROOT / "preregistration.json").resolve()
    if path != expected or path.is_symlink():
        raise CandidateError("fixed candidate preregistration path required")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"status": "FROZEN", "study_id": CONFIG["study_id"],
                "primary_variant": CONFIG["primary_variant"], "primary_comparator": CONFIG["primary_comparator"],
                "also_must_beat": CONFIG["also_must_beat"], "config_sha256": config_sha256()}
    if any(value.get(key) != expected for key, expected in required.items()):
        raise CandidateError("preregistration missing or incompatible with fixed experiment")
    return {"path": str(path), "sha256": frozen.sha256(path), "document": value}


def _baseline_reference(bundle):
    manifest = bundle["manifest"]
    return {"stream": manifest["stream"], "phase": manifest["phase"],
            "manifest_sha256": bundle["manifest_sha256"], "model_sha256": manifest["model_sha256"],
            "model_filename": manifest["model_filename"], "model_version": manifest["model_version"],
            "training_code_sha256": manifest["training_code_sha256"]}


def load_baseline(stream, phase, expected_reference=None):
    verify_frozen_code()
    if stream not in frozen.STREAMS or phase not in frozen.FOLDS:
        raise CandidateError("unknown stream or phase")
    expected = expected_reference.get("manifest_sha256") if expected_reference else None
    bundle = frozen.load_model_bundle(stream, phase, expected_manifest_sha256=expected)
    if bundle["manifest"].get("training_code_sha256") != BASE_CODE_SHA256:
        raise CandidateError("baseline training code pin mismatch")
    if expected_reference and _baseline_reference(bundle) != expected_reference:
        raise CandidateError("baseline reference mismatch")
    return bundle


def tail_training_targets(raw_returns):
    y = np.asarray(raw_returns, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise CandidateError("finite one-dimensional training returns required")
    q32, q68 = np.quantile(y, CONFIG["tail_quantiles"])
    result = {"q32": float(q32), "q68": float(q68)}
    for name, positive in (("up", y > q68), ("down", y < q32)):
        n_positive, n_negative = int(positive.sum()), int((~positive).sum())
        if not n_positive or not n_negative:
            raise CandidateError("both tail classes must be observed")
        weight = n_negative / n_positive
        result[name] = {"labels": positive.astype(np.int32), "sample_weight": np.where(positive, weight, 1.0),
                        "positive_count": n_positive, "negative_count": n_negative, "positive_weight": weight}
    return result


def _fit_tail(x_train, raw_returns, x_validation):
    train = np.ascontiguousarray(x_train, dtype=np.float64)
    validation = np.ascontiguousarray(x_validation, dtype=np.float64)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != 376 or validation.shape[1] != 376:
        raise CandidateError("tail estimator requires exactly 376 ordered features")
    if len(raw_returns) != len(train) or not len(validation) or not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise CandidateError("invalid tail training or validation matrix")
    targets = tail_training_targets(raw_returns)
    scaler = StandardScaler().fit(train)
    tr, va = scaler.transform(train), scaler.transform(validation)
    estimators = {}
    leaf = max(50, len(train)//100)
    for name in ("up", "down"):
        estimator = HistGradientBoostingClassifier(**CONFIG["hgb"], min_samples_leaf=leaf)
        estimator.fit(tr, targets[name]["labels"], sample_weight=targets[name]["sample_weight"])
        estimators[name] = estimator
    val_raw = estimators["up"].predict_proba(va)[:, 1] - estimators["down"].predict_proba(va)[:, 1]
    norm = frozen._normalizer(val_raw)
    metadata = {"q32": targets["q32"], "q68": targets["q68"], "min_samples_leaf": leaf,
                **{name: {k: v for k, v in targets[name].items() if k not in ("labels", "sample_weight")}
                   for name in ("up", "down")}}
    return {"scaler": scaler, **estimators, "normalization": norm}, frozen._z(val_raw, norm), metadata


def _tail_score(tail, x):
    transformed = tail["scaler"].transform(np.ascontiguousarray(x, dtype=np.float64))
    raw = tail["up"].predict_proba(transformed)[:, 1] - tail["down"].predict_proba(transformed)[:, 1]
    return frozen._z(raw, tail["normalization"])


def _row_receipt(index, labels, positions):
    positions = np.asarray(positions, dtype=np.int64)
    return {"count": len(positions), "first_bar": str(index[positions[0]]) if len(positions) else None,
            "last_bar": str(index[positions[-1]]) if len(positions) else None,
            "positions_sha256": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest(),
            "bar_times_sha256": hashlib.sha256(index.asi8[positions].astype("<i8").tobytes()).hexdigest(),
            "raw_returns_sha256": hashlib.sha256(np.asarray(labels[positions], dtype="<f8").tobytes()).hexdigest()}


def fit_candidate(stream, phase, frames, baseline_bundle=None, *, provenance=None):
    """Fit two tail classifiers only; all baseline estimators remain untouched."""
    verify_frozen_code()
    if stream not in frozen.STREAMS or phase not in frozen.FOLDS:
        raise CandidateError("unknown stream or phase")
    baseline = baseline_bundle or load_baseline(stream, phase)
    base, expert = frames["base"], frames["expert"]
    if len(base.columns) != 311 or len(expert.columns) != 65 or not base.index.equals(expert.index) or not base.index.equals(frames["raw_target"].index):
        raise CandidateError("unaligned 311+65 candidate frames")
    if baseline["manifest"]["stream"] != stream or baseline["manifest"]["phase"] != phase:
        raise CandidateError("wrong baseline identity")
    _, k, quantile = frozen.STREAMS[stream]
    labels = frozen._label(frames["raw_target"], k)
    tr, va, te = frozen._split(base.index, labels, frozen.FOLDS[phase], k)
    validation_scores = frozen.predict_scores(baseline, base.iloc[va], expert.iloc[va])
    train_x = np.hstack([base.iloc[tr].to_numpy(np.float32), expert.iloc[tr].to_numpy(np.float32)])
    validation_x = np.hstack([base.iloc[va].to_numpy(np.float32), expert.iloc[va].to_numpy(np.float32)])
    tail, tail_z, tail_metadata = _fit_tail(train_x, labels[tr], validation_x)
    current_norm = frozen._normalizer(validation_scores["LOCAL_V4_FROZEN"])
    base_norm = frozen._normalizer(validation_scores["BASE"])
    current_z = frozen._z(validation_scores["LOCAL_V4_FROZEN"], current_norm)
    base_z = frozen._z(validation_scores["BASE"], base_norm)
    current_mix = .85*current_z + .15*tail_z
    base_mix = .85*base_z + .15*tail_z
    normalizers = {"current": current_norm, "base": base_norm,
                   "tail_current": frozen._normalizer(current_mix), "tail_base": frozen._normalizer(base_mix)}
    primary_z = frozen._z(current_mix, normalizers["tail_current"])
    base_tail_z = frozen._z(base_mix, normalizers["tail_base"])
    threshold = lambda z: float(np.quantile(np.abs(z), quantile))
    thresholds = {"CURRENT_FROZEN": baseline["package"]["threshold_frozen"], "CURRENT_RESET": threshold(current_z),
                  "TAIL_CURRENT_RESET": threshold(primary_z), "DOC_BASE_RESET": threshold(base_z),
                  "TAIL_BASE_RESET": threshold(base_tail_z), "TAIL_CURRENT_FROZEN": baseline["package"]["threshold_frozen"]}
    package = {"tail": tail, "normalizers": normalizers, "thresholds": thresholds,
               "base_columns": list(base.columns), "expert_columns": list(expert.columns),
               "baseline_reference": _baseline_reference(baseline), "config_sha256": config_sha256()}
    manifest = {"classification": CLASSIFICATION, "purpose": "RESEARCH_CANDIDATE_ONLY_NOT_PUBLISHED",
                "stream": stream, "phase": phase, "model_version": f"local311_65_tail15_{phase}_v1",
                "fold": dict(frozen.FOLDS[phase]), "config": CONFIG, "config_sha256": config_sha256(),
                "baseline_reference": package["baseline_reference"], "baseline_code_sha256": BASE_CODE_SHA256,
                "candidate_code_sha256": frozen.sha256(__file__), "data_manifest_sha256": DATA_MANIFEST_SHA256,
                "base_columns": list(base.columns), "expert_columns": list(expert.columns),
                "tail_columns": list(base.columns)+list(expert.columns), "tail_metadata": tail_metadata,
                "normalizers": {**normalizers, "tail": tail["normalization"]}, "thresholds": thresholds,
                "stream_quantile": quantile, "train_stride": 2,
                "rows": {name: _row_receipt(base.index, labels, ix) for name, ix in (("train", tr), ("validation", va), ("test", te))},
                "train_last_label_end": str(base.index[tr[-1]]+pd.Timedelta(minutes=5*(k+1))),
                "validation_last_label_end": str(base.index[va[-1]]+pd.Timedelta(minutes=5*(k+1))),
                "feature_provenance": frames.get("provenance", {}), "run_provenance": provenance or {},
                "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "python_version": platform.python_version(),
                "numeric_policy": "float32 stored features; C-contiguous float64 scaler and estimators",
                "live_enabled": False, "production_published": False}
    return {"package": package, "manifest": manifest, "baseline": baseline}


def predict_candidate_scores(bundle, base_frame, expert_frame):
    package = bundle["package"]
    if list(base_frame.columns) != package["base_columns"] or list(expert_frame.columns) != package["expert_columns"]:
        raise CandidateError("candidate ordered feature schema mismatch")
    if not base_frame.index.equals(expert_frame.index) or not len(base_frame):
        raise CandidateError("candidate feature time index mismatch")
    legacy = frozen.predict_scores(bundle["baseline"], base_frame, expert_frame)
    x = np.hstack([base_frame.to_numpy(np.float32), expert_frame.to_numpy(np.float32)])
    tail_z = _tail_score(package["tail"], x)
    normalizers = package["normalizers"]
    current = frozen._z(legacy["LOCAL_V4_FROZEN"], normalizers["current"])
    base = frozen._z(legacy["BASE"], normalizers["base"])
    primary = frozen._z(.85*current+.15*tail_z, normalizers["tail_current"])
    return {"CURRENT_FROZEN": legacy["LOCAL_V4_FROZEN"], "CURRENT_RESET": current,
            "TAIL_CURRENT_RESET": primary, "DOC_BASE_RESET": base,
            "TAIL_BASE_RESET": frozen._z(.85*base+.15*tail_z, normalizers["tail_base"]),
            "TAIL_CURRENT_FROZEN": primary*normalizers["current"]["std"]+normalizers["current"]["mean"]}


def _candidate_folder(stream, phase):
    if stream not in frozen.STREAMS or phase not in frozen.FOLDS:
        raise CandidateError("unknown candidate stream or phase")
    root = MODEL_ROOT.resolve()
    if root == frozen.MODEL_ROOT.resolve() or root.is_relative_to(frozen.MODEL_ROOT.resolve()):
        raise CandidateError("candidate output cannot overlap baseline model root")
    folder = MODEL_ROOT / phase
    if folder.is_symlink() or not folder.resolve().is_relative_to(root):
        raise CandidateError("candidate output path not trusted")
    return folder


def save_candidate_bundle(bundle):
    manifest = dict(bundle["manifest"])
    stream, phase = manifest["stream"], manifest["phase"]
    folder = _candidate_folder(stream, phase)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stream}.joblib"
    manifest_path = folder / f"{stream}.manifest.json"
    if path.exists() or manifest_path.exists():
        raise CandidateError("candidate artifacts already exist; refusing overwrite")
    temporary = path.with_suffix(".joblib.tmp")
    joblib.dump(bundle["package"], temporary, compress=3)
    temporary.replace(path)
    manifest.update(model_filename=path.name, model_sha256=frozen.sha256(path), created_at=_now(),
                    serialization="Locally generated joblib only; never accept external pickle")
    _write_json(manifest_path, manifest)
    digest = frozen.sha256(manifest_path)
    manifest_path.with_suffix(".json.sha256").write_text(digest+"\n", encoding="ascii")
    return digest


def load_candidate_bundle(stream, phase, expected_manifest_sha256):
    verify_frozen_code()
    folder = _candidate_folder(stream, phase)
    path = folder / f"{stream}.manifest.json"
    if path.is_symlink() or not expected_manifest_sha256 or frozen.sha256(path) != expected_manifest_sha256:
        raise CandidateError("candidate manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("stream") != stream or manifest.get("phase") != phase or manifest.get("classification") != CLASSIFICATION
            or manifest.get("candidate_code_sha256") != frozen.sha256(__file__)
            or manifest.get("baseline_code_sha256") != BASE_CODE_SHA256
            or manifest.get("config_sha256") != config_sha256() or manifest.get("config") != CONFIG
            or manifest.get("sklearn_version") != sklearn.__version__ or manifest.get("numpy_version") != np.__version__):
        raise CandidateError("candidate identity, code, config or runtime mismatch")
    if manifest.get("model_filename") != f"{stream}.joblib":
        raise CandidateError("candidate filename not allowed")
    model_path = folder / manifest["model_filename"]
    if model_path.is_symlink() or frozen.sha256(model_path) != manifest.get("model_sha256"):
        raise CandidateError("candidate model hash mismatch")
    baseline = load_baseline(stream, phase, manifest["baseline_reference"])
    package = joblib.load(model_path)
    for key in ("base_columns", "expert_columns", "baseline_reference", "thresholds", "config_sha256"):
        if package.get(key) != manifest.get(key):
            raise CandidateError("candidate persisted schema mismatch")
    return {"package": package, "manifest": manifest, "manifest_sha256": expected_manifest_sha256, "baseline": baseline}


def evaluate_rows(bundle, frames, split="test", *, start=None, end=None, row_indices=None):
    """Score all rows, retaining ties/unsettled labels. Recent bounds are decision UTC.

    Recent snapshots are retrospective publication assumptions, not proof that
    those observations or candidate models were available at the historic time.
    """
    manifest = bundle["manifest"]
    stream, phase = manifest["stream"], manifest["phase"]
    base, expert = frames["base"], frames["expert"]
    _, k, _ = frozen.STREAMS[stream]
    labels = frozen._label(frames["raw_target"], k)
    if split in ("validation", "test"):
        if any(x is not None for x in (start, end, row_indices)):
            raise CandidateError("frozen fold evaluation does not accept row overrides")
        _, va, te = frozen._split(base.index, labels, manifest["fold"], k)
        chosen = va if split == "validation" else te
    elif split == "recent":
        if phase != "deployment_q4":
            raise CandidateError("recent research evaluation requires deployment candidate")
        decision = base.index+pd.Timedelta(minutes=5)
        due = np.asarray(decision.minute % (k*5) == 0)
        if row_indices is not None:
            if start is not None or end is not None:
                raise CandidateError("choose explicit indices or recent bounds")
            chosen = np.asarray(row_indices)
            if chosen.ndim != 1 or not np.issubdtype(chosen.dtype, np.integer) or (len(chosen) and
                (chosen.min() < 0 or chosen.max() >= len(base) or np.any(np.diff(chosen) <= 0))):
                raise CandidateError("ordered unique in-bounds row indices required")
            chosen = chosen.astype(np.int64)
        else:
            if start is None or end is None or frozen._timestamp(start) > frozen._timestamp(end):
                raise CandidateError("explicit ordered recent decision bounds required")
            chosen = np.flatnonzero((decision >= frozen._timestamp(start)) & (decision <= frozen._timestamp(end)) & due)
        if len(chosen) and (not due[chosen].all() or (decision[chosen] < frozen._timestamp(manifest["fold"]["valid_from"])).any()
                           or (decision[chosen] >= frozen._timestamp(manifest["fold"]["valid_until"])).any()):
            raise CandidateError("recent rows outside model validity or due clock")
    else:
        raise CandidateError("unsupported evaluation split")
    if not len(chosen):
        return pd.DataFrame()
    selected_base, selected_expert = base.iloc[chosen], expert.iloc[chosen]
    scores = predict_candidate_scores(bundle, selected_base, selected_expert)
    returns = labels[chosen]
    records = []
    for variant in VARIANTS:
        score = scores[variant]
        threshold = bundle["package"]["thresholds"][variant]
        active = np.abs(score) >= threshold
        settled = active & np.isfinite(returns)
        records.append(pd.DataFrame({"stream": stream, "phase": phase, "split": split, "variant": variant,
            "bar_open_utc": selected_base.index, "decision_close_utc": selected_base.index+pd.Timedelta(minutes=5),
            "score": score, "threshold": threshold, "selected": active, "side": np.where(score > 0, "UP", "DOWN"),
            "forward_return": returns, "label_tie": returns == 0, "settled": settled,
            "win": settled & ((score > 0) == (returns >= 0)),
            "model_sha256": manifest.get("model_sha256"), "manifest_sha256": bundle.get("manifest_sha256"),
            "baseline_model_sha256": manifest["baseline_reference"]["model_sha256"], "classification": CLASSIFICATION,
            "availability_scope": "RETROSPECTIVE_PUBLICATION_ASSUMPTION_NOT_HISTORIC_TRADEABILITY" if split == "recent" else "HISTORICAL_RESEARCH_PREVIOUSLY_INSPECTED"}))
    return pd.concat(records, ignore_index=True)


def assert_baseline_cache(stream, test_predictions, cache_path=None):
    path = Path(cache_path) if cache_path else BASELINE_REPORT_ROOT / f"{stream}_oos_predictions.parquet"
    old = pd.read_parquet(path)
    old = old.loc[old.variant == "LOCAL_V4_FROZEN"].reset_index(drop=True)
    new = test_predictions.loc[test_predictions.variant == "CURRENT_FROZEN"].reset_index(drop=True)
    if len(old) != len(new) or not len(old):
        raise CandidateError("baseline cache row-count mismatch")
    for column in ("stream", "bar_open_utc", "decision_close_utc", "threshold", "selected", "side", "label_tie", "settled", "win"):
        if not old[column].equals(new[column]):
            raise CandidateError(f"baseline cache row mismatch: {column}")
    if not np.array_equal(old.forward_return.to_numpy(), new.forward_return.to_numpy(), equal_nan=True):
        raise CandidateError("baseline cache label mismatch")
    if not np.array_equal(old.model_sha256.to_numpy(), new.baseline_model_sha256.to_numpy()):
        raise CandidateError("baseline cache artifact mismatch")
    delta = np.abs(old.score.to_numpy()-new.score.to_numpy())
    if not np.allclose(old.score, new.score, rtol=CONFIG["baseline_score_rtol"], atol=CONFIG["baseline_score_atol"]):
        raise CandidateError("baseline cache score mismatch")
    return {"status": "PASS", "path": str(path), "sha256": frozen.sha256(path), "rows": len(old),
            "max_abs_score_difference": float(delta.max()), "all_decisions_and_labels_exact": True}


def verify_score_parity(before, after, base, expert):
    original = predict_candidate_scores(before, base, expert)
    restored = predict_candidate_scores(after, base, expert)
    for name in VARIANTS:
        if not np.array_equal(original[name], restored[name]):
            raise CandidateError("saved candidate batch parity failed")
    maximum = 0.0
    for index in sorted({0, len(base)//2, len(base)-1}):
        single = predict_candidate_scores(after, base.iloc[[index]], expert.iloc[[index]])
        for name in VARIANTS:
            a, b = single[name], restored[name][[index]]
            if not np.allclose(a, b, rtol=1e-12, atol=1e-12):
                raise CandidateError("candidate one-row/batch score parity failed")
            threshold = after["package"]["thresholds"][name]
            if not np.array_equal(np.abs(a) >= threshold, np.abs(b) >= threshold) or not np.array_equal(a > 0, b > 0):
                raise CandidateError("candidate one-row/batch decision parity failed")
            maximum = max(maximum, float(np.max(np.abs(a-b))))
    return {"saved_batch": "EXACT", "one_row_batch": "PASS", "maximum_one_row_difference": maximum}


def train_all_candidate(*, preregistration_path, threads=2, phases=("oos_2026", "deployment_q4"), streams=None):
    if not 1 <= threads <= 2 or not phases or any(p not in frozen.FOLDS for p in phases) or len(set(phases)) != len(phases):
        raise CandidateError("candidate threads or phases invalid")
    selected = list(streams or frozen.STREAMS)
    if not selected or len(set(selected)) != len(selected) or any(s not in frozen.STREAMS for s in selected):
        raise CandidateError("candidate streams invalid")
    preregistration = validate_preregistration(preregistration_path)
    verify_frozen_code()
    verify_historical_data()
    policy_path = ROOT / "config/argon4_forward_policy.json"
    provenance = {"preregistration_path": preregistration["path"], "preregistration_sha256": preregistration["sha256"],
                  "paper_policy_reference": str(policy_path), "paper_policy_sha256": frozen.sha256(policy_path),
                  "training_script_sha256": frozen.sha256(ROOT / "scripts/train_argon4_gen4_candidate.py")}
    run = {"status": "TRAINING", "started_at": _now(), "config": CONFIG, "config_sha256": config_sha256(),
           "threads": threads, "phases": list(phases), "streams": selected, "provenance": provenance,
           "completed": [], "production_published": False, "live_enabled": False}
    _write_json(REPORT_ROOT / "run.json", run)
    try:
        with threadpool_limits(limits=threads):
            for symbol in frozen.SYMBOLS:
                own = [s for s in selected if frozen.STREAMS[s][0] == symbol]
                if not own:
                    continue
                print(f"FEATURES {symbol}", flush=True)
                frames = frozen.build_feature_frames(frozen.DATA_ROOT, symbol)
                for stream in own:
                    for phase in phases:
                        print(f"FIT_TAIL {stream} {phase}", flush=True)
                        baseline = load_baseline(stream, phase)
                        candidate = fit_candidate(stream, phase, frames, baseline, provenance=provenance)
                        pin = save_candidate_bundle(candidate)
                        restored = load_candidate_bundle(stream, phase, pin)
                        labels = frozen._label(frames["raw_target"], frozen.STREAMS[stream][1])
                        _, va, _ = frozen._split(frames["base"].index, labels, frozen.FOLDS[phase], frozen.STREAMS[stream][1])
                        parity_rows = va[np.linspace(0, len(va)-1, min(64, len(va)), dtype=int)]
                        parity = verify_score_parity(candidate, restored, frames["base"].iloc[parity_rows], frames["expert"].iloc[parity_rows])
                        outputs, baseline_check = {}, None
                        for split in ("validation", "test"):
                            prediction = evaluate_rows(restored, frames, split)
                            if prediction.empty:
                                continue
                            if split == "test":
                                baseline_check = assert_baseline_cache(stream, prediction)
                            path = REPORT_ROOT / phase / f"{stream}_{split}_predictions.parquet"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            if path.exists():
                                raise CandidateError("candidate prediction output already exists")
                            prediction.to_parquet(path, index=False)
                            outputs[split] = {"path": str(path), "sha256": frozen.sha256(path), "rows": len(prediction)}
                            del prediction
                        item = {"stream": stream, "phase": phase, "manifest_sha256": pin,
                                "model_sha256": restored["manifest"]["model_sha256"], "parity": parity,
                                "baseline_cache_check": baseline_check, "outputs": outputs}
                        run["completed"].append(item)
                        _write_json(REPORT_ROOT / "run.json", run)
                        print("COMPLETE " + canonical(item), flush=True)
                        del baseline, candidate, restored
                del frames
                gc.collect()
        run.update(status="COMPLETED", completed_at=_now())
    except Exception as exc:
        run.update(status="FAILED", error=type(exc).__name__, failed_at=_now())
        _write_json(REPORT_ROOT / "run.json", run)
        raise
    _write_json(REPORT_ROOT / "run.json", run)
    return run
