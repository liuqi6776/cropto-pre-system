"""Non-mutating bimonthly TWAP60 + exponential half-life decay trainer for ARGON-4.

Preserves exact byte-level SHA256 of frozen ``tools/argon4_forward/model.py``
and ``SOURCE_HASHES`` while implementing:
1. 1-minute Spot OHLC4 TWAP60 labels (``SPOT_TWAP60`` and ``SPOT_TWAP60_BLEND``)
   aligned to Polymarket's Chainlink 60-second TWAP settlement rule;
2. Full ETF-era history (from ``2024-01-05``) weighted by exponential sample decay
   (``half_life_days = 270.0``, 9-month half-life) across ``StandardScaler``,
   ``Ridge``, and ``HistGradientBoostingClassifier``;
3. Pure ``Spot TWAP60`` validation accuracy weight selection (``select_weight``)
   over a rolling 60-to-90-day validation window and extended ``valid_until``
   deployment window (defusing the ``2027-01-01T00:00:00Z`` hard stop).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import model, model_v14

FROZEN_MODEL_PY_SHA256 = "1d9629abf05c27eeec0ed6e63ae4b4fc65c4f2b904dca4de9549698cf1a009f7"
LABEL_MODES = ("BINANCE_PERP_POINT", "SPOT_TWAP60", "SPOT_TWAP60_BLEND")

BIMONTHLY_FOLDS = {
    "deployment_q4": {
        "train_start": "2024-01-05T00:00:00Z",
        "train_end": "2026-09-10T21:55:00Z",
        "validation_start": "2026-09-11T00:00:00Z",
        "validation_end": "2026-11-11T21:55:00Z",
        "test_start": None,
        "test_end": None,
        "valid_from": "2026-10-01T00:00:00Z",
        "valid_until": "2027-03-01T00:00:00Z",
        "half_life_days": 270.0,
        "label_mode": "SPOT_TWAP60_BLEND",
    },
    "deployment_2027_q1": {
        "train_start": "2024-01-05T00:00:00Z",
        "train_end": "2026-09-30T21:55:00Z",
        "validation_start": "2026-10-01T00:00:00Z",
        "validation_end": "2026-11-30T21:55:00Z",
        "test_start": None,
        "test_end": None,
        "valid_from": "2026-12-01T00:00:00Z",
        "valid_until": "2027-03-01T00:00:00Z",
        "half_life_days": 270.0,
        "label_mode": "SPOT_TWAP60_BLEND",
    },
}


def verify_frozen_model_untouched() -> str:
    actual = model.sha256(Path(model.__file__))
    if actual != FROZEN_MODEL_PY_SHA256:
        raise model.ModelInputError(
            f"frozen model.py hash mismatch: expected {FROZEN_MODEL_PY_SHA256}, got {actual}"
        )
    return actual


def compute_exponential_sample_weights(
    train_index: pd.DatetimeIndex,
    train_end: str | pd.Timestamp,
    half_life_days: float | None,
) -> np.ndarray | None:
    if half_life_days is None or float(half_life_days) <= 0:
        return None
    hl = float(half_life_days)
    end_ts = model._timestamp(train_end)
    idx_utc = pd.to_datetime(train_index, utc=True)
    age_days = (end_ts - idx_utc).total_seconds().to_numpy(dtype=np.float64) / 86400.0
    return np.power(0.5, np.maximum(0.0, age_days) / hl)


def fit_family_weighted(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    expert: bool = False,
    sample_weight: np.ndarray | None = None,
):
    x_train = np.ascontiguousarray(x_train, dtype=np.float64)
    x_validation = np.ascontiguousarray(x_validation, dtype=np.float64)
    sw = np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else None
    if sw is not None:
        if sw.shape != (len(x_train),) or not np.isfinite(sw).all() or (sw <= 0).any():
            raise model.ModelInputError("invalid exponential sample_weight vector")
    scaler = StandardScaler().fit(x_train, sample_weight=sw)
    tr, va = scaler.transform(x_train), scaler.transform(x_validation)
    ridge = Ridge(alpha=5000.0, random_state=42).fit(tr, y_train, sample_weight=sw)
    hgb = HistGradientBoostingClassifier(
        max_iter=50 if expert else 60,
        learning_rate=0.03 if expert else 0.04,
        max_leaf_nodes=7 if expert else 15,
        min_samples_leaf=max(100, len(tr) // 60) if expert else max(50, len(tr) // 100),
        l2_regularization=40.0 if expert else 20.0,
        early_stopping=False,
        random_state=42,
    ).fit(tr, (y_train > 0).astype(np.int32), sample_weight=sw)
    r, h = ridge.predict(va), hgb.predict_proba(va)[:, 1]
    rp, hp = model._normalizer(r), model._normalizer(h)
    ensemble = 0.5 * model._z(r, rp) + 0.5 * model._z(h, hp)
    ep = model._normalizer(ensemble)
    family = {
        "scaler": scaler,
        "ridge": ridge,
        "hgb": hgb,
        "normalization": {"ridge": rp, "hgb": hp, "ensemble": ep},
    }
    return family, model._z(ensemble, ep)


def build_twap60_labels(
    raw: pd.DataFrame,
    k: int,
    *,
    index: pd.DatetimeIndex | None = None,
    data_root: Path | str = model.DATA_ROOT,
    symbol: str | None = None,
    mode: str = "BINANCE_PERP_POINT",
    spot_1m_frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (train_labels, eval_twap_labels) with smooth fallback if spot_1m is partial."""
    if mode not in LABEL_MODES:
        raise model.ModelInputError(f"unsupported label mode: {mode}")
    result = np.full(len(raw), np.nan, dtype=np.float64)
    o, c = raw["open"].to_numpy(dtype=np.float64), raw["close"].to_numpy(dtype=np.float64)
    result[:-k] = c[k:] / o[1 : len(o) - k + 1] - 1.0

    if mode in ("SPOT_TWAP60", "SPOT_TWAP60_BLEND") and index is not None and (spot_1m_frame is not None or symbol is not None):
        s1 = spot_1m_frame
        if s1 is None and symbol is not None:
            spot_1m_path = Path(data_root) / "spot_1m" / f"{symbol}_1m_spot.parquet"
            if spot_1m_path.is_file():
                s1 = pd.read_parquet(spot_1m_path)
        if s1 is not None and not s1.empty:
            if "bound_ts" in s1.columns:
                s1 = s1.set_index(pd.to_datetime(s1["bound_ts"], utc=True))
            s1.index = pd.to_datetime(s1.index, utc=True)
            if s1.index.has_duplicates:
                s1 = s1.loc[~s1.index.duplicated(keep="last")]
            close_ts = pd.to_datetime(index, utc=True) + pd.Timedelta(minutes=5)
            tw60 = s1["twap60"].reindex(close_ts).to_numpy(dtype=np.float64)
            spc_col = "spot_close" if "spot_close" in s1.columns else "close"
            spc = s1[spc_col].reindex(close_ts).to_numpy(dtype=np.float64)
            twap_ret = np.full(len(raw), np.nan, dtype=np.float64)
            twap_ret[:-k] = tw60[k:] / tw60[:-k] - 1.0
            if mode == "SPOT_TWAP60_BLEND":
                s2t_ret = np.full(len(raw), np.nan, dtype=np.float64)
                s2t_ret[:-k] = tw60[k:] / spc[:-k] - 1.0
                target_ret = 0.5 * twap_ret + 0.5 * s2t_ret
            else:
                target_ret = twap_ret
            has_twap = np.isfinite(target_ret)
            result[has_twap] = target_ret[has_twap]
            eval_ret = np.where(np.isfinite(twap_ret), twap_ret, result)
            return result, eval_ret
    return result, result.copy()


def train_twap60_streams(
    *,
    data_root: Path | str,
    model_root: Path | str = model.MODEL_ROOT,
    report_root: Path | str | None = None,
    phase: str = "deployment_q4",
    fold_override: dict | None = None,
    selected_streams: tuple[str, ...] | list[str] | None = None,
    threads: int = 3,
    model_version_tag: str | None = None,
) -> dict:
    """Train and verify model.py-compatible bundles without mutating frozen sources."""
    if not 1 <= threads <= 4:
        raise model.ModelInputError("threads must be between 1 and 4")
    frozen_sha = verify_frozen_model_untouched()
    data_root = Path(data_root).resolve()
    model_root = Path(model_root).resolve()
    if report_root is None:
        report_root = model.ROOT / "reports" / f"argon4_twap60_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    report_root = Path(report_root).resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    base_fold = dict(BIMONTHLY_FOLDS.get(phase, model.FOLDS.get(phase, BIMONTHLY_FOLDS["deployment_q4"])))
    if fold_override:
        base_fold.update(fold_override)
    label_mode = str(base_fold.get("label_mode", "SPOT_TWAP60_BLEND"))
    half_life_days = base_fold.get("half_life_days", 270.0)

    streams = list(selected_streams or model.STREAMS.keys())
    if any(s not in model_v14.ALL_STREAMS for s in streams):
        raise model.ModelInputError("unknown training stream")

    version_str = model_version_tag or f"local311_independent65_{phase}_{label_mode.lower()}_hl{int(half_life_days or 0)}"
    run_summary = {
        "status": "TRAINING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "classification": model.CLASSIFICATION,
        "phase": phase,
        "fold": base_fold,
        "label_mode": label_mode,
        "half_life_days": half_life_days,
        "frozen_model_py_sha256": frozen_sha,
        "trainer_code_sha256": model.sha256(Path(__file__)),
        "source_sha256": dict(model.SOURCE_HASHES),
        "completed": [],
        "deployment_manifest_sha256": {},
    }

    symbols_needed = []
    for s in streams:
        sym = model_v14.ALL_STREAMS[s][0]
        if sym not in symbols_needed:
            symbols_needed.append(sym)

    with threadpool_limits(limits=threads):
        for symbol in symbols_needed:
            own_streams = [s for s in streams if model_v14.ALL_STREAMS[s][0] == symbol]
            t_sym = time.monotonic()
            frames = model_v14.build_feature_frames_v14(data_root, symbol)
            x = frames["base"].to_numpy(np.float32)
            xd = frames["expert"].to_numpy(np.float32)
            spot_1m_path = data_root / "spot_1m" / f"{symbol}_1m_spot.parquet"
            spot_1m_df = pd.read_parquet(spot_1m_path) if spot_1m_path.is_file() else None
            spot_1m_sha = model.sha256(spot_1m_path) if spot_1m_path.is_file() else None

            for stream in own_streams:
                t0 = time.monotonic()
                _, k, quantile = model_v14.ALL_STREAMS[stream]
                train_labels, eval_twap_labels = build_twap60_labels(
                    frames["raw_target"],
                    k,
                    index=frames["index"],
                    data_root=data_root,
                    symbol=symbol,
                    mode=label_mode,
                    spot_1m_frame=spot_1m_df,
                )
                tr, va, te = model._split(frames["index"], train_labels, base_fold, k)
                # Ensure validation labels also exclude exact zero ties during weight selection
                va_eval = eval_twap_labels[va]
                y = np.clip(train_labels[tr], -0.008 if k == 1 else -0.015, 0.008 if k == 1 else 0.015)
                sw = compute_exponential_sample_weights(frames["index"][tr], base_fold["train_end"], half_life_days)
                base_fam, z0 = fit_family_weighted(x[tr], y, x[va], expert=False, sample_weight=sw)
                exp_fam, zd = fit_family_weighted(xd[tr], y, xd[va], expert=True, sample_weight=sw)
                weight, threshold_reset, grid = model.select_weight(z0, zd, va_eval, quantile)
                threshold_frozen = float(np.quantile(np.abs(z0), quantile))

                package = {
                    "base": base_fam,
                    "expert": exp_fam,
                    "base_columns": list(frames["base"].columns),
                    "expert_columns": list(frames["expert"].columns),
                    "weight": weight,
                    "threshold_frozen": threshold_frozen,
                    "threshold_reset": threshold_reset,
                }
                manifest_fold = {
                    "train_start": base_fold["train_start"],
                    "train_end": base_fold["train_end"],
                    "validation_start": base_fold["validation_start"],
                    "validation_end": base_fold["validation_end"],
                    "test_start": base_fold.get("test_start"),
                    "test_end": base_fold.get("test_end"),
                    "valid_from": base_fold["valid_from"],
                    "valid_until": base_fold["valid_until"],
                    "half_life_days": half_life_days,
                    "label_mode": label_mode,
                }
                metadata = {
                    "stream": stream,
                    "phase": phase,
                    "classification": model.CLASSIFICATION,
                    "model_version": version_str,
                    "target_oracle": f"POLYMARKET_CHAINLINK_60S_TWAP_VIA_1M_{label_mode}",
                    "fold": manifest_fold,
                    "base_columns": package["base_columns"],
                    "expert_columns": package["expert_columns"],
                    "base_features": 311,
                    "expert_features": 65,
                    "train_stride": 2,
                    "train_rows": int(len(tr)),
                    "validation_rows": int(len(va)),
                    "test_rows": int(len(te)),
                    "sample_weight_policy": (
                        f"exponential_decay_half_life_{float(half_life_days):.1f}d"
                        if half_life_days
                        else "uniform"
                    ),
                    "sample_weight_min": float(sw.min()) if sw is not None else 1.0,
                    "sample_weight_max": float(sw.max()) if sw is not None else 1.0,
                    "weight": weight,
                    "weight_grid_validation": grid,
                    "threshold_frozen": threshold_frozen,
                    "threshold_reset": threshold_reset,
                    "train_last_label_end": str(frames["index"][tr[-1]] + pd.Timedelta(minutes=5 * (k + 1))),
                    "validation_last_label_end": str(frames["index"][va[-1]] + pd.Timedelta(minutes=5 * (k + 1))),
                    "training_data_last_bar": str(frames["index"][-1]),
                    "spot_1m_parquet_sha256": spot_1m_sha,
                    "source_sha256": dict(model.SOURCE_HASHES),
                    "frozen_model_code_sha256": frozen_sha,
                    "training_code_sha256": model.sha256(Path(__file__)),
                    "sklearn_version": sklearn.__version__,
                    "python_version": platform.python_version(),
                    "normalization_fit": "validation only",
                    "numeric_policy": "C-contiguous float64 scaler/estimators/normalization; float32 stored features",
                    "training_ties": "excluded from train/validation only; never filters test or online selection",
                    "purpose": "EXPLORATORY_LOCAL_PAPER_ONLY",
                }
                folder = (model_root / phase).resolve()
                folder.mkdir(parents=True, exist_ok=True)
                model_path = folder / f"{stream}.joblib"
                tmp_joblib = model_path.with_suffix(".joblib.tmp")
                joblib.dump(package, tmp_joblib, compress=3)
                tmp_joblib.replace(model_path)
                metadata = {
                    **metadata,
                    "model_filename": model_path.name,
                    "model_sha256": model.sha256(model_path),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "serialization": "joblib locally generated; never accept external pickle",
                }
                manifest_path = folder / f"{stream}.manifest.json"
                model._write_json(manifest_path, metadata)
                digest = model.sha256(manifest_path)
                manifest_path.with_suffix(".json.sha256").write_text(digest + "\n", encoding="ascii")

                # Verify exact score parity using frozen model.py / model_v14.py loader!
                if model_root == model.MODEL_ROOT.resolve():
                    reloaded = model_v14.load_model_bundle_v14(stream, phase=phase, expected_manifest_sha256=digest)
                else:
                    reloaded = {
                        "package": joblib.load(model_path),
                        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
                        "manifest_sha256": digest,
                    }
                chk_b = frames["base"].iloc[va[:64]]
                chk_e = frames["expert"].iloc[va[:64]]
                before_scores = model.predict_scores({"package": package}, chk_b, chk_e)
                after_scores = model.predict_scores(reloaded, chk_b, chk_e)
                if any(not np.array_equal(after_scores[key], before_scores[key]) for key in before_scores):
                    raise model.ModelInputError(f"serialized-model score parity failed for {stream}")
                if not (
                    np.allclose(after_scores["BASE"], z0[:64], rtol=1e-12, atol=1e-12)
                    and np.allclose(after_scores["EXPERT"], zd[:64], rtol=1e-12, atol=1e-12)
                ):
                    raise model.ModelInputError(f"training-inference batch score parity failed for {stream}")

                best_acc = max(r["accuracy"] for r in grid)
                item = {
                    "stream": stream,
                    "phase": phase,
                    "elapsed_seconds": round(time.monotonic() - t0, 2),
                    "manifest_sha256": digest,
                    "model_sha256": metadata["model_sha256"],
                    "train_rows": int(len(tr)),
                    "validation_rows": int(len(va)),
                    "weight": weight,
                    "threshold_frozen": threshold_frozen,
                    "validation_twap_accuracy": best_acc,
                    "saved_model_parity": "PASS",
                }
                run_summary["completed"].append(item)
                run_summary["deployment_manifest_sha256"][stream] = digest
                model._write_json(report_root / "twap60_retrain_run.json", run_summary)

            del frames, x, xd, spot_1m_df

    run_summary["status"] = "COMPLETED"
    run_summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    model._write_json(report_root / "twap60_retrain_run.json", run_summary)
    return run_summary
