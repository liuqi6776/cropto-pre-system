"""Locally trained ARGON research model: 311 base + independent 65-column expert.

This is not the author's unavailable 374-column model. Training, saved-model
backtesting and forward inference share the same feature and score functions.
No network, wallet, order submission, or prediction-market settlement is here.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import types

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "research_data/argon4_clean_room_20261105/raw_data"
MODEL_ROOT = ROOT / "research/argon4_local_v4_20261106"
REPORT_ROOT = ROOT / "reports/argon4_local_v4_20261106"
ORIGINAL = ROOT / "research/argon4_clean_room_20261105/original"
CLASSIFICATION = "LOCAL_311_PLUS_INDEPENDENT_65_NOT_ORIGINAL_374_EXPLORATORY"
SOURCE_HASHES = {
    "step2_standalone_clean_room_master.py": "0ec3e3889b8c78959457b698f7e5ed84371512dbacb780f4cbb256f26b1cd6da",
    "coin_specific_v4_divergence.py": "028f686a39e23dd52fbe168fb82ab40831a9ca05eab8f715d49889eaacef7c14",
}
STREAMS = {
    "BTC_15m": ("BTCUSDT", 3, .85), "ETH_15m": ("ETHUSDT", 3, .90),
    "BTC_5m": ("BTCUSDT", 1, .92), "ETH_5m": ("ETHUSDT", 1, .92),
    "SOL_15m": ("SOLUSDT", 3, .92), "SOL_5m": ("SOLUSDT", 1, .92),
    "BNB_15m": ("BNBUSDT", 3, .92),
}
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
WEIGHTS = (.08, .15, .22, .30)
FOLDS = {
    "oos_2026": {
        "train_start": "2024-01-05T00:00:00Z", "train_end": "2025-09-30T21:55:00Z",
        "validation_start": "2025-10-01T00:00:00Z", "validation_end": "2025-12-31T21:55:00Z",
        "test_start": "2026-01-01T00:00:00Z", "test_end": "2026-09-24T22:00:00Z",
        "valid_from": "2026-01-01T00:00:00Z", "valid_until": "2026-10-01T00:00:00Z",
    },
    "deployment_q4": {
        "train_start": "2024-01-05T00:00:00Z", "train_end": "2026-06-30T21:55:00Z",
        "validation_start": "2026-07-01T00:00:00Z", "validation_end": "2026-09-30T21:55:00Z",
        "test_start": None, "test_end": None,
        "valid_from": "2026-10-01T00:00:00Z", "valid_until": "2027-01-01T00:00:00Z",
    },
}


class ModelInputError(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _timestamp(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ModelInputError("timezone-aware timestamp required")
    return stamp.tz_convert("UTC")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _source_modules(reader, data_root):
    """Compile verified supplied formulas, removing the future-label block."""
    for name, expected in SOURCE_HASHES.items():
        if sha256(ORIGINAL / name) != expected:
            raise ModelInputError("frozen source hash mismatch")
    name = "argon4_forward_verified_features"
    module = types.ModuleType(name)
    module.__file__ = str(ORIGINAL / "step2_standalone_clean_room_master.py")
    sys.modules[name] = module
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_clean_room_bundle")
    cut = next(i for i, n in enumerate(builder.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "o_raw" for t in n.targets))
    builder.body = builder.body[:cut] + ast.parse("return {'index': idx, 'features': features_259, 'spec_features': spec_df}").body
    class RedirectReader(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "pd" and node.func.attr in ("read_parquet", "read_feather"):
                node.func = ast.Name(id="_guarded_read", ctx=ast.Load())
            return node
    tree = RedirectReader().visit(tree)
    ast.fix_missing_locations(tree)
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    module._locate_data_root = lambda: data_root
    module._guarded_read = reader
    expert = types.ModuleType("argon4_forward_verified_expert")
    expert.__file__ = str(ORIGINAL / "coin_specific_v4_divergence.py")
    exec(compile(Path(expert.__file__).read_text(encoding="utf-8"), expert.__file__, "exec"), expert.__dict__)
    return module, expert


def _validate_ohlc(frame, label, *, volume=False):
    cols = ["open", "high", "low", "close"]
    values = frame[cols].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ModelInputError(f"invalid observed OHLC: {label}")
    o, h, l, c = values.T
    if np.any(h < np.maximum.reduce([o, l, c])) or np.any(l > np.minimum.reduce([o, h, c])):
        raise ModelInputError(f"inconsistent observed OHLC: {label}")
    if volume:
        vals = frame[["volume", "quote_volume", "count", "taker_buy_quote_volume"]].to_numpy(float)
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ModelInputError(f"invalid observed volume: {label}")


def build_feature_frames(data_root=DATA_ROOT, symbol="BTCUSDT", *, as_of=None, tail_rows=None):
    """Return base/expert frames computed only from closed supplied observations.

    `data_root` uses the frozen raw_data schema (intraday + auxiliary). It may
    contain a shorter *observed* live history, but must have >= 1728 contiguous
    5m bars. Prefer >= 4320 bars for stable recursive indicator warmup. tail_rows
    limits returned rows, not input history. Any available_at column is enforced.
    Derived original indicator startup zeros remain; raw gaps are never filled.
    """
    root = Path(data_root).resolve()
    if not root.is_relative_to(ROOT) or symbol not in SYMBOLS:
        raise ModelInputError("unsupported local feature source or symbol")
    cutoff = _timestamp(as_of) if as_of is not None else None
    frames = {}
    fingerprints = {}
    availability = {}
    loaded = {}

    def reader(path):
        p = Path(path).resolve()
        if not p.is_relative_to(root) or p.suffix != ".parquet":
            raise ModelInputError("only explicit local parquet sources are accepted")
        if p in loaded:
            return loaded[p].copy()
        frame = pd.read_parquet(p)
        received_field = "first_received_at" if "first_received_at" in frame.columns else "available_at" if "available_at" in frame.columns else None
        if received_field:
            published = pd.to_datetime(frame[received_field], utc=True, errors="raise")
            if published.isna().any():
                raise ModelInputError("missing observation availability")
            if cutoff is not None:
                frame = frame.loc[published <= cutoff].copy()
        fingerprints[str(p.relative_to(ROOT))] = {"bytes": p.stat().st_size, "sha256": sha256(p),
            "availability_policy": "OBSERVED_FIRST_RECEIVED_AT" if received_field == "first_received_at" else "OBSERVED_AVAILABLE_AT" if received_field else "HISTORICAL_PUBLICATION_ASSUMPTION"}
        name = p.name
        if p.parent.name == "intraday":
            if not isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.set_index(pd.to_datetime(frame["open_time"], utc=True))
            frame.index = pd.to_datetime(frame.index, utc=True)
            if cutoff is not None:
                frame = frame.loc[frame.index + pd.Timedelta(minutes=5) <= cutoff]
            frame = frame.loc[frame.index >= pd.Timestamp("2024-01-01", tz="UTC")]
            if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
                raise ModelInputError("unsorted or duplicate raw candles")
            if len(frame) < 1728 or np.any(np.diff(frame.index.asi8) != 300_000_000_000):
                raise ModelInputError("at least 1728 contiguous observed 5m candles required")
            _validate_ohlc(frame, name, volume=True)
            availability[name] = {"last_available_time": str(frame.index[-1] + pd.Timedelta(minutes=5)), "rows": len(frame)}
        elif "_8h_funding" in name:
            stamps = pd.to_datetime(frame["date"], utc=True)
            if cutoff is not None:
                frame = frame.loc[stamps <= cutoff]
                stamps = stamps.loc[stamps <= cutoff]
            if stamps.empty or stamps.duplicated().any() or not stamps.is_monotonic_increasing or not np.isfinite(frame["funding_rate"].to_numpy(float)).all():
                raise ModelInputError("invalid observed funding events")
            availability[name] = {"last_available_time": str(stamps.iloc[-1]), "rows": len(frame)}
        elif "_1h_" in name:
            stamps = pd.to_datetime(frame["date"], utc=True) + pd.Timedelta(hours=1)
            if cutoff is not None:
                frame = frame.loc[stamps <= cutoff]
                stamps = stamps.loc[stamps <= cutoff]
            if stamps.empty or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
                raise ModelInputError("invalid hourly observations")
            _validate_ohlc(frame, name)
            availability[name] = {"last_available_time": str(stamps.iloc[-1]), "rows": len(frame)}
        elif name == "binance_basis_4h.parquet":
            frame.index = pd.to_datetime(frame.index, utc=True)
            stamps = frame.index + pd.Timedelta(hours=4)
            if cutoff is not None:
                frame = frame.loc[stamps <= cutoff]
                stamps = stamps[stamps <= cutoff]
            if len(frame) < 4 or frame.index.has_duplicates or not frame.index.is_monotonic_increasing or not np.isfinite(frame[list(SYMBOLS[:3])].to_numpy(float)).all():
                raise ModelInputError("invalid observed 4h basis")
            availability[name] = {"last_available_time": str(stamps[-1]), "rows": len(frame)}
        elif name == "deribit_dvol_daily_2021_2026.parquet":
            stamps = pd.to_datetime(frame["bucket_utc"], utc=True) + pd.Timedelta(days=1, minutes=5)
            if cutoff is not None:
                frame = frame.loc[stamps <= cutoff]
            for coin in ("BTC", "ETH"):
                sub = frame.loc[frame.symbol == coin]
                dates = pd.to_datetime(sub.bucket_utc, utc=True)
                if len(sub) < 9 or dates.duplicated().any() or not dates.is_monotonic_increasing:
                    raise ModelInputError("insufficient daily observed DVOL history")
                _validate_ohlc(sub, name)
                availability[f"DVOL_{coin}"] = {"last_available_time": str(dates.iloc[-1] + pd.Timedelta(days=1, minutes=5)), "rows": len(sub)}
        else:
            raise ModelInputError("unrecognized feature source")
        loaded[p] = frame.copy()
        return frame

    for coin in SYMBOLS:
        p = root / "intraday" / f"{coin}_5m_futures_2023_2026.parquet"
        if not p.exists():
            p = root / "intraday" / f"{coin}_5m_futures.parquet"
        frames[coin] = reader(p)
    idx = frames[symbol].index
    if any(not frame.index.equals(idx) for frame in frames.values()):
        raise ModelInputError("four observed candle timelines must match exactly")
    effective_close = idx[-1] + pd.Timedelta(minutes=5)
    if cutoff is not None and not pd.Timedelta(0) <= cutoff - effective_close <= pd.Timedelta(minutes=5):
        raise ModelInputError("latest observed closed candle is stale")
    # All helper inputs are read using close/availability cutoffs even for training.
    if cutoff is None:
        cutoff = effective_close
    base_module, expert_module = _source_modules(reader, root)
    bundle = base_module.build_clean_room_bundle(symbol)
    base = pd.concat([bundle["features"], bundle["spec_features"]], axis=1).astype(np.float32)
    ofi = sum((2 * frame.taker_buy_quote_volume - frame.quote_volume) / (frame.quote_volume + 1e-12) for frame in frames.values()) / 4
    expert = expert_module.build_coin_own_structural_and_v4_mag_gap(frames[symbol], frames["BTCUSDT"], ofi)
    if base.shape[1] != 311 or expert.shape[1] != 65 or not base.index.equals(expert.index):
        raise ModelInputError("unexpected verified feature schema")
    if not np.isfinite(base.to_numpy()).all() or not np.isfinite(expert.to_numpy()).all():
        raise ModelInputError("nonfinite model features")
    # Carrying a published derived feature is permitted, but stale source inputs fail.
    for name, meta in availability.items():
        age = effective_close - _timestamp(meta["last_available_time"])
        maximum = (pd.Timedelta(hours=8, minutes=5) if "funding" in name else
                   pd.Timedelta(hours=1, minutes=5) if "_1h_" in name else
                   pd.Timedelta(hours=4, minutes=5) if "basis" in name else
                   pd.Timedelta(days=1, minutes=10) if name.startswith("DVOL_") else pd.Timedelta(minutes=5))
        if age > maximum:
            raise ModelInputError(f"stale source input: {name}")
        meta["age_seconds_at_feature_close"] = age.total_seconds()
    raw_target = frames[symbol][["open", "close"]].copy()
    if tail_rows is not None:
        if not isinstance(tail_rows, int) or tail_rows < 1:
            raise ModelInputError("positive tail_rows required")
        base, expert, raw_target = base.tail(tail_rows), expert.tail(tail_rows), raw_target.tail(tail_rows)
    return {"base": base, "expert": expert, "index": base.index, "raw_target": raw_target,
            "availability": availability, "provenance": fingerprints,
            "classification": CLASSIFICATION, "source_sha256": dict(SOURCE_HASHES)}


def _normalizer(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ModelInputError("nonfinite validation scores")
    return {"mean": float(values.mean()), "std": float(values.std()) + 1e-12}


def _z(values, params):
    return (np.asarray(values, dtype=np.float64) - params["mean"]) / params["std"]


def _fit_family(x_train, y_train, x_validation, expert=False):
    # Ridge's small raw returns are amplified by score normalization. Keep both
    # fit and inference in C-contiguous float64, including one-row inference.
    x_train = np.ascontiguousarray(x_train, dtype=np.float64)
    x_validation = np.ascontiguousarray(x_validation, dtype=np.float64)
    scaler = StandardScaler().fit(x_train)
    tr, va = scaler.transform(x_train), scaler.transform(x_validation)
    ridge = Ridge(alpha=5000., random_state=42).fit(tr, y_train)
    hgb = HistGradientBoostingClassifier(max_iter=50 if expert else 60,
        learning_rate=.03 if expert else .04, max_leaf_nodes=7 if expert else 15,
        min_samples_leaf=max(100, len(tr)//60) if expert else max(50, len(tr)//100),
        l2_regularization=40. if expert else 20., early_stopping=False, random_state=42).fit(tr, (y_train > 0).astype(np.int32))
    r, h = ridge.predict(va), hgb.predict_proba(va)[:, 1]
    rp, hp = _normalizer(r), _normalizer(h)
    ensemble = .5 * _z(r, rp) + .5 * _z(h, hp)
    ep = _normalizer(ensemble)
    family = {"scaler": scaler, "ridge": ridge, "hgb": hgb,
              "normalization": {"ridge": rp, "hgb": hp, "ensemble": ep}}
    return family, _z(ensemble, ep)


def _score_family(family, x):
    transformed = family["scaler"].transform(np.ascontiguousarray(x, dtype=np.float64))
    norm = family["normalization"]
    r = _z(family["ridge"].predict(transformed), norm["ridge"])
    h = _z(family["hgb"].predict_proba(transformed)[:, 1], norm["hgb"])
    return _z(.5*r + .5*h, norm["ensemble"])


def select_weight(z_base, z_expert, validation_returns, quantile):
    candidates = []
    for weight in WEIGHTS:
        score = z_base + weight * z_expert
        threshold = float(np.quantile(np.abs(score), quantile))
        active = np.abs(score) >= threshold
        if not active.any():
            raise ModelInputError("empty validation selection")
        accuracy = float(((score[active] > 0) == (validation_returns[active] > 0)).mean())
        candidates.append({"weight": weight, "accuracy": accuracy, "selected": int(active.sum()), "threshold_reset": threshold})
    best = max(candidates, key=lambda row: row["accuracy"])
    return best["weight"], best["threshold_reset"], candidates


def _label(raw, k):
    result = np.full(len(raw), np.nan)
    o, c = raw.open.to_numpy(float), raw.close.to_numpy(float)
    result[:-k] = c[k:] / o[1:len(o)-k+1] - 1
    return result


def _split(index, label, fold, k):
    valid = np.isfinite(label) & (label != 0)
    clock = (index + pd.Timedelta(minutes=5)).minute % (k*5) == 0
    def inside(start, end):
        return np.asarray((index >= _timestamp(start)) & (index <= _timestamp(end)))
    tr = np.flatnonzero(inside(fold["train_start"], fold["train_end"]) & valid)[::2]
    va = np.flatnonzero(inside(fold["validation_start"], fold["validation_end"]) & valid & clock)
    if len(tr) < 1000 or len(va) < 100:
        raise ModelInputError("insufficient training/validation observations")
    if index[tr[-1]] + pd.Timedelta(minutes=5*(k+1)) >= _timestamp(fold["validation_start"]):
        raise ModelInputError("training label overlaps validation")
    if fold["test_start"]:
        if index[va[-1]] + pd.Timedelta(minutes=5*(k+1)) >= _timestamp(fold["test_start"]):
            raise ModelInputError("validation label overlaps test")
        # Selection has no future-return/tie filter. Unsettled labels are separate.
        te = np.flatnonzero(inside(fold["test_start"], fold["test_end"]) & clock)
    else:
        te = np.array([], dtype=int)
    return tr, va, te


def _save_stream(folder, stream, package, metadata):
    folder = Path(folder).resolve()
    if not folder.is_relative_to(MODEL_ROOT.resolve()):
        raise ModelInputError("model output outside fixed research root")
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / f"{stream}.joblib"
    temporary = model_path.with_suffix(".joblib.tmp")
    joblib.dump(package, temporary, compress=3)
    temporary.replace(model_path)
    metadata = {**metadata, "model_filename": model_path.name, "model_sha256": sha256(model_path),
                "created_at": _now(), "serialization": "joblib locally generated; never accept external pickle"}
    manifest_path = folder / f"{stream}.manifest.json"
    _write_json(manifest_path, metadata)
    digest = sha256(manifest_path)
    manifest_path.with_suffix(".json.sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def load_model_bundle(stream, phase="deployment_q4", *, expected_manifest_sha256=None):
    """Only load our own fixed-path artifacts after identity/hash/schema checks.

    For deployment pin expected_manifest_sha256 from the local training receipt.
    The adjacent checksum detects corruption, not a malicious local administrator.
    """
    if stream not in STREAMS or phase not in FOLDS:
        raise ModelInputError("unknown stream or phase")
    folder = (MODEL_ROOT / phase).resolve()
    manifest_path = folder / f"{stream}.manifest.json"
    if not folder.is_relative_to(MODEL_ROOT.resolve()) or manifest_path.is_symlink():
        raise ModelInputError("model path not trusted")
    digest = sha256(manifest_path)
    expected = expected_manifest_sha256 or manifest_path.with_suffix(".json.sha256").read_text(encoding="ascii").strip()
    if digest != expected:
        raise ModelInputError("manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("classification") != CLASSIFICATION or manifest.get("stream") != stream or manifest.get("phase") != phase:
        raise ModelInputError("manifest identity mismatch")
    if manifest.get("sklearn_version") != sklearn.__version__:
        raise ModelInputError("scikit-learn version mismatch")
    filename = manifest.get("model_filename")
    if filename != f"{stream}.joblib":
        raise ModelInputError("model filename is not allowed")
    path = folder / filename
    if path.is_symlink() or sha256(path) != manifest.get("model_sha256"):
        raise ModelInputError("model artifact hash mismatch")
    package = joblib.load(path)
    if package["base_columns"] != manifest["base_columns"] or package["expert_columns"] != manifest["expert_columns"]:
        raise ModelInputError("persisted feature schema mismatch")
    return {"package": package, "manifest": manifest, "manifest_sha256": digest}


def predict_scores(bundle, base, expert):
    package = bundle["package"]
    if list(base.columns) != package["base_columns"] or list(expert.columns) != package["expert_columns"]:
        raise ModelInputError("ordered feature schema mismatch")
    if not base.index.equals(expert.index) or not len(base):
        raise ModelInputError("feature time index mismatch")
    x, xd = base.to_numpy(np.float32), expert.to_numpy(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(xd).all():
        raise ModelInputError("missing model feature value")
    z0, zd = _score_family(package["base"], x), _score_family(package["expert"], xd)
    result = {"BASE": z0, "EXPERT": zd, "LOCAL_V4_FROZEN": z0 + package["weight"]*zd}
    if "append" in package:
        result["APPEND376_SAME_FOLD"] = _score_family(package["append"], np.hstack([x, xd]))
    return result


def predict_stream(stream, base_frame, v4_frame, decision_time, *, bundle=None, phase="deployment_q4"):
    decision = _timestamp(decision_time)
    if len(base_frame) != 1 or len(v4_frame) != 1:
        raise ModelInputError("one closed feature row required for online inference")
    bundle = bundle or load_model_bundle(stream, phase)
    manifest, package = bundle["manifest"], bundle["package"]
    if manifest["stream"] != stream:
        raise ModelInputError("wrong model stream")
    bar_open = _timestamp(base_frame.index[0])
    close = bar_open + pd.Timedelta(minutes=5)
    if not pd.Timedelta(0) <= decision - close <= pd.Timedelta(seconds=120):
        raise ModelInputError("feature row unclosed or too late")
    if close.minute % (STREAMS[stream][1]*5):
        raise ModelInputError("stream clock is not due")
    if not _timestamp(manifest["fold"]["valid_from"]) <= decision < _timestamp(manifest["fold"]["valid_until"]):
        raise ModelInputError("model outside declared deployment window")
    score = predict_scores(bundle, base_frame, v4_frame)
    zv = float(score["LOCAL_V4_FROZEN"][0])
    return {"stream": stream, "decision_time": decision.isoformat(), "feature_bar_open": bar_open.isoformat(),
            "z_base": float(score["BASE"][0]), "z_expert": float(score["EXPERT"][0]), "z_v4": zv,
            "threshold_frozen": package["threshold_frozen"], "threshold_reset": package["threshold_reset"],
            "w": package["weight"], "selected": abs(zv) >= package["threshold_frozen"],
            "side": "UP" if zv > 0 else "DOWN", "model_sha256": manifest["model_sha256"],
            "manifest_sha256": bundle["manifest_sha256"], "model_version": manifest["model_version"],
            "trained_through": manifest["fold"]["train_end"], "validation_through": manifest["fold"]["validation_end"],
            "classification": CLASSIFICATION, "label_proxy": "BINANCE_NEXT_OPEN_TO_WINDOW_CLOSE_NOT_CHAINLINK_TWAP"}


def _evaluate(bundle, stream, frames, test_rows, labels, report_dir):
    base, expert = frames["base"].iloc[test_rows], frames["expert"].iloc[test_rows]
    scores = predict_scores(bundle, base, expert)
    package = bundle["package"]
    returns = labels[test_rows]
    records, summaries = [], []
    thresholds = {"BASE": package["threshold_frozen"], "LOCAL_V4_FROZEN": package["threshold_frozen"],
                  "APPEND376_SAME_FOLD": package.get("threshold_append")}
    for variant, threshold in thresholds.items():
        if variant not in scores:
            continue
        z = scores[variant]
        selected = np.abs(z) >= threshold
        settled = selected & np.isfinite(returns)
        tied = settled & (returns == 0)
        wins = settled & ((z > 0) == (returns >= 0))
        n, w = int(settled.sum()), int(wins.sum())
        summaries.append({"stream": stream, "variant": variant, "period": "2026-01-01_to_2026-09-24",
            "signals": int(selected.sum()), "settled": n, "ties": int(tied.sum()), "unsettled": int((selected & ~np.isfinite(returns)).sum()),
            "wins": w, "raw_win_rate_pct": 100*w/n if n else None,
            "hypothetical_200_at_0_515_no_fee_pnl": 200*(w/.515-n),
            "fee_model": "NONE; research payout proxy, not Polymarket executions",
            "tie_rule": "BINANCE_PROXY_UP_IF_END_EQUALS_START; ties retained, never prefiltered",
            "model_sha256": bundle["manifest"]["model_sha256"], "manifest_sha256": bundle["manifest_sha256"],
            "threshold": threshold, "weight": package["weight"] if variant == "LOCAL_V4_FROZEN" else 0,
            "classification": CLASSIFICATION})
        records.append(pd.DataFrame({"stream": stream, "variant": variant, "bar_open_utc": base.index,
            "decision_close_utc": base.index + pd.Timedelta(minutes=5), "score": z, "threshold": threshold,
            "selected": selected, "side": np.where(z>0, "UP", "DOWN"), "forward_return": returns,
            "label_tie": returns == 0, "win": wins, "settled": settled,
            "model_sha256": bundle["manifest"]["model_sha256"]}))
    pd.concat(records, ignore_index=True).to_parquet(report_dir / f"{stream}_oos_predictions.parquet", index=False)
    return summaries


def train_all(*, threads=3, phases=("oos_2026", "deployment_q4"), selected_streams=None):
    if not 1 <= threads <= 4 or any(p not in FOLDS for p in phases):
        raise ModelInputError("invalid thread count or phase")
    streams = list(selected_streams or STREAMS)
    if any(s not in STREAMS for s in streams):
        raise ModelInputError("unknown training stream")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    # Verify registered historical observations before using any feature cache.
    from scripts.argon4_clean_room_reproduce import verify_data
    verify_data()
    run = {"status": "TRAINING", "started_at": _now(), "classification": CLASSIFICATION,
           "threads": threads, "phases": list(phases), "streams": streams, "completed": [],
           "source_sha256": SOURCE_HASHES, "training_code_sha256": sha256(__file__),
           "historical_data_manifest_sha256": sha256(DATA_ROOT.parent / "data_manifest.json"),
           "limitations": ["311 actual supplied base/rolling columns, not author's 374 causal-pivot columns",
              "Independent 65-feature expert, validation-only weights and frozen base threshold",
              "Same-fold locally retrained append376 comparator; no author table imported",
              "Historical windows were previously researched; not untouched holdout",
              "Binance candle direction proxy differs from Chainlink TWAP resolution",
              "Fixed 0.515/200 payout backtest excludes actual fees, fills, capital overlap and capacity",
              "Original derived indicator startup zeros retained; no raw OHLCV/funding events fabricated",
              "No strategy effectiveness or live readiness claim"]}
    _write_json(REPORT_ROOT / "run.json", run)
    progress = {"status": "TRAINING", "phase": "FEATURES", "current_stream": None,
                "completed_streams": [], "model_version": "local311_independent65_v1", "error": None}
    def update_progress(**changes):
        progress.update(changes, updated_at=_now())
        _write_json(REPORT_ROOT / "progress.json", progress)
    update_progress()
    summaries = []
    with threadpool_limits(limits=threads):
        for symbol in SYMBOLS:
            own_streams = [s for s in streams if STREAMS[s][0] == symbol]
            if not own_streams:
                continue
            started = time.monotonic()
            update_progress(phase="FEATURES", current_stream=symbol)
            print(f"FEATURES {symbol}", flush=True)
            frames = build_feature_frames(DATA_ROOT, symbol)
            x, xd = frames["base"].to_numpy(np.float32), frames["expert"].to_numpy(np.float32)
            _write_json(REPORT_ROOT / f"{symbol}_features.json", {"base_columns": list(frames["base"]),
                "expert_columns": list(frames["expert"]), "rows": len(x), "first_bar": str(frames["index"][0]),
                "last_bar": str(frames["index"][-1]), "provenance": frames["provenance"], "availability": frames["availability"]})
            for stream in own_streams:
                _, k, quantile = STREAMS[stream]
                labels = _label(frames["raw_target"], k)
                for phase in phases:
                    t0 = time.monotonic()
                    update_progress(phase=phase, current_stream=stream)
                    fold = FOLDS[phase]
                    tr, va, te = _split(frames["index"], labels, fold, k)
                    y = np.clip(labels[tr], -.008 if k == 1 else -.015, .008 if k == 1 else .015)
                    print(f"FIT {stream} {phase} train={len(tr)} val={len(va)}", flush=True)
                    base, z0 = _fit_family(x[tr], y, x[va])
                    expert, zd = _fit_family(xd[tr], y, xd[va], expert=True)
                    weight, threshold_reset, grid = select_weight(z0, zd, labels[va], quantile)
                    package = {"base": base, "expert": expert, "base_columns": list(frames["base"]),
                               "expert_columns": list(frames["expert"]), "weight": weight,
                               "threshold_frozen": float(np.quantile(np.abs(z0), quantile)),
                               "threshold_reset": threshold_reset}
                    if phase == "oos_2026":
                        print(f"FIT_APPEND {stream}", flush=True)
                        appended, za = _fit_family(np.hstack([x[tr], xd[tr]]), y, np.hstack([x[va], xd[va]]))
                        package.update(append=appended, threshold_append=float(np.quantile(np.abs(za), quantile)))
                    metadata = {"stream": stream, "phase": phase, "classification": CLASSIFICATION,
                        "model_version": f"local311_independent65_{phase}_v1", "fold": fold,
                        "base_columns": package["base_columns"], "expert_columns": package["expert_columns"],
                        "base_features": 311, "expert_features": 65, "train_stride": 2,
                        "train_rows": len(tr), "validation_rows": len(va), "test_rows": len(te),
                        "weight": weight, "weight_grid_validation": grid,
                        "threshold_frozen": package["threshold_frozen"], "threshold_reset": threshold_reset,
                        "train_last_label_end": str(frames["index"][tr[-1]] + pd.Timedelta(minutes=5*(k+1))),
                        "validation_last_label_end": str(frames["index"][va[-1]] + pd.Timedelta(minutes=5*(k+1))),
                        "training_data_last_bar": str(frames["index"][-1]), "source_sha256": SOURCE_HASHES,
                        "data_manifest_sha256": run["historical_data_manifest_sha256"],
                        "training_code_sha256": run["training_code_sha256"], "sklearn_version": sklearn.__version__,
                        "python_version": platform.python_version(), "normalization_fit": "validation only",
                        "numeric_policy": "C-contiguous float64 scaler/estimators/normalization; float32 stored features",
                        "training_ties": "excluded from train/validation only; never filters test or online selection",
                        "purpose": "EXPLORATORY_LOCAL_PAPER_ONLY"}
                    digest = _save_stream(MODEL_ROOT / phase, stream, package, metadata)
                    # The backtest deliberately reloads the bytes that forward inference loads.
                    reloaded = load_model_bundle(stream, phase, expected_manifest_sha256=digest)
                    check_base, check_expert = frames["base"].iloc[va[:64]], frames["expert"].iloc[va[:64]]
                    before_save = predict_scores({"package": package}, check_base, check_expert)
                    parity = predict_scores(reloaded, check_base, check_expert)
                    if any(not np.array_equal(parity[key], before_save[key]) for key in before_save):
                        raise ModelInputError("serialized-model score parity failed")
                    if not (np.allclose(parity["BASE"], z0[:64], rtol=1e-12, atol=1e-12)
                            and np.allclose(parity["EXPERT"], zd[:64], rtol=1e-12, atol=1e-12)):
                        raise ModelInputError("training-inference batch score parity failed")
                    if len(te):
                        summaries += _evaluate(reloaded, stream, frames, te, labels, REPORT_ROOT)
                        pd.DataFrame(summaries).to_csv(REPORT_ROOT / "oos_summary.csv", index=False, encoding="utf-8-sig")
                    item = {"stream": stream, "phase": phase, "elapsed_seconds": time.monotonic()-t0,
                            "manifest_sha256": digest, "model_sha256": reloaded["manifest"]["model_sha256"],
                            "weight": weight, "threshold": package["threshold_frozen"], "saved_model_parity": "PASS"}
                    run["completed"].append(item)
                    _write_json(REPORT_ROOT / "run.json", run)
                    update_progress(completed_streams=[{"stream": r["stream"], "phase": r["phase"]} for r in run["completed"]])
                    print("COMPLETE " + json.dumps(item), flush=True)
                    del package, base, expert, reloaded
            print(f"SYMBOL_DONE {symbol} elapsed={time.monotonic()-started:.1f}s", flush=True)
    run.update(status="COMPLETED", completed_at=_now())
    _write_json(REPORT_ROOT / "run.json", run)
    update_progress(status="COMPLETED", phase="DONE", current_stream=None)
    if summaries:
        ranks = sorted([r for r in summaries if r["variant"] == "LOCAL_V4_FROZEN"], key=lambda row: row["raw_win_rate_pct"] or 0, reverse=True)
        _write_json(REPORT_ROOT / "local_oos_ranking.json", {"classification": CLASSIFICATION,
            "ranking_basis": "local 2026 research OOS raw win rate; previously touched historical period",
            "streams": [{"rank": i+1, **row} for i, row in enumerate(ranks)]})
        pd.DataFrame([{"rank": i+1, **row} for i, row in enumerate(ranks)]).to_csv(REPORT_ROOT / "ranking.csv", index=False, encoding="utf-8-sig")
        _write_json(REPORT_ROOT / "summary.json", {"classification": CLASSIFICATION,
            "status": "COMPLETED", "rows": summaries, "ranking": [r["stream"] for r in ranks],
            "backtest_execution": "same locally saved bundle reloaded then predicted; no test refit",
            "returns_scope": "hypothetical 200 USD stake at 0.515, no fees or observed fills; not account returns"})
    return run
