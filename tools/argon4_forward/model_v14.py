"""v14 non-mutating adapter over frozen tools.argon4_forward.model.

Preserves exact byte-level SHA256 of tools/argon4_forward/model.py while
enabling the 8th stream (XRP_15m / XRPUSDT) and Hybrid Golden Quantile
threshold overrides.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from . import model

ALL_STREAMS = {**model.STREAMS, "XRP_15m": ("XRPUSDT", 3, .85)}
ALL_SYMBOLS = (*model.SYMBOLS, "XRPUSDT")
_ORIGINAL_BUILD_FEATURE_FRAMES = model.build_feature_frames
_ORIGINAL_LOAD_MODEL_BUNDLE = model.load_model_bundle
_ORIGINAL_PREDICT_STREAM = model.predict_stream


def build_feature_frames_v14(data_root=None, symbol="BTCUSDT", *, as_of=None, tail_rows=None):
    root = model.DATA_ROOT if data_root is None else data_root
    if symbol in model.SYMBOLS or model.build_feature_frames is not _ORIGINAL_BUILD_FEATURE_FRAMES:
        return model.build_feature_frames(root, symbol, as_of=as_of, tail_rows=tail_rows)
    if symbol not in ALL_SYMBOLS:
        raise model.ModelInputError("unsupported local feature source or symbol")
    tree = ast.parse(inspect.getsource(_ORIGINAL_BUILD_FEATURE_FRAMES))
    replacement = ast.parse(
        "ofi = sum((2 * frames[c].taker_buy_quote_volume - frames[c].quote_volume) / (frames[c].quote_volume + 1e-12) for c in CORE_4) / 4"
    ).body[0]
    for i, node in enumerate(tree.body[0].body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ofi" for t in node.targets):
            tree.body[0].body[i] = replacement

    def sources(reader, r):
        base, expert = model._source_modules(reader, r)
        base.CORE_SYMBOLS = (*model.SYMBOLS, symbol)
        return base, expert

    namespace = dict(vars(model))
    namespace.update(SYMBOLS=(*model.SYMBOLS, symbol), CORE_4=model.SYMBOLS, _source_modules=sources)
    ast.fix_missing_locations(tree)
    exec(compile(tree, "argon4_v14_feature_adapter", "exec"), namespace)
    return namespace["build_feature_frames"](root, symbol, as_of=as_of, tail_rows=tail_rows)


def load_model_bundle_v14(stream, phase="deployment_q4", *, expected_manifest_sha256=None):
    if stream in model.STREAMS or model.load_model_bundle is not _ORIGINAL_LOAD_MODEL_BUNDLE:
        if phase == "deployment_q4":
            return model.load_model_bundle(stream, expected_manifest_sha256=expected_manifest_sha256)
        return model.load_model_bundle(stream, phase=phase, expected_manifest_sha256=expected_manifest_sha256)
    if stream not in ALL_STREAMS:
        raise model.ModelInputError("unknown stream or phase")
    tree = ast.parse(inspect.getsource(_ORIGINAL_LOAD_MODEL_BUNDLE))
    namespace = dict(vars(model))
    namespace.update(STREAMS=ALL_STREAMS, MODEL_ROOT=model.MODEL_ROOT)
    ast.fix_missing_locations(tree)
    exec(compile(tree, "argon4_v14_load_bundle_adapter", "exec"), namespace)
    return namespace["load_model_bundle"](stream, phase=phase, expected_manifest_sha256=expected_manifest_sha256)


def compute_ex_ante_rv12_bps(data_root=None, symbol="BTCUSDT", *, as_of=None, snapshot=None, raw_frame=None):
    """Compute causal 1h realized body volatility (rv_12_bps) over last 12 closed 5m candles."""
    frame = None
    if (raw_frame is not None and isinstance(raw_frame, pd.DataFrame)
            and len(raw_frame) >= 12 and {"open", "close"} <= set(raw_frame.columns)):
        frame = raw_frame
    if frame is None and snapshot is not None and hasattr(snapshot, "_validated"):
        target_names = {f"{symbol}_5m_futures.parquet", f"{symbol}_5m_futures_2023_2026.parquet"}
        for p, entry in snapshot._validated.items():
            if getattr(p, "name", "") in target_names and isinstance(entry, tuple) and len(entry) >= 1:
                candidate = entry[0]
                if isinstance(candidate, pd.DataFrame) and len(candidate) >= 12:
                    frame = candidate
                    break
    if frame is None and data_root is not None:
        try:
            intraday_dir = Path(str(data_root)) / "intraday"
            old_path = intraday_dir / f"{symbol}_5m_futures_2023_2026.parquet"
            path = old_path if old_path.is_file() else intraday_dir / f"{symbol}_5m_futures.parquet"
            if path.is_file():
                df = pd.read_parquet(path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    col = "open_time" if "open_time" in df.columns else "date"
                    df = df.set_index(pd.to_datetime(df[col], utc=True))
                df.index = pd.to_datetime(df.index, utc=True)
                if as_of is not None:
                    cutoff = pd.Timestamp(as_of).tz_convert("UTC") if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of, tz="UTC")
                    df = df.loc[df.index + pd.Timedelta(minutes=5) <= cutoff]
                frame = df
        except Exception:
            frame = None
    if frame is None or len(frame) < 12 or not ({"open", "close"} <= set(frame.columns)):
        return None
    tail = frame.tail(12)
    if isinstance(tail.index, pd.DatetimeIndex) and len(tail.index) == 12:
        if not np.all(np.diff(tail.index.asi8) == pd.Timedelta(minutes=5).value):
            return None
    opens = tail["open"].to_numpy(dtype=float)
    closes = tail["close"].to_numpy(dtype=float)
    if len(opens) != 12 or not (np.isfinite(opens).all() and np.isfinite(closes).all()) or (opens <= 0).any():
        return None
    return float(np.mean(np.abs(closes - opens) / opens) * 10000.0)


def apply_volatility_regime_guard(stream, prediction, *, ex_ante_rv12_bps=None, guard_config=None):
    """Attach ex_ante_rv12_bps and optionally apply V5_Extreme / UltraLow_TWAP volatility regime guard."""
    res = dict(prediction)
    if ex_ante_rv12_bps is not None:
        res["ex_ante_rv12_bps"] = float(ex_ante_rv12_bps)
    if isinstance(guard_config, dict) and guard_config.get("enabled") is True and ex_ante_rv12_bps is not None:
        action = str(guard_config.get("action", "SKIP_OR_HALF_STAKE")).upper()
        limits = guard_config.get("max_rv12_bps")
        cap = limits.get(stream) if isinstance(limits, dict) else None
        min_limits = guard_config.get("min_rv12_bps")
        min_cap = min_limits.get(stream) if isinstance(min_limits, dict) else None
        if cap is not None and float(ex_ante_rv12_bps) > float(cap):
            res["volatility_regime_guarded"] = True
            res["volatility_regime_max_rv12_bps"] = float(cap)
            res["volatility_regime_reason"] = "ABOVE_MAX_RV12_BPS"
            res["volatility_regime_action"] = action
            if action in ("SKIP", "SKIP_OR_HALF_STAKE"):
                res["selected"] = False
        elif min_cap is not None and float(ex_ante_rv12_bps) < float(min_cap):
            res["volatility_regime_guarded"] = True
            res["volatility_regime_min_rv12_bps"] = float(min_cap)
            res["volatility_regime_reason"] = "BELOW_MIN_RV12_BPS"
            res["volatility_regime_action"] = action
            if action in ("SKIP", "SKIP_OR_HALF_STAKE"):
                res["selected"] = False
    return res


def predict_stream_v14(
    stream,
    base_frame,
    v4_frame,
    decision_time,
    *,
    bundle=None,
    phase="deployment_q4",
    threshold_override=None,
    ex_ante_rv12_bps=None,
    volatility_regime_guard=None,
):
    if stream in model.STREAMS or model.predict_stream is not _ORIGINAL_PREDICT_STREAM:
        if phase == "deployment_q4":
            res = model.predict_stream(stream, base_frame, v4_frame, decision_time, bundle=bundle)
        else:
            res = model.predict_stream(stream, base_frame, v4_frame, decision_time, bundle=bundle, phase=phase)
    elif stream in ALL_STREAMS:
        tree = ast.parse(inspect.getsource(_ORIGINAL_PREDICT_STREAM))
        namespace = dict(vars(model))
        namespace.update(STREAMS=ALL_STREAMS, load_model_bundle=load_model_bundle_v14)
        ast.fix_missing_locations(tree)
        exec(compile(tree, "argon4_v14_predict_adapter", "exec"), namespace)
        res = namespace["predict_stream"](stream, base_frame, v4_frame, decision_time, bundle=bundle, phase=phase)
    else:
        raise model.ModelInputError("unknown stream or phase")
    if threshold_override is not None:
        th = float(threshold_override)
        zv = float(res["z_v4"])
        res = {**res, "threshold_frozen": th, "selected": abs(zv) >= th}
    if ex_ante_rv12_bps is not None or volatility_regime_guard is not None:
        res = apply_volatility_regime_guard(
            stream,
            res,
            ex_ante_rv12_bps=ex_ante_rv12_bps,
            guard_config=volatility_regime_guard,
        )
    return res
