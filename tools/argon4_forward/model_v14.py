"""v14 non-mutating adapter over frozen tools.argon4_forward.model.

Preserves exact byte-level SHA256 of tools/argon4_forward/model.py while
enabling the 8th stream (XRP_15m / XRPUSDT) and Hybrid Golden Quantile
threshold overrides.
"""
from __future__ import annotations

import ast
import inspect

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


def predict_stream_v14(stream, base_frame, v4_frame, decision_time, *, bundle=None, phase="deployment_q4", threshold_override=None):
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
    return res
