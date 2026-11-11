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


def _resolve_5m_frame(data_root=None, symbol="BTCUSDT", *, as_of=None, snapshot=None, raw_frame=None, min_rows=12):
    frame = None
    if (raw_frame is not None and isinstance(raw_frame, pd.DataFrame)
            and len(raw_frame) >= min_rows and {"open", "close"} <= set(raw_frame.columns)):
        frame = raw_frame
    if frame is None and snapshot is not None:
        target_names = {f"{symbol}_5m_futures.parquet", f"{symbol}_5m_futures_2023_2026.parquet"}
        if hasattr(snapshot, "_validated"):
            for p, entry in snapshot._validated.items():
                if getattr(p, "name", "") in target_names and isinstance(entry, tuple) and len(entry) >= 1:
                    candidate = entry[0]
                    if isinstance(candidate, pd.DataFrame) and len(candidate) >= min_rows:
                        frame = candidate
                        break
        if frame is None and hasattr(snapshot, "_raw") and hasattr(snapshot, "_read_validated"):
            for p in list(getattr(snapshot, "_raw", {}).keys()):
                if getattr(p, "name", "") in target_names:
                    try:
                        candidate, _, _ = snapshot._read_validated(p)
                        if isinstance(candidate, pd.DataFrame) and len(candidate) >= min_rows:
                            frame = candidate
                            break
                    except Exception:
                        pass
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
    return frame


def compute_ex_ante_rv12_bps(data_root=None, symbol="BTCUSDT", *, as_of=None, snapshot=None, raw_frame=None):
    """Compute causal 1h realized body volatility (rv_12_bps) over last 12 closed 5m candles."""
    frame = _resolve_5m_frame(data_root, symbol, as_of=as_of, snapshot=snapshot, raw_frame=raw_frame, min_rows=12)
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


def compute_ex_ante_btc_macro_returns_pct(data_root=None, symbol="BTCUSDT", *, as_of=None, snapshot=None, raw_frame=None):
    """Compute causal 1h (12 bars), 2h (24 bars), 4h (48 bars) BTC returns and EMA12-EMA36 gap (bps)."""
    out = {
        "btc_ret_1h_pct": None,
        "btc_ret_2h_pct": None,
        "btc_ret_4h_pct": None,
        "btc_ema_gap_bps": None,
    }
    frame = _resolve_5m_frame(data_root, symbol, as_of=as_of, snapshot=snapshot, raw_frame=raw_frame, min_rows=12)
    if frame is None or len(frame) < 12 or not ({"open", "close"} <= set(frame.columns)):
        return out

    def _window_ret_pct(n_bars):
        if len(frame) < n_bars:
            return None
        tail = frame.tail(n_bars)
        if isinstance(tail.index, pd.DatetimeIndex) and len(tail.index) == n_bars:
            if not np.all(np.diff(tail.index.asi8) == pd.Timedelta(minutes=5).value):
                return None
        o_val = float(tail["open"].iloc[0])
        c_val = float(tail["close"].iloc[-1])
        if np.isfinite(o_val) and np.isfinite(c_val) and o_val > 0:
            return float((c_val / o_val - 1.0) * 100.0)
        return None

    out["btc_ret_1h_pct"] = _window_ret_pct(12)
    out["btc_ret_2h_pct"] = _window_ret_pct(24)
    out["btc_ret_4h_pct"] = _window_ret_pct(48)
    if len(frame) >= 36:
        tail_ema = frame["close"].tail(144).astype(float)
        if np.isfinite(tail_ema.to_numpy()).all() and (tail_ema > 0).all():
            ema12 = float(tail_ema.ewm(span=12, adjust=False).mean().iloc[-1])
            ema36 = float(tail_ema.ewm(span=36, adjust=False).mean().iloc[-1])
            if ema36 > 0:
                out["btc_ema_gap_bps"] = float((ema12 / ema36 - 1.0) * 10000.0)
    return out


def compute_ex_ante_symbol_returns(data_root=None, symbol="BTCUSDT", *, as_of=None, snapshot=None, raw_frame=None):
    """Compute causal 15m (3-bar bps), 1h (12-bar pct), and 3-bar directional streak for any symbol."""
    out = {
        "own_ret_15m_bps": None,
        "own_ret_1h_pct": None,
        "own_streak_3bar": None,
    }
    frame = _resolve_5m_frame(data_root, symbol, as_of=as_of, snapshot=snapshot, raw_frame=raw_frame, min_rows=3)
    if frame is None or len(frame) < 3 or not ({"open", "close"} <= set(frame.columns)):
        return out

    tail3 = frame.tail(3)
    if not (isinstance(tail3.index, pd.DatetimeIndex) and len(tail3.index) == 3
            and not np.all(np.diff(tail3.index.asi8) == pd.Timedelta(minutes=5).value)):
        o3 = tail3["open"].to_numpy(dtype=float)
        c3 = tail3["close"].to_numpy(dtype=float)
        if len(o3) == 3 and np.isfinite(o3).all() and np.isfinite(c3).all() and (o3 > 0).all():
            out["own_ret_15m_bps"] = float((c3[-1] / o3[0] - 1.0) * 10000.0)
            diffs = c3 - o3
            if np.all(diffs > 0):
                out["own_streak_3bar"] = 3
            elif np.all(diffs < 0):
                out["own_streak_3bar"] = -3
            else:
                out["own_streak_3bar"] = 0

    if len(frame) >= 12:
        tail12 = frame.tail(12)
        if not (isinstance(tail12.index, pd.DatetimeIndex) and len(tail12.index) == 12
                and not np.all(np.diff(tail12.index.asi8) == pd.Timedelta(minutes=5).value)):
            o12 = float(tail12["open"].iloc[0])
            c12 = float(tail12["close"].iloc[-1])
            if np.isfinite(o12) and np.isfinite(c12) and o12 > 0:
                out["own_ret_1h_pct"] = float((c12 / o12 - 1.0) * 100.0)
    return out


def _resolve_cohort_utc(cohort_time=None, prediction=None):
    if cohort_time is not None:
        ts = pd.Timestamp(cohort_time)
        return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    if isinstance(prediction, dict):
        if prediction.get("feature_bar_open"):
            ts = pd.Timestamp(prediction["feature_bar_open"])
            ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
            return ts + pd.Timedelta(minutes=5)
        if prediction.get("decision_time"):
            ts = pd.Timestamp(prediction["decision_time"])
            ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
            return ts.floor("5min")
    return None


def apply_volatility_regime_guard(
    stream,
    prediction,
    *,
    ex_ante_rv12_bps=None,
    guard_config=None,
    cohort_time=None,
    btc_ret_1h_pct=None,
    btc_ret_2h_pct=None,
    btc_ret_4h_pct=None,
    btc_ema_gap_bps=None,
    own_ret_1h_pct=None,
    own_ret_15m_bps=None,
    own_streak_3bar=None,
):
    """Attach ex_ante_rv12_bps / macro returns and apply volatility, US-open blackout & macro trend guards."""
    res = dict(prediction)
    if ex_ante_rv12_bps is not None:
        res["ex_ante_rv12_bps"] = float(ex_ante_rv12_bps)
    if btc_ret_1h_pct is not None:
        res["ex_ante_btc_ret_1h_pct"] = float(btc_ret_1h_pct)
    if btc_ret_2h_pct is not None:
        res["ex_ante_btc_ret_2h_pct"] = float(btc_ret_2h_pct)
    if btc_ret_4h_pct is not None:
        res["ex_ante_btc_ret_4h_pct"] = float(btc_ret_4h_pct)
    if btc_ema_gap_bps is not None:
        res["ex_ante_btc_ema_gap_bps"] = float(btc_ema_gap_bps)
    if own_ret_1h_pct is not None:
        res["ex_ante_own_ret_1h_pct"] = float(own_ret_1h_pct)
    if own_ret_15m_bps is not None:
        res["ex_ante_own_ret_15m_bps"] = float(own_ret_15m_bps)
    if own_streak_3bar is not None:
        res["ex_ante_own_streak_3bar"] = int(own_streak_3bar)
    if not (isinstance(guard_config, dict) and guard_config.get("enabled") is True):
        return res

    action = str(guard_config.get("action", "SKIP_OR_HALF_STAKE")).upper()
    if ex_ante_rv12_bps is not None:
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
            return res
        if min_cap is not None and float(ex_ante_rv12_bps) < float(min_cap):
            res["volatility_regime_guarded"] = True
            res["volatility_regime_min_rv12_bps"] = float(min_cap)
            res["volatility_regime_reason"] = "BELOW_MIN_RV12_BPS"
            res["volatility_regime_action"] = action
            if action in ("SKIP", "SKIP_OR_HALF_STAKE"):
                res["selected"] = False
            return res

    us_blackout = guard_config.get("us_open_blackout")
    if isinstance(us_blackout, dict) and us_blackout.get("enabled") is True:
        cohort_dt = _resolve_cohort_utc(cohort_time=cohort_time, prediction=res)
        if cohort_dt is not None:
            weekdays_only = bool(us_blackout.get("weekdays_only", True))
            day_ok = (cohort_dt.weekday() < 5) if weekdays_only else True
            minute_of_day = int(cohort_dt.hour) * 60 + int(cohort_dt.minute)
            start_min = int(us_blackout.get("utc_start_minute", 870))
            end_min = int(us_blackout.get("utc_end_minute", 960))
            skip_streams = us_blackout.get("skip_streams") or []
            if day_ok and start_min <= minute_of_day < end_min and stream in skip_streams:
                res["volatility_regime_guarded"] = True
                res["volatility_regime_reason"] = "US_OPEN_BLACKOUT"
                res["volatility_regime_action"] = action
                res["us_open_blackout_minute_utc"] = minute_of_day
                if action in ("SKIP", "SKIP_OR_HALF_STAKE"):
                    res["selected"] = False
                return res

    macro_guard = guard_config.get("macro_trend_guard")
    if isinstance(macro_guard, dict) and macro_guard.get("enabled") is True:
        max_4h = macro_guard.get("max_btc_4h_abs_return_pct")
        apply_streams = macro_guard.get("apply_streams")
        in_apply = (stream in apply_streams) if isinstance(apply_streams, (list, tuple, set)) else True
        if (max_4h is not None and btc_ret_4h_pct is not None and in_apply
                and abs(float(btc_ret_4h_pct)) > float(max_4h)):
            res["volatility_regime_guarded"] = True
            res["volatility_regime_reason"] = "BTC_4H_MACRO_BREAKOUT"
            res["volatility_regime_action"] = action
            res["macro_trend_max_btc_4h_abs_return_pct"] = float(max_4h)
            if action in ("SKIP", "SKIP_OR_HALF_STAKE"):
                res["selected"] = False
            return res

        chase_th = macro_guard.get("skip_1h_momentum_chase_above_pct")
        side = str(res.get("side", "")).upper()
        if chase_th is not None and btc_ret_1h_pct is not None and side in ("UP", "DOWN"):
            ret_1h = float(btc_ret_1h_pct)
            th_1h = float(chase_th)
            if (ret_1h > th_1h and side == "UP") or (ret_1h < -th_1h and side == "DOWN"):
                res["volatility_regime_guarded"] = True
                res["volatility_regime_reason"] = "BTC_1H_MOMENTUM_CHASE"
                res["volatility_regime_action"] = action
                res["macro_trend_skip_1h_momentum_chase_above_pct"] = th_1h
                if action in ("SKIP", "SKIP_OR_HALF_STAKE"):
                    res["selected"] = False
                return res

        bleed_cfg = macro_guard.get("single_order_per_window_on_2h_bleed")
        if isinstance(bleed_cfg, dict) and bleed_cfg.get("enabled") is True:
            min_2h = float(bleed_cfg.get("min_btc_2h_abs_return_pct", 0.50))
            min_gap = float(bleed_cfg.get("min_btc_ema12_36_gap_bps", 15.0))
            if btc_ret_2h_pct is not None and btc_ema_gap_bps is not None:
                r2h = float(btc_ret_2h_pct)
                egap = float(btc_ema_gap_bps)
                is_down_bleed = (r2h <= -min_2h and egap <= -min_gap and side == "UP")
                is_up_squeeze = (r2h >= min_2h and egap >= min_gap and side == "DOWN")
                if is_down_bleed or is_up_squeeze:
                    res["bleed_single_order_window_cap"] = True

    own_guard = guard_config.get("own_asset_crash_guard")
    if isinstance(own_guard, dict) and own_guard.get("enabled") is True:
        side = str(res.get("side", "")).upper()
        max_1h_map = own_guard.get("max_own_1h_counter_return_pct")
        max_1h = (
            max_1h_map.get(stream)
            if isinstance(max_1h_map, dict)
            else (float(max_1h_map) if isinstance(max_1h_map, (int, float)) else None)
        )
        if max_1h is not None and own_ret_1h_pct is not None and side in ("UP", "DOWN"):
            r1h = float(own_ret_1h_pct)
            th1h = float(max_1h)
            if (side == "UP" and r1h <= -th1h) or (side == "DOWN" and r1h >= th1h):
                res["volatility_regime_guarded"] = True
                res["volatility_regime_reason"] = "OWN_ASSET_1H_WATERFALL_GUARD"
                res["volatility_regime_action"] = action
                res["own_asset_max_1h_counter_return_pct"] = th1h
                if action in ("SKIP", "SKIP_OR_HALF_STAKE"):
                    res["selected"] = False
                return res

        max_15m_map = own_guard.get("max_own_15m_counter_return_bps")
        max_15m = (
            max_15m_map.get(stream)
            if isinstance(max_15m_map, dict)
            else (float(max_15m_map) if isinstance(max_15m_map, (int, float)) else None)
        )
        if max_15m is not None and own_ret_15m_bps is not None and side in ("UP", "DOWN"):
            r15m = float(own_ret_15m_bps)
            th15m = float(max_15m)
            if (side == "UP" and r15m <= -th15m) or (side == "DOWN" and r15m >= th15m):
                res["volatility_regime_guarded"] = True
                res["volatility_regime_reason"] = "OWN_ASSET_15M_WATERFALL_GUARD"
                res["volatility_regime_action"] = action
                res["own_asset_max_15m_counter_return_bps"] = th15m
                if action in ("SKIP", "SKIP_OR_HALF_STAKE"):
                    res["selected"] = False
                return res

    tiered_cfg = guard_config.get("confidence_tiered_max_buy_price")
    if isinstance(tiered_cfg, dict) and tiered_cfg.get("enabled") is True and res.get("selected") is True:
        side = str(res.get("side", "")).upper()
        z_abs = abs(float(res.get("z_v4", 0.0)))
        th_val = float(res.get("threshold_frozen", 0.0))
        delta_z = max(0.0, z_abs - th_val)
        res["signal_excess_z"] = round(delta_z, 6)

        streak_min_bps = float(tiered_cfg.get("counter_15m_streak_min_bps", 45.0))
        streak_marginal_dz = float(tiered_cfg.get("counter_15m_marginal_delta_z", 0.12))
        streak_action = str(tiered_cfg.get("counter_15m_marginal_action", "CAP_OR_SKIP")).upper()
        streak_max_buy = float(tiered_cfg.get("counter_15m_marginal_max_buy_price", 0.475))
        strong_min_dz = float(tiered_cfg.get("strong_signal_min_delta_z", 0.15))
        strong_cap = float(tiered_cfg.get("strong_max_buy_price", 0.515))
        marginal_cap = float(tiered_cfg.get("marginal_max_buy_price", 0.49))

        is_counter_15m_streak = False
        if own_ret_15m_bps is not None and side in ("UP", "DOWN"):
            r15 = float(own_ret_15m_bps)
            s3 = int(own_streak_3bar or 0)
            if (side == "DOWN" and r15 >= streak_min_bps and s3 >= 3) or (
                side == "UP" and r15 <= -streak_min_bps and s3 <= -3
            ):
                is_counter_15m_streak = True

        if is_counter_15m_streak and delta_z < streak_marginal_dz:
            if streak_action == "SKIP":
                res["volatility_regime_guarded"] = True
                res["volatility_regime_reason"] = "MARGINAL_SIGNAL_AGAINST_15M_STREAK"
                res["volatility_regime_action"] = action
                res["selected"] = False
                return res
            res["effective_max_buy_price"] = streak_max_buy
            res["price_tier_reason"] = "MARGINAL_AGAINST_15M_STREAK"
        elif delta_z < strong_min_dz:
            res["effective_max_buy_price"] = marginal_cap
            res["price_tier_reason"] = "MARGINAL_SIGNAL_CAP"
        else:
            res["effective_max_buy_price"] = strong_cap
            res["price_tier_reason"] = "STRONG_SIGNAL_CAP"

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
    cohort_time=None,
    btc_ret_1h_pct=None,
    btc_ret_2h_pct=None,
    btc_ret_4h_pct=None,
    btc_ema_gap_bps=None,
    own_ret_1h_pct=None,
    own_ret_15m_bps=None,
    own_streak_3bar=None,
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
    if any(
        x is not None
        for x in (
            ex_ante_rv12_bps,
            volatility_regime_guard,
            cohort_time,
            btc_ret_1h_pct,
            btc_ret_2h_pct,
            btc_ret_4h_pct,
            btc_ema_gap_bps,
            own_ret_1h_pct,
            own_ret_15m_bps,
            own_streak_3bar,
        )
    ):
        res = apply_volatility_regime_guard(
            stream,
            res,
            ex_ante_rv12_bps=ex_ante_rv12_bps,
            guard_config=volatility_regime_guard,
            cohort_time=cohort_time,
            btc_ret_1h_pct=btc_ret_1h_pct,
            btc_ret_2h_pct=btc_ret_2h_pct,
            btc_ret_4h_pct=btc_ret_4h_pct,
            btc_ema_gap_bps=btc_ema_gap_bps,
            own_ret_1h_pct=own_ret_1h_pct,
            own_ret_15m_bps=own_ret_15m_bps,
            own_streak_3bar=own_streak_3bar,
        )
    return res
