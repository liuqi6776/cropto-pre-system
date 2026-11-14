"""Autonomous rolling Chainlink TWAP60 + 270d decay retrainer, gatekeeper & hot-reload promoter.

Workflow:
1. Incrementally merges active runtime's validated `5m/1h/4h/mark/funding/basis/dvol` parquets
   into the full-history training `raw_data/` directory (and bridges any intermediate gaps via
   public Binance/Deribit endpoints) + updates `spot_1m/` OHLC4 TWAP60 parquets.
2. Computes dynamic rolling fold dates as of run time (`validation = [T-60d, T]`,
   `train = [2024-01-05, T-60d)`, `valid_until = T + 95d` — permanently preventing expiration cliffs).
3. Trains all 7 streams inside an isolated staging directory (never mutating active `.joblib` files
   in place and preserving frozen `tools/argon4_forward/model.py` byte-for-byte).
4. Enforces a strict Validation Gatekeeper (`saved_model_parity == "PASS"`, minimum mean & per-stream
   TWAP60 validation accuracy, and full `load_model_bundle_v14` reload verification).
5. Waits for a safe inter-window gap (`[T+25s, T+292s]`), archives old bundles, atomically replaces
   `deployment_q4/` files, and atomically updates `config/argon4_forward_policy.json`
   `deployment_manifest_sha256` so `argon4-forward` hot-reloads with zero restart or downtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_argon4_twap60_decay import fetch_spot_1m_incremental
from tools.argon4_forward import model, model_v14, twap60_retrain


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        # Write exact UTF-8 LF bytes across Python 3.9+ on both Windows and Linux
        tmp.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        df.to_parquet(tmp, index=index)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build_rolling_fold(
    as_of_utc: pd.Timestamp | str | datetime | None = None,
    *,
    validation_days: int = 60,
    valid_horizon_days: int = 95,
    half_life_days: float = 270.0,
    label_mode: str = "SPOT_TWAP60_BLEND",
) -> dict[str, Any]:
    """Compute rolling train/validation/deployment timestamps anchored to ``as_of_utc``."""
    now_ts = pd.Timestamp.now(tz="UTC") if as_of_utc is None else pd.Timestamp(as_of_utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")

    val_end = (now_ts.floor("5min") - pd.Timedelta(hours=2)).floor("5min")
    val_start = (val_end - pd.Timedelta(days=int(validation_days))).floor("1D")
    train_end = val_start - pd.Timedelta(minutes=30)
    valid_from = (now_ts - pd.Timedelta(days=1)).floor("1h")
    valid_until = (now_ts + pd.Timedelta(days=int(valid_horizon_days))).floor("1D")
    return {
        "train_start": "2024-01-05T00:00:00Z",
        "train_end": train_end.isoformat().replace("+00:00", "Z"),
        "validation_start": val_start.isoformat().replace("+00:00", "Z"),
        "validation_end": val_end.isoformat().replace("+00:00", "Z"),
        "test_start": None,
        "test_end": None,
        "valid_from": valid_from.isoformat().replace("+00:00", "Z"),
        "valid_until": valid_until.isoformat().replace("+00:00", "Z"),
        "half_life_days": float(half_life_days),
        "label_mode": str(label_mode),
    }


def _fetch_binance_gap_klines(base_url: str, endpoint: str, symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    step_ms = {"5m": 300_000, "1h": 3_600_000, "4h": 14_400_000}[interval]
    cur = start_ms
    out = []
    with requests.Session() as sess:
        sess.trust_env = False
        while cur < end_ms:
            batch = []
            for att in range(4):
                try:
                    resp = sess.get(
                        f"{base_url}/{endpoint}",
                        params={"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1000},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    batch = resp.json()
                    break
                except Exception:
                    if att == 3:
                        raise
                    time.sleep(1 + att)
            if not batch:
                break
            out.extend(batch)
            cur = int(batch[-1][0]) + step_ms
            if len(batch) < 1000:
                break
    return out


def sync_runtime_raw_to_training_root(
    runtime_raw_dir: Path,
    training_raw_dir: Path,
    *,
    fetch_network_gaps: bool = True,
) -> dict[str, int]:
    """Merge latest validated hot-runtime parquets (and any intermediate public API gaps) into training_raw_dir."""
    runtime_raw_dir = Path(runtime_raw_dir).resolve()
    training_raw_dir = Path(training_raw_dir).resolve()
    (training_raw_dir / "intraday").mkdir(parents=True, exist_ok=True)
    (training_raw_dir / "auxiliary").mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    now_ms = int(pd.Timestamp.now(tz="UTC").floor("5min").timestamp() * 1000) - 1

    # 1. Merge 5m futures, 1h futures/mark, and 8h funding for all 4 core symbols
    pending_5m: dict[str, tuple[pd.DataFrame, Path]] = {}
    for sym in model.SYMBOLS:
        # 5m intraday
        src_5m = runtime_raw_dir / "intraday" / f"{sym}_5m_futures.parquet"
        dst_5m = training_raw_dir / "intraday" / f"{sym}_5m_futures.parquet"
        old_5m = dst_5m if dst_5m.is_file() else (training_raw_dir / "intraday" / f"{sym}_5m_futures_2023_2026.parquet")
        parts = []
        if old_5m.is_file():
            d0 = pd.read_parquet(old_5m)
            if "open_time" not in d0.columns:
                d0 = d0.reset_index().rename(columns={d0.index.name or "index": "open_time"})
            d0["open_time"] = pd.to_datetime(d0["open_time"], utc=True)
            d0["date"] = d0["open_time"]
            parts.append(d0)
        if src_5m.is_file():
            d1 = pd.read_parquet(src_5m)
            if "open_time" not in d1.columns and "date" in d1.columns:
                d1["open_time"] = pd.to_datetime(d1["date"], utc=True)
            else:
                d1["open_time"] = pd.to_datetime(d1["open_time"], utc=True)
            d1["date"] = d1["open_time"]
            if fetch_network_gaps and parts:
                max_old = parts[0]["open_time"].max()
                min_new = d1["open_time"].min()
                if min_new > max_old + pd.Timedelta(minutes=5):
                    gap_rows = _fetch_binance_gap_klines(
                        "https://fapi.binance.com/fapi/v1", "klines", sym, "5m",
                        int((max_old + pd.Timedelta(minutes=5)).timestamp() * 1000),
                        int(min_new.timestamp() * 1000) - 1,
                    )
                    if gap_rows:
                        gd = pd.DataFrame(gap_rows, columns=["ms", "open", "high", "low", "close", "volume", "cms", "quote_volume", "count", "taker_buy_base_volume", "taker_buy_quote_volume", "ig"])
                        for col in ("open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_base_volume", "taker_buy_quote_volume"):
                            gd[col] = gd[col].astype(float)
                        gd["open_time"] = pd.to_datetime(gd["ms"].astype(np.int64), unit="ms", utc=True)
                        gd["date"] = gd["open_time"]
                        parts.append(gd)
            parts.append(d1)
        if parts:
            keep_cols = ["date", "open_time", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_base_volume", "taker_buy_quote_volume"]
            avail_cols = [c for c in keep_cols if all(c in p.columns for p in parts)]
            m5 = pd.concat([p[avail_cols] for p in parts], ignore_index=True).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
            pending_5m[sym] = (m5, old_5m if old_5m.is_file() else dst_5m)

    if pending_5m:
        common_max_ts = min(df["open_time"].max() for df, _ in pending_5m.values())
        for sym, (m5, target_5m) in pending_5m.items():
            aligned_5m = m5.loc[m5["open_time"] <= common_max_ts].reset_index(drop=True)
            _atomic_write_parquet(aligned_5m, target_5m, index=False)
            row_counts[f"{sym}:5m"] = len(aligned_5m)

    for sym in model.SYMBOLS:

        # 1h futures & mark
        for kind, ep in (("futures", "klines"), ("mark", "markPriceKlines")):
            src_1h = runtime_raw_dir / "auxiliary" / f"{sym}_1h_{kind}.parquet"
            dst_1h = training_raw_dir / "auxiliary" / f"{sym}_1h_{kind}.parquet"
            h_parts = []
            if dst_1h.is_file():
                h0 = pd.read_parquet(dst_1h)
                h0["date"] = pd.to_datetime(h0["date"], utc=True)
                h_parts.append(h0)
            if src_1h.is_file():
                h1 = pd.read_parquet(src_1h)
                h1["date"] = pd.to_datetime(h1["date"], utc=True)
                if fetch_network_gaps and h_parts:
                    max_h0 = h_parts[0]["date"].max()
                    min_h1 = h1["date"].min()
                    if min_h1 > max_h0 + pd.Timedelta(hours=1):
                        g_h = _fetch_binance_gap_klines(
                            "https://fapi.binance.com/fapi/v1", ep, sym, "1h",
                            int((max_h0 + pd.Timedelta(hours=1)).timestamp() * 1000),
                            int(min_h1.timestamp() * 1000) - 1,
                        )
                        if g_h:
                            gh_df = pd.DataFrame(g_h, columns=["ms", "open", "high", "low", "close", "volume", "cms", "qv", "n", "tbv", "tbqv", "ig"])
                            for col in ("open", "high", "low", "close", "volume"):
                                gh_df[col] = gh_df[col].astype(float)
                            gh_df["date"] = pd.to_datetime(gh_df["ms"].astype(np.int64), unit="ms", utc=True)
                            h_parts.append(gh_df)
                h_parts.append(h1)
            if h_parts:
                h_cols = ["date", "open", "high", "low", "close", "volume"]
                mh = pd.concat([p[h_cols] for p in h_parts], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
                _atomic_write_parquet(mh, dst_1h, index=False)
                row_counts[f"{sym}:1h:{kind}"] = len(mh)

        # 8h funding
        src_fr = runtime_raw_dir / "auxiliary" / f"{sym}_8h_funding.parquet"
        dst_fr = training_raw_dir / "auxiliary" / f"{sym}_8h_funding.parquet"
        f_parts = []
        if dst_fr.is_file():
            f0 = pd.read_parquet(dst_fr)
            f0["date"] = pd.to_datetime(f0["date"], utc=True)
            f_parts.append(f0[["date", "funding_rate"]])
        if src_fr.is_file():
            f1 = pd.read_parquet(src_fr)
            f1["date"] = pd.to_datetime(f1["date"], utc=True)
            f_parts.append(f1[["date", "funding_rate"]])
        if f_parts:
            mf = pd.concat(f_parts, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
            _atomic_write_parquet(mf, dst_fr, index=False)
            row_counts[f"{sym}:funding"] = len(mf)

    # 2. Merge binance_basis_4h & deribit_dvol_daily
    src_bas = runtime_raw_dir / "auxiliary/binance_basis_4h.parquet"
    dst_bas = training_raw_dir / "auxiliary/binance_basis_4h.parquet"
    b_parts = []
    if dst_bas.is_file():
        b0 = pd.read_parquet(dst_bas)
        if not isinstance(b0.index, pd.DatetimeIndex):
            b0 = b0.set_index(pd.to_datetime(b0["date"], utc=True))
        b0.index = pd.to_datetime(b0.index, utc=True)
        b_parts.append(b0[list(model.SYMBOLS[:3])])
    if src_bas.is_file():
        b1 = pd.read_parquet(src_bas)
        if not isinstance(b1.index, pd.DatetimeIndex):
            b1 = b1.set_index(pd.to_datetime(b1["date"], utc=True))
        b1.index = pd.to_datetime(b1.index, utc=True)
        b_parts.append(b1[list(model.SYMBOLS[:3])])
    if b_parts:
        mb = pd.concat(b_parts)
        mb = mb[~mb.index.duplicated(keep="last")].sort_index()
        mb.index.name = "date"
        _atomic_write_parquet(mb, dst_bas, index=True)
        row_counts["basis:4h"] = len(mb)

    src_dv = runtime_raw_dir / "auxiliary/deribit_dvol_daily_2021_2026.parquet"
    dst_dv = training_raw_dir / "auxiliary/deribit_dvol_daily_2021_2026.parquet"
    dv_parts = []
    for p_dv in (dst_dv, src_dv):
        if p_dv.is_file():
            dv = pd.read_parquet(p_dv)
            dv["bucket_utc"] = pd.to_datetime(dv["bucket_utc"], utc=True)
            dv_parts.append(dv[["bucket_utc", "symbol", "open", "high", "low", "close"]])
    if dv_parts:
        mdv = pd.concat(dv_parts, ignore_index=True).drop_duplicates(["symbol", "bucket_utc"], keep="last").sort_values(["symbol", "bucket_utc"]).reset_index(drop=True)
        _atomic_write_parquet(mdv, dst_dv, index=False)
        row_counts["dvol:1d"] = len(mdv)

    if fetch_network_gaps and now_ms > 0:
        fetch_spot_1m_incremental(training_raw_dir)
    return row_counts


def evaluate_retrain_gatekeeper(
    run_summary: Mapping[str, Any],
    enabled_streams: Sequence[str],
    *,
    min_mean_val_accuracy: float = 0.535,
    min_stream_val_accuracy: float = 0.515,
) -> tuple[bool, str, dict[str, Any]]:
    """Verify staged retrain summary meets strict parity & validation accuracy thresholds before promotion."""
    if run_summary.get("status") != "COMPLETED":
        return False, f"TRAINING_NOT_COMPLETED:{run_summary.get('status')}", {}
    by_stream = {str(item["stream"]): item for item in (run_summary.get("completed") or []) if isinstance(item, Mapping)}
    accs = []
    per_stream = {}
    for st in enabled_streams:
        item = by_stream.get(str(st))
        if item is None:
            return False, f"MISSING_ENABLED_STREAM:{st}", {}
        if item.get("saved_model_parity") != "PASS":
            return False, f"PARITY_FAILED:{st}", {}
        acc = float(item.get("validation_twap_accuracy") or 0.0)
        if not np.isfinite(acc) or acc < min_stream_val_accuracy:
            return False, f"STREAM_VAL_ACCURACY_BELOW_FLOOR:{st}={acc:.4f}<{min_stream_val_accuracy:.4f}", {}
        accs.append(acc)
        per_stream[str(st)] = round(acc, 6)
    mean_acc = float(np.mean(accs)) if accs else 0.0
    if mean_acc < min_mean_val_accuracy:
        return False, f"MEAN_VAL_ACCURACY_BELOW_FLOOR:{mean_acc:.4f}<{min_mean_val_accuracy:.4f}", {"per_stream": per_stream, "mean_accuracy": mean_acc}
    return True, "GATEKEEPER_PASSED", {"per_stream": per_stream, "mean_accuracy": round(mean_acc, 6)}


def atomic_promote_and_pin_policy(
    *,
    staging_model_root: Path,
    live_model_root: Path = model.MODEL_ROOT,
    policy_path: Path = ROOT / "config/argon4_forward_policy.json",
    phase: str = "deployment_q4",
    oracle_tag: str | None = None,
    wait_for_safe_window: bool = True,
) -> dict[str, Any]:
    """Atomically swap staged bundles into live_model_root and pin policy.json for zero-downtime hot-reload."""
    staging_phase = Path(staging_model_root).resolve() / phase
    live_phase = Path(live_model_root).resolve() / phase
    live_phase.mkdir(parents=True, exist_ok=True)
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    streams = list(policy.get("deployment_manifest_sha256", {}).keys()) or list(model.STREAMS.keys())

    # Pre-verify all staged artifacts before touching live directory
    new_pins: dict[str, str] = {}
    for st in streams:
        m_file = staging_phase / f"{st}.manifest.json"
        j_file = staging_phase / f"{st}.joblib"
        s_file = staging_phase / f"{st}.manifest.json.sha256"
        if not (m_file.is_file() and j_file.is_file() and s_file.is_file()):
            raise RuntimeError(f"Missing staged artifact for {st}")
        digest = model.sha256(m_file)
        if s_file.read_text(encoding="ascii").strip() != digest:
            raise RuntimeError(f"Staged manifest sha256 mismatch for {st}")
        meta = json.loads(m_file.read_text(encoding="utf-8"))
        if model.sha256(j_file) != meta.get("model_sha256"):
            raise RuntimeError(f"Staged joblib sha256 mismatch for {st}")
        new_pins[st] = digest

    if wait_for_safe_window:
        elapsed = time.time() % 300.0
        if elapsed < 25.0 or elapsed > 292.0:
            sleep_s = (25.5 - elapsed) if elapsed < 25.0 else (300.0 - elapsed + 25.5)
            time.sleep(max(0.1, sleep_s))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = Path(live_model_root).resolve() / "backups" / f"{phase}_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for existing in live_phase.glob("*"):
        if existing.is_file() and not existing.is_symlink():
            shutil.copy2(existing, backup_dir / existing.name)

    for st in streams:
        for suffix in (".joblib", ".manifest.json", ".manifest.json.sha256"):
            src = staging_phase / f"{st}{suffix}"
            dst = live_phase / f"{st}{suffix}"
            tmp_dst = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
            shutil.copy2(src, tmp_dst)
            os.replace(tmp_dst, dst)

    # Verify reload via model_v14 loader when promoting into canonical MODEL_ROOT
    if Path(live_model_root).resolve() == model.MODEL_ROOT.resolve():
        for st, digest in new_pins.items():
            model_v14.load_model_bundle_v14(st, phase=phase, expected_manifest_sha256=digest)

    old_pins = dict(policy.get("deployment_manifest_sha256") or {})
    policy["previous_deployment_manifest_sha256"] = old_pins
    policy["deployment_manifest_sha256"] = new_pins
    policy["deployment_model_target_oracle"] = oracle_tag or f"S4_SPOT_TWAP60_BLEND_HL270D_AUTO_{stamp[:8]}"
    _atomic_write_json(Path(policy_path), policy)
    return {
        "promoted_at_utc": stamp,
        "backup_dir": str(backup_dir),
        "deployment_manifest_sha256": new_pins,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Autonomous ARGON-4 TWAP60 rolling retrainer & hot-reload promoter.")
    ap.add_argument("--runtime-raw", type=Path, default=ROOT / "paper_runtime/argon4_forward_20261105/features/raw_data")
    ap.add_argument("--training-raw", type=Path, default=model.DATA_ROOT)
    ap.add_argument("--model-root", type=Path, default=model.MODEL_ROOT)
    ap.add_argument("--policy", type=Path, default=ROOT / "config/argon4_forward_policy.json")
    ap.add_argument("--phase", type=str, default="deployment_q4")
    ap.add_argument("--validation-days", type=int, default=60)
    ap.add_argument("--valid-horizon-days", type=int, default=95)
    ap.add_argument("--half-life-days", type=float, default=270.0)
    ap.add_argument("--min-mean-val-acc", type=float, default=0.535)
    ap.add_argument("--min-stream-val-acc", type=float, default=0.515)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-network-fetch", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    twap60_retrain.verify_frozen_model_untouched()
    sync_counts = sync_runtime_raw_to_training_root(
        args.runtime_raw, args.training_raw, fetch_network_gaps=not args.no_network_fetch
    )
    fold = build_rolling_fold(
        validation_days=args.validation_days,
        valid_horizon_days=args.valid_horizon_days,
        half_life_days=args.half_life_days,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging_root = Path(args.model_root).resolve() / f".staging_retrain_{stamp}"
    try:
        run_summary = twap60_retrain.train_twap60_streams(
            data_root=args.training_raw,
            model_root=staging_root,
            phase=args.phase,
            fold_override=fold,
            threads=args.threads,
            model_version_tag=f"local311_independent65_{args.phase}_auto_{stamp[:8]}",
        )
        pol = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        enabled = (pol.get("execution_stage") or {}).get("enabled_streams") or list(model.STREAMS.keys())[:6]
        ok, reason, gate_stats = evaluate_retrain_gatekeeper(
            run_summary,
            enabled,
            min_mean_val_accuracy=args.min_mean_val_acc,
            min_stream_val_accuracy=args.min_stream_val_acc,
        )
        if not ok:
            print(json.dumps({"status": "REJECTED_BY_GATEKEEPER", "reason": reason, "gate_stats": gate_stats}, indent=2))
            return 2
        promoted = atomic_promote_and_pin_policy(
            staging_model_root=staging_root,
            live_model_root=args.model_root,
            policy_path=args.policy,
            phase=args.phase,
        )
        print(json.dumps({
            "status": "PROMOTED_HOT_RELOAD_READY",
            "synced_sources": sync_counts,
            "fold": fold,
            "gatekeeper": gate_stats,
            "promotion": promoted,
        }, indent=2))
        return 0
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
