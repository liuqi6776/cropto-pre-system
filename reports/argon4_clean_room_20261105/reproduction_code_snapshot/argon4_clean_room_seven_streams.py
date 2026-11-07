"""Disclosed extension: apply the supplied binary recipe to seven streams and V4.

The attachment provides no seven-stream/V4 training driver. Here V4's 65 columns
are appended to the actual supplied base+structural columns, with the four-stream
training parameters unchanged. This is an explicit experiment, not exact parity
with the headline seven-stream numbers or a prediction-market execution backtest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.argon4_clean_room_reproduce import DATA, OUT, ORIGINAL, load_module, verify_data, load_original, export_binary
from scripts.argon4_clean_room_fetch import SYMBOLS, sha, write_json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

STREAMS = [("BTC_15m", "BTCUSDT", 3, .85), ("ETH_15m", "ETHUSDT", 3, .90),
           ("BTC_5m", "BTCUSDT", 1, .92), ("ETH_5m", "ETHUSDT", 1, .92),
           ("SOL_15m", "SOLUSDT", 3, .92), ("SOL_5m", "SOLUSDT", 1, .92),
           ("BNB_15m", "BNBUSDT", 3, .92)]


def load_v4():
    spec = importlib.util.spec_from_file_location("argon4_supplied_v4", ORIGINAL / "coin_specific_v4_divergence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_stream(mod, b_obj, X_mat, name, k_bars, q_th, out, label):
    idx = b_obj["index"]
    c_mask = (idx + pd.Timedelta("5min")).minute % (k_bars * 5) == 0
    y_ret = b_obj["y0_ret"][k_bars * 5]
    valid = np.isfinite(y_ret) & (y_ret != 0.0)
    for fold in mod.FOLDS:
        tr_m = mod._slice_mask(idx, fold.train_start, fold.train_end, valid)
        va_m = mod._slice_mask(idx, fold.val_start, fold.val_end, valid & c_mask)
        te_m = mod._slice_mask(idx, fold.test_start, fold.test_end, valid & c_mask)
        tr_idx = np.flatnonzero(tr_m)[::2]
        va_idx, te_idx = np.flatnonzero(va_m), np.flatnonzero(te_m)
        sc = StandardScaler().fit(X_mat[tr_idx])
        X_tr, X_va, X_te = sc.transform(X_mat[tr_idx]), sc.transform(X_mat[va_idx]), sc.transform(X_mat[te_idx])
        y_tr = y_ret[tr_idx]
        clip_v = .008 if k_bars == 1 else .015
        ridge = Ridge(alpha=5000., random_state=42).fit(X_tr, np.clip(y_tr, -clip_v, clip_v))
        hgb = HistGradientBoostingClassifier(max_iter=60, learning_rate=.04, max_leaf_nodes=15,
            min_samples_leaf=max(50, len(X_tr) // 100), l2_regularization=20., early_stopping=False,
            random_state=42).fit(X_tr, (y_tr > 0).astype(np.int32))
        rv, rt = ridge.predict(X_va), ridge.predict(X_te)
        hv, ht = hgb.predict_proba(X_va)[:, 1], hgb.predict_proba(X_te)[:, 1]
        zscore = mod._zscore_from_val
        va_ens = .5 * zscore(rv, rv) + .5 * zscore(hv, hv)
        te_ens = .5 * zscore(rv, rt) + .5 * zscore(hv, ht)
        va_z, te_z = zscore(va_ens, va_ens), zscore(va_ens, te_ens)
        th = float(np.quantile(np.abs(va_z), q_th))
        export_binary(locals(), out, label)


def summarize(out):
    rows = []
    for label in ("baseline", "v4_append"):
        for name, sym, k, q in STREAMS:
            parts = []
            for year in (2025, 2026):
                p = out / f"{label}_{name}_{year}_predictions.parquet"
                if not p.exists() and label == "baseline" and sym in ("BTCUSDT", "ETHUSDT"):
                    p = OUT / "literal" / f"literal_{name}_{year}_predictions.parquet"
                d = pd.read_parquet(p)
                d = d[d.selected].copy()
                d["year"] = year
                parts.append(d)
            d = pd.concat(parts, ignore_index=True)
            for period in ("2025", "2026", "combined"):
                sub = d if period == "combined" else d[d.year == int(period)]
                wr = sub.correct.mean()
                rows.append({"variant": label, "stream": name, "period": period, "signals": len(sub),
                             "wins": int(sub.correct.sum()), "win_rate_pct": 100 * wr,
                             "up_signals": int((sub.side > 0).sum()), "up_win_rate_pct": 100 * sub.loc[sub.side > 0, "correct"].mean(),
                             "down_signals": int((sub.side < 0).sum()), "down_win_rate_pct": 100 * sub.loc[sub.side < 0, "correct"].mean(),
                             "calendar_days": 632 if period == "combined" else (365 if period == "2025" else 267),
                             "assumed_200_usd_at_0_515_no_cost_profit": float(np.where(sub.correct, 200 * (1 / .515 - 1), -200).sum()),
                             "scope": "Binance candle direction; ideal hypothetical payout only; no oracle/fills/fees/capacity"})
    pd.DataFrame(rows).to_csv(out / "seven_stream_summary.csv", index=False, encoding="utf-8-sig")
    return rows


def run(threads):
    load_original()
    verify_data()
    literal_meta = json.loads((OUT / "literal/run.json").read_text(encoding="utf8"))
    if literal_meta["status"] != "COMPLETED":
        raise ValueError("Complete the original four-stream run before extension")
    out = OUT / "seven_stream_extension"
    out.mkdir(parents=True, exist_ok=True)
    mod = load_module(instrument=False)
    frames = {}
    for s in SYMBOLS:
        d = pd.read_parquet(DATA / "raw_data/intraday" / f"{s}_5m_futures_2023_2026.parquet").set_index("open_time").loc["2024-01-01":]
        frames[s] = d
    if any(not frames[s].index.equals(frames["BTCUSDT"].index) for s in SYMBOLS):
        raise ValueError("Unaligned raw observations; refuse V4's implicit BTC price forward fill")
    ofi = sum((2 * d.taker_buy_quote_volume - d.quote_volume) / (d.quote_volume + 1e-12) for d in frames.values()) / 4
    v4mod = load_v4()
    start = time.time()
    metadata = {"status": "RUNNING", "classification": "EXPLORATORY_EXPLICIT_EXTENSION_NOT_LITERAL_REPLICATION",
                "data_manifest_sha256": sha(DATA / "data_manifest.json"), "runner_sha256": sha(Path(__file__)),
                "source_main_sha256": sha(ORIGINAL / "step2_standalone_clean_room_master.py"),
                "source_v4_sha256": sha(ORIGINAL / "coin_specific_v4_divergence.py"),
                "design": "Original binary recipe, thresholds and folds; seven target streams; append all supplied 65 V4 features; no unprovided risk/routing/waterfall engine inferred",
                "prior_four_stream_results": literal_meta["artifacts"], "features": {}}
    write_json(out / "run.json", metadata)
    with threadpool_limits(limits=threads):
        for sym in SYMBOLS:
            b = mod.build_clean_room_bundle(sym)
            base = np.hstack([b["features"].to_numpy(np.float32), b["spec_features"].to_numpy(np.float32)])
            v4 = v4mod.build_coin_own_structural_and_v4_mag_gap(frames[sym], frames["BTCUSDT"], ofi)
            v4.to_parquet(out / f"{sym}_v4_features.parquet")
            metadata["features"][sym] = {"baseline": base.shape[1], "v4": v4.shape[1], "v4_columns": list(v4)}
            for label in ("baseline", "v4_append"):
                if label == "baseline" and sym in ("BTCUSDT", "ETHUSDT"):
                    continue
                x = base if label == "baseline" else np.hstack([base, v4.to_numpy(np.float32)])
                for name, target, k, q in STREAMS:
                    if target == sym:
                        print(f"TRAINING {label} {name} {x.shape}", flush=True)
                        train_stream(mod, b, x, name, k, q, out, label)
            write_json(out / "run.json", metadata)
    summarize(out)
    verify_data()
    metadata.update(status="COMPLETED", elapsed_seconds=time.time() - start,
                    artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "run.json"})
    write_json(out / "run.json", metadata)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"streams": STREAMS, "design": "original recipe plus appended supplied V4"}))
    else:
        run(a.threads)
