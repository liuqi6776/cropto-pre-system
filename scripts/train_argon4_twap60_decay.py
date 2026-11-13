"""CLI for ARGON-4 bimonthly Chainlink TWAP60 + 270-day exponential decay retraining.

Preserves frozen ``tools/argon4_forward/model.py`` byte-for-byte while building
or updating ``spot_1m`` OHLC4 TWAP60 parquets and training deployment bundles.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.argon4_forward import model, twap60_retrain


def fetch_spot_1m_incremental(data_root: Path, symbols: tuple[str, ...] = model.SYMBOLS, months_back: int = 14) -> None:
    """Incrementally fetch public 1m spot klines from data-api.binance.vision into data_root/spot_1m."""
    out_dir = Path(data_root) / "spot_1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    now_utc = pd.Timestamp.now(tz="UTC").floor("1min")
    default_start = now_utc - pd.Timedelta(days=int(months_back * 31))

    def _fetch_one(sym: str):
        out_path = out_dir / f"{sym}_1m_spot.parquet"
        existing = pd.DataFrame()
        start_ts = default_start
        if out_path.is_file():
            existing = pd.read_parquet(out_path)
            if not existing.empty and "bound_ts" in existing.columns:
                existing["bound_ts"] = pd.to_datetime(existing["bound_ts"], utc=True)
                start_ts = existing["bound_ts"].max() - pd.Timedelta(minutes=5)
        cur_ms = int((start_ts - pd.Timedelta(minutes=1)).timestamp() * 1000)
        end_ms = int((now_utc - pd.Timedelta(minutes=1)).timestamp() * 1000)
        sess = requests.Session()
        sess.trust_env = False
        rows = []
        while cur_ms <= end_ms:
            for att in range(4):
                try:
                    resp = sess.get(
                        "https://data-api.binance.vision/api/v3/klines",
                        params={"symbol": sym, "interval": "1m", "startTime": cur_ms, "endTime": end_ms, "limit": 1000},
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
            for b in batch:
                open_ms = int(b[0])
                if open_ms > end_ms:
                    continue
                o, h, l, c = float(b[1]), float(b[2]), float(b[3]), float(b[4])
                rows.append({
                    "bound_ts": pd.to_datetime(open_ms + 60_000, unit="ms", utc=True),
                    "spot_open": o,
                    "spot_high": h,
                    "spot_low": l,
                    "spot_close": c,
                    "twap60": 0.25 * (o + h + l + c),
                })
            cur_ms = int(batch[-1][0]) + 60_000
            if len(batch) < 1000:
                break
        new_df = pd.DataFrame(rows)
        if not existing.empty and not new_df.empty:
            comb = pd.concat([existing, new_df], ignore_index=True)
        elif not new_df.empty:
            comb = new_df
        else:
            comb = existing
        comb = comb.drop_duplicates("bound_ts", keep="last").sort_values("bound_ts").reset_index(drop=True)
        tmp = out_path.with_suffix(".parquet.tmp")
        comb.to_parquet(tmp, index=False)
        tmp.replace(out_path)
        return sym, len(comb)

    with ThreadPoolExecutor(max_workers=min(4, len(symbols))) as pool:
        for sym, count in pool.map(_fetch_one, symbols):
            print(f"SPOT_1M {sym}: {count:,} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ARGON-4 TWAP60 + exponential decay deployment models.")
    ap.add_argument("--data-root", type=Path, default=ROOT / "research_data/argon4_twap60_20261113/raw_data")
    ap.add_argument("--model-root", type=Path, default=model.MODEL_ROOT)
    ap.add_argument("--phase", type=str, default="deployment_q4")
    ap.add_argument("--label-mode", type=str, choices=twap60_retrain.LABEL_MODES, default=None)
    ap.add_argument("--half-life-days", type=float, default=None)
    ap.add_argument("--train-end", type=str, default=None)
    ap.add_argument("--validation-start", type=str, default=None)
    ap.add_argument("--validation-end", type=str, default=None)
    ap.add_argument("--valid-from", type=str, default=None)
    ap.add_argument("--valid-until", type=str, default=None)
    ap.add_argument("--model-version-tag", type=str, default=None)
    ap.add_argument("--fetch-spot-1m", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    if args.fetch_spot_1m:
        fetch_spot_1m_incremental(args.data_root)

    overrides = {}
    if args.label_mode is not None:
        overrides["label_mode"] = args.label_mode
    if args.half_life_days is not None:
        overrides["half_life_days"] = args.half_life_days
    if args.train_end is not None:
        overrides["train_end"] = args.train_end
    if args.validation_start is not None:
        overrides["validation_start"] = args.validation_start
    if args.validation_end is not None:
        overrides["validation_end"] = args.validation_end
    if args.valid_from is not None:
        overrides["valid_from"] = args.valid_from
    if args.valid_until is not None:
        overrides["valid_until"] = args.valid_until

    summary = twap60_retrain.train_twap60_streams(
        data_root=args.data_root,
        model_root=args.model_root,
        phase=args.phase,
        fold_override=overrides or None,
        threads=args.threads,
        model_version_tag=args.model_version_tag,
    )
    print(json.dumps(summary["deployment_manifest_sha256"], indent=2))


if __name__ == "__main__":
    main()
