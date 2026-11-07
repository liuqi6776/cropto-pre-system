#!/usr/bin/env python3
"""
Step 1: Zero-Dependency Public Data Crawler (`step1_fetch_all_public_data.py`)
==============================================================================
Downloads all required 2023-01-01 -> 2026-09-24 historical datasets directly from:
  1. Binance Official Archive (`https://data.binance.vision`):
     - USD-M Futures 5m klines: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT
     - USD-M Futures 1h klines & 1h markPriceKlines: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT
     - USD-M Futures 8h fundingRate: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT
     - Spot 4h klines (for 4h spot-futures basis): BTCUSDT, ETHUSDT, SOLUSDT
  2. Deribit Public API (`https://www.deribit.com/api/v2/public/get_volatility_index_data`):
     - Daily BTC & ETH Implied Volatility Index (DVOL)
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path("./raw_data")
INTRADAY_DIR = RAW_DIR / "intraday"
AUX_DIR = RAW_DIR / "auxiliary"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
MONTHS_2023_2026 = [
    f"{yr}-{m:02d}"
    for yr in (2023, 2024, 2025, 2026)
    for m in range(1, 13)
    if not (yr == 2026 and m > 9)
]
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]


def _fetch_bytes(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.read()
    except Exception:
        return None


def _download_vision_klines(base_path: str, sym: str, interval: str, months: list[str]) -> pd.DataFrame:
    dfs = []
    for ym in months:
        url = f"https://data.binance.vision/data/{base_path}/monthly/klines/{sym}/{interval}/{sym}-{interval}-{ym}.zip"
        raw = _fetch_bytes(url)
        if raw is None:
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = zf.namelist()[0]
            df = pd.read_csv(zf.open(csv_name), header=None, names=KLINE_COLS)
            df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()].copy()
            dfs.append(df)
    if not dfs:
        raise RuntimeError(f"Failed to download {sym} {interval} from data.binance.vision")
    out = pd.concat(dfs, ignore_index=True)
    out["open_time"] = pd.to_numeric(out["open_time"], errors="coerce").astype("int64")
    out.loc[out["open_time"] > 10**14, "open_time"] //= 1000
    out["open_time"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)
    return out


def fetch_all() -> None:
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    AUX_DIR.mkdir(parents=True, exist_ok=True)

    for sym in SYMBOLS:
        p5 = INTRADAY_DIR / f"{sym}_5m_futures_2023_2026.parquet"
        if not p5.exists():
            print(f"[1/4] Downloading {sym} 5m futures klines (2023-2026)...")
            df5 = _download_vision_klines("futures/um", sym, "5m", MONTHS_2023_2026)
            df5.to_parquet(p5)
            print(f"  -> Saved {p5} ({len(df5):,} rows)")

    for sym in SYMBOLS:
        p_mk = AUX_DIR / f"{sym}_1h_mark.parquet"
        if not p_mk.exists():
            print(f"[2/4] Downloading {sym} 1h mark & 1h futures klines...")
            dfs_mk = []
            for ym in MONTHS_2023_2026:
                u = f"https://data.binance.vision/data/futures/um/monthly/markPriceKlines/{sym}/1h/{sym}-1h-{ym}.zip"
                raw = _fetch_bytes(u)
                if raw:
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        df = pd.read_csv(zf.open(zf.namelist()[0]), header=None, names=KLINE_COLS)
                        df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()].copy()
                        dfs_mk.append(df)
            df_mk = pd.concat(dfs_mk, ignore_index=True)
            df_mk["date"] = pd.to_datetime(pd.to_numeric(df_mk["open_time"]).astype("int64"), unit="ms", utc=True)
            for c in ("open", "high", "low", "close"):
                df_mk[c] = pd.to_numeric(df_mk[c], errors="coerce")
            df_mk[["date", "open", "high", "low", "close"]].drop_duplicates("date").sort_values("date").to_parquet(p_mk)

        p_tr = AUX_DIR / f"{sym}_1h_futures.parquet"
        if not p_tr.exists():
            df_tr = _download_vision_klines("futures/um", sym, "1h", MONTHS_2023_2026)
            df_tr = df_tr.rename(columns={"open_time": "date"})[["date", "open", "high", "low", "close", "volume"]]
            df_tr.to_parquet(p_tr)

        p_fr = AUX_DIR / f"{sym}_8h_funding.parquet"
        if not p_fr.exists():
            dfs_fr = []
            for ym in MONTHS_2023_2026:
                u = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"
                raw = _fetch_bytes(u)
                if raw:
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        df = pd.read_csv(zf.open(zf.namelist()[0]))
                        dfs_fr.append(df)
            df_fr = pd.concat(dfs_fr, ignore_index=True)
            t_col = "calc_time" if "calc_time" in df_fr.columns else df_fr.columns[0]
            r_col = "last_funding_rate" if "last_funding_rate" in df_fr.columns else df_fr.columns[-1]
            df_fr["date"] = pd.to_datetime(pd.to_numeric(df_fr[t_col], errors="coerce"), unit="ms", utc=True)
            df_fr["funding_rate"] = pd.to_numeric(df_fr[r_col], errors="coerce")
            df_fr[["date", "funding_rate"]].dropna().drop_duplicates("date").sort_values("date").to_parquet(p_fr)

    p_b4 = AUX_DIR / "binance_basis_4h.parquet"
    if not p_b4.exists():
        print("[3/4] Computing 4h Spot-Futures Basis for BTCUSDT, ETHUSDT, SOLUSDT...")
        basis_dict = {}
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            f4 = _download_vision_klines("futures/um", sym, "4h", MONTHS_2023_2026).set_index("open_time")["close"]
            s4 = _download_vision_klines("spot", sym, "4h", MONTHS_2023_2026).set_index("open_time")["close"]
            joined = pd.concat([f4.rename("fut"), s4.rename("spot")], axis=1).dropna()
            basis_dict[sym] = (joined["fut"] - joined["spot"]) / joined["spot"]
        pd.DataFrame(basis_dict).sort_index().to_parquet(p_b4)

    p_dv = AUX_DIR / "deribit_dvol_daily_2021_2026.parquet"
    if not p_dv.exists():
        print("[4/4] Fetching Deribit Daily DVOL for BTC & ETH...")
        rows = []
        for ccy in ("BTC", "ETH"):
            cur_end = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)
            start_lim = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            while cur_end > start_lim:
                u = (
                    f"https://www.deribit.com/api/v2/public/get_volatility_index_data"
                    f"?currency={ccy}&start_timestamp={start_lim}&end_timestamp={cur_end}&resolution=1D"
                )
                raw = _fetch_bytes(u)
                if not raw:
                    break
                res = json.loads(raw.decode())["result"]["data"]
                if not res:
                    break
                for ts_ms, o, h, l, c in res:
                    rows.append(
                        {
                            "bucket_utc": pd.to_datetime(ts_ms, unit="ms", utc=True),
                            "symbol": ccy,
                            "open": float(o),
                            "high": float(h),
                            "low": float(l),
                            "close": float(c),
                        }
                    )
                min_ts = min(r[0] for r in res)
                if min_ts >= cur_end:
                    break
                cur_end = min_ts - 1
        pd.DataFrame(rows).drop_duplicates(["symbol", "bucket_utc"]).sort_values(["symbol", "bucket_utc"]).to_parquet(p_dv)
    print("[DONE] All public datasets downloaded and verified in ./raw_data/")


if __name__ == "__main__":
    fetch_all()
