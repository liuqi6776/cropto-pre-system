"""Public, checksum-verified inputs for the 2026-11-05 supplied clean-room code.

No credentials, exchange order endpoints, legacy caches, or source_data mutations.
Monthly archives retain the attachment's complete September download; model folds
stop on September 24. Missing partitions are errors, never silently skipped.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = "argon4_clean_room_20261105"
DATA = ROOT / "research_data" / STUDY
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
MONTHS = [str(p) for p in pd.period_range("2023-01", "2026-09", freq="M")]
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")
    tmp.replace(path)


def fetch(url):
    if not url.startswith(("https://data.binance.vision/data/", "https://www.deribit.com/api/v2/public/get_volatility_index_data?")):
        raise ValueError("URL outside public-data allowlist")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "argon4-reproduction-research/1"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def specifications():
    specs = []
    for sym in SYMBOLS:
        for kind, tf in (("klines", "5m"), ("klines", "1h"), ("markPriceKlines", "1h"), ("fundingRate", "events")):
            specs.append(("futures/um", kind, sym, tf))
    for sym in SYMBOLS[:3]:
        specs.extend([("futures/um", "klines", sym, "4h"), ("spot", "klines", sym, "4h")])
    return specs


def partition(spec, month, frequency="monthly"):
    market, kind, sym, tf = spec
    tail = f"{kind}/{sym}/{tf}/{sym}-{tf}-{month}.zip" if kind != "fundingRate" else f"{kind}/{sym}/{sym}-fundingRate-{month}.zip"
    url = f"https://data.binance.vision/data/{market}/{frequency}/{tail}"
    p = DATA / "archives" / market / ("" if frequency == "monthly" else "daily") / tail
    p.parent.mkdir(parents=True, exist_ok=True)
    receipt = p.with_suffix(".receipt.json")
    if p.exists() and receipt.exists():
        meta = json.loads(receipt.read_text(encoding="utf8"))
        if sha(p) != meta["official_sha256"]:
            raise ValueError(f"Cached archive changed: {p}")
        return meta
    check = fetch(url + ".CHECKSUM")
    official = check.decode().split()[0].lower()
    if len(official) != 64:
        raise ValueError(f"Invalid checksum: {url}")
    raw = fetch(url)
    if hashlib.sha256(raw).hexdigest() != official:
        raise ValueError(f"Checksum mismatch: {url}")
    p.write_bytes(raw)
    p.with_suffix(".zip.CHECKSUM").write_bytes(check)
    meta = {"url": url, "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "official_sha256": official, "bytes": len(raw), "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "spec": list(spec), "month": month, "frequency": frequency}
    write_json(receipt, meta)
    return meta


def read_partition(p, funding=False):
    with zipfile.ZipFile(p) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Unexpected archive members: {p}")
        with z.open(names[0]) as f:
            d = pd.read_csv(f) if funding else pd.read_csv(f, header=None, names=COLS, low_memory=False)
    if funding:
        if not {"calc_time", "last_funding_rate"}.issubset(d):
            raise ValueError(f"Unknown funding schema: {p}")
        d["date"] = pd.to_datetime(pd.to_numeric(d.calc_time, errors="raise"), unit="ms", utc=True)
        d["funding_rate"] = pd.to_numeric(d.last_funding_rate, errors="raise")
        return d[["date", "funding_rate"]]
    if len(d) and d.iloc[0].open_time == "open_time":
        d = d.iloc[1:].copy()
    for col in COLS:
        d[col] = pd.to_numeric(d[col], errors="raise")
    ts = d.open_time.astype("int64")
    # Binance changed SPOT archive units to microseconds in January 2025.
    units = np.where(ts > 10**14, 1000, 1)
    d["open_time"] = pd.to_datetime(ts // units, unit="ms", utc=True)
    return d


def validate(d, col, step=None):
    t = pd.DatetimeIndex(d[col])
    if len(t) == 0 or t.has_duplicates or not t.is_monotonic_increasing:
        raise ValueError("Empty, duplicated or unordered observations")
    numeric = [c for c in ("open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_quote_volume", "funding_rate") if c in d]
    if not np.isfinite(d[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite observed market data")
    if "close" in d:
        if (d[["open", "high", "low", "close"]] <= 0).any().any() or (d.high < d[["open", "close", "low"]].max(axis=1)).any() or (d.low > d[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError("Invalid OHLC")
    if "volume" in d and (d.volume < 0).any():
        raise ValueError("Negative volume")
    deltas = pd.Series(t[1:] - t[:-1])
    counts = {str(k): int(v) for k, v in deltas.value_counts().items()}
    gaps = [] if step is None else [str(t[i + 1]) for i in np.flatnonzero(deltas.to_numpy() != pd.Timedelta(step).to_timedelta64())]
    return {"rows": len(d), "first_utc": t[0].isoformat(), "last_utc": t[-1].isoformat(), "interval_counts": counts, "gaps": gaps}


def save_frame(d, path, audit):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        pd.testing.assert_frame_equal(old, d)
    else:
        d.to_parquet(path, index=False)
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path), "bytes": path.stat().st_size, **audit}


def registry_entry(id_, path, market_type, kind, symbols, tf, evidence, verified=False):
    return {"id": f"{STUDY}_{id_}", "source_path": str(path.resolve()).replace("\\", "/"),
            "symbols": list(symbols), "exchange": "deribit" if kind == "dvol" else "binance",
            "market_type": market_type, "data_type": kind, "timeframe": tf, "frequency": tf,
            "coverage": "2023-01-01 .. 2026-09-30 UTC; see study data_manifest.json",
            "verification_status": "VERIFIED_PERPETUAL_LABEL" if verified else "EXTERNAL_REFERENCE_ONLY",
            "label_evidence": evidence, "convertible": False, "default_selected": False,
            "notes": "Independent clean-room public-data download. Research-only; no Freqtrade gate effect."}


def register(manifest):
    p = ROOT / "archive/catalog/dataset_registry.json"
    reg = json.loads(p.read_text(encoding="utf8"))
    entries = []
    for item in manifest["datasets"]:
        spec = item.get("spec")
        path = ROOT / item["path"]
        if spec:
            market, kind, sym, tf = spec
            mt = "funding_rate" if kind == "fundingRate" else "futures_usds_m_perpetual"
            if market == "spot":
                mt = "basis"  # explicitly spot research input, not perpetual/convertible
            ev = f"Official {market}/monthly/{kind}/{sym}/{tf} archive; every source ZIP matched .CHECKSUM. Manifest: research_data/{STUDY}/data_manifest.json. SHA256={item['sha256']}."
            if market == "spot":
                ev += " SPOT observations retained solely for spot-futures basis, never relabelled as perpetual."
            entries.append(registry_entry("_".join(spec).replace("/", "_"), path, mt, kind, [sym], tf, ev, market != "spot"))
        else:
            entries.append(registry_entry(item["kind"], path, "basis" if item["kind"] == "basis" else "options_volatility_index", item["kind"], SYMBOLS[:3] if item["kind"] == "basis" else ["BTC", "ETH"], "4h" if item["kind"] == "basis" else "1d", f"Public source receipts and SHA256 in research_data/{STUDY}/data_manifest.json; SHA256={item['sha256']}"))
    ids = {e["id"] for e in entries}
    reg["datasets"] = [d for d in reg["datasets"] if d["id"] not in ids] + entries
    write_json(p, reg)
    p = ROOT / "archive/catalog/mark_candle_registry.json"
    reg = json.loads(p.read_text(encoding="utf8"))
    refs = []
    for item in manifest["datasets"]:
        if item.get("spec", [None, None])[1] != "markPriceKlines":
            continue
        sym = item["spec"][2]
        refs.append({"dataset_id": f"{STUDY}_mark_{sym}", "pair": f"{sym[:-4]}/USDT:USDT", "timeframe": "1h", "candle_type": "mark", "local_path": item["path"], "extension": "parquet", "sha256": item["sha256"], "rows": item["rows"], "first_utc": item["first_utc"], "last_utc": item["last_utc"], "verification_status": "VERIFIED_PERPETUAL_LABEL", "provenance_evidence": "Official Binance USD-M markPriceKlines; independently downloaded and checksum matched; original provider volume retained. See study data_manifest.json.", "installed": False, "research_only": True, "gate_effect": "NONE"})
    ids = {e["dataset_id"] for e in refs}
    reg["research_references"] = [d for d in reg.get("research_references", []) if d["dataset_id"] not in ids] + refs
    write_json(p, reg)


def run(workers=8):
    DATA.mkdir(parents=True, exist_ok=True)
    jobs = [(s, m) for s in specifications() for m in MONTHS]
    write_json(DATA / "download_plan.json", {"status": "PUBLIC_READ_ONLY", "specifications": specifications(), "months": MONTHS})
    receipts, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(partition, s, m): (s, m) for s, m in jobs}
        for i, f in enumerate(as_completed(tasks), 1):
            try:
                receipts.append(f.result())
            except Exception as e:
                failures.append({"partition": tasks[f], "error": repr(e)})
            if i % 50 == 0:
                print(f"ARCHIVES {i}/{len(jobs)} failures={len(failures)}", flush=True)
    write_json(DATA / "download_failures.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} missing partitions; no complete-data claim")
    manifest = {"status": "OBSERVED_PUBLIC_DATA", "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(), "archives": sorted(receipts, key=lambda x: x["url"]), "datasets": [], "observed_daily_repairs": []}
    frames = {}
    for spec in specifications():
        market, kind, sym, tf = spec
        items = sorted([r for r in receipts if r["spec"] == list(spec)], key=lambda x: x["month"])
        d = pd.concat([read_partition(ROOT / r["path"], kind == "fundingRate") for r in items], ignore_index=True)
        col = "date" if kind == "fundingRate" else "open_time"
        d = d.sort_values(col).reset_index(drop=True)
        audit = validate(d, col, None if kind == "fundingRate" else {"5m": "5min", "1h": "1h", "4h": "4h"}[tf])
        if kind != "fundingRate":
            step = {"5m": "5min", "1h": "1h", "4h": "4h"}[tf]
            expected = pd.date_range("2023-01-01", "2026-10-01", freq=step, inclusive="left", tz="UTC")
            missing = expected.difference(pd.DatetimeIndex(d[col]))
            if len(missing):
                write_json(DATA / f"gaps_{sym}_{market.replace('/', '_')}_{kind}_{tf}.json", {**audit, "missing_utc": [str(t) for t in missing]})
                for day in sorted(set(missing.strftime("%Y-%m-%d"))):
                    receipt = partition(spec, day, "daily")
                    observed = read_partition(ROOT / receipt["path"])
                    repair = observed[observed[col].isin(missing)]
                    if repair.empty:
                        raise ValueError(f"Official daily archive does not repair {spec} {day}")
                    manifest["observed_daily_repairs"].append({**receipt, "added_utc": [str(t) for t in repair[col]], "added_rows": len(repair)})
                    d = pd.concat([d, repair], ignore_index=True).sort_values(col).reset_index(drop=True)
                audit = validate(d, col, step)
            if not pd.DatetimeIndex(d[col]).equals(expected):
                raise ValueError(f"Incomplete candle coverage: {spec}; no imputation")
        if kind == "fundingRate":
            dominant = max(audit["interval_counts"], key=audit["interval_counts"].get)
            if pd.Timedelta(dominant) != pd.Timedelta("8h"):
                raise ValueError("Attachment filename 8h disagrees with dominant observed funding grid")
            path = DATA / "raw_data/auxiliary" / f"{sym}_8h_funding.parquet"
        elif kind == "markPriceKlines":
            d = d.rename(columns={"open_time": "date"})[["date", "open", "high", "low", "close", "volume"]]
            path = DATA / "raw_data/auxiliary" / f"{sym}_1h_mark.parquet"
        elif tf == "5m":
            path = DATA / "raw_data/intraday" / f"{sym}_5m_futures_2023_2026.parquet"
        elif tf == "1h":
            d = d.rename(columns={"open_time": "date"})[["date", "open", "high", "low", "close", "volume"]]
            path = DATA / "raw_data/auxiliary" / f"{sym}_1h_futures.parquet"
        else:
            path = DATA / "basis_inputs" / f"{sym}_{market.replace('/', '_')}_4h.parquet"
        manifest["datasets"].append({**save_frame(d, path, audit), "spec": list(spec)})
        frames[spec] = d
        print(f"NORMALIZED {spec}: {len(d)} rows", flush=True)
    basis = {}
    for sym in SYMBOLS[:3]:
        f4 = frames[("futures/um", "klines", sym, "4h")].set_index("open_time").close
        s4 = frames[("spot", "klines", sym, "4h")].set_index("open_time").close
        if not f4.index.equals(s4.index):
            raise ValueError("Basis grid mismatch")
        basis[sym] = (f4 - s4) / s4
    b = pd.DataFrame(basis)
    bp = DATA / "raw_data/auxiliary/binance_basis_4h.parquet"
    if bp.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(bp), b)
    else:
        b.to_parquet(bp)
    manifest["datasets"].append({"kind": "basis", "path": str(bp.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(bp), "rows": len(b)})
    rows, dv_receipts = [], []
    for sym in ("BTC", "ETH"):
        start = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)
        end = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp() * 1000)
        while end > start:
            url = f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={sym}&start_timestamp={start}&end_timestamp={end}&resolution=1D"
            jp = DATA / "deribit" / f"{sym}_{end}.json"
            jp.parent.mkdir(parents=True, exist_ok=True)
            if not jp.exists():
                jp.write_bytes(fetch(url))
            payload = json.loads(jp.read_bytes())
            if "error" in payload:
                raise ValueError(payload["error"])
            got = payload["result"]["data"]
            dv_receipts.append({"url": url, "path": str(jp.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(jp)})
            if not got:
                break
            rows.extend({"bucket_utc": pd.to_datetime(t, unit="ms", utc=True), "symbol": sym, "open": o, "high": h, "low": l, "close": c} for t, o, h, l, c in got)
            earliest = min(r[0] for r in got)
            if earliest >= end:
                raise ValueError("DVOL pagination stalled")
            end = earliest - 1
    dv = pd.DataFrame(rows).sort_values(["symbol", "bucket_utc"]).reset_index(drop=True)
    # Provider pages can repeat an endpoint; reject conflicting copies.
    for _, grp in dv[dv.duplicated(["symbol", "bucket_utc"], keep=False)].groupby(["symbol", "bucket_utc"]):
        if len(grp.drop_duplicates()) != 1:
            raise ValueError("Conflicting DVOL duplicates")
    dv = dv.drop_duplicates(["symbol", "bucket_utc"]).reset_index(drop=True)
    audits = {s: validate(g, "bucket_utc", "1d") for s, g in dv.groupby("symbol")}
    for s in ("BTC", "ETH"):
        a = audits[s]
        if a["gaps"] or pd.Timestamp(a["first_utc"]) > pd.Timestamp("2023-01-01", tz="UTC") or pd.Timestamp(a["last_utc"]) < pd.Timestamp("2026-09-30", tz="UTC"):
            raise ValueError(f"Incomplete DVOL {s}")
    manifest["datasets"].append({**save_frame(dv, DATA / "raw_data/auxiliary/deribit_dvol_daily_2021_2026.parquet", {"symbols": audits}), "kind": "dvol"})
    manifest["deribit_receipts"] = dv_receipts
    write_json(DATA / "data_manifest.json", manifest)
    register(manifest)
    print(f"DONE {len(receipts)} verified archives; {len(manifest['datasets'])} datasets", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"specifications": specifications(), "months": MONTHS, "partitions": len(specifications()) * len(MONTHS)}))
    else:
        run(a.workers)
