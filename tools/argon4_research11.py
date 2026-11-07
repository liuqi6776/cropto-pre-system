"""Isolated, non-deployable eleven-stream research preparation and evaluation.

Run ``python -m tools.argon4_research11 --help``. Nothing in this module imports
the execution engine, writes production policies, or submits exchange orders.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.request

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from tools.argon4_forward import model
from scripts.argon4_clean_room_fetch import read_partition, validate

ROOT = Path(__file__).resolve().parents[1]
STUDY = "argon4_research11_20261107"
DATA = ROOT / "research_data" / STUDY
RAW = DATA / "raw_data"
OUTPUT = ROOT / "research" / STUDY
NONCORE = ROOT / "research_data/binance_noncore_dynamic_20261013"
NEW_SYMBOLS = ("XRPUSDT", "DOGEUSDT")
MONTHS = tuple(str(x) for x in pd.period_range("2024-01", "2026-09", freq="M"))
CORE = tuple(model.SYMBOLS)
CANDIDATES = {
    "BTC_15m": ("BTCUSDT", 3, .80), "ETH_5m": ("ETHUSDT", 1, .88),
    "SOL_15m": ("SOLUSDT", 3, .85), "SOL_5m": ("SOLUSDT", 1, .88),
    "BTC_5m": ("BTCUSDT", 1, .90), "ETH_15m": ("ETHUSDT", 3, .85),
    "BNB_15m": ("BNBUSDT", 3, .88), "XRP_5m": ("XRPUSDT", 1, .88),
    "XRP_15m": ("XRPUSDT", 3, .85), "DOGE_5m": ("DOGEUSDT", 1, .88),
    "DOGE_15m": ("DOGEUSDT", 3, .85),
}
LIMITATIONS = [
    "EXPLORATORY_RESEARCH_ONLY; never a production configuration or effective strategy verdict",
    "Existing seven model bytes and weights are reused, never refitted or overwritten",
    "Candidate base-score quantiles and new-model weights use validation only; historical test was already researched",
    "Binance next-open/window-close labels are not Chainlink TWAP market resolutions",
    "Historical publication-time assumption; archives do not recover as-received revisions or live latency",
    "Hypothetical fee-adjusted payouts are not executions, account PnL, capacity or profit forecasts",
    "Original derived indicator startup handling retained; no raw OHLCV or funding events imputed",
    "Noisy observations and ties are not removed from test signal selection",
    "Mandated strategy-validation-gates skill missing locally; efficacy gate remains closed",
]


def write_json(path, value):
    path = Path(path).resolve()
    if not any(path.is_relative_to(p.resolve()) for p in (DATA, OUTPUT)):
        raise ValueError("research11 output must stay inside its isolated roots")
    if path.exists():
        raise ValueError(f"immutable research output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def archive_url(symbol, month):
    if symbol not in NEW_SYMBOLS or month not in MONTHS:
        raise ValueError("only declared XRP/DOGE historical study months are allowed")
    return f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/5m/{symbol}-5m-{month}.zip"


def _public_get(url, maximum):
    monthly = re.fullmatch(r"https://data\.binance\.vision/data/futures/um/monthly/klines/(XRPUSDT|DOGEUSDT)/5m/\1-5m-20\d{2}-\d{2}\.zip(?:\.CHECKSUM)?", url)
    repair = re.fullmatch(r"https://data\.binance\.vision/data/futures/um/daily/markPriceKlines/(XRPUSDT|DOGEUSDT)/1h/\1-1h-2026-06-29\.zip(?:\.CHECKSUM)?", url)
    if not (monthly or repair):
        raise ValueError("not an allowlisted public archive")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research11/1"}), timeout=30) as response:
                if response.geturl() != url:
                    raise ValueError("archive redirect refused")
                data = response.read(maximum + 1)
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            # Sequential retries only; a provider Retry-After is respected up to
            # one minute, beyond which this bounded acquisition stops for review.
            delay = 2 ** (attempt + 1)
            if isinstance(exc, urllib.error.HTTPError):
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    if not retry_after.isdigit() or int(retry_after) > 60:
                        raise ValueError("provider requested a longer retry pause; stop acquisition") from exc
                    delay = max(delay, int(retry_after))
            time.sleep(delay)
    if len(data) > maximum:
        raise ValueError("archive exceeds bounded research download")
    return data


def candle_audit(frame, timeframe="5m", month=None):
    """Strict observed grid/close check, with no repair, noise removal or filling."""
    step = pd.Timedelta(timeframe)
    audit = validate(frame, "open_time", timeframe)
    if audit["gaps"]:
        raise ValueError("observed candle gap; do not fill")
    model._validate_ohlc(frame, "research11", volume=True)
    opens = pd.DatetimeIndex(frame.open_time)
    closes = pd.to_datetime(frame.close_time, unit="ms", utc=True)
    if not np.array_equal(closes.astype("int64"), (opens + step - pd.Timedelta(milliseconds=1)).asi8):
        raise ValueError("candle close does not match the declared interval")
    if month:
        first = pd.Timestamp(month + "-01", tz="UTC")
        stop = first + pd.offsets.MonthBegin(1)
        if opens[0] != first or opens[-1] + step != stop or len(opens) != (stop-first)//step:
            raise ValueError("incomplete monthly candle coverage")
    return audit


def fetch_month(month):
    """Fetch exactly two 5m monthly ZIPs, each checksum-verified and immutable."""
    receipts = []
    for symbol in NEW_SYMBOLS:
        url = archive_url(symbol, month)
        path = DATA / "archives" / symbol / f"{symbol}-5m-{month}.zip"
        receipt_path = path.with_suffix(".receipt.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf8"))
            if model.sha256(path) != receipt["official_sha256"]:
                raise ValueError("cached research archive hash mismatch")
            receipts.append(receipt)
            continue
        if path.exists():
            raise ValueError("unreceipted archive exists; manual review required")
        check = _public_get(url + ".CHECKSUM", 2048).decode("ascii")
        expected = check.split()[0].lower()
        if not re.fullmatch("[0-9a-f]{64}", expected):
            raise ValueError("invalid official checksum")
        raw = _public_get(url, 8 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("official archive checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        # A bad schema is retained as unreceipted evidence and cannot be reused.
        audit = candle_audit(read_partition(path), month=month)
        receipt = {"symbol": symbol, "month": month, "url": url, "official_sha256": expected,
                   "path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": len(raw),
                   "retrieved_utc": model._now(), "audit": audit, "classification": "RESEARCH_ONLY_OBSERVED_ARCHIVE"}
        write_json(receipt_path, receipt)
        receipts.append(receipt)
    proposal = {"datasets": [{
        "id": f"{STUDY}_{r['symbol']}_5m_{month}", "source_path": str((ROOT/r["path"]).resolve()),
        "symbols": [r["symbol"]], "exchange": "binance", "market_type": "futures_usds_m_perpetual",
        "data_type": "official_archive_ohlcv", "timeframe": "5m", "frequency": "5m",
        "coverage": r["audit"], "verification_status": "VERIFIED_PERPETUAL_LABEL",
        "label_evidence": f"Official USD-M monthly klines URL {r['url']}; official checksum matched {r['official_sha256']}",
        "convertible": False, "default_selected": False,
        "notes": "Research sample only; no Freqtrade readiness gate effect; see adjacent immutable receipt.",
    } for r in receipts], "mark_research_references": []}
    proposal_path = DATA / f"registration_proposal_{month}.json"
    if not proposal_path.exists():
        write_json(proposal_path, proposal)
    return {"status": "SAMPLE_VERIFIED_REGISTRATION_REQUIRED", "archives": receipts,
            "registration_proposal": str(proposal_path), "training_ready": False}


def _mark_reference_file_metadata(path, expected_sha256):
    """Canonical registry metadata for an unchanged isolated research file."""
    path = Path(path).resolve()
    if not path.is_relative_to(DATA.resolve()) or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("mark reference must remain in isolated research data")
    if model.sha256(path) != expected_sha256:
        raise ValueError("mark reference content hash mismatch")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("mark reference is empty")
    return {"local_path": path.relative_to(ROOT.resolve()).as_posix(), "bytes": size}


def fetch_mark_repair():
    """Only the independently diagnosed 2026-06-29 mark gap; never alter sources."""
    proposal = {"datasets": [], "mark_research_references": []}
    for symbol in NEW_SYMBOLS:
        receipt_path = DATA/"supplemental"/f"{symbol}_mark_2026-06-29.receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        else:
            url = f"https://data.binance.vision/data/futures/um/daily/markPriceKlines/{symbol}/1h/{symbol}-1h-2026-06-29.zip"
            expected = _public_get(url+".CHECKSUM", 2048).decode("ascii").split()[0].lower()
            raw = _public_get(url, 1024*1024)
            if not re.fullmatch("[0-9a-f]{64}", expected) or hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("official daily mark checksum mismatch")
            source = receipt_path.with_suffix(".zip")
            if source.exists():
                raise ValueError("unreceipted daily mark archive exists")
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(raw)
            data = read_partition(source)
            candle_audit(data, "1h")
            required = pd.date_range("2026-06-29", periods=24, freq="1h", tz="UTC")
            if not pd.DatetimeIndex(data.open_time).equals(required):
                raise ValueError("official daily mark archive does not cover exactly the diagnosed day")
            normalized = data.rename(columns={"open_time": "date"})[["date", "open", "high", "low", "close", "volume"]]
            path = receipt_path.with_suffix(".parquet")
            if path.exists():
                raise ValueError("unreceipted daily mark normalization exists")
            normalized.to_parquet(path, index=False)
            receipt = {"symbol": symbol, "url": url, "official_sha256": expected,
                       "source_path": str(source.relative_to(ROOT)), "path": str(path.relative_to(ROOT)),
                       "sha256": model.sha256(path), "rows": 24, "retrieved_utc": model._now()}
            write_json(receipt_path, receipt)
        source, path = ROOT/receipt["source_path"], ROOT/receipt["path"]
        if model.sha256(source) != receipt["official_sha256"] or model.sha256(path) != receipt["sha256"]:
            raise ValueError("daily mark supplement hash mismatch")
        identity = f"{STUDY}_{symbol}_mark_daily_20260629"
        evidence = f"Official daily mark archive {receipt['url']}; ZIP matched .CHECKSUM {receipt['official_sha256']}; normalized SHA256 {receipt['sha256']}"
        proposal["datasets"].append({"id": identity, "source_path": str(path.resolve()), "symbols": [symbol],
            "exchange": "binance", "market_type": "futures_usds_m_perpetual", "data_type": "markPriceKlines",
            "timeframe": "1h", "frequency": "1h", "coverage": "2026-06-29 00:00 through 23:00 UTC; 24 actual observations",
            "verification_status": "VERIFIED_PERPETUAL_LABEL", "label_evidence": evidence,
            "convertible": False, "default_selected": False, "notes": "Independent observed supplement; original monthly source remains immutable."})
        proposal["mark_research_references"].append({"dataset_id": identity, "pair": f"{symbol[:-4]}/USDT:USDT",
            "timeframe": "1h", "candle_type": "mark", **_mark_reference_file_metadata(path, receipt["sha256"]), "extension": "parquet",
            "sha256": receipt["sha256"], "rows": 24, "first_utc": "2026-06-29T00:00:00+00:00",
            "last_utc": "2026-06-29T23:00:00+00:00", "verification_status": "VERIFIED_PERPETUAL_LABEL",
            "provenance_evidence": evidence, "installed": False, "research_only": True, "gate_effect": "NONE"})
    path = DATA/"mark_repair_registration_proposal.json"
    if not path.exists():
        write_json(path, proposal)
    return {"status": "OFFICIAL_MARK_SUPPLEMENT_VERIFIED_REGISTRATION_REQUIRED", "proposal": str(path), "observed_rows": 48}


def merge_mark_supplement(frame, symbol):
    """Merge actual missing timestamps only; conflicting overlaps stop research."""
    receipt = json.loads((DATA/"supplemental"/f"{symbol}_mark_2026-06-29.receipt.json").read_text(encoding="utf8"))
    path, source = (ROOT/receipt["path"]).resolve(), (ROOT/receipt["source_path"]).resolve()
    if (not path.is_relative_to(DATA.resolve()) or not source.is_relative_to(DATA.resolve())
            or model.sha256(path) != receipt["sha256"] or model.sha256(source) != receipt["official_sha256"]):
        raise ValueError("daily mark supplement provenance changed")
    observed = pd.read_parquet(path)
    original = frame.set_index("date")
    supplement = observed.set_index("date")
    if original.index.has_duplicates or supplement.index.has_duplicates:
        raise ValueError("duplicate mark observation timestamps")
    overlap = original.index.intersection(supplement.index)
    if len(overlap):
        pd.testing.assert_frame_equal(original.loc[overlap], supplement.loc[overlap], check_freq=False, check_exact=True)
    missing = supplement.index.difference(original.index)
    combined = pd.concat([original, supplement.loc[missing]]).sort_index().reset_index()
    audit = validate(combined, "date", "1h")
    if audit["gaps"]:
        raise ValueError("additional mark gaps remain after the observed supplement")
    return combined, {"receipt_path": str((DATA/"supplemental"/f"{symbol}_mark_2026-06-29.receipt.json").relative_to(ROOT)),
                      "added_rows": len(missing), "added_timestamps": [x.isoformat() for x in missing],
                      "supplement_sha256": receipt["sha256"]}


def inventory():
    missing = []
    for symbol in NEW_SYMBOLS:
        for month in MONTHS:
            p = DATA / "archives" / symbol / f"{symbol}-5m-{month}.zip"
            if not p.exists() or not p.with_suffix(".receipt.json").exists():
                missing.append(f"{symbol}/5m/{month}")
    return {"status": "BLOCKED_MISSING_5M_HISTORY" if missing else "ARCHIVES_PRESENT_NOT_YET_PREPARED",
            "required_5m_months": len(MONTHS)*2, "missing_5m_months": missing,
            "streams": {s: {"symbol": v[0], "bars": v[1], "candidate_validation_quantile": v[2],
                            "existing_model_readonly": s in model.STREAMS} for s,v in CANDIDATES.items()},
            "model_training_started": False, "limitations": LIMITATIONS}


def _noncore_receipts():
    records = {}
    with (NONCORE / "manifests/normalization_receipts.jsonl").open(encoding="utf8") as handle:
        for line in handle:
            if not any(symbol in line for symbol in NEW_SYMBOLS):
                continue
            row = json.loads(line)
            if row.get("symbol") in NEW_SYMBOLS and row.get("status") == "NORMALIZED":
                records[row["normalized_relative_path"].replace("\\", "/")] = row
    return records


def audit_auxiliary():
    """Check already acquired hourly/funding sources without writing or relabelling."""
    receipts = _noncore_receipts()
    downloads = {}
    with (NONCORE / "manifests/download_receipts.jsonl").open(encoding="utf8") as handle:
        for line in handle:
            if not any(symbol in line for symbol in NEW_SYMBOLS):
                continue
            row = json.loads(line)
            if row.get("symbol") in NEW_SYMBOLS and row.get("status") == "DOWNLOADED_VERIFIED":
                downloads[str(Path(row["path"]).resolve())] = row
    items = []
    for symbol in NEW_SYMBOLS:
        for kind, tf in (("klines", "1h"), ("markPriceKlines", "1h"), ("fundingRate", "")):
            for month in MONTHS:
                name = f"{symbol}-{tf or 'fundingRate'}-{month}.parquet"
                relative = "/".join(x for x in ("normalized", kind, symbol, tf, name) if x)
                row = receipts.get(relative)
                if not row:
                    raise ValueError(f"missing normalization provenance: {relative}")
                path = NONCORE / relative
                if not path.is_file() or model.sha256(path) != row["normalized_sha256"]:
                    raise ValueError(f"auxiliary partition hash mismatch: {relative}")
                source = NONCORE / row["source_relative_path"]
                if not source.is_file() or model.sha256(source) != row["source_sha256"]:
                    raise ValueError(f"auxiliary source archive hash mismatch: {relative}")
                receipt = downloads.get(str(source.resolve()))
                if (not receipt or receipt.get("official_sha256") != row["source_sha256"]
                        or not receipt.get("url", "").startswith(f"https://data.binance.vision/data/futures/um/monthly/{kind}/{symbol}/")):
                    raise ValueError(f"auxiliary official download evidence missing: {relative}")
                items.append({"symbol": symbol, "kind": kind, "timeframe": tf or "events", "month": month,
                              "path": str(path.relative_to(ROOT)), "sha256": row["normalized_sha256"],
                              "source_sha256": row["source_sha256"], "source_url": receipt["url"], "rows": row["rows"]})
    return {"status": "LOCAL_AUXILIARY_HASH_VERIFIED", "partitions": items,
            "note": "Existing acquisition provenance; check root registries before preparing new derived files"}


def prepare():
    """Create a separate exact-schema view only after the complete history exists.

    Shared core data are byte copies. Funding event timestamps are preserved.
    Registration proposals must be merged by the workspace owner before training.
    """
    inv = inventory()
    if inv["missing_5m_months"]:
        raise ValueError(f"training blocked: missing {len(inv['missing_5m_months'])} of 66 required 5m months")
    if RAW.exists():
        raise ValueError("prepared view already exists; never overwrite research observations")
    auxiliary = audit_auxiliary()
    source_manifest = json.loads((model.DATA_ROOT.parent / "data_manifest.json").read_text(encoding="utf8"))
    known = {str((ROOT / x["path"]).resolve()): x for x in source_manifest["datasets"]}
    copied = []
    for path in model.DATA_ROOT.rglob("*.parquet"):
        entry = known.get(str(path.resolve()))
        if not entry or model.sha256(path) != entry["sha256"]:
            raise ValueError("shared historical data manifest/hash mismatch")
        copied.append((path, path.relative_to(model.DATA_ROOT)))
    datasets = []
    for source, relative in copied:
        dest = RAW / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        datasets.append({"path": str(dest.relative_to(ROOT)), "sha256": model.sha256(dest),
                         "provenance_path": str(source), "shared_core_byte_copy": True})
    for symbol in NEW_SYMBOLS:
        frames = []
        for month in MONTHS:
            p = DATA / "archives" / symbol / f"{symbol}-5m-{month}.zip"
            receipt = json.loads(p.with_suffix(".receipt.json").read_text(encoding="utf8"))
            if model.sha256(p) != receipt["official_sha256"]:
                raise ValueError("5m archive hash mismatch")
            f = read_partition(p)
            candle_audit(f, month=month)
            frames.append(f)
        candles = pd.concat(frames, ignore_index=True)
        candle_audit(candles)
        candles = candles.set_index("open_time")
        path = RAW / "intraday" / f"{symbol}_5m_futures.parquet"
        candles.to_parquet(path)
        datasets.append({"path": str(path.relative_to(ROOT)), "sha256": model.sha256(path),
                         "symbol": symbol, "kind": "klines", "timeframe": "5m", "rows": len(candles)})
        for kind, suffix in (("klines", "1h_futures"), ("markPriceKlines", "1h_mark"), ("fundingRate", "8h_funding")):
            rows = [r for r in auxiliary["partitions"] if r["symbol"] == symbol and r["kind"] == kind]
            data = pd.concat([pd.read_parquet(ROOT/r["path"]) for r in rows], ignore_index=True)
            if kind == "fundingRate":
                intervals = pd.to_numeric(data.funding_interval_hours, errors="raise")
                if not (intervals == 8).all():
                    raise ValueError("funding event grid changed; the legacy 8h filename needs an explicit new mapping")
                data = pd.DataFrame({"date": pd.to_datetime(data.calc_time, unit="ms", utc=True),
                                     "funding_rate": data.last_funding_rate.astype(float)})
                # Events remain on their actual irregular/millisecond timestamp grid.
                audit = validate(data, "date")
            else:
                data = data.rename(columns={"open_time_utc": "date"})[["date", "open", "high", "low", "close", "volume"]]
                if kind == "markPriceKlines":
                    data, observed_supplement = merge_mark_supplement(data, symbol)
                audit = validate(data, "date", "1h")
                if audit["gaps"]:
                    raise ValueError("auxiliary hourly gap")
            path = RAW / "auxiliary" / f"{symbol}_{suffix}.parquet"
            data.to_parquet(path, index=False)
            datasets.append({"path": str(path.relative_to(ROOT)), "sha256": model.sha256(path),
                             "symbol": symbol, "kind": kind, "timeframe": "events" if kind == "fundingRate" else "1h",
                             "audit": audit, "source_partitions": rows,
                             "observed_supplement": observed_supplement if kind == "markPriceKlines" else None})
    manifest = {"status": "PREPARED_REGISTRATION_REQUIRED", "datasets": datasets,
                "classification": "RESEARCH11_ONLY", "created_utc": model._now(), "limitations": LIMITATIONS}
    write_json(DATA / "prepared_manifest.json", manifest)
    proposal = {"datasets": [], "mark_research_references": []}
    for entry in datasets:
        if entry.get("shared_core_byte_copy"):
            continue
        symbol, kind, tf = entry["symbol"], entry["kind"], entry["timeframe"]
        dataset_id = f"{STUDY}_{symbol}_{kind}_{tf}"
        evidence = f"Official Binance USD-M archives with checked download/normalization SHA256; research11 prepared_manifest.json; SHA256={entry['sha256']}"
        proposal["datasets"].append({"id": dataset_id, "source_path": str((ROOT/entry["path"]).resolve()),
            "symbols": [symbol], "exchange": "binance", "market_type": "funding_rate" if kind == "fundingRate" else "futures_usds_m_perpetual",
            "data_type": kind, "timeframe": tf, "frequency": tf, "coverage": "2024-01 through 2026-09; see manifest",
            "verification_status": "VERIFIED_PERPETUAL_LABEL", "label_evidence": evidence,
            "convertible": False, "default_selected": False, "notes": "Isolated exploratory research; no readiness gate effect."})
        if kind == "markPriceKlines":
            proposal["mark_research_references"].append({"dataset_id": dataset_id, "pair": f"{symbol[:-4]}/USDT:USDT",
                "timeframe": "1h", "candle_type": "mark", **_mark_reference_file_metadata(ROOT/entry["path"], entry["sha256"]), "extension": "parquet",
                "sha256": entry["sha256"], "rows": entry["audit"]["rows"], "first_utc": entry["audit"]["first_utc"],
                "last_utc": entry["audit"]["last_utc"], "verification_status": "VERIFIED_PERPETUAL_LABEL",
                "provenance_evidence": evidence, "installed": False, "research_only": True, "gate_effect": "NONE"})
    write_json(DATA / "prepared_registration_proposal.json", proposal)
    return manifest


def _registered_research_paths(registry):
    # Other legacy registry entries contain descriptive templates, not Windows
    # paths. Only this study's concrete source registrations can satisfy its gate.
    return {str(Path(r["source_path"]).resolve()) for r in registry["datasets"]
            if r.get("source_path") and r.get("id", "").startswith(STUDY+"_")}


def verify_prepared():
    path = DATA / "prepared_manifest.json"
    if not path.exists():
        raise ValueError("training blocked: prepared research11 manifest missing; run inventory/prepare")
    manifest = json.loads(path.read_text(encoding="utf8"))
    expected = {p.relative_to(model.DATA_ROOT).as_posix() for p in model.DATA_ROOT.rglob("*.parquet")}
    expected |= {f"intraday/{s}_5m_futures.parquet" for s in NEW_SYMBOLS}
    expected |= {f"auxiliary/{s}_{suffix}.parquet" for s in NEW_SYMBOLS for suffix in ("8h_funding", "1h_mark", "1h_futures")}
    actual = [(ROOT/e["path"]).resolve().relative_to(RAW.resolve()).as_posix() for e in manifest["datasets"]]
    if len(actual) != 26 or len(set(actual)) != 26 or set(actual) != expected:
        raise ValueError("complete isolated 26-source feature manifest required before training")
    registry = json.loads((ROOT/"archive/catalog/dataset_registry.json").read_text(encoding="utf8"))
    registered = _registered_research_paths(registry)
    marks = json.loads((ROOT/"archive/catalog/mark_candle_registry.json").read_text(encoding="utf8"))
    mark_hashes = {r.get("sha256") for r in marks.get("research_references", [])}
    for entry in manifest["datasets"]:
        p = (ROOT / entry["path"]).resolve()
        if not p.is_relative_to(RAW.resolve()) or model.sha256(p) != entry["sha256"]:
            raise ValueError("prepared data changed or escaped isolated root")
        if not entry.get("shared_core_byte_copy"):
            if str(p) not in registered:
                raise ValueError("new research source not registered")
            if entry["kind"] == "markPriceKlines" and entry["sha256"] not in mark_hashes:
                raise ValueError("research mark source/hash not registered")
    return manifest


def build_features(symbol):
    """Reuse all existing guards/formulas without mutating the production module.

    The isolated new-target adapter adds target files to the reader universe and
    compiled formula module, while retaining ONLY the original four-coin expert
    OFI context. Existing target functions are invoked completely unchanged.
    """
    if symbol in CORE:
        return model.build_feature_frames(RAW, symbol)
    if symbol not in NEW_SYMBOLS:
        raise ValueError("unknown research symbol")
    tree = ast.parse(inspect.getsource(model.build_feature_frames))
    replacement = ast.parse("ofi = sum((2 * frames[c].taker_buy_quote_volume - frames[c].quote_volume) / (frames[c].quote_volume + 1e-12) for c in RESEARCH_CORE) / 4").body[0]
    replaced = 0
    for i, node in enumerate(tree.body[0].body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ofi" for t in node.targets):
            tree.body[0].body[i] = replacement
            replaced += 1
    if replaced != 1:
        raise ValueError("production feature adapter structure changed; independent review required")
    def sources(reader, root):
        base, expert = model._source_modules(reader, root)
        base.CORE_SYMBOLS = CORE + (symbol,)
        return base, expert
    namespace = dict(vars(model))
    namespace.update(SYMBOLS=CORE+(symbol,), RESEARCH_CORE=CORE, _source_modules=sources)
    ast.fix_missing_locations(tree)
    exec(compile(tree, "research11_isolated_feature_adapter", "exec"), namespace)
    result = namespace["build_feature_frames"](RAW, symbol)
    result["research_mapping"] = {"target": symbol, "partner": "BTCUSDT", "expert_ofi_context": CORE,
                                  "cross_dispersion_members": [symbol, "BTCUSDT", "ETHUSDT", "BNBUSDT"]}
    return result


def payout_proxy(scores, labels, threshold, price=.515, fee_rate=.07):
    """Per-share hypothetical fully-filled payout, retaining ties/unsettled rows."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    if scores.shape != labels.shape or not np.isfinite(scores).all() or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("invalid score/threshold arrays")
    if not 0 < price < 1 or not 0 <= fee_rate <= 1:
        raise ValueError("invalid explicit fee scenario")
    selected = np.abs(scores) >= threshold
    settled = selected & np.isfinite(labels)
    wins = settled & ((scores > 0) == (labels >= 0))
    fee = fee_rate * price * (1-price)
    n, w = int(settled.sum()), int(wins.sum())
    return {"signals": int(selected.sum()), "settled_proxy_labels": n, "unsettled_proxy_labels": int((selected & ~np.isfinite(labels)).sum()),
            "ties_retained": int((settled & (labels == 0)).sum()), "wins": w, "raw_win_rate": w/n if n else None,
            "entry_price_assumption": price, "fee_rate_assumption": fee_rate, "fee_per_share": fee,
            "break_even_win_rate": price+fee, "hypothetical_pnl_one_share_per_signal": w-n*(price+fee),
            "payout_scope": "DIRECTION_PROXY_ALL_SIGNALS_ASSUMED_FILLED; NOT_ACCOUNT_RETURNS",
            "test_selection": "score only; no future return magnitude/tie/noise filter"}


def train(threads=2):
    if not 1 <= threads <= 3:
        raise ValueError("research CPU thread limit is 1..3")
    manifest = verify_prepared()  # Must fail before fitting or creating model output.
    if OUTPUT.exists():
        raise ValueError("immutable candidate run already exists")
    fold = dict(model.FOLDS["oos_2026"])
    result = {"status": "RESEARCH_COMPLETE_NOT_DEPLOYABLE", "started_utc": model._now(), "fold": fold,
              "test_status": "PREVIOUSLY_RESEARCHED_CONTAMINATED_HISTORICAL_COMPARISON", "streams": [],
              "data_manifest_sha256": model.sha256(DATA/"prepared_manifest.json"), "limitations": LIMITATIONS}
    started = time.monotonic()
    with threadpool_limits(limits=threads):
        for symbol in CORE + NEW_SYMBOLS:
            features = build_features(symbol)
            x, xd = features["base"], features["expert"]
            for stream, (coin, bars, quantile) in CANDIDATES.items():
                if coin != symbol:
                    continue
                labels = model._label(features["raw_target"], bars)
                tr, va, te = model._split(features["index"], labels, fold, bars)
                if not len(te):
                    raise ValueError("no chronological comparison window")
                if stream in model.STREAMS:
                    bundle = model.load_model_bundle(stream, "oos_2026")
                    if bundle["manifest"]["fold"] != fold:
                        raise ValueError("frozen model fold changed")
                    z = model.predict_scores(bundle, x.iloc[va], xd.iloc[va])
                    threshold = float(np.quantile(np.abs(z["BASE"]), quantile))
                    package = bundle["package"]
                    identity = {"existing_manifest_sha256": bundle["manifest_sha256"],
                                "existing_model_sha256": bundle["manifest"]["model_sha256"], "mode": "READ_ONLY_FROZEN_WEIGHTS"}
                else:
                    clip = .008 if bars == 1 else .015
                    y = np.clip(labels[tr], -clip, clip)
                    base, z0 = model._fit_family(x.iloc[tr], y, x.iloc[va])
                    expert, zd = model._fit_family(xd.iloc[tr], y, xd.iloc[va], expert=True)
                    weight, reset, grid = model.select_weight(z0, zd, labels[va], quantile)
                    threshold = float(np.quantile(np.abs(z0), quantile))
                    package = {"base": base, "expert": expert, "base_columns": list(x), "expert_columns": list(xd),
                               "weight": weight, "threshold_frozen": threshold, "threshold_reset": reset}
                    OUTPUT.mkdir(parents=True, exist_ok=True)
                    path = OUTPUT / f"{stream}.joblib"
                    joblib.dump(package, path, compress=3)
                    loaded = joblib.load(path)  # Only bytes generated locally in this call.
                    before = model.predict_scores({"package": package}, x.iloc[va[:64]], xd.iloc[va[:64]])
                    after = model.predict_scores({"package": loaded}, x.iloc[va[:64]], xd.iloc[va[:64]])
                    if any(not np.array_equal(before[k], after[k]) for k in before):
                        raise ValueError("new-model serialization score parity failed")
                    bundle = {"package": loaded}
                    identity = {"new_model_sha256": model.sha256(path), "mode": "NEW_TARGET_LOCAL_TRAINING",
                                "validation_weight_grid": grid, "saved_score_parity": "PASS", "mapping": features["research_mapping"]}
                scores = model.predict_scores(bundle, x.iloc[te], xd.iloc[te])["LOCAL_V4_FROZEN"]
                row = {"stream": stream, "candidate_validation_quantile": quantile, "candidate_threshold": threshold,
                       "threshold_basis": "ABS_BASE_SCORE_VALIDATION_ONLY; APPLIED_TO_FROZEN_BLEND", "weight": package["weight"],
                       "train_rows": len(tr), "validation_rows": len(va), "test_rows": len(te), **identity,
                       "candidate_proxy": payout_proxy(scores, labels[te], threshold)}
                if stream in model.STREAMS:
                    row["original_proxy"] = payout_proxy(scores, labels[te], package["threshold_frozen"])
                result["streams"].append(row)
    result.update(elapsed_seconds=time.monotonic()-started, completed_utc=model._now(),
                  feature_source_sha256=dict(model.SOURCE_HASHES), model_code_sha256=model.sha256(model.__file__))
    write_json(OUTPUT / "comparison.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "audit-auxiliary", "fetch-month", "fetch-mark-repair", "prepare", "train"))
    parser.add_argument("--month", default="2026-09")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        value = {"inventory": inventory, "audit-auxiliary": audit_auxiliary,
                 "fetch-mark-repair": fetch_mark_repair,
                 "fetch-month": lambda: fetch_month(args.month), "prepare": prepare,
                 "train": lambda: train(args.threads)}[args.action]()
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc), "production_changed": False}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
