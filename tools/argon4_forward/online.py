"""Public, GET-only online inputs; no account, signer or order endpoints.

Runtime data is separate from the frozen research corpus. Candle timestamps are
provider open times. ``first_received_at`` is local receipt time, never inferred
from a historical candle timestamp. The legacy ``8h_funding`` filename is only
a clean-room reader compatibility name: its contents remain observed events.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
BASIS_SYMBOLS = SYMBOLS[:3]
BINANCE_DOC = "https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data"
SPOT_DOC = "https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market"
DVOL_DOC = "https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data"
ENDPOINTS = {
    "clock": ("https://fapi.binance.com", "/fapi/v1/time"),
    "spot_clock": ("https://api.binance.com", "/api/v3/time"),
    "futures": ("https://fapi.binance.com", "/fapi/v1/klines"),
    "mark": ("https://fapi.binance.com", "/fapi/v1/markPriceKlines"),
    "funding": ("https://fapi.binance.com", "/fapi/v1/fundingRate"),
    "spot": ("https://api.binance.com", "/api/v3/klines"),
    "dvol": ("https://www.deribit.com", "/api/v2/public/get_volatility_index_data"),
}
STEPS = {"5m": pd.Timedelta(minutes=5), "1h": pd.Timedelta(hours=1),
         "4h": pd.Timedelta(hours=4), "1D": pd.Timedelta(days=1)}
PROVENANCE = ("first_received_at", "receipt_sha256", "receipt_time_basis", "clock_evidence_sha256")


def _now():
    return pd.Timestamp(datetime.now(timezone.utc))


def _utc(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("UTC-aware timestamp required")
    return ts.tz_convert("UTC")


def _iso(value):
    return _utc(value).isoformat()


def _ms(value):
    return int(_utc(value).value // 1_000_000)


def _atomic_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _parameters(kind, params):
    """Allow only the exact public parameter vocabulary; never forward auth."""
    if kind not in ENDPOINTS:
        raise ValueError("Endpoint is not a public input endpoint")
    allowed = ({"currency", "start_timestamp", "end_timestamp", "resolution"}
               if kind == "dvol" else set() if kind.endswith("clock")
               else {"symbol", "startTime", "endTime", "limit"} |
               ({"interval"} if kind != "funding" else set()))
    if set(params) != allowed:
        raise ValueError("Unexpected public input parameters")
    if not allowed:
        return
    if kind == "dvol":
        if params["currency"] not in ("BTC", "ETH") or params["resolution"] != "1D":
            raise ValueError("Unsupported DVOL request")
        start, end = params["start_timestamp"], params["end_timestamp"]
    else:
        if params["symbol"] not in SYMBOLS:
            raise ValueError("Unsupported symbol")
        if kind != "funding" and params["interval"] not in ("5m", "1h", "4h"):
            raise ValueError("Unsupported interval")
        if isinstance(params["limit"], bool) or not 1 <= params["limit"] <= 1000:
            raise ValueError("Unsupported page size")
        start, end = params["startTime"], params["endTime"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (start, end)) or end < start:
        raise ValueError("Invalid time range")


class PublicInputsHTTP:
    """Bounded public GETs and exact raw receipts; no environment credentials."""
    def __init__(self, receipts_path, *, clock=_now, min_spacing=1.0,
                 time_basis="SYSTEM_UTC", clock_evidence_sha256=None):
        if isinstance(min_spacing, bool) or not math.isfinite(min_spacing) or min_spacing < 0:
            raise ValueError("Invalid public request spacing")
        self.path = Path(receipts_path)
        self.clock = clock
        self.min_spacing = min_spacing
        self.time_basis = time_basis
        self.clock_evidence_sha256 = clock_evidence_sha256
        # One response, its receipt and any backoff must be fully handled before
        # another queued worker may check the rate limit and reach the network.
        # Keep this separate from the reentrant receipt/database lock below.
        self._request_lock = threading.Lock()
        self._lock = threading.RLock()
        self._last_request = 0.0
        self._session = None

    def _session_for_request(self):
        # The full request lock also serializes access to the connection pool.
        # Reuse TCP/TLS connections, never server cookies or ambient credentials.
        if self._session is None:
            self._session = requests.Session()
        self._session.trust_env = False
        self._session.auth = None
        proxies = getattr(self._session, "proxies", None)
        if proxies is not None:
            proxies.clear()
        cookies = getattr(self._session, "cookies", None)
        if cookies is not None:
            cookies.clear()
        return self._session

    def _close_session(self):
        session, self._session = self._session, None
        close = getattr(session, "close", None)
        if close:
            close()

    def close(self):
        """Release the connection pool after any active request has completed."""
        with self._request_lock:
            self._close_session()

    def _check_backoff(self, family):
        deadlines = []
        if self.path.exists():
            with sqlite3.connect(self.path, timeout=30) as conn:
                exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='backoff'").fetchone()
                row = conn.execute("SELECT until_utc FROM backoff WHERE family=?", (family,)).fetchone() if exists else None
            if row:
                deadlines.append(_utc(row[0]))
        shared = self.path.parent / "binance_backoff.json"
        if family == "binance" and shared.exists():
            deadlines.append(_utc(json.loads(shared.read_text(encoding="utf-8"))["blocked_until_utc"]))
        if deadlines and _utc(self.clock()) < max(deadlines):
            raise ValueError("PUBLIC_RATE_LIMIT_BACKOFF_UNTIL_" + _iso(max(deadlines)))

    def _record_backoff(self, family, raw, retry_after, status):
        # Follow the provider's explicit ban time; never rotate hosts/proxies.
        now = _utc(self.clock())
        until = now + pd.Timedelta(seconds=3600 if status == 418 else 60)
        match = re.search(rb"banned until (\d{13})", raw)
        if match:
            until = pd.to_datetime(int(match.group(1)), unit="ms", utc=True)
        elif retry_after:
            try:
                until = now + pd.Timedelta(seconds=float(retry_after))
            except (TypeError, ValueError):
                try:
                    until = _utc(retry_after)
                except (TypeError, ValueError):
                    pass
        # A small margin also covers an injected calibrated UTC upper bound.
        until += pd.Timedelta(seconds=15)
        response_until = until
        with self._lock, sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS backoff(family TEXT PRIMARY KEY, until_utc TEXT NOT NULL)")
            old = conn.execute("SELECT until_utc FROM backoff WHERE family=?", (family,)).fetchone()
            if old:
                until = max(until, _utc(old[0]))
            conn.execute("INSERT OR REPLACE INTO backoff(family,until_utc) VALUES(?,?)", (family, _iso(until)))
        if family == "binance":
            target = self.path.parent / "binance_backoff.json"
            temporary = target.with_suffix(".json.tmp")
            with self._lock:
                previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
                if previous.get("blocked_until_utc"):
                    until = max(until, _utc(previous["blocked_until_utc"]))
                evidence = ({"reason": previous["reason"],
                             "source_receipt_sha256": previous["source_receipt_sha256"]}
                            if previous.get("blocked_until_utc") and
                            _utc(previous["blocked_until_utc"]) > response_until else
                            {"reason": "PUBLIC_HTTP_" + str(status),
                             "source_receipt_sha256": hashlib.sha256(raw).hexdigest()})
                temporary.write_text(json.dumps({"blocked_until_utc": _iso(until),
                    **evidence,
                    "scope": "ALL_BINANCE_HOSTS", "safety_margin_seconds": 15}, indent=2), encoding="utf-8")
                temporary.replace(target)

    def _save(self, receipt, raw):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS receipts(id INTEGER PRIMARY KEY, "
                         "url TEXT NOT NULL, requested_at TEXT NOT NULL, received_at TEXT NOT NULL, "
                         "sha256 TEXT NOT NULL, http_status INTEGER NOT NULL, raw BLOB NOT NULL)")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(receipts)")}
            for name in ("requested_at_local", "received_at_local", "time_basis", "clock_evidence_sha256"):
                if name not in columns:
                    conn.execute("ALTER TABLE receipts ADD COLUMN " + name + " TEXT")
            conn.execute("INSERT INTO receipts(url,requested_at,received_at,sha256,http_status,raw,"
                         "requested_at_local,received_at_local,time_basis,clock_evidence_sha256) VALUES(?,?,?,?,?,?,?,?,?,?)",
                         (receipt["url"], receipt["requested_at"], receipt["received_at"],
                          receipt["sha256"], receipt["http_status"], raw,
                          receipt["requested_at_local"], receipt["received_at_local"],
                          receipt["time_basis"], receipt["clock_evidence_sha256"]))

    def get(self, kind, params=None):
        with self._request_lock:
            try:
                return self._get_serial(kind, params)
            except Exception:
                # No transparent retry: retain the original failure and let the
                # next authorized request establish a clean connection pool.
                self._close_session()
                raise

    def _get_serial(self, kind, params=None):
        params = params or {}
        _parameters(kind, params)
        base, path = ENDPOINTS[kind]
        family = "deribit" if kind == "dvol" else "binance"
        url = base + path + ("?" + urlencode(params) if params else "")
        with self._lock:
            self._check_backoff(family)
            delay = self.min_spacing - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._check_backoff(family)
            self._last_request = time.monotonic()
        requested_at_local = _iso(_now())
        requested_at = _iso(self.clock())
        session = self._session_for_request()
        try:
            with session.get(url, timeout=(4, 12), allow_redirects=False, stream=True,
                             headers={"User-Agent": "argon4-public-paper-inputs/1"}) as response:
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > 4_000_000:
                        raise ValueError("Public input response exceeds size limit")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                received_at_local = _iso(_now())
                received_at = _iso(self.clock())
                receipt = {"url": url, "requested_at": requested_at, "received_at": received_at,
                           "sha256": hashlib.sha256(raw).hexdigest(), "http_status": response.status_code,
                           "requested_at_local": requested_at_local, "received_at_local": received_at_local,
                           "time_basis": self.time_basis, "clock_evidence_sha256": self.clock_evidence_sha256}
                retry_after = getattr(response, "headers", {}).get("Retry-After")
        finally:
            cookies = getattr(session, "cookies", None)
            if cookies is not None:
                cookies.clear()
        self._save(receipt, raw)
        if receipt["http_status"] in (418, 429):
            self._record_backoff(family, raw, retry_after, receipt["http_status"])
        if receipt["http_status"] != 200:
            raise ValueError("PUBLIC_HTTP_STATUS_" + str(receipt["http_status"]))
        return json.loads(raw), receipt


def _validate_prices(frame):
    fields = [c for c in ("open", "high", "low", "close", "volume", "quote_volume",
                          "count", "taker_buy_base_volume", "taker_buy_quote_volume") if c in frame]
    if frame.empty or not np.isfinite(frame[fields].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite prices/flow")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Nonpositive prices")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
            (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC ordering")
    for col in set(fields) - {"open", "high", "low", "close"}:
        if (frame[col] < 0).any():
            raise ValueError("Negative volume/count")
    if "count" in frame and not (frame["count"] % 1 == 0).all():
        raise ValueError("Noninteger trade count")


def _merge(old, new, keys):
    """Keep the first real receipt, rejecting revised/conflicting closed data."""
    if old.empty:
        combined = new.copy()
    else:
        combined = pd.concat([old, new], ignore_index=True)
    if combined.empty:
        return combined
    value_cols = [c for c in combined if c not in PROVENANCE]
    duplicates = combined[combined.duplicated(keys, keep=False)]
    for _, group in duplicates.groupby(keys, dropna=False):
        if len(group[value_cols].drop_duplicates()) != 1:
            raise ValueError("CONFLICTING_CLOSED_OBSERVATION")
    return combined.sort_values("first_received_at").drop_duplicates(keys, keep="first").sort_values(keys).reset_index(drop=True)


class OnlineInputs:
    """Use the account runtime directory; outputs live in ``features/raw_data``.

    ``refresh`` returns a snapshot at completion, plus the requested candle
    cutoff. An explicit historical ``snapshot(as_of)`` only exposes metadata for
    rows actually received by that time; readers must apply the same row filter
    to ``data_root`` (all frames include ``first_received_at``). No historical
    observations are claimed to have been available before first receipt.
    """
    def __init__(self, runtime_root, *, window_days=20, client=None, clock=_now,
                 max_clock_skew_seconds=2.0, time_basis="SYSTEM_UTC", clock_evidence_sha256=None,
                 http_min_spacing_seconds=1.0):
        if not 15 <= window_days <= 60:
            raise ValueError("Online warmup requires 15 to 60 days")
        if (isinstance(http_min_spacing_seconds, bool) or not math.isfinite(http_min_spacing_seconds)
                or not .25 <= http_min_spacing_seconds <= 60):
            raise ValueError("Online HTTP spacing must be finite and within 0.25 to 60 seconds")
        self.root = Path(runtime_root) / "features"
        self.data_root = self.root / "raw_data"
        self.state_path = self.root / "online_state.json"
        self.clock = clock
        self.window_days = window_days
        self.max_clock_skew_seconds = max_clock_skew_seconds
        if time_basis not in ("SYSTEM_UTC", "CALIBRATED_UTC_UPPER_BOUND"):
            raise ValueError("Unknown time basis")
        if time_basis != "SYSTEM_UTC" and (not isinstance(clock_evidence_sha256, str) or
                                          re.fullmatch(r"[a-f0-9]{64}", clock_evidence_sha256) is None):
            raise ValueError("Calibrated clock requires evidence SHA256")
        self.time_basis = time_basis
        self.clock_evidence_sha256 = clock_evidence_sha256
        self.client = client or PublicInputsHTTP(self.root / "http_receipts.sqlite3", clock=clock,
            time_basis=time_basis, clock_evidence_sha256=clock_evidence_sha256,
            min_spacing=http_min_spacing_seconds)
        self._lock = threading.RLock()
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self._checks = dict(self.state.get("checks", {}))

    def close(self):
        """Close the owned/injected public client after refresh leaves its lock."""
        with self._lock:
            close = getattr(self.client, "close", None)
            if close:
                close()

    def _path(self, kind, symbol=None, interval=None):
        if kind == "dvol":
            return self.data_root / "auxiliary/deribit_dvol_daily_2021_2026.parquet"
        if kind == "basis":
            return self.data_root / "auxiliary/binance_basis_4h.parquet"
        if kind == "funding":
            name = f"{symbol}_8h_funding.parquet"
        else:
            name = f"{symbol}_{interval}_{kind}.parquet"
        return self.data_root / ("intraday" if interval == "5m" else "auxiliary") / name

    def _read(self, path):
        frame = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        if not frame.empty:
            if "receipt_time_basis" not in frame:
                frame["receipt_time_basis"] = "SYSTEM_UTC"
            if "clock_evidence_sha256" not in frame:
                frame["clock_evidence_sha256"] = None
        return frame

    def _receipt_fields(self, receipt):
        return {"first_received_at": _utc(receipt["received_at"]), "receipt_sha256": receipt["sha256"],
                "receipt_time_basis": receipt.get("time_basis", self.time_basis),
                "clock_evidence_sha256": receipt.get("clock_evidence_sha256", self.clock_evidence_sha256)}

    def _checked(self, name, receipt):
        self._checks[name] = {"received_at": receipt["received_at"], "source": receipt["url"],
                              "sha256": receipt["sha256"]}

    def _start(self, old, key, cutoff, step):
        initial = cutoff.floor("D") - pd.Timedelta(days=self.window_days + 1)
        if old.empty:
            return initial
        dates = pd.DatetimeIndex(old[key]).sort_values().unique()
        gaps = np.flatnonzero(np.diff(dates.asi8) != step.value)
        start = dates[gaps[0]] if len(gaps) else dates[-1] - 2 * step
        return max(initial, start)

    def _closed_cache_complete(self, frame, key, step, expected_last_open, min_rows=1):
        if frame.empty or len(frame) < min_rows:
            return False
        dates = pd.DatetimeIndex(frame[key]).sort_values()
        if dates[-1] != expected_last_open or (np.diff(dates.asi8) != step.value).any():
            return False
        _validate_prices(frame)
        return True

    def _candles(self, kind, symbol, interval, cutoff):
        path = self._path(kind, symbol, interval)
        old = self._read(path)
        step = STEPS[interval]
        if interval != "5m" and self._closed_cache_complete(
                old, "date", step, cutoff.floor(step)-step,
                min_rows=int(pd.Timedelta(days=15)/step)):
            # The closed data is already held. This creates no new receipt and
            # preserves its first-received time; freshness is candle based.
            return str(path)
        start = self._start(old, "date", cutoff, step)
        end = cutoff.floor(step) - pd.Timedelta(milliseconds=1)
        rows = []
        while start <= end:
            try:
                payload, receipt = self.client.get(kind, {"symbol": symbol, "interval": interval,
                    "startTime": _ms(start), "endTime": _ms(end), "limit": 1000})
            except Exception:
                # Preserve already received, validated pages when a later page
                # fails. Snapshot still blocks on missing coverage/errors.
                if rows:
                    partial = pd.DataFrame(rows)
                    _validate_prices(partial)
                    _atomic_frame(_merge(old, partial, ["date"]), path)
                raise
            self._checked(f"{symbol}:{kind}:{interval}", receipt)
            if not isinstance(payload, list):
                raise ValueError("Invalid kline response")
            if not payload:
                break
            next_start = start
            for raw in payload:
                if not isinstance(raw, list) or len(raw) < 11:
                    raise ValueError("Invalid kline row")
                opened = pd.to_datetime(raw[0], unit="ms", utc=True)
                closed = opened + step
                if opened < start or opened > end or opened.value % step.value:
                    raise ValueError("Kline timestamp outside requested grid")
                if int(raw[6]) != _ms(closed) - 1:
                    raise ValueError("Provider close timestamp mismatch")
                next_start = max(next_start, closed)
                if closed > cutoff or closed > _utc(receipt["received_at"]):
                    continue
                rows.append({"date": opened, "open_time": opened,
                    **dict(zip(("open", "high", "low", "close", "volume"), map(float, raw[1:6]))),
                    "close_time": pd.to_datetime(raw[6], unit="ms", utc=True),
                    "quote_volume": float(raw[7]), "count": float(raw[8]),
                    "taker_buy_base_volume": float(raw[9]), "taker_buy_quote_volume": float(raw[10]),
                    **self._receipt_fields(receipt)})
            if next_start <= start:
                raise ValueError("Kline pagination stalled")
            start = next_start
            if len(payload) < 1000:
                break
        new = pd.DataFrame(rows)
        if not new.empty:
            _validate_prices(new)
        merged = _merge(old, new, ["date"])
        if merged.empty:
            raise ValueError("EMPTY_CANDLES")
        _atomic_frame(merged, path)
        return str(path)

    def _funding(self, symbol, cutoff):
        path = self._path("funding", symbol)
        old = self._read(path)
        start = (cutoff.floor("D") - pd.Timedelta(days=self.window_days + 3) if old.empty
                 else pd.Timestamp(old.date.max()) - pd.Timedelta(days=2))
        rows = []
        while start <= cutoff:
            payload, receipt = self.client.get("funding", {"symbol": symbol, "startTime": _ms(start),
                                                "endTime": _ms(cutoff), "limit": 1000})
            self._checked(f"{symbol}:funding", receipt)
            if not isinstance(payload, list):
                raise ValueError("Invalid funding response")
            if not payload:
                break
            dates = []
            for raw in payload:
                ts = pd.to_datetime(raw["fundingTime"], unit="ms", utc=True)
                rate = float(raw["fundingRate"])
                if raw["symbol"] != symbol or not math.isfinite(rate) or not start <= ts <= cutoff:
                    raise ValueError("Invalid funding event")
                dates.append(ts)
                if ts <= _utc(receipt["received_at"]):
                    rows.append({"date": ts, "funding_rate": rate, **self._receipt_fields(receipt)})
            following = max(dates) + pd.Timedelta(milliseconds=1)
            if following <= start:
                raise ValueError("Funding pagination stalled")
            start = following
            if len(payload) < 1000:
                break
        merged = _merge(old, pd.DataFrame(rows), ["date"])
        if merged.empty:
            raise ValueError("EMPTY_FUNDING_EVENTS")
        _atomic_frame(merged, path)
        return str(path)

    def _dvol_one(self, symbol, cutoff, old):
        if self._closed_cache_complete(old, "bucket_utc", STEPS["1D"],
                (cutoff-pd.Timedelta(minutes=5)).floor("D")-STEPS["1D"], min_rows=10):
            return old
        end = cutoff - pd.Timedelta(minutes=5)
        start = (cutoff.floor("D") - pd.Timedelta(days=self.window_days + 1) if old.empty
                 else pd.Timestamp(old.bucket_utc.max()) - pd.Timedelta(days=2))
        rows = []
        while end >= start:
            payload, receipt = self.client.get("dvol", {"currency": symbol, "resolution": "1D",
                "start_timestamp": _ms(start), "end_timestamp": _ms(end)})
            self._checked(f"{symbol}:dvol", receipt)
            if "error" in payload or not isinstance(payload.get("result", {}).get("data"), list):
                raise ValueError("Invalid DVOL response")
            data = payload["result"]["data"]
            if not data:
                break
            dates = []
            for raw in data:
                ts = pd.to_datetime(raw[0], unit="ms", utc=True)
                if len(raw) != 5 or ts.value % STEPS["1D"].value or not start <= ts <= end:
                    raise ValueError("Invalid DVOL day")
                dates.append(ts)
                if ts + STEPS["1D"] + pd.Timedelta(minutes=5) <= cutoff and ts + STEPS["1D"] <= _utc(receipt["received_at"]):
                    rows.append({"bucket_utc": ts, "symbol": symbol,
                        **dict(zip(("open", "high", "low", "close"), map(float, raw[1:]))),
                        **self._receipt_fields(receipt)})
            next_end = min(dates) - pd.Timedelta(milliseconds=1)
            if next_end >= end:
                raise ValueError("DVOL pagination stalled")
            end = next_end
        new = pd.DataFrame(rows)
        if not new.empty:
            _validate_prices(new)
        merged = _merge(old, new, ["symbol", "bucket_utc"])
        if merged.empty:
            raise ValueError("EMPTY_DVOL")
        return merged

    def _basis(self):
        parts, receipts, bases, calibrations = [], [], [], []
        for symbol in BASIS_SYMBOLS:
            futures = self._read(self._path("futures", symbol, "4h")).set_index("date")
            spot = self._read(self._path("spot", symbol, "4h")).set_index("date")
            if not futures.index.equals(spot.index):
                raise ValueError("BASIS_LEG_COVERAGE_MISMATCH")
            parts.append(((futures.close - spot.close) / spot.close).rename(symbol))
            receipts.extend([futures.first_received_at, spot.first_received_at])
            bases.extend([futures.receipt_time_basis, spot.receipt_time_basis])
            calibrations.extend([futures.clock_evidence_sha256, spot.clock_evidence_sha256])
        frame = pd.concat(parts, axis=1)
        if frame.isna().any().any():
            raise ValueError("BASIS_SYMBOL_COVERAGE_MISMATCH")
        frame["first_received_at"] = pd.concat(receipts, axis=1).max(axis=1)
        frame["receipt_time_basis"] = pd.concat(bases, axis=1).apply(
            lambda row: row.iloc[0] if row.nunique() == 1 else "MIXED_RECEIPT_CLOCKS", axis=1)
        frame["clock_evidence_sha256"] = pd.concat(calibrations, axis=1).apply(
            lambda row: row.iloc[0] if row.notna().all() and row.nunique() == 1 else None, axis=1)
        frame.index.name = "date"
        # The legacy basis reader requires an open-time DatetimeIndex.
        path = self._path("basis")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary)
        temporary.replace(path)

    def _check_clock(self):
        errors = []
        for kind in ("clock", "spot_clock"):
            try:
                data, receipt = self.client.get(kind)
                server = pd.to_datetime(int(data["serverTime"]), unit="ms", utc=True)
                requested, received = _utc(receipt["requested_at"]), _utc(receipt["received_at"])
                rtt = (received - requested).total_seconds()
                skew = (server - (requested + (received - requested) / 2)).total_seconds()
                valid = rtt >= 0 and rtt <= 4 and abs(skew) <= self.max_clock_skew_seconds
                return {"status": "VERIFIED" if valid else "CLOCK_SKEW_OR_LATENCY", "source": receipt["url"],
                        "server_time": _iso(server), "received_at": receipt["received_at"],
                        "skew_seconds": skew, "round_trip_seconds": rtt,
                        "max_skew_seconds": self.max_clock_skew_seconds, "fallback_errors": errors}
            except Exception as exc:
                errors.append(kind + ":" + type(exc).__name__)
        return {"status": "UNAVAILABLE", "errors": errors, "skew_seconds": None}

    def refresh(self, now=None):
        with self._lock:
            cutoff = _utc(now) if now is not None else _utc(self.clock())
            if cutoff > _utc(self.clock()) + pd.Timedelta(seconds=2):
                raise ValueError("Future refresh cutoff")
            errors = []
            # Create shared parent directories before parallel publishers on
            # Windows; workers only replace their own distinct data files.
            for directory in (self.data_root / "intraday", self.data_root / "auxiliary"):
                directory.mkdir(parents=True, exist_ok=True)
            clock = self._check_clock()
            jobs = [(self._candles, ("futures", s, "5m", cutoff)) for s in SYMBOLS]
            jobs += [(self._candles, (k, s, "1h", cutoff)) for s in SYMBOLS for k in ("futures", "mark")]
            jobs += [(self._candles, (k, s, "4h", cutoff)) for s in BASIS_SYMBOLS for k in ("futures", "spot")]
            jobs += [(self._funding, (s, cutoff)) for s in SYMBOLS]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [(args, pool.submit(fn, *args)) for fn, args in jobs]
                for args, future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append({"source": ":".join(str(a) for a in args[:-1]),
                                       "error": type(exc).__name__ + ":" + str(exc)[:160]})
            try:
                old = self._read(self._path("dvol"))
                parts = [self._dvol_one(s, cutoff, old[old.symbol == s] if not old.empty else old) for s in ("BTC", "ETH")]
                _atomic_frame(pd.concat(parts, ignore_index=True), self._path("dvol"))
            except Exception as exc:
                errors.append({"source": "dvol", "error": type(exc).__name__ + ":" + str(exc)[:160]})
            try:
                self._basis()
            except Exception as exc:
                errors.append({"source": "basis", "error": type(exc).__name__ + ":" + str(exc)[:160]})
            self.state = {"schema_version": 1, "requested_cutoff": _iso(cutoff),
                          "refresh_completed_at": _iso(self.clock()), "clock": clock, "errors": errors,
                          "time_basis": self.time_basis, "clock_evidence_sha256": self.clock_evidence_sha256,
                          "checks": self._checks}
            self.root.mkdir(parents=True, exist_ok=True)
            temp = self.state_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(self.state_path)
            return self.snapshot()

    def _source_specs(self):
        for symbol in SYMBOLS:
            for kind, interval in (("futures", "5m"), ("futures", "1h"), ("mark", "1h")):
                yield f"{symbol}:{kind}:{interval}", self._path(kind, symbol, interval), "date", STEPS[interval], None
            yield f"{symbol}:funding", self._path("funding", symbol), "date", None, None
        for symbol in BASIS_SYMBOLS:
            for kind in ("futures", "spot"):
                yield f"{symbol}:{kind}:4h", self._path(kind, symbol, "4h"), "date", STEPS["4h"], None
        yield "basis:4h", self._path("basis"), "date", STEPS["4h"], None
        for symbol in ("BTC", "ETH"):
            yield f"{symbol}:dvol", self._path("dvol"), "bucket_utc", STEPS["1D"], symbol

    def snapshot(self, as_of=None):
        with self._lock:
            now = _utc(as_of) if as_of is not None else _utc(self.clock())
            if now > _utc(self.clock()) + pd.Timedelta(seconds=2):
                raise ValueError("Future snapshot requested")
            sources, gaps, latest5, available = {}, [], [], []
            for name, path, key, step, symbol in self._source_specs():
                info = {"path": str(path), "status": "UNAVAILABLE", "rows": 0,
                        "source_doc": DVOL_DOC if symbol else SPOT_DOC if ":spot:" in name else BINANCE_DOC}
                sources[name] = info
                try:
                    frame = self._read(path)
                    if frame.empty:
                        raise ValueError("SOURCE_ABSENT")
                    if key not in frame and frame.index.name == key:
                        frame = frame.reset_index()
                    if symbol:
                        frame = frame[frame.symbol == symbol]
                    extra = pd.Timedelta(minutes=5) if symbol else pd.Timedelta(0)
                    closed = frame[key] + (step or pd.Timedelta(0)) + extra
                    frame = frame[(closed <= now) & (frame.first_received_at <= now)].sort_values(key)
                    if frame.empty:
                        raise ValueError("NOT_YET_RECEIVED_OR_CLOSED")
                    if "close" in frame:
                        _validate_prices(frame)
                    elif "funding_rate" in frame and not np.isfinite(frame.funding_rate.to_numpy(dtype=float)).all():
                        raise ValueError("NONFINITE_FUNDING_EVENT")
                    elif name == "basis:4h" and not np.isfinite(frame[list(BASIS_SYMBOLS)].to_numpy(dtype=float)).all():
                        raise ValueError("NONFINITE_BASIS")
                    dates = pd.DatetimeIndex(frame[key])
                    first_received, last_received = frame.first_received_at.min(), frame.first_received_at.max()
                    latest = dates[-1] + (step or pd.Timedelta(0))
                    info.update(rows=len(frame), first_open=_iso(dates[0]), latest_closed=_iso(latest),
                                first_received_at=_iso(first_received), latest_received_at=_iso(last_received),
                                age_seconds=(now - latest).total_seconds(),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    info["receipt_time_bases"] = sorted(frame.receipt_time_basis.dropna().unique().tolist())
                    check = self.state.get("checks", {}).get(name)
                    if check and _utc(check["received_at"]) <= now:
                        info.update(latest_receipt_at=check["received_at"], latest_receipt_sha256=check["sha256"],
                                    last_checked_source=check["source"])
                    available.append(last_received)
                    issues = []
                    if step is None:
                        intervals = np.diff(dates.asi8) / 1e9
                        if len(intervals) < 2 or (intervals <= 0).any():
                            issues.append("INSUFFICIENT_FUNDING_INTERVAL_EVIDENCE")
                        else:
                            maximum = float(intervals.max())
                            info.update(event_series=True, resampled=False, compatibility_filename_only=True,
                                        max_observed_interval_seconds=maximum,
                                        observed_interval_seconds=sorted(set(intervals.tolist())),
                                        freshness_limit_seconds=maximum + 300,
                                        completeness="PROVIDER_RETURNED_EVENTS_NOT_GRID_CERTIFIED")
                            if (now - dates[-1]).total_seconds() > maximum + 300:
                                issues.append("FUNDING_STALE_VS_MAX_OBSERVED_INTERVAL")
                    else:
                        bad = np.flatnonzero(np.diff(dates.asi8) != step.value)
                        if len(bad):
                            issues.append("CANDLE_GAPS")
                            info["gaps"] = [{"after": _iso(dates[i]), "before": _iso(dates[i+1])} for i in bad[:30]]
                        expected = (now - extra).floor(step)
                        if latest != expected:
                            issues.append("LATEST_CLOSED_BAR_MISSING")
                        if step == STEPS["5m"]:
                            latest5.append(latest)
                            if len(frame) < 4320:
                                issues.append("WARMUP_UNDER_4320_5M_BARS")
                        if symbol and len(frame) < 10:
                            issues.append("WARMUP_UNDER_10_DVOL_DAYS")
                    info["status"] = "READY" if not issues else "BLOCKED"
                    info["issues"] = issues
                    gaps.extend({"source": name, "error": issue} for issue in issues)
                except Exception as exc:
                    info["error"] = type(exc).__name__ + ":" + str(exc)[:160]
                    gaps.append({"source": name, "error": info["error"]})
            errors = list(self.state.get("errors", []))
            clock = dict(self.state.get("clock", {"status": "UNAVAILABLE"}))
            if clock.get("received_at"):
                age = (now - _utc(clock["received_at"])).total_seconds()
                clock["age_seconds"] = age
                if not 0 <= age <= 120:
                    clock["status"] = "STALE_OR_NOT_YET_RECEIVED"
            if clock["status"] != "VERIFIED":
                errors.append({"source": "clock", "error": clock["status"]})
            if self.state.get("refresh_completed_at") and _utc(self.state["refresh_completed_at"]) > now:
                errors.append({"source": "refresh", "error": "NOT_COMPLETE_BY_AS_OF"})
            return {"status": "READY" if not errors and not gaps else "BLOCKED", "mode": "PUBLIC_INPUTS_ONLY",
                    "data_root": str(self.data_root), "as_of": _iso(now), "local_now": _iso(self.clock()),
                    "system_local_now": _iso(_now()), "time_basis": self.time_basis,
                    "clock_evidence_sha256": self.clock_evidence_sha256,
                    "requested_cutoff": self.state.get("requested_cutoff"),
                    "refresh_completed_at": self.state.get("refresh_completed_at"),
                    "latest_closed_5m": _iso(min(latest5)) if len(latest5) == 4 else None,
                    "feature_available_at": _iso(max(available)) if available else None,
                    "clock": clock, "sources": sources, "gaps": gaps, "errors": errors}

    def mark_registry_evidence(self):
        """Evidence only. The caller registers these files before model use."""
        result = []
        for symbol in SYMBOLS:
            path = self._path("mark", symbol, "1h")
            frame = self._read(path)
            if frame.empty:
                continue
            result.append({"symbol": symbol, "path": str(path.resolve()), "timeframe": "1h",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(frame),
                "first_open": _iso(frame.date.min()), "last_close": _iso(frame.date.max() + STEPS["1h"]),
                "source_endpoint": "".join(ENDPOINTS["mark"]), "source_doc": BINANCE_DOC,
                "first_received_at": _iso(frame.first_received_at.min()),
                "last_received_at": _iso(frame.first_received_at.max()),
                "receipt_sha256": sorted(frame.receipt_sha256.unique().tolist()),
                "receipts_path": str((self.root / "http_receipts.sqlite3").resolve()),
                "purpose": "ONLINE_RESEARCH_FEATURE_ONLY", "converted_for_freqtrade": False})
        return result
