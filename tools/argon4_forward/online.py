"""Public, GET-only online inputs; no account, signer or order endpoints.

Runtime data is separate from the frozen research corpus. Candle timestamps are
provider open times. ``first_received_at`` is local receipt time, never inferred
from a historical candle timestamp. The legacy ``8h_funding`` filename is only
a clean-room reader compatibility name: its contents remain observed events.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
ALL_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
BASIS_SYMBOLS = SYMBOLS[:3]
BINANCE_DOC = "https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data"
SPOT_DOC = "https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market"
DVOL_DOC = "https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data"
ENDPOINTS = {
    "clock": ("https://fapi.binance.com", "/fapi/v1/time"),
    "spot_clock": ("https://data-api.binance.vision", "/api/v3/time"),
    "futures": ("https://fapi.binance.com", "/fapi/v1/klines"),
    "mark": ("https://fapi.binance.com", "/fapi/v1/markPriceKlines"),
    "funding": ("https://fapi.binance.com", "/fapi/v1/fundingRate"),
    "spot": ("https://data-api.binance.vision", "/api/v3/klines"),
    "dvol": ("https://www.deribit.com", "/api/v2/public/get_volatility_index_data"),
    "book_ticker": ("https://fapi.binance.com", "/fapi/v1/ticker/bookTicker"),
}
STEPS = {"5m": pd.Timedelta(minutes=5), "1h": pd.Timedelta(hours=1),
         "4h": pd.Timedelta(hours=4), "1D": pd.Timedelta(days=1)}
PROVENANCE = ("first_received_at", "receipt_sha256", "receipt_time_basis", "clock_evidence_sha256")


class InputDeadlineExceeded(TimeoutError):
    """A retired input generation must never publish a working view."""


def _remaining(deadline):
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise InputDeadlineExceeded("INPUT_DEADLINE_EXCEEDED")
    return remaining


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
    if kind == "book_ticker":
        if set(params) not in (set(), {"symbol"}):
            raise ValueError("Unexpected public input parameters")
        if "symbol" in params and params["symbol"] not in ALL_SYMBOLS:
            raise ValueError("Unsupported symbol")
        return
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
        if params["symbol"] not in ALL_SYMBOLS:
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
        self._base_min_spacing = min_spacing
        self._weight_throttle_until = 0.0
        self.time_basis = time_basis
        self.clock_evidence_sha256 = clock_evidence_sha256
        # Only sending is serialized. Each worker owns its connection pool;
        # waiting on the network never holds the shared rate/receipt lock.
        self._request_lock = threading.Lock()
        self._lock = threading.RLock()
        self._last_request = 0.0
        self._local = threading.local()
        self._sessions = {}
        self._sessions_lock = threading.Lock()
        self._active_sessions = set()
        self._closed = False
        self._memory_backoff = {}
        self._pending_backoff = {}
        self._backoff_lock = threading.Lock()
        self.backoff_persistence_failure = None
        self.backoff_persistence_warning = None

    @property
    def _session(self):
        """The calling thread's session; retained for diagnostic compatibility."""
        return getattr(self._local, "session", None)

    def _session_for_request(self):
        with self._sessions_lock:
            if self._closed:
                raise ValueError("PUBLIC_INPUT_CLIENT_CLOSED")
            if self._session is None:
                self._local.session = requests.Session()
                self._sessions[threading.get_ident()] = self._session
            self._active_sessions.add(threading.get_ident())
        session = self._session
        session.trust_env = False
        session.auth = None
        proxies = getattr(session, "proxies", None)
        if proxies is not None:
            proxies.clear()
        cookies = getattr(session, "cookies", None)
        if cookies is not None:
            cookies.clear()
        return session

    def _close_session(self):
        session = self._session
        self._local.session = None
        with self._sessions_lock:
            self._sessions.pop(threading.get_ident(), None)
        close = getattr(session, "close", None)
        if close:
            close()

    def close(self):
        """Never wait on retired requests; each active worker closes on exit."""
        with self._sessions_lock:
            self._closed = True
            inactive = [key for key in self._sessions if key not in self._active_sessions]
            sessions = [self._sessions.pop(key) for key in inactive]
        for session in sessions:
            close = getattr(session, "close", None)
            if close:
                close()
        self._local.session = None

    def _check_backoff(self, family):
        if self.backoff_persistence_failure:
            raise ValueError("PUBLIC_BACKOFF_PERSISTENCE_FAILED")
        pending_until = self._pending_backoff.get(family)
        if pending_until is not None and _utc(self.clock()) < pending_until:
            raise ValueError("PUBLIC_RATE_LIMIT_BACKOFF_UNTIL_RESPONSE_HANDLED")
        self._pending_backoff.pop(family, None)
        deadlines = [self._memory_backoff[family]] if family in self._memory_backoff else []
        shared = self.path.parent / (family + "_backoff.json")
        if shared.exists():
            deadlines.append(_utc(json.loads(shared.read_text(encoding="utf-8"))["blocked_until_utc"]))
        # A durable host block must not wait for an unrelated receipt database
        # lock before stopping a new request (including after process restart).
        if deadlines and _utc(self.clock()) < max(deadlines):
            raise ValueError("PUBLIC_RATE_LIMIT_BACKOFF_UNTIL_" + _iso(max(deadlines)))
        if self.path.exists():
            with sqlite3.connect(self.path, timeout=self._sqlite_timeout()) as conn:
                exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='backoff'").fetchone()
                row = conn.execute("SELECT until_utc FROM backoff WHERE family=?", (family,)).fetchone() if exists else None
            if row:
                deadlines.append(_utc(row[0]))
        if deadlines and _utc(self.clock()) < max(deadlines):
            raise ValueError("PUBLIC_RATE_LIMIT_BACKOFF_UNTIL_" + _iso(max(deadlines)))

    def _backoff_now(self):
        """Clock fallback is allowed only for conservative no-request periods."""
        try:
            return (_utc(getattr(self._local, "receipt_clock", self.clock)()),
                    "CAPTURED_REQUEST_CLOCK")
        except Exception:
            return _utc(_now()), "SYSTEM_UTC_UNCALIBRATED_BACKOFF_ONLY"

    def _record_backoff(self, family, raw, retry_after, status):
        # This is safety evidence, not a cohort data publication. A late 429 or
        # 418 must remain durable after its generation/clock expires. It runs
        # only in the bounded HTTP worker; the coordinator never waits for it.
        now, time_basis = self._backoff_now()
        until = now + pd.Timedelta(seconds=3600 if status == 418 else 60)
        match = re.search(rb"banned until (\d{13})", raw or b"")
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
        self._memory_backoff[family] = max(until, self._memory_backoff.get(family, until))
        if not self._backoff_lock.acquire(timeout=2):
            self.backoff_persistence_failure = "SAFETY_EVIDENCE_LOCK_TIMEOUT"
            raise ValueError("PUBLIC_BACKOFF_PERSISTENCE_FAILED")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            target = self.path.parent / (family + "_backoff.json")
            previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
            if previous.get("blocked_until_utc"):
                until = max(until, _utc(previous["blocked_until_utc"]))
            until = max(until, self._memory_backoff[family])
            evidence = ({key: previous.get(key) for key in
                         ("reason", "source_receipt_sha256", "evidence_type", "retry_after", "backoff_time_basis")}
                        if previous.get("blocked_until_utc") and
                        _utc(previous["blocked_until_utc"]) > response_until else
                        {"reason": "PUBLIC_HTTP_" + str(status),
                         "source_receipt_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                         "evidence_type": "RAW_RESPONSE" if raw is not None else "HTTP_STATUS_AND_HEADERS_ONLY",
                         "retry_after": retry_after, "backoff_time_basis": time_basis})
            def publish_shared():
                temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    temporary.write_text(json.dumps({"blocked_until_utc": _iso(until), **evidence,
                        "scope": "ALL_BINANCE_HOSTS" if family == "binance" else "DERIBIT_PUBLIC_HOST",
                        "safety_margin_seconds": 15}, indent=2), encoding="utf-8")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            # Publish the restart guard first. SQLite busy timeouts may exceed
            # the expired cohort budget, but cannot erase this durable guard.
            publish_shared()
            try:
                with sqlite3.connect(self.path, timeout=.5) as conn:
                    conn.execute("CREATE TABLE IF NOT EXISTS backoff(family TEXT PRIMARY KEY, until_utc TEXT NOT NULL)")
                    old = conn.execute("SELECT until_utc FROM backoff WHERE family=?", (family,)).fetchone()
                    if old:
                        until = max(until, _utc(old[0]))
                    conn.execute("INSERT OR REPLACE INTO backoff(family,until_utc) VALUES(?,?)", (family, _iso(until)))
                publish_shared()
            except sqlite3.Error:
                self.backoff_persistence_warning = "SQLITE_UNAVAILABLE_SHARED_HOST_GUARD_DURABLE"
            self._memory_backoff[family] = until
            self._pending_backoff.pop(family, None)
        except Exception as exc:
            self.backoff_persistence_failure = type(exc).__name__
            raise ValueError("PUBLIC_BACKOFF_PERSISTENCE_FAILED") from exc
        finally:
            self._backoff_lock.release()

    def _save(self, receipt, raw):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._receipt_lock(), sqlite3.connect(self.path, timeout=self._sqlite_timeout()) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS receipts(id INTEGER PRIMARY KEY, "
                         "url TEXT NOT NULL, requested_at TEXT NOT NULL, received_at TEXT NOT NULL, "
                         "sha256 TEXT NOT NULL, http_status INTEGER NOT NULL, raw BLOB NOT NULL)")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(receipts)")}
            for name in ("requested_at_local", "received_at_local", "time_basis", "clock_evidence_sha256",
                         "generation_id", "cohort_id"):
                if name not in columns:
                    conn.execute("ALTER TABLE receipts ADD COLUMN " + name + " TEXT")
            conn.execute("INSERT INTO receipts(url,requested_at,received_at,sha256,http_status,raw,"
                         "requested_at_local,received_at_local,time_basis,clock_evidence_sha256,generation_id,cohort_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                         (receipt["url"], receipt["requested_at"], receipt["received_at"],
                          receipt["sha256"], receipt["http_status"], raw,
                          receipt["requested_at_local"], receipt["received_at_local"],
                          receipt["time_basis"], receipt["clock_evidence_sha256"],
                          receipt.get("generation_id"), receipt.get("cohort_id")))

    def _sqlite_timeout(self):
        remaining = _remaining(getattr(self._local, "deadline", None))
        return 5. if remaining is None else min(5., remaining)

    @contextmanager
    def _receipt_lock(self):
        remaining = _remaining(getattr(self._local, "deadline", None))
        acquired = self._lock.acquire(timeout=remaining) if remaining is not None else self._lock.acquire()
        if not acquired:
            raise InputDeadlineExceeded("INPUT_DEADLINE_RECEIPT_LOCK")
        try:
            yield
        finally:
            self._lock.release()

    def get(self, kind, params=None, *, deadline_monotonic=None, generation_id=None, cohort_id=None,
            receipt_clock=None, receipt_time_basis=None, receipt_clock_evidence=None):
        _remaining(deadline_monotonic)
        self._local.deadline = deadline_monotonic
        self._local.receipt_clock = receipt_clock or self.clock
        try:
            return self._get_serial(kind, params, deadline_monotonic=deadline_monotonic,
                                    generation_id=generation_id, cohort_id=cohort_id,
                                    receipt_clock=receipt_clock or self.clock,
                                    receipt_time_basis=receipt_time_basis or self.time_basis,
                                    receipt_clock_evidence=receipt_clock_evidence)
        except Exception:
            # Rebuild only this worker's failed connection, never another
            # worker's active request. There is no transparent retry.
            self._close_session()
            raise
        finally:
            with self._sessions_lock:
                self._active_sessions.discard(threading.get_ident())
                closed = self._closed
            if closed:
                self._close_session()

    def _get_serial(self, kind, params=None, *, deadline_monotonic=None,
                    generation_id=None, cohort_id=None, receipt_clock=None,
                    receipt_time_basis=None, receipt_clock_evidence=None):
        params = params or {}
        _parameters(kind, params)
        base, path = ENDPOINTS[kind]
        family = "deribit" if kind == "dvol" else "binance"
        url = base + path + ("?" + urlencode(params) if params else "")
        while True:
            remaining = _remaining(deadline_monotonic)
            acquired = self._request_lock.acquire(timeout=remaining) if remaining is not None else self._request_lock.acquire()
            if not acquired:
                raise InputDeadlineExceeded("INPUT_DEADLINE_EXCEEDED")
            try:
                _remaining(deadline_monotonic)
                self._check_backoff(family)
                if self._weight_throttle_until and time.monotonic() >= self._weight_throttle_until:
                    self._weight_throttle_until = 0.0
                    self.min_spacing = self._base_min_spacing
                delay = self.min_spacing - (time.monotonic() - self._last_request)
                if delay <= 0:
                    session = self._session_for_request()
                    _remaining(deadline_monotonic)
                    self._last_request = time.monotonic()
                    break
            finally:
                self._request_lock.release()
            remaining = _remaining(deadline_monotonic)
            if remaining is not None and delay >= remaining:
                raise InputDeadlineExceeded("INPUT_DEADLINE_BEFORE_SEND_SLOT")
            time.sleep(delay)
        remaining = _remaining(deadline_monotonic)
        requested_at_local = _iso(_now())
        request_clock = receipt_clock or self.clock
        requested_at = _iso(request_clock())
        time_basis = receipt_time_basis or self.time_basis
        clock_evidence = receipt_clock_evidence if receipt_clock_evidence is not None else self.clock_evidence_sha256
        evict_noisy_session = False
        try:
            timeout = (4, 12) if remaining is None else (min(2., remaining), min(3.5, remaining))
            with session.get(url, timeout=timeout, allow_redirects=False, stream=True,
                             headers={"User-Agent": "argon4-public-paper-inputs/1"}) as response:
                headers = getattr(response, "headers", None) or {}
                used_weight = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
                if used_weight is not None:
                    try:
                        weight_val = float(used_weight)
                        if weight_val >= 1000:
                            evict_noisy_session = True
                        if weight_val >= 1600:
                            self._weight_throttle_until = time.monotonic() + 30.0
                            self.min_spacing = max(self._base_min_spacing, 1.0)
                        elif weight_val < 1200 or time.monotonic() >= self._weight_throttle_until:
                            self._weight_throttle_until = 0.0
                            self.min_spacing = self._base_min_spacing
                    except (TypeError, ValueError):
                        pass
                if response.status_code in (418, 429):
                    # Persist the observed status/Retry-After before waiting
                    # for a body that might stall beyond this generation. An
                    # incomplete body is explicitly not a completed receipt.
                    self._pending_backoff[family] = self._backoff_now()[0] + pd.Timedelta(
                        seconds=3615 if response.status_code == 418 else 75)
                    self._record_backoff(family, None, headers.get("Retry-After"),
                                         response.status_code)
                _remaining(deadline_monotonic)
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    _remaining(deadline_monotonic)
                    size += len(chunk)
                    if size > 4_000_000:
                        raise ValueError("Public input response exceeds size limit")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                received_at_local = _iso(_now())
                received_at = _iso(request_clock())
                receipt = {"url": url, "requested_at": requested_at, "received_at": received_at,
                           "sha256": hashlib.sha256(raw).hexdigest(), "http_status": response.status_code,
                           "requested_at_local": requested_at_local, "received_at_local": received_at_local,
                           "time_basis": time_basis, "clock_evidence_sha256": clock_evidence,
                           "generation_id": generation_id, "cohort_id": cohort_id}
                retry_after = getattr(response, "headers", {}).get("Retry-After")
        finally:
            cookies = getattr(session, "cookies", None)
            if cookies is not None:
                cookies.clear()
            if evict_noisy_session:
                self._close_session()
        if receipt["http_status"] in (418, 429):
            self._record_backoff(family, raw, retry_after, receipt["http_status"])
        _remaining(deadline_monotonic)
        self._save(receipt, raw)
        _remaining(deadline_monotonic)
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
                 http_min_spacing_seconds=1.0, symbols=SYMBOLS):
        if not 15 <= window_days <= 60:
            raise ValueError("Online warmup requires 15 to 60 days")
        if (isinstance(http_min_spacing_seconds, bool) or not math.isfinite(http_min_spacing_seconds)
                or not .25 <= http_min_spacing_seconds <= 60):
            raise ValueError("Online HTTP spacing must be finite and within 0.25 to 60 seconds")
        symbols = tuple(symbols)
        if not symbols or any(symbol not in ALL_SYMBOLS for symbol in symbols):
            raise ValueError("Unsupported online input symbols")
        self.symbols = symbols
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
        self._publication_lock = threading.RLock()
        self._job_local = threading.local()
        self._generation = 0
        self._active_generation = None
        self._outstanding = set()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="argon4-input")
        self._closed = False
        self._funding_cache = {}
        self._cached_clock = None
        self._basis_fingerprint = None

    def close(self):
        """Retire workers without waiting on blocked sockets or publications."""
        # Retirement cannot wait for a worker's filesystem operation. New
        # generations already refuse admission while any old worker remains.
        self._closed = True
        self._active_generation = None
        self._pool.shutdown(wait=False, cancel_futures=True)
        close = getattr(self.client, "close", None)
        if close:
            close()

    def _guard_generation(self):
        context = getattr(self._job_local, "context", None)
        if context is not None:
            generation, deadline, _, _, _, _ = context
            _remaining(deadline)
            if self._closed or generation != self._active_generation:
                raise InputDeadlineExceeded("INPUT_GENERATION_RETIRED")
        return context

    def _get(self, kind, params=None):
        context = self._guard_generation()
        if context is not None and isinstance(self.client, PublicInputsHTTP):
            generation, deadline, cohort_id, request_clock, time_basis, clock_evidence = context
            result = self.client.get(kind, params, deadline_monotonic=deadline,
                                     generation_id=str(generation), cohort_id=cohort_id,
                                     receipt_clock=request_clock, receipt_time_basis=time_basis,
                                     receipt_clock_evidence=clock_evidence)
        else:
            result = self.client.get(kind, params)
        self._guard_generation()
        return result

    def _run_job(self, context, fn, args):
        self._job_local.context = context
        try:
            self._guard_generation()
            result = fn(*args)
            self._guard_generation()
            return result
        finally:
            self._job_local.context = None

    def _publish_frame(self, frame, path, *, index=False):
        self._guard_generation()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            frame.to_parquet(temporary, index=index)
            # Generation verification and the actual replace are one atomic
            # publication gate. A retired worker cannot overwrite a new view.
            with self._publication_lock:
                self._guard_generation()
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

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
        self._guard_generation()
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
        with self._publication_lock:
            self._guard_generation()
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
        context = self._guard_generation()
        cohort_id = context[2] if context is not None else None
        if self._closed_cache_complete(
                old, "date", step, cutoff.floor(step)-step,
                min_rows=int(pd.Timedelta(days=15)/step)) and (
                interval != "5m" or (
                    cohort_id is not None
                    and not str(cohort_id).startswith("maintenance:")
                    and _utc(old["first_received_at"].iloc[-1]) >= cutoff.floor(step)
                )):
            # The closed data is already held. This creates no new receipt and
            # preserves its first-received time; freshness is candle based.
            return str(path)
        start = self._start(old, "date", cutoff, step)
        end = cutoff.floor(step) - pd.Timedelta(milliseconds=1)
        rows = []
        while start <= end:
            needed_bars = int((end - start) // step) + 4
            page_limit = min(1000, max(10, needed_bars))
            try:
                payload, receipt = self._get(kind, {"symbol": symbol, "interval": interval,
                    "startTime": _ms(start), "endTime": _ms(end), "limit": page_limit})
            except Exception:
                # Preserve already received, validated pages when a later page
                # fails. Snapshot still blocks on missing coverage/errors.
                if rows:
                    partial = pd.DataFrame(rows)
                    _validate_prices(partial)
                    self._publish_frame(_merge(old, partial, ["date"]), path)
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
            if len(payload) < page_limit:
                break
        new = pd.DataFrame(rows)
        if not new.empty:
            _validate_prices(new)
        merged = _merge(old, new, ["date"])
        if merged.empty:
            raise ValueError("EMPTY_CANDLES")
        self._publish_frame(merged, path)
        return str(path)

    def _funding(self, symbol, cutoff):
        path = self._path("funding", symbol)
        old = self._read(path)
        cached = self._funding_cache.get(symbol)
        if cached and not old.empty:
            age = (cutoff - _utc(cached["received_at"])).total_seconds()
            dates = pd.DatetimeIndex(old.date).sort_values()
            observed = np.diff(dates.asi8)
            # A receipt TTL is not a fabricated funding calendar. Near the
            # earliest next event suggested by actual observations, recheck.
            before_next = (len(observed) >= 2 and (observed > 0).all() and
                           cutoff < dates[-1] + pd.Timedelta(int(observed.min()), unit="ns") - pd.Timedelta(seconds=60))
            if (0 <= age < 60 and before_next and
                    cached["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()):
                return str(path)
        with self._publication_lock:
            self._guard_generation()
            self._funding_cache.pop(symbol, None)
        start = (cutoff.floor("D") - pd.Timedelta(days=self.window_days + 3) if old.empty
                 else pd.Timestamp(old.date.max()) - pd.Timedelta(days=2))
        rows = []
        while start <= cutoff:
            page_limit = 1000 if old.empty else 30
            payload, receipt = self._get("funding", {"symbol": symbol, "startTime": _ms(start),
                                                "endTime": _ms(cutoff), "limit": page_limit})
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
            if len(payload) < page_limit:
                break
        merged = _merge(old, pd.DataFrame(rows), ["date"])
        if merged.empty:
            raise ValueError("EMPTY_FUNDING_EVENTS")
        self._publish_frame(merged, path)
        # Reading/hash computation can stall on the filesystem. Keep it out of
        # the tiny publication gate and recheck retirement after it completes.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self._publication_lock:
            self._guard_generation()
            self._funding_cache[symbol] = {"received_at": receipt["received_at"],
                                           "sha256": digest}
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
            payload, receipt = self._get("dvol", {"currency": symbol, "resolution": "1D",
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
        self._publish_frame(frame, path, index=True)

    def _check_clock(self):
        errors = []
        for kind in ("clock", "spot_clock"):
            try:
                data, receipt = self._get(kind)
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

    def _dvol(self, cutoff):
        old = self._read(self._path("dvol"))
        parts = [self._dvol_one(s, cutoff, old[old.symbol == s] if not old.empty else old)
                 for s in ("BTC", "ETH")]
        self._publish_frame(pd.concat(parts, ignore_index=True), self._path("dvol"))

    def _blocked(self, cutoff, reason, *, generation=None, cohort_id=None):
        return {"status": "BLOCKED", "mode": "PUBLIC_INPUTS_ONLY", "data_root": str(self.data_root),
                "as_of": _iso(self.clock()), "local_now": _iso(self.clock()),
                "system_local_now": _iso(_now()), "time_basis": self.time_basis,
                "clock_evidence_sha256": self.clock_evidence_sha256,
                "requested_cutoff": _iso(cutoff), "refresh_completed_at": None,
                "latest_closed_5m": None, "feature_available_at": None,
                "clock": {"status": "UNAVAILABLE"}, "sources": {}, "gaps": [],
                "errors": [{"source": "refresh", "error": reason}],
                "generation_id": generation, "cohort_id": cohort_id}

    def _write_state(self, cutoff, clock, errors, generation, cohort_id):
        """Prepare state off-lock in a bounded worker, then gate publication."""
        self._guard_generation()
        state = {"schema_version": 1, "requested_cutoff": _iso(cutoff),
                 "refresh_completed_at": _iso(self.clock()), "clock": clock, "errors": errors,
                 "time_basis": self.time_basis, "clock_evidence_sha256": self.clock_evidence_sha256,
                 "checks": dict(self._checks), "generation_id": generation, "cohort_id": cohort_id}
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_name(self.state_path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temp.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
            with self._publication_lock:
                self._guard_generation()
                temp.replace(self.state_path)
                # If an OS-level atomic rename itself stalled past expiry, its
                # disk effect cannot be canceled. Never then publish READY in
                # memory; the outstanding-worker gate requires a fresh batch.
                self._guard_generation()
                self.state = state
        finally:
            temp.unlink(missing_ok=True)
        return True

    def _batch(self, now, deadline_monotonic, cohort_id, *, auxiliary):
        cutoff = _utc(now) if now is not None else _utc(self.clock())
        if cutoff > _utc(self.clock()) + pd.Timedelta(seconds=2):
            raise ValueError("Future refresh cutoff")
        try:
            remaining = _remaining(deadline_monotonic)
        except InputDeadlineExceeded:
            return self._blocked(cutoff, "INPUT_DEADLINE_EXCEEDED", cohort_id=cohort_id)
        acquired = self._lock.acquire(timeout=remaining) if remaining is not None else self._lock.acquire()
        if not acquired:
            return self._blocked(cutoff, "INPUT_DEADLINE_WAITING_FOR_BATCH", cohort_id=cohort_id)
        generation = None
        try:
            _remaining(deadline_monotonic)
            self._outstanding = {f for f in self._outstanding if not f.done()}
            if self._closed or self._outstanding:
                return self._blocked(cutoff, "INPUT_CLIENT_CLOSED" if self._closed else "INPUT_WORKERS_BUSY",
                                     cohort_id=cohort_id)
            if getattr(self.client, "backoff_persistence_failure", None):
                return self._blocked(cutoff, "PUBLIC_BACKOFF_PERSISTENCE_FAILED", cohort_id=cohort_id)
            # _lock serializes coordinators and _outstanding excludes retired
            # workers. No worker publication lock is needed for admission.
            self._generation += 1
            generation = self._generation
            self._active_generation = generation
            # Capture the generation's clock and calibration. A retired socket
            # must not acquire a newer cohort's mutable receipt-clock evidence.
            context = (generation, deadline_monotonic, cohort_id, self.clock,
                       self.time_basis, self.clock_evidence_sha256)
            errors, results = [], {}
            clock = self._cached_clock
            cached_clock_ok = (clock is not None and clock.get("status") == "VERIFIED" and
                               -5 <= (cutoff - _utc(clock["received_at"])).total_seconds() <= 30 and
                               getattr(self, "_cached_clock_identity", None) == context[4:])
            jobs = [] if cached_clock_ok else [("clock", self._check_clock, ())]
            if not auxiliary:
                jobs += [(f"futures:{s}:5m", self._candles, ("futures", s, "5m", cutoff)) for s in self.symbols]
            jobs += [(f"{k}:{s}:1h", self._candles, (k, s, "1h", cutoff)) for s in self.symbols for k in ("futures", "mark")]
            jobs += [(f"{k}:{s}:4h", self._candles, (k, s, "4h", cutoff)) for s in BASIS_SYMBOLS for k in ("futures", "spot")]
            jobs += [(s, self._funding, (s, cutoff)) for s in self.symbols]
            jobs += [("dvol", self._dvol, (cutoff,))]

            def execute(job_list):
                futures = {self._pool.submit(self._run_job, context, fn, args): name
                           for name, fn, args in job_list}
                self._outstanding.update(futures)
                _, pending = wait(futures, timeout=_remaining(deadline_monotonic))
                if pending:
                    raise InputDeadlineExceeded("INPUT_DEADLINE_EXCEEDED")
                for future, name in futures.items():
                    try:
                        results[name] = future.result()
                    except Exception as exc:
                        errors.append({"source": name, "error": type(exc).__name__ + ":" + str(exc)[:160]})
                _remaining(deadline_monotonic)

            execute(jobs)
            execute([("basis", self._basis, ())])
            if not cached_clock_ok:
                clock = results.get("clock", {"status": "UNAVAILABLE"})
            _remaining(deadline_monotonic)
            self._cached_clock = (clock if (auxiliary or cohort_id is not None)
                                  and isinstance(clock, dict) and clock.get("status") == "VERIFIED" else None)
            self._cached_clock_identity = context[4:]
            if auxiliary:
                return {"status": "AUXILIARY_PREFETCH_COMPLETE" if not errors else "BLOCKED",
                        "requested_cutoff": _iso(cutoff), "completed_at": _iso(self.clock()),
                        "generation_id": generation, "cohort_id": cohort_id, "errors": errors}
            execute([("state", self._write_state, (cutoff, clock, errors, generation, cohort_id))])
            if results.get("state") is not True:
                result = self._blocked(cutoff, "INPUT_STATE_PUBLICATION_FAILED", generation=generation, cohort_id=cohort_id)
                result["errors"].extend(errors)
                return result
            execute([("snapshot", self.snapshot, ())])
            result = results.get("snapshot")
            if result is None:
                result = self._blocked(cutoff, "INPUT_SNAPSHOT_FAILED", generation=generation, cohort_id=cohort_id)
                result["errors"].extend(errors)
            _remaining(deadline_monotonic)
            result.update(generation_id=generation, cohort_id=cohort_id)
            return result
        except InputDeadlineExceeded as exc:
            result = self._blocked(cutoff, str(exc), generation=generation, cohort_id=cohort_id)
            if not auxiliary:
                # The deadline path must not touch the filesystem or wait on a
                # worker lock. Pipeline persists the returned BLOCKED evidence
                # separately; the next valid generation publishes new state.
                self.state = {"schema_version": 1, "requested_cutoff": _iso(cutoff),
                    "refresh_completed_at": None, "clock": result["clock"], "errors": result["errors"],
                    "time_basis": self.time_basis, "clock_evidence_sha256": self.clock_evidence_sha256,
                    "checks": dict(self._checks), "generation_id": generation, "cohort_id": cohort_id}
            return result
        finally:
            if generation is not None and self._active_generation == generation:
                self._active_generation = None
            for future in self._outstanding:
                future.cancel()
            self._lock.release()

    def refresh(self, now=None, *, deadline_monotonic=None, cohort_id=None):
        """Refresh with a monotonic absolute budget; omitted budget supports warmup.

        An expired generation is retired without joining running HTTP workers.
        Until those bounded workers finish, another generation fails closed.
        """
        return self._batch(now, deadline_monotonic, cohort_id, auxiliary=False)

    def prefetch_auxiliary(self, now=None, *, deadline_monotonic=None, cohort_id=None):
        """Fetch only already-closed auxiliary bars and observed funding events."""
        return self._batch(now, deadline_monotonic, cohort_id, auxiliary=True)

    def capture_early_window_book_ticker(self, now=None, *, cohort_id=None, deadline_monotonic=None):
        """Capture pre-decision Binance futures bookTicker top-of-book with receipt provenance."""
        cutoff = _utc(now) if now is not None else _utc(self.clock())
        if isinstance(self.client, PublicInputsHTTP):
            payload, receipt = self.client.get(
                "book_ticker", {},
                deadline_monotonic=deadline_monotonic,
                cohort_id=cohort_id,
                receipt_clock=self.clock,
                receipt_time_basis=self.time_basis,
                receipt_clock_evidence=self.clock_evidence_sha256,
            )
        else:
            payload, receipt = self.client.get("book_ticker", {})
        items = payload if isinstance(payload, list) else ([payload] if isinstance(payload, dict) else [])
        target_set = set(self.symbols)
        tickers = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            sym = item.get("symbol")
            if sym in target_set:
                try:
                    bid_p = float(item["bidPrice"])
                    bid_q = float(item["bidQty"])
                    ask_p = float(item["askPrice"])
                    ask_q = float(item["askQty"])
                except (KeyError, TypeError, ValueError):
                    continue
                mid = 0.5 * (bid_p + ask_p) if (bid_p > 0 and ask_p > 0) else 0.0
                imbalance = (bid_q - ask_q) / (bid_q + ask_q) if (bid_q + ask_q) > 0 else 0.0
                tickers[sym] = {
                    "symbol": sym,
                    "bid_price": bid_p,
                    "bid_qty": bid_q,
                    "ask_price": ask_p,
                    "ask_qty": ask_q,
                    "mid_price": mid,
                    "top_book_imbalance": imbalance,
                    "event_time_ms": item.get("time"),
                }
        return {
            "status": "OBSERVED" if tickers else "EMPTY",
            "cohort_id": cohort_id,
            "requested_at": receipt.get("requested_at"),
            "received_at": receipt.get("received_at"),
            "receipt_sha256": receipt.get("sha256"),
            "time_basis": receipt.get("time_basis", self.time_basis),
            "clock_evidence_sha256": receipt.get("clock_evidence_sha256", self.clock_evidence_sha256),
            "as_of": _iso(cutoff),
            "tickers": tickers,
        }

    def _source_specs(self):
        for symbol in self.symbols:
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
        # Batch workers already have an exclusive generation while the caller
        # holds _lock; running validation in the bounded pool keeps disk reads
        # within the same non-blocking deadline envelope as HTTP.
        with self._lock if self._guard_generation() is None else nullcontext():
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
                    "latest_closed_5m": _iso(min(latest5)) if len(latest5) == len(self.symbols) else None,
                    "feature_available_at": _iso(max(available)) if available else None,
                    "clock": clock, "sources": sources, "gaps": gaps, "errors": errors}

    def mark_registry_evidence(self):
        """Evidence only. The caller registers these files before model use."""
        result = []
        for symbol in self.symbols:
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
