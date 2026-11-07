"""Future-only public book evidence. This module has no order capability.

Sampling is independent of prediction readiness and never counts as a fill. Six
bounded daemon workers own separate GET-only sessions; neither a slow endpoint
nor this database can hold the paper executor's locks or connection pool.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
import queue
import re
import sqlite3
import threading
import time

from .clock_sync import calibrated_now, project_local, utc
from .polymarket import STREAMS, validate_book
from .polymarket_transport import PublicPolymarketTransport, SOURCE, public_url
from .stage import stage_profile


OFFSETS = (2, 6, 10, 15, 20, 30, 60)
EVIDENCE_CLASS = "PUBLIC_BOOK_OBSERVATION_ONLY_NOT_FILL_EVIDENCE"
SAMPLE_TOLERANCE_SECONDS = 1.0
BOOK_MAX_AGE_SECONDS = 2.0
MAX_WORKERS = 6


def _iso(value):
    return utc(value).isoformat().replace("+00:00", "Z")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def fee_inclusive_capacity(book, fee, limit=Decimal("0.515")):
    """Displayed ask depth under the existing ask limit; not executable size.

    Only the validated exponent-one taker formula used by the paper executor is
    supported. Unknown fees or other schedules have unknown capacity, not zero
    fees. Queue position, adverse moves and balances are deliberately absent.
    """
    unknown = {"status": "UNKNOWN_FEE", "usd": None, "shares": None,
               "limit_price": str(limit), "is_fill_estimate": False}
    try:
        if fee.get("status") == "KNOWN_DISABLED" and fee.get("fees_enabled") is False:
            rate = Decimal("0")
        elif (fee.get("status") == "KNOWN_SCHEDULE" and fee.get("fees_enabled") is True
              and fee.get("taker_only") is True and Decimal(str(fee.get("exponent"))) == 1):
            rate = Decimal(str(fee["rate"]))
            if not rate.is_finite() or not 0 <= rate <= 1:
                return unknown
        else:
            return unknown
        if book.get("status") != "OBSERVED":
            return {**unknown, "status": "BOOK_NOT_VALID"}
        cost, shares = Decimal("0"), Decimal("0")
        for row in book["asks"]:
            price, size = Decimal(str(row["price"])), Decimal(str(row["size"]))
            if not price.is_finite() or not size.is_finite() or not 0 < price < 1 or size < 0:
                return {**unknown, "status": "BOOK_NOT_VALID"}
            unit_cost = price + rate * price * (1-price)
            if price <= limit:
                cost += unit_cost * size
                shares += size
        return {"status": "OBSERVED_DEPTH", "usd": str(cost.quantize(Decimal("0.000001"), rounding=ROUND_FLOOR)),
                "shares": str(shares), "limit_price": str(limit),
                "fee_formula": "shares * rate * price * (1-price)",
                "is_fill_estimate": False}
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return unknown


class BookTelemetry:
    """A separate observational service; providers must be local snapshots.

    ``market_provider`` returns ObservationStore.latest("polymarket"). The
    optional ``decision_provider`` returns ForwardPipeline.status(). Providers
    are never invoked by the execution path. Call worker(stop_event) in its own
    daemon thread. tick() is also exposed for deterministic scheduling tests.
    """

    def __init__(self, runtime, *, policy_provider, market_provider, clock_provider,
                 decision_provider=None, transport_factory=None, request_spacing_seconds=.05):
        self.path = Path(runtime) / "book_telemetry.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_provider = policy_provider
        self.market_provider = market_provider
        self.clock_provider = clock_provider
        self.decision_provider = decision_provider
        self.transport_factory = transport_factory or (lambda: PublicPolymarketTransport(pool_size=1))
        self.spacing = max(.05, float(request_spacing_seconds))
        self._db_lock = threading.Lock()
        self._lock = threading.RLock()
        self._tick_lock = threading.Lock()
        self._pending = {}
        self._last_armed = None
        self._closed = threading.Event()
        self._queue = queue.Queue(maxsize=MAX_WORKERS)
        self._slots = threading.BoundedSemaphore(MAX_WORKERS)
        self._workers = []
        self._active = 0
        self._completed = 0
        self._last_sample = None
        self._last_error = None
        self._clock_status = "WAITING_CLOCK"
        self._next_request = 0.0
        self._backoff_until = None
        self._decision_seen = set()
        self._initialize()

    def _connection(self):
        return sqlite3.connect(self.path, timeout=.25)

    def _initialize(self):
        with self._db_lock, self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS samples (sample_id TEXT PRIMARY KEY, cohort TEXT NOT NULL, stream TEXT NOT NULL, outcome TEXT NOT NULL, target_offset INTEGER NOT NULL, target_at TEXT NOT NULL, status TEXT NOT NULL, payload_json TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS decision_links (cohort TEXT NOT NULL, stream TEXT NOT NULL, observed_at TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY(cohort, stream))")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
            # A restart never resumes a past plan or silently drops its gap.
            for sample_id, payload in db.execute("SELECT sample_id, payload_json FROM samples WHERE status IN ('PLANNED','DISPATCHED')").fetchall():
                row = json.loads(payload)
                row.update(status="MISSED_SAMPLE", reason="PROCESS_RESTART", recorded_at=_iso(datetime.now(timezone.utc)))
                db.execute("UPDATE samples SET status=?,payload_json=? WHERE sample_id=?", (row["status"], _json(row), sample_id))
            previous = db.execute("SELECT value FROM metadata WHERE name='backoff_until'").fetchone()
            if previous:
                self._backoff_until = utc(previous[0])

    def _save(self, row, *, insert=False):
        with self._db_lock, self._connection() as db:
            if insert:
                return db.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?,?)", (
                    row["sample_id"], row["cohort"], row["stream"], row["outcome"],
                    row["target_offset_seconds"], row["target_at"], row["status"], _json(row))).rowcount == 1
            db.execute("UPDATE samples SET status=?,payload_json=? WHERE sample_id=?", (row["status"], _json(row), row["sample_id"]))
        return True

    def _finish(self, row, status, reason=None):
        row.update(status=status, reason=reason, recorded_at=_iso(datetime.now(timezone.utc)))
        self._save(row)
        with self._lock:
            self._pending.pop(row["sample_id"], None)
            self._completed += 1
            self._last_sample = {k: row.get(k) for k in (
                "cohort", "stream", "outcome", "target_offset_seconds", "target_at", "status", "reason",
                "requested_at_local", "received_at_local", "best_bid", "best_ask", "capacity")}

    def _bounds(self):
        clock = _copy(self.clock_provider() or {})
        lower = calibrated_now(clock, upper=False)
        upper = calibrated_now(clock, upper=True)
        uncertainty = float(clock["uncertainty_seconds"])
        if (not 0 <= uncertainty <= .5 or upper < lower
                or clock.get("source") != "POLYMARKET_CLOB_SERVER_TIME"
                or not re.fullmatch(r"[0-9a-fA-F]{64}", str(clock.get("receipt_sha256", "")))):
            raise ValueError("clock uncertainty invalid")
        return clock, lower, upper

    def _arm(self, cohort):
        policy = _copy(self.policy_provider())
        profile = stage_profile(policy, cohort)
        policy_hash = hashlib.sha256(_json(policy).encode()).hexdigest()
        for stream in profile["enabled_streams"]:
            if stream not in STREAMS or int(cohort.timestamp()) % STREAMS[stream][1]:
                continue
            for offset in OFFSETS:
                for outcome in ("UP", "DOWN"):
                    identity = f"{int(cohort.timestamp())}:{stream}:{outcome}:{offset}"
                    row = {"sample_id": identity, "cohort": _iso(cohort), "stream": stream, "outcome": outcome,
                           "target_offset_seconds": offset, "target_at": _iso(cohort + timedelta(seconds=offset)),
                           "status": "PLANNED", "reason": None, "armed_at_local": _iso(datetime.now(timezone.utc)),
                           "source": SOURCE, "evidence_class": EVIDENCE_CLASS, "policy_sha256": policy_hash,
                           "stage_id": profile["id"], "execution_enabled": False, "live_enabled": False,
                           "best_bid": None, "best_ask": None, "capacity": None}
                    if self._save(row, insert=True):
                        self._pending[identity] = row
        self._last_armed = cohort

    def _market(self, snapshot, row):
        state = (snapshot or {}).get("streams", {}).get(row["stream"], {})
        for slot in ("current", "next"):
            market = state.get(slot)
            if not isinstance(market, dict):
                continue
            if market.get("status") != "VERIFIED_MARKET" or not market.get("window_start"):
                continue
            try:
                if utc(market["window_start"]) != utc(row["cohort"]):
                    continue
            except (TypeError, ValueError):
                continue
            token = str(market.get("outcome_ids", {}).get(row["outcome"], ""))
            if (not re.fullmatch(r"[0-9]{1,78}", token)
                    or not re.fullmatch(r"0x[0-9a-fA-F]{64}", str(market.get("condition_id", "")))):
                continue
            return _copy(market), token
        return None, None

    def _ensure_workers(self):
        with self._lock:
            if not self._workers and not self._closed.is_set():
                for i in range(MAX_WORKERS):
                    worker = threading.Thread(target=self._request_worker, name=f"book-evidence-{i}", daemon=True)
                    self._workers.append(worker)
                    worker.start()

    def tick(self):
        if self._closed.is_set() or not self._tick_lock.acquire(blocking=False):
            return
        try:
            try:
                clock, lower, upper = self._bounds()
                self._clock_status = "CALIBRATED"
            except (KeyError, TypeError, ValueError, OverflowError):
                self._clock_status = "WAITING_VALID_CLOCK"
                return
            # Only future boundaries are armed, including first start/restart.
            future = datetime.fromtimestamp((int(upper.timestamp()) // 300 + 1) * 300, timezone.utc)
            if self._last_armed is None or future > self._last_armed:
                self._arm(future)
            due = [r for r in self._pending.values() if utc(r["target_at"]) <= lower]
            snapshot = None
            for row in due:
                target = utc(row["target_at"])
                row["dispatch_clock"] = clock
                if upper > target + timedelta(seconds=SAMPLE_TOLERANCE_SECONDS):
                    self._finish(row, "MISSED_SAMPLE", "SCHEDULER_LATE")
                    continue
                with self._lock:
                    backoff = self._backoff_until
                if backoff and upper < backoff:
                    self._finish(row, "MISSED_SAMPLE", "HOST_BACKOFF")
                    continue
                if snapshot is None:
                    snapshot = self.market_provider() or {}
                market, token = self._market(snapshot, row)
                if market is None:
                    self._finish(row, "MISSED_SAMPLE", "MARKET_METADATA_UNAVAILABLE")
                    continue
                if not self._slots.acquire(blocking=False):
                    self._finish(row, "MISSED_SAMPLE", "OBSERVATION_WORKERS_BUSY")
                    continue
                row.update(status="DISPATCHED", market_id=market.get("market_id"), condition_id=market["condition_id"],
                           slug=market.get("slug"), token_id=token, fee=market.get("fee"),
                           metadata_received_at_local=market.get("metadata_received_at_local", market.get("metadata_received_at")),
                           metadata_sha256=market.get("metadata_sha256"))
                try:
                    self._save(row)
                except Exception:
                    self._slots.release()
                    raise
                with self._lock:
                    self._pending.pop(row["sample_id"], None)
                    self._active += 1
                self._ensure_workers()
                self._queue.put_nowait(row)
            self._poll_decisions()
        except Exception as exc:
            self._last_error = type(exc).__name__
        finally:
            self._tick_lock.release()

    def _request_worker(self):
        transport = None
        try:
            while not self._closed.is_set():
                try:
                    row = self._queue.get(timeout=.25)
                except queue.Empty:
                    continue
                try:
                    if transport is None:
                        transport = self.transport_factory()
                    self._collect(row, transport)
                except Exception as exc:
                    self._last_error = type(exc).__name__
                    try:
                        self._finish(row, "FAILED_OBSERVATION", type(exc).__name__)
                    except Exception:
                        pass
                finally:
                    with self._lock:
                        self._active -= 1
                    self._slots.release()
                    self._queue.task_done()
        finally:
            if transport is not None:
                transport.close()

    def _host_backoff(self, http_status):
        if http_status not in (418, 429):
            return
        until = datetime.now(timezone.utc) + timedelta(seconds=3600 if http_status == 418 else 60)
        with self._lock:
            self._backoff_until = max(until, self._backoff_until or until)
            value = _iso(self._backoff_until)
        with self._db_lock, self._connection() as db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('backoff_until', ?)", (value,))

    def _collect(self, row, transport):
        # Only reserve a rate token under the lock. No network or sleep inside it.
        with self._lock:
            scheduled = max(time.monotonic(), self._next_request)
            self._next_request = scheduled + self.spacing
        if self._closed.wait(max(0, scheduled-time.monotonic())):
            self._finish(row, "MISSED_SAMPLE", "SERVICE_STOPPED")
            return
        clock, lower, upper = self._bounds()
        target = utc(row["target_at"])
        if upper > target + timedelta(seconds=SAMPLE_TOLERANCE_SECONDS):
            self._finish(row, "MISSED_SAMPLE", "REQUEST_SLOT_LATE")
            return
        with self._lock:
            backoff = self._backoff_until
        if backoff and datetime.now(timezone.utc) < backoff:
            self._finish(row, "MISSED_SAMPLE", "HOST_BACKOFF")
            return
        row["request_clock"] = clock
        raw, receipt, error = transport.get("book", row["token_id"])
        # The receipt retains exact raw_text even if the decoded payload includes
        # invalid non-finite JSON values, which must never enter derived quotes.
        row.update(receipt=receipt, requested_at_local=receipt.get("requested_at"),
                   received_at_local=receipt.get("received_at"))
        self._host_backoff(receipt.get("http_status"))
        if error or receipt.get("http_status") != 200 or not receipt.get("body_complete"):
            self._finish(row, "FAILED_OBSERVATION", error or "INCOMPLETE_RECEIPT")
            return
        if (receipt.get("source") != SOURCE or receipt.get("method") != "GET"
                or receipt.get("url") != public_url("book", row["token_id"])
                or not receipt.get("requested_at") or not receipt.get("received_at")
                or not isinstance(receipt.get("raw_text"), str)
                or hashlib.sha256(receipt["raw_text"].encode()).hexdigest() != receipt.get("sha256")):
            self._finish(row, "INVALID_OBSERVATION", "RECEIPT_PROOF_INVALID")
            return
        # Reject an intervening local-clock step / expired calibration. Keep raw
        # receipts even when their timing cannot be trusted.
        try:
            calibrated_now(clock, upper=True)
            requested_lower = project_local(receipt["requested_at"], clock, upper=False)
            received_lower = project_local(receipt["received_at"], clock, upper=False)
            received_upper = project_local(receipt["received_at"], clock, upper=True)
            row.update(requested_at_server_lower=_iso(requested_lower),
                       received_at_server_lower=_iso(received_lower), received_at_server_upper=_iso(received_upper),
                       clock_uncertainty_seconds=clock["uncertainty_seconds"],
                       receipt_lateness_upper_seconds=(received_upper-target).total_seconds())
            book = validate_book(raw, row["token_id"], row["condition_id"], received_upper)
            row["parsed_book"] = book
            source = utc(book["source_at"])
            row.update(source_at=book["source_at"], source_book_age_upper_seconds=(received_upper-source).total_seconds(),
                       source_book_age_lower_seconds=(received_lower-source).total_seconds())
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._finish(row, "INVALID_OBSERVATION", type(exc).__name__)
            return
        if requested_lower < target or received_lower < requested_lower:
            self._finish(row, "INVALID_OBSERVATION", "RECEIPT_TIMING_INVALID")
            return
        if received_upper > target + timedelta(seconds=SAMPLE_TOLERANCE_SECONDS):
            self._finish(row, "MISSED_SAMPLE", "RESPONSE_LATE_RAW_RETAINED")
            return
        if book["status"] != "OBSERVED" or source > received_upper or (received_upper-source).total_seconds() > BOOK_MAX_AGE_SECONDS:
            self._finish(row, "INVALID_OBSERVATION", "BOOK_STALE_FUTURE_OR_INCOMPLETE")
            return
        row.update(best_bid=book["best_bid"], best_ask=book["best_ask"])
        try:
            if not row.get("metadata_received_at_local"):
                raise ValueError("metadata receipt missing")
            metadata_age = (received_upper - project_local(row["metadata_received_at_local"], clock, upper=False)).total_seconds()
            valid_fee_proof = bool(re.fullmatch(r"[0-9a-f]{64}", row["metadata_sha256"])) and 0 <= metadata_age <= 90
        except (KeyError, TypeError, ValueError):
            valid_fee_proof = False
        row["capacity"] = (fee_inclusive_capacity(book, row.get("fee") or {}) if valid_fee_proof
                           else {"status": "FEE_METADATA_UNVERIFIED", "usd": None, "shares": None, "is_fill_estimate": False})
        self._finish(row, "OBSERVED")

    def publish_decisions(self, cohort, decisions, *, inputs_ready_at=None, observed_at=None):
        """Attach later evidence without changing a single sample timestamp."""
        if utc(cohort).timestamp() % 300:
            raise ValueError("decision cohort must be a five-minute UTC boundary")
        cohort = _iso(cohort)
        seen_at = _iso(observed_at or datetime.now(timezone.utc))
        with self._db_lock, self._connection() as db:
            for stream, decision in decisions.items():
                if stream not in STREAMS or not isinstance(decision, dict):
                    continue
                prediction = decision.get("prediction")
                if prediction and (not prediction.get("feature_bar_open") or not prediction.get("decision_time")
                        or utc(prediction["feature_bar_open"]) + timedelta(minutes=5) != utc(cohort)
                        or utc(prediction["decision_time"]) < utc(cohort)):
                    continue
                value = {"cohort": cohort, "stream": stream, "decision_observed_at": seen_at,
                         "inputs_ready_at": inputs_ready_at, "decision": _copy(decision),
                         "association_only": True, "sample_times_unchanged": True}
                db.execute("INSERT OR IGNORE INTO decision_links VALUES (?,?,?,?)", (cohort, stream, seen_at, _json(value)))

    def _poll_decisions(self):
        if self.decision_provider is None:
            return
        state = self.decision_provider() or {}
        cohort = state.get("timing", {}).get("cohort")
        decisions = state.get("decisions", {})
        if not cohort or not decisions:
            return
        matched = {}
        for stream, decision in decisions.items():
            prediction = decision.get("prediction", {}) if isinstance(decision, dict) else {}
            if ((cohort, stream) not in self._decision_seen
                    and prediction.get("feature_bar_open") and prediction.get("decision_time")
                    and utc(prediction["feature_bar_open"]) + timedelta(minutes=5) == utc(cohort)
                    and utc(prediction["decision_time"]) >= utc(cohort)):
                matched[stream] = decision
        if matched:
            self.publish_decisions(cohort, matched, inputs_ready_at=state.get("inputs", {}).get("feature_available_at"))
            self._decision_seen.update((cohort, stream) for stream in matched)
            if len(self._decision_seen) > 512:
                self._decision_seen = set(sorted(self._decision_seen)[-256:])

    def status(self):
        with self._lock:
            return {"status": "STOPPED" if self._closed.is_set() else self._clock_status,
                    "evidence_class": EVIDENCE_CLASS, "source": SOURCE, "database": str(self.path),
                    "target_offsets_seconds": list(OFFSETS), "sample_tolerance_seconds": SAMPLE_TOLERANCE_SECONDS,
                    "pending_samples": len(self._pending), "active_requests": self._active, "max_requests": MAX_WORKERS,
                    "completed_this_process": self._completed, "last_sample": _copy(self._last_sample),
                    "last_error": self._last_error, "last_armed_cohort": _iso(self._last_armed) if self._last_armed else None,
                    "backoff_until": _iso(self._backoff_until) if self._backoff_until else None,
                    "execution_enabled": False, "live_enabled": False, "counts_as_fills": False}

    def worker(self, stop_event):
        while not stop_event.is_set() and not self._closed.is_set():
            self.tick()
            stop_event.wait(.1)
        self.close(wait=False)

    def close(self, wait=False):
        self._closed.set()
        if wait:
            deadline = time.monotonic() + 2
            for worker in self._workers:
                worker.join(timeout=max(0, deadline-time.monotonic()))
