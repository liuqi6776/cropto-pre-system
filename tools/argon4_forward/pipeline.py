"""Local, credential-free orchestration of saved models and paper execution.

Never replay a missed decision window or count a historical backtest as cash.
The only execution class used here is the isolated SQLite paper executor.
"""
from datetime import datetime, timedelta, timezone
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import threading
import time

from .clock_sync import calibrate, calibrated_now, project_local
from .stage import stage_profile, disabled_decision
from .runtime_profile import (engineering_profile, minimum_decision_delay, V12_PROFILE_ID,
                              revision_fastpath_profile)

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
ALL_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
_MARK_REGISTRY_LOCK = threading.RLock()
_MARK_REGISTRY_CACHE = {}


def _policy_symbols(policy, profile=None):
    if not isinstance(policy, dict):
        return SYMBOLS
    disabled = set((profile or {}).get("disabled_streams", ()))
    active_streams = [s for s in policy.get("order", ()) if s not in disabled]
    if any(str(s).startswith("XRP_") for s in active_streams):
        return (*SYMBOLS, "XRPUSDT")
    return SYMBOLS


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temp = path.with_suffix(".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def cohort_at(value):
    return datetime.fromtimestamp(int(value.timestamp()) // 300 * 300, timezone.utc)


def register_online_marks(data_root):
    """Keep observed marks as research references, never install canonical data."""
    with _MARK_REGISTRY_LOCK:
        return _register_online_marks(data_root)


def _register_online_marks(data_root):
    import pandas as pd
    root = Path(data_root).resolve()
    allowed = (ROOT / "paper_runtime/argon4_forward_20261105/features/raw_data").resolve()
    if root != allowed:
        raise ValueError("online mark registry scope mismatch")
    path = ROOT / "archive/catalog/mark_candle_registry.json"
    registry_raw = path.read_bytes()
    originals = {symbol: (root / "auxiliary" / f"{symbol}_1h_mark.parquet") for symbol in ALL_SYMBOLS}
    raw_sources = {symbol: target.read_bytes() for symbol, target in originals.items() if target.exists()}
    source_hashes = tuple((symbol, hashlib.sha256(raw).hexdigest()) for symbol, raw in raw_sources.items())
    identity = (hashlib.sha256(registry_raw).hexdigest(), source_hashes)
    # Check complete bytes every time, not mtime/size. Any external edit,
    # revision or new bar invalidates the cache and gets full validation.
    if _MARK_REGISTRY_CACHE.get(str(root)) == identity:
        return {"status": "UNCHANGED_VERIFIED", "source_count": len(raw_sources)}
    registry = json.loads(registry_raw)
    refs = []
    for symbol in ALL_SYMBOLS:
        target = originals[symbol]
        if symbol not in raw_sources:
            continue
        raw = raw_sources[symbol]
        data = pd.read_parquet(io.BytesIO(raw))
        if len(data) == 0 or not {"date", "open", "high", "low", "close", "volume", "first_received_at", "receipt_sha256"}.issubset(data):
            raise ValueError("observed mark schema missing")
        import numpy as np
        import re
        digests = sorted(set(data.receipt_sha256))
        if (not np.isfinite(data[["open", "high", "low", "close", "volume"]].to_numpy(float)).all()
                or data.date.duplicated().any() or not data.date.is_monotonic_increasing
                or data.first_received_at.isna().any()
                or not all(re.fullmatch(r"[a-f0-9]{64}", str(d)) for d in digests)):
            raise ValueError("observed mark integrity invalid")
        refs.append({"dataset_id": f"argon4_local_v4_online_mark_{symbol}_20261106",
                     "pair": f"{symbol[:-4]}/USDT:USDT", "timeframe": "1h", "candle_type": "mark",
                     "local_path": target.relative_to(ROOT).as_posix(), "extension": "parquet",
                     "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(data),
                     "first_utc": str(data.date.min()), "last_utc": str(data.date.max()),
                     "verification_status": "VERIFIED_PERPETUAL_LABEL", "installed": False,
                     "research_only": True, "gate_effect": "NONE",
                     "source_endpoint": "https://fapi.binance.com/fapi/v1/markPriceKlines",
                     "source_receipt_sha256": digests,
                     "provenance_evidence": "Observed public Binance USD-M /fapi/v1/markPriceKlines; raw responses, hashes and first receive times in features/http_receipts.sqlite3. Closed observations only; no synthetic marks or canonical conversion."})
    ids = {r["dataset_id"] for r in refs}
    registry["research_references"] = [r for r in registry.get("research_references", []) if r["dataset_id"] not in ids] + refs
    if path.read_bytes() != registry_raw:
        raise ValueError("mark registry changed during publication")
    save_json(path, registry)
    expected = json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False)
    published = path.read_bytes()
    # save_json uses the platform text newline convention. Validate the exact
    # serialized text, then pin the actual bytes (including Windows CRLF).
    if published.decode("utf-8").replace("\r\n", "\n") != expected:
        raise ValueError("mark registry publication changed before verification")
    _MARK_REGISTRY_CACHE[str(root)] = (hashlib.sha256(published).hexdigest(), source_hashes)
    return {"status": "VALIDATED_AND_REGISTERED", "source_count": len(raw_sources)}


class ForwardPipeline:
    def __init__(self, account, runtime, observations, monitor_log=None):
        self.account, self.runtime, self.observations = account, Path(runtime), observations
        from .monitor_logger import MonitorAuditLogger
        self.monitor_log = monitor_log or MonitorAuditLogger(self.runtime)
        self._last_logged_block_key = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.state = {"status": "PREPARING", "active": False, "models": {}, "decisions": {},
                      "live_enabled": False, "reason": "等待本地训练与数据准备"}
        self.bundles = {}
        self.clock = None
        self.last_cohort = None
        self.inputs = None
        self._input_spacing = None
        self.last_settlement_poll = 0
        self.warmup_complete = False
        self.last_warmup = 0
        self._pending_timing = None
        self._decision_timing = {}
        self._prefetch_cohort = None
        self._pending_execution = None
        self._worker_engineering = {}
        self._last_idle_publish = 0.
        self._revision_fastpath = {"enabled": False}
        self._same_cohort_revision = None
        self._blocked_input_maintenance = None
        from .polymarket_transport import PublicPolymarketTransport
        # Clock calibration keeps this CLOB connection warm without additional
        # requests. Background discovery has its own bounded connection pool.
        self.public_transport = PublicPolymarketTransport(pool_size=1)

    def finish_timing(self, status, error=None):
        timing = self._pending_timing
        if timing is None:
            return None
        timing = dict(timing)
        timing["duration_seconds"] = time.perf_counter() - timing.pop("started_monotonic")
        timing["status"] = status
        timing["computation"] = dict(self._decision_timing)
        try:
            completed = calibrated_now(self.clock)
            timing["completed_at"] = completed.isoformat()
            boundary = datetime.fromisoformat(timing["cohort"])
            timing["completed_after_boundary_seconds"] = (completed-boundary).total_seconds()
            timing["deadline_margin_seconds"] = 20-timing["completed_after_boundary_seconds"]
        except ValueError:
            timing["clock_status"] = "UNAVAILABLE_AT_COMPLETION"
        if error is not None:
            timing["error"] = type(error).__name__ + ":" + str(error)[:160]
        self.observations.save("cycle_latency", timing)
        self._pending_timing = None
        return timing

    def status(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def publish(self, **changes):
        with self.lock:
            self.state.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
            state = json.loads(json.dumps(self.state))
        save_json(self.runtime / "pipeline_state.json", state)
        self.account.heartbeat(state["status"], state.get("reason"), details={"pipeline": state["status"]})

    def load_ready(self):
        from .model import STREAMS
        from .model_v14 import ALL_STREAMS, load_model_bundle_v14 as load_model_bundle
        policy = json.loads((ROOT / "config/argon4_forward_policy.json").read_text(encoding="utf-8"))
        if not policy.get("paper_trading_active") or policy.get("ranking_model_scope") != "LOCAL_SELF_TRAINED_OOS":
            self.abandon_prepared("PAPER_DISABLED")
            self.publish(status="WAITING_LOCAL_BACKTEST", active=False, reason="本地训练与回测尚未完成，等待本地胜率排序")
            return None
        profile = stage_profile(policy)
        with self.lock:
            self.state["stage"] = profile
        order = policy.get("order")
        target_streams = order if isinstance(order, list) and set(order) <= set(ALL_STREAMS) else STREAMS
        for stream in target_streams:
            # The saved artifacts and policy pins remain untouched. A
            # disabled stream need not load or validate a model for new work.
            if stream in profile["disabled_streams"]:
                continue
            expected = policy.get("deployment_manifest_sha256", {}).get(stream)
            if not expected:
                raise ValueError("deployment model not pinned by policy")
            if stream not in self.bundles or self.bundles[stream]["manifest_sha256"] != expected:
                self.bundles[stream] = load_model_bundle(stream, expected_manifest_sha256=expected)
        return policy

    def update_clock(self):
        age = time.monotonic()-self.clock.get("calibrated_monotonic", 0) if self.clock else None
        refresh = age is None or age > 40
        reason = "PERIODIC_OR_INITIAL"
        if not refresh and self.clock.get("status") == "CALIBRATED" and age > 20:
            lower = calibrated_now(self.clock, upper=False)
            until_boundary = (cohort_at(lower)+timedelta(minutes=5)-lower).total_seconds()
            if 0 < until_boundary <= 10:
                # Pull the same six-sample calibration ahead of the critical
                # input window. Successful calibration resets age, preventing
                # a second proactive request group at this boundary.
                refresh, reason = True, "PRE_DECISION_WINDOW"
        if refresh:
            kwargs = {}
            if self._revision_fastpath.get("enabled"):
                extra_allowed, budget = False, 6.
                if self.clock and self.clock.get("status") == "CALIBRATED" and age <= 60:
                    lower, upper = calibrated_now(self.clock, upper=False), calibrated_now(self.clock)
                    elapsed = (lower-cohort_at(lower)).total_seconds()
                    until = (cohort_at(lower)+timedelta(minutes=5)-upper).total_seconds()
                    extra_allowed = (25 < elapsed < 290 and until > 6
                                     and not self._critical_window())
                    if reason == "PRE_DECISION_WINDOW":
                        # A still-valid clock is safer than starting proactive
                        # collection too late. This does not extend its TTL.
                        if until <= 2 or self._critical_window():
                            return self.clock
                        budget = min(budget, until-.25)
                        extra_allowed = True
                    elif 0 < until <= 6:
                        budget = max(.1, until-.25)
                kwargs = {"retry_borderline": extra_allowed,
                          "budget_monotonic": time.monotonic()+budget}
            self.clock = calibrate(transport=self.public_transport, **kwargs)
            self.clock["refresh_reason"] = reason
            self.observations.save("execution_clock", self.clock)
        if self.clock.get("status") != "CALIBRATED":
            raise ValueError("clock uncertainty prevents paper execution")
        return self.clock

    def refresh_book(self, market, outcome):
        from .polymarket import _get_public, validate_book
        token = market["outcome_ids"][outcome]
        data, receipt, error = _get_public("book", token, transport=self.public_transport)
        self.observations.save("paper_book_receipt", {"receipts": [receipt], "market_id": market["market_id"], "outcome": outcome})
        if error:
            raise ValueError("fresh book unavailable")
        return validate_book(data, token, market["condition_id"], receipt["received_at"])

    def refresh_market(self, stream, boundary, existing=None):
        """One exact public lookup only when the executor reaches a signal."""
        from .polymarket import ALL_STREAMS, _get_public, validate_market
        asset, duration, _ = ALL_STREAMS[stream]
        now = calibrated_now(self.clock)
        if not boundary <= now < boundary + timedelta(seconds=20):
            raise ValueError("market refresh outside decision window")
        slug = f"{asset}-updown-{duration // 60}m-{int(boundary.timestamp())}"
        data, receipt, error = _get_public("market", slug, transport=self.public_transport)
        self.observations.save("paper_market_receipt", {"stream": stream, "cohort": boundary.isoformat(),
            "receipts": [receipt], "error": error, "clock_evidence_sha256": self.clock["receipt_sha256"]})
        now = calibrated_now(self.clock)
        if not now < boundary + timedelta(seconds=20):
            raise ValueError("market refresh exceeded decision window")
        if error or not isinstance(data, list) or len(data) != 1:
            raise ValueError("fresh market unavailable")
        market = validate_market(data[0], stream, boundary, now)
        market.update(metadata_received_at=receipt["received_at"], metadata_sha256=receipt["sha256"])
        return market

    def fetch_resolution(self, market_id):
        from .execution import fetch_public_resolution
        if self._worker_engineering.get("profile") == V12_PROFILE_ID:
            if self._settlement_guard():
                raise ValueError("DEFERRED_FOR_DECISION_WINDOW")
            lower = calibrated_now(self.clock, upper=False)
            positions = [p for p in self.account.overview()["positions"] if p["market_id"] == market_id]
            if not positions or any(datetime.fromisoformat(p["window_end"]) > lower for p in positions):
                raise ValueError("WAITING_FOR_WINDOW_END")
        raw, receipt = fetch_public_resolution(market_id)
        self.observations.save("paper_resolution_receipt", {"receipts": [receipt],
            "market_id": market_id, "clock_evidence_sha256": self.clock["receipt_sha256"]})
        projected = dict(receipt, received_at_local=receipt["received_at"],
            received_at=project_local(receipt["received_at"], self.clock).isoformat(),
            time_basis="CALIBRATED_UTC_UPPER_BOUND")
        return raw, projected

    def poll_settlements(self, policy):
        from .execution import PaperExecutor
        if time.monotonic()-self.last_settlement_poll <= 30:
            return
        self.last_settlement_poll = time.monotonic()
        before_positions = {p["market_id"]: p for p in self.account.overview()["positions"]}
        if not before_positions:
            return
        self.update_clock()
        result = PaperExecutor(self.account, policy).settle_observed(
            now=lambda: calibrated_now(self.clock), fetcher=self.fetch_resolution)
        self.observations.save("paper_settlements", result)
        if result.get("settled"):
            try:
                from scripts.analyze_argon4_settlement_alignment import (
                    compute_settlement_alignment_row,
                    summarize_settlement_alignments,
                )
                raw_root = self.runtime / "features/raw_data"
                rows = []
                with self.account._connection() as con:
                    for item in result["settled"]:
                        mid = item.get("market_id")
                        res_obj = item.get("resolution") if isinstance(item.get("resolution"), dict) else {}
                        outcome = res_obj.get("outcome") or item.get("winning_outcome") or item.get("outcome")
                        order_row = con.execute(
                            "SELECT signal_id, underlying, outcome, window_end, payload FROM orders WHERE market_id=? ORDER BY fill_at ASC LIMIT 1",
                            (mid,),
                        ).fetchone()
                        sig_payload, order_payload = {}, {}
                        if order_row is not None:
                            order_payload = json.loads(order_row["payload"]) if order_row["payload"] else {}
                            sig_row = con.execute("SELECT payload FROM signals WHERE signal_id=?", (order_row["signal_id"],)).fetchone()
                            if sig_row and sig_row["payload"]:
                                sig_payload = json.loads(sig_row["payload"])
                        pos = before_positions.get(mid, {})
                        venue_mkt = order_payload.get("venue_market") if isinstance(order_payload.get("venue_market"), dict) else {}
                        stream = sig_payload.get("stream") or item.get("stream") or pos.get("stream")
                        w_start = sig_payload.get("window_start") or venue_mkt.get("window_start") or item.get("window_start") or pos.get("window_start")
                        w_end = (sig_payload.get("window_end") or venue_mkt.get("window_end")
                                 or (order_row["window_end"] if order_row is not None else None)
                                 or item.get("window_end") or pos.get("window_end"))
                        pred_side = (order_row["outcome"] if order_row is not None else None) or sig_payload.get("outcome") or pos.get("outcome")
                        if not stream and w_start and w_end and (order_row is not None or pos.get("underlying")):
                            und = (order_row["underlying"] if order_row is not None else None) or pos.get("underlying")
                            dur_min = int((pd.Timestamp(w_end) - pd.Timestamp(w_start)).total_seconds() // 60)
                            stream = f"{und}_{dur_min}m"
                        if stream and w_start and w_end and outcome in ("UP", "DOWN"):
                            rows.append(compute_settlement_alignment_row(
                                raw_root, stream=stream, window_start=w_start, window_end=w_end,
                                official_outcome=outcome, market_id=str(mid), prediction_side=pred_side))
                if rows:
                    self.observations.save("settlement_label_alignment", {
                        "rows": rows, "summary": summarize_settlement_alignments(rows)})
            except Exception:
                pass

    def reconcile_inputs(self, collector, inputs):
        if not any("CONFLICTING_CLOSED_OBSERVATION" in e.get("error", "")
                   for e in inputs.get("errors", [])):
            return None
        from .revisions import reconcile_confirmed_price_revisions
        self.update_clock()
        kwargs = {}
        if any(e.get("source", "").startswith("mark:") for e in inputs["errors"]):
            kwargs["mark_registry_path"] = ROOT / "archive/catalog/mark_candle_registry.json"
        with collector._lock:
            result = reconcile_confirmed_price_revisions(
                self.runtime, inputs["errors"], now=lambda: calibrated_now(self.clock),
                cutover_now=self.revision_cutover_now, **kwargs)
        self.observations.save("input_revision_reconciliation", result)
        return result

    def revision_cutover_now(self):
        # Archiving an audit database can exceed the clock's 60-second TTL.
        # Obtain fresh evidence after that work, before certifying a new view.
        self.update_clock()
        return calibrated_now(self.clock)

    def _cohort_pristine(self, cohort):
        """Durable evidence, not the in-memory last_cohort flag, gates repair."""
        if self._pending_execution is not None:
            raise ValueError("REVISION_COHORT_ALREADY_PREPARED")
        stamp = cohort.isoformat()
        with sqlite3.connect(Path(self.account.path).resolve().as_uri()+"?mode=ro", uri=True,
                             timeout=.1) as db:
            for table in ("decision_cohorts", "cohort_streams", "cohort_book_looks"):
                if db.execute(f"SELECT 1 FROM {table} WHERE cohort_id=? LIMIT 1", (stamp,)).fetchone():
                    raise ValueError("REVISION_COHORT_ALREADY_USED")
            for table in ("signals", "orders"):
                if db.execute("SELECT 1 FROM "+table+" WHERE json_extract(payload,'$.window_start')=? LIMIT 1",
                              (stamp,)).fetchone():
                    raise ValueError("REVISION_COHORT_ALREADY_USED")
            if db.execute("SELECT 1 FROM orders WHERE json_extract(payload,'$.allocation_authorization.cohort_id')=? "
                          "OR signal_id IN (SELECT signal_id FROM signals WHERE json_extract(payload,'$.window_start')=?) "
                          "LIMIT 1", (stamp, stamp)).fetchone():
                raise ValueError("REVISION_COHORT_ALREADY_USED")
        with sqlite3.connect(Path(self.observations.path).resolve().as_uri()+"?mode=ro", uri=True,
                             timeout=.1) as db:
            if db.execute("SELECT 1 FROM snapshots WHERE kind IN ('prediction_decisions','paper_execution',"
                          "'paper_execution_abandoned') AND json_extract(payload,'$.cohort')=? LIMIT 1",
                          (stamp,)).fetchone():
                raise ValueError("REVISION_COHORT_ALREADY_PREDICTED")
        return {"eligible": True, "cohort": stamp, "evidence": "NO_DURABLE_PREDICTION_SIGNAL_ORDER_OR_LOOK"}

    def _revision_guard(self, policy, cohort, deadline, clock):
        if self.stop.is_set() or (self.runtime/"verification_pause.json").exists():
            raise ValueError("REVISION_EXECUTION_CANCELLED")
        if self.clock != clock:
            raise ValueError("REVISION_CLOCK_CHANGED")
        if cohort_at(calibrated_now(clock, upper=False)) != cohort or calibrated_now(clock) >= deadline:
            raise ValueError("REVISION_INPUT_DEADLINE_EXCEEDED")
        current = json.loads((ROOT/"config/argon4_forward_policy.json").read_text(encoding="utf-8"))
        if current != policy:
            raise ValueError("REVISION_POLICY_CHANGED")
        evidence = self._cohort_pristine(cohort)
        evidence.update(clock_evidence_sha256=clock["receipt_sha256"],
                        policy_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest())
        return evidence

    def reconcile_current_cohort(self, policy, engineering, cohort, inputs):
        """One bounded aged-bar repair; fresh validation precedes any scoring."""
        from .revisions import reconcile_confirmed_price_revisions, ensure_view_committed
        if not self._revision_fastpath.get("enabled"):
            return inputs, None
        errors = inputs.get("errors", [])
        allow_hourly = self._revision_fastpath.get("profile") == "argon4-aged-revision-v14"
        def _supported_fast_source(source):
            text = str(source or "")
            if text.startswith("futures:") and text.endswith(":5m"):
                return True
            if allow_hourly and (text.startswith("futures:") or text.startswith("mark:")) and text.endswith(":1h"):
                return True
            return False
        if not errors or any(e.get("error") not in (
                "CONFLICTING_CLOSED_OBSERVATION", "ValueError:CONFLICTING_CLOSED_OBSERVATION")
                or not _supported_fast_source(e.get("source")) for e in errors):
            return inputs, None
        deadline = cohort+timedelta(seconds=engineering["input_deadline_after_boundary_seconds"])
        clock = dict(self.clock)
        guard = lambda: self._revision_guard(policy, cohort, deadline, clock)
        context = {"profile": self._revision_fastpath["profile"], "cohort": cohort.isoformat(),
                   "deadline": deadline.isoformat(), "minimum_spacing_seconds": 2.0,
                   "clock_evidence_sha256": clock["receipt_sha256"]}
        revision = None
        self._same_cohort_revision = None
        try:
            for attempt in range(2):
                guard()
                remaining = (deadline-calibrated_now(clock)).total_seconds()
                # Retired input jobs must never race the archived working view.
                if not self.inputs._lock.acquire(timeout=max(0, min(remaining, .1))):
                    raise ValueError("REVISION_INPUT_WRITER_BUSY")
                try:
                    if any(not f.done() for f in getattr(self.inputs, "_outstanding", ())):
                        raise ValueError("REVISION_INPUT_WORKERS_BUSY")
                    extra_kwargs = ({"mark_registry_path": ROOT / "archive/catalog/mark_candle_registry.json"}
                                    if any(str(e.get("source", "")).startswith("mark:") for e in inputs["errors"]) else {})
                    revision = reconcile_confirmed_price_revisions(self.runtime, inputs["errors"],
                        now=lambda: calibrated_now(clock), cutover_now=lambda: calibrated_now(clock),
                        fast_context=context, cohort_guard=guard, **extra_kwargs)
                finally:
                    self.inputs._lock.release()
                self.observations.save("input_revision_reconciliation", revision)
                if revision.get("status") == "COMMITTED":
                    break
                if (attempt or revision.get("status") != "WAITING_CONFIRMATION"
                        or revision.get("fast_confirmation_eligible") is not True):
                    return inputs, revision
                if calibrated_now(clock)+timedelta(seconds=2) >= deadline:
                    return inputs, revision
                if self.stop.wait(2):
                    raise ValueError("STOPPED_DURING_REVISION_CONFIRMATION")
                guard()
                inputs = self.refresh_decision_inputs(engineering, cohort, calibrated_now(clock, upper=False))
                self.observations.save("revision_confirmation_inputs", inputs)
                if inputs.get("status") in ("READY", "COMPLETE", "OBSERVED"):
                    # Conflicting responses which reverted do not authorize a
                    # silent repair or a decision from an unstable source.
                    raise ValueError("REVISION_RESPONSE_REVERTED_DURING_CONFIRMATION")
            if not revision or revision.get("same_cohort_eligible") is not True:
                return inputs, revision
            guard()
            view = ensure_view_committed(self.runtime)
            if any(view.get(k) != revision.get(k) for k in ("revision_id", "manifest_sha256")):
                raise ValueError("REVISION_VIEW_CHANGED_BEFORE_REFRESH")
            inputs = self.refresh_decision_inputs(engineering, cohort, calibrated_now(clock, upper=False))
            self.observations.save("post_revision_inputs", inputs)
            guard()
            if inputs.get("status") not in ("READY", "COMPLETE", "OBSERVED"):
                return inputs, revision
            effective = datetime.fromisoformat(view["view_effective_at"].replace("Z", "+00:00"))
            available = datetime.fromisoformat(inputs["feature_available_at"].replace("Z", "+00:00"))
            inputs = {**inputs, "feature_available_at": max(effective, available).isoformat(),
                      "revision_id": view["revision_id"], "revision_manifest_sha256": view["manifest_sha256"],
                      "revision_view_effective_at": view["view_effective_at"]}
            self._same_cohort_revision = {"cohort": cohort, "revision_id": view["revision_id"],
                                          "manifest_sha256": view["manifest_sha256"], "clock": clock}
            return inputs, revision
        except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
            self._same_cohort_revision = None
            blocked = {**inputs, "status": "BLOCKED", "errors": [*inputs.get("errors", []),
                       {"source": "revision_fastpath", "error": type(exc).__name__+":"+str(exc)[:120]}]}
            self.observations.save("revision_fastpath_blocked", {"cohort": cohort.isoformat(),
                "reason": blocked["errors"][-1]["error"]})
            return blocked, revision or {"status": "BLOCKED", "reason": "FAST_REVISION_NOT_ADMITTED"}

    def _same_cohort_view_admitted(self, view):
        admission = self._same_cohort_revision
        if not admission or self.clock != admission["clock"]:
            return False
        return (cohort_at(calibrated_now(self.clock, upper=False)) == admission["cohort"]
                and self.last_cohort == admission["cohort"]
                and all(view.get(k) == admission[k] for k in ("revision_id", "manifest_sha256")))

    def wait_for_window(self, models):
        if (self._worker_engineering.get("profile") == V12_PROFILE_ID
                and self.state.get("status") == "PAPER_WAITING_WINDOW"
                and time.monotonic()-self._last_idle_publish < .8):
            return
        self._last_idle_publish = time.monotonic()
        if self.state.get("inputs", {}).get("status") == "BLOCKED":
            revision = self.state.get("revision") or {}
            maint_res = (self.state.get("maintenance") or {}).get("result") or {}
            errors = self.state.get("inputs", {}).get("errors", [])
            deadline_block = any("DEADLINE" in str(e) or "BUSY" in str(e) for e in errors)
            reason = ("输入维护检查已就绪；本窗口不补单，等待下一实时决策窗口重新核验"
                      if maint_res.get("status") == "READY_FOR_FUTURE_WINDOW" else
                      "价格修订已保留旧版本并重复确认；等待下一窗口重新检查输入"
                      if revision.get("status") == "COMMITTED" or maint_res.get("status") == "COMMITTED" else
                      "采集超过本轮时间预算，已跳过；等待下一实时窗口，不补历史订单" if deadline_block else
                      "模型输入存在缺口或观测冲突；当前暂停新单，等待后续数据检查")
            self.publish(status="PAPER_DATA_BLOCKED", active=True, models=models, reason=reason)
        else:
            self.publish(status="PAPER_WAITING_WINDOW", active=True, models=models,
                         reason="等待下一个实时决策窗口；不补发历史订单")

    def prepare_decisions(self, policy, inputs, cohort):
        from .model_v14 import build_feature_frames_v14 as build_feature_frames, predict_stream_v14 as predict_stream
        profile = stage_profile(policy, cohort)
        now_lower = calibrated_now(self.clock, upper=False)
        now_upper = calibrated_now(self.clock)
        available = (datetime.fromisoformat(inputs["feature_available_at"].replace("Z", "+00:00"))
                     if inputs.get("time_basis") == "CALIBRATED_UTC_UPPER_BOUND"
                     else project_local(inputs["feature_available_at"], self.clock))
        if available > now_lower:
            delay = (available-now_lower).total_seconds()+.01
            if delay > 2 or self.stop.wait(delay):
                raise ValueError("feature receipt lower-bound timing not ready")
            now_lower = calibrated_now(self.clock, upper=False)
            now_upper = calibrated_now(self.clock)
            if available > now_lower:
                raise ValueError("feature receipt lower-bound timing not ready")
        if now_upper >= cohort + timedelta(seconds=20):
            raise ValueError("missed feature decision window")
        frames = {}
        decisions = {}
        self._decision_timing = {"features_seconds": {}, "inference_seconds": {}}
        snapshot = None
        if engineering_profile(policy)["feature_snapshot_cache"]:
            from .feature_cache import FeatureSnapshot
            capture_started = time.perf_counter()
            snapshot = FeatureSnapshot(inputs["data_root"], as_of=now_lower,
                                       expected_sources=inputs.get("sources"))
            self._decision_timing["snapshot_capture_seconds"] = time.perf_counter()-capture_started
        try:
            return self._score_decisions(policy, inputs, cohort, profile, now_lower,
                                         available, frames, decisions, snapshot,
                                         build_feature_frames, predict_stream)
        finally:
            if snapshot is not None:
                self._decision_timing["snapshot"] = snapshot.stats
                snapshot.close()

    def _score_decisions(self, policy, inputs, cohort, profile, now_lower,
                         available, frames, decisions, snapshot,
                         build_feature_frames, predict_stream):
        from .model_v14 import apply_volatility_regime_guard, compute_ex_ante_rv12_bps
        vol_guard = (policy.get("volatility_regime_guard")
                     if isinstance(policy.get("volatility_regime_guard"), dict) else None)
        rv12_cache = {}
        for stream in policy["order"]:
            if stream in profile["disabled_streams"]:
                decisions[stream] = disabled_decision(profile)
                continue
            underlying, timeframe = stream.split("_")
            duration = int(timeframe[:-1])*60
            if int(cohort.timestamp()) % duration:
                decisions[stream] = {"status": "NOT_DUE"}
                continue
            if calibrated_now(self.clock) >= cohort + timedelta(seconds=20):
                raise ValueError("missed feature decision window")
            symbol = underlying + "USDT"
            if symbol not in frames:
                started = time.perf_counter()
                frames[symbol] = (snapshot.build(symbol, tail_rows=1) if snapshot is not None else
                                  build_feature_frames(inputs["data_root"], symbol,
                                                       as_of=now_lower, tail_rows=1))
                self._decision_timing["features_seconds"][symbol] = time.perf_counter()-started
            if calibrated_now(self.clock) >= cohort + timedelta(seconds=20):
                raise ValueError("missed feature decision window")
            f = frames[symbol]
            if f["base"].index[-1].to_pydatetime() + timedelta(minutes=5) != cohort:
                raise ValueError("closed feature row does not match decision cohort")
            if symbol not in rv12_cache:
                rv12_cache[symbol] = compute_ex_ante_rv12_bps(
                    inputs.get("data_root"), symbol, as_of=now_lower, snapshot=snapshot)
            started = time.perf_counter()
            threshold_override = (policy.get("stream_thresholds", {}).get(stream)
                                  if isinstance(policy.get("stream_thresholds"), dict) else None)
            score_kwargs = {"bundle": self.bundles[stream]}
            if threshold_override is not None:
                score_kwargs["threshold_override"] = threshold_override
            score = predict_stream(stream, f["base"], f["expert"], calibrated_now(self.clock), **score_kwargs)
            if rv12_cache[symbol] is not None or vol_guard is not None:
                score = apply_volatility_regime_guard(
                    stream, score, ex_ante_rv12_bps=rv12_cache[symbol], guard_config=vol_guard)
            self._decision_timing["inference_seconds"][stream] = time.perf_counter()-started
            if calibrated_now(self.clock) >= cohort + timedelta(seconds=20):
                raise ValueError("missed feature decision window")
            decisions[stream] = {"status": "NO_SIGNAL", "prediction": score, "outcome": score["side"]}
            if score["selected"]:
                decision_time = calibrated_now(self.clock)
                identity = f"{score['model_sha256']}:{stream}:{cohort.isoformat()}"
                signal = {"signal_id": hashlib.sha256(identity.encode()).hexdigest(), "stream": stream,
                          "underlying": underlying, "outcome": score["side"],
                          "model_id": score["model_version"], "model_sha256": score["model_sha256"],
                          "validation_status": "INPUTS_VALIDATED", "feature_available_at": available.isoformat(),
                          "decision_at": decision_time.isoformat(), "valid_until": (cohort+timedelta(seconds=20)).isoformat(),
                          "window_start": cohort.isoformat(), "window_end": (cohort+timedelta(seconds=duration)).isoformat()}
                decisions[stream].update(status="SIGNAL", signal=signal)
        return decisions

    def set_input_clock(self, clock=None):
        # A generation receives an immutable calibration. Retired requests must
        # never inherit a subsequent generation's clock or evidence identifier.
        clock = dict(self.clock if clock is None else clock)
        frozen_now = lambda: calibrated_now(clock)
        self.inputs.clock = frozen_now
        self.inputs.client.clock = frozen_now
        self.inputs.clock_evidence_sha256 = clock["receipt_sha256"]
        self.inputs.client.clock_evidence_sha256 = clock["receipt_sha256"]
        return clock

    def _maintenance_guard(self, policy, cohort, clock, deadline, phase):
        """Admission proof for maintenance, never an execution authorization."""
        def permission_check():
            if (self.stop.is_set() or (self.runtime/"verification_pause.json").exists()
                    or (self.runtime/"maintenance_pause.json").exists()):
                raise ValueError("MAINTENANCE_STOPPED_OR_PAUSED")
            if (self._pending_execution is not None or self.last_cohort != cohort
                    or self.clock != clock):
                raise ValueError("MAINTENANCE_COHORT_OR_CLOCK_CHANGED")
            if json.loads((ROOT/"config/argon4_forward_policy.json").read_text(encoding="utf-8")) != policy:
                raise ValueError("MAINTENANCE_POLICY_CHANGED")
        permission_check()
        lower, upper = calibrated_now(clock, upper=False), calibrated_now(clock)
        if (cohort_at(lower) != cohort or cohort_at(upper) != cohort
                or lower < cohort+timedelta(seconds=70) or upper >= cohort+timedelta(seconds=180)
                or time.monotonic() >= deadline):
            raise ValueError("MAINTENANCE_PHASE_OR_BUDGET_EXPIRED")
        if any(not f.done() for f in getattr(self.inputs, "_outstanding", ())):
            raise ValueError("MAINTENANCE_RETIRED_WORKERS_PENDING")
        evidence = self._cohort_pristine(cohort)
        if evidence.get("eligible") is not True:
            raise ValueError("MAINTENANCE_COHORT_NOT_PRISTINE")
        permission_check()
        # Include durable pristine checks in the deadline/phase budget too.
        lower, upper = calibrated_now(clock, upper=False), calibrated_now(clock)
        if (cohort_at(lower) != cohort or cohort_at(upper) != cohort
                or lower < cohort+timedelta(seconds=70) or upper >= cohort+timedelta(seconds=180)
                or time.monotonic() >= deadline):
            raise ValueError("MAINTENANCE_PHASE_OR_BUDGET_EXPIRED")
        return {**evidence, "phase": phase, "lower_at": lower.isoformat(), "upper_at": upper.isoformat(),
                "clock_evidence_sha256": clock["receipt_sha256"],
                "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
                "execution_authorized": False}

    def _claim_maintenance(self, policy, cohort, clock, deadline):
        """Atomically append a bounded per-cohort claim in the observation log."""
        path = Path(self.observations.path).resolve()
        payload = {"cohort": cohort.isoformat(), "clock_evidence_sha256": clock["receipt_sha256"],
                   "claimed_at_upper": calibrated_now(clock).isoformat(),
                   "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
                   "maximum_attempts": 2, "purpose": "INPUT_MAINTENANCE_NO_EXECUTION"}
        with sqlite3.connect(path, timeout=max(.001, min(.1, deadline-time.monotonic()))) as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM snapshots WHERE kind='input_maintenance_claim' "
                               "AND json_extract(payload,'$.cohort')=?", (cohort.isoformat(),)).fetchone()[0]
            if count >= 2:
                return None
            if count == 1:
                prior = db.execute("SELECT payload FROM snapshots WHERE kind='input_maintenance_result' "
                    "AND json_extract(payload,'$.cohort')=? AND json_extract(payload,'$.sequence')=1 "
                    "ORDER BY id DESC LIMIT 1", (cohort.isoformat(),)).fetchone()
                if prior is None:
                    return None
                prior = json.loads(prior[0])
                not_before = prior.get("next_eligible_at")
                if (prior.get("result", {}).get("status") != "WAITING_CONFIRMATION"
                        or prior.get("policy_sha256") != payload["policy_sha256"] or not not_before
                        or calibrated_now(clock, upper=False) < datetime.fromisoformat(not_before)):
                    return None
            payload["sequence"] = count+1
            db.execute("INSERT INTO snapshots(kind,saved_at,payload) VALUES(?,?,?)", (
                "input_maintenance_claim", datetime.now(timezone.utc).isoformat(),
                json.dumps(payload, sort_keys=True, allow_nan=False)))
        return payload

    def maintain_blocked_inputs(self, policy, models, cohort):
        """At most two window-external legacy maintenance attempts; no replay."""
        from .revisions import ensure_view_committed, reconcile_confirmed_price_revisions
        pending = self._blocked_input_maintenance
        if (not self._revision_fastpath.get("enabled") or not pending or pending.get("done")
                or pending["cohort"] != cohort or pending["policy"] != policy
                or self.last_cohort != cohort or self._pending_execution is not None):
            return False
        clock = deepcopy(self.clock)
        lower, upper = calibrated_now(clock, upper=False), calibrated_now(clock)
        if (cohort_at(lower) != cohort or cohort_at(upper) != cohort
                or lower < cohort+timedelta(seconds=70) or upper >= cohort+timedelta(seconds=180)
                or (pending.get("next_eligible_at") is not None and lower < pending["next_eligible_at"])):
            return False
        # Phase ends at T+180; calibration is immutable for this entire attempt.
        deadline = min(time.monotonic()+14, clock["calibrated_monotonic"]+59,
                       time.monotonic()+(cohort+timedelta(seconds=180)-upper).total_seconds())
        if deadline <= time.monotonic():
            return False
        guard = lambda phase: self._maintenance_guard(policy, cohort, clock, deadline, phase)
        result, claim = None, None
        locked = False
        try:
            guard("BEFORE_CLAIM")
            ensure_view_committed(self.runtime)
            claim = self._claim_maintenance(policy, cohort, clock, deadline)
            if claim is None:
                pending["done"] = True
                return False
            # Claim is durable before network I/O; a failed attempt still counts.
            pending["attempts"] = claim["sequence"]
            guard("BEFORE_REFRESH")
            self.set_input_clock(clock)
            inputs = self.inputs.refresh(now=calibrated_now(clock, upper=False),
                deadline_monotonic=deadline,
                cohort_id="maintenance:"+cohort.isoformat()+":"+str(claim["sequence"]))
            self.observations.save("input_maintenance_refresh", {"cohort": cohort.isoformat(),
                "sequence": claim["sequence"], "inputs": inputs, "clock_evidence_sha256": clock["receipt_sha256"]})
            guard("AFTER_REFRESH")
            ready = (inputs.get("status") in ("READY", "COMPLETE", "OBSERVED")
                     and not inputs.get("errors") and not inputs.get("gaps"))
            if ready:
                result = {"status": "READY_FOR_FUTURE_WINDOW", "next_cohort_requires_fresh_validation": True}
            elif any("CONFLICTING_CLOSED_OBSERVATION" in str(e.get("error", ""))
                     for e in inputs.get("errors", []) if isinstance(e, dict)):
                # All actual errors are passed unchanged. Never use fast_context
                # here: confirmation still requires the legacy 60-second rule.
                locked = self.inputs._lock.acquire(timeout=max(0, min(.1, deadline-time.monotonic())))
                if not locked:
                    raise ValueError("MAINTENANCE_WRITER_BUSY")
                guard("BEFORE_RECONCILIATION")
                kwargs = ({"mark_registry_path": ROOT/"archive/catalog/mark_candle_registry.json"}
                          if any(str(e.get("source", "")).startswith("mark:")
                                 for e in inputs["errors"] if isinstance(e, dict)) else {})
                result = reconcile_confirmed_price_revisions(self.runtime, inputs["errors"],
                    now=lambda: calibrated_now(clock), cutover_now=lambda: calibrated_now(clock),
                    maintenance_guard=guard, **kwargs)
            else:
                result = {"status": "BLOCKED", "reason": "NON_REVISION_INPUT_FAILURE",
                          "errors": deepcopy(inputs.get("errors", [])), "gaps": deepcopy(inputs.get("gaps", []))}
        except (ValueError, TypeError, KeyError, OSError, sqlite3.Error) as exc:
            result = {"status": "BLOCKED", "reason": str(exc)[:200]}
        finally:
            if locked:
                self.inputs._lock.release()
        if claim is None:
            # Failed admission does not send requests or consume an attempt.
            return False
        pending["done"] = True
        if result.get("status") == "WAITING_CONFIRMATION" and claim["sequence"] < 2:
            try:
                guard("AFTER_WAITING_CONFIRMATION")
                # Completion upper bound is no earlier than any receipt in this
                # attempt. Next attempt's fresh lower bound must be >= this+60s.
                pending["next_eligible_at"] = calibrated_now(clock)+timedelta(seconds=60)
                pending["done"] = pending["next_eligible_at"] >= cohort+timedelta(seconds=180)
            except (ValueError, TypeError, KeyError):
                pass
        observation = {"cohort": cohort.isoformat(), "sequence": claim["sequence"], "result": result,
                       "policy_sha256": claim["policy_sha256"],
                       "clock_evidence_sha256": clock["receipt_sha256"], "current_cohort_skipped": True,
                       "next_eligible_at": (pending["next_eligible_at"].isoformat()
                                            if not pending["done"] else None),
                       "next_cohort_requires_fresh_validation": True, "execution_authorized": False}
        self.observations.save("input_maintenance_result", observation)
        label = {"COMMITTED": "输入修订已提交", "WAITING_CONFIRMATION": "输入修订仍待独立确认",
                 "READY_FOR_FUTURE_WINDOW": "输入维护检查已就绪"}.get(result.get("status"), "输入维护未通过")
        self.publish(status="PAPER_INPUT_MAINTENANCE", active=True, models=models, maintenance=observation,
                     reason=label+"；本窗口不补单，下个完整窗口重新核验")
        return True

    def prefetch_inputs(self, engineering, now_lower):
        if not engineering["auxiliary_prefetch"]:
            return
        next_cohort = cohort_at(now_lower)+timedelta(minutes=5)
        until_boundary = (next_cohort-calibrated_now(self.clock)).total_seconds()
        if (self._prefetch_cohort == next_cohort or not
                2 < until_boundary <= engineering["prefetch_lead_seconds"]):
            return
        self._prefetch_cohort = next_cohort
        self.set_input_clock()
        budget = min(engineering["prefetch_budget_seconds"], until_boundary-2)
        result = self.inputs.prefetch_auxiliary(
            now=now_lower, deadline_monotonic=time.monotonic()+budget,
            cohort_id="prefetch:"+next_cohort.isoformat())
        self.observations.save("input_prefetch", result)
        with self.lock:
            self.state["prefetch"] = {key: result.get(key) for key in
                ("status", "errors", "observed_at", "cohort_id", "duration_seconds")}

    @staticmethod
    def retryable_latest_candle(inputs, cohort):
        """Only a single not-yet-published 5m close is retryable, never gaps."""
        gaps = inputs.get("gaps") or []
        if inputs.get("errors") or not gaps or inputs.get("status") != "BLOCKED":
            return False
        allowed = {f"{symbol}:futures:5m" for symbol in ALL_SYMBOLS}
        for gap in gaps:
            if not isinstance(gap, dict) or gap.get("source") not in allowed or gap.get("error") != "LATEST_CLOSED_BAR_MISSING":
                return False
            info = inputs.get("sources", {}).get(gap["source"], {})
            try:
                latest = datetime.fromisoformat(info["latest_closed"].replace("Z", "+00:00"))
                if latest != cohort-timedelta(minutes=5) or info["rows"] < 4320:
                    return False
                if info.get("issues") != ["LATEST_CLOSED_BAR_MISSING"]:
                    return False
            except (KeyError, TypeError, ValueError):
                return False
        return True

    def refresh_decision_inputs(self, engineering, cohort, now_lower):
        self.set_input_clock()
        cutoff = cohort+timedelta(seconds=engineering.get("input_deadline_after_boundary_seconds", 14))
        if engineering["enabled"]:
            self._pending_timing["input_deadline_at"] = cutoff.isoformat()
        attempts = 0
        while True:
            attempts += 1
            if engineering["enabled"]:
                remaining = max(0, (cutoff-calibrated_now(self.clock)).total_seconds())
                inputs = self.inputs.refresh(now=now_lower,
                    deadline_monotonic=time.monotonic()+remaining, cohort_id=cohort.isoformat())
            else:
                inputs = self.inputs.refresh(now=now_lower)
            self._pending_timing["input_refresh_attempts"] = attempts
            if (engineering["profile"] != V12_PROFILE_ID
                    or not self.retryable_latest_candle(inputs, cohort) or attempts >= 24):
                return inputs
            interval = engineering["input_ready_retry_interval_seconds"]
            if calibrated_now(self.clock)+timedelta(seconds=interval) >= cutoff:
                return inputs
            self.observations.save("input_readiness_retry", {
                "cohort": cohort.isoformat(), "attempt": attempts,
                "observed_at": calibrated_now(self.clock).isoformat(), "gaps": inputs["gaps"],
                "reason": "LATEST_5M_CLOSE_NOT_YET_PUBLISHED", "retry_after_seconds": interval})
            if self.stop.wait(interval):
                return {**inputs, "errors": [{"source": "pipeline", "error": "STOPPED_DURING_INPUT_RETRY"}]}
            now_lower = calibrated_now(self.clock, upper=False)

    def _critical_window(self):
        if self._worker_engineering.get("profile") != V12_PROFILE_ID or not self.clock:
            return False
        try:
            lower, upper = calibrated_now(self.clock, upper=False), calibrated_now(self.clock)
            elapsed = (lower-cohort_at(lower)).total_seconds()
            return elapsed >= 299 or (upper-cohort_at(lower)).total_seconds() <= 20
        except ValueError:
            return False

    def _worker_delay(self):
        future_window_wait = (self._worker_engineering.get("profile") == V12_PROFILE_ID
                              and self.warmup_complete and self.state.get("status") in (
                                  "STAGE_WAITING_ACTIVATION", "REVISION_WAITING_FUTURE_WINDOW"))
        if self.state.get("active") or future_window_wait:
            return self._worker_engineering.get("window_poll_interval_seconds", 1) if self._critical_window() else 1
        return 10

    def _settlement_guard(self):
        """Leave 30 seconds before the boundary for bounded public GETs."""
        if self._worker_engineering.get("profile") != V12_PROFILE_ID or not self.clock:
            return False
        try:
            lower, upper = calibrated_now(self.clock, upper=False), calibrated_now(self.clock)
            elapsed = (lower-cohort_at(lower)).total_seconds()
            return elapsed >= 270 or (upper-cohort_at(lower)).total_seconds() <= 20
        except ValueError:
            return True

    def abandon_prepared(self, reason):
        pending = self._pending_execution
        if pending is None:
            return
        self._pending_execution = None
        outcome = {"status": reason, "fills": pending["fills"], "blocked": [reason], "live_enabled": False}
        self.observations.save("paper_execution_abandoned", {
            "cohort": pending["cohort"].isoformat(), "reason": reason, "result": outcome})
        timing = self.finish_timing(reason)
        self.monitor_log.log_cohort_blocked(
            pending["cohort"].isoformat(),
            reason,
            f"待重检或待执行状态中止: {reason}",
            prior_fills_count=len(pending.get("fills") or []),
        )
        with self.lock:
            self.state.update(execution=outcome, timing=timing)

    def execution_now(self, clock):
        if self.stop.is_set() or (self.runtime/"verification_pause.json").is_file():
            raise ValueError("EXECUTION_CANCELLED")
        keys = ("receipt_sha256", "calibrated_monotonic", "server_minus_local_seconds",
                "uncertainty_seconds", "calibrated_at_local")
        if any(self.clock.get(k) != clock.get(k) for k in keys):
            raise ValueError("EXECUTION_CLOCK_CHANGED")
        return calibrated_now(clock)

    def execute_prepared(self, policy, inputs, models, cohort, decisions, clock, prior_fills=None):
        """Reuse the exact score and clock within one live cohort; never rescore."""
        from .execution import PaperExecutor
        started = time.perf_counter()
        with self.monitor_log.phase("pipeline", "execute_priority_waterfall",
                                    cohort=cohort.isoformat(), slow_warn_seconds=5.0):
            outcome = PaperExecutor(self.account, policy).process_cohort(
                cohort.isoformat(), decisions, self.observations.latest("polymarket"),
                now=lambda: self.execution_now(clock), book_refresh=self.refresh_book,
                clock=clock, market_refresh=self.refresh_market)
        self._pending_timing["execution_seconds"] = self._pending_timing.get("execution_seconds", 0)+time.perf_counter()-started
        self._pending_timing["execution_passes"] = self._pending_timing.get("execution_passes", 0)+1
        self._pending_timing.setdefault("first_execution_completed_after_boundary_seconds", (calibrated_now(clock)-cohort).total_seconds())
        # A later pass can find prior fills in the immutable account; keep the
        # dashboard's cumulative cohort list without booking them a second time.
        fills = {fill["order_id"]: fill for fill in (prior_fills or [])+outcome.get("fills", [])}
        outcome["fills"] = list(fills.values())
        if outcome["status"] == "BLOCKED" and fills:
            outcome["status"] = "PARTIAL_BLOCKED"
        self.observations.save("paper_execution", {
            "cohort": cohort.isoformat(), "policy_version": policy["version"],
            "waterfall_semantics": policy.get("waterfall_semantics", "LEGACY_CAPACITY_GATE"),
            "clock_evidence_sha256": clock["receipt_sha256"],
            "completed_at": calibrated_now(clock).isoformat(), "result": outcome})
        waiting = outcome.get("status") == "WAITING_PRICE_RECHECK"
        if waiting:
            self._pending_execution = deepcopy({"policy": policy, "inputs": inputs, "models": models,
                "cohort": cohort, "decisions": decisions, "clock": clock,
                "next_retry_at": outcome["next_retry_at"], "fills": outcome["fills"]})
            timing = {k: v for k, v in self._pending_timing.items() if k != "started_monotonic"}
            timing.update(status="WAITING_PRICE_RECHECK", next_retry_at=outcome["next_retry_at"])
        else:
            self._pending_execution = None
            timing = self.finish_timing(outcome.get("status", "COMPLETE"))
        self.monitor_log.log_cohort_execution(cohort.isoformat(), policy, decisions, outcome, timing)
        self.publish(status="PAPER_WAITING_PRICE_RECHECK" if waiting else "PAPER_CYCLE_COMPLETE",
            active=True, reason="当前优先流价格未达限价，等待本窗口内重检；下层暂不占预算" if waiting else None,
            models=models, inputs=inputs, revision=None, decisions=decisions,
            last_cohort=cohort.isoformat(), last_attempt_cohort=cohort.isoformat(),
            execution=outcome, timing=timing)

    def resume_prepared(self, policy):
        pending = self._pending_execution
        if pending is None:
            return False
        now = calibrated_now(self.clock)
        reason = None
        if self.stop.is_set():
            reason = "STOPPED_DURING_RECHECK"
        elif policy != pending["policy"]:
            reason = "POLICY_CHANGED_DURING_WINDOW"
        elif now >= pending["cohort"]+timedelta(seconds=20):
            reason = "WINDOW_EXPIRED"
        elif self.clock != pending["clock"]:
            reason = "CLOCK_CHANGED_DURING_RECHECK"
        if reason:
            self.abandon_prepared(reason)
            self.publish(status="PAPER_CYCLE_COMPLETE", active=True,
                reason="本窗口重检已停止；不重算预测、不补历史订单",
                execution=self.state["execution"], timing=self.state["timing"])
            return True
        if now < datetime.fromisoformat(pending["next_retry_at"]):
            return True
        self.execute_prepared(pending["policy"], pending["inputs"], pending["models"],
            pending["cohort"], pending["decisions"], pending["clock"], pending["fills"])
        return True

    def _log_block_once(self, key, cohort, code, reason, **extra):
        if self._last_logged_block_key == key:
            return
        self._last_logged_block_key = key
        self.monitor_log.log_cohort_blocked(cohort, code, reason, **extra)

    def cycle(self):
        from .online import OnlineInputs
        from .revisions import ensure_view_committed
        self.monitor_log.heartbeat("pipeline")
        # Turning off new predictions or a model-pin failure must not freeze
        # previously recorded positions awaiting official final settlement.
        # Final official settlement is still required; move this read-only poll
        # outside the short decision/recheck window to reduce interference.
        # Each fetch checks this guard again; running GETs remain bounded by
        # their transport and entry still fails closed if scheduling is late.
        if not self._settlement_guard():
            with self.monitor_log.phase("pipeline", "poll_settlements", slow_warn_seconds=10.0):
                self.poll_settlements({})
        policy = self.load_ready()
        if policy is None:
            self.abandon_prepared("PAPER_DISABLED")
            return
        engineering = engineering_profile(policy)
        self._worker_engineering = engineering
        self._revision_fastpath = revision_fastpath_profile(policy)
        models = {s: {"version": b["manifest"]["model_version"], "sha256": b["manifest"]["model_sha256"]} for s,b in self.bundles.items()}
        if (self.runtime / "verification_pause.json").is_file():
            self.abandon_prepared("VERIFICATION_PAUSE")
            self._log_block_once(("VERIFICATION_PAUSE",), None, "VERIFICATION_PAUSE",
                                 "完整检查期间暂停新预测与输入更新")
            self.publish(status="VERIFICATION_PAUSE", active=False, models=models,
                         reason="完整检查期间暂停新预测与输入更新；市场观察和已有仓位官方结算继续")
            return
        revision_view = ensure_view_committed(self.runtime)
        backoff_path = self.runtime / "features/binance_backoff.json"
        if backoff_path.exists():
            backoff = json.loads(backoff_path.read_text(encoding="utf-8"))
            until = datetime.fromisoformat(backoff["blocked_until_utc"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) < until:
                self.abandon_prepared("PUBLIC_RATE_LIMIT_BACKOFF")
                self._log_block_once(("PUBLIC_RATE_LIMIT_BACKOFF", backoff.get("blocked_until_utc")),
                                     None, "PUBLIC_RATE_LIMIT_BACKOFF",
                                     "公开行情限流暂停；等待供应商解除限制", backoff=backoff)
                self.publish(status="PUBLIC_RATE_LIMIT_BACKOFF", active=False, models=models, backoff=backoff,
                             reason="公开行情限流暂停；本地模型已保存，等待供应商解除限制")
                return
        if revision_view.get("status") == "COMMITTED":
            self.update_clock()
            effective = datetime.fromisoformat(revision_view["view_effective_at"].replace("Z", "+00:00"))
            next_window = cohort_at(effective) + timedelta(minutes=5)
            current_cohort = cohort_at(calibrated_now(self.clock, upper=False))
            if (current_cohort < next_window
                    and not self._same_cohort_view_admitted(revision_view)):
                self.abandon_prepared("REVISION_CUTOVER_WINDOW")
                self._log_block_once(("REVISION_CUTOVER_WINDOW", current_cohort.isoformat(), revision_view.get("revision_id")),
                                     current_cohort.isoformat(), "REVISION_CUTOVER_WINDOW",
                                     "输入修订已归档并核验；切换完成所在窗口跳过",
                                     revision_id=revision_view.get("revision_id"))
                self.publish(status="REVISION_WAITING_FUTURE_WINDOW", active=False, models=models,
                             revision=revision_view, decisions={},
                             execution={"status": "NOT_EVALUATED", "fills": [], "blocked": ["REVISION_CUTOVER_WINDOW"]},
                             reason="输入修订已归档并核验；切换完成所在窗口跳过，等待下一个完整窗口重新检查")
                return
        if not self.warmup_complete:
            if time.monotonic()-self.last_warmup < 60:
                return
            self.last_warmup = time.monotonic()
            # Legacy cold preparation retains raw system times. The v13 path
            # uses one immutable calibrated epoch, bounded below its TTL: a
            # skewed host clock must not indefinitely prevent price-revision
            # maintenance. Never relabel old receipts or suppress diagnostics.
            collector_kwargs, refresh_kwargs = {}, {}
            active_symbols = _policy_symbols(policy, stage_profile(policy))
            if active_symbols != SYMBOLS:
                collector_kwargs["symbols"] = active_symbols
            if self._revision_fastpath.get("enabled"):
                self.update_clock()
                warmup_clock = deepcopy(self.clock)
                calibrated_now(warmup_clock)
                warmup_deadline = min(time.monotonic()+30,
                                      warmup_clock["calibrated_monotonic"]+59)
                if warmup_deadline <= time.monotonic():
                    raise ValueError("warmup clock budget exhausted")
                collector_kwargs.update(clock=lambda: calibrated_now(warmup_clock),
                    time_basis="CALIBRATED_UTC_UPPER_BOUND",
                    clock_evidence_sha256=warmup_clock["receipt_sha256"])
                refresh_kwargs = {"now": calibrated_now(warmup_clock, upper=False),
                                  "deadline_monotonic": warmup_deadline}
            collector = OnlineInputs(self.runtime, **collector_kwargs)
            try:
                with self.monitor_log.phase("pipeline", "warmup_refresh", slow_warn_seconds=15.0):
                    warmup = collector.refresh(**refresh_kwargs)
            finally:
                close = getattr(collector, "close", None)
                if close:
                    close()
            self.observations.save("model_inputs_warmup", warmup)
            self.warmup_complete = not warmup.get("gaps") and all(
                e.get("source") == "clock" and e.get("error") == "CLOCK_SKEW_OR_LATENCY"
                for e in warmup.get("errors", []))
            if self._revision_fastpath.get("enabled"):
                self.warmup_complete = (warmup.get("status") in ("READY", "COMPLETE", "OBSERVED")
                                       and not warmup.get("errors") and not warmup.get("gaps"))
                if self.warmup_complete:
                    # Cold collection is maintenance, never a late entry into
                    # the window it finishes in. Refresh again next window.
                    self.last_cohort = cohort_at(calibrated_now(warmup_clock))
            revision = self.reconcile_inputs(collector, warmup) if not self.warmup_complete else None
            if not self.warmup_complete:
                self.monitor_log.log_cohort_blocked(
                    None, "PAPER_WARMUP", "热身阶段模型输入尚未就绪，热身阶段不下单",
                    gaps=warmup.get("gaps"), errors=warmup.get("errors"),
                )
            else:
                self.monitor_log.record(
                    "INFO", "WARMUP_COMPLETED",
                    f"模型输入热身核验完成，等待下一实时决策窗口 (当前窗口={self.last_cohort.isoformat() if self.last_cohort else 'N/A'} 跳过不补单)",
                    active_symbols=list(active_symbols), status=warmup.get("status"),
                )
            self.publish(status="PAPER_WARMUP", active=False, models=models, inputs=warmup,
                         revision=revision, decisions={},
                         reason="正在补齐并核验模型真实输入；观测冲突须重复确认并保留旧版本，热身阶段不下单")
            return
        self.update_clock()
        now_lower = calibrated_now(self.clock, upper=False)
        profile = stage_profile(policy)
        if profile["configured"] and cohort_at(now_lower) < datetime.fromisoformat(profile["effective_from"]):
            self.abandon_prepared("STAGE_WAITING_ACTIVATION")
            self._log_block_once(("STAGE_WAITING_ACTIVATION", profile["effective_from"]),
                                 cohort_at(now_lower).isoformat(), "STAGE_WAITING_ACTIVATION",
                                 "新阶段等待指定的未来窗口启用", effective_from=profile["effective_from"])
            self.publish(status="STAGE_WAITING_ACTIVATION", active=False, models=models,
                         decisions={s: disabled_decision(profile) for s in profile["disabled_streams"]},
                         reason="新阶段等待指定的未来窗口启用；已有仓位继续按官方结果结算")
            return
        if self.resume_prepared(policy):
            return
        spacing = policy.get("input_http_min_spacing_seconds", 1.0)
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or not math.isfinite(spacing) or not .25 <= spacing <= 1:
            raise ValueError("unsupported live public-input request spacing")
        active_symbols = _policy_symbols(policy, profile)
        if self.inputs is not None and (self._input_spacing != spacing or getattr(self.inputs, "symbols", SYMBOLS) != active_symbols):
            close = getattr(self.inputs, "close", None)
            if close:
                close()
            self.inputs = None
        input_kwargs = {"clock": lambda: calibrated_now(self.clock),
                        "time_basis": "CALIBRATED_UTC_UPPER_BOUND",
                        "clock_evidence_sha256": self.clock["receipt_sha256"],
                        "http_min_spacing_seconds": spacing}
        if active_symbols != SYMBOLS:
            input_kwargs["symbols"] = active_symbols
        self.inputs = self.inputs or OnlineInputs(self.runtime, **input_kwargs)
        self._input_spacing = spacing
        cohort = cohort_at(now_lower)
        elapsed = (now_lower-cohort).total_seconds()
        earliest = minimum_decision_delay(policy, engineering)
        if not earliest <= elapsed <= 18 or self.last_cohort == cohort:
            if self.maintain_blocked_inputs(policy, models, cohort):
                return
            self.prefetch_inputs(engineering, now_lower)
            self.wait_for_window(models)
            return
        self.last_cohort = cohort
        self._blocked_input_maintenance = None
        self._same_cohort_revision = None
        self._decision_timing = {}
        self._pending_timing = {"cohort": cohort.isoformat(), "deadline_seconds": 20,
            "started_at": calibrated_now(self.clock).isoformat(), "started_monotonic": time.perf_counter(),
            "http_min_spacing_seconds": spacing, "policy_version": policy["version"]}
        started = time.perf_counter()
        with self.monitor_log.phase("pipeline", "refresh_decision_inputs",
                                    cohort=cohort.isoformat(), slow_warn_seconds=6.0):
            inputs = self.refresh_decision_inputs(engineering, cohort, now_lower)
        revision = None
        if inputs.get("status") not in ("READY", "COMPLETE", "OBSERVED"):
            with self.monitor_log.phase("pipeline", "reconcile_current_cohort",
                                        cohort=cohort.isoformat(), slow_warn_seconds=6.0):
                inputs, revision = self.reconcile_current_cohort(policy, engineering, cohort, inputs)
        self._pending_timing["input_refresh_seconds"] = time.perf_counter()-started
        self.observations.save("model_inputs", inputs)
        if inputs.get("status") not in ("READY", "COMPLETE", "OBSERVED"):
            self._blocked_input_maintenance = {"cohort": cohort, "policy": deepcopy(policy),
                "trigger": "INPUTS_NOT_VALIDATED", "attempts": 0, "done": False}
            if revision is None:
                revision = self.reconcile_inputs(self.inputs, inputs)
            elif (revision.get("status") != "COMMITTED" and self._revision_fastpath.get("enabled")
                    and not self.stop.is_set() and not (self.runtime/"verification_pause.json").exists()
                    and json.loads((ROOT/"config/argon4_forward_policy.json").read_text(encoding="utf-8")) == policy):
                # This cohort is already abandoned. Preserve legacy 60-second
                # maintenance when a large audit archive cannot fit the fast
                # budget, so it cannot repeatedly stall every future window.
                # Keep ALL latest real errors; discard only our own diagnostic.
                maintenance = {**inputs, "errors": [e for e in inputs.get("errors", [])
                    if e.get("source") != "revision_fastpath"]}
                ensure_view_committed(self.runtime)  # never overwrite PREPARED
                repaired = self.reconcile_inputs(self.inputs, maintenance)
                if repaired is not None:
                    revision = repaired
            finished_timing = self.finish_timing("INPUTS_NOT_VALIDATED")
            self.monitor_log.log_cohort_blocked(
                cohort.isoformat(),
                "INPUTS_NOT_VALIDATED",
                "模型输入存在缺口或观测冲突，本轮跳过；已确认修订仅供未来窗口使用",
                errors=inputs.get("errors"),
                gaps=inputs.get("gaps"),
                revision_status=(revision or {}).get("status"),
                input_refresh_seconds=self._pending_timing.get("input_refresh_seconds") if self._pending_timing else None,
            )
            self.publish(status="PAPER_DATA_BLOCKED", active=True, inputs=inputs, models=models,
                         revision=revision, decisions={}, last_attempt_cohort=cohort.isoformat(),
                         timing=finished_timing,
                         execution={"status": "NOT_EVALUATED", "fills": [], "blocked": ["INPUTS_NOT_VALIDATED"]},
                         reason="模型输入存在缺口或观测冲突，本轮跳过；已确认修订仅供未来窗口使用")
            return
        started = time.perf_counter()
        with self.monitor_log.phase("pipeline", "register_online_marks",
                                    cohort=cohort.isoformat(), slow_warn_seconds=3.0):
            registered = register_online_marks(inputs["data_root"])
        self._pending_timing["mark_registration_seconds"] = time.perf_counter()-started
        self._pending_timing["mark_registration"] = registered
        ticker_cfg = policy.get("early_window_book_ticker")
        if isinstance(ticker_cfg, dict) and ticker_cfg.get("enabled") is True:
            capture_fn = getattr(self.inputs, "capture_early_window_book_ticker", None)
            if callable(capture_fn):
                try:
                    self.set_input_clock()
                    now_upper = calibrated_now(self.clock)
                    remaining_window = (cohort + timedelta(seconds=20) - now_upper).total_seconds()
                    max_budget = float(ticker_cfg.get("max_budget_seconds", 0.8))
                    if remaining_window > 6.5 and max_budget > 0:
                        ticker_deadline = time.monotonic() + min(max_budget, remaining_window - 6.0)
                        snap = capture_fn(
                            now=calibrated_now(self.clock, upper=False),
                            cohort_id=cohort.isoformat(),
                            deadline_monotonic=ticker_deadline,
                        )
                        self.observations.save("early_window_book_ticker", snap)
                except Exception as exc:
                    self.observations.save("early_window_book_ticker", {
                        "status": "ERROR", "cohort_id": cohort.isoformat(),
                        "error": f"{type(exc).__name__}:{str(exc)[:160]}"})
        started = time.perf_counter()
        with self.monitor_log.phase("pipeline", "prepare_decisions",
                                    cohort=cohort.isoformat(), slow_warn_seconds=5.0):
            decisions = self.prepare_decisions(policy, inputs, cohort)
        self._pending_timing["decisions_seconds"] = time.perf_counter()-started
        self.observations.save("prediction_decisions", {"cohort": cohort.isoformat(), "decisions": decisions})
        # The observation worker discovers current AND next contracts. Reuse its
        # identity/fee evidence within the executor's 90-second metadata limit;
        # refresh executable books immediately through the isolated callback.
        started = time.perf_counter()
        if self.stop.wait(2*self.clock["uncertainty_seconds"]+.01):
            self.finish_timing("STOPPED_BEFORE_EXECUTION")
            return
        self._pending_timing["causal_wait_seconds"] = time.perf_counter()-started
        self.execute_prepared(policy, inputs, models, cohort, decisions, dict(self.clock))

    def worker(self):
        self.monitor_log.start_watchdog()
        try:
            while not self.stop.is_set():
                try:
                    self.cycle()
                except Exception as exc:
                    self.abandon_prepared("CHECK_FAILED_DURING_RECHECK")
                    self.monitor_log.log_cohort_blocked(
                        self.last_cohort.isoformat() if self.last_cohort else None,
                        "PIPELINE_CYCLE_EXCEPTION",
                        f"本地模拟检查未通过：{type(exc).__name__}: {exc}",
                        error_type=type(exc).__name__,
                        error=str(exc)[:240],
                    )
                    self.publish(status="PAPER_BLOCKED", active=False,
                                 timing=self.finish_timing("FAILED", exc),
                                 reason="本地模拟检查未通过："+type(exc).__name__)
                self.stop.wait(self._worker_delay())
        finally:
            self.close()

    def close(self):
        self.monitor_log.close()
        close = getattr(self.inputs, "close", None)
        if close:
            close()
        self.public_transport.close()
