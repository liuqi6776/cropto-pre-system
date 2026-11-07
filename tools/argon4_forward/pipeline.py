"""Local, credential-free orchestration of saved models and paper execution.

Never replay a missed decision window or count a historical backtest as cash.
The only execution class used here is the isolated SQLite paper executor.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time

from .clock_sync import calibrate, calibrated_now, project_local
from .stage import stage_profile, disabled_decision
from .runtime_profile import engineering_profile

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
_MARK_REGISTRY_LOCK = threading.RLock()
_MARK_REGISTRY_CACHE = {}


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
    originals = {symbol: (root / "auxiliary" / f"{symbol}_1h_mark.parquet") for symbol in SYMBOLS}
    raw_sources = {symbol: target.read_bytes() for symbol, target in originals.items() if target.exists()}
    source_hashes = tuple((symbol, hashlib.sha256(raw).hexdigest()) for symbol, raw in raw_sources.items())
    identity = (hashlib.sha256(registry_raw).hexdigest(), source_hashes)
    # Check complete bytes every time, not mtime/size. Any external edit,
    # revision or new bar invalidates the cache and gets full validation.
    if _MARK_REGISTRY_CACHE.get(str(root)) == identity:
        return {"status": "UNCHANGED_VERIFIED", "source_count": len(raw_sources)}
    registry = json.loads(registry_raw)
    refs = []
    for symbol in SYMBOLS:
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
    def __init__(self, account, runtime, observations):
        self.account, self.runtime, self.observations = account, Path(runtime), observations
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
        from .model import STREAMS, load_model_bundle
        policy = json.loads((ROOT / "config/argon4_forward_policy.json").read_text(encoding="utf-8"))
        if not policy.get("paper_trading_active") or policy.get("ranking_model_scope") != "LOCAL_SELF_TRAINED_OOS":
            self.publish(status="WAITING_LOCAL_BACKTEST", active=False, reason="本地训练与回测尚未完成，等待本地胜率排序")
            return None
        profile = stage_profile(policy)
        with self.lock:
            self.state["stage"] = profile
        for stream in STREAMS:
            # The seven saved artifacts and policy pins remain untouched. A
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
            self.clock = calibrate(transport=self.public_transport)
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
        from .polymarket import STREAMS, _get_public, validate_market
        asset, duration, _ = STREAMS[stream]
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
        if not self.account.overview()["positions"]:
            return
        self.update_clock()
        result = PaperExecutor(self.account, policy).settle_observed(
            now=lambda: calibrated_now(self.clock), fetcher=self.fetch_resolution)
        self.observations.save("paper_settlements", result)

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

    def wait_for_window(self, models):
        if self.state.get("inputs", {}).get("status") == "BLOCKED":
            revision = self.state.get("revision") or {}
            errors = self.state.get("inputs", {}).get("errors", [])
            deadline_block = any("DEADLINE" in str(e) or "BUSY" in str(e) for e in errors)
            reason = ("价格修订已保留旧版本并重复确认；等待下一窗口重新检查输入"
                      if revision.get("status") == "COMMITTED" else
                      "采集超过本轮时间预算，已跳过；等待下一实时窗口，不补历史订单" if deadline_block else
                      "模型输入存在缺口或观测冲突；当前暂停新单，等待后续数据检查")
            self.publish(status="PAPER_DATA_BLOCKED", active=True, models=models, reason=reason)
        else:
            self.publish(status="PAPER_WAITING_WINDOW", active=True, models=models,
                         reason="等待下一个实时决策窗口；不补发历史订单")

    def prepare_decisions(self, policy, inputs, cohort):
        from .model import build_feature_frames, predict_stream
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
            started = time.perf_counter()
            score = predict_stream(stream, f["base"], f["expert"], calibrated_now(self.clock), bundle=self.bundles[stream])
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

    def set_input_clock(self):
        # A generation receives an immutable calibration. Retired requests must
        # never inherit a subsequent generation's clock or evidence identifier.
        clock = dict(self.clock)
        frozen_now = lambda: calibrated_now(clock)
        self.inputs.clock = frozen_now
        self.inputs.client.clock = frozen_now
        self.inputs.clock_evidence_sha256 = clock["receipt_sha256"]
        self.inputs.client.clock_evidence_sha256 = clock["receipt_sha256"]

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

    def cycle(self):
        from .online import OnlineInputs
        from .execution import PaperExecutor
        from .revisions import ensure_view_committed
        # Turning off new predictions or a model-pin failure must not freeze
        # previously recorded positions awaiting official final settlement.
        self.poll_settlements({})
        policy = self.load_ready()
        if policy is None:
            return
        engineering = engineering_profile(policy)
        models = {s: {"version": b["manifest"]["model_version"], "sha256": b["manifest"]["model_sha256"]} for s,b in self.bundles.items()}
        if (self.runtime / "verification_pause.json").is_file():
            self.publish(status="VERIFICATION_PAUSE", active=False, models=models,
                         reason="完整检查期间暂停新预测与输入更新；市场观察和已有仓位官方结算继续")
            return
        revision_view = ensure_view_committed(self.runtime)
        backoff_path = self.runtime / "features/binance_backoff.json"
        if backoff_path.exists():
            backoff = json.loads(backoff_path.read_text(encoding="utf-8"))
            until = datetime.fromisoformat(backoff["blocked_until_utc"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) < until:
                self.publish(status="PUBLIC_RATE_LIMIT_BACKOFF", active=False, models=models, backoff=backoff,
                             reason="公开行情限流暂停；本地模型已保存，等待供应商解除限制")
                return
        if revision_view.get("status") == "COMMITTED":
            self.update_clock()
            effective = datetime.fromisoformat(revision_view["view_effective_at"].replace("Z", "+00:00"))
            next_window = cohort_at(effective) + timedelta(minutes=5)
            if cohort_at(calibrated_now(self.clock, upper=False)) < next_window:
                self.publish(status="REVISION_WAITING_FUTURE_WINDOW", active=False, models=models,
                             revision=revision_view, decisions={},
                             execution={"status": "NOT_EVALUATED", "fills": [], "blocked": ["REVISION_CUTOVER_WINDOW"]},
                             reason="输入修订已归档并核验；切换完成所在窗口跳过，等待下一个完整窗口重新检查")
                return
        if not self.warmup_complete:
            if time.monotonic()-self.last_warmup < 60:
                return
            self.last_warmup = time.monotonic()
            # Cold preparation can take over a minute: retain raw system receive
            # times, do not perform decisions, then obtain a fresh calibrated
            # input refresh at a subsequent live decision boundary.
            collector = OnlineInputs(self.runtime)
            try:
                warmup = collector.refresh()
            finally:
                close = getattr(collector, "close", None)
                if close:
                    close()
            self.observations.save("model_inputs_warmup", warmup)
            self.warmup_complete = not warmup.get("gaps") and all(
                e.get("source") == "clock" and e.get("error") == "CLOCK_SKEW_OR_LATENCY"
                for e in warmup.get("errors", []))
            revision = self.reconcile_inputs(collector, warmup) if not self.warmup_complete else None
            self.publish(status="PAPER_WARMUP", active=False, models=models, inputs=warmup,
                         revision=revision, decisions={},
                         reason="正在补齐并核验模型真实输入；观测冲突须重复确认并保留旧版本，热身阶段不下单")
            return
        self.update_clock()
        now_lower = calibrated_now(self.clock, upper=False)
        profile = stage_profile(policy)
        if profile["configured"] and cohort_at(now_lower) < datetime.fromisoformat(profile["effective_from"]):
            self.publish(status="STAGE_WAITING_ACTIVATION", active=False, models=models,
                         decisions={s: disabled_decision(profile) for s in profile["disabled_streams"]},
                         reason="新阶段等待指定的未来窗口启用；已有仓位继续按官方结果结算")
            return
        spacing = policy.get("input_http_min_spacing_seconds", 1.0)
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or not math.isfinite(spacing) or not .25 <= spacing <= 1:
            raise ValueError("unsupported live public-input request spacing")
        if self.inputs is not None and self._input_spacing != spacing:
            close = getattr(self.inputs, "close", None)
            if close:
                close()
            self.inputs = None
        self.inputs = self.inputs or OnlineInputs(self.runtime, clock=lambda: calibrated_now(self.clock),
                                                  time_basis="CALIBRATED_UTC_UPPER_BOUND",
                                                  clock_evidence_sha256=self.clock["receipt_sha256"],
                                                  http_min_spacing_seconds=spacing)
        self._input_spacing = spacing
        executor = PaperExecutor(self.account, policy)
        cohort = cohort_at(now_lower)
        elapsed = (now_lower-cohort).total_seconds()
        earliest = float(policy.get("min_decision_delay_seconds", 5))
        if not 5 <= earliest <= 10:
            raise ValueError("unsupported minimum decision delay")
        if not earliest <= elapsed <= 18 or self.last_cohort == cohort:
            self.prefetch_inputs(engineering, now_lower)
            self.wait_for_window(models)
            return
        self.last_cohort = cohort
        self._decision_timing = {}
        self._pending_timing = {"cohort": cohort.isoformat(), "deadline_seconds": 20,
            "started_at": calibrated_now(self.clock).isoformat(), "started_monotonic": time.perf_counter(),
            "http_min_spacing_seconds": spacing, "policy_version": policy["version"]}
        self.set_input_clock()
        started = time.perf_counter()
        if engineering["enabled"]:
            input_cutoff = cohort+timedelta(seconds=engineering["input_deadline_after_boundary_seconds"])
            remaining = max(0, (input_cutoff-calibrated_now(self.clock)).total_seconds())
            self._pending_timing["input_deadline_at"] = input_cutoff.isoformat()
            inputs = self.inputs.refresh(now=now_lower,
                deadline_monotonic=time.monotonic()+remaining, cohort_id=cohort.isoformat())
        else:
            inputs = self.inputs.refresh(now=now_lower)
        self._pending_timing["input_refresh_seconds"] = time.perf_counter()-started
        self.observations.save("model_inputs", inputs)
        if inputs.get("status") not in ("READY", "COMPLETE", "OBSERVED"):
            revision = self.reconcile_inputs(self.inputs, inputs)
            self.publish(status="PAPER_DATA_BLOCKED", active=True, inputs=inputs, models=models,
                         revision=revision, decisions={}, last_attempt_cohort=cohort.isoformat(),
                         timing=self.finish_timing("INPUTS_NOT_VALIDATED"),
                         execution={"status": "NOT_EVALUATED", "fills": [], "blocked": ["INPUTS_NOT_VALIDATED"]},
                         reason="模型输入存在缺口或观测冲突，本轮跳过；已确认修订仅供未来窗口使用")
            return
        started = time.perf_counter()
        registered = register_online_marks(inputs["data_root"])
        self._pending_timing["mark_registration_seconds"] = time.perf_counter()-started
        self._pending_timing["mark_registration"] = registered
        started = time.perf_counter()
        decisions = self.prepare_decisions(policy, inputs, cohort)
        self._pending_timing["decisions_seconds"] = time.perf_counter()-started
        self.observations.save("prediction_decisions", {"cohort": cohort.isoformat(), "decisions": decisions})
        # The observation worker discovers current AND next contracts. Reuse its
        # identity/fee evidence within the executor's 90-second metadata limit;
        # refresh executable books immediately through the isolated callback.
        markets = self.observations.latest("polymarket")
        started = time.perf_counter()
        if self.stop.wait(2*self.clock["uncertainty_seconds"]+.01):
            self.finish_timing("STOPPED_BEFORE_EXECUTION")
            return
        self._pending_timing["causal_wait_seconds"] = time.perf_counter()-started
        started = time.perf_counter()
        outcome = executor.process_cohort(cohort.isoformat(), decisions, markets,
                                           now=lambda: calibrated_now(self.clock),
                                           book_refresh=self.refresh_book, clock=self.clock,
                                           market_refresh=self.refresh_market)
        self._pending_timing["execution_seconds"] = time.perf_counter()-started
        self.observations.save("paper_execution", {
            "cohort": cohort.isoformat(), "policy_version": policy["version"],
            "waterfall_semantics": policy.get("waterfall_semantics", "LEGACY_CAPACITY_GATE"),
            "clock_evidence_sha256": self.clock["receipt_sha256"],
            "completed_at": calibrated_now(self.clock).isoformat(), "result": outcome})
        self.publish(status="PAPER_CYCLE_COMPLETE", active=True, reason=None, models=models,
                     inputs=inputs, revision=None, decisions=decisions, last_cohort=cohort.isoformat(),
                     last_attempt_cohort=cohort.isoformat(), execution=outcome,
                     timing=self.finish_timing(outcome.get("status", "COMPLETE")))

    def worker(self):
        try:
            while not self.stop.is_set():
                try:
                    self.cycle()
                except Exception as exc:
                    self.publish(status="PAPER_BLOCKED", active=False,
                                 timing=self.finish_timing("FAILED", exc),
                                 reason="本地模拟检查未通过："+type(exc).__name__)
                self.stop.wait(1 if self.state.get("active") else 10)
        finally:
            self.close()

    def close(self):
        close = getattr(self.inputs, "close", None)
        if close:
            close()
        self.public_transport.close()
