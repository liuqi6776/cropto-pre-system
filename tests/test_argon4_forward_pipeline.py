"""Offline integration: pinned saved-model scores become isolated paper fills.

Every price, feature and contract below is an explicit test fixture. No public
network is used and no fixture is installed into research or runtime datasets.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from types import SimpleNamespace
import threading
import time

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from tools.argon4_forward import model as model
from tools.argon4_forward import pipeline as pipeline
from tools.argon4_forward.account import Account, STREAMS
from tools.argon4_forward.execution import PaperExecutor


BOUNDARY = datetime(2026, 11, 6, tzinfo=timezone.utc)
ORDER = ["BTC_15m", "ETH_5m", "SOL_15m", "SOL_5m", "BTC_5m", "ETH_15m", "BNB_15m"]


def test_save_json_preserves_mtime_only_when_serialized_content_is_unchanged(tmp_path):
    path = tmp_path / "offline_registry.json"
    original = {"source": "测试观测", "rows": 12}
    pipeline.save_json(path, original)
    old_ns = 1_700_000_000_000_000_000
    os.utime(path, ns=(old_ns, old_ns))
    before = path.stat().st_mtime_ns
    before_bytes = path.read_bytes()
    pipeline.save_json(path, deepcopy(original))
    assert path.stat().st_mtime_ns == before
    assert path.read_bytes() == before_bytes
    changed = {**original, "rows": 13}
    pipeline.save_json(path, changed)
    assert path.stat().st_mtime_ns != before
    assert path.read_bytes() != before_bytes
    assert json.loads(path.read_text(encoding="utf-8")) == changed
    assert not path.with_suffix(".tmp").exists()


def test_v11_uses_one_snapshot_for_exact_saved_predictions_and_budgeted_inputs(harness, monkeypatch):
    from tools.argon4_forward import online, feature_cache
    from tools.argon4_forward.runtime_profile import V11_ENGINEERING
    h = harness
    enable_v5(h)
    h.policy["engineering"] = deepcopy(V11_ENGINEERING)
    h.policy_path.write_text(json.dumps(h.policy))
    captured = []
    old_refresh = online.OnlineInputs.refresh
    def budgeted(self, now=None, **kwargs):
        captured.append(kwargs)
        return old_refresh(self, now)
    monkeypatch.setattr(online.OnlineInputs, "refresh", budgeted)
    snapshots = []
    class Snapshot:
        def __init__(self, root, *, as_of, expected_sources):
            self.root, self.cutoff = root, as_of
            self.builds, self.closed = [], False
            snapshots.append(self)
        def build(self, symbol, tail_rows):
            self.builds.append(symbol)
            return model.build_feature_frames(self.root, symbol, as_of=self.cutoff, tail_rows=tail_rows)
        @property
        def stats(self):
            return {"symbols": self.builds[:]}
        def close(self):
            self.closed = True
    monkeypatch.setattr(feature_cache, "FeatureSnapshot", Snapshot)
    before = time.monotonic()
    h.p.cycle()
    assert len(captured) == 1
    assert captured[0]["cohort_id"] == BOUNDARY.isoformat()
    assert before+6.5 < captured[0]["deadline_monotonic"] < time.monotonic()+7.1
    assert len(snapshots) == 1 and snapshots[0].closed
    assert len(snapshots[0].builds) == len(set(snapshots[0].builds)) == 4
    assert h.p.status()["status"] == "PAPER_CYCLE_COMPLETE"
    for stream, decision in h.p.status()["decisions"].items():
        assert decision["prediction"]["z_v4"] == pytest.approx(h.trained.scores[stream]["score"], abs=1e-12)
    assert h.account.overview()["live_enabled"] is False


def test_v11_deadline_failure_never_scores_or_replays_same_cohort(harness, monkeypatch):
    from tools.argon4_forward import online
    from tools.argon4_forward.runtime_profile import V11_ENGINEERING
    h = harness
    h.policy["engineering"] = deepcopy(V11_ENGINEERING)
    h.policy_path.write_text(json.dumps(h.policy))
    calls = []
    def late(self, now=None, **kwargs):
        calls.append(kwargs)
        return {"status": "BLOCKED", "errors": [{"source": "batch", "error": "INPUT_DEADLINE_EXCEEDED"}]}
    monkeypatch.setattr(online.OnlineInputs, "refresh", late)
    h.p.cycle()
    h.p.cycle()
    assert len(calls) == 1
    assert h.state.feature_calls == []
    assert h.account.overview()["order_count"] == 0
    assert "时间预算" in h.p.status()["reason"]


def test_v11_prefetch_is_before_boundary_once_and_never_executes(harness, monkeypatch):
    from tools.argon4_forward import online
    from tools.argon4_forward.runtime_profile import V11_ENGINEERING
    h = harness
    h.policy["engineering"] = deepcopy(V11_ENGINEERING)
    h.policy_path.write_text(json.dumps(h.policy))
    h.state.upper = BOUNDARY+timedelta(seconds=292)
    calls = []
    def prefetch(self, now=None, **kwargs):
        calls.append((now, kwargs))
        return {"status": "PREFETCH_COMPLETE", "cohort_id": kwargs["cohort_id"]}
    monkeypatch.setattr(online.OnlineInputs, "prefetch_auxiliary", prefetch, raising=False)
    before = time.monotonic()
    h.p.cycle()
    h.p.cycle()
    assert len(calls) == 1
    assert calls[0][0] < BOUNDARY+timedelta(minutes=5)
    assert calls[0][1]["cohort_id"] == "prefetch:"+(BOUNDARY+timedelta(minutes=5)).isoformat()
    assert before+2.5 < calls[0][1]["deadline_monotonic"] <= time.monotonic()+3
    assert h.state.refreshes == 0 and h.state.feature_calls == []
    assert h.account.overview()["order_count"] == 0


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request",
                        lambda *a, **k: pytest.fail("offline integration attempted network"))


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline_models")
    rng = np.random.default_rng(481)
    x = rng.normal(size=(500, 3)).astype("float32")
    xd = rng.normal(size=(500, 2)).astype("float32")
    y = x[:, 0] * .001 + rng.normal(0, .0005, len(x))
    with threadpool_limits(1):
        base, _ = model._fit_family(x[:400], y[:400], x[400:])
        expert, _ = model._fit_family(xd[:400], y[:400], xd[400:], expert=True)
    package = {"base": base, "expert": expert, "base_columns": ["a", "b", "c"],
               "expert_columns": ["d", "e"], "weight": .15,
               "threshold_frozen": 0., "threshold_reset": 0.}
    index = pd.date_range(BOUNDARY - timedelta(minutes=5), periods=100, freq="5min")
    frames = {"base": pd.DataFrame(x[400:], index=index, columns=package["base_columns"]),
              "expert": pd.DataFrame(xd[400:], index=index, columns=package["expert_columns"])}
    pins, scores = {}, {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model, "MODEL_ROOT", root)
        reports = root / "offline_test_reports"
        reports.mkdir()
        for stream in ORDER:
            metadata = {"stream": stream, "phase": "deployment_q4", "classification": model.CLASSIFICATION,
                        "model_version": "offline-integration-v1", "fold": model.FOLDS["deployment_q4"],
                        "base_columns": package["base_columns"], "expert_columns": package["expert_columns"],
                        "sklearn_version": model.sklearn.__version__}
            pins[stream] = model._save_stream(root / "deployment_q4", stream, package, metadata)
            saved = model.load_model_bundle(stream, expected_manifest_sha256=pins[stream])
            # The real backtest path uses precisely this reloaded bundle.
            with threadpool_limits(1):
                summaries = model._evaluate(saved, stream, frames, np.arange(100), y[400:], reports)
            scored = pd.read_parquet(reports / f"{stream}_oos_predictions.parquet")
            scores[stream] = scored[scored.variant == "LOCAL_V4_FROZEN"].iloc[0].to_dict()
            assert next(r for r in summaries if r["variant"] == "LOCAL_V4_FROZEN")["manifest_sha256"] == pins[stream]
    return SimpleNamespace(root=root, pins=pins, frames=frames, scores=scores)


class MemoryObservations:
    def __init__(self):
        self.data = {}

    def save(self, kind, data):
        self.data[kind] = deepcopy(data)

    def latest(self, kind):
        return deepcopy(self.data.get(kind, {}))


@pytest.fixture
def harness(tmp_path, monkeypatch, trained_artifacts):
    from tools.argon4_forward import online
    monkeypatch.setattr(model, "MODEL_ROOT", trained_artifacts.root)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    state = SimpleNamespace(upper=BOUNDARY + timedelta(seconds=7), boundary=BOUNDARY,
                            capacity=100, feature_shift=0, metadata_age=30,
                            source_age=.1, feature_error=None, input_status="READY",
                            input_time_basis="CALIBRATED_UTC_UPPER_BOUND", refreshes=0,
                            input_errors=[], input_gaps=None, available_age=.3, refresh_delay=0,
                            book_calls=[], feature_calls=[], registered=[], waits=[])
    policy = {"status": "CONFIRMED", "version": "offline-integration-policy-v1", "order": ORDER[:],
              "ranking_source": "offline fixture saved-model backtest",
              "paper_trading_active": True, "ranking_model_scope": "LOCAL_SELF_TRAINED_OOS",
              "deployment_manifest_sha256": dict(trained_artifacts.pins)}
    (tmp_path / "config").mkdir()
    policy_path = tmp_path / "config/argon4_forward_policy.json"
    policy_path.write_text(json.dumps(policy))
    clock = {"status": "CALIBRATED", "source": "POLYMARKET_CLOB_SERVER_TIME",
             "uncertainty_seconds": .1, "server_minus_local_seconds": 5,
             "receipt_sha256": "d" * 64, "observed_at": BOUNDARY.isoformat(),
             "calibrated_monotonic": time.monotonic()}
    monkeypatch.setattr(pipeline, "calibrated_now", lambda c, upper=True:
                        state.upper if upper else state.upper - timedelta(seconds=.2))
    monkeypatch.setattr(pipeline, "register_online_marks", lambda root: state.registered.append(root))

    def feature_frames(root, symbol, *, as_of, tail_rows):
        state.feature_calls.append((symbol, as_of, tail_rows))
        if state.feature_error:
            raise model.ModelInputError(state.feature_error)
        frames = {k: v.iloc[:1].copy() for k, v in trained_artifacts.frames.items()}
        for frame in frames.values():
            frame.index = pd.DatetimeIndex([state.boundary - timedelta(minutes=5) +
                                            timedelta(seconds=state.feature_shift)])
        return frames
    monkeypatch.setattr(model, "build_feature_frames", feature_frames)

    class Inputs:
        def __init__(self, runtime, **kwargs):
            self.client = SimpleNamespace(clock_evidence_sha256=None)
            self.clock_evidence_sha256 = None
            self._lock = threading.RLock()

        def refresh(self, now=None):
            state.refreshes += 1
            state.upper += timedelta(seconds=state.refresh_delay)
            available = state.upper - timedelta(seconds=state.available_age)
            if state.input_time_basis != "CALIBRATED_UTC_UPPER_BOUND":
                available -= timedelta(seconds=5.1)
            gaps = state.input_gaps if state.input_gaps is not None else ([] if state.input_status == "READY" else ["fixture gap"])
            return {"status": state.input_status, "gaps": deepcopy(gaps), "errors": deepcopy(state.input_errors),
                    "data_root": "offline-fixture", "feature_available_at": available.isoformat(),
                    "time_basis": state.input_time_basis}
    monkeypatch.setattr(online, "OnlineInputs", Inputs)
    observations = MemoryObservations()
    account = Account(tmp_path / "paper.sqlite3", started_at=BOUNDARY - timedelta(seconds=1))
    p = pipeline.ForwardPipeline(account, tmp_path / "runtime", observations)
    # Missing metadata may now trigger an explicit public refresh. This offline
    # harness never reaches the network; stale/missing fixtures stay blocked.
    monkeypatch.setattr(p.public_transport, "get", lambda *args: (None, {
        "received_at": state.upper.isoformat(), "sha256": None, "raw_text": None,
        "http_status": None, "body_complete": False}, "OFFLINE_UNAVAILABLE"))
    p.clock = clock
    p.warmup_complete = True
    p.last_settlement_poll = time.monotonic()
    monkeypatch.setattr(p, "update_clock", lambda: clock)

    class Stop:
        def wait(self, seconds):
            state.waits.append(seconds)
            state.upper += timedelta(seconds=seconds)
            return False

        def is_set(self):
            return False
    p.stop = Stop()

    def book(market, side):
        state.book_calls.append((market["market_id"], side))
        return {"status": "OBSERVED", "asset_id": market["outcome_ids"][side],
                "condition_id": market["condition_id"],
                "source_at": (state.upper - timedelta(seconds=state.source_age)).isoformat(),
                "received_at": (state.upper - timedelta(seconds=5.1)).isoformat(),
                "asks": [{"price": .5, "size": 2 * state.capacity}], "bids": [{"price": .49, "size": 1000}]}
    monkeypatch.setattr(p, "refresh_book", book)

    def refresh_markets():
        markets = {}
        for i, stream in enumerate(ORDER, 1):
            asset, duration = STREAMS[stream]
            start = datetime.fromtimestamp(int(state.boundary.timestamp()) // duration * duration, timezone.utc)
            market = {"status": "VERIFIED_MARKET", "market_id": str(100 + i), "condition_id": "0x" + str(i) * 64,
                      "protocol": "v2", "asset": asset, "duration_seconds": duration,
                      "window_start": start.isoformat(), "window_end": (start + timedelta(seconds=duration)).isoformat(),
                      "active": True, "closed": False, "accepting_orders": True,
                      "outcome_ids": {"UP": str(i * 10 + 1), "DOWN": str(i * 10 + 2)},
                      "metadata_received_at": (state.upper - timedelta(seconds=5.1 + state.metadata_age)).isoformat(),
                      "metadata_sha256": "a" * 64, "resolution": {"source": "https://data.chain.link/fixture"},
                      "fee": {"status": "KNOWN_DISABLED", "fees_enabled": False}, "books": {}}
            old = deepcopy(market)
            old.update(window_start=(start - timedelta(seconds=duration)).isoformat(), window_end=start.isoformat())
            # Simulates the observer's local clock still in the old window;
            # the executor must select the exact new contract from `next`.
            markets[stream] = {"current": old, "next": market}
        observations.save("polymarket", {"streams": markets})
    refresh_markets()
    return SimpleNamespace(p=p, account=account, state=state, policy=policy, policy_path=policy_path,
                           clock=clock, observations=observations, refresh_markets=refresh_markets,
                           trained=trained_artifacts)


def test_saved_backtest_bundle_to_pipeline_to_persisted_paper_order(harness):
    h = harness
    with threadpool_limits(1):
        h.p.cycle()
    status = h.p.status()
    assert status["status"] == "PAPER_CYCLE_COMPLETE"
    assert [o["cost_usd"] for o in status["execution"]["fills"]] == [100, 100, 20]
    assert h.account.overview()["locked_cost"] == 220
    assert h.account.overview()["live_enabled"] is False
    for stream, decision in status["decisions"].items():
        expected = h.trained.scores[stream]
        assert decision["prediction"]["z_v4"] == pytest.approx(expected["score"], abs=1e-12)
        assert decision["prediction"]["model_sha256"] == expected["model_sha256"]
        assert decision["prediction"]["manifest_sha256"] == h.policy["deployment_manifest_sha256"][stream]
    assert len(h.state.feature_calls) == 4  # shared symbols reuse one causal frame
    assert all(call[1] == BOUNDARY + timedelta(seconds=6.8) for call in h.state.feature_calls)
    fill = status["execution"]["fills"][0]
    assert fill["book_received_at_local"] != fill["fill_at"]
    assert fill["clock_projection"]["receipt_sha256"] == "d" * 64
    assert h.state.registered == ["offline-fixture"]
    assert h.state.book_calls  # cached metadata cannot substitute for a new book


def test_poll_and_process_restart_do_not_reissue_cohort_budget(harness):
    h = harness
    h.p.cycle()
    h.p.cycle()
    assert h.state.refreshes == 1
    assert h.account.overview()["order_count"] == 3
    # A process restart forgets last_cohort, but SQLite is authoritative.
    h.p.last_cohort = None
    h.p.cycle()
    assert h.state.refreshes == 2
    assert h.account.overview()["order_count"] == 3
    assert h.account.cohort_state(BOUNDARY.isoformat())["used_usd"] == 220


def test_nonboundary_highest_stream_keeps_lower_locked_without_capacity(harness):
    h = harness
    h.state.boundary += timedelta(minutes=5)
    h.state.upper += timedelta(minutes=5)
    h.clock["observed_at"] = h.state.boundary.isoformat()
    h.state.capacity = 600
    h.refresh_markets()
    h.p.cycle()
    status = h.p.status()
    assert status["decisions"][ORDER[0]] == {"status": "NOT_DUE"}
    assert status["execution"]["fills"] == []
    assert status["execution"]["cohort"]["streams"][0]["state"] == "CAPACITY_LOCKED"
    assert h.account.overview()["cash"] == 10000


def test_nonboundary_can_unlock_only_after_upper_observed_capacity_gate(harness):
    h = harness
    h.state.boundary += timedelta(minutes=5)
    h.state.upper += timedelta(minutes=5)
    h.clock["observed_at"] = h.state.boundary.isoformat()
    h.refresh_markets()
    h.p.cycle()
    status = h.p.status()
    assert status["execution"]["cohort"]["streams"][0]["state"] == "NOT_DUE"
    assert status["execution"]["fills"][0]["underlying"] == "ETH"


@pytest.mark.parametrize("capacity,expected_orders", [(600, 0), (100, 3)])
def test_real_model_no_signal_does_not_bypass_upper_capacity_gate(harness, monkeypatch, capacity, expected_orders):
    h = harness
    h.p.load_ready()
    # Save and pin a real alternative fixture whose frozen threshold declines
    # the top stream; do not mock selected=False or mutate a loaded artifact.
    root = h.policy_path.parent.parent / "declining_model"
    monkeypatch.setattr(model, "MODEL_ROOT", root)
    saved = h.p.bundles[ORDER[0]]
    package = deepcopy(saved["package"])
    package["threshold_frozen"] = 1000.
    pin = model._save_stream(root / "deployment_q4", ORDER[0], package, saved["manifest"])
    h.policy["deployment_manifest_sha256"][ORDER[0]] = pin
    h.policy_path.write_text(json.dumps(h.policy))
    h.state.capacity = capacity
    h.p.cycle()
    assert h.p.status()["decisions"][ORDER[0]]["status"] == "NO_SIGNAL"
    assert h.account.overview()["order_count"] == expected_orders
    expected_gate = "CAPACITY_LOCKED" if capacity == 600 else "NO_SIGNAL"
    assert h.p.status()["execution"]["cohort"]["streams"][0]["state"] == expected_gate


@pytest.mark.parametrize("metadata_age,source_age", [(91, .1), (30, 2.01)])
def test_stale_metadata_or_provider_book_cannot_be_promoted_to_fresh(harness, metadata_age, source_age):
    h = harness
    h.state.metadata_age, h.state.source_age = metadata_age, source_age
    h.refresh_markets()
    h.p.cycle()
    assert h.p.status()["execution"]["fills"] == []
    assert h.account.overview()["order_count"] == 0


def test_missing_exact_current_window_fails_closed(harness):
    h = harness
    data = h.observations.data["polymarket"]["streams"][ORDER[0]]
    data["next"] = None
    h.p.cycle()
    assert h.p.status()["execution"]["fills"] == []
    assert h.account.overview()["order_count"] == 0


@pytest.mark.parametrize("failure", ["gap", "wrong_closed_row", "future_availability", "expired"])
def test_model_or_timing_failure_never_creates_paper_order(harness, failure):
    h = harness
    if failure == "gap":
        h.state.feature_error = "fixture data gap"
    elif failure == "wrong_closed_row":
        h.state.feature_shift = -300
    elif failure == "future_availability":
        h.p.load_ready()
        with pytest.raises(ValueError, match="receipt lower-bound timing not ready"):
            h.p.prepare_decisions(h.policy, {"feature_available_at": (h.state.upper + timedelta(seconds=3)).isoformat(),
                "data_root": "fixture", "time_basis": "CALIBRATED_UTC_UPPER_BOUND"}, BOUNDARY)
        assert h.state.waits == []
    else:
        h.state.upper = BOUNDARY + timedelta(seconds=20.1)
    if failure in ("gap", "wrong_closed_row"):
        with pytest.raises(ValueError):
            h.p.cycle()
    elif failure == "expired":
        h.p.cycle()
        assert h.p.status()["status"] == "PAPER_WAITING_WINDOW"
    assert h.account.overview()["order_count"] == 0


def test_calibrated_upper_crossing_new_boundary_cannot_close_future_bar(harness):
    h = harness
    h.state.upper = BOUNDARY + timedelta(seconds=.1)
    h.p.cycle()
    assert h.state.refreshes == 0
    assert h.account.overview()["order_count"] == 0


@pytest.mark.parametrize("lower_seconds", [1, 3, 4.999])
def test_first_five_seconds_after_boundary_do_not_start_input_refresh(harness, lower_seconds):
    h = harness
    h.state.upper = BOUNDARY + timedelta(seconds=lower_seconds + .2)
    h.p.cycle()
    assert h.p.status()["status"] == "PAPER_WAITING_WINDOW"
    assert h.state.refreshes == 0
    assert h.state.feature_calls == [] and h.state.book_calls == []
    assert h.p.last_cohort is None
    assert h.account.overview()["order_count"] == 0


def test_exact_fifth_second_lower_bound_allows_input_refresh(harness):
    h = harness
    h.state.upper = BOUNDARY + timedelta(seconds=5.2)
    h.p.cycle()
    assert h.state.refreshes == 1
    assert all(call[1] == BOUNDARY + timedelta(seconds=5) for call in h.state.feature_calls)
    assert h.account.overview()["order_count"] == 3


def test_verification_pause_keeps_settlement_polling_and_resumes_only_in_normal_window(harness, monkeypatch):
    h = harness
    calls = []
    original_poll = h.p.poll_settlements
    def poll(policy):
        calls.append(deepcopy(policy))
        return original_poll(policy)
    monkeypatch.setattr(h.p, "poll_settlements", poll)
    pause = h.p.runtime / "verification_pause.json"
    pause.parent.mkdir(parents=True, exist_ok=True)
    pause.write_text(json.dumps({"reason": "offline integration fixture"}))
    policy_before = h.policy_path.read_bytes()
    h.p.cycle()
    assert calls == [{}]
    assert h.p.status()["status"] == "VERIFICATION_PAUSE"
    assert h.p.status()["active"] is False
    assert h.state.refreshes == 0
    assert h.state.feature_calls == [] and h.state.book_calls == []
    assert h.state.registered == []
    assert h.p.last_cohort is None
    assert h.account.overview()["order_count"] == 0
    pause.unlink()
    h.state.upper = BOUNDARY + timedelta(seconds=21)
    h.p.cycle()
    assert calls == [{}, {}]
    assert h.p.status()["status"] == "PAPER_WAITING_WINDOW"
    assert h.state.refreshes == 0
    assert h.account.overview()["order_count"] == 0
    h.state.boundary += timedelta(minutes=5)
    h.state.upper = h.state.boundary + timedelta(seconds=7)
    h.clock["observed_at"] = h.state.boundary.isoformat()
    h.refresh_markets()
    h.p.cycle()
    assert calls == [{}, {}, {}]
    assert h.p.status()["status"] == "PAPER_CYCLE_COMPLETE"
    assert h.state.refreshes == 1
    assert h.account.overview()["order_count"] == 3
    assert h.policy_path.read_bytes() == policy_before


@pytest.mark.parametrize("phase", ["input_refresh", "book_refresh"])
def test_twenty_second_deadline_remains_closed_after_delayed_data(harness, phase):
    h = harness
    if phase == "input_refresh":
        h.state.refresh_delay = 13  # Normal entry is +7; completed input is exactly +20.
        with pytest.raises(ValueError, match="missed feature decision window"):
            h.p.cycle()
        assert h.state.feature_calls == [] and h.state.book_calls == []
    else:
        original_refresh = h.p.refresh_book
        def delayed_refresh(market, side):
            h.state.upper = BOUNDARY + timedelta(seconds=20)
            return original_refresh(market, side)
        h.p.refresh_book = delayed_refresh
        h.p.cycle()
        assert h.p.status()["execution"]["fills"] == []
        assert h.p.status()["execution"]["status"] == "BLOCKED"
    assert h.state.refreshes == 1
    assert h.account.overview()["order_count"] == 0


def test_slow_pre_window_calibration_rechecks_window_before_inputs(harness, monkeypatch):
    h = harness
    h.state.upper = BOUNDARY-timedelta(seconds=5)
    def slow_calibration():
        h.state.upper = BOUNDARY+timedelta(seconds=21)
        return h.clock
    monkeypatch.setattr(h.p, "update_clock", slow_calibration)
    h.p.cycle()
    assert h.p.status()["status"] == "PAPER_WAITING_WINDOW"
    assert h.state.refreshes == 0 and h.state.feature_calls == [] and h.state.book_calls == []
    assert h.account.overview()["order_count"] == 0


def test_local_first_receive_time_is_projected_once(harness):
    h = harness
    h.state.input_time_basis = "SYSTEM_UTC"
    h.p.cycle()
    signal = h.p.status()["decisions"][ORDER[0]]["signal"]
    assert datetime.fromisoformat(signal["feature_available_at"]) == BOUNDARY + timedelta(seconds=6.7)
    assert h.account.overview()["order_count"] == 3


def test_feature_receipt_within_uncertainty_waits_then_scores_and_fills(harness):
    h = harness
    h.state.available_age = .05
    h.p.cycle()
    assert h.state.waits == pytest.approx([.16, .21])
    signal = h.p.status()["decisions"][ORDER[0]]["signal"]
    available = datetime.fromisoformat(signal["feature_available_at"])
    assert available == BOUNDARY + timedelta(seconds=6.95)
    assert all(call[1] > available for call in h.state.feature_calls)
    assert datetime.fromisoformat(signal["decision_at"]) == BOUNDARY + timedelta(seconds=7.16)
    assert h.account.overview()["order_count"] == 3


@pytest.mark.parametrize("failure", ["stopped", "wait_did_not_advance_clock", "deadline"])
def test_feature_receipt_wait_rechecks_stop_clock_and_decision_deadline(harness, failure):
    h = harness
    h.p.load_ready()
    if failure == "deadline":
        h.state.upper = BOUNDARY + timedelta(seconds=19.8)
        available = h.state.upper + timedelta(seconds=.1)
    else:
        available = h.state.upper - timedelta(seconds=.05)
        h.p.stop.wait = lambda seconds: failure == "stopped"
    with pytest.raises(ValueError, match="receipt lower-bound|missed feature decision"):
        h.p.prepare_decisions(h.policy, {"feature_available_at": available.isoformat(), "data_root": "fixture",
            "time_basis": "CALIBRATED_UTC_UPPER_BOUND"}, BOUNDARY)
    assert h.state.feature_calls == [] and h.state.book_calls == []
    assert h.account.overview()["order_count"] == 0


def test_incomplete_online_inputs_do_not_call_model_or_executor(harness):
    h = harness
    h.state.input_status = "BLOCKED"
    h.p.cycle()
    assert h.p.status()["status"] == "PAPER_DATA_BLOCKED"
    assert h.state.feature_calls == [] and h.state.book_calls == []
    assert h.account.overview()["order_count"] == 0


def test_committed_revision_skips_current_cohort_and_only_refreshes_next_window(harness, monkeypatch):
    from tools.argon4_forward import revisions
    h = harness
    reconciliations = []
    def reconcile(runtime, errors, now, *, cutover_now):
        assert h.p.inputs._lock._is_owned()
        assert runtime == h.p.runtime
        assert errors == h.state.input_errors
        assert now() == h.state.upper
        assert cutover_now() == h.state.upper
        reconciliations.append(deepcopy(errors))
        return {"status": "COMMITTED", "revision_id": "offline-fixture-revision"}
    monkeypatch.setattr(revisions, "reconcile_confirmed_price_revisions", reconcile)
    h.state.input_status = "BLOCKED"
    h.state.input_errors = [{"source": "futures:BTCUSDT", "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"}]
    h.p.state["decisions"] = {"stale_decision": {"status": "SIGNAL"}}
    h.p.cycle()
    status = h.p.status()
    assert len(reconciliations) == 1
    assert status["revision"]["status"] == "COMMITTED"
    assert status["decisions"] == {}
    assert status["execution"]["status"] == "NOT_EVALUATED"
    assert h.state.refreshes == 1
    assert h.state.feature_calls == [] and h.state.book_calls == []
    assert h.account.overview()["order_count"] == 0
    h.state.input_status, h.state.input_errors = "READY", []
    h.p.cycle()
    assert h.state.refreshes == 1
    assert len(reconciliations) == 1
    assert h.p.status()["status"] == "PAPER_DATA_BLOCKED"
    assert "下一窗口" in h.p.status()["reason"]
    assert h.account.overview()["order_count"] == 0
    h.state.boundary += timedelta(minutes=5)
    h.state.upper += timedelta(minutes=5)
    h.clock["observed_at"] = h.state.boundary.isoformat()
    h.refresh_markets()
    h.p.cycle()
    assert h.state.refreshes == 2
    assert len(reconciliations) == 1
    assert h.p.status()["revision"] is None
    assert h.p.status()["last_attempt_cohort"] == h.state.boundary.isoformat()
    assert h.account.overview()["order_count"] == 3
    assert h.p.status()["inputs"]["status"] == "READY"
    assert h.p.status()["inputs"]["errors"] == []
    h.p.cycle()
    assert h.p.status()["status"] == "PAPER_WAITING_WINDOW"


def test_nonrevision_input_error_does_not_invoke_revision_recovery(harness, monkeypatch):
    from tools.argon4_forward import revisions
    h = harness
    monkeypatch.setattr(revisions, "reconcile_confirmed_price_revisions",
                        lambda *a, **k: pytest.fail("unrelated input error invoked revision recovery"))
    h.state.input_status = "BLOCKED"
    h.state.input_errors = [{"source": "futures:BTCUSDT", "error": "HTTPError"}]
    h.p.cycle()
    assert h.p.status()["revision"] is None
    assert h.p.status()["decisions"] == {}
    assert h.account.overview()["order_count"] == 0


@pytest.mark.parametrize("source", ["futures:BTCUSDT:5m", "mark:ETHUSDT:1h"])
def test_revision_reconciliation_supplies_explicit_registry_only_for_mark(harness, monkeypatch, source):
    from tools.argon4_forward import online, revisions
    h = harness
    collector = online.OnlineInputs(h.p.runtime)
    errors = [{"source": source, "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"}]
    def reconcile(runtime, actual_errors, now, **kwargs):
        assert collector._lock._is_owned()
        assert runtime == h.p.runtime and actual_errors == errors and now() == h.state.upper
        expected = ({"mark_registry_path": pipeline.ROOT / "archive/catalog/mark_candle_registry.json"}
                    if source.startswith("mark:") else {})
        expected["cutover_now"] = h.p.revision_cutover_now
        assert kwargs == expected
        return {"status": "WAITING_CONFIRMATION"}
    monkeypatch.setattr(revisions, "reconcile_confirmed_price_revisions", reconcile)
    assert h.p.reconcile_inputs(collector, {"errors": errors}) == {"status": "WAITING_CONFIRMATION"}
    assert h.observations.latest("input_revision_reconciliation")["status"] == "WAITING_CONFIRMATION"
    assert h.account.overview()["order_count"] == 0


@pytest.mark.parametrize("effective_seconds", [0, 299, 602])
def test_persisted_cutover_skips_whole_effective_window_even_after_restart(harness, monkeypatch, effective_seconds):
    from tools.argon4_forward import revisions
    h = harness
    effective = BOUNDARY + timedelta(seconds=effective_seconds)
    cutover_cohort = pipeline.cohort_at(effective)
    marker_path = h.p.runtime / "fixture_committed_marker.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps({"status": "COMMITTED", "revision_id": "offline-fixture",
                                      "view_effective_at": effective.isoformat()}))
    def committed(runtime):
        assert runtime == h.p.runtime
        return json.loads(marker_path.read_text())
    monkeypatch.setattr(revisions, "ensure_view_committed", committed)
    h.state.boundary = cutover_cohort
    h.state.upper = max(cutover_cohort + timedelta(seconds=7), effective + timedelta(seconds=.5))
    h.p.state["decisions"] = {"stale": {"status": "SIGNAL"}}
    h.p.cycle()
    assert h.p.status()["status"] == "REVISION_WAITING_FUTURE_WINDOW"
    assert h.p.status()["decisions"] == {}
    assert h.p.status()["execution"]["blocked"] == ["REVISION_CUTOVER_WINDOW"]
    assert h.state.refreshes == 0 and h.state.feature_calls == [] and h.state.book_calls == []
    # A fresh process loses its in-memory cohort flag, but the committed marker
    # still prevents warmup, scoring or execution during the cutover window.
    restarted = pipeline.ForwardPipeline(h.account, h.p.runtime, h.observations)
    restarted.clock = h.clock
    restarted.stop = h.p.stop
    monkeypatch.setattr(restarted, "update_clock", lambda: h.clock)
    monkeypatch.setattr(restarted, "refresh_book", h.p.refresh_book)
    assert restarted.warmup_complete is False and restarted.last_cohort is None
    restarted.cycle()
    assert restarted.status()["status"] == "REVISION_WAITING_FUTURE_WINDOW"
    assert h.state.refreshes == 0 and h.account.overview()["order_count"] == 0
    # The next whole cohort must obtain fresh inputs; no old decision is replayed.
    h.state.boundary = cutover_cohort + timedelta(minutes=5)
    h.state.upper = h.state.boundary + timedelta(seconds=7)
    h.clock["observed_at"] = h.state.boundary.isoformat()
    h.refresh_markets()
    restarted.warmup_complete = True
    restarted.cycle()
    assert h.state.refreshes == 1 and h.state.feature_calls
    assert restarted.status()["last_attempt_cohort"] == h.state.boundary.isoformat()
    assert restarted.status()["status"] == "PAPER_CYCLE_COMPLETE"
    assert h.observations.latest("prediction_decisions")["cohort"] == h.state.boundary.isoformat()


def test_deployment_enable_and_each_manifest_pin_are_required(harness, monkeypatch):
    h = harness
    h.policy["paper_trading_active"] = False
    h.policy_path.write_text(json.dumps(h.policy))
    h.p.cycle()
    assert h.p.status()["status"] == "WAITING_LOCAL_BACKTEST"
    assert h.p.bundles == {}
    h.policy["paper_trading_active"] = True
    h.policy["deployment_manifest_sha256"].pop(ORDER[0])
    h.policy_path.write_text(json.dumps(h.policy))
    with pytest.raises(ValueError, match="not pinned"):
        h.p.load_ready()
    h.policy["deployment_manifest_sha256"][ORDER[0]] = "e" * 64
    h.policy_path.write_text(json.dumps(h.policy))
    monkeypatch.setattr(model.joblib, "load", lambda *_: pytest.fail("untrusted model loaded"))
    with pytest.raises(model.ModelInputError, match="manifest hash"):
        h.p.load_ready()
    assert h.account.overview()["order_count"] == 0


def test_cold_start_prepares_inputs_without_paper_decision(harness):
    h = harness
    h.p.warmup_complete = False
    h.p.cycle()
    assert h.p.status()["status"] == "PAPER_WARMUP"
    assert h.state.refreshes == 1
    assert h.state.feature_calls == [] and h.state.book_calls == []
    assert h.account.overview()["order_count"] == 0


@pytest.mark.parametrize("errors,gaps,ready", [
    ([{"source": "clock", "error": "CLOCK_SKEW_OR_LATENCY"}], [], True),
    ([{"source": "clock", "error": "UNAVAILABLE"}], [], False),
    ([{"source": "futures:BTCUSDT", "error": "HTTPError"}], [], False),
    ([{"source": "clock", "error": "CLOCK_SKEW_OR_LATENCY"},
      {"source": "dvol", "error": "HTTPError"}], [], False),
    ([{"source": "clock", "error": "CLOCK_SKEW_OR_LATENCY"}], ["missing closed bar"], False),
])
def test_warmup_only_tolerates_clock_skew_and_never_data_errors(harness, errors, gaps, ready):
    h = harness
    h.p.warmup_complete = False
    h.state.input_status, h.state.input_errors, h.state.input_gaps = "BLOCKED", errors, gaps
    h.p.cycle()
    assert h.p.warmup_complete is ready
    assert h.p.status()["active"] is False
    assert h.state.feature_calls == [] and h.state.book_calls == []
    assert h.account.overview()["order_count"] == 0


def test_stop_during_causal_wait_prevents_new_paper_fills(harness):
    h = harness
    def stopped_wait(seconds):
        h.state.upper += timedelta(seconds=seconds)
        return True
    h.p.stop.wait = stopped_wait
    h.p.cycle()
    assert h.account.overview()["order_count"] == 0
    assert h.state.book_calls == []


def official_fetcher(h, monkeypatch):
    """Completed GET fixtures advance time, as a real HTTP round trip does."""
    from tools.argon4_forward import execution
    calls = []
    def fetch(market_id):
        calls.append(market_id)
        h.state.upper += timedelta(seconds=1)
        market = next(v["next"] for v in h.observations.data["polymarket"]["streams"].values()
                      if v["next"]["market_id"] == market_id)
        raw = {"id": market_id, "conditionId": market["condition_id"], "version": "v2",
               "eventStartTime": market["window_start"], "endDate": market["window_end"],
               "resolutionSource": market["resolution"]["source"], "resolutionStatus": "resolved",
               "outcomes": ["Up", "Down"], "positionIds": [market["outcome_ids"]["UP"], market["outcome_ids"]["DOWN"]],
               "outcomePrices": ["1", "0"]}
        text = json.dumps(raw)
        receipt = {"url": f"https://gamma-api.polymarket.com/markets/{market_id}", "method": "GET", "http_status": 200,
                   "body_complete": True, "raw_text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "received_at": (h.state.upper - timedelta(seconds=h.clock["server_minus_local_seconds"] +
                                                                 h.clock["uncertainty_seconds"])).isoformat()}
        return raw, receipt
    monkeypatch.setattr(execution, "fetch_public_resolution", fetch)
    return calls


def test_prepared_revision_blocks_predictions_without_freezing_official_settlement(harness, monkeypatch):
    from tools.argon4_forward import revisions
    h = harness
    h.p.cycle()
    assert h.account.overview()["order_count"] == 3
    initial_refreshes = h.state.refreshes
    h.state.upper = BOUNDARY + timedelta(minutes=16, seconds=1)
    h.clock.update(server_minus_local_seconds=0, observed_at=h.state.upper.isoformat())
    h.p.last_settlement_poll = 0
    calls = official_fetcher(h, monkeypatch)
    def prepared(runtime):
        assert runtime == h.p.runtime
        assert h.account.overview()["positions"] == []
        raise ValueError("PREPARED revision has not been committed")
    monkeypatch.setattr(revisions, "ensure_view_committed", prepared)
    with pytest.raises(ValueError, match="PREPARED"):
        h.p.cycle()
    assert len(calls) == 3
    assert h.account.overview()["positions"] == []
    assert h.account.overview()["order_count"] == 3
    assert h.state.refreshes == initial_refreshes


def test_committed_cutover_wait_does_not_freeze_official_settlement(harness, monkeypatch):
    from tools.argon4_forward import revisions
    h = harness
    h.p.cycle()
    assert h.account.overview()["order_count"] == 3
    initial_refreshes = h.state.refreshes
    h.state.upper = BOUNDARY + timedelta(minutes=15, seconds=7)
    h.clock.update(server_minus_local_seconds=0, observed_at=h.state.upper.isoformat())
    h.p.last_settlement_poll = 0
    calls = official_fetcher(h, monkeypatch)
    monkeypatch.setattr(revisions, "ensure_view_committed", lambda runtime: {
        "status": "COMMITTED", "revision_id": "offline-fixture",
        "view_effective_at": (BOUNDARY + timedelta(minutes=15, seconds=1)).isoformat()})
    h.p.cycle()
    assert len(calls) == 3 and h.account.overview()["positions"] == []
    assert h.account.overview()["order_count"] == 3
    assert h.state.refreshes == initial_refreshes
    assert h.p.status()["status"] == "REVISION_WAITING_FUTURE_WINDOW"


@pytest.mark.parametrize("offset", [-5, 0, 5])
def test_official_settlement_uses_clock_after_http_completion(harness, monkeypatch, offset):
    h = harness
    h.p.cycle()
    assert h.account.overview()["order_count"] == 3
    h.state.upper = BOUNDARY + timedelta(minutes=16, seconds=1)
    h.clock.update(server_minus_local_seconds=offset, observed_at=h.state.upper.isoformat())
    h.p.last_settlement_poll = 0
    calls = official_fetcher(h, monkeypatch)
    h.p.cycle()
    assert len(calls) == 3
    assert h.account.overview()["positions"] == []
    assert h.account.overview()["wins"] + h.account.overview()["losses"] == 3
    original = h.observations.data["paper_resolution_receipt"]["receipts"][0]
    assert datetime.fromisoformat(original["received_at"]) == h.state.upper - timedelta(seconds=offset + .1)
    assert h.observations.data["paper_resolution_receipt"]["clock_evidence_sha256"] == "d" * 64


@pytest.mark.parametrize("blocked_source", ["backoff", "cold_restart", "prediction_disabled", "missing_model_pin"])
def test_official_settlement_is_independent_of_binance_input_availability(harness, monkeypatch, blocked_source):
    h = harness
    h.p.cycle()
    h.state.upper = BOUNDARY + timedelta(minutes=16, seconds=1)
    h.clock.update(server_minus_local_seconds=0, observed_at=h.state.upper.isoformat())
    h.p.last_settlement_poll = 0
    calls = official_fetcher(h, monkeypatch)
    if blocked_source == "backoff":
        path = h.p.runtime / "features/binance_backoff.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"blocked_until_utc": "2099-01-01T00:00:00Z"}))
    elif blocked_source == "cold_restart":
        h.p.warmup_complete = False
    elif blocked_source == "prediction_disabled":
        h.policy["paper_trading_active"] = False
        h.policy_path.write_text(json.dumps(h.policy))
    else:
        h.policy["deployment_manifest_sha256"] = {}
        h.policy_path.write_text(json.dumps(h.policy))
    if blocked_source == "missing_model_pin":
        with pytest.raises(ValueError, match="not pinned"):
            h.p.cycle()
    else:
        h.p.cycle()
    assert len(calls) == 3
    assert h.account.overview()["positions"] == []


def enable_v5(h):
    h.policy.update(version="offline-eligible-waterfall-v5",
                    waterfall_semantics="ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1")
    h.policy_path.write_text(json.dumps(h.policy))


def enable_v12(h, monkeypatch):
    from tools.argon4_forward import online
    from tools.argon4_forward.runtime_profile import V12_ENGINEERING
    enable_v5(h)
    active = ["BTC_15m", "ETH_5m", "SOL_15m", "SOL_5m", "BTC_5m"]
    engineering = deepcopy(V12_ENGINEERING)
    # This harness uses explicit tiny model frames; real snapshot parity has a
    # separate full-history test. Do not pretend these frames are disk inputs.
    engineering["feature_snapshot_cache"] = False
    h.policy.update(version="offline-v12-five-stream", order=active+["ETH_15m", "BNB_15m"],
        engineering=engineering, min_decision_delay_seconds=2.5,
        execution={"multi_look_ioc": True, "max_price_rechecks": 3,
            "price_recheck_interval_seconds": 2.5, "minimum_recheck_remaining_seconds": 3.,
            "last_recheck_start_after_boundary_seconds": 18.5},
        execution_stage={"id": "offline-stage2", "status": "CONFIRMED", "enabled_streams": active,
            "effective_from": BOUNDARY.isoformat()})
    h.policy_path.write_text(json.dumps(h.policy))
    old_refresh = online.OnlineInputs.refresh
    monkeypatch.setattr(online.OnlineInputs, "refresh", lambda self, now=None, **kw: old_refresh(self, now))
    h.state.upper = BOUNDARY+timedelta(seconds=2.7)
    h.state.capacity = 1000
    return active


def test_v12_starts_at_lower_bound_2_5_and_scores_five_pinned_streams_once(harness, monkeypatch):
    h = harness
    active = enable_v12(h, monkeypatch)
    h.state.upper -= timedelta(microseconds=1)
    h.p.cycle()
    assert h.state.refreshes == 0
    h.state.upper += timedelta(microseconds=1)
    h.p.cycle()
    status = h.p.status()
    assert status["status"] == "PAPER_CYCLE_COMPLETE"
    assert [x[0] for x in h.state.feature_calls] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert list(status["timing"]["computation"]["inference_seconds"]) == active
    assert status["timing"]["completed_after_boundary_seconds"] < 5
    for stream in active:
        assert status["decisions"][stream]["prediction"]["z_v4"] == h.trained.scores[stream]["score"]
    assert h.account.overview()["live_enabled"] is False


def test_v12_recheck_reuses_frozen_scores_without_input_reads_or_lower_allocation(harness, monkeypatch):
    h = harness
    active = enable_v12(h, monkeypatch)
    original = h.p.refresh_book
    price = [.52]
    def book(market, side):
        result = original(market, side)
        result["asks"][0]["price"] = price[0]
        return result
    monkeypatch.setattr(h.p, "refresh_book", book)
    h.p.cycle()
    state = h.p.status()
    assert state["status"] == "PAPER_WAITING_PRICE_RECHECK"
    assert state["execution"]["stream_results"][active[1]]["status"] == "UNREACHED"
    scores = deepcopy(state["decisions"])
    calls, reads, features = len(h.state.book_calls), h.state.refreshes, len(h.state.feature_calls)
    h.p.cycle()
    assert len(h.state.book_calls) == calls
    h.state.upper = datetime.fromisoformat(state["execution"]["next_retry_at"])+timedelta(milliseconds=1)
    price[0] = .5
    h.p.cycle()
    state = h.p.status()
    assert state["status"] == "PAPER_CYCLE_COMPLETE"
    assert state["decisions"] == scores
    assert h.state.refreshes == reads and len(h.state.feature_calls) == features
    assert h.account.overview()["order_count"] == 2
    assert state["timing"]["execution_passes"] == 2
    h.p.cycle()
    assert h.account.overview()["order_count"] == 2


@pytest.mark.parametrize("mode", ["expired", "changed_policy"])
def test_v12_pending_is_abandoned_without_late_order_or_rescore(harness, monkeypatch, mode):
    h = harness
    enable_v12(h, monkeypatch)
    original = h.p.refresh_book
    def too_high(market, side):
        result = original(market, side)
        result["asks"][0]["price"] = .52
        return result
    monkeypatch.setattr(h.p, "refresh_book", too_high)
    h.p.cycle()
    assert h.p._pending_execution
    calls = len(h.state.book_calls)
    if mode == "expired":
        h.state.upper = BOUNDARY+timedelta(seconds=20)
    else:
        h.policy["version"] += "-changed"
        h.policy_path.write_text(json.dumps(h.policy))
    h.p.cycle()
    assert h.p._pending_execution is None and h.p._pending_timing is None
    assert len(h.state.book_calls) == calls and h.state.refreshes == 1
    assert h.account.overview()["order_count"] == 0


def last_close_missing():
    return {"status": "BLOCKED", "errors": [],
        "gaps": [{"source": "BTCUSDT:futures:5m", "error": "LATEST_CLOSED_BAR_MISSING"}],
        "sources": {"BTCUSDT:futures:5m": {"rows": 4320,
            "latest_closed": (BOUNDARY-timedelta(minutes=5)).isoformat(),
            "issues": ["LATEST_CLOSED_BAR_MISSING"]}}}


def test_v12_retries_only_unpublished_last_close_and_then_validates(harness, monkeypatch):
    from tools.argon4_forward import online
    h = harness
    enable_v12(h, monkeypatch)
    old = online.OnlineInputs.refresh
    attempts = []
    def delayed(self, now=None, **kwargs):
        attempts.append((now, kwargs["deadline_monotonic"]))
        return last_close_missing() if len(attempts) == 1 else old(self, now, **kwargs)
    monkeypatch.setattr(online.OnlineInputs, "refresh", delayed)
    h.p.cycle()
    assert len(attempts) == 2 and attempts[1][0] > attempts[0][0]
    assert .5 in h.state.waits
    assert h.p.status()["timing"]["input_refresh_attempts"] == 2
    assert h.p.status()["execution"]["status"] == "PROCESSED"


@pytest.mark.parametrize("change", ["error", "interior_gap", "old_bar", "hourly", "warmup"])
def test_v12_does_not_retry_integrity_failure_as_publication_delay(change):
    value = last_close_missing()
    if change == "error": value["errors"] = [{"error": "CLOCK_SKEW_OR_LATENCY"}]
    if change == "interior_gap": value["gaps"][0]["error"] = "CANDLE_GAPS"
    if change == "old_bar": value["sources"]["BTCUSDT:futures:5m"]["latest_closed"] = (BOUNDARY-timedelta(minutes=10)).isoformat()
    if change == "hourly": value["gaps"][0]["source"] = "BTCUSDT:futures:1h"
    if change == "warmup": value["sources"]["BTCUSDT:futures:5m"]["rows"] = 300
    assert pipeline.ForwardPipeline.retryable_latest_candle(value, BOUNDARY) is False


def test_v12_stops_missing_latest_close_at_input_deadline(harness, monkeypatch):
    from tools.argon4_forward import online
    h = harness
    enable_v12(h, monkeypatch)
    attempts = []
    def not_published(self, now=None, **kwargs):
        attempts.append(now)
        h.state.upper += timedelta(seconds=1)
        return last_close_missing()
    monkeypatch.setattr(online.OnlineInputs, "refresh", not_published)
    h.p.cycle()
    assert len(attempts) < 10
    assert h.p.status()["status"] == "PAPER_DATA_BLOCKED"
    assert h.state.feature_calls == [] and h.account.overview()["order_count"] == 0
    h.p.cycle()
    assert len(attempts) < 10


def test_v12_fast_polling_and_settlement_reads_avoid_critical_window(harness, monkeypatch):
    h = harness
    enable_v12(h, monkeypatch)
    h.p._worker_engineering = deepcopy(h.policy["engineering"])
    polled = []
    monkeypatch.setattr(h.p, "poll_settlements", lambda policy: polled.append(True))
    h.p.cycle()
    assert not polled and h.p._worker_delay() == .2
    h.state.upper = BOUNDARY+timedelta(seconds=25)
    h.p.cycle()
    assert polled == [True] and h.p._worker_delay() == 1


@pytest.mark.parametrize("cancel", ["pause", "disabled", "stop", "clock"])
def test_v12_cancelled_wait_cannot_resume_when_condition_clears(harness, monkeypatch, cancel):
    h = harness
    enable_v12(h, monkeypatch)
    original = h.p.refresh_book
    def high(market, side):
        value = original(market, side)
        value["asks"][0]["price"] = .52
        return value
    monkeypatch.setattr(h.p, "refresh_book", high)
    h.p.cycle()
    assert h.p._pending_execution
    calls = len(h.state.book_calls)
    original_clock = h.p.clock
    if cancel == "pause":
        (h.p.runtime/"verification_pause.json").write_text("{}")
    elif cancel == "disabled":
        h.policy["paper_trading_active"] = False
        h.policy_path.write_text(json.dumps(h.policy))
    elif cancel == "stop":
        monkeypatch.setattr(h.p.stop, "is_set", lambda: True)
    else:
        h.p.clock = {**h.p.clock, "receipt_sha256": "e"*64}
    h.p.cycle()
    assert h.p._pending_execution is None
    if cancel == "pause": (h.p.runtime/"verification_pause.json").unlink()
    if cancel == "disabled":
        h.policy["paper_trading_active"] = True
        h.policy_path.write_text(json.dumps(h.policy))
    if cancel == "stop": monkeypatch.setattr(h.p.stop, "is_set", lambda: False)
    h.p.clock = original_clock
    h.state.upper += timedelta(seconds=3)
    h.p.cycle()
    assert len(h.state.book_calls) == calls
    assert h.state.refreshes == 1 and h.account.overview()["order_count"] == 0


def test_v12_settlement_guard_rechecks_each_fetch_and_defers_before_boundary(harness, monkeypatch):
    from tools.argon4_forward import execution
    h = harness
    enable_v12(h, monkeypatch)
    h.p._worker_engineering = h.policy["engineering"]
    h.state.upper = BOUNDARY+timedelta(seconds=280)
    polled = []
    monkeypatch.setattr(h.p, "poll_settlements", lambda policy: polled.append(True))
    monkeypatch.setattr(execution, "fetch_public_resolution", lambda *a: pytest.fail("must not fetch near boundary"))
    h.p.cycle()
    assert polled == [] and h.p._worker_delay() == 1
    with pytest.raises(ValueError, match="DEFERRED_FOR_DECISION_WINDOW"):
        h.p.fetch_resolution("123")


def test_v12_future_stage_activation_does_not_sleep_through_early_window(harness, monkeypatch):
    h = harness
    enable_v12(h, monkeypatch)
    h.p._worker_engineering = h.policy["engineering"]
    h.p.state.update(status="STAGE_WAITING_ACTIVATION", active=False)
    h.state.upper = BOUNDARY+timedelta(seconds=295)
    assert h.p._worker_delay() == 1
    h.state.upper = BOUNDARY+timedelta(seconds=299.5)
    assert h.p._worker_delay() == .2


def test_v5_saved_model_scores_fill_approved_budget_and_persist_execution(harness):
    h = harness
    enable_v5(h)
    h.state.capacity = 1000
    h.p.cycle()
    state = h.p.status()
    result = state["execution"]
    assert h.account.overview()["order_count"] == 2
    assert result["cohort"]["used_usd"] == 220
    assert result["stream_results"][ORDER[0]]["status"] == "FILLED_BUDGET"
    assert result["stream_results"][ORDER[1]]["status"] == "FILLED_BUDGET"
    assert result["stream_results"][ORDER[0]]["approved_usd"] == 150
    assert result["stream_results"][ORDER[1]]["approved_usd"] == 70
    assert result["stream_results"][ORDER[0]]["raw_capacity_usd"] == 1000
    saved = h.observations.data["paper_execution"]
    assert saved["cohort"] == BOUNDARY.isoformat()
    assert saved["policy_version"] == h.policy["version"]
    assert saved["waterfall_semantics"] == h.policy["waterfall_semantics"]
    assert saved["clock_evidence_sha256"] == h.clock["receipt_sha256"]
    assert saved["result"] == result
    assert state["inputs"] == h.observations.data["model_inputs"]
    h.p.cycle()
    assert h.account.overview()["order_count"] == 2
    assert h.observations.data["paper_execution"] == saved


@pytest.mark.parametrize("top_state", ["NO_SIGNAL", "NOT_DUE"])
def test_v5_ineligible_top_needs_no_market_and_lower_real_score_is_executed(harness, top_state):
    h = harness
    enable_v5(h)
    h.state.capacity = 1000
    if top_state == "NOT_DUE":
        h.state.boundary += timedelta(minutes=5)
        h.state.upper += timedelta(minutes=5)
        h.clock["observed_at"] = h.state.boundary.isoformat()
        h.refresh_markets()
    else:
        original = h.p.prepare_decisions
        def explicit_no_signal(policy, inputs, cohort):
            decisions = original(policy, inputs, cohort)
            top = decisions[ORDER[0]]
            # An explicit offline ineligible-score fixture, never runtime data.
            top["prediction"]["threshold_frozen"] = abs(top["prediction"]["z_v4"]) + 1
            top["prediction"]["selected"] = False
            top["status"] = "NO_SIGNAL"
            top.pop("signal")
            return decisions
        h.p.prepare_decisions = explicit_no_signal
    h.observations.data["polymarket"]["streams"].pop(ORDER[0])
    h.p.cycle()
    result = h.p.status()["execution"]
    assert result["stream_results"][ORDER[0]]["status"] == top_state
    assert result["stream_results"][ORDER[1]]["status"] == "FILLED_BUDGET"
    assert h.account.overview()["order_count"] == 2
    assert result["cohort"]["used_usd"] == 220
    assert all(market_id != "101" for market_id, _ in h.state.book_calls)


def test_v5_known_high_price_is_saved_as_skip_before_lower_model_fill(harness):
    h = harness
    enable_v5(h)
    h.state.capacity = 1000
    original = h.p.refresh_book
    def priced_book(market, side):
        book = original(market, side)
        if market["market_id"] == "101":
            book["asks"] = [{"price": .60, "size": 1000}]
        return book
    h.p.refresh_book = priced_book
    h.p.cycle()
    result = h.p.status()["execution"]
    assert result["stream_results"][ORDER[0]]["status"] == "PRICE_LIMIT_NOT_MET"
    assert result["stream_results"][ORDER[1]]["status"] == "FILLED_BUDGET"
    assert result["cohort"]["used_usd"] == 220
    assert h.account.overview()["order_count"] == 2
    assert h.observations.data["paper_execution"]["result"] == result


def test_live_http_pacing_is_explicit_and_replacement_closes_old_client(harness, monkeypatch):
    from tools.argon4_forward import online
    h = harness
    original = online.OnlineInputs
    opened, closed = [], []

    def tracked(runtime, **kwargs):
        instance = original(runtime, **kwargs)
        opened.append(kwargs)
        instance.close = lambda: closed.append(instance)
        return instance

    monkeypatch.setattr(online, "OnlineInputs", tracked)
    h.policy["input_http_min_spacing_seconds"] = .25
    h.policy_path.write_text(json.dumps(h.policy))
    h.p.cycle()
    initial_orders = h.account.overview()["order_count"]
    assert opened[0]["http_min_spacing_seconds"] == .25
    assert opened[0]["time_basis"] == "CALIBRATED_UTC_UPPER_BOUND"
    h.policy["input_http_min_spacing_seconds"] = .5
    h.policy_path.write_text(json.dumps(h.policy))
    h.p.cycle()
    assert len(closed) == 1 and len(opened) == 2
    assert opened[1]["http_min_spacing_seconds"] == .5
    assert h.state.refreshes == 1  # replacing transport never replays the same window
    assert h.account.overview()["order_count"] == initial_orders


@pytest.mark.parametrize("spacing", [True, 0, .249, 1.001, float("nan"), float("inf"), "0.25"])
def test_unsafe_live_http_pacing_is_rejected_without_input_or_orders(harness, spacing):
    h = harness
    h.policy["input_http_min_spacing_seconds"] = spacing
    h.policy_path.write_text(json.dumps(h.policy))
    with pytest.raises(ValueError, match="request spacing"):
        h.p.cycle()
    assert h.state.refreshes == 0 and h.state.book_calls == []
    assert h.account.overview()["order_count"] == 0


def test_cold_warmup_keeps_default_pacing_and_closes_collector(harness, monkeypatch):
    from tools.argon4_forward import online
    h = harness
    original = online.OnlineInputs
    calls, closed = [], []

    def tracked(runtime, **kwargs):
        calls.append(kwargs)
        instance = original(runtime, **kwargs)
        instance.close = lambda: closed.append(True)
        return instance

    monkeypatch.setattr(online, "OnlineInputs", tracked)
    h.p.warmup_complete = False
    h.policy["input_http_min_spacing_seconds"] = .25
    h.policy_path.write_text(json.dumps(h.policy))
    h.p.cycle()
    assert calls == [{}] and closed == [True]
    assert h.account.overview()["order_count"] == 0
    assert h.p.status()["status"] == "PAPER_WARMUP"


@pytest.mark.parametrize("phase", ["features", "inference"])
def test_slow_computation_cannot_admit_a_signal_after_twenty_seconds(harness, monkeypatch, phase):
    h = harness
    name = "build_feature_frames" if phase == "features" else "predict_stream"
    original = getattr(model, name)

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        h.state.upper = BOUNDARY + timedelta(seconds=20)
        return result

    monkeypatch.setattr(model, name, slow)
    with pytest.raises(ValueError, match="missed feature decision window"):
        h.p.cycle()
    assert h.state.book_calls == [] and h.account.overview()["order_count"] == 0
    assert "prediction_decisions" not in h.observations.data


def test_latency_evidence_preserves_pinned_model_scores_and_records_future_window(harness):
    h = harness
    h.policy["input_http_min_spacing_seconds"] = .25
    h.policy_path.write_text(json.dumps(h.policy))
    h.p.cycle()
    state = h.p.status()
    timing = h.observations.latest("cycle_latency")
    assert timing == state["timing"]
    assert timing["cohort"] == BOUNDARY.isoformat()
    assert timing["deadline_seconds"] == 20 and timing["deadline_margin_seconds"] > 0
    assert timing["http_min_spacing_seconds"] == .25
    assert "started_monotonic" not in timing
    for field in ("input_refresh_seconds", "mark_registration_seconds", "decisions_seconds",
                  "causal_wait_seconds", "execution_seconds", "duration_seconds"):
        assert timing[field] >= 0
    assert set(timing["computation"]["features_seconds"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}
    for stream, decision in state["decisions"].items():
        assert decision["prediction"]["z_v4"] == pytest.approx(h.trained.scores[stream]["score"], abs=1e-12)
        assert decision["prediction"]["threshold_frozen"] == 0
    assert h.p._pending_timing is None


def test_worker_preserves_late_failure_evidence_and_closes_transport(harness, monkeypatch):
    h = harness
    h.state.refresh_delay = 13
    closed, iterations = [], []
    original_stop = h.p.stop
    monkeypatch.setattr(original_stop, "is_set", lambda: bool(iterations))
    original_wait = original_stop.wait

    def last_wait(seconds):
        iterations.append(True)
        return original_wait(seconds)

    monkeypatch.setattr(original_stop, "wait", last_wait)
    from tools.argon4_forward import online
    original = online.OnlineInputs

    def tracked(runtime, **kwargs):
        instance = original(runtime, **kwargs)
        instance.close = lambda: closed.append(True)
        return instance

    monkeypatch.setattr(online, "OnlineInputs", tracked)
    h.p.worker()
    timing = h.observations.latest("cycle_latency")
    assert timing["status"] == "FAILED" and "missed feature decision window" in timing["error"]
    assert timing["deadline_margin_seconds"] <= 0
    assert h.account.overview()["order_count"] == 0 and closed == [True]


def test_fast_stage_one_keeps_eth_btc_priority_limits_and_one_frame_per_asset(harness):
    h = harness
    enable_v5(h)
    enabled = ["ETH_5m", "BTC_15m", "BTC_5m"]
    h.policy.update(order=enabled + [s for s in ORDER if s not in enabled],
                    input_http_min_spacing_seconds=.25,
                    execution_stage={"id": "latency-stage-fixture", "status": "CONFIRMED",
                                     "enabled_streams": enabled, "effective_from": BOUNDARY.isoformat()})
    h.policy_path.write_text(json.dumps(h.policy))
    h.state.capacity = 1000
    before = h.policy_path.read_bytes()
    h.p.cycle()
    status = h.p.status()
    assert list(status["decisions"]) == h.policy["order"]
    assert [x[0] for x in h.state.feature_calls] == ["ETHUSDT", "BTCUSDT"]
    assert list(status["timing"]["computation"]["inference_seconds"]) == enabled
    assert status["execution"]["stream_results"]["ETH_5m"]["spent_usd"] == 150
    assert status["execution"]["stream_results"]["BTC_15m"]["spent_usd"] == 70
    assert status["execution"]["stream_results"]["BTC_5m"] == {
        "status": "UNREACHED", "reason": "COHORT_BUDGET_EXHAUSTED"}
    assert h.account.overview()["order_count"] == 2
    assert status["execution"]["cohort"]["used_usd"] == 220
    assert h.policy_path.read_bytes() == before
    assert all(not v.get("model_evaluated", True) for k, v in status["decisions"].items() if k not in enabled)
    assert all(float(price["price"]) <= .515 for fill in status["execution"]["fills"] for price in fill["fills"])
