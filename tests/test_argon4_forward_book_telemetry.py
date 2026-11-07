"""Public observations remain timely evidence, never simulated fills."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import sqlite3
import threading
import time

import pytest

from tools.argon4_forward import book_telemetry as mod
from tools.argon4_forward.polymarket_transport import SOURCE, public_url


START = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
ORDER = ["ETH_5m", "BTC_15m", "BTC_5m", "SOL_15m", "SOL_5m", "ETH_15m", "BNB_15m"]


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def policy():
    return {"status": "CONFIRMED", "waterfall_semantics": "ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1",
            "order": ORDER, "execution_stage": {"id": "stage1", "status": "CONFIRMED",
            "enabled_streams": ORDER[:3], "effective_from": iso(START-timedelta(days=1))}}


def snapshot(start=START, fee=None, enabled=None):
    streams = {}
    for index, stream in enumerate(enabled or ORDER[:3]):
        duration = 900 if stream.endswith("15m") else 300
        asset = stream.split("_")[0].lower()
        market = {"status": "VERIFIED_MARKET", "market_id": str(index+1),
                  "condition_id": "0x" + str(index+1)*64,
                  "slug": f"{asset}-updown-{duration//60}m-{int(start.timestamp())}",
                  "window_start": iso(start), "window_end": iso(start+timedelta(seconds=duration)),
                  "outcome_ids": {"UP": str(index*2+101), "DOWN": str(index*2+102)},
                  "fee": fee or {"status": "KNOWN_SCHEDULE", "fees_enabled": True, "rate": .02,
                                 "exponent": 1, "taker_only": True},
                  "metadata_received_at": iso(start-timedelta(seconds=10)), "metadata_sha256": "a"*64}
        streams[stream] = {"current": market, "next": None}
    return {"streams": streams}


def rows(service, *, status=None):
    with sqlite3.connect(service.path) as db:
        query = "SELECT payload_json FROM samples" + (" WHERE status=?" if status else "")
        return [json.loads(row[0]) for row in db.execute(query, (status,) if status else ())]


def wait_idle(service):
    deadline = time.monotonic()+3
    while service.status()["active_requests"] and time.monotonic() < deadline:
        time.sleep(.01)
    assert service.status()["active_requests"] == 0


@pytest.fixture
def harness(tmp_path, monkeypatch):
    state = {"now": START-timedelta(seconds=1), "snapshot": snapshot(), "calls": [], "response_delay": .05,
             "uncertainty": 0.0, "http_status": 200, "gate": None, "source_age": 0.0, "omit_asks": False}
    def clock():
        return {"status": "CALIBRATED", "source": "POLYMARKET_CLOB_SERVER_TIME", "receipt_sha256": "b"*64,
                "server_minus_local_seconds": 0, "uncertainty_seconds": state["uncertainty"]}
    def now(clock, *, upper=True):
        if state.get("bad_clock"):
            raise ValueError("test clock gap")
        return state["now"] + timedelta(seconds=(1 if upper else -1)*clock["uncertainty_seconds"])
    monkeypatch.setattr(mod, "calibrated_now", now)
    class FakeTransport:
        def get(self, kind, token):
            assert kind == "book"
            requested = state["now"]
            state["calls"].append((kind, token, requested))
            if state["gate"] is not None:
                state["gate"].wait(3)
            received = state["now"] + timedelta(seconds=state["response_delay"])
            index = (int(token)-101)//2
            raw = {"market": "0x"+str(index+1)*64, "asset_id": token,
                   "timestamp": str(int((received-timedelta(seconds=state["source_age"])).timestamp()*1000)),
                   "bids": [{"price": ".49", "size": "10"}],
                   "asks": [{"price": ".51", "size": "10"}, {"price": ".515", "size": "20"}, {"price": ".52", "size": "500"}],
                   "hash": "fixture-book-hash"}
            if state["omit_asks"]:
                del raw["asks"]
            text = json.dumps(raw)
            receipt = {"source": SOURCE, "method": "GET", "url": public_url("book", token),
                       "requested_at": iso(requested), "received_at": iso(received),
                       "http_status": state["http_status"], "body_complete": True,
                       "raw_text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
            return raw, receipt, None if state["http_status"] == 200 else f"HTTP_{state['http_status']}"
        def close(self):
            pass
    service = mod.BookTelemetry(tmp_path, policy_provider=policy, market_provider=lambda: state["snapshot"],
                                clock_provider=clock, transport_factory=FakeTransport)
    state["service"] = service
    yield state
    if state["gate"]:
        state["gate"].set()
    service.close(wait=True)
    assert not any(worker.is_alive() for worker in service._workers)


def arm_and_sample(state, offset=2):
    state["service"].tick()
    state["now"] = START+timedelta(seconds=offset+.1)
    state["service"].tick()
    wait_idle(state["service"])


def test_first_tick_arms_only_future_due_stage_and_both_outcomes(harness):
    service = harness["service"]
    service.tick()
    planned = rows(service)
    assert len(planned) == 3*2*7
    assert {r["stream"] for r in planned} == set(ORDER[:3])
    assert {r["outcome"] for r in planned} == {"UP", "DOWN"}
    assert {r["target_offset_seconds"] for r in planned} == set(mod.OFFSETS)
    assert harness["calls"] == []
    assert all(r["cohort"] == iso(START) and not r["execution_enabled"] for r in planned)


def test_start_midwindow_never_catches_up_old_samples(harness):
    service = harness["service"]
    harness["now"] = START+timedelta(seconds=1)
    service.tick()
    planned = rows(service)
    assert {r["cohort"] for r in planned} == {iso(START+timedelta(minutes=5))}
    assert {r["stream"] for r in planned} == {"ETH_5m", "BTC_5m"}
    assert harness["calls"] == []


def test_both_sides_observed_before_model_readiness_and_fees_match_executor(harness):
    service = harness["service"]
    arm_and_sample(harness)
    observed = rows(service, status="OBSERVED")
    assert len(observed) == 6
    assert len(harness["calls"]) == 6
    for row in observed:
        assert row["best_bid"] == .49 and row["best_ask"] == .51
        expected = 10*(Decimal(".51")+Decimal(".02")*Decimal(".51")*Decimal(".49"))
        expected += 20*(Decimal(".515")+Decimal(".02")*Decimal(".515")*Decimal(".485"))
        assert Decimal(row["capacity"]["usd"]) == expected
        assert Decimal(row["capacity"]["shares"]) == 30
        assert row["source_book_age_upper_seconds"] == 0
        assert row["receipt"]["method"] == "GET"
        assert row["evidence_class"] == mod.EVIDENCE_CLASS
        assert not row["capacity"]["is_fill_estimate"]


def test_scheduler_late_is_gap_not_historical_get(harness):
    service = harness["service"]
    service.tick()
    harness["now"] = START+timedelta(seconds=5)
    service.tick()
    missed = rows(service, status="MISSED_SAMPLE")
    assert len(missed) == 6 and {r["reason"] for r in missed} == {"SCHEDULER_LATE"}
    assert harness["calls"] == []
    assert all(r["best_ask"] is None for r in missed)


def test_late_response_keeps_raw_receipt_without_claiming_target_quote(harness):
    harness["response_delay"] = 2
    arm_and_sample(harness)
    missed = rows(harness["service"], status="MISSED_SAMPLE")
    assert len(missed) == 6
    assert all(r["reason"] == "RESPONSE_LATE_RAW_RETAINED" and r["receipt"]["raw_text"] for r in missed)
    assert all(r["best_ask"] is None and r["capacity"] is None for r in missed)


def test_t30_and_t60_are_only_new_readonly_observations(harness):
    service = harness["service"]
    service.tick()
    for offset in (30, 60):
        harness["now"] = START+timedelta(seconds=offset+.1)
        service.tick()
        wait_idle(service)
    observed = rows(service, status="OBSERVED")
    assert len(observed) == 12
    assert {r["target_offset_seconds"] for r in observed} == {30, 60}
    assert len(harness["calls"]) == 12
    assert all(not r["execution_enabled"] and not r["live_enabled"] for r in observed)


def test_clock_uncertainty_cannot_disguise_missed_sampling_deadline(harness):
    service = harness["service"]
    service.tick()
    harness["uncertainty"] = .45
    harness["now"] = START+timedelta(seconds=2.7)
    service.tick()
    assert len(rows(service, status="MISSED_SAMPLE")) == 6
    assert harness["calls"] == []


@pytest.mark.parametrize("fee", [{"status": "UNKNOWN", "rate": None},
                                  {"status": "KNOWN_SCHEDULE", "fees_enabled": True, "rate": .02,
                                   "exponent": 2, "taker_only": True}])
def test_missing_or_unsupported_fee_never_becomes_free_capacity(harness, fee):
    harness["snapshot"] = snapshot(fee=fee)
    arm_and_sample(harness)
    observed = rows(harness["service"], status="OBSERVED")
    assert len(observed) == 6
    assert all(r["capacity"]["status"] == "UNKNOWN_FEE" and r["capacity"]["usd"] is None for r in observed)


def test_explicit_disabled_fee_is_zero_and_missing_book_is_unknown():
    book = {"status": "OBSERVED", "asks": [{"price": .515, "size": 10}, {"price": .52, "size": 50}]}
    fee = {"status": "KNOWN_DISABLED", "fees_enabled": False}
    assert mod.fee_inclusive_capacity(book, fee)["usd"] == "5.150000"
    assert mod.fee_inclusive_capacity({"status": "INVALID"}, fee)["usd"] is None


@pytest.mark.parametrize("key,value", [("source_age", 3), ("source_age", -1), ("omit_asks", True)])
def test_stale_future_or_missing_depth_never_yields_capacity(harness, key, value):
    harness[key] = value
    arm_and_sample(harness)
    invalid = rows(harness["service"], status="INVALID_OBSERVATION")
    assert len(invalid) == 6
    assert all(r["capacity"] is None and r["best_ask"] is None for r in invalid)


def test_unknown_metadata_receipt_cannot_attest_fees(harness):
    for state in harness["snapshot"]["streams"].values():
        state["current"]["metadata_sha256"] = None
    arm_and_sample(harness)
    assert all(r["capacity"]["status"] == "FEE_METADATA_UNVERIFIED" for r in rows(harness["service"], status="OBSERVED"))


def test_bounded_requests_do_not_queue_later_targets_or_hold_tick(harness):
    service = harness["service"]
    harness["gate"] = threading.Event()
    service.tick()
    harness["now"] = START+timedelta(seconds=2.1)
    service.tick()
    started = time.monotonic()
    while len(harness["calls"]) < 6 and time.monotonic()-started < 2:
        time.sleep(.01)
    assert len(harness["calls"]) == 6
    harness["now"] = START+timedelta(seconds=6.1)
    started = time.monotonic()
    service.tick()
    assert time.monotonic()-started < .5
    assert service.status()["active_requests"] == 6
    assert len(harness["calls"]) == 6
    assert len([r for r in rows(service) if r.get("reason") == "OBSERVATION_WORKERS_BUSY"]) == 6
    harness["gate"].set()
    wait_idle(service)


def test_http_rate_limit_stops_following_samples_without_retries(harness):
    harness["http_status"] = 429
    arm_and_sample(harness)
    service = harness["service"]
    first_calls = len(harness["calls"])
    assert 1 <= first_calls <= 6
    assert service.status()["backoff_until"]
    harness["now"] = START+timedelta(seconds=6.1)
    service.tick()
    wait_idle(service)
    assert len(harness["calls"]) == first_calls
    assert any(r.get("reason") == "HOST_BACKOFF" for r in rows(service))


def test_restart_marks_unfinished_samples_and_arms_next_future(harness):
    original = harness["service"]
    original.tick()
    original.close(wait=True)
    harness["now"] = START+timedelta(seconds=10)
    restarted = mod.BookTelemetry(original.path.parent, policy_provider=policy,
        market_provider=lambda: harness["snapshot"], clock_provider=original.clock_provider,
        transport_factory=original.transport_factory)
    try:
        restarted.tick()
        old = [r for r in rows(restarted) if r["cohort"] == iso(START)]
        assert len(old) == 42
        assert all(r["status"] == "MISSED_SAMPLE" and r["reason"] == "PROCESS_RESTART" for r in old)
        assert restarted.status()["last_armed_cohort"] == iso(START+timedelta(minutes=5))
        assert harness["calls"] == []
    finally:
        restarted.close(wait=True)


def test_clock_gap_cannot_dispatch_or_backfill(harness):
    service = harness["service"]
    service.tick()
    harness["bad_clock"] = True
    harness["now"] = START+timedelta(seconds=2.1)
    service.tick()
    assert service.status()["status"] == "WAITING_VALID_CLOCK"
    assert harness["calls"] == []
    harness["bad_clock"] = False
    harness["now"] = START+timedelta(seconds=4)
    service.tick()
    assert len(rows(service, status="MISSED_SAMPLE")) == 6
    assert harness["calls"] == []


def test_decision_association_preserves_actual_ready_time_and_samples(harness):
    service = harness["service"]
    ledger = service.path.parent / "paper_account.sqlite3"
    ledger.write_bytes(b"sentinel unrelated ledger")
    arm_and_sample(harness)
    before = rows(service, status="OBSERVED")
    ready = START+timedelta(seconds=12)
    prediction = {"feature_bar_open": iso(START-timedelta(minutes=5)), "decision_time": iso(ready), "selected": False}
    service.decision_provider = lambda: {"timing": {"cohort": iso(START)},
        "decisions": {"ETH_5m": {"status": "NO_SIGNAL", "prediction": prediction}},
        "inputs": {"feature_available_at": iso(START+timedelta(seconds=11))}}
    service._poll_decisions()
    with sqlite3.connect(service.path) as db:
        linked = json.loads(db.execute("SELECT payload_json FROM decision_links").fetchone()[0])
    assert linked["decision"]["prediction"]["decision_time"] == iso(ready)
    assert linked["inputs_ready_at"] == iso(START+timedelta(seconds=11))
    assert rows(service, status="OBSERVED") == before
    assert ledger.read_bytes() == b"sentinel unrelated ledger"
    assert not service.status()["counts_as_fills"]


def test_wrong_window_model_cannot_link_to_samples(harness):
    service = harness["service"]
    service.decision_provider = lambda: {"timing": {"cohort": iso(START)}, "decisions": {
        "ETH_5m": {"prediction": {"feature_bar_open": iso(START-timedelta(minutes=10)), "decision_time": iso(START)}}}}
    service._poll_decisions()
    with sqlite3.connect(service.path) as db:
        assert db.execute("SELECT count(*) FROM decision_links").fetchone()[0] == 0


def test_partial_decisions_can_link_remaining_streams_later(harness):
    service = harness["service"]
    decision = {"prediction": {"feature_bar_open": iso(START-timedelta(minutes=5)),
                               "decision_time": iso(START+timedelta(seconds=8))}}
    state = {"timing": {"cohort": iso(START)}, "decisions": {"ETH_5m": decision}}
    service.decision_provider = lambda: state
    service._poll_decisions()
    state["decisions"]["BTC_5m"] = decision
    service._poll_decisions()
    with sqlite3.connect(service.path) as db:
        assert db.execute("SELECT count(*) FROM decision_links").fetchone()[0] == 2


@pytest.fixture
def harness_v12(harness):
    old = harness["service"]
    old.close(wait=True)
    service = mod.BookTelemetry(old.path.parent, policy_provider=old.policy_provider,
        market_provider=old.market_provider, clock_provider=old.clock_provider,
        transport_factory=old.transport_factory, sampling_profile="clock_center_v12")
    harness["service"] = service
    yield harness
    if harness["gate"]:
        harness["gate"].set()
    service.close(wait=True)
    assert not any(worker.is_alive() for worker in service._workers)


def test_sampling_profile_is_explicit_and_invalid_profile_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="sampling profile"):
        mod.BookTelemetry(tmp_path / "invalid", policy_provider=policy,
            market_provider=lambda: {}, clock_provider=lambda: {}, sampling_profile="v12_typo")
    assert not (tmp_path / "invalid").exists()


def test_v12_center_dispatch_preserves_clock_overlap_not_falsely_observed(harness_v12):
    state = harness_v12
    state["uncertainty"] = .35
    arm_and_sample(state)
    collected = rows(state["service"], status="AMBIGUOUS_SAMPLE")
    assert len(state["calls"]) == len(collected) == 6
    assert not rows(state["service"], status="OBSERVED")
    for row in collected:
        assert row["reason"] == "REQUEST_TARGET_CLOCK_OVERLAP"
        quality = row["timing_quality"]
        assert quality["request_interval_relation"] == "CLOCK_OVERLAP_TARGET"
        assert quality["requested_offset_center_seconds"] == pytest.approx(.1)
        assert quality["received_offset_center_seconds"] == pytest.approx(.15)
        assert not quality["strict_target_window"] and not quality["exact_target_instant"]
        assert row["best_ask"] is None and row["capacity"] is None
        actual = row["actual_observation"]
        assert actual["best_ask"] == .51 and Decimal(actual["capacity"]["shares"]) == 30
        assert actual["capacity_scope"] == "ACTUAL_RECEIPT_NOT_TARGET_INSTANT"
        assert not actual["strict_target_window"] and not actual["is_fill_estimate"]
        assert actual["requested_at_server_lower"] == iso(START+timedelta(seconds=1.75))
        assert actual["requested_at_server_upper"] == iso(START+timedelta(seconds=2.45))
    status = state["service"].status()
    assert status["sampling_profile"] == "clock_center_v12"
    assert status["dispatch_time_basis"] == "CALIBRATED_CENTER"
    assert status["observed_requires_full_target_interval"]
    assert status["actual_quote_samples_this_process"] == 6
    assert status["actual_depth_samples_this_process"] == 6
    assert status["quality_counts_this_process"] == {"AMBIGUOUS_SAMPLE": 6}


def test_v12_strict_observed_remains_bounded_by_both_clock_sides(harness_v12):
    state = harness_v12
    state["uncertainty"] = .1
    state["service"].tick()
    state["now"] = START+timedelta(seconds=2.2)
    state["service"].tick()
    wait_idle(state["service"])
    collected = rows(state["service"], status="OBSERVED")
    assert len(collected) == 6
    for row in collected:
        assert row["timing_quality"]["strict_target_window"]
        assert row["timing_quality"]["request_interval_relation"] == "WITHIN_TARGET_WINDOW"
        assert row["timing_quality"]["response_interval_relation"] == "WITHIN_TARGET_WINDOW"
        assert row["capacity"] == row["actual_observation"]["capacity"]


def test_v12_late_complete_book_keeps_actual_capacity_without_relabelling(harness_v12):
    state = harness_v12
    state["response_delay"] = 2
    arm_and_sample(state)
    collected = rows(state["service"], status="MISSED_SAMPLE")
    assert len(collected) == 6
    for row in collected:
        assert row["reason"] == "RESPONSE_LATE_RAW_RETAINED"
        assert row["receipt"]["raw_text"]
        assert row["best_ask"] is None and row["capacity"] is None
        assert row["timing_quality"]["response_interval_relation"] == "AFTER_TARGET_WINDOW"
        assert not row["timing_quality"]["strict_target_window"]
        assert row["actual_observation"]["received_at_server_center"] == iso(START+timedelta(seconds=4.1))
        assert row["actual_observation"]["capacity"]["status"] == "OBSERVED_DEPTH"


def test_v12_lower_bound_before_deadline_does_not_prove_timeliness(harness_v12):
    state = harness_v12
    state["uncertainty"] = .35
    state["service"].tick()
    state["now"] = START+timedelta(seconds=2.8)
    state["service"].tick()
    wait_idle(state["service"])
    collected = rows(state["service"], status="MISSED_SAMPLE")
    assert len(collected) == 6
    for row in collected:
        assert row["reason"] == "RESPONSE_LATE_RAW_RETAINED"
        assert row["timing_quality"]["request_interval_relation"] == "CLOCK_OVERLAP_DEADLINE"
        assert row["timing_quality"]["response_interval_relation"] == "CLOCK_OVERLAP_DEADLINE"
        assert not row["timing_quality"]["strict_target_window"]
        assert row["actual_observation"]["capacity"]["usd"] is not None


def test_v12_centre_past_window_is_missed_without_backfill(harness_v12):
    state = harness_v12
    state["uncertainty"] = .35
    state["service"].tick()
    state["now"] = START+timedelta(seconds=3.1)
    state["service"].tick()
    assert not state["calls"]
    missed = rows(state["service"], status="MISSED_SAMPLE")
    assert len(missed) == 6
    assert all(row["reason"] == "SCHEDULER_LATE" and row["actual_observation"] is None for row in missed)


@pytest.mark.parametrize("key,value", [("source_age", 3), ("source_age", -1), ("omit_asks", True)])
def test_v12_late_invalid_depth_has_no_actual_capacity(harness_v12, key, value):
    state = harness_v12
    state["response_delay"] = 2
    state[key] = value
    arm_and_sample(state)
    invalid = rows(state["service"], status="INVALID_OBSERVATION")
    assert len(invalid) == 6
    assert all(row["actual_observation"] is None and row["capacity"] is None for row in invalid)


@pytest.mark.parametrize("fee", [{"status": "UNKNOWN"},
    {"status": "KNOWN_SCHEDULE", "fees_enabled": True, "rate": .02, "exponent": 2, "taker_only": True}])
def test_v12_late_unknown_fee_remains_unknown_capacity(harness_v12, fee):
    state = harness_v12
    state["snapshot"] = snapshot(fee=fee)
    state["response_delay"] = 2
    arm_and_sample(state)
    missed = rows(state["service"], status="MISSED_SAMPLE")
    assert len(missed) == 6
    assert all(row["actual_observation"]["capacity"]["status"] == "UNKNOWN_FEE"
               and row["actual_observation"]["capacity"]["usd"] is None for row in missed)


def test_v12_late_invalid_metadata_proof_does_not_attest_fee(harness_v12):
    state = harness_v12
    state["response_delay"] = 2
    for market in state["snapshot"]["streams"].values():
        market["current"]["metadata_sha256"] = None
    arm_and_sample(state)
    missed = rows(state["service"], status="MISSED_SAMPLE")
    assert len(missed) == 6
    assert all(row["actual_observation"]["capacity"]["status"] == "FEE_METADATA_UNVERIFIED"
               and row["actual_observation"]["capacity"]["usd"] is None for row in missed)


def test_v12_derivatives_use_hashed_raw_body_not_alternate_decoded_payload(harness_v12):
    state = harness_v12
    original = state["service"].transport_factory
    class AlternateDecoded(original):
        def get(self, *args):
            raw, receipt, error = super().get(*args)
            raw["asks"][0]["price"] = ".505"
            return raw, receipt, error
    state["service"].transport_factory = AlternateDecoded
    arm_and_sample(state)
    observed = rows(state["service"], status="OBSERVED")
    assert len(observed) == 6
    assert all(row["best_ask"] == .51 and row["actual_observation"]["best_ask"] == .51 for row in observed)


def test_v12_backwards_receipt_has_no_usable_depth(harness_v12):
    state = harness_v12
    state["response_delay"] = -.05
    arm_and_sample(state)
    invalid = rows(state["service"], status="INVALID_OBSERVATION")
    assert len(invalid) == 6
    assert all(row["reason"] == "RECEIPT_TIMING_INVALID" and row["actual_observation"] is None for row in invalid)


def test_v12_workers_start_before_targets_and_remain_bounded(harness_v12):
    state = harness_v12
    service = state["service"]
    state["gate"] = threading.Event()
    service.tick()
    assert len(service._workers) == mod.MAX_WORKERS
    assert not state["calls"] and not service.status()["active_requests"]
    state["now"] = START+timedelta(seconds=2.1)
    service.tick()
    deadline = time.monotonic()+2
    while len(state["calls"]) < 6 and time.monotonic() < deadline:
        time.sleep(.01)
    assert len(state["calls"]) == 6
    state["now"] = START+timedelta(seconds=6.1)
    started = time.monotonic()
    service.tick()
    assert time.monotonic()-started < .5
    assert len(state["calls"]) == 6
    assert service.status()["active_requests"] == service.status()["max_requests"] == 12
    assert service.status()["queued_requests"] == 6
    state["now"] = START+timedelta(seconds=10.1)
    service.tick()
    assert len(state["calls"]) == 6 and service.status()["active_requests"] == 12
    assert len([row for row in rows(service) if row.get("reason") == "OBSERVATION_WORKERS_BUSY"]) == 6
    state["gate"].set()
    wait_idle(service)
    assert len(state["calls"]) == 6
    assert len([row for row in rows(service) if row.get("reason") == "REQUEST_SLOT_LATE"]) == 6


def test_v12_stage_two_ten_samples_use_six_workers_with_bounded_queue(harness_v12):
    state = harness_v12
    service = state["service"]
    enabled = ORDER[:5]
    def stage_two_policy():
        result = policy()
        result["execution_stage"]["enabled_streams"] = enabled
        return result
    service.policy_provider = stage_two_policy
    state["snapshot"] = snapshot(enabled=enabled)
    arm_and_sample(state)
    observed = rows(service, status="OBSERVED")
    assert len(observed) == len(state["calls"]) == 10
    assert {row["stream"] for row in observed} == set(enabled)
    assert len(service._workers) == service.status()["max_workers"] == 6
    assert service.status()["max_requests"] == 12
    assert not service.status()["queued_requests"] and not service.status()["active_requests"]
    assert service.status()["actual_depth_samples_this_process"] == 10
    assert not service.status()["execution_enabled"]
