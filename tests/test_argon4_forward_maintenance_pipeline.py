"""Offline window-external maintenance: timing, durability and no replay."""
from copy import deepcopy
from datetime import timedelta
import json
import sqlite3
import time
from concurrent.futures import Future

import pytest

from test_argon4_forward_pipeline import (BOUNDARY, harness, trained_artifacts,
    forbid_network, enable_v13_fixture)
from tools.argon4_forward import pipeline, revisions, online


def arm(h, monkeypatch):
    enable_v13_fixture(h, monkeypatch)
    h.state.input_status = "BLOCKED"
    h.state.input_errors = [{"source": "futures:BNBUSDT:5m",
        "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"}]
    h.state.input_gaps = []
    monkeypatch.setattr(h.p, "reconcile_current_cohort", lambda p, e, c, i: (i, None))
    monkeypatch.setattr(h.p, "reconcile_inputs", lambda *a: {"status": "WAITING_CONFIRMATION"})
    h.p.cycle()
    assert h.p.status()["timing"]["status"] == "INPUTS_NOT_VALIDATED"
    h.state.upper = BOUNDARY+timedelta(seconds=70.2)
    return h


def claims(h):
    with sqlite3.connect(h.observations.path) as db:
        return [json.loads(r[0]) for r in db.execute(
            "SELECT payload FROM snapshots WHERE kind='input_maintenance_claim' ORDER BY id")]


@pytest.mark.parametrize("upper,allowed", [(69.999, False), (70, False), (70.199, False),
    (70.2, True), (179, True), (180, False), (180.001, False), (300.2, False)])
def test_phase_uses_both_clock_bounds(harness, monkeypatch, upper, allowed):
    h = arm(harness, monkeypatch)
    h.state.upper = BOUNDARY+timedelta(seconds=upper)
    h.state.input_status, h.state.input_errors = "READY", []
    initial = h.state.refreshes
    assert h.p.maintain_blocked_inputs(h.policy, {}, BOUNDARY) is allowed
    assert h.state.refreshes-initial == int(allowed)
    assert len(claims(h)) == int(allowed)
    assert not h.state.feature_calls and h.account.overview()["order_count"] == 0


@pytest.mark.parametrize("condition", ["paused", "stopped", "changed_policy", "clock_expired",
    "retired_worker", "prepared", "pending_execution", "used_cohort", "pristine_false"])
def test_admission_refusals_do_not_claim_or_refresh(harness, monkeypatch, condition):
    h = arm(harness, monkeypatch)
    if condition == "paused": (h.p.runtime/"verification_pause.json").write_text("{}")
    if condition == "stopped": monkeypatch.setattr(h.p.stop, "is_set", lambda: True)
    if condition == "changed_policy": h.policy_path.write_text(json.dumps({**h.policy, "version": "changed"}))
    if condition == "clock_expired": h.clock["calibrated_monotonic"] = time.monotonic()-60
    if condition == "retired_worker": h.p.inputs._outstanding = [Future()]
    if condition == "prepared":
        monkeypatch.setattr(revisions, "ensure_view_committed", lambda r: (_ for _ in ()).throw(ValueError("PREPARED")))
    if condition == "pending_execution": h.p._pending_execution = {"cohort": BOUNDARY}
    if condition == "used_cohort":
        h.observations.save("prediction_decisions", {"cohort": BOUNDARY.isoformat(), "decisions": {}})
    if condition == "pristine_false": monkeypatch.setattr(h.p, "_cohort_pristine", lambda c: {"eligible": False})
    initial = h.state.refreshes
    assert not h.p.maintain_blocked_inputs(h.policy, {}, BOUNDARY)
    assert h.state.refreshes == initial and claims(h) == []


@pytest.mark.parametrize("failure", ["pause", "stop", "policy", "clock_epoch", "clock_jump", "retired_worker"])
def test_refresh_postcheck_prevents_reconciliation(harness, monkeypatch, failure):
    h = arm(harness, monkeypatch)
    refresh = h.p.inputs.refresh
    def changed(*a, **kw):
        inputs = refresh(*a, **kw)
        if failure == "pause": (h.p.runtime/"maintenance_pause.json").write_text("{}")
        if failure == "stop": monkeypatch.setattr(h.p.stop, "is_set", lambda: True)
        if failure == "policy": h.policy_path.write_text(json.dumps({**h.policy, "version": "changed"}))
        if failure == "clock_epoch": h.p.clock = {**h.clock, "receipt_sha256": "e"*64}
        if failure == "clock_jump": monkeypatch.setattr(pipeline, "calibrated_now", lambda *a, **k: (_ for _ in ()).throw(ValueError("CLOCK_JUMP")))
        if failure == "retired_worker": h.p.inputs._outstanding = [Future()]
        return inputs
    monkeypatch.setattr(h.p.inputs, "refresh", changed)
    monkeypatch.setattr(revisions, "reconcile_confirmed_price_revisions", lambda *a, **kw: pytest.fail("unsafe archive"))
    assert h.p.maintain_blocked_inputs(h.policy, {}, BOUNDARY)
    assert len(claims(h)) == 1
    result = h.observations.latest("input_maintenance_result")
    assert result["result"]["status"] == "BLOCKED"
    assert result["next_eligible_at"] is None


def test_two_attempts_wait_from_completion_upper_and_keep_original_timing(harness, monkeypatch):
    h = arm(harness, monkeypatch)
    timing = deepcopy(h.p.status()["timing"])
    h.state.refresh_delay = 3
    seen = []
    def reconcile(*a, **kw):
        assert "fast_context" not in kw
        assert kw["maintenance_guard"]("BEFORE_VALIDATION")["eligible"]
        seen.append(kw["now"]())
        return {"status": "WAITING_CONFIRMATION"}
    monkeypatch.setattr(revisions, "reconcile_confirmed_price_revisions", reconcile)
    h.p.cycle()
    first = h.observations.latest("input_maintenance_result")
    assert first["next_eligible_at"] == (BOUNDARY+timedelta(seconds=133.2)).isoformat()
    assert h.p.last_cohort == BOUNDARY and h.p._pending_execution is None
    h.state.upper = BOUNDARY+timedelta(seconds=133.399)
    h.p.cycle()
    assert len(claims(h)) == 1
    h.state.upper = BOUNDARY+timedelta(seconds=133.4)
    h.p.clock = {**h.clock, "calibrated_monotonic": time.monotonic(), "receipt_sha256": "e"*64}
    h.p.cycle()
    assert [c["sequence"] for c in claims(h)] == [1, 2]
    assert len(seen) == 2 and (seen[1]-seen[0]).total_seconds() >= 60
    h.p.cycle()
    assert len(seen) == 2
    assert h.p.status()["timing"] == timing
    assert not h.state.feature_calls and h.account.overview()["order_count"] == 0


@pytest.mark.parametrize("status", [None, "READY_FOR_FUTURE_WINDOW", "COMMITTED", "BLOCKED", "WAITING_NOT_DUE", "WAITING_BAD_POLICY"])
def test_durable_claim_refuses_unproven_second_attempt(harness, monkeypatch, status):
    h = arm(harness, monkeypatch)
    deadline = time.monotonic()+10
    first = h.p._claim_maintenance(h.policy, BOUNDARY, h.clock, deadline)
    if status is not None:
        h.observations.save("input_maintenance_result", {"cohort": BOUNDARY.isoformat(), "sequence": 1,
            "result": {"status": "WAITING_CONFIRMATION" if status.startswith("WAITING") else status},
            "policy_sha256": "wrong" if status == "WAITING_BAD_POLICY" else first["policy_sha256"],
            "next_eligible_at": (BOUNDARY+timedelta(seconds=130)).isoformat()})
    if status == "WAITING_BAD_POLICY": h.state.upper = BOUNDARY+timedelta(seconds=131)
    h.p._blocked_input_maintenance = {"cohort": BOUNDARY, "policy": deepcopy(h.policy), "done": False}
    initial = h.state.refreshes
    assert not h.p.maintain_blocked_inputs(h.policy, {}, BOUNDARY)
    assert h.state.refreshes == initial and len(claims(h)) == 1


@pytest.mark.parametrize("status", ["READY", "COMMITTED", "BLOCKED"])
def test_terminal_maintenance_never_reenters_current_decision(harness, monkeypatch, status):
    h = arm(harness, monkeypatch)
    if status == "READY": h.state.input_status, h.state.input_errors = "READY", []
    monkeypatch.setattr(revisions, "reconcile_confirmed_price_revisions", lambda *a, **k: {"status": status})
    h.p.cycle()
    h.state.upper = BOUNDARY+timedelta(seconds=160)
    h.p.cycle()
    assert len(claims(h)) == 1
    assert not h.state.feature_calls and h.account.overview()["order_count"] == 0
    assert h.p.last_cohort == BOUNDARY
    # A later window must refresh again and can still reject fresh bad inputs.
    h.state.boundary = BOUNDARY+timedelta(minutes=5)
    h.state.upper = h.state.boundary+timedelta(seconds=4.2)
    h.state.input_status = "BLOCKED"
    h.state.input_errors = [{"source": "fresh", "error": "HTTP_TIMEOUT"}]
    prior = h.state.refreshes
    h.p.cycle()
    assert h.state.refreshes == prior+1 and not h.state.feature_calls


def test_preserves_mixed_errors_and_cannot_maintain_legacy_profile(harness, monkeypatch):
    h = arm(harness, monkeypatch)
    h.state.input_errors.append({"source": "mark:ETHUSDT:1h", "error": "HTTP_TIMEOUT"})
    received = []
    def reconcile(runtime, errors, **kw):
        received.extend(deepcopy(errors))
        return {"status": "BLOCKED", "reason": "UNSUPPORTED_REVISION_ERROR"}
    monkeypatch.setattr(revisions, "reconcile_confirmed_price_revisions", reconcile)
    h.p.cycle()
    assert received == h.state.input_errors
    assert h.p._blocked_input_maintenance["done"]
    h.p._revision_fastpath = {"enabled": False}
    h.p._blocked_input_maintenance["done"] = False
    assert not h.p.maintain_blocked_inputs(h.policy, {}, BOUNDARY)


def test_writer_busy_is_bounded_and_claim_is_consumed(harness, monkeypatch):
    h = arm(harness, monkeypatch)
    class Busy:
        def acquire(self, *, timeout):
            assert 0 <= timeout <= .1
            return False
    h.p.inputs._lock = Busy()
    monkeypatch.setattr(revisions, "reconcile_confirmed_price_revisions", lambda *a, **kw: pytest.fail("writer busy"))
    assert h.p.maintain_blocked_inputs(h.policy, {}, BOUNDARY)
    assert h.observations.latest("input_maintenance_result")["result"]["reason"] == "MAINTENANCE_WRITER_BUSY"
    assert len(claims(h)) == 1


@pytest.mark.parametrize("change", ["pause", "policy", "clock", "deadline"])
def test_pristine_io_cannot_hide_late_permission_or_budget_loss(harness, monkeypatch, change):
    h = arm(harness, monkeypatch)
    check = h.p._cohort_pristine
    deadline = time.monotonic()+14
    def slow_check(cohort):
        result = check(cohort)
        if change == "pause": (h.p.runtime/"verification_pause.json").write_text("{}")
        if change == "policy": h.policy_path.write_text(json.dumps({**h.policy, "version": "changed"}))
        if change == "clock": h.p.clock = {**h.clock, "receipt_sha256": "e"*64}
        if change == "deadline": monkeypatch.setattr(pipeline.time, "monotonic", lambda: deadline)
        return result
    monkeypatch.setattr(h.p, "_cohort_pristine", slow_check)
    with pytest.raises(ValueError):
        h.p._maintenance_guard(h.policy, BOUNDARY, deepcopy(h.clock), deadline, "BEFORE_CUTOVER")
    assert not claims(h)
