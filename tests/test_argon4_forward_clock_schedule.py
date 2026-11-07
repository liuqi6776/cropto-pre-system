"""Clock collection is scheduled earlier, never accepted for longer."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tools.argon4_forward import pipeline


BOUNDARY = datetime(2026, 11, 6, 16, 0, tzinfo=timezone.utc)


@pytest.fixture
def schedule(tmp_path, monkeypatch):
    state = SimpleNamespace(monotonic=100., lower=BOUNDARY-timedelta(seconds=5), calls=[], saved=[], fail=False)
    p = pipeline.ForwardPipeline(None, tmp_path,
        SimpleNamespace(save=lambda kind, data: state.saved.append((kind, data))))
    p.clock = {"status": "CALIBRATED", "calibrated_monotonic": 70.}
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: state.monotonic)
    monkeypatch.setattr(pipeline, "calibrated_now", lambda clock, upper=True: state.lower)
    def calibrate(*, transport):
        assert transport is p.public_transport
        state.calls.append(state.lower)
        return {"status": "UNCALIBRATED" if state.fail else "CALIBRATED", "calibrated_monotonic": state.monotonic}
    monkeypatch.setattr(pipeline, "calibrate", calibrate)
    yield p, state
    p.close()


@pytest.mark.parametrize("before,age,expected", [(11,30,False),(10,20,False),(10,20.01,True),
    (1,39,True),(0,39,False),(-5,39,False),(-5,40.01,True),(20,41,True)])
def test_refresh_only_when_periodic_or_approaching_next_boundary(schedule,before,age,expected):
    p,state = schedule
    state.lower = BOUNDARY-timedelta(seconds=before)
    p.clock["calibrated_monotonic"] = state.monotonic-age
    p.update_clock()
    assert bool(state.calls) is expected
    assert len(state.saved) == int(expected)
    if expected:
        assert p.clock["refresh_reason"] == ("PERIODIC_OR_INITIAL" if age>40 else "PRE_DECISION_WINDOW")


def test_successful_precalibration_is_not_repeated_at_same_boundary(schedule):
    p,state = schedule
    p.update_clock()
    assert len(state.calls) == 1
    for elapsed in (1,4,10,15,23):
        state.monotonic = 100+elapsed
        state.lower = BOUNDARY+timedelta(seconds=elapsed-5)
        p.update_clock()
    assert len(state.calls) == 1  # normal +5..18 decision window has fresh clock


def test_failed_precalibration_does_not_fall_back_to_old_success(schedule):
    p,state = schedule
    state.fail = True
    with pytest.raises(ValueError,match="clock uncertainty"):
        p.update_clock()
    assert p.clock["status"] == "UNCALIBRATED"
    assert len(state.calls) == 1 and len(state.saved) == 1


def test_clock_step_detected_during_scheduling_is_not_ignored(schedule,monkeypatch):
    p,state = schedule
    def stepped(*args,**kwargs):
        raise ValueError("local clock changed after calibration")
    monkeypatch.setattr(pipeline,"calibrated_now",stepped)
    with pytest.raises(ValueError,match="local clock changed"):
        p.update_clock()
    assert not state.calls


@pytest.mark.parametrize("before,expected_calls", [(5, 1), (1, 0), (.1, 0)])
def test_v13_proactive_sampling_budget_stops_before_boundary(schedule, monkeypatch, before, expected_calls):
    from tools.argon4_forward.runtime_profile import V12_ENGINEERING
    p, state = schedule
    p._revision_fastpath = {"enabled": True}
    p._worker_engineering = dict(V12_ENGINEERING)
    state.lower = BOUNDARY-timedelta(seconds=before)
    calls = []
    def calibrate(**kwargs):
        calls.append(kwargs)
        return {"status": "CALIBRATED", "calibrated_monotonic": state.monotonic}
    monkeypatch.setattr(pipeline, "calibrate", calibrate)
    p.update_clock()
    assert len(calls) == expected_calls
    if calls:
        assert calls[0]["retry_borderline"]
        assert calls[0]["budget_monotonic"] < state.monotonic+before


def test_v13_periodic_clock_inside_decision_window_never_adds_samples(schedule, monkeypatch):
    from tools.argon4_forward.runtime_profile import V12_ENGINEERING
    p, state = schedule
    p._revision_fastpath = {"enabled": True}
    p._worker_engineering = dict(V12_ENGINEERING)
    state.lower = BOUNDARY+timedelta(seconds=5)
    p.clock["calibrated_monotonic"] = state.monotonic-41
    calls = []
    def calibrate(**kwargs):
        calls.append(kwargs)
        return {"status": "CALIBRATED", "calibrated_monotonic": state.monotonic}
    monkeypatch.setattr(pipeline, "calibrate", calibrate)
    p.update_clock()
    assert len(calls) == 1 and not calls[0]["retry_borderline"]
    assert calls[0]["budget_monotonic"] <= state.monotonic+6
