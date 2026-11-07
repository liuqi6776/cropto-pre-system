"""Coordinator deadline covers worker publication locks and filesystem stalls."""
from concurrent.futures import wait
import json
from pathlib import Path
import threading
import time

import pandas as pd

from tools.argon4_forward import online
from test_argon4_forward_online import FixtureHTTP


NOW = pd.Timestamp("2026-11-06T15:12:00Z")


def collector_fixture(tmp_path, monkeypatch):
    collector = online.OnlineInputs(tmp_path, clock=lambda: NOW, client=FixtureHTTP(lambda: NOW))
    collector.root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(collector, "_check_clock", lambda: {"status": "VERIFIED", "received_at": NOW.isoformat()})
    monkeypatch.setattr(collector, "_candles", lambda *args: None)
    monkeypatch.setattr(collector, "_funding", lambda *args: None)
    monkeypatch.setattr(collector, "_dvol", lambda *args: None)
    monkeypatch.setattr(collector, "_basis", lambda: None)
    monkeypatch.setattr(collector, "snapshot", lambda: {"status": "READY"})
    return collector


def bounded_timeout(collector, *, seconds=.2):
    started = time.monotonic()
    result = collector.refresh(deadline_monotonic=started+seconds, cohort_id="expired")
    assert time.monotonic()-started < seconds+.5
    assert result["status"] == "BLOCKED"
    assert "DEADLINE" in result["errors"][0]["error"]
    assert collector._active_generation is None
    return result


def release_workers(collector, event):
    event.set()
    _, pending = wait(collector._outstanding, timeout=3)
    collector.close()
    assert not pending


def test_held_publication_lock_cannot_hold_timeout_or_admit_new_generation(tmp_path, monkeypatch):
    collector = collector_fixture(tmp_path, monkeypatch)
    before = json.dumps({"marker": "old state"})
    collector.state_path.write_text(before)
    locked, release = threading.Event(), threading.Event()
    def holder():
        with collector._publication_lock:
            locked.set()
            release.wait(3)
    thread = threading.Thread(target=holder)
    thread.start()
    assert locked.wait(1)
    try:
        bounded_timeout(collector)
        assert collector.state["clock"]["status"] == "UNAVAILABLE"
        busy = collector.refresh(deadline_monotonic=time.monotonic()+.1, cohort_id="too soon")
        assert busy["errors"][0]["error"] == "INPUT_WORKERS_BUSY"
        release.set()
        thread.join(2)
        _, pending = wait(collector._outstanding, timeout=2)
        assert not pending
        assert collector.state_path.read_text() == before
        assert collector.refresh(deadline_monotonic=time.monotonic()+2, cohort_id="future")["status"] == "READY"
        assert json.loads(collector.state_path.read_text())["cohort_id"] == "future"
    finally:
        release_workers(collector, release)
        thread.join(2)


def test_stalled_state_file_write_retires_without_late_state_publication(tmp_path, monkeypatch):
    collector = collector_fixture(tmp_path, monkeypatch)
    before = json.dumps({"marker": "old state"})
    collector.state_path.write_text(before)
    entered, release = threading.Event(), threading.Event()
    original = Path.write_text
    def stalled_write(path, text, *args, **kwargs):
        result = original(path, text, *args, **kwargs)
        if path.name.startswith("online_state.json.") and path.name.endswith(".tmp"):
            entered.set()
            release.wait(3)
        return result
    monkeypatch.setattr(Path, "write_text", stalled_write)
    try:
        bounded_timeout(collector)
        assert entered.is_set()
        assert collector.state_path.read_text() == before
        blocked_state = dict(collector.state)
        release.set()
        _, pending = wait(collector._outstanding, timeout=2)
        assert not pending
        assert collector.state == blocked_state
        assert collector.state_path.read_text() == before
        assert not list(collector.root.glob("online_state.json.*.tmp"))
        assert collector.refresh(deadline_monotonic=time.monotonic()+2, cohort_id="future")["status"] == "READY"
        assert json.loads(collector.state_path.read_text())["cohort_id"] == "future"
    finally:
        release_workers(collector, release)


def test_funding_hash_stall_is_outside_gate_and_cannot_publish_retired_cache(tmp_path, monkeypatch):
    collector = collector_fixture(tmp_path, monkeypatch)
    actual_funding = online.OnlineInputs._funding.__get__(collector)
    actual_funding("BTCUSDT", NOW)
    collector._funding_cache.clear()
    path = collector._path("funding", "BTCUSDT")
    entered, release = threading.Event(), threading.Event()
    original = Path.read_bytes
    def stalled_hash_read(target):
        result = original(target)
        if target == path:
            entered.set()
            release.wait(3)
        return result
    monkeypatch.setattr(Path, "read_bytes", stalled_hash_read)
    monkeypatch.setattr(collector, "_funding", lambda symbol, cutoff: actual_funding(symbol, cutoff) if symbol == "BTCUSDT" else None)
    try:
        bounded_timeout(collector, seconds=.35)
        assert entered.is_set()
        assert collector._publication_lock.acquire(blocking=False)
        collector._publication_lock.release()
        accepted_before_timeout = original(path)
        release.set()
        _, pending = wait(collector._outstanding, timeout=2)
        assert not pending
        assert collector._funding_cache == {}
        assert original(path) == accepted_before_timeout
        assert not collector.state_path.exists()
    finally:
        release_workers(collector, release)


def test_atomic_rename_stall_never_publishes_ready_in_memory_or_starts_new_generation(tmp_path, monkeypatch):
    collector = collector_fixture(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = Path.replace
    def stalled_rename(path, target):
        if target == collector.state_path:
            entered.set()
            release.wait(3)
        return original(path, target)
    monkeypatch.setattr(Path, "replace", stalled_rename)
    try:
        bounded_timeout(collector)
        assert entered.is_set()
        blocked_state = dict(collector.state)
        busy = collector.refresh(deadline_monotonic=time.monotonic()+.1)
        assert busy["errors"][0]["error"] == "INPUT_WORKERS_BUSY"
        release.set()
        _, pending = wait(collector._outstanding, timeout=2)
        assert not pending
        # An atomic OS operation already entered cannot be canceled, so its
        # eventual disk effect is possible. It may not certify a valid batch.
        assert collector.state == blocked_state
        assert collector.state["clock"]["status"] == "UNAVAILABLE"
        assert collector.refresh(deadline_monotonic=time.monotonic()+2, cohort_id="fresh validation")["status"] == "READY"
        assert json.loads(collector.state_path.read_text())["cohort_id"] == "fresh validation"
    finally:
        release_workers(collector, release)
