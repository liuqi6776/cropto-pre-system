"""Late rate limits are durable safety evidence, never accepted model inputs."""
from concurrent.futures import wait
import json
from pathlib import Path
import sqlite3
import threading
import time

import pandas as pd
import pytest

from tools.argon4_forward import online
from test_argon4_forward_deadline_publication import collector_fixture


NOW = pd.Timestamp("2026-11-06T15:12:00Z")


@pytest.mark.parametrize("status", [418, 429])
@pytest.mark.parametrize("expired_clock", [False, True])
def test_headers_after_retirement_keep_restart_backoff_without_fake_receipt(
        tmp_path, monkeypatch, status, expired_clock):
    collector = collector_fixture(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    state = {"clock_expired": False, "calls": 0}
    def request_clock():
        if state["clock_expired"]:
            raise ValueError("calibration expired")
        return NOW
    class Session:
        def get(self, url, **kwargs):
            state["calls"] += 1
            entered.set()
            release.wait(3)
            state["clock_expired"] = expired_clock
            class Response:
                status_code = status
                headers = {"Retry-After": "120"}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_content(self, size):
                    pytest.fail("expired response must not be accepted as a completed input")
            return Response()
        def close(self): pass
    monkeypatch.setattr(online.requests, "Session", Session)
    monkeypatch.setattr(online, "_now", lambda: NOW)
    client = online.PublicInputsHTTP(collector.root / "http_receipts.sqlite3", clock=request_clock, min_spacing=0)
    collector.client = client
    collector.clock = request_clock
    monkeypatch.setattr(collector, "_check_clock", online.OnlineInputs._check_clock.__get__(collector))
    try:
        started = time.monotonic()
        result = collector.refresh(deadline_monotonic=started+.15, cohort_id="expired")
        assert time.monotonic()-started < .65
        assert entered.is_set() and result["status"] == "BLOCKED"
        release.set()
        _, pending = wait(collector._outstanding, timeout=2)
        assert not pending
        evidence = json.loads((collector.root / "binance_backoff.json").read_text())
        assert pd.Timestamp(evidence["blocked_until_utc"]) == NOW+pd.Timedelta(seconds=135)
        assert evidence["evidence_type"] == "HTTP_STATUS_AND_HEADERS_ONLY"
        assert evidence["source_receipt_sha256"] is None
        assert evidence["backoff_time_basis"] == (
            "SYSTEM_UTC_UNCALIBRATED_BACKOFF_ONLY" if expired_clock else "CAPTURED_REQUEST_CLOCK")
        with sqlite3.connect(client.path) as db:
            assert not db.execute("SELECT name FROM sqlite_master WHERE name='receipts'").fetchall()
        restarted = online.PublicInputsHTTP(client.path, clock=lambda: NOW, min_spacing=0)
        try:
            with pytest.raises(ValueError, match="PUBLIC_RATE_LIMIT_BACKOFF"):
                restarted.get("spot_clock")
        finally:
            restarted.close()
        assert state["calls"] == 1
    finally:
        release.set()
        wait(collector._outstanding, timeout=3)
        collector.close()


def test_rate_limit_persistence_stall_does_not_hold_coordinator_deadline(tmp_path, monkeypatch):
    collector = collector_fixture(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    class Session:
        def get(self, url, **kwargs):
            class Response:
                status_code = 429
                headers = {"Retry-After": "120"}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def iter_content(self, size): yield b'{"code":-1003}'
            return Response()
        def close(self): pass
    monkeypatch.setattr(online.requests, "Session", Session)
    client = online.PublicInputsHTTP(collector.root / "http_receipts.sqlite3", clock=lambda: NOW, min_spacing=0)
    collector.client = client
    monkeypatch.setattr(collector, "_check_clock", online.OnlineInputs._check_clock.__get__(collector))
    original = Path.write_text
    def stalled_write(path, text, *args, **kwargs):
        result = original(path, text, *args, **kwargs)
        if path.name.startswith("binance_backoff.json."):
            entered.set()
            release.wait(3)
        return result
    monkeypatch.setattr(Path, "write_text", stalled_write)
    try:
        started = time.monotonic()
        result = collector.refresh(deadline_monotonic=started+.15)
        assert time.monotonic()-started < .65
        assert result["status"] == "BLOCKED" and entered.is_set()
        assert collector.refresh(deadline_monotonic=time.monotonic()+.1)["errors"][0]["error"] == "INPUT_WORKERS_BUSY"
        release.set()
        _, pending = wait(collector._outstanding, timeout=2)
        assert not pending
        restarted = online.PublicInputsHTTP(client.path, clock=lambda: NOW, min_spacing=0)
        try:
            with pytest.raises(ValueError, match="PUBLIC_RATE_LIMIT_BACKOFF"):
                restarted.get("clock")
        finally:
            restarted.close()
    finally:
        release.set()
        wait(collector._outstanding, timeout=3)
        collector.close()


def test_durable_write_failure_is_explicit_and_future_batches_fail_closed(tmp_path, monkeypatch):
    collector = collector_fixture(tmp_path, monkeypatch)
    client = online.PublicInputsHTTP(collector.root / "http_receipts.sqlite3", clock=lambda: NOW, min_spacing=0)
    collector.client = client
    original = Path.replace
    def failed_write(path, target):
        if Path(target).name == "binance_backoff.json":
            raise OSError("fixture storage failure")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", failed_write)
    try:
        with pytest.raises(ValueError, match="PUBLIC_BACKOFF_PERSISTENCE_FAILED"):
            client._record_backoff("binance", None, "120", 429)
        assert client.backoff_persistence_failure == "OSError"
        with pytest.raises(ValueError, match="PUBLIC_BACKOFF_PERSISTENCE_FAILED"):
            client.get("clock")
        result = collector.refresh(deadline_monotonic=time.monotonic()+.1)
        assert result["status"] == "BLOCKED"
        assert result["errors"][0]["error"] == "PUBLIC_BACKOFF_PERSISTENCE_FAILED"
    finally:
        collector.close()


def test_sqlite_unavailable_still_has_durable_host_guard(tmp_path):
    client = online.PublicInputsHTTP(tmp_path / "receipts.sqlite3", clock=lambda: NOW, min_spacing=0)
    db = sqlite3.connect(client.path)
    db.execute("CREATE TABLE existing(value TEXT)")
    db.commit()
    db.execute("BEGIN EXCLUSIVE")
    try:
        client._record_backoff("binance", None, "120", 429)
        assert client.backoff_persistence_failure is None
        assert client.backoff_persistence_warning == "SQLITE_UNAVAILABLE_SHARED_HOST_GUARD_DURABLE"
        restarted = online.PublicInputsHTTP(client.path, clock=lambda: NOW, min_spacing=0)
        try:
            started = time.monotonic()
            with pytest.raises(ValueError, match="PUBLIC_RATE_LIMIT_BACKOFF"):
                restarted.get("spot_clock")
            assert time.monotonic()-started < .3
        finally:
            restarted.close()
    finally:
        db.rollback()
        db.close()
        client.close()
