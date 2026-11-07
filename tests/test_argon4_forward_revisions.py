"""Explicit fixtures only; no production views, network or account mutations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlencode

import pandas as pd
import pytest

from tools.argon4_forward import revisions as revisions

OPEN = pd.Timestamp("2026-11-06T01:25:00Z")
NOW = pd.Timestamp("2026-11-06T01:45:00Z")
OLD = [int(OPEN.timestamp()*1000), "100.00", "110.00", "90.00", "101.00", "10.000",
       int((OPEN+pd.Timedelta(minutes=5)).timestamp()*1000)-1, "1000.000", 20, "5.000", "500.000", "0"]
NEW = [*OLD]
NEW[4], NEW[5], NEW[7], NEW[8], NEW[9], NEW[10] = "102.00", "11.000", "1102.000", 22, "5.500", "551.000"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("revision recovery attempted network"))


def _row(raw):
    return {"date": pd.to_datetime(raw[0], unit="ms", utc=True),
            "open_time": pd.to_datetime(raw[0], unit="ms", utc=True),
            "close_time": pd.to_datetime(raw[6], unit="ms", utc=True),
            **{name: float(raw[i]) for i, name in revisions.PRICE_FIELDS.items()},
            "first_received_at": pd.Timestamp("2026-11-06T01:30:03Z"), "receipt_sha256": "a"*64,
            "receipt_time_basis": "CALIBRATED_UTC_UPPER_BOUND", "clock_evidence_sha256": "b"*64}


def _error(symbol="BTCUSDT"):
    return {"source": f"futures:{symbol}:5m", "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"}


def _add_receipt(runtime, received="2026-11-06T01:35:04Z", bar=None, *, symbol="BTCUSDT",
                 bad_hash=False, endpoint="https://fapi.binance.com/fapi/v1/klines", include_new_bar=True,
                 http_status=200, params_extra=None):
    bar = deepcopy(NEW if bar is None else bar)
    payload = [bar]
    if include_new_bar:
        new_bar = deepcopy(bar)
        new_bar[0] += 300000
        new_bar[6] += 300000
        payload.append(new_bar)
    raw = json.dumps(payload).encode()
    params = {"symbol": symbol, "interval": "5m", "startTime": bar[0], "endTime": payload[-1][6], "limit": 1000}
    params.update(params_extra or {})
    url = endpoint + "?" + urlencode(params)
    received = pd.Timestamp(received)
    digest = hashlib.sha256(raw).hexdigest()
    with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
        conn.execute("INSERT INTO receipts(url,requested_at,received_at,sha256,http_status,raw,time_basis,clock_evidence_sha256) VALUES(?,?,?,?,?,?,?,?)",
            (url, (received-pd.Timedelta(seconds=.4)).isoformat(), received.isoformat(), "f"*64 if bad_hash else digest,
             http_status, raw, "CALIBRATED_UTC_UPPER_BOUND", "c"*64))
    return digest


@pytest.fixture
def runtime(tmp_path):
    runtime = tmp_path / "runtime"
    raw = runtime / "features/raw_data"
    (raw / "intraday").mkdir(parents=True)
    (raw / "auxiliary").mkdir()
    previous = deepcopy(OLD)
    previous[0] -= 300000
    previous[6] -= 300000
    for symbol in ("BTCUSDT", "ETHUSDT"):
        pd.DataFrame([_row(previous), _row(OLD)]).to_parquet(raw / "intraday" / f"{symbol}_5m_futures.parquet", index=False)
    (raw / "auxiliary/unrelated_reference.txt").write_text("explicit unchanged fixture", encoding="utf-8")
    with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
        conn.execute("CREATE TABLE receipts(id INTEGER PRIMARY KEY,url TEXT,requested_at TEXT,received_at TEXT,sha256 TEXT,http_status INTEGER,raw BLOB,time_basis TEXT,clock_evidence_sha256 TEXT)")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        digest = _add_receipt(runtime, "2026-11-06T01:30:03Z", OLD, symbol=symbol, include_new_bar=False)
        path = raw / "intraday" / f"{symbol}_5m_futures.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.date == OPEN, "receipt_sha256"] = digest
        frame.loc[frame.date == OPEN, "clock_evidence_sha256"] = "c"*64
        frame.to_parquet(path, index=False)
    with sqlite3.connect(runtime / "observations.sqlite3") as conn:
        conn.execute("CREATE TABLE decisions(id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO decisions(content) VALUES(?)", ("old decision uses original receipt, not a revised history",))
    return runtime


def _files(runtime):
    return revisions._inventory(runtime / "features/raw_data")


def test_unconfirmed_revision_changes_nothing(runtime):
    _add_receipt(runtime)
    before = _files(runtime)
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert result["status"] == "WAITING_CONFIRMATION"
    assert _files(runtime) == before
    assert not (runtime / "features/revisions").exists()


@pytest.mark.parametrize("case,expected", [
    ("different", "WAITING_CONFIRMATION"), ("time_short", "WAITING_CONFIRMATION"),
    ("future", "BLOCKED"), ("bad_hash", "BLOCKED"), ("wrong_endpoint", "BLOCKED"),
    ("extra_parameter", "BLOCKED"), ("unclosed", "BLOCKED"), ("http_error", "WAITING_CONFIRMATION"),
    ("raw_ignored_diff", "WAITING_CONFIRMATION"), ("same_request", "WAITING_CONFIRMATION"),
])
def test_invalid_confirmation_never_mutates_cache(runtime, case, expected):
    _add_receipt(runtime)
    kwargs = {"received": "2026-11-06T01:40:05Z"}
    if case == "different":
        other = deepcopy(NEW)
        other[4] = "103.00"
        kwargs["bar"] = other
    elif case == "time_short": kwargs["received"] = "2026-11-06T01:35:40Z"
    elif case == "future": kwargs["received"] = "2026-11-06T01:46:00Z"
    elif case == "bad_hash": kwargs["bad_hash"] = True
    elif case == "wrong_endpoint": kwargs["endpoint"] = "https://elsewhere.invalid/fapi/v1/klines"
    elif case == "extra_parameter": kwargs["params_extra"] = {"apiKey": "fixture-never-a-secret"}
    elif case == "unclosed": kwargs.update(received="2026-11-06T01:29:59Z", include_new_bar=False)
    elif case == "http_error": kwargs["http_status"] = 418
    elif case == "raw_ignored_diff":
        other = deepcopy(NEW)
        other[11] = "1"
        kwargs["bar"] = other
    _add_receipt(runtime, **kwargs)
    if case == "same_request":
        with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
            conn.execute("UPDATE receipts SET requested_at=? WHERE received_at=?",
                ((pd.Timestamp("2026-11-06T01:35:04Z")-pd.Timedelta(seconds=.4)).isoformat(), pd.Timestamp(kwargs["received"]).isoformat()))
    before = _files(runtime)
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert result["status"] == expected, result
    assert _files(runtime) == before
    assert not (runtime / "features/revisions").exists()


def test_confirmed_revision_archives_full_old_view_and_changes_only_existing_target(runtime):
    first = _add_receipt(runtime)
    latest = _add_receipt(runtime, "2026-11-06T01:40:05Z")
    raw = runtime / "features/raw_data"
    path = raw / "intraday/BTCUSDT_5m_futures.parquet"
    old = pd.read_parquet(path)
    before = _files(runtime)
    stamps = iter((NOW, NOW+pd.Timedelta(seconds=3)))
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], lambda: next(stamps))
    assert result["status"] == "COMMITTED", result
    assert result["view_effective_at"] == (NOW+pd.Timedelta(seconds=3)).isoformat()
    assert result["current_cohort_must_be_skipped"] is True
    archive = Path(result["archive_path"])
    assert revisions._inventory(archive / "raw_data") == before
    pd.testing.assert_frame_equal(pd.read_parquet(archive / "raw_data/intraday/BTCUSDT_5m_futures.parquet"), old)
    current = pd.read_parquet(path)
    assert len(current) == len(old)  # receipt's new 01:30 bar is never inserted
    assert current.columns.equals(old.columns)
    pd.testing.assert_series_equal(current.iloc[0], old.iloc[0])
    assert current.iloc[-1].close == 102
    assert current.iloc[-1].first_received_at == pd.Timestamp("2026-11-06T01:40:05Z")
    assert current.iloc[-1].receipt_sha256 == latest
    after = _files(runtime)
    assert {k:v for k,v in after.items() if not k.endswith("BTCUSDT_5m_futures.parquet")} == {
        k:v for k,v in before.items() if not k.endswith("BTCUSDT_5m_futures.parquet")}
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["snapshot_meaning"] == "PRE_REVISION_WORKING_VIEW_NOT_A_HISTORICAL_DECISION_TIME_SNAPSHOT"
    assert manifest["model_weights_modified_by_this_operation"] is False
    assert [r["sha256"] for r in manifest["changes"][0]["confirmation_receipts"]] == [first, latest]
    with sqlite3.connect(archive / "receipts.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 4
    with sqlite3.connect(archive / "observations.sqlite3") as conn:
        assert conn.execute("SELECT content FROM decisions").fetchone()[0].startswith("old decision")
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"


def test_batch_waits_until_every_conflicting_source_is_confirmed(runtime):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    _add_receipt(runtime, symbol="ETHUSDT")
    before = _files(runtime)
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error(), _error("ETHUSDT")], NOW)
    assert result["status"] == "WAITING_CONFIRMATION"
    assert _files(runtime) == before
    _add_receipt(runtime, "2026-11-06T01:40:05Z", symbol="ETHUSDT")
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error(), _error("ETHUSDT")], NOW)
    assert result["status"] == "COMMITTED", result
    assert {c["symbol"] for c in result["changes"]} == {"BTCUSDT", "ETHUSDT"}


def test_recovery_is_idempotent_and_later_normal_append_is_allowed(runtime):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert result["status"] == "COMMITTED", result
    before = _files(runtime)
    assert revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)["status"] == "NO_CHANGE"
    assert _files(runtime) == before
    path = runtime / "features/raw_data/intraday/BTCUSDT_5m_futures.parquet"
    frame = pd.read_parquet(path)
    # Explicit fixture standing for a subsequent independently observed bar.
    extra = deepcopy(NEW)
    extra[0] += 300000
    extra[6] += 300000
    pd.concat([frame, pd.DataFrame([_row(extra)])], ignore_index=True).to_parquet(path, index=False)
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"


def test_prepared_marker_blocks_and_empty_runtime_is_allowed(runtime):
    assert revisions.ensure_view_committed(runtime) == {"status": "NO_REVISIONS"}
    folder = runtime / "features/revisions"
    folder.mkdir()
    (folder / "view_state.json").write_text(json.dumps({"schema_version": 1, "status": "PREPARED", "revision_id": "fixture"}))
    with pytest.raises(revisions.RevisionSafetyError, match="^REVISION_VIEW_NOT_COMMITTED$"):
        revisions.ensure_view_committed(runtime)
    assert revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)["status"] == "BLOCKED"


def test_failure_during_cutover_leaves_prepared_marker_and_old_evidence(runtime, monkeypatch):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    before = _files(runtime)
    replace = revisions.os.replace

    def interrupted(source, target):
        if str(target).endswith("BTCUSDT_5m_futures.parquet"):
            raise OSError("fixture interrupted cutover")
        return replace(source, target)

    monkeypatch.setattr(revisions.os, "replace", interrupted)
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert result["status"] == "BLOCKED"
    assert Path(result["archive_path"]).is_dir()
    assert revisions._inventory(Path(result["archive_path"]) / "raw_data") == before
    with pytest.raises(revisions.RevisionSafetyError, match="^REVISION_VIEW_NOT_COMMITTED$"):
        revisions.ensure_view_committed(runtime)


@pytest.mark.parametrize("source", ["BTCUSDT:funding", "mark:BTCUSDT:5m", "futures:OTHER:5m"])
def test_unsupported_sources_are_never_restored(runtime, source):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    before = _files(runtime)
    result = revisions.reconcile_confirmed_price_revisions(runtime,
        [{"source": source, "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"}], NOW)
    assert result["status"] == "BLOCKED"
    assert _files(runtime) == before
    assert not (runtime / "features/revisions").exists()


def test_changed_archive_is_detected_but_working_append_is_not_a_commit_failure(runtime):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    archived = Path(result["archive_path"]) / "raw_data/auxiliary/unrelated_reference.txt"
    archived.write_text("fixture corruption of supposedly immutable evidence")
    with pytest.raises(revisions.RevisionSafetyError, match="^REVISION_VIEW_NOT_COMMITTED$"):
        revisions.ensure_view_committed(runtime)


def test_insufficient_snapshot_space_blocks_before_prepared_marker(runtime, monkeypatch):
    from types import SimpleNamespace
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    before = _files(runtime)
    monkeypatch.setattr(revisions.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "INSUFFICIENT_SNAPSHOT_DISK_SPACE"
    assert result["disk_budget"]["reserve_bytes"] == 256*1024*1024
    assert result["disk_budget"]["required_free_bytes"] > result["disk_budget"]["reserve_bytes"]
    assert _files(runtime) == before
    assert not (runtime / "features/revisions").exists()


@pytest.mark.parametrize("corruption", ["missing_receipt", "wrong_old_value", "wrong_old_time", "wrong_clock"])
def test_existing_cache_requires_matching_original_receipt(runtime, corruption):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    path = runtime / "features/raw_data/intraday/BTCUSDT_5m_futures.parquet"
    frame = pd.read_parquet(path)
    if corruption == "missing_receipt": frame.loc[frame.date == OPEN, "receipt_sha256"] = "d"*64
    elif corruption == "wrong_old_value": frame.loc[frame.date == OPEN, "close"] = 100.5
    elif corruption == "wrong_old_time": frame.loc[frame.date == OPEN, "first_received_at"] = pd.Timestamp("2026-11-06T01:31:00Z")
    elif corruption == "wrong_clock": frame.loc[frame.date == OPEN, "clock_evidence_sha256"] = "d"*64
    frame.to_parquet(path, index=False)
    before = _files(runtime)
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert result["status"] == "BLOCKED", result
    assert result["reason"].startswith("CACHE_ORIGINAL_RECEIPT_")
    assert _files(runtime) == before
    assert not (runtime / "features/revisions").exists()
