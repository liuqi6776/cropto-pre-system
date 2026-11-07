"""Offline mark-1h recovery fixtures; never read or write the managed runtime."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlencode

import pandas as pd
import pytest

from tools.argon4_forward import revisions

OPEN = pd.Timestamp("2026-11-06T08:00:00Z")
NOW = pd.Timestamp("2026-11-06T10:20:00Z")
OLD_RECEIVED = pd.Timestamp("2026-11-06T09:00:03Z")
FIRST = pd.Timestamp("2026-11-06T10:05:04Z")
SECOND = pd.Timestamp("2026-11-06T10:10:05Z")
MARK_ENDPOINT = "https://fapi.binance.com/fapi/v1/markPriceKlines"
FUTURES_ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"
MARK_PATH = "auxiliary/ETHUSDT_1h_mark.parquet"
FUTURES_PATH = "intraday/BTCUSDT_5m_futures.parquet"


def bar(opened=OPEN, minutes=60, *, mark=True):
    # Explicit test numbers mirror the observed kind of close/count correction.
    return [int(opened.timestamp()*1000), "2220.00", "2235.00", "2210.00", "2225.5879992",
            "0" if mark else "10", int((opened+pd.Timedelta(minutes=minutes)).timestamp()*1000)-1,
            "0" if mark else "22255.879992", 3594 if mark else 20, "0", "0", "0"]


OLD_MARK = bar()
NEW_MARK = deepcopy(OLD_MARK)
NEW_MARK[4], NEW_MARK[8] = "2225.56431783", 3600
OLD_FUTURES = bar(OPEN+pd.Timedelta(minutes=55), 5, mark=False)
NEW_FUTURES = deepcopy(OLD_FUTURES)
NEW_FUTURES[4], NEW_FUTURES[8] = "2226.50", 24


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("mark recovery attempted network"))


def _error(source="mark:ETHUSDT:1h"):
    return {"source": source, "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"}


def _receipt(runtime, raw_bar=None, received=FIRST, *, endpoint=MARK_ENDPOINT, symbol="ETHUSDT",
             interval="1h", extra_bar=False, bad_hash=False, http_status=200,
             basis="CALIBRATED_UTC_UPPER_BOUND", clock_hash="d"*64, requested=None):
    raw_bar = deepcopy(NEW_MARK if raw_bar is None else raw_bar)
    payload = [raw_bar]
    if extra_bar:
        additional = deepcopy(raw_bar)
        step = 3_600_000 if interval == "1h" else 300_000
        additional[0] += step
        additional[6] += step
        payload.append(additional)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    received = pd.Timestamp(received)
    requested = pd.Timestamp(requested) if requested is not None else received-pd.Timedelta(milliseconds=400)
    params = {"symbol": symbol, "interval": interval, "startTime": raw_bar[0],
              "endTime": payload[-1][6], "limit": 1000}
    url = endpoint+"?"+urlencode(params)
    with sqlite3.connect(runtime/"features/http_receipts.sqlite3") as con:
        cursor = con.execute("INSERT INTO receipts(url,requested_at,received_at,sha256,http_status,raw,time_basis,clock_evidence_sha256) VALUES(?,?,?,?,?,?,?,?)",
            (url, requested.isoformat(), received.isoformat(), "f"*64 if bad_hash else digest,
             http_status, raw, basis, clock_hash))
        identity = cursor.lastrowid
    return {"id": identity, "sha256": digest, "received_at": received.isoformat(),
            "time_basis": basis, "clock_evidence_sha256": clock_hash, "url": url}


def _row(raw, receipt):
    return {"date": pd.to_datetime(raw[0], unit="ms", utc=True),
            "open_time": pd.to_datetime(raw[0], unit="ms", utc=True),
            "close_time": pd.to_datetime(raw[6], unit="ms", utc=True),
            **{name: float(raw[i]) for i, name in revisions.PRICE_FIELDS.items()},
            "first_received_at": pd.Timestamp(receipt["received_at"]), "receipt_sha256": receipt["sha256"],
            "receipt_time_basis": receipt["time_basis"], "clock_evidence_sha256": receipt["clock_evidence_sha256"],
            "untouched_fixture_annotation": "must remain byte-for-byte semantically identical"}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recover(runtime, errors, now):
    return revisions.reconcile_confirmed_price_revisions(
        runtime, errors, now, mark_registry_path=runtime/"fixture_mark_registry.json")


def _inventory(runtime):
    root = runtime/"features/raw_data"
    return {p.relative_to(root).as_posix(): _digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def runtime(tmp_path):
    runtime = tmp_path/"isolated_mark_runtime"
    root = runtime/"features/raw_data"
    (root/"intraday").mkdir(parents=True)
    (root/"auxiliary").mkdir()
    with sqlite3.connect(runtime/"features/http_receipts.sqlite3") as con:
        con.execute("CREATE TABLE receipts(id INTEGER PRIMARY KEY,url TEXT,requested_at TEXT,received_at TEXT,sha256 TEXT,http_status INTEGER,raw BLOB,time_basis TEXT,clock_evidence_sha256 TEXT)")
    old = _receipt(runtime, OLD_MARK, OLD_RECEIVED, basis="SYSTEM_UTC", clock_hash=None)
    previous = bar(OPEN-pd.Timedelta(hours=1))
    pd.DataFrame([_row(previous, old), _row(OLD_MARK, old)]).to_parquet(root/MARK_PATH, index=False)
    old_future = _receipt(runtime, OLD_FUTURES, OLD_RECEIVED, endpoint=FUTURES_ENDPOINT,
                          symbol="BTCUSDT", interval="5m", basis="SYSTEM_UTC", clock_hash=None)
    pd.DataFrame([_row(OLD_FUTURES, old_future)]).to_parquet(root/FUTURES_PATH, index=False)
    # Both real source families naturally coexist for ETH/1h. They must stay separate.
    own_future = _receipt(runtime, OLD_MARK, OLD_RECEIVED, endpoint=FUTURES_ENDPOINT,
                          basis="SYSTEM_UTC", clock_hash=None)
    pd.DataFrame([_row(OLD_MARK, own_future)]).to_parquet(root/"auxiliary/ETHUSDT_1h_futures.parquet", index=False)
    (root/"auxiliary/unrelated_funding_events.txt").write_text("one explicitly observed event; never grid-filled", encoding="utf-8")
    registry = {"schema_version": 1, "datasets": [], "research_references": [{
        "dataset_id": "argon4_local_v4_online_mark_ETHUSDT_20261106", "pair": "ETH/USDT:USDT",
        "timeframe": "1h", "candle_type": "mark", "local_path": str((root/MARK_PATH).resolve()),
        "sha256": _digest(root/MARK_PATH), "source_endpoint": MARK_ENDPOINT,
        "verification_status": "VERIFIED_PERPETUAL_LABEL", "installed": False,
        "research_only": True, "gate_effect": "NONE"}]}
    (runtime/"fixture_mark_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    with sqlite3.connect(runtime/"observations.sqlite3") as con:
        con.execute("CREATE TABLE decisions(id INTEGER PRIMARY KEY, model_hash TEXT, view_note TEXT)")
        con.execute("INSERT INTO decisions VALUES(1,?,?)", ("a"*64, "original decision retains its original working view"))
    with sqlite3.connect(runtime/"account.sqlite3") as con:
        con.execute("CREATE TABLE canary(content TEXT)")
        con.execute("INSERT INTO canary VALUES('no new orders, signals or simulated profit')")
    return runtime


def _assert_no_recovery(runtime, before, result):
    assert result["status"] in ("WAITING_CONFIRMATION", "BLOCKED", "NO_CHANGE"), result
    assert _inventory(runtime) == before
    assert not (runtime/"features/revisions").exists()


def test_confirmed_mark_revision_preserves_all_old_evidence_and_only_replaces_existing_row(runtime):
    first = _receipt(runtime, extra_bar=True)
    second = _receipt(runtime, received=SECOND, extra_bar=True)
    before = _inventory(runtime)
    path = runtime/"features/raw_data"/MARK_PATH
    original = pd.read_parquet(path)
    account_hash = _digest(runtime/"account.sqlite3")
    observations_hash = _digest(runtime/"observations.sqlite3")
    receipts_hash = _digest(runtime/"features/http_receipts.sqlite3")
    registry_before = (runtime/"fixture_mark_registry.json").read_bytes()
    stamps = iter((NOW, NOW+pd.Timedelta(seconds=4)))
    result = _recover(runtime, [_error()], lambda: next(stamps))
    assert result["status"] == "COMMITTED", result
    assert result["current_cohort_must_be_skipped"] is True
    assert result["next_cohort_requires_fresh_validation"] is True
    assert result["view_effective_at"] == (NOW+pd.Timedelta(seconds=4)).isoformat()
    current = pd.read_parquet(path)
    assert current.shape == original.shape and current.columns.equals(original.columns)
    pd.testing.assert_series_equal(current.iloc[0], original.iloc[0])
    allowed = list(revisions.PRICE_FIELDS.values())+list(revisions.PROVENANCE)
    pd.testing.assert_frame_equal(current.drop(columns=allowed), original.drop(columns=allowed))
    assert current.iloc[-1].close == float(NEW_MARK[4]) and current.iloc[-1]["count"] == 3600
    assert current.iloc[-1].first_received_at == SECOND
    assert current.iloc[-1].receipt_sha256 == second["sha256"]
    assert current.iloc[-1].receipt_time_basis == "CALIBRATED_UTC_UPPER_BOUND"
    assert current.iloc[-1].clock_evidence_sha256 == "d"*64
    assert pd.Timestamp("2026-11-06T09:00:00Z") not in set(current.date)  # No receipt-only bar is inserted.
    assert {k: v for k, v in _inventory(runtime).items() if k != MARK_PATH} == {k: v for k, v in before.items() if k != MARK_PATH}
    archive = Path(result["archive_path"])
    assert {p.relative_to(archive/"raw_data").as_posix(): _digest(p) for p in (archive/"raw_data").rglob("*") if p.is_file()} == before
    pd.testing.assert_frame_equal(pd.read_parquet(archive/"raw_data"/MARK_PATH), original)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    registry_after = (runtime/"fixture_mark_registry.json").read_bytes()
    registered = json.loads(registry_after)["research_references"][0]
    assert registered["sha256"] == _digest(path) and registered["rows"] == len(current)
    assert registered["source_endpoint"] == MARK_ENDPOINT and registered["installed"] is False
    assert registered["research_only"] is True and registered["gate_effect"] == "NONE"
    assert (archive/"mark_registry_before.json").read_bytes() == registry_before
    assert (archive/"mark_registry_proposed.json").read_bytes() == registry_after
    assert manifest["mark_registry_update"]["canonical_entries_modified"] is False
    change = manifest["changes"][0]
    assert change["before"]["close"] == float(OLD_MARK[4]) and change["after"]["close"] == float(NEW_MARK[4])
    assert change["before"]["count"] == 3594 and change["after"]["count"] == 3600
    assert change["provenance_before"]["receipt_time_basis"] == "SYSTEM_UTC"
    assert change["provenance_before"]["clock_evidence_sha256"] is None
    assert change["provenance_after"]["receipt_sha256"] == second["sha256"]
    assert [x["id"] for x in change["confirmation_receipts"]] == [first["id"], second["id"]]
    assert all(x["url"].startswith(MARK_ENDPOINT+"?") for x in change["confirmation_receipts"])
    assert manifest["model_weights_modified_by_this_operation"] is False
    assert manifest["working_view_for_future_cohorts_only"] is True
    assert manifest["current_view_must_not_replay_as_of_before"] == result["view_effective_at"]
    assert manifest["snapshot_meaning"] == "PRE_REVISION_WORKING_VIEW_NOT_A_HISTORICAL_DECISION_TIME_SNAPSHOT"
    with sqlite3.connect(archive/"receipts.sqlite3") as con:
        assert con.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 5
        assert con.execute("SELECT raw FROM receipts WHERE id=?", (first["id"],)).fetchone()
    with sqlite3.connect(archive/"observations.sqlite3") as con:
        assert con.execute("SELECT model_hash FROM decisions").fetchone()[0] == "a"*64
    assert _digest(runtime/"account.sqlite3") == account_hash
    assert _digest(runtime/"observations.sqlite3") == observations_hash
    assert _digest(runtime/"features/http_receipts.sqlite3") == receipts_hash
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"
    after = _inventory(runtime)
    assert _recover(runtime, [_error()], NOW+pd.Timedelta(minutes=5))["status"] == "NO_CHANGE"
    assert _inventory(runtime) == after


@pytest.mark.parametrize("case", ["one_only", "different_close", "different_count", "ignored_raw_field_diff",
                                  "less_than_60_seconds", "same_request", "overlapping_request", "future_received", "unclosed", "bad_hash",
                                  "wrong_step", "http_error", "wrong_endpoint", "futures_endpoint", "wrong_host"])
def test_unconfirmed_or_invalid_mark_receipts_never_rewrite_cache(runtime, case):
    first = _receipt(runtime)
    kwargs = {"received": SECOND}
    if case == "one_only": pass
    elif case == "different_close":
        changed = deepcopy(NEW_MARK); changed[4] = "2226"; kwargs["raw_bar"] = changed
    elif case == "different_count":
        changed = deepcopy(NEW_MARK); changed[8] = 3601; kwargs["raw_bar"] = changed
    elif case == "ignored_raw_field_diff":
        changed = deepcopy(NEW_MARK); changed[11] = "1"; kwargs["raw_bar"] = changed
    elif case == "less_than_60_seconds": kwargs["received"] = FIRST+pd.Timedelta(seconds=59)
    elif case == "same_request": kwargs["requested"] = FIRST-pd.Timedelta(milliseconds=400)
    elif case == "overlapping_request": kwargs["requested"] = FIRST-pd.Timedelta(milliseconds=100)
    elif case == "future_received": kwargs["received"] = NOW+pd.Timedelta(seconds=1)
    elif case == "unclosed": kwargs["received"] = OPEN+pd.Timedelta(minutes=59)
    elif case == "bad_hash": kwargs["bad_hash"] = True
    elif case == "wrong_step":
        changed = deepcopy(NEW_MARK); changed[6] = changed[0]+299999; kwargs["raw_bar"] = changed
    elif case == "http_error": kwargs["http_status"] = 418
    elif case == "wrong_endpoint": kwargs["endpoint"] = "https://fapi.binance.com/fapi/v1/indexPriceKlines"
    elif case == "futures_endpoint": kwargs["endpoint"] = FUTURES_ENDPOINT
    elif case == "wrong_host": kwargs["endpoint"] = "https://fapi.binance.com.evil.invalid/fapi/v1/markPriceKlines"
    if case != "one_only": _receipt(runtime, **kwargs)
    before = _inventory(runtime)
    result = _recover(runtime, [_error()], NOW)
    _assert_no_recovery(runtime, before, result)
    assert first["sha256"] != pd.read_parquet(runtime/"features/raw_data"/MARK_PATH).iloc[-1].receipt_sha256


def test_exactly_sixty_seconds_with_two_sequential_requests_is_sufficient(runtime):
    _receipt(runtime)
    _receipt(runtime, received=FIRST+pd.Timedelta(seconds=60))
    result = _recover(runtime, [_error()], NOW)
    assert result["status"] == "COMMITTED", result


def test_mark_recovery_requires_an_explicit_registry_and_never_finds_production_registry(runtime):
    _receipt(runtime)
    _receipt(runtime, received=SECOND)
    before = _inventory(runtime)
    registry_before = (runtime/"fixture_mark_registry.json").read_bytes()
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert result["status"] == "BLOCKED" and result["reason"] == "MARK_REGISTRY_REQUIRED"
    _assert_no_recovery(runtime, before, result)
    assert (runtime/"fixture_mark_registry.json").read_bytes() == registry_before


@pytest.mark.parametrize("corruption", ["missing_receipt", "old_value", "old_time", "old_clock", "old_hash"])
def test_mark_original_cache_requires_its_own_original_receipt(runtime, corruption):
    _receipt(runtime)
    _receipt(runtime, received=SECOND)
    path = runtime/"features/raw_data"/MARK_PATH
    frame = pd.read_parquet(path)
    target = frame.date == OPEN
    if corruption == "missing_receipt": frame.loc[target, "receipt_sha256"] = "e"*64
    elif corruption == "old_value": frame.loc[target, "close"] = 2225.1
    elif corruption == "old_time": frame.loc[target, "first_received_at"] = OLD_RECEIVED+pd.Timedelta(seconds=1)
    elif corruption == "old_clock": frame.loc[target, "clock_evidence_sha256"] = "e"*64
    elif corruption == "old_hash":
        with sqlite3.connect(runtime/"features/http_receipts.sqlite3") as con:
            con.execute("UPDATE receipts SET sha256=? WHERE id=1", ("e"*64,))
    frame.to_parquet(path, index=False)
    before = _inventory(runtime)
    result = _recover(runtime, [_error()], NOW)
    assert result["status"] == "BLOCKED", result
    _assert_no_recovery(runtime, before, result)


def test_two_futures_1h_responses_cannot_confirm_one_mark_1h_response(runtime):
    _receipt(runtime)
    _receipt(runtime, received=SECOND, endpoint=FUTURES_ENDPOINT)
    _receipt(runtime, received=SECOND+pd.Timedelta(minutes=5), endpoint=FUTURES_ENDPOINT)
    before = _inventory(runtime)
    result = _recover(runtime, [_error()], NOW)
    _assert_no_recovery(runtime, before, result)


def test_mixed_mark_and_futures_batch_is_atomic_until_both_sources_are_confirmed(runtime):
    _receipt(runtime)
    _receipt(runtime, received=SECOND)
    _receipt(runtime, NEW_FUTURES, FIRST, endpoint=FUTURES_ENDPOINT, symbol="BTCUSDT", interval="5m", extra_bar=True)
    errors = [_error(), _error("futures:BTCUSDT:5m")]
    before = _inventory(runtime)
    waiting = _recover(runtime, errors, NOW)
    assert waiting["status"] == "WAITING_CONFIRMATION", waiting
    _assert_no_recovery(runtime, before, waiting)
    _receipt(runtime, NEW_FUTURES, SECOND, endpoint=FUTURES_ENDPOINT, symbol="BTCUSDT", interval="5m", extra_bar=True)
    result = _recover(runtime, errors, NOW)
    assert result["status"] == "COMMITTED", result
    assert {change["symbol"] for change in result["changes"]} == {"ETHUSDT", "BTCUSDT"}
    assert {key for key in before if before[key] != _inventory(runtime)[key]} == {MARK_PATH, FUTURES_PATH}
    assert len(pd.read_parquet(runtime/"features/raw_data"/FUTURES_PATH)) == 1
    assert len(pd.read_parquet(runtime/"features/raw_data"/MARK_PATH)) == 2


@pytest.mark.parametrize("unsupported", ["ETHUSDT:funding", "funding:ETHUSDT:8h", "mark:ETHUSDT:5m",
                                         "mark:ETHUSDT:4h", "futures:ETHUSDT:1h", "index:ETHUSDT:1h"])
def test_unsupported_source_or_interval_blocks_the_entire_otherwise_confirmed_batch(runtime, unsupported):
    _receipt(runtime)
    _receipt(runtime, received=SECOND)
    before = _inventory(runtime)
    result = _recover(runtime, [_error(), _error(unsupported)], NOW)
    assert result["status"] == "BLOCKED", result
    _assert_no_recovery(runtime, before, result)
