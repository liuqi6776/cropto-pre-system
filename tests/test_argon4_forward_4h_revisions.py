"""Offline 4h futures/spot and derived basis-mismatch revision tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlencode

import pandas as pd
import pytest

from tools.argon4_forward import revisions

OPEN_4H = pd.Timestamp("2026-11-12T08:00:00Z")
PREV_4H = pd.Timestamp("2026-11-12T04:00:00Z")
COHORT = pd.Timestamp("2026-11-12T16:00:00Z")
STEP_4H = 14400000

OLD_FUT = [int(OPEN_4H.timestamp() * 1000), "72800.0", "73100.0", "72500.0", "72935.1", "17088.942",
           int((OPEN_4H + pd.Timedelta(hours=4)).timestamp() * 1000) - 1, "1251182885.13", 377616, "7716.168", "565046874.46", "0"]
NEW_FUT = [*OLD_FUT]
NEW_FUT[4], NEW_FUT[5], NEW_FUT[7], NEW_FUT[8] = "72930.0", "17126.248", "1253903773.80", 377840

OLD_SPOT = [int(OPEN_4H.timestamp() * 1000), "2010.0", "2030.0", "2000.0", "2020.5", "85000.0",
            int((OPEN_4H + pd.Timedelta(hours=4)).timestamp() * 1000) - 1, "171742500.0", 92000, "42000.0", "84861000.0", "0"]
NEW_SPOT = [*OLD_SPOT]
NEW_SPOT[4], NEW_SPOT[5], NEW_SPOT[8] = "2020.2", "85010.0", 92012


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("revision recovery attempted network"))


def _row(raw, received="2026-11-12T12:00:03Z", sha="a" * 64):
    return {
        "date": pd.to_datetime(raw[0], unit="ms", utc=True),
        "open_time": pd.to_datetime(raw[0], unit="ms", utc=True),
        "close_time": pd.to_datetime(raw[6], unit="ms", utc=True),
        **{name: float(raw[i]) for i, name in revisions.PRICE_FIELDS.items()},
        "first_received_at": pd.Timestamp(received),
        "receipt_sha256": sha,
        "receipt_time_basis": "CALIBRATED_UTC_UPPER_BOUND",
        "clock_evidence_sha256": "c" * 64,
    }


def _add_4h_receipt(runtime, bar, received, *, symbol="BTCUSDT", kind="futures", endpoint_override=None, include_next=True):
    payload = [deepcopy(bar)]
    if include_next:
        nxt = deepcopy(bar)
        nxt[0] += STEP_4H
        nxt[6] += STEP_4H
        payload.append(nxt)
    raw = json.dumps(payload).encode()
    default_ep = (
        "https://fapi.binance.com/fapi/v1/klines"
        if kind == "futures"
        else "https://data-api.binance.vision/api/v3/klines"
    )
    url = (endpoint_override or default_ep) + "?" + urlencode(
        {"symbol": symbol, "interval": "4h", "startTime": bar[0], "endTime": payload[-1][6], "limit": 1000}
    )
    rec = pd.Timestamp(received)
    digest = hashlib.sha256(raw).hexdigest()
    with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
        conn.execute(
            "INSERT INTO receipts(url,requested_at,received_at,sha256,http_status,raw,time_basis,clock_evidence_sha256) VALUES(?,?,?,?,?,?,?,?)",
            (url, (rec - pd.Timedelta(seconds=0.4)).isoformat(), rec.isoformat(), digest, 200, raw, "CALIBRATED_UTC_UPPER_BOUND", "c" * 64),
        )
    return digest


@pytest.fixture
def runtime_4h(tmp_path):
    rt = tmp_path / "runtime"
    raw = rt / "features/raw_data"
    (raw / "intraday").mkdir(parents=True)
    (raw / "auxiliary").mkdir()
    with sqlite3.connect(rt / "features/http_receipts.sqlite3") as conn:
        conn.execute(
            "CREATE TABLE receipts(id INTEGER PRIMARY KEY,url TEXT,requested_at TEXT,received_at TEXT,sha256 TEXT,http_status INTEGER,raw BLOB,time_basis TEXT,clock_evidence_sha256 TEXT)"
        )
    with sqlite3.connect(rt / "observations.sqlite3") as conn:
        conn.execute("CREATE TABLE decisions(id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO decisions(content) VALUES(?)", ("untouched prior decision",))

    prev_fut = deepcopy(OLD_FUT)
    prev_fut[0] -= STEP_4H
    prev_fut[6] -= STEP_4H
    fut_sha = _add_4h_receipt(rt, OLD_FUT, "2026-11-12T12:00:03Z", symbol="BTCUSDT", kind="futures", include_next=False)
    pd.DataFrame([_row(prev_fut, "2026-11-12T08:00:05Z", "b" * 64), _row(OLD_FUT, "2026-11-12T12:00:03Z", fut_sha)]).to_parquet(
        raw / "auxiliary/BTCUSDT_4h_futures.parquet", index=False
    )

    prev_spot = deepcopy(OLD_SPOT)
    prev_spot[0] -= STEP_4H
    prev_spot[6] -= STEP_4H
    spot_sha = _add_4h_receipt(rt, OLD_SPOT, "2026-11-12T12:00:03Z", symbol="ETHUSDT", kind="spot", include_next=False)
    pd.DataFrame([_row(prev_spot, "2026-11-12T08:00:05Z", "b" * 64), _row(OLD_SPOT, "2026-11-12T12:00:03Z", spot_sha)]).to_parquet(
        raw / "auxiliary/ETHUSDT_4h_spot.parquet", index=False
    )
    return rt


def test_confirmed_4h_futures_and_spot_with_derived_basis_mismatch_commits(runtime_4h):
    _add_4h_receipt(runtime_4h, NEW_FUT, "2026-11-12T16:00:03Z", symbol="BTCUSDT", kind="futures")
    _add_4h_receipt(runtime_4h, NEW_FUT, "2026-11-12T16:05:03Z", symbol="BTCUSDT", kind="futures")
    _add_4h_receipt(runtime_4h, NEW_SPOT, "2026-11-12T16:00:03Z", symbol="ETHUSDT", kind="spot")
    _add_4h_receipt(runtime_4h, NEW_SPOT, "2026-11-12T16:05:03Z", symbol="ETHUSDT", kind="spot")

    errors = [
        {"source": "futures:BTCUSDT:4h", "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"},
        {"source": "spot:ETHUSDT:4h", "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"},
        {"source": "basis", "error": "ValueError:BASIS_LEG_COVERAGE_MISMATCH"},
    ]
    result = revisions.reconcile_confirmed_price_revisions(runtime_4h, errors, pd.Timestamp("2026-11-12T16:06:00Z"))
    assert result["status"] == "COMMITTED", result
    fut = pd.read_parquet(runtime_4h / "features/raw_data/auxiliary/BTCUSDT_4h_futures.parquet")
    spot = pd.read_parquet(runtime_4h / "features/raw_data/auxiliary/ETHUSDT_4h_spot.parquet")
    assert fut.iloc[-1]["close"] == 72930.0
    assert spot.iloc[-1]["close"] == 2020.2
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    endpoints = {c["kind"]: c["source_endpoint"] for c in manifest["changes"]}
    assert endpoints == {
        "futures": "https://fapi.binance.com/fapi/v1/klines",
        "spot": "https://data-api.binance.vision/api/v3/klines",
    }
    assert revisions.ensure_view_committed(runtime_4h)["status"] == "COMMITTED"


def test_standalone_basis_mismatch_or_non_basis_4h_symbol_fails_closed(runtime_4h):
    standalone = revisions.reconcile_confirmed_price_revisions(
        runtime_4h,
        [{"source": "basis", "error": "ValueError:BASIS_LEG_COVERAGE_MISMATCH"}],
        pd.Timestamp("2026-11-12T16:06:00Z"),
    )
    assert standalone["status"] == "BLOCKED"
    assert standalone["reason"] == "UNSUPPORTED_REVISION_ERROR"

    bnb_4h = revisions.reconcile_confirmed_price_revisions(
        runtime_4h,
        [{"source": "futures:BNBUSDT:4h", "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"}],
        pd.Timestamp("2026-11-12T16:06:00Z"),
    )
    assert bnb_4h["status"] == "BLOCKED"
    assert bnb_4h["reason"] == "UNSUPPORTED_REVISION_ERROR"


def test_cross_host_4h_receipt_fails_closed(runtime_4h):
    _add_4h_receipt(runtime_4h, NEW_SPOT, "2026-11-12T16:00:03Z", symbol="ETHUSDT", kind="spot")
    _add_4h_receipt(
        runtime_4h,
        NEW_SPOT,
        "2026-11-12T16:05:03Z",
        symbol="ETHUSDT",
        kind="spot",
        endpoint_override="https://fapi.binance.com/api/v3/klines",
    )
    res = revisions.reconcile_confirmed_price_revisions(
        runtime_4h,
        [{"source": "spot:ETHUSDT:4h", "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"}],
        pd.Timestamp("2026-11-12T16:06:00Z"),
    )
    assert res["status"] == "BLOCKED"
    assert res["reason"] == "UNSUPPORTED_RECEIPT_ENDPOINT"


def test_v14_fastpath_admits_aged_4h_conflict_with_derived_basis_mismatch(runtime_4h):
    _add_4h_receipt(runtime_4h, NEW_FUT, COHORT + pd.Timedelta(seconds=3), symbol="BTCUSDT", kind="futures")
    _add_4h_receipt(runtime_4h, NEW_FUT, COHORT + pd.Timedelta(seconds=5.5), symbol="BTCUSDT", kind="futures")
    context = {
        "profile": revisions.FAST_V14_PROFILE,
        "cohort": COHORT.isoformat(),
        "deadline": (COHORT + pd.Timedelta(seconds=18.5)).isoformat(),
        "minimum_spacing_seconds": 1.2,
        "clock_evidence_sha256": "c" * 64,
    }
    errors = [
        {"source": "futures:BTCUSDT:4h", "error": "ValueError:CONFLICTING_CLOSED_OBSERVATION"},
        {"source": "basis", "error": "ValueError:BASIS_LEG_COVERAGE_MISMATCH"},
    ]
    res = revisions.reconcile_confirmed_price_revisions(
        runtime_4h,
        errors,
        now=lambda: COHORT + pd.Timedelta(seconds=6),
        cutover_now=lambda: COHORT + pd.Timedelta(seconds=7),
        fast_context=context,
        cohort_guard=lambda: {"eligible": True, "clock_evidence_sha256": "c" * 64, "policy_sha256": "d" * 64},
    )
    assert res["status"] == "COMMITTED", res
    assert res["same_cohort_eligible"] is True
