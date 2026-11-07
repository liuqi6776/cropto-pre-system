"""Offline, read-only receipt verification; no synthetic fills or current fees."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from scripts import analyze_argon4_book_telemetry as mod
from tools.argon4_forward.clock_sync import bounds_from_samples, project_local
from tools.argon4_forward.polymarket import parse_fee, validate_book

C = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)


def fixture(*, offset=2, delay=.2, status="OBSERVED", profile=None, unknown_fee=False):
    slug = f"btc-updown-15m-{int(C.timestamp())}"
    condition = "0x"+"a"*64
    market = {"id": "1", "slug": slug, "version": "v1", "conditionId": condition,
        "question": "Bitcoin Up or Down - fixture", "eventStartTime": mod.iso(C),
        "endDate": mod.iso(C+timedelta(minutes=15)),
        "events": [{"slug": slug, "startTime": mod.iso(C), "endDate": mod.iso(C+timedelta(minutes=15))}],
        "active": True, "closed": False, "archived": False, "acceptingOrders": True,
        "enableOrderBook": True, "negRisk": False,
        "cryptoMarketConfig": {"asset": "btc", "duration": "15m", "twapEnabled": True, "twapLookbackSeconds": 60},
        "resolutionSource": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
        "description": 'resolve to "Up" if Chainlink BTC/USD is greater than or equal to; resolve to "Down" otherwise',
        "outcomes": '["Up","Down"]', "clobTokenIds": '["111","222"]',
        "feesEnabled": True, "feeSchedule": {"rate": .07, "exponent": 1, "takerOnly": True, "rebateRate": .2}}
    if unknown_fee:
        market.pop("feeSchedule")
    meta_raw = json.dumps([market])
    meta = {"source": mod.SOURCE, "method": "GET", "url": mod.public_url("market", slug),
        "http_status": 200, "body_complete": True,
        "requested_at": mod.iso(C-timedelta(seconds=4.2)), "received_at": mod.iso(C-timedelta(seconds=4)),
        "sha256": mod.sha(meta_raw)}
    samples = []
    for second, fraction in ((-12, .1), (-11, .5), (-10, .9)):
        value = int(C.timestamp())+second
        requested = C+timedelta(seconds=second+fraction)
        received = requested+timedelta(seconds=.05)
        samples.append({"url": mod.public_url("time"), "raw_text": str(value),
            "sha256": mod.sha(str(value)), "http_status": 200, "server_second": value,
            "requested_at_local": mod.iso(requested), "received_at_local": mod.iso(received),
            "received_at": mod.iso(received), "monotonic_elapsed_seconds": .05})
    clock = {"status": "CALIBRATED", "source": "POLYMARKET_CLOB_SERVER_TIME",
        "receipt": {"samples": samples, "quantization_seconds": 1}, "receipts": samples,
        "receipt_sha256": mod.sha(mod.canonical(samples)), **bounds_from_samples(samples),
        "calibrated_at_local": mod.iso(C-timedelta(seconds=8.8)), "calibrated_monotonic": 123.0}
    clock["observed_at"] = mod.iso(project_local(clock["calibrated_at_local"], clock))
    requested, received = C+timedelta(seconds=offset+.3), C+timedelta(seconds=offset+.3+delay)
    raw = {"market": condition, "asset_id": "111", "timestamp": str(int((received-timedelta(seconds=.2)).timestamp()*1000)),
        "bids": [{"price": ".49", "size": "20"}], "asks": [{"price": ".50", "size": "20"}], "hash": "raw-book"}
    text = json.dumps(raw)
    receipt = {"source": mod.SOURCE, "method": "GET", "url": mod.public_url("book", "111"),
        "http_status": 200, "body_complete": True, "requested_at": mod.iso(requested), "received_at": mod.iso(received),
        "raw_text": text, "sha256": mod.sha(text), "transport_timing": {"request_elapsed_seconds": delay}}
    row = {"sample_id": f"{int(C.timestamp())}:BTC_15m:UP:{offset}", "stream": "BTC_15m", "outcome": "UP",
        "cohort": mod.iso(C), "target_at": mod.iso(C+timedelta(seconds=offset)), "target_offset_seconds": offset,
        "status": status, "token_id": "111", "condition_id": condition, "market_id": "1", "slug": slug,
        "request_clock": clock, "receipt": receipt, "requested_at_local": receipt["requested_at"],
        "received_at_local": receipt["received_at"],
        "requested_at_server_lower": mod.iso(project_local(requested, clock, upper=False)),
        "received_at_server_lower": mod.iso(project_local(received, clock, upper=False)),
        "received_at_server_upper": mod.iso(project_local(received, clock)),
        "parsed_book": validate_book(raw, "111", condition, project_local(received, clock)),
        "metadata_sha256": meta["sha256"], "metadata_received_at_local": meta["received_at"], "fee": parse_fee(market)}
    if profile:
        row["sampling_profile"] = profile
    proof = {"receipt": meta, "body": [market], "raw_text": meta_raw}
    return row, {mod.metadata_key(row): proof}


def databases(tmp_path, samples):
    database, observations = tmp_path/"books.sqlite3", tmp_path/"observations.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE samples(sample_id TEXT,cohort TEXT,stream TEXT,outcome TEXT,target_offset INTEGER,target_at TEXT,status TEXT,payload_json TEXT)")
        for row, _ in samples:
            db.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?)", (row["sample_id"], row["cohort"], row["stream"], row["outcome"],
                row["target_offset_seconds"], row["target_at"], row["status"], json.dumps(row)))
    unique = {key: value for _, metadata in samples for key, value in metadata.items()}
    with sqlite3.connect(observations) as db:
        db.executescript("CREATE TABLE receipts(id INTEGER PRIMARY KEY,received_at TEXT,url TEXT,sha256 TEXT,http_status INTEGER,raw_text TEXT); CREATE TABLE snapshots(id INTEGER PRIMARY KEY,kind TEXT,saved_at TEXT,payload TEXT);")
        for proof in unique.values():
            receipt = deepcopy(proof["receipt"])
            cursor = db.execute("INSERT INTO receipts(received_at,url,sha256,http_status,raw_text) VALUES(?,?,?,?,?)",
                (receipt["received_at"], receipt["url"], receipt["sha256"], receipt["http_status"], proof["raw_text"]))
            receipt["receipt_id"] = cursor.lastrowid
            db.execute("INSERT INTO snapshots(kind,saved_at,payload) VALUES(?,?,?)", ("polymarket", receipt["received_at"], json.dumps({"receipt_metadata": [receipt]})))
    return database, observations


def test_late_complete_receipt_has_actual_capacity_but_never_becomes_strict():
    row, metadata = fixture(delay=1.1, status="MISSED_SAMPLE")
    result = mod.verify_sample(row, metadata)
    assert result["actual_quote"] and result["capacity_available"]
    assert result["capacity_usd"] == 10.35
    assert not result["strict_slice"] and not result["strict_timing_verified"]
    assert row["status"] == "MISSED_SAMPLE"
    assert result["receipt_lag_upper_seconds"] > 1


def test_missed_original_status_cannot_be_promoted_even_if_timing_is_strict():
    row, metadata = fixture(status="MISSED_SAMPLE")
    result = mod.verify_sample(row, metadata)
    assert result["strict_timing_verified"] and not result["strict_slice"]


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r["receipt"].update(sha256="0"*64), "BOOK_RECEIPT_HASH_MISMATCH"),
    (lambda r: r["request_clock"].update(receipt_sha256="0"*64), "CLOCK_RECEIPT_HASH_MISMATCH"),
    (lambda r: r["request_clock"].update(uncertainty_seconds=.001), "CLOCK_BOUNDS_MISMATCH"),
    (lambda r: r["receipt"]["transport_timing"].update(request_elapsed_seconds=3), "REQUEST_CLOCK_ELAPSED_UNVERIFIABLE"),
    (lambda r: r["parsed_book"].update(best_ask=.1), "NORMALIZED_BOOK_MISMATCH"),
    (lambda r: r["fee"].update(rate=.001), "STORED_FEE_DIFFERS_FROM_HISTORICAL_GAMMA"),
    (lambda r: r.update(token_id="222"), "COMPLETE_PUBLIC_BOOK_RECEIPT_REQUIRED"),
    (lambda r: r.update(market_id="2"), "GAMMA_BOOK_IDENTITY_MISMATCH"),
])
def test_modified_proof_never_yields_quotes_or_capacity(mutation, reason):
    row, metadata = fixture(delay=1.1, status="MISSED_SAMPLE")
    mutation(row)
    result = mod.verify_sample(row, metadata)
    assert result["status"] == "UNVERIFIABLE" and result["reason"] == reason
    assert not result["actual_quote"] and not result["capacity_available"]


def test_missing_historical_gamma_cannot_use_stored_or_current_fee():
    row, _ = fixture()
    result = mod.verify_sample(row, {})
    assert result["reason"] == "HISTORICAL_METADATA_RECEIPT_MISSING"
    assert not result["capacity_available"]


def test_unknown_fee_allows_verified_quote_but_no_capacity():
    row, metadata = fixture(unknown_fee=True)
    result = mod.verify_sample(row, metadata)
    assert result["actual_quote"] and not result["capacity_available"]
    assert result["capacity_status"] == "UNKNOWN_FEE" and result["capacity_usd"] is None


def test_metadata_raw_body_mutation_is_rejected():
    row, metadata = fixture()
    metadata[mod.metadata_key(row)]["raw_text"] += " "
    result = mod.verify_sample(row, metadata)
    assert result["reason"] == "METADATA_RAW_RECEIPT_INVALID"


def test_original_clock_must_cover_the_historical_request():
    row, metadata = fixture(offset=60)
    result = mod.verify_sample(row, metadata)
    assert result["reason"] == "HISTORICAL_CLOCK_EXPIRED_OR_REVERSED"


def test_readonly_aggregate_preserves_databases_and_groups_actual_receipts(tmp_path):
    first = fixture()
    late = fixture(offset=6, delay=1.1, status="MISSED_SAMPLE", profile="clock_center_v12")
    no_receipt = fixture(offset=10)
    no_receipt[0].pop("receipt")
    no_receipt[0]["status"] = "MISSED_SAMPLE"
    database, observations = databases(tmp_path, [first, late, no_receipt])
    before = database.read_bytes(), observations.read_bytes()
    report = mod.analyze(database, observations)
    assert report["counts"] == {"total_samples": 3, "strict_slices": 1, "actual_quote_samples": 2,
        "fee_inclusive_depth_samples": 2, "non_strict_actual_depth_samples": 1,
        "unverifiable_samples": 0, "no_receipt_samples": 1}
    assert {g["profile"] for g in report["groups"]} == {"legacy_unlabelled", "clock_center_v12"}
    group = next(g for g in report["groups"] if g["profile"] == "clock_center_v12")
    assert group["actual_receipt_distributions"]["receipt_lag_center_seconds"]["median"] > 1
    assert group["actual_receipt_distributions"]["capacity_usd"]["median"] == 10.35
    assert not report["historical_status_modified"] and not report["current_clock_or_fee_used"]
    assert before == (database.read_bytes(), observations.read_bytes())
    with mod.readonly(database) as db:
        with pytest.raises(sqlite3.OperationalError):
            db.execute("DELETE FROM samples")


def test_tampered_metadata_in_observation_database_is_unverifiable(tmp_path):
    database, observations = databases(tmp_path, [fixture()])
    with sqlite3.connect(observations) as db:
        db.execute("UPDATE receipts SET raw_text=raw_text || ' '")
    report = mod.analyze(database, observations)
    assert report["counts"]["unverifiable_samples"] == 1
    assert report["groups"][0]["reason_counts"] == {"METADATA_RAW_RECEIPT_INVALID": 1}


def test_sql_and_payload_identity_conflict_never_changes_original_status(tmp_path):
    database, observations = databases(tmp_path, [fixture()])
    with sqlite3.connect(database) as db:
        db.execute("UPDATE samples SET status='MISSED_SAMPLE'")
    report = mod.analyze(database, observations)
    assert report["counts"]["strict_slices"] == 0
    assert report["groups"][0]["reason_counts"] == {"SQL_PAYLOAD_IDENTITY_MISMATCH": 1}


def test_missing_input_is_not_created(tmp_path):
    missing = tmp_path/"absent.sqlite3"
    with pytest.raises(FileNotFoundError):
        mod.analyze(missing)
    assert not missing.exists()


def test_cli_default_writes_nothing_and_output_requires_new_local_report(tmp_path, monkeypatch, capsys):
    database, observations = databases(tmp_path, [fixture()])
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    args = ["--database", str(database), "--observations-database", str(observations)]
    files_before = sorted(p.name for p in tmp_path.iterdir())
    assert mod.main(args) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == files_before
    report = tmp_path/"reports"/"audit.json"
    assert mod.main(args+["--output", str(report)]) == 0
    before = report.read_bytes()
    assert mod.main(args+["--output", str(report)]) == 2
    assert report.read_bytes() == before
    outside = tmp_path/"outside.json"
    assert mod.main(args+["--output", str(outside)]) == 2
    assert not outside.exists()
    assert "raw-book" not in capsys.readouterr().out


def test_analysis_bounds_request_volume_without_loading_unbounded_rows(tmp_path, monkeypatch):
    database, observations = databases(tmp_path, [fixture(), fixture(offset=6)])
    monkeypatch.setattr(mod, "MAX_SAMPLES", 1)
    with pytest.raises(mod.EvidenceError, match="SAMPLE_BUDGET_EXCEEDED"):
        mod.analyze(database, observations)
    monkeypatch.setattr(mod, "MAX_SAMPLES", 10)
    monkeypatch.setattr(mod, "MAX_METADATA_BYTES", 1)
    with pytest.raises(mod.EvidenceError, match="METADATA_BUDGET_EXCEEDED"):
        mod.analyze(database, observations)
