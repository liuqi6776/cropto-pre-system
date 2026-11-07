"""Offline registry publication fixtures; never target the workspace registry."""
from copy import deepcopy
import hashlib
import json
import sqlite3
from urllib.parse import urlencode

import pandas as pd
import pytest

from tools.argon4_forward import mark_revision_registry as registry


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    raw = tmp_path / "runtime/features/raw_data"
    path = raw / "auxiliary/ETHUSDT_1h_mark.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-11-06T01:00:00Z"]),
        "open": [100.], "high": [110.], "low": [90.], "close": [101.], "volume": [0.],
        "first_received_at": pd.to_datetime(["2026-11-06T02:00:03Z"]),
        "receipt_sha256": ["a"*64], "receipt_time_basis": ["CALIBRATED_UTC_UPPER_BOUND"],
        "clock_evidence_sha256": ["c"*64]})
    frame.to_parquet(path, index=False)
    before = digest(path)
    proposed = frame.copy(deep=True)
    proposed.loc[0, "close"] = 102.
    proposed.loc[0, "receipt_sha256"] = "b"*64
    proposed.loc[0, "first_received_at"] = pd.Timestamp("2026-11-06T02:05:05Z")
    archive = tmp_path / "runtime/features/revisions/fixture-revision"
    relative = "auxiliary/ETHUSDT_1h_mark.parquet"
    proposed_path = archive / "proposed_raw_data" / relative
    proposed_path.parent.mkdir(parents=True)
    proposed.to_parquet(proposed_path, index=False)
    plan = {"kind": "mark", "interval": "1h", "symbol": "ETHUSDT", "path": path,
            "relative_path": relative, "before_sha256": before, "frame": proposed}
    document = {"datasets": [{"dataset_id": "canonical-original", "sha256": "d"*64}],
        "metadata": {"unrelated": "unchanged"}, "research_references": [
            {"dataset_id": "argon4_local_v4_online_mark_ETHUSDT_20261106", "pair": "ETH/USDT:USDT",
             "timeframe": "1h", "candle_type": "mark", "local_path": str(path), "sha256": before,
             "source_endpoint": registry.ENDPOINT, "verification_status": "VERIFIED_PERPETUAL_LABEL",
             "installed": False, "research_only": True, "gate_effect": "NONE"},
            {"dataset_id": "unrelated-research", "local_path": str(tmp_path / "other.parquet"), "untouched": [1, 2]}]}
    path_registry = tmp_path / "fixture_mark_candle_registry.json"
    path_registry.write_text(json.dumps(document), encoding="utf-8")
    return {"registry": path_registry, "raw": raw, "archive": archive, "plans": [plan],
            "after": {relative: digest(proposed_path)}, "document": document}


def prepare(f):
    return registry.prepare_registry_update(f["registry"], f["raw"], f["plans"], f["after"], f["archive"])


def test_prepare_archives_bytes_publish_preserves_all_other_entries_and_future_append(fixture):
    f = fixture
    original = f["registry"].read_bytes()
    receipt = prepare(f)
    assert f["registry"].read_bytes() == original
    assert (f["archive"] / receipt["before_snapshot"]).read_bytes() == original
    assert json.loads(json.dumps(receipt)) == receipt
    proposed = json.loads((f["archive"] / receipt["proposed_snapshot"]).read_text(encoding="utf-8"))
    assert proposed["datasets"] == f["document"]["datasets"]
    assert proposed["metadata"] == f["document"]["metadata"]
    assert proposed["research_references"][1] == f["document"]["research_references"][1]
    entry = proposed["research_references"][0]
    assert entry["sha256"] == f["after"]["auxiliary/ETHUSDT_1h_mark.parquet"]
    assert entry["rows"] == 1 and entry["first_utc"] == entry["last_utc"] == "2026-11-06T01:00:00+00:00"
    assert entry["source_receipt_sha256"] == ["b"*64]
    assert entry["revision_provenance"]["future_cohorts_only"] is True
    assert registry.publish_registry_update(receipt)["status"] == "PUBLISHED"
    assert digest(f["registry"]) == receipt["after_sha256"]
    # A later normal update is valid; immutable recovery evidence still verifies.
    proposed["research_references"][0]["rows"] = 2
    f["registry"].write_text(json.dumps(proposed), encoding="utf-8")
    assert registry.verify_registry_archive(f["archive"], receipt)["status"] == "VERIFIED"


@pytest.mark.parametrize("field,value", [("sha256", "f"*64), ("pair", "BTC/USDT:USDT"),
    ("timeframe", "5m"), ("candle_type", "futures"), ("source_endpoint", "https://fapi.binance.com/fapi/v1/klines"),
    ("installed", True), ("research_only", False), ("gate_effect", "PASS"),
    ("verification_status", "UNKNOWN"), ("dataset_id", "canonical-other")])
def test_wrong_old_identity_cannot_prepare_or_write_registry(fixture, field, value):
    f = fixture
    document = deepcopy(f["document"])
    document["research_references"][0][field] = value
    f["registry"].write_text(json.dumps(document), encoding="utf-8")
    before = f["registry"].read_bytes()
    with pytest.raises(ValueError, match="REFERENCE_IDENTITY"):
        prepare(f)
    assert f["registry"].read_bytes() == before
    assert not (f["archive"] / registry.SNAPSHOTS[0]).exists()


@pytest.mark.parametrize("case", ["missing_registry", "missing_entry", "wrong_local_path", "duplicate_path", "duplicate_id"])
def test_missing_or_nonunique_reference_fails_closed(fixture, case):
    f = fixture
    doc = deepcopy(f["document"])
    if case == "missing_registry": f["registry"].unlink()
    else:
        if case == "missing_entry": doc["research_references"].pop(0)
        elif case == "wrong_local_path": doc["research_references"][0]["local_path"] = str(f["raw"] / "wrong.parquet")
        else:
            extra = deepcopy(doc["research_references"][0])
            if case == "duplicate_id": extra["local_path"] = str(f["raw"] / "other.parquet")
            doc["research_references"].append(extra)
        f["registry"].write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError): prepare(f)
    assert not (f["archive"] / registry.SNAPSHOTS[0]).exists()


@pytest.mark.parametrize("case", ["old_raw", "proposed_raw", "frame", "duplicate_plan", "wrong_path", "wrong_kind"])
def test_changed_raw_or_plan_never_produces_registration(fixture, case):
    f = fixture; plan = f["plans"][0]
    if case == "old_raw": plan["path"].write_bytes(b"changed")
    elif case == "proposed_raw": (f["archive"] / "proposed_raw_data" / plan["relative_path"]).write_bytes(b"changed")
    elif case == "frame": plan["frame"].loc[0, "close"] = 103.
    elif case == "duplicate_plan": f["plans"].append(plan)
    elif case == "wrong_path": plan["relative_path"] = "../ETHUSDT_1h_mark.parquet"
    elif case == "wrong_kind": plan["kind"] = "funding"
    with pytest.raises(ValueError): prepare(f)
    assert not (f["archive"] / registry.SNAPSHOTS[0]).exists()


def test_relative_production_style_reference_resolves_only_to_exact_raw_file(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(registry, "ROOT", f["raw"].parents[2])
    doc = deepcopy(f["document"])
    doc["research_references"][0]["local_path"] = f["plans"][0]["path"].relative_to(registry.ROOT).as_posix()
    f["registry"].write_text(json.dumps(doc), encoding="utf-8")
    assert prepare(f)["updates"][0]["symbol"] == "ETHUSDT"


def test_archive_files_are_write_once_and_registry_race_is_rejected(fixture):
    f = fixture
    receipt = prepare(f)
    with pytest.raises(ValueError, match="ALREADY_EXISTS"): prepare(f)
    f["registry"].write_text('{"unexpected_concurrent_change": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="CHANGED_BEFORE_PUBLICATION"):
        registry.publish_registry_update(receipt)
    assert "unexpected_concurrent_change" in f["registry"].read_text()


@pytest.mark.parametrize("name", registry.SNAPSHOTS)
def test_immutable_archive_tamper_blocks_verification_and_publication(fixture, name):
    f = fixture; receipt = prepare(f); before = f["registry"].read_bytes()
    (f["archive"] / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="ARCHIVE_HASH_MISMATCH"):
        registry.verify_registry_archive(f["archive"], receipt)
    with pytest.raises(ValueError): registry.publish_registry_update(receipt)
    assert f["registry"].read_bytes() == before


def test_atomic_publish_failure_preserves_old_registry_and_archive(fixture, monkeypatch):
    f = fixture; receipt = prepare(f); before = f["registry"].read_bytes()
    def fail(*args): raise OSError("fixture failure")
    monkeypatch.setattr(registry.os, "replace", fail)
    with pytest.raises(OSError): registry.publish_registry_update(receipt)
    assert f["registry"].read_bytes() == before
    assert registry.verify_registry_archive(f["archive"], receipt)["status"] == "VERIFIED"
    assert not list(f["registry"].parent.glob("*.tmp"))


@pytest.fixture
def append_fixture(fixture):
    """Explicit synthetic observations; no workspace data or registry is used."""
    f = fixture
    plan = f["plans"][0]
    relative = plan["relative_path"]
    old = pd.read_parquet(plan["path"])
    old["open_time"] = old.date
    old["close_time"] = old.date+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1)
    for col in ("quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"):
        old[col] = 0.
    old["count"] = 3600.
    old_archive = f["raw"].parent/"revisions/previous-committed-fixture"
    old_path = old_archive/"raw_data"/relative
    old_path.parent.mkdir(parents=True)
    old.to_parquet(old_path, index=False)
    old_hash = digest(old_path)
    old_prepared = {"status": "PREPARED", "revision_id": old_archive.name, "archive_path": str(old_archive),
                    "data_root": str(f["raw"]), "validation_at": "2026-11-06T02:10:00Z", "before_files": {relative: old_hash}}
    (old_archive/"prepared.json").write_text(json.dumps(old_prepared), encoding="utf8")
    old_manifest = {**old_prepared, "status": "COMMITTED", "view_effective_at": "2026-11-06T02:10:00Z",
                    "prepared_sha256": digest(old_archive/"prepared.json"), "after_files": {relative: old_hash}}
    (old_archive/"manifest.json").write_text(json.dumps(old_manifest), encoding="utf8")
    added = old.copy()
    added["date"] += pd.Timedelta(hours=1)
    added["open_time"] = added.date
    added["close_time"] += pd.Timedelta(hours=1)
    added["first_received_at"] = pd.Timestamp("2026-11-06T03:00:05Z")
    opened = int(added.date.iloc[0].timestamp()*1000)
    raw_bar = [opened, "100", "110", "90", "101", "0", opened+3599999, "0", 3600, "0", "0", "0"]
    raw = json.dumps([raw_bar], separators=(",", ":")).encode()
    added["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    baseline = pd.concat([old, added], ignore_index=True)
    for column in ("date", "open_time", "close_time", "first_received_at"):
        baseline[column] = pd.to_datetime(baseline[column], utc=True).astype("datetime64[ns, UTC]")
    baseline.to_parquet(plan["path"], index=False)
    before_path = f["archive"]/"raw_data"/relative
    before_path.parent.mkdir(parents=True)
    baseline.to_parquet(before_path, index=False)
    plan["before_sha256"] = digest(plan["path"])
    prepared = {"status": "PREPARED", "revision_id": f["archive"].name, "archive_path": str(f["archive"]),
                "data_root": str(f["raw"]), "validation_at": "2026-11-06T03:10:00Z",
                "before_files": {relative: plan["before_sha256"]}}
    (f["archive"]/"prepared.json").write_text(json.dumps(prepared), encoding="utf8")
    proposal = baseline.copy()
    proposal.loc[1, "close"] = 102.
    proposal.loc[1, "receipt_sha256"] = "b"*64
    proposal.loc[1, "first_received_at"] = pd.Timestamp("2026-11-06T03:05:05Z")
    proposal["first_received_at"] = pd.to_datetime(proposal.first_received_at, utc=True).astype("datetime64[ns, UTC]")
    proposed_path = f["archive"]/"proposed_raw_data"/relative
    proposal.to_parquet(proposed_path, index=False)
    plan["frame"] = proposal
    f["after"][relative] = digest(proposed_path)
    f["document"]["research_references"][0]["sha256"] = old_hash
    f["registry"].write_text(json.dumps(f["document"]), encoding="utf8")
    db_path = f["archive"]/"receipts.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE receipts(id INTEGER PRIMARY KEY,url TEXT,requested_at TEXT,received_at TEXT,sha256 TEXT,http_status INTEGER,raw BLOB,time_basis TEXT,clock_evidence_sha256 TEXT)")
        url = registry.ENDPOINT+"?"+urlencode({"symbol": "ETHUSDT", "interval": "1h", "startTime": opened,
                                              "endTime": opened+3599999, "limit": 1000})
        db.execute("INSERT INTO receipts VALUES(1,?,?,?,?,?,?,?,?)", (url, "2026-11-06T03:00:04.8Z",
                   "2026-11-06T03:00:05Z", hashlib.sha256(raw).hexdigest(), 200, raw,
                   "CALIBRATED_UTC_UPPER_BOUND", "c"*64))
    return {**f, "old_path": old_path, "old_archive": old_archive, "before_path": before_path,
            "baseline": baseline, "receipt_db": db_path}


def _append_proof(f):
    return registry.prove_append_baseline(f["document"]["research_references"][0], f["raw"], "ETHUSDT",
                                         f["plans"][0]["before_sha256"], f["archive"])


def _rewrite_fixture_before(f, frame):
    frame.to_parquet(f["before_path"], index=False)
    f["plans"][0]["before_sha256"] = digest(f["before_path"])
    p = f["archive"]/"prepared.json"
    value = json.loads(p.read_text())
    value["before_files"][f["plans"][0]["relative_path"]] = digest(f["before_path"])
    p.write_text(json.dumps(value), encoding="utf8")


def test_verified_append_archives_old_registry_and_proof_in_single_publication(append_fixture):
    f = append_fixture
    before = f["registry"].read_bytes()
    proof = _append_proof(f)
    assert proof["old_rows"] == 1 and proof["before_rows"] == 2 and proof["appended_rows"] == 1
    assert proof["registered_sha256"] == digest(f["old_path"])
    assert proof["added_observations"][0]["original_receipt"]["raw_bar"][4] == "101"
    assert _append_proof(f) == proof
    assert not (f["archive"]/registry.APPEND_PROOF_FILE).exists()  # helper is read-only
    receipt = prepare(f)
    assert f["registry"].read_bytes() == before
    assert (f["archive"]/registry.SNAPSHOTS[0]).read_bytes() == before
    assert receipt["append_baseline_proofs"] == [proof]
    proposed = json.loads((f["archive"]/registry.SNAPSHOTS[1]).read_text())
    updated = proposed["research_references"][0]
    assert updated["revision_provenance"]["append_baseline_proof"] == proof
    assert updated["sha256"] == f["after"][f["plans"][0]["relative_path"]]
    assert proposed["research_references"][1] == f["document"]["research_references"][1]
    assert registry.publish_registry_update(receipt)["status"] == "PUBLISHED"
    assert registry.verify_registry_archive(f["archive"], receipt)["status"] == "VERIFIED"
    (f["archive"]/registry.APPEND_PROOF_FILE).write_text("[]")
    with pytest.raises(ValueError, match="APPEND_PROOF_ARCHIVE"):
        registry.verify_registry_archive(f["archive"], receipt)


@pytest.mark.parametrize("case", ["missing_snapshot", "bare_snapshot", "wrong_manifest_binding", "changed_snapshot",
                                 "changed_old_value", "float_epsilon", "float_sign_zero", "changed_dtype", "columns_reordered",
                                 "no_append", "duplicate_date", "fake_new_value", "wrong_clock", "unclosed_new_row",
                                 "bad_receipt_hash", "wrong_receipt_endpoint", "wrong_receipt_time"])
def test_append_baseline_never_accepts_unproven_or_changed_observations(append_fixture, case):
    f = append_fixture
    frame = f["baseline"].copy()
    rewrite = True
    if case == "missing_snapshot": f["old_path"].unlink(); rewrite = False
    elif case == "bare_snapshot": (f["old_archive"]/"manifest.json").unlink(); rewrite = False
    elif case == "wrong_manifest_binding":
        p = f["old_archive"]/"manifest.json"; doc = json.loads(p.read_text())
        doc["before_files"][f["plans"][0]["relative_path"]] = "f"*64; p.write_text(json.dumps(doc)); rewrite = False
    elif case == "changed_snapshot": f["old_path"].write_bytes(b"changed"); rewrite = False
    elif case == "changed_old_value": frame.loc[0, "close"] = 101.01
    elif case == "float_epsilon": frame.loc[0, "close"] += 1e-12
    elif case == "float_sign_zero": frame.loc[0, "volume"] = -0.0
    elif case == "changed_dtype": frame["count"] = frame["count"].astype("int64")
    elif case == "columns_reordered": frame = frame[list(reversed(frame.columns))]
    elif case == "no_append": frame = frame.iloc[:1]
    elif case == "duplicate_date": frame.loc[1, "date"] = frame.loc[0, "date"]
    elif case == "fake_new_value": frame.loc[1, "close"] = 102.
    elif case == "wrong_clock": frame.loc[1, "clock_evidence_sha256"] = "e"*64
    elif case == "unclosed_new_row": frame.loc[1, "first_received_at"] = pd.Timestamp("2026-11-06T02:59:59Z")
    else:
        with sqlite3.connect(f["receipt_db"]) as db:
            if case == "bad_receipt_hash": db.execute("UPDATE receipts SET sha256=?", ("f"*64,))
            elif case == "wrong_receipt_endpoint": db.execute("UPDATE receipts SET url=replace(url,'markPriceKlines','klines')")
            elif case == "wrong_receipt_time": db.execute("UPDATE receipts SET received_at='2026-11-06T03:00:06Z'")
        rewrite = False
    if rewrite:
        _rewrite_fixture_before(f, frame)
    original_registry = f["registry"].read_bytes()
    with pytest.raises((ValueError, OSError)):
        _append_proof(f)
    assert f["registry"].read_bytes() == original_registry
    assert not (f["archive"]/registry.APPEND_PROOF_FILE).exists()


def test_append_proof_does_not_relax_other_registry_identity_fields(append_fixture):
    f = append_fixture
    f["document"]["research_references"][0]["installed"] = True
    f["registry"].write_text(json.dumps(f["document"]))
    with pytest.raises(ValueError, match="REFERENCE_IDENTITY"):
        prepare(f)
    assert not (f["archive"]/registry.SNAPSHOTS[0]).exists()
