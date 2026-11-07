"""Offline registry publication fixtures; never target the workspace registry."""
from copy import deepcopy
import hashlib
import json

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
