"""Byte-keyed mark registration cache, using isolated synthetic fixtures only."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from tools.argon4_forward import pipeline


def _write_frame(path, frame):
    # Fixed-width, uncompressed values permit a genuine same-size byte edit.
    frame.to_parquet(path, index=False, compression=None, write_statistics=False)


@pytest.fixture
def registry_fixture(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("mark registry tests must never access a network")
    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "_MARK_REGISTRY_CACHE", {})
    data_root = tmp_path / "paper_runtime/argon4_forward_20261105/features/raw_data"
    auxiliary = data_root / "auxiliary"
    auxiliary.mkdir(parents=True)
    date = pd.date_range("2026-11-06T07:00:00Z", periods=2, freq="h")
    frame = pd.DataFrame({
        "date": date, "open": [100., 101.], "high": [102., 103.],
        "low": [99., 100.], "close": [101., 102.], "volume": [0., 0.],
        "first_received_at": date + pd.Timedelta(hours=1, seconds=2),
        "receipt_sha256": ["a" * 64, "b" * 64],
        "receipt_time_basis": ["CALIBRATED_UTC_UPPER_BOUND"] * 2,
        "clock_evidence_sha256": ["c" * 64] * 2,
    })
    paths = {}
    for symbol in pipeline.SYMBOLS:
        paths[symbol] = auxiliary / f"{symbol}_1h_mark.parquet"
        _write_frame(paths[symbol], frame)
    path = tmp_path / "archive/catalog/mark_candle_registry.json"
    path.parent.mkdir(parents=True)
    initial = {
        "schema_version": 1,
        "datasets": [{"dataset_id": "canonical_fixture", "installed": True, "sha256": "d" * 64}],
        "research_references": [{"dataset_id": "unrelated_fixture", "note": "preserve exactly"}],
        "outside_note": "old",
    }
    path.write_text(json.dumps(initial, indent=2), encoding="utf-8")
    return SimpleNamespace(root=tmp_path, data_root=data_root, paths=paths,
                           registry=path, initial=initial, frame=frame)


def _spy_parquet(monkeypatch):
    original = pd.read_parquet
    calls = []
    def read(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", read)
    return calls


def _reference(fixture, symbol="ETHUSDT"):
    registry = json.loads(fixture.registry.read_text(encoding="utf-8"))
    target = f"argon4_local_v4_online_mark_{symbol}_20261106"
    return next(r for r in registry["research_references"] if r["dataset_id"] == target)


def test_first_registration_validates_all_then_cache_hit_reads_all_bytes(registry_fixture, monkeypatch):
    f = registry_fixture
    parses = _spy_parquet(monkeypatch)
    assert pipeline.register_online_marks(f.data_root) == {
        "status": "VALIDATED_AND_REGISTERED", "source_count": 4}
    assert len(parses) == 4
    registry_before = f.registry.read_bytes()
    original_read_bytes = Path.read_bytes
    byte_reads = []
    def read_bytes(path):
        byte_reads.append(path)
        return original_read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert pipeline.register_online_marks(f.data_root) == {
        "status": "UNCHANGED_VERIFIED", "source_count": 4}
    assert len(parses) == 4  # parsing is the only skipped data work
    assert set(byte_reads) == {f.registry, *f.paths.values()}
    assert original_read_bytes(f.registry) == registry_before
    result = json.loads(registry_before)
    assert result["datasets"] == f.initial["datasets"]
    assert f.initial["research_references"][0] in result["research_references"]
    assert all(_reference(f, s)["gate_effect"] == "NONE" for s in pipeline.SYMBOLS)
    assert all(_reference(f, s)["installed"] is False for s in pipeline.SYMBOLS)


def test_same_size_same_mtime_changed_parquet_cannot_hit_cache(registry_fixture, monkeypatch):
    f = registry_fixture
    parses = _spy_parquet(monkeypatch)
    pipeline.register_online_marks(f.data_root)
    target = f.paths["ETHUSDT"]
    before = target.read_bytes()
    stat = target.stat()
    revised = f.frame.copy(deep=True)
    revised.loc[1, "close"] = 102.25
    _write_frame(target, revised)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert target.stat().st_size == stat.st_size
    assert target.stat().st_mtime_ns == stat.st_mtime_ns
    assert target.read_bytes() != before
    assert pipeline.register_online_marks(f.data_root)["status"] == "VALIDATED_AND_REGISTERED"
    assert len(parses) == 8
    assert _reference(f)["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert pipeline.register_online_marks(f.data_root)["status"] == "UNCHANGED_VERIFIED"


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_existing_equivalent_registry_encoding_is_verified_by_actual_published_bytes(registry_fixture, monkeypatch, newline):
    f = registry_fixture
    pipeline.register_online_marks(f.data_root)
    logical = json.loads(f.registry.read_text(encoding="utf-8"))
    content = json.dumps(logical, ensure_ascii=False, indent=2, allow_nan=False)
    actual = content.replace("\n", newline).encode("utf-8")
    f.registry.write_bytes(actual)
    pipeline._MARK_REGISTRY_CACHE.clear()  # process restart, all bytes revalidated
    parses = _spy_parquet(monkeypatch)
    assert pipeline.register_online_marks(f.data_root)["status"] == "VALIDATED_AND_REGISTERED"
    assert len(parses) == 4
    assert f.registry.read_bytes() == actual  # equal logical JSON need not be rewritten
    assert pipeline._MARK_REGISTRY_CACHE[str(f.data_root.resolve())][0] == hashlib.sha256(actual).hexdigest()
    assert pipeline.register_online_marks(f.data_root)["status"] == "UNCHANGED_VERIFIED"
    assert len(parses) == 4


@pytest.mark.parametrize("same_metadata", [False, True])
def test_external_registry_edit_misses_and_preserves_unrelated_records(registry_fixture, monkeypatch, same_metadata):
    f = registry_fixture
    parses = _spy_parquet(monkeypatch)
    pipeline.register_online_marks(f.data_root)
    stat = f.registry.stat()
    external = json.loads(f.registry.read_text(encoding="utf-8"))
    external["outside_note"] = "new"
    if not same_metadata:
        external["datasets"].append({"dataset_id": "second_canonical_fixture", "value": 73})
        external["research_references"].append({"dataset_id": "second_research_fixture", "value": 19})
    f.registry.write_text(json.dumps(external, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if same_metadata:
        os.utime(f.registry, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert f.registry.stat().st_size == stat.st_size
        assert f.registry.stat().st_mtime_ns == stat.st_mtime_ns
    assert pipeline.register_online_marks(f.data_root)["status"] == "VALIDATED_AND_REGISTERED"
    assert len(parses) == 8
    saved = json.loads(f.registry.read_text(encoding="utf-8"))
    # Registration groups its own entries last; external entry contents and
    # their relative order must survive, but the whole list need not be stable.
    assert {k: v for k, v in saved.items() if k != "research_references"} == {
        k: v for k, v in external.items() if k != "research_references"}
    assert len(saved["research_references"]) == len(external["research_references"])
    assert {r["dataset_id"]: r for r in saved["research_references"]} == {
        r["dataset_id"]: r for r in external["research_references"]}
    assert [r for r in saved["research_references"] if not r["dataset_id"].startswith("argon4_")] == [
        r for r in external["research_references"] if not r["dataset_id"].startswith("argon4_")]
    assert pipeline.register_online_marks(f.data_root)["status"] == "UNCHANGED_VERIFIED"


def test_new_closed_bar_is_registered_with_exact_hash_and_receipts(registry_fixture, monkeypatch):
    f = registry_fixture
    parses = _spy_parquet(monkeypatch)
    pipeline.register_online_marks(f.data_root)
    before = {s: _reference(f, s) for s in pipeline.SYMBOLS}
    added = f.frame.iloc[[-1]].copy(deep=True)
    added["date"] += pd.Timedelta(hours=1)
    added["first_received_at"] += pd.Timedelta(hours=1)
    added["receipt_sha256"] = "e" * 64
    extended = pd.concat([f.frame, added], ignore_index=True)
    assert (extended.first_received_at >= extended.date + pd.Timedelta(hours=1)).all()
    _write_frame(f.paths["ETHUSDT"], extended)
    assert pipeline.register_online_marks(f.data_root)["status"] == "VALIDATED_AND_REGISTERED"
    assert len(parses) == 8
    updated = _reference(f)
    assert updated["rows"] == 3 and updated["last_utc"] == str(added.date.iloc[0])
    assert updated["sha256"] == hashlib.sha256(f.paths["ETHUSDT"].read_bytes()).hexdigest()
    assert updated["source_receipt_sha256"] == ["a" * 64, "b" * 64, "e" * 64]
    for symbol in set(pipeline.SYMBOLS) - {"ETHUSDT"}:
        assert _reference(f, symbol) == before[symbol]


@pytest.mark.parametrize("defect", ["schema", "empty", "nan", "duplicate", "reverse", "missing_receipt_time", "bad_receipt_hash", "corrupt_file"])
def test_changed_invalid_source_fails_closed_without_publishing_or_poisoning_cache(registry_fixture, monkeypatch, defect):
    f = registry_fixture
    parses = _spy_parquet(monkeypatch)
    pipeline.register_online_marks(f.data_root)
    original_registry = f.registry.read_bytes()
    original_cache = deepcopy(pipeline._MARK_REGISTRY_CACHE)
    target = f.paths["ETHUSDT"]
    original_source = target.read_bytes()
    broken = f.frame.copy(deep=True)
    if defect == "schema": broken = broken.drop(columns="close")
    elif defect == "empty": broken = broken.iloc[:0]
    elif defect == "nan": broken.loc[1, "close"] = np.nan
    elif defect == "duplicate": broken.loc[1, "date"] = broken.loc[0, "date"]
    elif defect == "reverse": broken = broken.iloc[::-1].reset_index(drop=True)
    elif defect == "missing_receipt_time": broken.loc[1, "first_received_at"] = pd.NaT
    elif defect == "bad_receipt_hash": broken.loc[1, "receipt_sha256"] = "not-a-sha256"
    if defect == "corrupt_file": target.write_bytes(b"not-a-parquet-fixture")
    else: _write_frame(target, broken)
    for _ in range(2):
        with pytest.raises(ValueError):
            pipeline.register_online_marks(f.data_root)
        assert f.registry.read_bytes() == original_registry
        assert pipeline._MARK_REGISTRY_CACHE == original_cache
    assert len(parses) > 4
    target.write_bytes(original_source)
    assert pipeline.register_online_marks(f.data_root)["status"] == "UNCHANGED_VERIFIED"


def test_invalid_external_registry_is_not_hidden_by_prior_cache_hit(registry_fixture):
    f = registry_fixture
    pipeline.register_online_marks(f.data_root)
    original_cache = deepcopy(pipeline._MARK_REGISTRY_CACHE)
    f.registry.write_bytes(b"not-json-fixture")
    with pytest.raises(json.JSONDecodeError):
        pipeline.register_online_marks(f.data_root)
    assert f.registry.read_bytes() == b"not-json-fixture"
    assert pipeline._MARK_REGISTRY_CACHE == original_cache


def test_registry_external_edit_during_validation_aborts_own_publication(registry_fixture, monkeypatch):
    f = registry_fixture
    original = pd.read_parquet
    external = deepcopy(f.initial)
    external["research_references"].append({"dataset_id": "concurrent_fixture", "value": 9})
    edited = False
    def read(*args, **kwargs):
        nonlocal edited
        if not edited:
            f.registry.write_text(json.dumps(external, indent=2), encoding="utf-8")
            edited = True
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", read)
    with pytest.raises(ValueError, match="changed during publication"):
        pipeline.register_online_marks(f.data_root)
    assert json.loads(f.registry.read_text(encoding="utf-8")) == external
    assert pipeline._MARK_REGISTRY_CACHE == {}


def test_scope_mismatch_rejected_before_registry_or_source_reads(registry_fixture, monkeypatch):
    f = registry_fixture
    def forbidden_read(*args, **kwargs):
        pytest.fail("out-of-scope registry must not read any bytes")
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(ValueError, match="scope mismatch"):
        pipeline.register_online_marks(f.root / "different_raw_data")
