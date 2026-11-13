"""Exact old/new formula parity and snapshot isolation (offline fixtures only)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from tools.argon4_forward import feature_cache as cache
from tools.argon4_forward import model as reference
from test_argon4_forward_model import raw_fixture


def test_real_model_verifier_requires_every_pin_and_does_not_swallow_missing_model(monkeypatch):
    from scripts import verify_argon4_feature_cache as verify
    with pytest.raises(ValueError, match="all three"):
        verify.load_bundles({"deployment_manifest_sha256": {"ETH_5m": "a"}}, "deployment_q4")
    called = []
    def missing(stream, phase, *, expected_manifest_sha256):
        called.append((stream, phase, expected_manifest_sha256))
        if stream == "BTC_5m":
            raise FileNotFoundError("required deployed model unavailable")
        return {"stream": stream}
    monkeypatch.setattr(reference, "load_model_bundle", missing)
    pins = {s: s + "-pin" for s in verify.STREAMS}
    with pytest.raises(FileNotFoundError, match="required deployed"):
        verify.load_bundles({"deployment_manifest_sha256": pins}, "deployment_q4")
    assert called == [(s, "deployment_q4", pins[s]) for s in verify.STREAMS]


def test_real_model_verifier_rejects_zero_cohort_claims():
    from scripts import verify_argon4_feature_cache as verify
    with pytest.raises(ValueError, match="positive"):
        verify.select_cohorts(Path("does-not-exist.sqlite3"), "deployment_q4", 0)


def test_real_snapshot_verifier_rejects_bytes_different_from_recorded_collector(raw_fixture):
    from scripts import verify_argon4_feature_cache as verify
    root, _ = raw_fixture
    with pytest.raises(ValueError, match="recorded collector hash"):
        verify.frozen_inputs(root, root.parent / "audit_copy", expected_hashes={})


def _equal(left, right):
    for name in ("base", "expert", "raw_target"):
        pd.testing.assert_frame_equal(left[name], right[name], check_exact=True)
        # Numeric equality alone treats +0 and -0 as equal; compare payload too.
        assert left[name].to_numpy().tobytes() == right[name].to_numpy().tobytes()
    assert left["availability"] == right["availability"]
    assert left["provenance"] == right["provenance"]
    assert left["source_sha256"] == right["source_sha256"]


def test_actual_distinct_implementation_matches_all_features_and_reduces_reads(raw_fixture, monkeypatch):
    root, idx = raw_fixture
    cutoff = idx[-1] + pd.Timedelta(minutes=5)
    reads = []
    original = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda path, *a, **kw: (reads.append(path), original(path, *a, **kw))[1])
    old = {s: reference.build_feature_frames(root, s, as_of=cutoff, tail_rows=None)
           for s in ("ETHUSDT", "BTCUSDT")}
    old_reads = len(reads)
    reads.clear()
    modules_before = set(sys.modules)
    with cache.FeatureSnapshot(root, as_of=cutoff) as snapshot:
        new = {s: snapshot.build(s, tail_rows=None) for s in ("ETHUSDT", "BTCUSDT")}
        for symbol in old:
            _equal(old[symbol], new[symbol])
            assert new[symbol]["base"].shape == (len(idx), 311)
            assert new[symbol]["expert"].shape == (len(idx), 65)
        assert snapshot.stats["physical_reads"] == 18
        assert snapshot.stats["parquet_decodes"] == len(reads) == 18
        assert snapshot.stats["validated_cache_hits"] >= 18
        assert snapshot.stats["input_history_truncated"] is False
    assert old_reads == 36
    assert not any(n.startswith("argon4_cohort_features_") for n in set(sys.modules) - modules_before)


def test_compiled_code_reused_but_input_namespaces_are_fresh(raw_fixture, monkeypatch):
    root, idx = raw_fixture
    cutoff = idx[-1] + pd.Timedelta(minutes=5)
    cache._CODE_CACHE.clear()
    first = cache.FeatureSnapshot(root, as_of=cutoff)
    assert first.stats["compile_cache_hit"] is False
    second = cache.FeatureSnapshot(root, as_of=cutoff)
    assert second.stats["compile_cache_hit"] is True
    assert first._codes[0] is second._codes[0]
    with cache._fresh_modules(first._codes, lambda p: "first", root) as (a, _):
        with cache._fresh_modules(second._codes, lambda p: "second", root) as (b, _):
            assert a is not b and a.__dict__ is not b.__dict__
            assert a._guarded_read(None) == "first"
            assert b._guarded_read(None) == "second"
    first.close()
    second.close()
    monkeypatch.setitem(reference.SOURCE_HASHES, "step2_standalone_clean_room_master.py", "0" * 64)
    with pytest.raises(reference.ModelInputError, match="source hash"):
        cache.FeatureSnapshot(root, as_of=cutoff)


def test_captured_bytes_survive_later_disk_publication_without_cross_cohort_cache(raw_fixture):
    root, idx = raw_fixture
    cutoff = idx[-1] + pd.Timedelta(minutes=5)
    old = reference.build_feature_frames(root, "ETHUSDT", as_of=cutoff, tail_rows=1)
    with cache.FeatureSnapshot(root, as_of=cutoff) as snapshot:
        path = root / "intraday/ETHUSDT_5m_futures.parquet"
        frame = pd.read_parquet(path)
        frame.loc[len(frame)-1, "close"] += .2
        frame.to_parquet(path, index=False)
        _equal(old, snapshot.build("ETHUSDT"))
        old_digest = snapshot.stats["source_hashes"]["intraday/ETHUSDT_5m_futures.parquet"]
    with cache.FeatureSnapshot(root, as_of=cutoff) as new:
        assert new.stats["source_hashes"]["intraday/ETHUSDT_5m_futures.parquet"] != old_digest
        actual = new.build("ETHUSDT")
        expected = reference.build_feature_frames(root, "ETHUSDT", as_of=cutoff, tail_rows=1)
        _equal(expected, actual)
        assert not np.array_equal(old["base"].to_numpy(), actual["base"].to_numpy())


def test_copies_cannot_mutate_shared_validated_frame(raw_fixture):
    root, idx = raw_fixture
    cutoff = idx[-1] + pd.Timedelta(minutes=5)
    with cache.FeatureSnapshot(root, as_of=cutoff) as snapshot:
        with pytest.raises(AttributeError):
            snapshot.cutoff = cutoff + pd.Timedelta(minutes=5)
        with pytest.raises(AttributeError):
            snapshot.root = root / "other"
        path = root / "intraday/BTCUSDT_5m_futures.parquet"
        frame, meta, fingerprint = snapshot._read_validated(path)
        value = frame.close.iloc[-1]
        frame.loc[frame.index[-1], "close"] = -1
        meta.clear()
        fingerprint.clear()
        again, fresh_meta, fresh_fp = snapshot._read_validated(path)
        assert again.close.iloc[-1] == value
        assert fresh_meta and fresh_fp


def test_same_cutoff_filters_actual_received_time_identically(raw_fixture):
    root, idx = raw_fixture
    cutoff = idx[-1] + pd.Timedelta(minutes=5)
    for coin in reference.SYMBOLS:
        path = root / "intraday" / f"{coin}_5m_futures.parquet"
        frame = pd.read_parquet(path)
        frame["first_received_at"] = pd.to_datetime(frame.open_time, utc=True) + pd.Timedelta(minutes=5)
        frame.loc[len(frame)-1, "first_received_at"] = cutoff + pd.Timedelta(seconds=3)
        frame.to_parquet(path, index=False)
    for delta, expected_index in ((0, idx[-2]), (4, idx[-1])):
        as_of = cutoff + pd.Timedelta(seconds=delta)
        old = reference.build_feature_frames(root, "BTCUSDT", as_of=as_of, tail_rows=1)
        with cache.FeatureSnapshot(root, as_of=as_of) as snapshot:
            _equal(old, snapshot.build("BTCUSDT"))
            assert snapshot.build("BTCUSDT")["index"][-1] == expected_index


@pytest.mark.parametrize("damage,match", [("gap", "contiguous"), ("duplicate", "duplicate"), ("nan", "invalid observed OHLC")])
def test_corrupt_raw_inputs_fail_same_way_as_reference(raw_fixture, damage, match):
    root, idx = raw_fixture
    path = root / "intraday/BTCUSDT_5m_futures.parquet"
    frame = pd.read_parquet(path)
    if damage == "gap":
        frame = frame.drop(index=1800)
    elif damage == "duplicate":
        frame.loc[1800, "open_time"] = frame.loc[1799, "open_time"]
    else:
        frame.loc[1800, "close"] = np.nan
    frame.to_parquet(path, index=False)
    cutoff = idx[-1] + pd.Timedelta(minutes=5)
    with pytest.raises(reference.ModelInputError, match=match):
        reference.build_feature_frames(root, "BTCUSDT", as_of=cutoff)
    with cache.FeatureSnapshot(root, as_of=cutoff) as snapshot:
        with pytest.raises(reference.ModelInputError, match=match):
            snapshot.build("BTCUSDT")


def test_snapshot_requires_collector_hashes_for_every_input_when_supplied(raw_fixture):
    root, idx = raw_fixture
    expected = {str(p): {"path": str(p), "sha256": reference.sha256(p)} for p in cache.required_paths(root)}
    cutoff = idx[-1] + pd.Timedelta(minutes=5)
    with cache.FeatureSnapshot(root, as_of=cutoff, expected_sources=expected) as snapshot:
        assert snapshot.stats["physical_reads"] == 18
    expected[next(iter(expected))]["sha256"] = "f" * 64
    with pytest.raises(reference.ModelInputError, match="collector source"):
        cache.FeatureSnapshot(root, as_of=cutoff, expected_sources=expected)
    with pytest.raises(reference.ModelInputError, match="collector source"):
        cache.FeatureSnapshot(root, as_of=cutoff, expected_sources={})


def test_capture_detects_replacement_while_collecting_files(raw_fixture, monkeypatch):
    root, idx = raw_fixture
    first, second, *_ = cache.required_paths(root)
    original = Path.read_bytes
    modified = []
    def read(path):
        result = original(path)
        if path == second and not modified:
            first.write_bytes(original(first) + b"changed")
            modified.append(True)
        return result
    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(reference.ModelInputError, match="changed during snapshot"):
        cache.FeatureSnapshot(root, as_of=idx[-1] + pd.Timedelta(minutes=5))


def test_snapshot_cannot_be_reused_with_different_root_or_cutoff_or_after_close(raw_fixture):
    root, idx = raw_fixture
    cutoff = idx[-1] + pd.Timedelta(minutes=5)
    snapshot = cache.FeatureSnapshot(root, as_of=cutoff)
    with pytest.raises(reference.ModelInputError, match="scope mismatch"):
        cache.build_feature_frames(root, as_of=cutoff + pd.Timedelta(seconds=1), snapshot=snapshot)
    with pytest.raises(reference.ModelInputError, match="scope mismatch"):
        cache.build_feature_frames(root.parent, as_of=cutoff, snapshot=snapshot)
    snapshot.close()
    with pytest.raises(reference.ModelInputError, match="already closed"):
        snapshot.build("BTCUSDT")


def test_concurrent_three_symbol_builds_are_byte_identical(raw_fixture):
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits
    root, idx = raw_fixture
    cutoff = idx[-1] + pd.Timedelta(minutes=5)
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    with cache.FeatureSnapshot(root, as_of=cutoff) as serial_snap:
        serial = {s: serial_snap.build(s, tail_rows=1) for s in symbols}
        assert serial_snap.stats["builds"] == 3
    modules_before = set(sys.modules)
    with cache.FeatureSnapshot(root, as_of=cutoff) as conc_snap:
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futs = {s: pool.submit(conc_snap.build, s, tail_rows=1) for s in symbols}
                concurrent = {s: futs[s].result() for s in symbols}
        assert conc_snap.stats["builds"] == 3
        assert conc_snap.stats["parquet_decodes"] == 18
        for s in symbols:
            _equal(serial[s], concurrent[s])
    assert not any(n.startswith("argon4_cohort_features_") for n in set(sys.modules) - modules_before)
