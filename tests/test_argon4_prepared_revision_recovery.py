"""Offline interrupted-cutover recovery; never production evidence or network."""
from pathlib import Path
import json
import sqlite3

import pandas as pd
import pytest

from tools.argon4_forward import revisions
from test_argon4_forward_revisions import runtime, no_network, _add_receipt, _error, _files, NOW
from test_argon4_forward_mark_revisions import runtime as mark_runtime, _receipt as mark_receipt, _error as mark_error, NOW as MARK_NOW, SECOND


def prepared_after_clock_expiry(runtime):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    calls = []
    def expiring_clock():
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("clock calibration expired")
        return NOW
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], expiring_clock)
    assert result["status"] == "BLOCKED" and result["reason"] == "clock calibration expired"
    archive = Path(result["archive_path"])
    assert not (archive / "manifest.json").exists()
    with pytest.raises(revisions.RevisionSafetyError):
        revisions.ensure_view_committed(runtime)
    return archive


def test_readonly_verification_then_fresh_future_only_commit(runtime):
    archive = prepared_after_clock_expiry(runtime)
    before = revisions._inventory(runtime)
    verified = revisions.recover_prepared_price_revision(runtime)
    assert verified["status"] == "VERIFIED_PREPARED_CUTOVER", verified
    assert revisions._inventory(runtime) == before
    effective = NOW+pd.Timedelta(hours=2)
    current_view = _files(runtime)
    committed = revisions.recover_prepared_price_revision(runtime, apply=True, cutover_now=lambda: effective)
    assert committed["status"] == "COMMITTED", committed
    assert committed["view_effective_at"] == effective.isoformat()
    assert committed["current_cohort_must_be_skipped"] and committed["next_cohort_requires_fresh_validation"]
    assert _files(runtime) == current_view
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"
    for relative, digest in before.items():
        if relative != "features/revisions/view_state.json":
            assert revisions._sha(runtime / relative) == digest
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cutover_clock_mode"] == "FRESH_CALIBRATED_PREPARED_RECOVERY_CALLBACK"
    assert manifest["current_view_must_not_replay_as_of_before"] == effective.isoformat()
    again = revisions.recover_prepared_price_revision(runtime, apply=True, cutover_now=lambda: pytest.fail("idempotent recovery requested clock"))
    assert again["recovery_action"] == "NONE_ALREADY_COMMITTED"
    (archive / manifest["recovery_validation_file"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(revisions.RevisionSafetyError):
        revisions.ensure_view_committed(runtime)


@pytest.mark.parametrize("case", ["partial_raw", "old_archive", "proposed_raw", "tiny_proposed_and_current_change", "receipt", "audit_prefix", "unsupported_table", "prepared_changes"])
def test_recovery_refuses_ambiguous_or_corrupt_evidence_without_writes(runtime, case):
    archive = prepared_after_clock_expiry(runtime)
    relative = "intraday/BTCUSDT_5m_futures.parquet"
    if case == "partial_raw":
        (runtime / "features/raw_data" / relative).write_bytes((archive / "raw_data" / relative).read_bytes())
    elif case == "old_archive":
        (archive / "raw_data/auxiliary/unrelated_reference.txt").write_text("corrupt", encoding="utf-8")
    elif case == "proposed_raw":
        path = archive / "proposed_raw_data" / relative
        frame = pd.read_parquet(path)
        frame.loc[frame.index[-1], "close"] = 103
        frame.to_parquet(path, index=False)
    elif case == "tiny_proposed_and_current_change":
        path = archive / "proposed_raw_data" / relative
        frame = pd.read_parquet(path)
        frame.loc[frame.index[-1], "close"] += 1e-8
        frame.to_parquet(path, index=False)
        (runtime / "features/raw_data" / relative).write_bytes(path.read_bytes())
    elif case == "receipt":
        with sqlite3.connect(archive / "receipts.sqlite3") as conn:
            conn.execute("UPDATE receipts SET sha256=? WHERE id=3", ("f"*64,))
    elif case == "audit_prefix":
        with sqlite3.connect(archive / "observations.sqlite3") as conn:
            conn.execute("UPDATE decisions SET content='changed prior decision'")
    elif case == "unsupported_table":
        for path in (archive / "observations.sqlite3", runtime / "observations.sqlite3"):
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE unexpected(id INTEGER PRIMARY KEY)")
    elif case == "prepared_changes":
        path = archive / "prepared.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["changes"][0]["after"]["close"] = 104
        path.write_text(json.dumps(data), encoding="utf-8")
    before = revisions._inventory(runtime)
    result = revisions.recover_prepared_price_revision(runtime, apply=True, cutover_now=lambda: pytest.fail("unverified evidence requested cutover clock"))
    assert result["status"] == "BLOCKED", result
    assert revisions._inventory(runtime) == before
    with pytest.raises(revisions.RevisionSafetyError):
        revisions.ensure_view_committed(runtime)


def test_missing_or_expired_cutover_callback_leaves_prepared_evidence(runtime):
    prepared_after_clock_expiry(runtime)
    before = revisions._inventory(runtime)
    missing = revisions.recover_prepared_price_revision(runtime, apply=True)
    assert missing["reason"] == "RECOVERY_FRESH_CUTOVER_CLOCK_CALLBACK_REQUIRED"
    def expired(): raise ValueError("clock calibration expired")
    result = revisions.recover_prepared_price_revision(runtime, apply=True, cutover_now=expired)
    assert result["status"] == "BLOCKED"
    assert revisions._inventory(runtime) == before


def test_slow_archive_uses_new_cutover_clock_instead_of_expired_validation_clock(runtime, monkeypatch):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    archived = []
    backup = revisions._backup_sqlite
    def slow_backup(source, target):
        result = backup(source, target)
        archived.append(target)
        return result
    monkeypatch.setattr(revisions, "_backup_sqlite", slow_backup)
    calls = []
    def original_clock():
        calls.append(True)
        assert len(calls) == 1, "expired validation clock was reused at cutover"
        return NOW
    def fresh_clock():
        assert len(archived) == 2
        return NOW+pd.Timedelta(seconds=90)
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], original_clock, cutover_now=fresh_clock)
    assert result["status"] == "COMMITTED", result
    assert result["view_effective_at"] == (NOW+pd.Timedelta(seconds=90)).isoformat()


@pytest.mark.parametrize("case", ["valid", "unpublished_registry", "unrelated_registry_change", "unrelated_archived_registry_change"])
def test_mark_registry_must_be_exact_already_published_proposal(mark_runtime, case):
    runtime = mark_runtime
    mark_receipt(runtime)
    mark_receipt(runtime, received=SECOND)
    def expired(): raise ValueError("clock calibration expired")
    registry = runtime / "fixture_mark_registry.json"
    result = revisions.reconcile_confirmed_price_revisions(runtime, [mark_error()], MARK_NOW,
        mark_registry_path=registry, cutover_now=expired)
    assert result["status"] == "BLOCKED"
    archive = Path(result["archive_path"])
    if case == "unpublished_registry":
        registry.write_bytes((archive / "mark_registry_before.json").read_bytes())
    elif case in ("unrelated_registry_change", "unrelated_archived_registry_change"):
        document = json.loads(registry.read_text(encoding="utf-8"))
        document["datasets"].append({"should_never_change_canonical_entries": True})
        registry.write_text(json.dumps(document), encoding="utf-8")
        if case == "unrelated_archived_registry_change":
            (archive / "mark_registry_proposed.json").write_bytes(registry.read_bytes())
    before = revisions._inventory(runtime)
    verified = revisions.recover_prepared_price_revision(runtime, mark_registry_path=registry)
    assert revisions._inventory(runtime) == before
    if case == "valid":
        assert verified["status"] == "VERIFIED_PREPARED_CUTOVER", verified
        result = revisions.recover_prepared_price_revision(runtime, mark_registry_path=registry,
            apply=True, cutover_now=lambda: MARK_NOW+pd.Timedelta(hours=2))
        assert result["status"] == "COMMITTED", result
        assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"
        for relative, digest in before.items():
            if relative != "features/revisions/view_state.json":
                assert revisions._sha(runtime / relative) == digest
    else:
        assert verified["status"] == "BLOCKED", verified


def test_prepared_recovery_reproduces_verified_append_baseline_proof(mark_runtime):
    from copy import deepcopy
    import pandas as pd
    from test_argon4_forward_mark_revisions import (
        bar, _row, MARK_PATH, OPEN, NEW_MARK,
    )
    runtime = mark_runtime
    registry = runtime/"fixture_mark_registry.json"
    mark_receipt(runtime)
    mark_receipt(runtime, received=SECOND)
    first = revisions.reconcile_confirmed_price_revisions(
        runtime, [mark_error()], MARK_NOW, mark_registry_path=registry)
    assert first["status"] == "COMMITTED", first
    registry_before_append = registry.read_bytes()
    # A later real-observation fixture appends one closed hour. The registry's
    # prior digest remains an immutable committed anchor, not the new full file.
    appended_bar = bar(OPEN+pd.Timedelta(hours=1))
    appended_receipt = mark_receipt(runtime, appended_bar, received=MARK_NOW+pd.Timedelta(minutes=5))
    path = runtime/"features/raw_data"/MARK_PATH
    frame = pd.read_parquet(path)
    pd.concat([frame, pd.DataFrame([_row(appended_bar, appended_receipt)])], ignore_index=True).to_parquet(path, index=False)
    assert registry.read_bytes() == registry_before_append
    revised = deepcopy(NEW_MARK)
    revised[4], revised[8] = "2226.0", 3601
    mark_receipt(runtime, revised, received=MARK_NOW+pd.Timedelta(minutes=10))
    mark_receipt(runtime, revised, received=MARK_NOW+pd.Timedelta(minutes=15))
    def expired(): raise ValueError("synthetic post-publication clock expiry")
    pending = revisions.reconcile_confirmed_price_revisions(
        runtime, [mark_error()], MARK_NOW+pd.Timedelta(minutes=20),
        mark_registry_path=registry, cutover_now=expired)
    assert pending["status"] == "BLOCKED" and pending["reason"] == "synthetic post-publication clock expiry", pending
    archive = Path(pending["archive_path"])
    assert (archive/"mark_append_baseline_proofs.json").is_file()
    before = revisions._inventory(runtime)
    checked = revisions.recover_prepared_price_revision(runtime, mark_registry_path=registry)
    assert checked["status"] == "VERIFIED_PREPARED_CUTOVER", checked
    assert revisions._inventory(runtime) == before
    result = revisions.recover_prepared_price_revision(runtime, mark_registry_path=registry,
        apply=True, cutover_now=lambda: MARK_NOW+pd.Timedelta(hours=1))
    assert result["status"] == "COMMITTED", result
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    proof_receipt = manifest["mark_registry_update"]
    assert len(proof_receipt["append_baseline_proofs"]) == 1
    assert proof_receipt["append_baseline_proof_archive"]["sha256"] == revisions._sha(archive/"mark_append_baseline_proofs.json")
    assert result["current_cohort_must_be_skipped"] and result.get("same_cohort_eligible") is not True
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"
