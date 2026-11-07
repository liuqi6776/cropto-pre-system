"""Synthetic stopped-writer evidence for retiring an unapplied transaction."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from tools.argon4_forward import revisions
from test_argon4_forward_revisions import runtime, no_network, _add_receipt, _error, NEW, NOW


def interrupted_before_replacement(runtime, monkeypatch):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    prior = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert prior["status"] == "COMMITTED"
    prior_marker = revisions.ensure_view_committed(runtime)
    changed = deepcopy(NEW)
    changed[4], changed[8] = "103.0", 24
    _add_receipt(runtime, "2026-11-06T01:50:05Z", changed)
    _add_receipt(runtime, "2026-11-06T01:55:05Z", changed)
    atomic = revisions._atomic_json
    def interrupted(path, value):
        atomic(path, value)
        if Path(path).name == "view_state.json" and value.get("status") == "PREPARED":
            raise OSError("synthetic crash after barrier, before any replacement")
    with monkeypatch.context() as scoped:
        scoped.setattr(revisions, "_atomic_json", interrupted)
        failed = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW+pd.Timedelta(minutes=15))
    assert failed["status"] == "BLOCKED"
    with pytest.raises(revisions.RevisionSafetyError): revisions.ensure_view_committed(runtime)
    return Path(failed["archive_path"]), prior_marker


def test_abort_only_exact_before_preserves_all_evidence_and_restores_verified_prior(runtime, monkeypatch):
    archive, prior = interrupted_before_replacement(runtime, monkeypatch)
    before = revisions._inventory(runtime)
    result = revisions.abort_unapplied_price_revision(runtime)
    assert result["status"] == "VERIFIED_UNAPPLIED_REVISION", result
    assert revisions._inventory(runtime) == before
    result = revisions.abort_unapplied_price_revision(runtime, apply=True,
                abort_now=lambda: NOW+pd.Timedelta(minutes=20))
    assert result["status"] == "ABORTED_UNAPPLIED", result
    assert result["same_cohort_eligible"] is False
    assert revisions.ensure_view_committed(runtime) == prior
    record = Path(result["abort_evidence_path"])
    assert record.parent == archive
    assert json.loads(record.read_text())["status"] == "ABORTED_UNAPPLIED"
    for relative, digest in before.items():
        if relative != "features/revisions/view_state.json":
            assert revisions._sha(runtime/relative) == digest
    again = revisions.abort_unapplied_price_revision(runtime)
    assert again["status"] == "BLOCKED" and again["reason"] == "ABORT_REQUIRES_PREPARED_TRANSACTION"


@pytest.mark.parametrize("corruption", ["partial_current", "current_append", "archive_before", "proposal",
                                        "receipt", "audit_prefix", "prior_manifest", "prior_identity",
                                        "registry_snapshot", "manifest"])
def test_abort_refuses_any_ambiguous_or_corrupt_transaction_without_writes(runtime, monkeypatch, corruption):
    archive, prior = interrupted_before_replacement(runtime, monkeypatch)
    relative = "intraday/BTCUSDT_5m_futures.parquet"
    if corruption == "partial_current":
        (runtime/"features/raw_data"/relative).write_bytes((archive/"proposed_raw_data"/relative).read_bytes())
    elif corruption == "current_append":
        (runtime/"features/raw_data/new_observation.txt").write_text("later valid writes still prohibit exact-before abort")
    elif corruption == "archive_before":
        (archive/"raw_data/auxiliary/unrelated_reference.txt").write_text("changed")
    elif corruption == "proposal":
        path = archive/"proposed_raw_data"/relative
        frame = pd.read_parquet(path)
        frame.loc[frame.index[-1], "close"] += 1e-8
        frame.to_parquet(path, index=False)
    elif corruption == "receipt":
        with sqlite3.connect(archive/"receipts.sqlite3") as db:
            db.execute("UPDATE receipts SET raw=? WHERE id=(SELECT MAX(id) FROM receipts)", (b"[]",))
    elif corruption == "audit_prefix":
        with sqlite3.connect(archive/"observations.sqlite3") as db:
            db.execute("UPDATE decisions SET content='mutated prior decision'")
    elif corruption == "prior_manifest":
        (archive.parent/prior["revision_id"]/"manifest.json").write_text("invalid")
    elif corruption == "prior_identity":
        path = archive/"prepared.json"
        prepared = json.loads(path.read_text())
        prepared["prior_view"] = {"status": "NO_REVISIONS"}
        path.write_text(json.dumps(prepared))
    elif corruption == "registry_snapshot": (archive/"mark_registry_before.json").write_text("{}")
    elif corruption == "manifest": (archive/"manifest.json").write_text("{}")
    before = revisions._inventory(runtime)
    result = revisions.abort_unapplied_price_revision(runtime, apply=True,
        abort_now=lambda: pytest.fail("unverified abort requested time"))
    assert result["status"] == "BLOCKED", result
    assert revisions._inventory(runtime) == before
    with pytest.raises(revisions.RevisionSafetyError): revisions.ensure_view_committed(runtime)


@pytest.mark.parametrize("case", ["missing", "reversed"])
def test_abort_requires_fresh_nonreversed_publication_time(runtime, monkeypatch, case):
    interrupted_before_replacement(runtime, monkeypatch)
    before = revisions._inventory(runtime)
    callback = None if case == "missing" else lambda: NOW
    result = revisions.abort_unapplied_price_revision(runtime, apply=True, abort_now=callback)
    assert result["status"] == "BLOCKED"
    assert revisions._inventory(runtime) == before


def test_preparation_failure_keeps_previous_active_marker_and_preserves_failed_archive(runtime, monkeypatch):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    before = revisions._inventory(runtime/"features/raw_data")
    def broken_backup(source, target): raise OSError("synthetic archive failure")
    monkeypatch.setattr(revisions, "_backup_sqlite", broken_backup)
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert result["status"] == "BLOCKED"
    assert Path(result["archive_path"]).exists()
    assert revisions._inventory(runtime/"features/raw_data") == before
    assert revisions.ensure_view_committed(runtime) == {"status": "NO_REVISIONS"}


@pytest.mark.parametrize("change", ["current", "proposal", "marker", "prepared", "prior", "publication", "receipt_archive", "live_decision"])
def test_abort_rechecks_after_slow_clock_callback_before_writing_any_abort_record(runtime, monkeypatch, change):
    archive, prior = interrupted_before_replacement(runtime, monkeypatch)
    def clock():
        if change == "current":
            (runtime/"features/raw_data/new_observation.txt").write_text("concurrent input writer")
        elif change == "proposal":
            (archive/"proposed_raw_data/new_proposal.txt").write_text("changed proposal")
        elif change == "marker":
            (archive.parent/"view_state.json").write_text("changed active identity")
        elif change == "prepared":
            (archive/"prepared.json").write_text("changed prepared identity")
        elif change == "prior":
            (archive.parent/prior["revision_id"]/"manifest.json").write_text("changed prior identity")
        elif change == "publication":
            (archive/"mark_registry_proposed.json").write_text("{}")
        elif change == "receipt_archive":
            with sqlite3.connect(archive/"receipts.sqlite3") as db:
                db.execute("UPDATE receipts SET raw=? WHERE id=(SELECT MAX(id) FROM receipts)", (b"[]",))
        elif change == "live_decision":
            with sqlite3.connect(runtime/"observations.sqlite3") as db:
                db.execute("UPDATE decisions SET content='concurrent historical mutation'")
        return NOW+pd.Timedelta(minutes=20)
    result = revisions.abort_unapplied_price_revision(runtime, apply=True, abort_now=clock)
    assert result["status"] == "BLOCKED", result
    assert not list(archive.glob("abort_unapplied_validation_*.json"))
    marker = (archive.parent/"view_state.json").read_text()
    if change == "marker":
        assert marker == "changed active identity"
    else:
        assert json.loads(marker)["status"] == "PREPARED"
