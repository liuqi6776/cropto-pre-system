"""Synthetic, offline revision fixtures; no production views or network."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from tools.argon4_forward import revisions
from test_argon4_forward_revisions import (
    runtime, no_network, _add_receipt, _error, _files, NEW, NOW,
)


def _context(cohort=NOW):
    return {"profile": revisions.FAST_PROFILE, "cohort": cohort.isoformat(),
            "deadline": (cohort+pd.Timedelta(seconds=14)).isoformat(),
            "minimum_spacing_seconds": 2.0, "clock_evidence_sha256": "c"*64}


def _guard():
    return {"eligible": True, "clock_evidence_sha256": "c"*64, "policy_sha256": "d"*64,
            "prediction_count": 0, "cohort_order_count": 0}


def _confirm(runtime, cohort=NOW, symbol="BTCUSDT"):
    _add_receipt(runtime, cohort+pd.Timedelta(seconds=4), symbol=symbol)
    _add_receipt(runtime, cohort+pd.Timedelta(seconds=6.4), symbol=symbol)


def _run(runtime, *, cohort=NOW, now=None, cutover_now=None, context=None, guard=_guard, errors=None):
    return revisions.reconcile_confirmed_price_revisions(
        runtime, errors or [_error()], now=now or (lambda: cohort+pd.Timedelta(seconds=7)),
        cutover_now=cutover_now or (lambda: cohort+pd.Timedelta(seconds=8)),
        fast_context=_context(cohort) if context is None else context, cohort_guard=guard)


def test_two_real_current_receipts_allow_only_process_local_same_cohort(runtime):
    before = _files(runtime)
    _confirm(runtime)
    result = _run(runtime)
    assert result["status"] == "COMMITTED", result
    assert result["same_cohort_eligible"] is True
    assert result["same_cohort"] == NOW.isoformat()
    # Persistent identities still require conservative restart/recovery handling.
    assert result["current_cohort_must_be_skipped"] is True
    marker = revisions.ensure_view_committed(runtime)
    assert "same_cohort_eligible" not in marker
    archive = Path(result["archive_path"])
    assert revisions._inventory(archive / "raw_data") == before
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["fast_confirmation_context"] == _context()
    assert manifest["same_cohort_admission"]["process_local_only"] is True
    assert manifest["same_cohort_admission"]["restart_or_recovery_must_skip"] is True
    assert [g["phase"] for g in manifest["fast_cohort_guards"]] == [
        "BEFORE_VALIDATION", "BEFORE_CUTOVER", "AFTER_CUTOVER_BEFORE_ADMISSION"]
    receipts = manifest["changes"][0]["confirmation_receipts"]
    assert receipts[0]["id"] != receipts[1]["id"]
    assert (pd.Timestamp(receipts[1]["requested_at"])-pd.Timestamp(receipts[0]["received_at"])).total_seconds() == 2
    with sqlite3.connect(archive / "observations.sqlite3") as conn:
        assert conn.execute("SELECT content FROM decisions").fetchone()[0].startswith("old decision")
    changed = pd.read_parquet(runtime / "features/raw_data/intraday/BTCUSDT_5m_futures.parquet")
    assert changed.iloc[-1]["count"] == NEW[8]


def test_age_alone_never_confirms_but_reports_precise_next_request_time(runtime):
    _add_receipt(runtime, NOW+pd.Timedelta(seconds=4))
    before = _files(runtime)
    result = _run(runtime)
    assert result["status"] == "WAITING_CONFIRMATION"
    assert result["fast_confirmation_eligible"] is True
    assert result["pending"][0]["next_confirmation_request_at"] == (NOW+pd.Timedelta(seconds=6)).isoformat()
    assert _files(runtime) == before
    assert not (runtime / "features/revisions").exists()


@pytest.mark.parametrize("case", ["short_request_spacing", "same_request", "different_field",
                                  "ignored_field", "changed_then_changed_back", "wrong_clock",
                                  "system_clock", "prior_cohort"])
def test_insufficient_or_inconsistent_fast_receipts_never_change_view(runtime, case):
    first_at = NOW+pd.Timedelta(seconds=4)
    second_at = NOW+pd.Timedelta(seconds=6.4)
    if case == "prior_cohort":
        first_at = NOW-pd.Timedelta(seconds=1)
    _add_receipt(runtime, first_at)
    second = deepcopy(NEW)
    if case == "short_request_spacing": second_at = NOW+pd.Timedelta(seconds=6.399)
    elif case == "different_field": second[8] += 1
    elif case == "ignored_field": second[11] = "1"
    elif case == "changed_then_changed_back":
        intermediate = deepcopy(NEW)
        intermediate[8] += 1
        _add_receipt(runtime, NOW+pd.Timedelta(seconds=5), intermediate)
    _add_receipt(runtime, second_at, second)
    with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
        if case == "same_request":
            conn.execute("UPDATE receipts SET requested_at=? WHERE id=(SELECT MAX(id) FROM receipts)",
                         ((first_at-pd.Timedelta(seconds=.4)).isoformat(),))
        elif case == "wrong_clock":
            conn.execute("UPDATE receipts SET clock_evidence_sha256=? WHERE id>2", ("e"*64,))
        elif case == "system_clock":
            conn.execute("UPDATE receipts SET time_basis='SYSTEM_UTC',clock_evidence_sha256=NULL WHERE id>2")
    before = _files(runtime)
    result = _run(runtime)
    assert result["status"] == "WAITING_CONFIRMATION", result
    assert _files(runtime) == before
    assert not (runtime / "features/revisions").exists()


def test_exact_aged_boundary_is_allowed_but_last_closed_bar_is_not(runtime):
    cohort = NOW-pd.Timedelta(minutes=10)
    _confirm(runtime, cohort)
    result = _run(runtime, cohort=cohort)
    # Fixture bar closes 01:30; at 01:35 it is precisely five minutes old.
    assert result["status"] == "COMMITTED" and result["same_cohort_eligible"], result


def test_young_revision_falls_back_to_unchanged_slow_confirmation(runtime):
    cohort = NOW-pd.Timedelta(minutes=15)
    _add_receipt(runtime, cohort+pd.Timedelta(seconds=4), include_new_bar=False)
    _add_receipt(runtime, cohort+pd.Timedelta(seconds=6.4), include_new_bar=False)
    before = _files(runtime)
    result = _run(runtime, cohort=cohort)
    assert result["status"] == "WAITING_CONFIRMATION", result
    assert result["fast_confirmation_eligible"] is False
    assert result["pending"][0]["reason"] == "TWO_INDEPENDENT_MATCHING_RECEIPTS_60S_APART_REQUIRED"
    assert _files(runtime) == before


def test_batch_cannot_commit_only_one_confirmed_symbol(runtime):
    _confirm(runtime)
    _add_receipt(runtime, NOW+pd.Timedelta(seconds=4), symbol="ETHUSDT")
    before = _files(runtime)
    result = _run(runtime, errors=[_error(), _error("ETHUSDT")])
    assert result["status"] == "WAITING_CONFIRMATION" and result["fast_confirmation_eligible"]
    assert _files(runtime) == before
    _add_receipt(runtime, NOW+pd.Timedelta(seconds=6.4), symbol="ETHUSDT")
    result = _run(runtime, errors=[_error(), _error("ETHUSDT")])
    assert result["status"] == "COMMITTED" and result["same_cohort_eligible"]
    assert {c["symbol"] for c in result["changes"]} == {"BTCUSDT", "ETHUSDT"}


@pytest.mark.parametrize("case", ["raw_hash", "future_receipt", "wrong_endpoint", "clock_syntax", "extra_error"])
def test_existing_receipt_source_hash_time_validation_stays_fail_closed(runtime, case):
    _add_receipt(runtime, NOW+pd.Timedelta(seconds=4))
    kwargs = {"received": NOW+pd.Timedelta(seconds=6.4)}
    if case == "raw_hash": kwargs["bad_hash"] = True
    elif case == "future_receipt": kwargs["received"] = NOW+pd.Timedelta(seconds=8)
    elif case == "wrong_endpoint": kwargs["endpoint"] = "https://elsewhere.invalid/fapi/v1/klines"
    _add_receipt(runtime, **kwargs)
    if case == "clock_syntax":
        with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
            conn.execute("UPDATE receipts SET clock_evidence_sha256='bad' WHERE id=(SELECT MAX(id) FROM receipts)")
    errors = [_error()]
    if case == "extra_error": errors.append({"source": "funding:BTCUSDT", "error": "MISSING"})
    before = _files(runtime)
    result = _run(runtime, errors=errors)
    assert result["status"] == "BLOCKED", result
    assert _files(runtime) == before


@pytest.mark.parametrize("case", ["profile", "spacing", "deadline", "off_grid", "hash", "past_deadline", "no_guard", "no_calibrated_now"])
def test_fast_context_is_explicit_and_bounded(runtime, case):
    _confirm(runtime)
    context, kwargs = _context(), {}
    if case == "profile": context["profile"] = "argon4-latency-v12"
    elif case == "spacing": context["minimum_spacing_seconds"] = 1.5
    elif case == "deadline": context["deadline"] = (NOW+pd.Timedelta(seconds=15)).isoformat()
    elif case == "off_grid": context["cohort"] = (NOW+pd.Timedelta(seconds=1)).isoformat()
    elif case == "hash": context["clock_evidence_sha256"] = "invalid"
    elif case == "past_deadline": kwargs["now"] = lambda: NOW+pd.Timedelta(seconds=14.001)
    elif case == "no_guard": kwargs["guard"] = None
    elif case == "no_calibrated_now": kwargs["now"] = NOW+pd.Timedelta(seconds=7)
    before = _files(runtime)
    result = _run(runtime, context=context, **kwargs)
    assert result["status"] == "BLOCKED", result
    assert _files(runtime) == before


@pytest.mark.parametrize("phase", [1, 2, 3])
@pytest.mark.parametrize("refusal", ["used", "error", "clock_changed", "policy_changed"])
def test_guards_cannot_admit_used_or_unverified_cohort(runtime, phase, refusal):
    _confirm(runtime)
    before = _files(runtime)
    calls = []
    def guard():
        calls.append(True)
        result = _guard()
        if len(calls) == phase:
            if refusal == "used": result["eligible"] = False
            elif refusal == "error": raise RuntimeError("synthetic guard unavailable")
            elif refusal == "clock_changed": result["clock_evidence_sha256"] = "e"*64
            elif refusal == "policy_changed": result["policy_sha256"] = "e"*64 if phase > 1 else "malformed"
        return result
    result = _run(runtime, guard=guard)
    if phase < 3:
        assert result["status"] == "BLOCKED", result
        assert _files(runtime) == before
        assert revisions.ensure_view_committed(runtime) == {"status": "NO_REVISIONS"}
    else:
        assert result["status"] == "COMMITTED", result
        assert result["same_cohort_eligible"] is False
        assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"


@pytest.mark.parametrize("overrun_phase", ["archive", "final_guard"])
def test_slow_archive_or_guard_finishes_safe_commit_but_never_admits_late(runtime, overrun_phase):
    _confirm(runtime)
    stamps = iter([NOW+pd.Timedelta(seconds=15)] if overrun_phase == "archive" else
                  [NOW+pd.Timedelta(seconds=8), NOW+pd.Timedelta(seconds=14.001)])
    result = _run(runtime, cutover_now=lambda: next(stamps))
    assert result["status"] == "COMMITTED", result
    assert result["same_cohort_eligible"] is False
    assert result["same_cohort_admission_reason"] == "FAST_REVISION_INPUT_BUDGET_EXPIRED"
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"


def test_prepared_fast_revision_recovers_original_two_second_proof_future_only(runtime):
    _confirm(runtime)
    def expired(): raise ValueError("synthetic cutover clock expired")
    result = _run(runtime, cutover_now=expired)
    assert result["status"] == "BLOCKED"
    with pytest.raises(revisions.RevisionSafetyError): revisions.ensure_view_committed(runtime)
    before = revisions._inventory(runtime)
    verified = revisions.recover_prepared_price_revision(runtime)
    assert verified["status"] == "VERIFIED_PREPARED_CUTOVER", verified
    assert revisions._inventory(runtime) == before
    recovered = revisions.recover_prepared_price_revision(
        runtime, apply=True, cutover_now=lambda: NOW+pd.Timedelta(minutes=20))
    assert recovered["status"] == "COMMITTED", recovered
    assert recovered["current_cohort_must_be_skipped"] is True
    assert recovered.get("same_cohort_eligible") is not True
    manifest = json.loads(Path(recovered["manifest_path"]).read_text())
    assert manifest["working_view_for_future_cohorts_only"] is True
    assert "same_cohort_admission" not in manifest


def test_legacy_path_still_requires_sixty_seconds_with_fast_receipts_present(runtime):
    _confirm(runtime)
    result = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW+pd.Timedelta(seconds=7))
    assert result["status"] == "WAITING_CONFIRMATION"
    assert result["fast_confirmation_eligible"] is False
    assert result["pending"][0]["reason"] == "TWO_INDEPENDENT_MATCHING_RECEIPTS_60S_APART_REQUIRED"


def test_manifest_publication_time_is_included_in_final_return_admission(runtime):
    _confirm(runtime)
    stamps = iter([NOW+pd.Timedelta(seconds=8), NOW+pd.Timedelta(seconds=9),
                   NOW+pd.Timedelta(seconds=14.001)])
    result = _run(runtime, cutover_now=lambda: next(stamps))
    assert result["status"] == "COMMITTED" and result["same_cohort_eligible"] is False
    assert result["same_cohort_admission_reason"] == "FAST_REVISION_PUBLICATION_OUTSIDE_INPUT_BUDGET"
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"


def test_repeated_call_cannot_reissue_same_cohort_admission(runtime):
    _confirm(runtime)
    first = _run(runtime)
    assert first["same_cohort_eligible"] is True
    before = revisions._inventory(runtime)
    second = _run(runtime)
    assert second["status"] == "NO_CHANGE"
    assert second.get("same_cohort_eligible") is not True
    assert revisions._inventory(runtime) == before


@pytest.mark.parametrize("case", ["missing", "raw_tampered", "hash_tampered", "nearby_time_only"])
def test_fast_selective_original_receipt_still_requires_exact_identity_and_raw(runtime, case):
    _confirm(runtime)
    with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
        if case in ("missing", "nearby_time_only"):
            conn.execute("DELETE FROM receipts WHERE id=1")
        elif case == "raw_tampered":
            conn.execute("UPDATE receipts SET raw=? WHERE id=1", (b"[]",))
        elif case == "hash_tampered":
            conn.execute("UPDATE receipts SET sha256=? WHERE id=1", ("e"*64,))
    if case == "nearby_time_only":
        from test_argon4_forward_revisions import OLD
        # Same payload hash and the same SQLite millisecond bucket is not the
        # original receipt at exactly 01:30:03.000000.
        _add_receipt(runtime, pd.Timestamp("2026-11-06T01:30:03.000100Z"), OLD, include_new_bar=False)
    before = _files(runtime)
    result = _run(runtime)
    assert result["status"] == "BLOCKED", result
    assert result["reason"] == ("RAW_RECEIPT_HASH_MISMATCH" if case == "raw_tampered" else
                                "CACHE_ORIGINAL_RECEIPT_MISSING")
    assert _files(runtime) == before


def test_sql_scope_never_hides_contradictory_current_receipt_with_other_clock(runtime):
    _add_receipt(runtime, NOW+pd.Timedelta(seconds=4))
    contradicted = deepcopy(NEW)
    contradicted[8] += 1
    _add_receipt(runtime, NOW+pd.Timedelta(seconds=5), contradicted)
    with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
        conn.execute("UPDATE receipts SET clock_evidence_sha256=? WHERE id=(SELECT MAX(id) FROM receipts)", ("e"*64,))
    _add_receipt(runtime, NOW+pd.Timedelta(seconds=6.4))
    before = _files(runtime)
    result = _run(runtime)
    assert result["status"] == "WAITING_CONFIRMATION" and result["fast_confirmation_eligible"]
    assert _files(runtime) == before


def test_sql_scope_includes_response_from_request_before_cohort(runtime):
    _confirm(runtime)
    with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
        conn.execute("UPDATE receipts SET requested_at=? WHERE id=(SELECT MAX(id) FROM receipts)",
                     ((NOW-pd.Timedelta(seconds=1)).isoformat(),))
    before = _files(runtime)
    result = _run(runtime)
    assert result["status"] == "WAITING_CONFIRMATION" and not result["fast_confirmation_eligible"]
    assert _files(runtime) == before


@pytest.mark.parametrize("field", ["requested_at", "received_at"])
def test_sql_scope_cannot_hide_invalid_current_timestamp(runtime, field):
    _confirm(runtime)
    with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
        conn.execute("UPDATE receipts SET "+field+"='invalid' WHERE id=(SELECT MAX(id) FROM receipts)")
    before = _files(runtime)
    result = _run(runtime)
    assert result["status"] == "BLOCKED", result
    assert _files(runtime) == before


def test_sql_scope_uses_utc_semantics_for_current_and_original_receipts(runtime):
    _confirm(runtime)
    with sqlite3.connect(runtime / "features/http_receipts.sqlite3") as conn:
        rows = conn.execute("SELECT id,requested_at,received_at FROM receipts").fetchall()
        for identifier, requested, received in rows:
            conn.execute("UPDATE receipts SET requested_at=?,received_at=? WHERE id=?",
                         (pd.Timestamp(requested).tz_convert("Asia/Tokyo").isoformat(),
                          pd.Timestamp(received).tz_convert("Asia/Tokyo").isoformat(), identifier))
    result = _run(runtime)
    assert result["status"] == "COMMITTED" and result["same_cohort_eligible"], result


def test_fast_scope_does_not_parse_unrelated_history_but_legacy_still_does(runtime):
    _add_receipt(runtime, "2026-11-06T01:40:00Z", bad_hash=True)
    _confirm(runtime)
    legacy = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW+pd.Timedelta(seconds=7))
    assert legacy["status"] == "BLOCKED" and legacy["reason"] == "RAW_RECEIPT_HASH_MISMATCH"
    result = _run(runtime)
    assert result["status"] == "COMMITTED" and result["same_cohort_eligible"], result
    # The unrelated historical row remains present in the full archive.
    with sqlite3.connect(Path(result["archive_path"]) / "receipts.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM receipts WHERE sha256=?", ("f"*64,)).fetchone()[0] == 1
