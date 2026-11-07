"""Legacy receipt rules with a separate cancellable maintenance publication guard."""
import json
from pathlib import Path

import pandas as pd
import pytest

from test_argon4_forward_revisions import runtime, no_network, _add_receipt, _error, _files, NOW
from tools.argon4_forward import revisions


def confirmed(runtime):
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")


def allow(phase):
    return {"eligible": True, "phase": phase, "execution_authorized": False}


def run(runtime, guard=allow, clock=lambda: NOW, **kwargs):
    return revisions.reconcile_confirmed_price_revisions(runtime, [_error()],
        now=lambda: NOW, cutover_now=clock, maintenance_guard=guard, **kwargs)


@pytest.mark.parametrize("phase", ["BEFORE_VALIDATION", "BEFORE_ARCHIVE", "BEFORE_CUTOVER"])
@pytest.mark.parametrize("refusal", ["raise", "false"])
def test_precutover_refusal_keeps_raw_and_prior_marker(runtime, phase, refusal):
    confirmed(runtime)
    before = _files(runtime)
    def guard(current):
        if current == phase:
            if refusal == "raise": raise ValueError("fixture cancellation")
            return {"eligible": False, "execution_authorized": False}
        return allow(current)
    result = run(runtime, guard)
    assert result["status"] == "BLOCKED"
    assert _files(runtime) == before
    assert revisions.ensure_view_committed(runtime) == {"status": "NO_REVISIONS"}
    assert not (runtime/"features/revisions/view_state.json").exists()
    if phase == "BEFORE_CUTOVER": assert (Path(result["archive_path"])/"prepared.json").exists()


def test_maintenance_commit_is_future_only_and_records_all_guards(runtime):
    confirmed(runtime)
    result = run(runtime)
    assert result["status"] == "COMMITTED", result
    assert not result["same_cohort_eligible"]
    assert result["current_cohort_must_be_skipped"] and result["next_cohort_requires_fresh_validation"]
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert [g["phase"] for g in manifest["maintenance_guards"]] == [
        "BEFORE_VALIDATION", "BEFORE_ARCHIVE", "BEFORE_CUTOVER", "AFTER_CUTOVER"]
    assert "fast_confirmation_context" not in manifest
    assert manifest["maintenance_policy"] == "WINDOW_EXTERNAL_LEGACY_60S_FUTURE_COHORTS_ONLY"


@pytest.mark.parametrize("reason", ["STOPPED", "POLICY_CHANGED", "PHASE_EXPIRED", "DEADLINE_EXPIRED"])
def test_postcutover_permission_loss_commits_real_view_without_admission(runtime, reason):
    confirmed(runtime)
    before = _files(runtime)
    def guard(phase):
        if phase == "AFTER_CUTOVER": raise ValueError(reason)
        return allow(phase)
    result = run(runtime, guard)
    assert result["status"] == "COMMITTED" and _files(runtime) != before
    assert result["maintenance_postcutover"]["reason"] == reason
    assert not result["same_cohort_eligible"]
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"


@pytest.mark.parametrize("failure", ["expiry", "jump", "reverse"])
def test_postcutover_unverifiable_time_leaves_necessary_prepared(runtime, failure):
    confirmed(runtime)
    before = _files(runtime)
    changed = False
    def guard(phase):
        nonlocal changed
        if phase == "AFTER_CUTOVER": changed = True
        return allow(phase)
    def clock():
        if changed:
            if failure == "reverse": return NOW-pd.Timedelta(seconds=1)
            raise ValueError("CLOCK_"+failure.upper())
        return NOW
    result = run(runtime, guard, clock)
    assert result["status"] == "BLOCKED" and _files(runtime) != before
    with pytest.raises(revisions.RevisionSafetyError, match="REVISION_VIEW_NOT_COMMITTED"):
        revisions.ensure_view_committed(runtime)
    assert not (Path(result["archive_path"])/"manifest.json").exists()


@pytest.mark.parametrize("spacing", [0, 2, 59])
def test_maintenance_does_not_weaken_sixty_second_receipt_rule(runtime, spacing):
    first = pd.Timestamp("2026-11-06T01:35:04Z")
    _add_receipt(runtime, first)
    _add_receipt(runtime, first+pd.Timedelta(seconds=spacing))
    before = _files(runtime)
    assert run(runtime)["status"] == "WAITING_CONFIRMATION"
    assert _files(runtime) == before


def test_mixed_and_unconfirmed_sources_still_block_entire_batch(runtime):
    confirmed(runtime)
    _add_receipt(runtime, symbol="ETHUSDT")
    before = _files(runtime)
    def reconcile(errors):
        return revisions.reconcile_confirmed_price_revisions(runtime, errors,
            now=lambda: NOW, cutover_now=lambda: NOW, maintenance_guard=allow)
    assert reconcile([_error(), _error("ETHUSDT")])["status"] == "WAITING_CONFIRMATION"
    assert reconcile([_error(), {"source": "mark:ETHUSDT:1h", "error": "HTTP_TIMEOUT"}])["status"] == "BLOCKED"
    assert _files(runtime) == before


def test_fast_context_cannot_be_combined_with_maintenance(runtime):
    confirmed(runtime)
    result = run(runtime, fast_context={})
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "MAINTENANCE_REQUIRES_EXCLUSIVE_CALIBRATED_LEGACY_PATH"
