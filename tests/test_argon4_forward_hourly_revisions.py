"""Offline futures-1h revisions; never touch production views or the network."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from tools.argon4_forward import revisions
from test_argon4_forward_mark_revisions import (
    runtime, forbid_network, _receipt, _error, _inventory, _digest,
    FUTURES_ENDPOINT, MARK_ENDPOINT, MARK_PATH, FIRST, SECOND, NOW, NEW_MARK,
)

PATH = "auxiliary/ETHUSDT_1h_futures.parquet"
SOURCE = "futures:ETHUSDT:1h"
NEW_HOURLY = deepcopy(NEW_MARK)
NEW_HOURLY[5], NEW_HOURLY[7], NEW_HOURLY[8], NEW_HOURLY[9], NEW_HOURLY[10] = "7", "15578.95", 23, "3", "6676.69"


def receipt(runtime, received=FIRST, **kwargs):
    return _receipt(runtime, NEW_HOURLY, received, endpoint=FUTURES_ENDPOINT, **kwargs)


def recover(runtime, now=NOW, **kwargs):
    return revisions.reconcile_confirmed_price_revisions(runtime, [_error(SOURCE)], now, **kwargs)


def test_confirmed_hourly_trade_revision_updates_only_trade_view_without_mark_registry(runtime):
    first, second = receipt(runtime), receipt(runtime, SECOND, extra_bar=True)
    before = _inventory(runtime)
    root = runtime/"features/raw_data"
    original = pd.read_parquet(root/PATH)
    registry = runtime/"fixture_mark_registry.json"
    registry_bytes = registry.read_bytes()
    account_hash = _digest(runtime/"account.sqlite3")
    # No mark_registry_path is needed for a trade candle correction.
    result = recover(runtime)
    assert result["status"] == "COMMITTED", result
    assert result["current_cohort_must_be_skipped"] is True
    assert result["same_cohort_eligible"] is False
    changed = pd.read_parquet(root/PATH)
    assert changed.shape == original.shape
    assert changed.iloc[-1]["count"] == 23 and changed.iloc[-1]["volume"] == 7
    assert changed.iloc[-1]["taker_buy_quote_volume"] == 6676.69
    assert changed.iloc[-1]["receipt_sha256"] == second["sha256"]
    assert {k:v for k,v in _inventory(runtime).items() if k != PATH} == {k:v for k,v in before.items() if k != PATH}
    assert registry.read_bytes() == registry_bytes
    assert _digest(runtime/"account.sqlite3") == account_hash
    archive = Path(result["archive_path"])
    assert revisions._inventory(archive/"raw_data") == before
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["mark_registry_update"] is None
    assert manifest["changes"][0]["kind"] == "futures"
    assert manifest["changes"][0]["interval"] == "1h"
    assert [r["id"] for r in manifest["changes"][0]["confirmation_receipts"]] == [first["id"], second["id"]]
    assert not (archive/"mark_registry_before.json").exists()
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"


@pytest.mark.parametrize("seconds,expected", [(2, "WAITING_CONFIRMATION"), (59.999, "WAITING_CONFIRMATION"), (60, "COMMITTED")])
def test_hourly_trade_still_requires_full_sixty_second_confirmation(runtime, seconds, expected):
    receipt(runtime)
    receipt(runtime, FIRST+pd.Timedelta(seconds=seconds))
    before = _inventory(runtime)
    result = recover(runtime)
    assert result["status"] == expected, result
    if expected != "COMMITTED":
        assert _inventory(runtime) == before
        assert not (runtime/"features/revisions").exists()


def test_hourly_trade_cannot_use_aged_five_minute_fast_exception(runtime):
    receipt(runtime, NOW+pd.Timedelta(seconds=4))
    receipt(runtime, NOW+pd.Timedelta(seconds=6.4))
    context = {"profile": revisions.FAST_PROFILE, "cohort": NOW.isoformat(),
               "deadline": (NOW+pd.Timedelta(seconds=14)).isoformat(),
               "minimum_spacing_seconds": 2.0, "clock_evidence_sha256": "d"*64}
    result = recover(runtime, now=lambda: NOW+pd.Timedelta(seconds=7),
        cutover_now=lambda: NOW+pd.Timedelta(seconds=8), fast_context=context,
        cohort_guard=lambda: {"eligible": True, "clock_evidence_sha256": "d"*64, "policy_sha256": "e"*64})
    assert result["status"] == "WAITING_CONFIRMATION", result
    assert result["fast_confirmation_eligible"] is False
    assert result["pending"][0]["reason"] == "TWO_INDEPENDENT_MATCHING_RECEIPTS_60S_APART_REQUIRED"


@pytest.mark.parametrize("case", ["bad_raw_hash", "missing_original", "original_raw_tampered", "future_receipt", "wrong_endpoint", "unclosed_bar"])
def test_hourly_trade_receipt_proof_remains_fail_closed(runtime, case):
    receipt(runtime)
    if case == "bad_raw_hash": receipt(runtime, SECOND, bad_hash=True)
    elif case == "future_receipt": receipt(runtime, NOW+pd.Timedelta(seconds=1))
    elif case == "wrong_endpoint":
        _receipt(runtime, NEW_HOURLY, SECOND, endpoint="https://elsewhere.invalid/fapi/v1/klines")
    elif case == "unclosed_bar":
        receipt(runtime, pd.Timestamp("2026-11-06T08:59:59Z"))
    else: receipt(runtime, SECOND)
    if case in ("missing_original", "original_raw_tampered"):
        with sqlite3.connect(runtime/"features/http_receipts.sqlite3") as conn:
            # Fixture id 3 is the original ETH hourly trade receipt, not mark.
            if case == "missing_original": conn.execute("DELETE FROM receipts WHERE id=3")
            else: conn.execute("UPDATE receipts SET raw=? WHERE id=3", (b"[]",))
    before = _inventory(runtime)
    result = recover(runtime)
    assert result["status"] == "BLOCKED", result
    assert _inventory(runtime) == before
    assert not (runtime/"features/revisions").exists()


def test_mixed_hourly_mark_and_trade_batch_waits_for_both_then_registers_only_mark(runtime):
    receipt(runtime)
    receipt(runtime, SECOND)
    _receipt(runtime)
    before = _inventory(runtime)
    registry = runtime/"fixture_mark_registry.json"
    errors = [_error(SOURCE), _error()]
    waiting = revisions.reconcile_confirmed_price_revisions(runtime, errors, NOW, mark_registry_path=registry)
    assert waiting["status"] == "WAITING_CONFIRMATION"
    assert _inventory(runtime) == before
    _receipt(runtime, received=SECOND)
    result = revisions.reconcile_confirmed_price_revisions(runtime, errors, NOW, mark_registry_path=registry)
    assert result["status"] == "COMMITTED", result
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert {c["kind"] for c in manifest["changes"]} == {"mark", "futures"}
    updates = manifest["mark_registry_update"]["updates"]
    assert len(updates) == 1 and updates[0]["relative_path"] == MARK_PATH
    assert json.loads(registry.read_text())["research_references"][0]["candle_type"] == "mark"


def test_interrupted_hourly_trade_revision_recovers_only_for_future_cohorts(runtime):
    receipt(runtime)
    receipt(runtime, SECOND)
    def expired(): raise ValueError("synthetic expired cutover clock")
    result = recover(runtime, cutover_now=expired)
    assert result["status"] == "BLOCKED"
    with pytest.raises(revisions.RevisionSafetyError): revisions.ensure_view_committed(runtime)
    before = revisions._inventory(runtime)
    checked = revisions.recover_prepared_price_revision(runtime)
    assert checked["status"] == "VERIFIED_PREPARED_CUTOVER", checked
    assert revisions._inventory(runtime) == before
    committed = revisions.recover_prepared_price_revision(runtime, apply=True,
        cutover_now=lambda: NOW+pd.Timedelta(minutes=20))
    assert committed["status"] == "COMMITTED", committed
    assert committed["current_cohort_must_be_skipped"] is True
    assert committed.get("same_cohort_eligible") is not True
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"
