from copy import deepcopy
import hashlib
import json

import pytest

from tools.argon4_forward.runtime_profile import engineering_profile, V11_ENGINEERING
from tools.argon4_forward.runtime_profile import V12_ENGINEERING, minimum_decision_delay
from tools.argon4_forward.live_preflight import preparation_status, verify_artifacts


def fixture_policy():
    return {"order": ["ETH_5m"], "live_enabled": False, "engineering": deepcopy(V11_ENGINEERING),
            "max_order_usd": 200, "max_open_risk_usd": 1000, "max_buy_price": .515}


def fixture_config():
    return {"mode": "disabled", "live_enabled": False, "signer_connected": False,
            "order_transport_connected": False, "max_order_usd": 200, "max_open_risk_usd": 1000}


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), 20, 9])
def test_invalid_input_budget_fails_closed(value):
    p = fixture_policy()
    p["engineering"]["input_deadline_after_boundary_seconds"] = value
    with pytest.raises(ValueError):
        engineering_profile(p)


def test_legacy_defaults_and_shared_preparation_never_enable_live():
    assert engineering_profile({})["enabled"] is False
    p, c = fixture_policy(), fixture_config()
    report = preparation_status(p, c)
    assert report["configuration_safe"] and report["risk_limits_match"]
    assert report["ready_for_live"] is False and report["order_adapter_implemented"] is False
    c["live_enabled"] = True
    report = preparation_status(p, c)
    assert not report["configuration_safe"] and not report["live_enabled"]


def test_early_decision_is_explicit_v12_only():
    p = fixture_policy()
    p["min_decision_delay_seconds"] = 2.5
    with pytest.raises(ValueError, match="minimum decision delay"):
        minimum_decision_delay(p)
    p["engineering"] = deepcopy(V12_ENGINEERING)
    assert minimum_decision_delay(p) == 2.5
    assert preparation_status(p, fixture_config())["ready_for_live"] is False


@pytest.mark.parametrize("key,value", [("window_poll_interval_seconds", True),
    ("window_poll_interval_seconds", .01), ("input_ready_retry_interval_seconds", float("nan")),
    ("input_ready_retry_interval_seconds", .1), ("telemetry_sampling_profile", "ignore_uncertainty")])
def test_v12_timing_and_sampling_contract_fails_closed(key, value):
    p = fixture_policy()
    p["engineering"] = deepcopy(V12_ENGINEERING)
    p["engineering"][key] = value
    with pytest.raises(ValueError): engineering_profile(p)


@pytest.mark.parametrize("value", [True, None, float("nan"), 0.79, 10.1])
def test_v12_rejects_unapproved_earliest_delay(value):
    p = fixture_policy()
    p.update(engineering=deepcopy(V12_ENGINEERING), min_decision_delay_seconds=value)
    with pytest.raises(ValueError): minimum_decision_delay(p)


def test_v12_accepts_stage1b_high_speed_bounds():
    p = fixture_policy()
    eng = deepcopy(V12_ENGINEERING)
    eng.update(window_poll_interval_seconds=0.1, input_ready_retry_interval_seconds=0.15,
               minimum_decision_delay_seconds=0.8, prefetch_lead_seconds=9.5, prefetch_budget_seconds=6.5)
    p.update(engineering=eng, min_decision_delay_seconds=0.8)
    parsed = engineering_profile(p)
    assert parsed["window_poll_interval_seconds"] == pytest.approx(0.1)
    assert parsed["input_ready_retry_interval_seconds"] == pytest.approx(0.15)
    assert minimum_decision_delay(p, parsed) == pytest.approx(0.8)


def test_preflight_checks_bytes_without_deserializing_or_connecting(tmp_path):
    p, c = fixture_policy(), fixture_config()
    folder = tmp_path / "research/argon4_local_v4_20261106/deployment_q4"
    folder.mkdir(parents=True)
    blob = b"not-an-executable-model; hash-only fixture"
    (folder / "ETH_5m.joblib").write_bytes(blob)
    manifest = {"stream": "ETH_5m", "model_filename": "ETH_5m.joblib",
                "model_sha256": hashlib.sha256(blob).hexdigest()}
    raw = json.dumps(manifest).encode()
    (folder / "ETH_5m.manifest.json").write_bytes(raw)
    p["deployment_manifest_sha256"] = {"ETH_5m": hashlib.sha256(raw).hexdigest()}
    report = verify_artifacts(tmp_path, p, c)
    assert report["preparation_checks_passed"] and not report["ready_for_live"]
    (folder / "ETH_5m.joblib").write_bytes(blob+b"changed")
    report = verify_artifacts(tmp_path, p, c)
    assert not report["preparation_checks_passed"]
    assert report["artifacts"]["ETH_5m"]["status"] == "BLOCKED"


@pytest.mark.parametrize("mutation", ["age", "spacing", "late_budget", "switch_type", "legacy"])
def test_v13_revision_contract_rejects_unsupported_scope(mutation):
    from tools.argon4_forward.runtime_profile import AGED_REVISION_FASTPATH, revision_fastpath_profile
    p = fixture_policy()
    p["engineering"] = deepcopy(V12_ENGINEERING)
    p["input_revision_policy"] = {"aged_futures_fastpath": deepcopy(AGED_REVISION_FASTPATH)}
    value = p["input_revision_policy"]["aged_futures_fastpath"]
    if mutation == "age": value["minimum_bar_age_seconds"] = 60
    if mutation == "spacing": value["minimum_receipt_spacing_seconds"] = 0
    if mutation == "late_budget": p["engineering"]["input_deadline_after_boundary_seconds"] = 10
    if mutation == "switch_type": value["enabled"] = 1
    if mutation == "legacy": p["engineering"] = deepcopy(V11_ENGINEERING)
    with pytest.raises(ValueError): revision_fastpath_profile(p)


def test_v13_preparation_reports_guarded_revision_scope_without_live_authority():
    from tools.argon4_forward.runtime_profile import AGED_REVISION_FASTPATH, revision_fastpath_profile
    assert revision_fastpath_profile({}) == {"enabled": False}
    p = fixture_policy()
    p.update(engineering=deepcopy(V12_ENGINEERING), min_decision_delay_seconds=4.,
             input_revision_policy={"aged_futures_fastpath": deepcopy(AGED_REVISION_FASTPATH)})
    report = preparation_status(p, fixture_config())
    assert report["aged_revision_fastpath"]["same_cohort_before_prediction_only"]
    assert report["min_decision_delay_seconds"] == 4.
    assert not report["ready_for_live"] and not report["live_enabled"]
