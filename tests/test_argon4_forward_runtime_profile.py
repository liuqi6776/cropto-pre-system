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


@pytest.mark.parametrize("value", [True, None, float("nan"), 1.9, 10.1])
def test_v12_rejects_unapproved_earliest_delay(value):
    p = fixture_policy()
    p.update(engineering=deepcopy(V12_ENGINEERING), min_decision_delay_seconds=value)
    with pytest.raises(ValueError): minimum_decision_delay(p)


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
