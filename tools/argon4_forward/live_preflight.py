"""Disabled live-preparation contract. GET data and local checks only.

There is intentionally no signer, exchange-order client, or activation function.
This report never turns technical checks into trading or regional eligibility.
"""
import hashlib
import json
from pathlib import Path

from .runtime_profile import engineering_profile, minimum_decision_delay, revision_fastpath_profile
from .stage import stage_profile
from .account import AccountRejected, _policy_risk_limits, multi_look_settings


def preparation_status(policy, config):
    engineering = engineering_profile(policy)
    stage = stage_profile(policy)
    earliest = minimum_decision_delay(policy, engineering)
    multi_look = multi_look_settings(policy)
    revision_profile = revision_fastpath_profile(policy)
    try:
        risk_limits = _policy_risk_limits(policy)
    except AccountRejected as exc:
        raise ValueError(str(exc)) from exc
    disabled = (policy.get("live_enabled") is False and
                config.get("live_enabled") is False and
                config.get("mode") == "disabled" and
                config.get("signer_connected") is False and
                config.get("order_transport_connected") is False)
    risk_block = policy.get("risk") if isinstance(policy.get("risk"), dict) else {}
    expected_order = float(risk_limits["max_order_usd"]) if ("max_order_usd" in policy or "max_order_usd" in risk_block or "max_order_usd" in config) else None
    expected_open = float(risk_limits["max_open_risk_usd"]) if ("max_open_risk_usd" in policy or "max_open_risk_usd" in risk_block or "max_open_risk_usd" in config) else None
    limits_match = (config.get("max_order_usd") == expected_order and
                    config.get("max_open_risk_usd") == expected_open)
    return {"status": "DISABLED", "live_enabled": False,
            "configuration_safe": disabled, "risk_limits_match": limits_match,
            "risk_limits": {k: (bool(v) if isinstance(v, bool) else ({sk: float(sv) for sk, sv in v.items()} if isinstance(v, dict) else (int(v) if v == int(v) else float(v))))
                            for k, v in risk_limits.items() if not k.endswith("_atoms")},
            "engineering": engineering, "policy_version": policy.get("version"),
            "min_decision_delay_seconds": earliest, "multi_look_ioc": multi_look,
            "aged_revision_fastpath": revision_profile,
            "price_limit_basis": "ASK_PRICE_EXCLUDING_FEE_FEES_INCLUDED_IN_CAPITAL_LIMITS",
            "execution_stage": stage, "execution_style": "TAKER_IOC_PAPER_ONLY",
            "max_buy_price": float(risk_limits["max_buy_price"]) if risk_limits.get("has_explicit_max_buy_price") else None,
            "signer_connected": False, "order_transport_connected": False,
            "order_adapter_implemented": False, "ready_for_live": False,
            "blockers": ["WORKSPACE_LIVE_TRADING_PROHIBITED", "NO_LIVE_ORDER_ADAPTER",
                         "NO_SIGNER", "LIVE_ELIGIBILITY_NOT_CERTIFIED"]}


def verify_artifacts(root, policy, config):
    """Hash local artifacts without deserializing executable model packages."""
    report = preparation_status(policy, config)
    folder = Path(root) / "research/argon4_local_v4_20261106/deployment_q4"
    artifacts = {}
    for stream in report["execution_stage"]["enabled_streams"]:
        try:
            manifest_path = folder / (stream+".manifest.json")
            raw = manifest_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != policy.get("deployment_manifest_sha256", {}).get(stream):
                raise ValueError("manifest pin mismatch")
            manifest = json.loads(raw)
            if manifest.get("stream") != stream or manifest.get("model_filename") != stream+".joblib":
                raise ValueError("model identity mismatch")
            model_path = folder / manifest["model_filename"]
            if manifest_path.is_symlink() or model_path.is_symlink():
                raise ValueError("symlink artifacts are not allowed")
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if model_sha != manifest.get("model_sha256"):
                raise ValueError("model digest mismatch")
            artifacts[stream] = {"status": "HASH_VERIFIED", "manifest_sha256": digest,
                                 "model_sha256": model_sha}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            artifacts[stream] = {"status": "BLOCKED", "error": type(exc).__name__}
    report["artifacts"] = artifacts
    report["preparation_checks_passed"] = (report["configuration_safe"] and report["risk_limits_match"]
        and bool(artifacts) and all(v["status"] == "HASH_VERIFIED" for v in artifacts.values()))
    return report
