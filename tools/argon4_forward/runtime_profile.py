"""Shared, read-only engineering contract; never grants live-order authority."""
from copy import deepcopy
import math

PROFILE_ID = "argon4-latency-v11"
V11_ENGINEERING = {
    "profile": PROFILE_ID,
    "input_deadline_after_boundary_seconds": 14.0,
    "feature_snapshot_cache": True,
    "auxiliary_prefetch": True,
    "prefetch_lead_seconds": 10.0,
    "prefetch_budget_seconds": 3.0,
    "book_telemetry": True,
}
V12_PROFILE_ID = "argon4-latency-v12"
V12_ENGINEERING = {
    **V11_ENGINEERING,
    "profile": V12_PROFILE_ID,
    "window_poll_interval_seconds": .2,
    "input_ready_retry_interval_seconds": .5,
    "telemetry_sampling_profile": "clock_center_v12",
}


def engineering_profile(policy):
    """Legacy policies stay legacy; explicitly enabled profiles fail closed."""
    value = policy.get("engineering")
    if value is None:
        return {"enabled": False, "profile": "legacy", "book_telemetry": False,
                "feature_snapshot_cache": False, "auxiliary_prefetch": False}
    if not isinstance(value, dict):
        raise ValueError("invalid engineering profile fields")
    expected = V12_ENGINEERING if value.get("profile") == V12_PROFILE_ID else V11_ENGINEERING
    if set(value) != set(expected):
        raise ValueError("invalid engineering profile fields")
    if value.get("profile") not in (PROFILE_ID, V12_PROFILE_ID):
        raise ValueError("unsupported engineering profile")
    for key in ("feature_snapshot_cache", "auxiliary_prefetch", "book_telemetry"):
        if not isinstance(value[key], bool):
            raise ValueError("engineering switches must be boolean")
    for key, low, high in (("input_deadline_after_boundary_seconds", 10, 14),
                          ("prefetch_lead_seconds", 8, 10),
                          ("prefetch_budget_seconds", .5, 3)):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError("unsafe engineering budget")
    if value["profile"] == V12_PROFILE_ID:
        for key, expected_number in (("window_poll_interval_seconds", .2),
                                     ("input_ready_retry_interval_seconds", .5)):
            if isinstance(value[key], bool) or value[key] != expected_number:
                raise ValueError("unsafe opportunity timing")
        if value["telemetry_sampling_profile"] != "clock_center_v12":
            raise ValueError("unsupported telemetry sampling profile")
    return {"enabled": True, **deepcopy(value)}


def minimum_decision_delay(policy, engineering=None):
    profile = engineering or engineering_profile(policy)
    value = policy.get("min_decision_delay_seconds", 5)
    lower = 2 if profile["profile"] == V12_PROFILE_ID else 5
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or not lower <= value <= 10):
        raise ValueError("unsupported minimum decision delay")
    return float(value)
