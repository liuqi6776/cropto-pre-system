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


def engineering_profile(policy):
    """Legacy policies stay legacy; explicitly enabled profiles fail closed."""
    value = policy.get("engineering")
    if value is None:
        return {"enabled": False, "profile": "legacy", "book_telemetry": False,
                "feature_snapshot_cache": False, "auxiliary_prefetch": False}
    if not isinstance(value, dict) or set(value) != set(V11_ENGINEERING):
        raise ValueError("invalid engineering profile fields")
    if value.get("profile") != PROFILE_ID:
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
    return {"enabled": True, **deepcopy(value)}
