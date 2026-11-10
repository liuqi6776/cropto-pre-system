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

AGED_REVISION_PROFILE = "argon4-aged-revision-v13"
AGED_REVISION_FASTPATH = {
    "profile": AGED_REVISION_PROFILE,
    "enabled": True,
    "minimum_bar_age_seconds": 300,
    "minimum_receipt_spacing_seconds": 2.0,
    "same_cohort_before_prediction_only": True,
}


def revision_fastpath_profile(policy):
    """An explicit, narrow exception to the legacy future-window-only repair."""
    revision = policy.get("input_revision_policy", {})
    if not isinstance(revision, dict):
        raise ValueError("invalid input revision policy")
    value = revision.get("aged_futures_fastpath")
    if value is None:
        return {"enabled": False}
    if not isinstance(value, dict) or set(value) != set(AGED_REVISION_FASTPATH):
        raise ValueError("invalid aged revision profile")
    for key, expected in AGED_REVISION_FASTPATH.items():
        if isinstance(expected, bool):
            valid = type(value[key]) is bool and value[key] is expected
        else:
            valid = not isinstance(value[key], bool) and value[key] == expected
        if not valid:
            raise ValueError("unsupported aged revision profile")
    engineering = engineering_profile(policy)
    if (engineering["profile"] != V12_PROFILE_ID
            or engineering["input_deadline_after_boundary_seconds"] not in (14, 16.5)):
        raise ValueError("aged revision requires bounded opportunity profile")
    return deepcopy(value)


def engineering_profile(policy):
    """Legacy policies stay legacy; explicitly enabled profiles fail closed."""
    value = policy.get("engineering")
    if value is None:
        return {"enabled": False, "profile": "legacy", "book_telemetry": False,
                "feature_snapshot_cache": False, "auxiliary_prefetch": False}
    if not isinstance(value, dict):
        raise ValueError("invalid engineering profile fields")
    expected = V12_ENGINEERING if value.get("profile") == V12_PROFILE_ID else V11_ENGINEERING
    optional_keys = {"enabled", "minimum_decision_delay_seconds"} if value.get("profile") == V12_PROFILE_ID else set()
    if not (set(expected) <= set(value) <= (set(expected) | optional_keys)):
        raise ValueError("invalid engineering profile fields")
    if "enabled" in value and (not isinstance(value["enabled"], bool) or value["enabled"] is not True):
        raise ValueError("engineering switches must be boolean")
    if value.get("profile") not in (PROFILE_ID, V12_PROFILE_ID):
        raise ValueError("unsupported engineering profile")
    for key in ("feature_snapshot_cache", "auxiliary_prefetch", "book_telemetry"):
        if not isinstance(value[key], bool):
            raise ValueError("engineering switches must be boolean")
    for key, low, high in (("input_deadline_after_boundary_seconds", 10, 16.5),
                          ("prefetch_lead_seconds", 8, 25),
                          ("prefetch_budget_seconds", .5, 8)):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError("unsafe engineering budget")
    if value["profile"] == V12_PROFILE_ID:
        for key, allowed_numbers in (("window_poll_interval_seconds", (.2, .25)),
                                     ("input_ready_retry_interval_seconds", (.35, .5))):
            if isinstance(value[key], bool) or value[key] not in allowed_numbers:
                raise ValueError("unsafe opportunity timing")
        if value["telemetry_sampling_profile"] != "clock_center_v12":
            raise ValueError("unsupported telemetry sampling profile")
    return {"enabled": True, **deepcopy(value)}


def minimum_decision_delay(policy, engineering=None):
    profile = engineering or engineering_profile(policy)
    eng_block = policy.get("engineering") if isinstance(policy.get("engineering"), dict) else {}
    value = policy.get("min_decision_delay_seconds", eng_block.get("minimum_decision_delay_seconds", 5))
    lower = 2 if profile["profile"] == V12_PROFILE_ID else 5
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or not lower <= value <= 10):
        raise ValueError("unsupported minimum decision delay")
    return float(value)
