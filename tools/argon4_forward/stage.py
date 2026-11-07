"""Pure stage allowlist for future local-paper decisions; no trading capability.

The complete seven-model ranking remains intact. An optional confirmed stage
selects its active subset, in that ranking's order, beginning at a future UTC
cohort boundary. Absence preserves the historical seven-stream semantics.
"""
from collections.abc import Mapping
from datetime import datetime, timezone

STREAMS = frozenset(("BTC_15m", "ETH_5m", "ETH_15m", "SOL_15m", "SOL_5m", "BTC_5m", "BNB_15m"))
SEMANTICS = "ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1"
DISABLED = "DISABLED_BY_STAGE"


class StageRejected(ValueError):
    pass


def _utc(value):
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError
        return stamp.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StageRejected("stage requires an aware UTC timestamp") from exc


def stage_profile(policy, cohort=None):
    """Validate and return a detached display/execution profile.

    policy.execution_stage schema: {id: nonempty str, status: CONFIRMED,
    enabled_streams: nonempty ordered subset of policy.order,
    effective_from: aware ISO timestamp at a 5-minute boundary}.
    With ``cohort``, historical/pre-activation use raises, never falls back to
    the old seven-stream scope. Callers retain old cohorts' immutable policies.
    """
    if not isinstance(policy, Mapping):
        raise StageRejected("stage policy must be an object")
    order = policy.get("order", [])
    spec = policy.get("execution_stage")
    if "execution_stage" not in policy:
        return {"configured": False, "id": None, "status": "ALL_STREAMS",
                "enabled_streams": list(order), "disabled_streams": [],
                "effective_from": None, "live_enabled": False}
    if (not isinstance(order, list) or len(order) != 7 or any(not isinstance(s, str) for s in order)
            or set(order) != STREAMS or policy.get("status") != "CONFIRMED"
            or policy.get("waterfall_semantics") != SEMANTICS):
        raise StageRejected("stage requires a confirmed complete eligible-budget ranking")
    if (not isinstance(spec, Mapping) or set(spec) != {"id", "status", "enabled_streams", "effective_from"}
            or spec.get("status") != "CONFIRMED" or not isinstance(spec.get("id"), str)
            or not spec["id"].strip()):
        raise StageRejected("stage requires a complete confirmed specification")
    enabled = spec["enabled_streams"]
    if (not isinstance(enabled, list) or not enabled or any(not isinstance(s, str) for s in enabled)
            or len(set(enabled)) != len(enabled) or not set(enabled) <= STREAMS
            or enabled != [s for s in order if s in enabled]):
        raise StageRejected("stage allowlist must be unique and follow the confirmed ranking")
    effective = _utc(spec["effective_from"])
    if effective.timestamp() % 300:
        raise StageRejected("stage activation must be a five-minute UTC boundary")
    if cohort is not None:
        boundary = _utc(cohort)
        if boundary.timestamp() % 300 or boundary < effective:
            raise StageRejected("stage cannot execute a pre-activation or off-grid cohort")
    return {"configured": True, "id": spec["id"], "status": "CONFIRMED",
            "enabled_streams": list(enabled), "disabled_streams": [s for s in order if s not in enabled],
            "effective_from": effective.isoformat(), "live_enabled": False}


def disabled_decision(profile):
    return {"status": DISABLED, "stage_id": profile["id"], "approved_usd": 0.0,
            "spent_usd": 0.0, "book_checked": False, "model_evaluated": False}
