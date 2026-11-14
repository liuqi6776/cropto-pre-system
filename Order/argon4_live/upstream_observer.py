"""Non-blocking read-only observer for upstream `crypto-predict_system`.

Implements Section 2 (2.1, 2.2, 2.3) & Section 3 of the ARGON4 External Live Executor Spec:
1. Primary Trigger: Polls `pipeline_state.json` every 50ms via `os.stat().st_mtime_ns`
   with zero SQLite lock impact.
2. Secondary Interface: Queries `http://127.0.0.1:8897/api/status` and `/api/ledger`
   with mandatory `Host: 127.0.0.1:8897` and no `Origin` header, strictly outside
   upstream's `T+0s..T+20s` critical SQLite transaction window.
3. Downtime Catch-Up Fallback (Design Gap #3): If a live position's settlement rolled
   past the 100-event window of `/api/ledger` while the executor was offline, resolves
   the market via `/api/status` and Polymarket CLOB `/markets/{condition_id}` / Gamma API.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping
import requests

from .math_utils import parse_utc_timestamp
from .safety import assert_safe_read_path


REQUIRED_POLICY_FIELDS: tuple[str, ...] = (
    "initial_capital_usd",
    "max_tick_budget_usd",
    "max_order_usd",
    "max_active_window_risk_usd",
    "max_open_risk_usd",
    "max_underlying_active_risk_usd",
    "max_underlying_open_risk_usd",
    "max_buy_price",
    "min_buy_price",
    "daily_loss_fraction",
)


class PolicyVerificationError(RuntimeError):
    """Fail-closed error raised whenever active trading policy cannot be verified."""

    def __init__(self, reason_code: str, message: str | None = None) -> None:
        self.reason_code = str(reason_code or "POLICY_UNVERIFIED")
        super().__init__(message or self.reason_code)


def _as_finite_float(val: Any, field_name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise PolicyVerificationError("POLICY_INVALID_VALUE", f"Policy field {field_name} must be numeric, got {val!r}")
    num = float(val)
    if not math.isfinite(num):
        raise PolicyVerificationError("POLICY_NON_FINITE_VALUE", f"Policy field {field_name} must be finite, got {val!r}")
    return num


def validate_policy_dict(payload: Any) -> dict[str, Any]:
    """Strictly validate all required risk caps and price bounds in `argon4_forward_policy.json`.

    Nested `payload["risk"]` overrides top-level `payload` fields using the exact same
    precedence and active-risk inheritance rules as `tools.argon4_forward.account._policy_risk_limits`.
    """
    if not isinstance(payload, Mapping):
        raise PolicyVerificationError("POLICY_INVALID_TYPE", "Policy JSON root must be an object")
    raw_risk = payload.get("risk")
    if "risk" in payload and raw_risk is not None and not isinstance(raw_risk, Mapping):
        raise PolicyVerificationError("POLICY_INVALID_VALUE", "Policy field risk must be a mapping")
    risk_block: Mapping[str, Any] = raw_risk if isinstance(raw_risk, Mapping) else {}

    def _raw_field(key: str) -> Any:
        if key in risk_block:
            return risk_block.get(key)
        return payload.get(key)

    resolved_raw: dict[str, Any] = {}
    for k in REQUIRED_POLICY_FIELDS:
        if k == "max_active_window_risk_usd":
            if "max_active_window_risk_usd" in risk_block:
                resolved_raw[k] = risk_block.get("max_active_window_risk_usd")
            elif "max_open_risk_usd" in risk_block:
                resolved_raw[k] = risk_block.get("max_open_risk_usd")
            else:
                resolved_raw[k] = payload.get("max_active_window_risk_usd")
        elif k == "max_underlying_active_risk_usd":
            if "max_underlying_active_risk_usd" in risk_block:
                resolved_raw[k] = risk_block.get("max_underlying_active_risk_usd")
            elif "max_underlying_open_risk_usd" in risk_block or "max_open_risk_usd" in risk_block:
                # Derived after underlying_open and active_window are validated as finite floats below
                resolved_raw[k] = "__DERIVE_MIN_UNDERLYING_ACTIVE__"
            else:
                resolved_raw[k] = payload.get("max_underlying_active_risk_usd")
        else:
            resolved_raw[k] = _raw_field(k)

    missing = [k for k in REQUIRED_POLICY_FIELDS if resolved_raw.get(k) is None]
    if missing:
        raise PolicyVerificationError(
            "POLICY_MISSING_FIELDS",
            f"Missing required policy fields: {', '.join(missing)}",
        )
    out = dict(payload)
    nums: dict[str, float] = {}
    for k in REQUIRED_POLICY_FIELDS:
        if resolved_raw[k] == "__DERIVE_MIN_UNDERLYING_ACTIVE__":
            continue
        nums[k] = _as_finite_float(resolved_raw[k], k)
    if resolved_raw.get("max_underlying_active_risk_usd") == "__DERIVE_MIN_UNDERLYING_ACTIVE__":
        nums["max_underlying_active_risk_usd"] = min(
            nums["max_underlying_open_risk_usd"],
            nums["max_active_window_risk_usd"],
        )

    for pos_key in (
        "initial_capital_usd",
        "max_tick_budget_usd",
        "max_order_usd",
        "max_active_window_risk_usd",
        "max_open_risk_usd",
        "max_underlying_active_risk_usd",
        "max_underlying_open_risk_usd",
    ):
        if nums[pos_key] <= 0.0:
            raise PolicyVerificationError("POLICY_INVALID_BOUND", f"Policy field {pos_key} must be > 0, got {nums[pos_key]}")

    if nums["max_order_usd"] > nums["max_tick_budget_usd"]:
        raise PolicyVerificationError(
            "POLICY_INVALID_BOUND",
            f"max_order_usd ({nums['max_order_usd']}) exceeds max_tick_budget_usd ({nums['max_tick_budget_usd']})",
        )
    if nums["max_underlying_open_risk_usd"] > nums["max_open_risk_usd"]:
        raise PolicyVerificationError(
            "POLICY_INVALID_BOUND",
            f"max_underlying_open_risk_usd ({nums['max_underlying_open_risk_usd']}) exceeds max_open_risk_usd ({nums['max_open_risk_usd']})",
        )
    if nums["max_active_window_risk_usd"] > nums["max_open_risk_usd"]:
        raise PolicyVerificationError(
            "POLICY_INVALID_BOUND",
            f"max_active_window_risk_usd ({nums['max_active_window_risk_usd']}) exceeds max_open_risk_usd ({nums['max_open_risk_usd']})",
        )
    if nums["max_underlying_active_risk_usd"] > nums["max_underlying_open_risk_usd"]:
        raise PolicyVerificationError(
            "POLICY_INVALID_BOUND",
            f"max_underlying_active_risk_usd ({nums['max_underlying_active_risk_usd']}) exceeds max_underlying_open_risk_usd ({nums['max_underlying_open_risk_usd']})",
        )
    if nums["max_underlying_active_risk_usd"] > nums["max_active_window_risk_usd"]:
        raise PolicyVerificationError(
            "POLICY_INVALID_BOUND",
            f"max_underlying_active_risk_usd ({nums['max_underlying_active_risk_usd']}) exceeds max_active_window_risk_usd ({nums['max_active_window_risk_usd']})",
        )
    if not (
        0.0 < nums["min_buy_price"] <= 0.500
        and 0.450 <= nums["max_buy_price"] <= 0.530
        and nums["min_buy_price"] <= nums["max_buy_price"]
    ):
        raise PolicyVerificationError(
            "POLICY_INVALID_BOUND",
            f"Invalid buy price bounds: min={nums['min_buy_price']}, max={nums['max_buy_price']}",
        )
    if not (0.0 < nums["daily_loss_fraction"] <= 1.0):
        raise PolicyVerificationError(
            "POLICY_INVALID_BOUND",
            f"daily_loss_fraction must be in (0, 1], got {nums['daily_loss_fraction']}",
        )

    for k, v in nums.items():
        out[k] = v

    for map_key in ("stream_max_order_usd", "stream_min_buy_price"):
        has_in_risk = map_key in risk_block
        sub = risk_block.get(map_key) if has_in_risk else payload.get(map_key)
        if sub is not None:
            if not isinstance(sub, Mapping):
                raise PolicyVerificationError("POLICY_INVALID_VALUE", f"Policy field {map_key} must be a mapping")
            clean_sub: dict[str, float] = {}
            for s_name, s_val in sub.items():
                f_val = _as_finite_float(s_val, f"{map_key}.{s_name}")
                if map_key == "stream_max_order_usd" and f_val <= 0.0:
                    raise PolicyVerificationError("POLICY_INVALID_BOUND", f"{map_key}.{s_name} must be > 0")
                if map_key == "stream_min_buy_price" and not (0.0 < f_val <= 0.500 and f_val <= nums["max_buy_price"]):
                    raise PolicyVerificationError("POLICY_INVALID_BOUND", f"{map_key}.{s_name} out of bounds")
                clean_sub[str(s_name)] = f_val
            out[map_key] = clean_sub
        elif has_in_risk:
            out[map_key] = {}
    return out


def _normalize_outcome_key(raw_label: Any) -> str:
    txt = str(raw_label or "").upper().strip()
    if txt in ("UP", "YES", "0"):
        return "UP"
    if txt in ("DOWN", "NO", "1"):
        return "DOWN"
    return txt


def _validate_complete_binary_payout_map(raw: Mapping[str, float]) -> dict[str, float] | None:
    """Require complete binary `{UP, DOWN}` keys with finite values in `[0.0, 1.0]` summing to `1.0`."""
    if not isinstance(raw, Mapping) or set(raw.keys()) != {"UP", "DOWN"}:
        return None
    try:
        up_val = float(raw["UP"])
        down_val = float(raw["DOWN"])
    except (ValueError, TypeError):
        return None
    if not (math.isfinite(up_val) and math.isfinite(down_val)):
        return None
    if not (0.0 <= up_val <= 1.0 and 0.0 <= down_val <= 1.0):
        return None
    if abs((up_val + down_val) - 1.0) > 1e-5:
        return None
    return {"UP": round(up_val, 8), "DOWN": round(down_val, 8)}


def extract_payout_ratios_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    allow_outcome_prices: bool = False,
) -> dict[str, float] | None:
    """Extract verified complete binary settlement payout ratios (`UP` and `DOWN` summing to `1.0`).

    By default (`allow_outcome_prices=False`), market trading `outcomePrices` are NEVER treated as
    authoritative settlement payout ratios so disputed or unfinalized trading prices cannot leak into
    settlement accounting.
    """
    if not isinstance(payload, Mapping):
        return None

    direct = payload.get("payout_ratios") or payload.get("payoutRatios")
    if isinstance(direct, Mapping) and direct:
        cleaned: dict[str, float] = {}
        valid_direct = True
        for k, v in direct.items():
            if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                valid_direct = False
                break
            try:
                num = float(v)
            except (ValueError, TypeError):
                valid_direct = False
                break
            norm_k = _normalize_outcome_key(k)
            if norm_k not in ("UP", "DOWN") or norm_k in cleaned:
                valid_direct = False
                break
            cleaned[norm_k] = num
        if valid_direct:
            validated = _validate_complete_binary_payout_map(cleaned)
            if validated is not None:
                return validated

    raw_outcomes = payload.get("outcomes")
    outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
    if isinstance(outcomes, list) and len(outcomes) >= 2:
        raw_nums = (
            payload.get("payoutNumerators")
            if payload.get("payoutNumerators") is not None
            else payload.get("payout_numerators")
        )
        if raw_nums is not None:
            nums = json.loads(raw_nums) if isinstance(raw_nums, str) else raw_nums
            denom_raw = payload.get("payoutDenominator") or payload.get("payout_denominator")
            if isinstance(nums, list) and len(nums) == len(outcomes):
                try:
                    floats = [float(x) for x in nums]
                except (ValueError, TypeError):
                    floats = []
                if floats and all(math.isfinite(x) and x >= 0.0 for x in floats):
                    if denom_raw is not None:
                        try:
                            denom = float(denom_raw)
                        except (ValueError, TypeError):
                            denom = 0.0
                    else:
                        total = sum(floats)
                        denom = total if total > 1.0 else 1.0
                    if denom > 0.0:
                        ratios: dict[str, float] = {}
                        valid_num_map = True
                        for lbl, val in zip(outcomes, floats):
                            norm_k = _normalize_outcome_key(lbl)
                            if norm_k not in ("UP", "DOWN") or norm_k in ratios:
                                valid_num_map = False
                                break
                            ratios[norm_k] = val / denom
                        if valid_num_map:
                            validated = _validate_complete_binary_payout_map(ratios)
                            if validated is not None:
                                return validated

    # Inspect CLOB `tokens` list when explicit token payout fields exist
    tokens = payload.get("tokens")
    if isinstance(tokens, list) and len(tokens) >= 2:
        tok_ratios: dict[str, float] = {}
        valid_tokens = True
        for tok in tokens:
            if not isinstance(tok, Mapping):
                valid_tokens = False
                break
            key = _normalize_outcome_key(tok.get("outcome"))
            pr = tok.get("payout_ratio")
            if pr is None:
                pr = tok.get("payout")
            if pr is None or isinstance(pr, bool):
                valid_tokens = False
                break
            try:
                num = float(pr)
            except (ValueError, TypeError):
                valid_tokens = False
                break
            if key not in ("UP", "DOWN") or key in tok_ratios:
                valid_tokens = False
                break
            tok_ratios[key] = num
        if valid_tokens:
            validated = _validate_complete_binary_payout_map(tok_ratios)
            if validated is not None:
                return validated

    if allow_outcome_prices and isinstance(outcomes, list) and len(outcomes) >= 2:
        raw_prices = payload.get("outcomePrices")
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        if isinstance(prices, list) and len(prices) == len(outcomes):
            parsed: dict[str, float] = {}
            valid_prices = True
            for out_label, pr_val in zip(outcomes, prices):
                if isinstance(pr_val, bool):
                    valid_prices = False
                    break
                try:
                    num = float(pr_val)
                except (ValueError, TypeError):
                    valid_prices = False
                    break
                norm_k = _normalize_outcome_key(out_label)
                if norm_k not in ("UP", "DOWN") or norm_k in parsed:
                    valid_prices = False
                    break
                parsed[norm_k] = num
            if valid_prices:
                validated = _validate_complete_binary_payout_map(parsed)
                if validated is not None:
                    return validated

    return None


NON_FINAL_RESOLUTION_STATUSES = frozenset(
    {
        "disputed",
        "proposed",
        "requested",
        "pending",
        "open",
        "in_dispute",
        "active",
        "inactive",
        "unresolved",
        "unsettled",
        "flagged",
        "paused",
        "under_review",
        "challenged",
    }
)
FINAL_RESOLUTION_STATUSES = frozenset({"resolved", "settled", "finalized"})


def _is_market_officially_finalized(payload: Mapping[str, Any] | None) -> bool:
    """Return True only when a market is closed, undisputed, not in a pending UMA stage, and positively finalized."""
    if not isinstance(payload, Mapping) or payload.get("closed") is not True:
        return False
    if payload.get("disputed") is True:
        return False
    res_block = payload.get("resolution") if isinstance(payload.get("resolution"), Mapping) else {}
    if res_block.get("disputed") is True:
        return False

    statuses = [
        str(val or "").strip().lower()
        for val in (
            payload.get("umaResolutionStatus"),
            payload.get("uma_resolution_status"),
            payload.get("resolutionStatus"),
            payload.get("resolution_status"),
            res_block.get("umaResolutionStatus"),
            res_block.get("uma_resolution_status"),
            res_block.get("status"),
            res_block.get("resolutionStatus"),
            res_block.get("resolution_status"),
        )
    ]
    if any(st in NON_FINAL_RESOLUTION_STATUSES for st in statuses if st):
        return False

    if any(st in FINAL_RESOLUTION_STATUSES for st in statuses if st):
        return True
    if payload.get("resolved") is True or res_block.get("resolved") is True:
        return True

    tokens = payload.get("tokens")
    if isinstance(tokens, list):
        winning_tokens = [tok for tok in tokens if isinstance(tok, Mapping) and tok.get("winner") is True]
        if len(winning_tokens) == 1:
            return True

    return False


class UpstreamObserver:
    """Strictly read-only, non-blocking observer for `crypto-predict_system`."""

    def __init__(
        self,
        pipeline_state_path: str | Path,
        policy_path: str | Path,
        http_base_url: str = "http://127.0.0.1:8897",
        required_host_header: str = "127.0.0.1:8897",
        upstream_root: str | Path = r"C:\Users\liuqi\crypto-predict_system",
        clob_host: str = "https://clob.polymarket.com",
        gamma_host: str = "https://gamma-api.polymarket.com",
    ) -> None:
        self.upstream_root = Path(upstream_root).resolve()
        self.pipeline_state_path = assert_safe_read_path(pipeline_state_path, self.upstream_root)
        self.policy_path = assert_safe_read_path(policy_path, self.upstream_root)
        self.http_base_url = http_base_url.rstrip("/")
        self.required_host_header = required_host_header
        self.clob_host = clob_host.rstrip("/")
        self.gamma_host = gamma_host.rstrip("/")

        self._last_mtime_ns: int | None = None
        self._last_size: int | None = None
        self._cached_state: dict[str, Any] | None = None
        self._cached_status: dict[str, Any] | None = None
        self._last_policy_mtime_ns: int | None = None
        self._last_policy_size: int | None = None
        self._cached_policy: dict[str, Any] | None = None
        self._cached_policy_sha256: str | None = None
        self._last_policy_error: str | None = None
        self._last_policy_verified_at: str | None = None

        self._session = requests.Session()
        self._session.trust_env = False

    def close(self) -> None:
        self._session.close()

    @staticmethod
    def is_in_critical_window(now_utc: datetime | None = None) -> bool:
        """Return True during `T-2s .. T+21s` around any 5-minute boundary.

        Upstream executes `BEGIN IMMEDIATE` on `account.sqlite3` during `T+4s..T+20s`.
        Background HTTP calls to `/api/status` or `/api/ledger` are deferred outside this window.
        """
        dt = now_utc or datetime.now(timezone.utc)
        elapsed = dt.timestamp() % 300.0
        return elapsed <= 21.0 or elapsed >= 298.0

    def get_diagnostic_policy_state(self) -> dict[str, Any]:
        """Return diagnostic-only policy status & last known policy (NEVER authorizes new orders)."""
        return {
            "verified": self._last_policy_error is None and self._cached_policy is not None,
            "error": self._last_policy_error,
            "last_verified_at": self._last_policy_verified_at,
            "sha256": self._cached_policy_sha256,
            "diagnostic_cached_policy": dict(self._cached_policy) if self._cached_policy is not None else None,
        }

    def read_policy(self, force: bool = False) -> dict[str, Any]:
        """Read and strictly validate `config/argon4_forward_policy.json` (fail-closed on any error)."""
        assert_safe_read_path(self.policy_path, self.upstream_root)
        if not self.policy_path.is_file():
            self._last_policy_mtime_ns = None
            self._last_policy_size = None
            self._last_policy_error = f"POLICY_FILE_MISSING:{self.policy_path}"
            raise PolicyVerificationError("POLICY_FILE_MISSING", f"Policy file not found: {self.policy_path}")
        try:
            st = os.stat(self.policy_path)
            if (
                not force
                and self._last_policy_error is None
                and self._cached_policy is not None
                and st.st_mtime_ns == self._last_policy_mtime_ns
                and st.st_size == self._last_policy_size
            ):
                return dict(self._cached_policy)
            raw_bytes = self.policy_path.read_bytes()
            sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
            parsed = json.loads(raw_bytes.decode("utf-8"))
            validated = validate_policy_dict(parsed)
            self._cached_policy = validated
            self._cached_policy_sha256 = sha256_hex
            self._last_policy_mtime_ns = st.st_mtime_ns
            self._last_policy_size = st.st_size
            self._last_policy_error = None
            self._last_policy_verified_at = datetime.now(timezone.utc).isoformat()
            return dict(validated)
        except PolicyVerificationError as exc:
            self._last_policy_mtime_ns = None
            self._last_policy_size = None
            self._last_policy_error = f"{exc.reason_code}:{exc}"
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            self._last_policy_mtime_ns = None
            self._last_policy_size = None
            self._last_policy_error = f"POLICY_READ_FAILED:{type(exc).__name__}:{exc}"
            raise PolicyVerificationError("POLICY_READ_FAILED", f"Failed to verify policy {self.policy_path}: {exc}") from exc

    def poll_pipeline_state(self, force: bool = False) -> tuple[bool, dict[str, Any] | None, datetime]:
        """Check `pipeline_state.json` via `os.stat().st_mtime_ns`.

        Returns:
            `(changed, state_dict, detected_at_local_utc)`
        """
        assert_safe_read_path(self.pipeline_state_path, self.upstream_root)
        try:
            st = os.stat(self.pipeline_state_path)
        except FileNotFoundError:
            return False, None, datetime.now(timezone.utc)
        except OSError:
            return False, self._cached_state, datetime.now(timezone.utc)

        if not force and st.st_mtime_ns == self._last_mtime_ns and st.st_size == self._last_size:
            return False, self._cached_state, datetime.now(timezone.utc)

        # Read with micro-retry in case atomic .tmp replace is mid-rename on Windows
        detected_at = datetime.now(timezone.utc)
        for attempt in range(4):
            try:
                text = self.pipeline_state_path.read_text(encoding="utf-8")
                state = json.loads(text)
                self._last_mtime_ns = st.st_mtime_ns
                self._last_size = st.st_size
                self._cached_state = state
                return True, state, detected_at
            except (OSError, json.JSONDecodeError):
                if attempt < 3:
                    time.sleep(0.005)
        return False, self._cached_state, detected_at

    def _http_get_json(self, path: str, timeout: tuple[float, float] = (1.5, 2.5)) -> Any:
        """Issue GET request to `127.0.0.1:8897` with mandatory `Host` header and no `Origin` header."""
        url = f"{self.http_base_url}{path}"
        headers = {
            "Host": self.required_host_header,
            "Accept": "application/json",
            "User-Agent": "argon4-external-live-observer/2.0",
        }
        # Explicitly guarantee no Origin header is ever attached
        assert "Origin" not in headers and "origin" not in headers
        resp = self._session.get(url, headers=headers, timeout=timeout, allow_redirects=False)
        if resp.status_code != 200:
            raise RuntimeError(f"Upstream HTTP {resp.status_code} on {path}")
        return resp.json()

    def fetch_status(self, allow_in_critical_window: bool = False) -> dict[str, Any] | None:
        """Fetch `GET /api/status` safely."""
        if not allow_in_critical_window and self.is_in_critical_window():
            return self._cached_status
        try:
            data = self._http_get_json("/api/status")
            if isinstance(data, dict):
                self._cached_status = data
            return self._cached_status
        except Exception:
            return self._cached_status

    def fetch_ledger(self, allow_in_critical_window: bool = False) -> list[dict[str, Any]]:
        """Fetch `GET /api/ledger` (latest hash-chained account events)."""
        if not allow_in_critical_window and self.is_in_critical_window():
            return []
        try:
            data = self._http_get_json("/api/ledger")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def resolve_unsettled_market_fallback(
        self,
        market_id: str,
        condition_id: str,
        token_id: str = "",
        expected_outcome: str = "",
        allow_in_critical_window: bool = False,
    ) -> dict[str, Any] | None:
        """Resolve a market that aged out of the 100-event `/api/ledger` buffer during downtime.

        Checks:
          1. Upstream `/api/status` (`recent_resolutions` / `settled_markets`).
          2. Polymarket CLOB `GET {clob_host}/markets/{condition_id}` (`tokens[*].winner`).
          3. Polymarket Gamma `GET {gamma_host}/markets/{market_id}` (`outcomePrices`).
        """
        if not allow_in_critical_window and self.is_in_critical_window():
            return None

        # 1. Check cached or fresh upstream /api/status if it lists resolved markets
        status_data = self.fetch_status(allow_in_critical_window=allow_in_critical_window)
        if isinstance(status_data, Mapping):
            for key in ("recent_resolutions", "resolutions", "settled_markets"):
                items = status_data.get(key)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, Mapping) and str(item.get("market_id") or "") == str(market_id):
                            out = str(
                                item.get("official_outcome")
                                or (item.get("resolution") or {}).get("outcome")
                                or item.get("outcome")
                                or ""
                            ).upper().strip()
                            if out in ("UP", "DOWN", "VOID"):
                                ratios = extract_payout_ratios_from_payload(item)
                                if ratios is None and isinstance(item.get("resolution"), Mapping):
                                    ratios = extract_payout_ratios_from_payload(item["resolution"])
                                if ratios is None and out in ("UP", "DOWN"):
                                    ratios = {"UP": 1.0, "DOWN": 0.0} if out == "UP" else {"UP": 0.0, "DOWN": 1.0}
                                if ratios is None and out == "VOID":
                                    ratios = self.fetch_official_market_payout_ratios(
                                        str(market_id),
                                        str(condition_id or item.get("condition_id") or ""),
                                    )
                                return {
                                    "market_id": str(market_id),
                                    "condition_id": str(condition_id or item.get("condition_id") or ""),
                                    "official_outcome": out,
                                    "payout_ratios": ratios,
                                    "paper_payout_usd": float(item.get("payout_usd") or 0.0),
                                    "paper_realized_pnl_usd": float(item.get("realized_pnl_usd") or 0.0),
                                    "settled_at": item.get("utc") or item.get("settled_at"),
                                    "source": "upstream_status",
                                }

        # 2. Check Polymarket CLOB `/markets/{condition_id}`
        if condition_id and str(condition_id).startswith("0x"):
            try:
                resp = requests.get(f"{self.clob_host}/markets/{condition_id}", timeout=2.5)
                if resp.status_code == 200:
                    mkt = resp.json()
                    if _is_market_officially_finalized(mkt):
                        clob_ratios = extract_payout_ratios_from_payload(mkt, allow_outcome_prices=False)
                        tokens = mkt.get("tokens")
                        if isinstance(tokens, list):
                            winning_tokens = [
                                tok for tok in tokens if isinstance(tok, Mapping) and tok.get("winner") is True
                            ]
                            if len(winning_tokens) == 1:
                                tok = winning_tokens[0]
                                raw_out = str(tok.get("outcome") or "").upper().strip()
                                tok_id = str(tok.get("token_id") or "").strip()
                                if raw_out in ("UP", "YES"):
                                    winner = "UP"
                                elif raw_out in ("DOWN", "NO"):
                                    winner = "DOWN"
                                elif token_id and tok_id == str(token_id) and expected_outcome:
                                    winner = str(expected_outcome).upper().strip()
                                else:
                                    winner = raw_out
                                if winner in ("UP", "DOWN", "VOID"):
                                    if clob_ratios is None and winner in ("UP", "DOWN"):
                                        clob_ratios = {"UP": 1.0, "DOWN": 0.0} if winner == "UP" else {"UP": 0.0, "DOWN": 1.0}
                                    return {
                                        "market_id": str(market_id),
                                        "condition_id": str(condition_id),
                                        "official_outcome": winner,
                                        "payout_ratios": clob_ratios,
                                        "paper_payout_usd": 0.0,
                                        "paper_realized_pnl_usd": 0.0,
                                        "settled_at": None,
                                        "source": "polymarket_clob",
                                    }
                        if clob_ratios is not None and any(0.0 < float(v) < 1.0 for v in clob_ratios.values()):
                            return {
                                "market_id": str(market_id),
                                "condition_id": str(condition_id),
                                "official_outcome": "VOID",
                                "payout_ratios": clob_ratios,
                                "paper_payout_usd": 0.0,
                                "paper_realized_pnl_usd": 0.0,
                                "settled_at": None,
                                "source": "polymarket_clob",
                            }
            except Exception:
                pass

        # 3. Check Polymarket Gamma `/markets/{market_id}` — strictly require official final oracle resolution (`[P1-C]`)
        if market_id:
            try:
                resp = requests.get(f"{self.gamma_host}/markets/{market_id}", timeout=2.5)
                if resp.status_code == 200:
                    gm = resp.json()
                    if _is_market_officially_finalized(gm):
                        res_block = gm.get("resolution") if isinstance(gm.get("resolution"), Mapping) else {}
                        gamma_ratios = extract_payout_ratios_from_payload(gm, allow_outcome_prices=False)
                        if gamma_ratios is None and res_block:
                            gamma_ratios = extract_payout_ratios_from_payload(res_block, allow_outcome_prices=False)

                        # 3a. Check explicit winner / resolution outcome field first (preserving authoritative payout_ratios)
                        explicit_out = str(
                            gm.get("winningOutcome")
                            or gm.get("winner")
                            or res_block.get("outcome")
                            or res_block.get("winner")
                            or ""
                        ).upper().strip()
                        if explicit_out in ("UP", "YES", "DOWN", "NO", "VOID"):
                            winner = (
                                "UP"
                                if explicit_out in ("UP", "YES")
                                else ("DOWN" if explicit_out in ("DOWN", "NO") else explicit_out)
                            )
                            if gamma_ratios is None and winner in ("UP", "DOWN"):
                                gamma_ratios = {"UP": 1.0, "DOWN": 0.0} if winner == "UP" else {"UP": 0.0, "DOWN": 1.0}
                            return {
                                "market_id": str(market_id),
                                "condition_id": str(condition_id or gm.get("conditionId") or ""),
                                "official_outcome": winner,
                                "payout_ratios": gamma_ratios,
                                "paper_payout_usd": 0.0,
                                "paper_realized_pnl_usd": 0.0,
                                "settled_at": gm.get("resolvedAt") or gm.get("closedTime"),
                                "source": "polymarket_gamma",
                            }

                        # 3b. Check exact settled authoritative payout vector ([1, 0], [0, 1], or split VOID),
                        # OR exact binary `outcomePrices` ([1, 0] / [0, 1]) ONLY when finalized and binary.
                        # Arbitrary fractional `outcomePrices` (e.g. [0.7, 0.3]) are NEVER treated as VOID ratios!
                        if gamma_ratios is not None:
                            exact_winners = [lbl for lbl, p_num in gamma_ratios.items() if p_num == 1.0]
                            exact_losers = [lbl for lbl, p_num in gamma_ratios.items() if p_num == 0.0]
                            if len(exact_winners) == 1 and len(exact_losers) == len(gamma_ratios) - 1:
                                winner = _normalize_outcome_key(exact_winners[0])
                                if winner in ("UP", "DOWN", "VOID"):
                                    return {
                                        "market_id": str(market_id),
                                        "condition_id": str(condition_id or gm.get("conditionId") or ""),
                                        "official_outcome": winner,
                                        "payout_ratios": gamma_ratios,
                                        "paper_payout_usd": 0.0,
                                        "paper_realized_pnl_usd": 0.0,
                                        "settled_at": gm.get("resolvedAt") or gm.get("closedTime"),
                                        "source": "polymarket_gamma",
                                    }
                            elif sum(gamma_ratios.values()) > 0.0:
                                return {
                                    "market_id": str(market_id),
                                    "condition_id": str(condition_id or gm.get("conditionId") or ""),
                                    "official_outcome": "VOID",
                                    "payout_ratios": gamma_ratios,
                                    "paper_payout_usd": 0.0,
                                    "paper_realized_pnl_usd": 0.0,
                                    "settled_at": gm.get("resolvedAt") or gm.get("closedTime"),
                                    "source": "polymarket_gamma",
                                }
                        else:
                            price_ratios = extract_payout_ratios_from_payload(gm, allow_outcome_prices=True)
                            if price_ratios is not None:
                                exact_winners = [lbl for lbl, p_num in price_ratios.items() if p_num == 1.0]
                                exact_losers = [lbl for lbl, p_num in price_ratios.items() if p_num == 0.0]
                                if len(exact_winners) == 1 and len(exact_losers) == len(price_ratios) - 1:
                                    winner = _normalize_outcome_key(exact_winners[0])
                                    if winner in ("UP", "DOWN"):
                                        return {
                                            "market_id": str(market_id),
                                            "condition_id": str(condition_id or gm.get("conditionId") or ""),
                                            "official_outcome": winner,
                                            "payout_ratios": price_ratios,
                                            "paper_payout_usd": 0.0,
                                            "paper_realized_pnl_usd": 0.0,
                                            "settled_at": gm.get("resolvedAt") or gm.get("closedTime"),
                                            "source": "polymarket_gamma",
                                        }
            except Exception:
                pass

        return None

    def fetch_official_market_payout_ratios(
        self,
        market_id: str,
        condition_id: str | None = None,
    ) -> dict[str, float] | None:
        """Query Polymarket Gamma / CLOB directly for verified, finalized official oracle payout ratios."""
        if market_id:
            try:
                resp = requests.get(f"{self.gamma_host}/markets/{market_id}", timeout=2.5)
                if resp.status_code == 200:
                    gm = resp.json()
                    if _is_market_officially_finalized(gm):
                        ratios = extract_payout_ratios_from_payload(gm, allow_outcome_prices=False)
                        if ratios is None and isinstance(gm.get("resolution"), Mapping):
                            ratios = extract_payout_ratios_from_payload(gm["resolution"], allow_outcome_prices=False)
                        if ratios is not None:
                            return ratios
            except Exception:
                pass
        if condition_id and str(condition_id).startswith("0x"):
            try:
                resp = requests.get(f"{self.clob_host}/markets/{condition_id}", timeout=2.5)
                if resp.status_code == 200:
                    mkt = resp.json()
                    if _is_market_officially_finalized(mkt):
                        ratios = extract_payout_ratios_from_payload(mkt, allow_outcome_prices=False)
                        if ratios is not None:
                            return ratios
            except Exception:
                pass
        return None

    @staticmethod
    def extract_candidate_fills(state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Extract all valid paper fills and their matching stream decision/signal from `pipeline_state.json`.

        Handles both initial execution (`PAPER_CYCLE_COMPLETE`) and multi-look retries
        (`PAPER_WAITING_PRICE_RECHECK`) where `execution.fills` retains prior fills and appends new ones.
        """
        if not isinstance(state, Mapping):
            return []
        execution = state.get("execution")
        if not isinstance(execution, Mapping):
            return []
        fills = execution.get("fills")
        if not isinstance(fills, list) or not fills:
            return []

        decisions = state.get("decisions") if isinstance(state.get("decisions"), Mapping) else {}
        signal_map: dict[str, tuple[str, dict[str, Any]]] = {}
        for stream_name, dec in decisions.items():
            if isinstance(dec, Mapping) and isinstance(dec.get("signal"), Mapping):
                sig_obj = dec["signal"]
                sig_id = str(sig_obj.get("signal_id") or "")
                if sig_id:
                    signal_map[sig_id] = (str(stream_name), dict(sig_obj))

        candidates: list[dict[str, Any]] = []
        for fill in fills:
            if not isinstance(fill, Mapping):
                continue
            signal_id = str(fill.get("signal_id") or "").strip()
            if not signal_id or fill.get("status") != "OPEN":
                continue

            stream_name, sig_obj = signal_map.get(signal_id, ("", {}))
            venue_mkt = fill.get("venue_market") if isinstance(fill.get("venue_market"), Mapping) else {}
            if not stream_name:
                underlying = str(fill.get("underlying") or venue_mkt.get("asset") or "")
                w_start = venue_mkt.get("window_start")
                w_end = venue_mkt.get("window_end")
                if underlying and w_start and w_end:
                    try:
                        dur_min = int(
                            (parse_utc_timestamp(w_end) - parse_utc_timestamp(w_start)).total_seconds() // 60
                        )
                        stream_name = f"{underlying}_{dur_min}m"
                    except ValueError:
                        stream_name = underlying

            auth = fill.get("allocation_authorization") if isinstance(fill.get("allocation_authorization"), Mapping) else {}
            cohort_id = str(
                auth.get("cohort_id")
                or state.get("last_cohort")
                or sig_obj.get("window_start")
                or venue_mkt.get("window_start")
                or ""
            )

            book_snap = fill.get("book_snapshot") if isinstance(fill.get("book_snapshot"), Mapping) else {}
            observed_best_ask = book_snap.get("best_ask") or venue_mkt.get("best_ask")
            book_observed_at = (
                fill.get("book_observed_at")
                or book_snap.get("observed_at")
                or book_snap.get("timestamp")
                or venue_mkt.get("book_observed_at")
            )
            market_protocol = str(venue_mkt.get("protocol") or fill.get("protocol") or "v1").strip().lower()
            neg_risk = bool(venue_mkt.get("neg_risk") or fill.get("neg_risk") or False)

            candidates.append(
                {
                    "cohort_id": cohort_id,
                    "stream": stream_name or "UNKNOWN_STREAM",
                    "signal_id": signal_id,
                    "fill": dict(fill),
                    "signal": sig_obj,
                    "observed_best_ask": observed_best_ask,
                    "book_observed_at": book_observed_at,
                    "market_protocol": market_protocol,
                    "neg_risk": neg_risk,
                }
            )
        return candidates

    @staticmethod
    def extract_paper_mirror_snapshot(
        status_payload: Mapping[str, Any] | None,
        pipeline_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the `paper_account_mirror` block for `live_quota_snapshots.jsonl` (Spec Section 5.2)."""
        account = (status_payload or {}).get("account") if isinstance(status_payload, Mapping) else {}
        open_risk = (status_payload or {}).get("open_risk") if isinstance(status_payload, Mapping) else {}
        account = account if isinstance(account, Mapping) else {}
        open_risk = open_risk if isinstance(open_risk, Mapping) else {}

        cohort_rem = 0.0
        if isinstance(pipeline_state, Mapping):
            exec_block = pipeline_state.get("execution")
            if isinstance(exec_block, Mapping) and isinstance(exec_block.get("cohort"), Mapping):
                cohort_rem = float(exec_block["cohort"].get("remaining_usd") or 0.0)

        return {
            "paper_cash_usd": float(account.get("cash") or 0.0),
            "paper_locked_cost_usd": float(account.get("locked_cost") or 0.0),
            "paper_active_window_locked_usd": float(
                open_risk.get("active_window_cost_usd", account.get("locked_cost") or 0.0)
            ),
            "paper_unsettled_expired_usd": float(open_risk.get("awaiting_resolution_cost_usd") or 0.0),
            "paper_cohort_remaining_usd": cohort_rem,
            "paper_realized_pnl_usd": float(account.get("realized_pnl") or 0.0),
        }
