"""Main orchestration loop for the ARGON4 External Live Executor.

Coordinates:
1. 50ms non-blocking `pipeline_state.json` polling (`UpstreamObserver`).
2. Atomic `BEGIN IMMEDIATE` risk quota reservation (`AuditStore.claim_and_reserve_signal`)
   before signing/submitting so concurrent callers can never breach `max_open_risk_usd` ([P1-6]),
   coupled with `SingleInstanceFileLock`.
3. Enforcing wallet free balance & exchange allowance caps (`balance_verified`, [P1-7]),
   independent live daily realized loss stop & consecutive-loss cooldown (Design Gap #1),
   `min_buy_price = 0.37` floor (Design Gap #4), and pre-send `valid_until` re-check ([P1-5]).
4. Startup & periodic order reconciliation (`reconcile_pending_orders`, [P1-2] & [P1-3]).
5. Post-settlement resolution detection (`/api/ledger` + `last_ledger_seq` cursor +
   unresolved market fallback via `/api/status` & Polymarket CLOB/Gamma APIs, Design Gap #3)
   and strict confirmation before crediting redeemed pUSD ([P1-4]).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from dotenv import load_dotenv

from .audit_store import AuditStore
from .clob_executor import ClobExecutor
from .math_utils import (
    HARD_WINDOW_DEADLINE_SECONDS,
    USD_MICRO,
    align_limit_price,
    check_hard_deadline,
    compute_diff_vs_paper,
    compute_paper_benchmark_metrics,
    derive_effective_order_deadline,
    format_utc_iso,
    parse_utc_timestamp,
    project_to_calibrated_utc,
    to_decimal,
    validate_and_project_utc,
)
from .redeemer import AutoRedeemer, resolve_verified_payout_ratios
from .safety import (
    OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS,
    OFFICIAL_NEG_RISK_COLLATERAL_ADAPTER_ADDRESS,
    OFFICIAL_POLYGON_CTF_ADDRESS,
    OFFICIAL_POLYGON_PUSD_ADDRESS,
    OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS,
    SingleInstanceFileLock,
    assert_no_withdrawal_permission,
)
from .upstream_observer import (
    PolicyVerificationError,
    UpstreamObserver,
    extract_payout_ratios_from_payload,
)

logger = logging.getLogger("argon4_live.engine")


def load_live_config(config_path: str | Path = r"C:\Users\liuqi\Order\config\live_config.json") -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert_no_withdrawal_permission(cfg)
    return cfg


def resolve_mode_audit_paths(audit_cfg: Mapping[str, Any], mode: str) -> dict[str, str]:
    """Return isolated runtime/audit file paths for `shadow` vs `live` mode (Design Gap #2)."""
    mode_key = str(mode or "shadow").strip().lower()
    modes_map = audit_cfg.get("modes")
    if isinstance(modes_map, Mapping) and isinstance(modes_map.get(mode_key), Mapping):
        sub = modes_map[mode_key]
        return {
            "runtime_dir": str(sub["runtime_dir"]),
            "sqlite_db": str(sub["sqlite_db"]),
            "orders_jsonl": str(sub["orders_jsonl"]),
            "quota_snapshots_jsonl": str(sub["quota_snapshots_jsonl"]),
            "settlements_jsonl": str(sub["settlements_jsonl"]),
        }
    return {
        "runtime_dir": str(audit_cfg["runtime_dir"]),
        "sqlite_db": str(audit_cfg["sqlite_db"]),
        "orders_jsonl": str(audit_cfg["orders_jsonl"]),
        "quota_snapshots_jsonl": str(audit_cfg["quota_snapshots_jsonl"]),
        "settlements_jsonl": str(audit_cfg["settlements_jsonl"]),
    }


class LiveExecutionEngine:
    """Strictly isolated external live listener, V2 signer, FAK executor, and auto-redeemer."""

    def __init__(
        self,
        config: Mapping[str, Any],
        observer: UpstreamObserver,
        audit_store: AuditStore,
        clob_executor: ClobExecutor,
        redeemer: AutoRedeemer,
        instance_lock: SingleInstanceFileLock | None = None,
    ) -> None:
        assert_no_withdrawal_permission(dict(config))
        self.config = dict(config)
        self.observer = observer
        self.audit = audit_store
        self.clob = clob_executor
        self.redeemer = redeemer
        self.instance_lock = instance_lock
        self.stop_event = threading.Event()

        cap_cfg = self.config.get("capital_and_scaling", {})
        if cap_cfg.get("live_scale_factor") is not None:
            self.live_scale_factor = float(cap_cfg["live_scale_factor"])
        else:
            self.live_scale_factor = float(os.getenv("ARGON4_LIVE_SCALE_FACTOR") or 0.10)
        self.paper_max_daily_loss_usd = float(cap_cfg.get("paper_max_daily_realized_loss_usd", 300.0))
        self.max_daily_loss_usd_override = (
            float(cap_cfg["max_daily_realized_loss_usd"])
            if cap_cfg.get("max_daily_realized_loss_usd") is not None
            else None
        )
        self.max_consecutive_losses = int(cap_cfg.get("max_consecutive_losses", 4))
        self.cooldown_minutes_after_consecutive_losses = float(
            cap_cfg.get("cooldown_minutes_after_consecutive_losses", 30.0)
        )

        exec_cfg = self.config.get("order_execution", {})
        self.fak_no_match_max_retries = max(0, int(exec_cfg.get("fak_no_match_max_retries", 2)))
        self.fak_no_match_retry_interval_seconds = max(
            0.05, float(exec_cfg.get("fak_no_match_retry_interval_seconds", 1.2))
        )
        self.fak_no_match_min_remaining_seconds = max(
            0.20, float(exec_cfg.get("fak_no_match_min_remaining_seconds", 1.8))
        )
        self.hard_deadline_after_cohort_seconds = min(
            HARD_WINDOW_DEADLINE_SECONDS,
            max(0.5, float(exec_cfg.get("hard_deadline_after_cohort_seconds", HARD_WINDOW_DEADLINE_SECONDS))),
        )
        self._last_verified_policy_sha256: str | None = None

        up_cfg = self.config.get("upstream", {})
        self.poll_interval_seconds = max(0.01, float(up_cfg.get("poll_interval_ms", 50)) / 1000.0)
        self.settlement_poll_interval = float(up_cfg.get("settlement_poll_interval_seconds", 15.0))
        self.prewarm_interval = float(up_cfg.get("token_cache_prewarm_interval_seconds", 60.0))
        self.reconcile_interval = float(up_cfg.get("order_reconcile_interval_seconds", 10.0))

        self._last_settlement_poll_mono = 0.0
        self._last_prewarm_mono = 0.0
        self._last_reconcile_mono = 0.0

    @classmethod
    def from_config_and_env(
        cls,
        config_path: str | Path = r"C:\Users\liuqi\Order\config\live_config.json",
        env_path: str | Path = r"C:\Users\liuqi\Order\.env",
        mode_override: str | None = None,
        scale_override: float | None = None,
        acquire_instance_lock: bool = True,
    ) -> "LiveExecutionEngine":
        env_file = Path(env_path).expanduser().resolve()
        if env_file.is_file():
            load_dotenv(env_file, override=False)

        cfg = load_live_config(config_path)
        if scale_override is not None:
            cfg.setdefault("capital_and_scaling", {})["live_scale_factor"] = float(scale_override)
        elif os.getenv("ARGON4_LIVE_SCALE_FACTOR"):
            cfg.setdefault("capital_and_scaling", {})["live_scale_factor"] = float(
                os.getenv("ARGON4_LIVE_SCALE_FACTOR", "0.10")
            )

        mode = (mode_override or os.getenv("ARGON4_EXECUTION_MODE") or "shadow").strip().lower()
        up_cfg = cfg["upstream"]
        audit_cfg = cfg["audit"]
        exec_cfg = cfg["order_execution"]
        red_cfg = cfg["redemption"]

        mode_paths = resolve_mode_audit_paths(audit_cfg, mode)
        cfg["audit"] = {**dict(audit_cfg), **mode_paths}

        instance_lock: SingleInstanceFileLock | None = None
        if acquire_instance_lock:
            lock_path = Path(mode_paths["runtime_dir"]) / "live_executor.lock"
            instance_lock = SingleInstanceFileLock(lock_path, local_root=Path(config_path).resolve().parents[1])
            instance_lock.acquire()

        try:
            observer = UpstreamObserver(
                pipeline_state_path=up_cfg["pipeline_state_path"],
                policy_path=up_cfg["policy_path"],
                http_base_url=up_cfg["http_base_url"],
                required_host_header=up_cfg["required_host_header"],
                upstream_root=up_cfg["repo_root"],
                clob_host=cfg.get("clob_host", "https://clob.polymarket.com"),
            )

            audit_store = AuditStore(
                sqlite_path=mode_paths["sqlite_db"],
                orders_jsonl_path=mode_paths["orders_jsonl"],
                quota_jsonl_path=mode_paths["quota_snapshots_jsonl"],
                settlements_jsonl_path=mode_paths["settlements_jsonl"],
                local_root=Path(config_path).resolve().parents[1],
                upstream_root=up_cfg["repo_root"],
            )

            private_key = os.getenv("POLY_PRIVATE_KEY", "")
            funder_address = os.getenv("POLY_FUNDER_ADDRESS", "")
            raw_sig_env = os.getenv("POLY_SIGNATURE_TYPE", "").strip()
            sig_type: int | None = int(raw_sig_env) if raw_sig_env else None
            cfg_rpc_url = str(cfg.get("polygon_rpc_url") or "https://polygon-rpc.com").strip()
            env_rpc_url = os.getenv("POLYGON_RPC_URL", "").strip()
            rpc_url = env_rpc_url or cfg_rpc_url
            raw_env_fallbacks = os.getenv("POLYGON_RPC_FALLBACK_URLS", "").strip()
            cfg_fallbacks = cfg.get("polygon_rpc_fallback_urls")
            rpc_fallbacks: list[str] = []
            if raw_env_fallbacks:
                rpc_fallbacks.extend(x.strip() for x in raw_env_fallbacks.split(",") if x.strip())
            elif isinstance(cfg_fallbacks, (list, tuple)):
                rpc_fallbacks.extend(str(x).strip() for x in cfg_fallbacks if str(x).strip())
            if env_rpc_url and cfg_rpc_url and cfg_rpc_url != env_rpc_url:
                rpc_fallbacks.insert(0, cfg_rpc_url)
            creds_cache_path = Path(mode_paths["runtime_dir"]) / ".derived_clob_creds.json"
            pusd_addr = red_cfg.get("pusd_collateral_address", OFFICIAL_POLYGON_PUSD_ADDRESS)

            clob_executor = ClobExecutor(
                mode=mode,
                host=cfg.get("clob_host", "https://clob.polymarket.com"),
                relayer_host=cfg.get("relayer_host", "https://relayer-v2.polymarket.com"),
                chain_id=int(cfg.get("chain_id", 137)),
                private_key=private_key,
                api_key=os.getenv("POLY_API_KEY", ""),
                api_secret=os.getenv("POLY_API_SECRET", ""),
                api_passphrase=os.getenv("POLY_API_PASSPHRASE", ""),
                signature_type=sig_type,
                funder_address=funder_address,
                polygon_rpc_url=rpc_url,
                polygon_rpc_fallback_urls=rpc_fallbacks,
                pusd_collateral_address=pusd_addr,
                default_tick_size=str(exec_cfg.get("default_tick_size", "0.01")),
                default_clob_fee_rate_bps=int(exec_cfg.get("default_clob_fee_rate_bps", 1000)),
                tick_size_limit_prices=exec_cfg.get("tick_size_limit_prices"),
                min_buy_price=exec_cfg.get("min_buy_price", 0.37),
                max_buy_price=exec_cfg.get("max_buy_price", 0.515),
                creds_cache_path=creds_cache_path,
                fresh_observed_book_max_age_seconds=float(exec_cfg.get("fresh_observed_book_max_age_seconds", 1.5)),
            )

            redeemer = AutoRedeemer(
                mode=mode,
                relayer_host=cfg.get("relayer_host", "https://relayer-v2.polymarket.com"),
                chain_id=int(cfg.get("chain_id", 137)),
                private_key=private_key,
                signature_type=clob_executor.signature_type,
                funder_address=clob_executor.funder_address,
                relayer_api_key=os.getenv("RELAYER_API_KEY", ""),
                relayer_api_secret=os.getenv("RELAYER_API_SECRET", ""),
                relayer_api_key_address=os.getenv("RELAYER_API_KEY_ADDRESS", ""),
                builder_api_key=os.getenv("POLY_BUILDER_API_KEY", ""),
                builder_secret=os.getenv("POLY_BUILDER_SECRET", ""),
                builder_passphrase=os.getenv("POLY_BUILDER_PASSPHRASE", ""),
                polygon_rpc_url=rpc_url,
                polygon_rpc_fallback_urls=rpc_fallbacks,
                ctf_address=red_cfg.get("ctf_collateral_adapter_address", OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS),
                ctf_contract_address=red_cfg.get("ctf_contract_address", OFFICIAL_POLYGON_CTF_ADDRESS),
                neg_risk_adapter_address=red_cfg.get(
                    "neg_risk_collateral_adapter_address", OFFICIAL_NEG_RISK_COLLATERAL_ADAPTER_ADDRESS
                ),
                v2_router_address=red_cfg.get("protocol_v2_router_address", OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS),
                pusd_address=pusd_addr,
                withdrawal_permission=bool(red_cfg.get("withdrawal_permission", False)),
                require_onchain_receipt=bool(red_cfg.get("require_onchain_receipt", True)),
            )

            return cls(
                config=cfg,
                observer=observer,
                audit_store=audit_store,
                clob_executor=clob_executor,
                redeemer=redeemer,
                instance_lock=instance_lock,
            )
        except Exception:
            if instance_lock is not None:
                instance_lock.release()
            raise

    def close(self) -> None:
        self.observer.close()
        for client in (self.clob, self.redeemer):
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        if self.instance_lock is not None:
            self.instance_lock.release()

    def _paper_risk_caps(self) -> dict[str, Any]:
        policy = self.observer.read_policy()
        raw_stream_min = policy.get("stream_min_buy_price")
        stream_min_buy_price = (
            {str(k): float(v) for k, v in raw_stream_min.items() if v is not None}
            if isinstance(raw_stream_min, Mapping)
            else {}
        )
        raw_stream_max = policy.get("stream_max_order_usd")
        stream_max_order_usd = (
            {str(k): float(v) for k, v in raw_stream_max.items() if v is not None}
            if isinstance(raw_stream_max, Mapping)
            else {}
        )
        return {
            "max_order_usd": float(policy["max_order_usd"]),
            "stream_max_order_usd": stream_max_order_usd,
            "max_active_window_risk_usd": float(policy["max_active_window_risk_usd"]),
            "max_open_risk_usd": float(policy["max_open_risk_usd"]),
            "max_underlying_active_risk_usd": float(policy["max_underlying_active_risk_usd"]),
            "max_underlying_open_risk_usd": float(policy["max_underlying_open_risk_usd"]),
            "max_tick_budget_usd": float(policy["max_tick_budget_usd"]),
            "min_buy_price": float(policy["min_buy_price"]),
            "stream_min_buy_price": stream_min_buy_price,
            "max_buy_price": float(policy["max_buy_price"]),
        }

    def _live_loss_limits(self, scale_dec: Decimal, now_utc: datetime | None = None) -> tuple[Decimal, int, float]:
        if self.max_daily_loss_usd_override is not None:
            daily_loss_cap = to_decimal(self.max_daily_loss_usd_override, "max_daily_realized_loss_usd")
        else:
            policy = self.observer.read_policy()
            init_cap = policy.get("initial_capital_usd")
            loss_frac = policy.get("daily_loss_fraction")
            if init_cap is not None and loss_frac is not None:
                base_live_cap = scale_dec * to_decimal(init_cap, "initial_capital_usd")
                prior_live_pnl = (
                    self.audit.get_prior_days_realized_pnl_usdc(now_utc)
                    if hasattr(self.audit, "get_prior_days_realized_pnl_usdc")
                    else Decimal("0")
                )
                day_start_live_equity = max(Decimal("0"), base_live_cap + prior_live_pnl)
                daily_loss_cap = (
                    day_start_live_equity * to_decimal(loss_frac, "daily_loss_fraction")
                ).quantize(USD_MICRO, rounding=ROUND_DOWN)
            else:
                paper_daily_loss = to_decimal(self.paper_max_daily_loss_usd)
                daily_loss_cap = (scale_dec * paper_daily_loss).quantize(
                    USD_MICRO, rounding=ROUND_DOWN
                )
        return daily_loss_cap, self.max_consecutive_losses, self.cooldown_minutes_after_consecutive_losses

    def resolve_auto_equity_sizing(
        self,
        *,
        onchain_free_usdc: float | Decimal | None,
        balance_verified: bool,
        stream: str,
        scale_dec: Decimal,
        now_utc: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Compute true-wallet-driven tiered order sizing and risk caps when enabled."""
        cap_cfg = self.config.get("capital_and_scaling") or {}
        auto_cfg = cap_cfg.get("auto_equity_sizing") if isinstance(cap_cfg, Mapping) else None
        if not (isinstance(auto_cfg, Mapping) and auto_cfg.get("enabled") is True):
            return None
        clob_mode = str(getattr(self.clob, "mode", "live") or "live").lower()
        if clob_mode != "live" and not bool(auto_cfg.get("apply_in_shadow", False)):
            return None
        if not balance_verified or onchain_free_usdc is None or not hasattr(self.audit, "get_live_equity_and_today_pnl"):
            return None
        tiers = auto_cfg.get("tiers")
        if not isinstance(tiers, list) or not tiers:
            return None

        eq_info = self.audit.get_live_equity_and_today_pnl(onchain_free_usdc, now_utc=now_utc)
        eff_eq = (eq_info["true_live_equity_dec"] * scale_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
        eff_day_start = (eq_info["day_start_true_equity_dec"] * scale_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
        if eff_eq <= Decimal("0"):
            return {
                "tier_name": "DEPLETED_EQUITY_STOP",
                "true_live_equity_usdc": float(eff_eq),
                "target_stake_dec": Decimal("0"),
                "cohort_budget_usd": 0.0,
                "daily_loss_cap_dec": Decimal("0"),
                "caps_override": {
                    "max_order_usd": 0.0,
                    "max_tick_budget_usd": 0.0,
                    "max_active_window_risk_usd": 0.0,
                    "max_open_risk_usd": 0.0,
                    "max_underlying_active_risk_usd": 0.0,
                    "max_underlying_open_risk_usd": 0.0,
                },
            }

        ordered_tiers = sorted(
            (t for t in tiers if isinstance(t, Mapping) and t.get("min_equity_usd") is not None),
            key=lambda t: float(t["min_equity_usd"]),
        )
        if not ordered_tiers:
            return None
        selected_tier = ordered_tiers[0]
        for t in ordered_tiers:
            if eff_eq >= to_decimal(t["min_equity_usd"], "min_equity_usd"):
                selected_tier = t

        single_frac = to_decimal(selected_tier["single_stream_equity_fraction"], "single_stream_equity_fraction")
        tick_frac = to_decimal(selected_tier["tick_equity_fraction"], "tick_equity_fraction")
        loss_frac = to_decimal(selected_tier["daily_loss_fraction"], "daily_loss_fraction")
        max_ord = to_decimal(selected_tier["max_order_usd"], "max_order_usd")
        max_tick = to_decimal(selected_tier["max_tick_budget_usd"], "max_tick_budget_usd")
        max_open = to_decimal(selected_tier["max_open_risk_usd"], "max_open_risk_usd")
        max_und = to_decimal(selected_tier["max_underlying_open_risk_usd"], "max_underlying_open_risk_usd")
        s_map = selected_tier.get("stream_max_order_usd")
        stream_cap = (
            min(max_ord, to_decimal(s_map[stream], "stream_max_order_usd"))
            if isinstance(s_map, Mapping) and stream in s_map and s_map[stream] is not None
            else max_ord
        )
        max_active = (
            to_decimal(selected_tier["max_active_window_risk_usd"], "max_active_window_risk_usd")
            if selected_tier.get("max_active_window_risk_usd") is not None
            else max(max_tick, min(max_open, (max_open * Decimal("0.70")).quantize(Decimal("1"), rounding=ROUND_DOWN)))
        )
        max_und_active = (
            to_decimal(selected_tier["max_underlying_active_risk_usd"], "max_underlying_active_risk_usd")
            if selected_tier.get("max_underlying_active_risk_usd") is not None
            else max(stream_cap, min(max_und, (max_und * Decimal("0.68")).quantize(Decimal("1"), rounding=ROUND_DOWN)))
        )

        daily_loss_cap = (
            to_decimal(self.max_daily_loss_usd_override, "max_daily_realized_loss_usd")
            if self.max_daily_loss_usd_override is not None
            else (eff_day_start * loss_frac).quantize(USD_MICRO, rounding=ROUND_DOWN)
        )
        today_pnl = eq_info["today_realized_pnl_dec"]
        rem_loss_room = (
            max(Decimal("0"), (daily_loss_cap + today_pnl).quantize(USD_MICRO, rounding=ROUND_DOWN))
            if today_pnl < Decimal("0")
            else Decimal("1000000000")
        )
        target_stake = min(
            (eff_eq * single_frac).quantize(USD_MICRO, rounding=ROUND_DOWN),
            stream_cap,
            rem_loss_room,
        )
        cohort_budget = min(
            (eff_eq * tick_frac).quantize(USD_MICRO, rounding=ROUND_DOWN),
            max_tick,
            rem_loss_room,
        )
        return {
            "tier_name": str(selected_tier.get("name") or "AUTO_TIER"),
            "true_live_equity_usdc": float(eff_eq),
            "target_stake_dec": max(Decimal("0"), target_stake),
            "cohort_budget_usd": float(max(Decimal("0"), cohort_budget)),
            "daily_loss_cap_dec": daily_loss_cap,
            "caps_override": {
                "max_order_usd": float(stream_cap),
                "max_tick_budget_usd": float(max_tick),
                "max_active_window_risk_usd": float(max_active),
                "max_open_risk_usd": float(max_open),
                "max_underlying_active_risk_usd": float(max_und_active),
                "max_underlying_open_risk_usd": float(max_und),
            },
        }

    def _default_limit_price(self) -> float:
        """Resolve configured tick-aligned limit price (capped by `max_buy_price`) for abort records."""
        tick = str(getattr(self.clob, "default_tick_size", "0.01"))
        mapping = getattr(self.clob, "tick_size_limit_prices", None) or {}
        val = float(mapping[tick]) if tick in mapping else float(mapping.get("0.01", 0.48))
        max_p = getattr(self.clob, "max_buy_price", None)
        if max_p is not None and val > float(max_p):
            try:
                tk_dec = to_decimal(tick, "tick_size").normalize()
                floored = float((to_decimal(max_p, "max_buy_price") // tk_dec) * tk_dec)
                if floored > 0:
                    val = floored
            except ValueError:
                pass
        return val

    def _wait_retry_interval(self, seconds: float) -> bool:
        """Sleep up to `seconds` between bounded FAK retries; returns True if `stop_event` interrupted."""
        return bool(self.stop_event.wait(max(0.0, float(seconds))))

    @staticmethod
    def _is_retryable_fak_no_match(live_exec: Mapping[str, Any]) -> bool:
        status = str(live_exec.get("clob_status") or "").strip().upper()
        err_lower = str(live_exec.get("error_code") or "").strip().lower()
        if status not in ("CANCELED", "CANCELLED", "KILLED", "UNMATCHED"):
            return False
        if not (
            "fak_killed_no_match" in err_lower
            or "no orders found to match" in err_lower
            or "killed if no match is found" in err_lower
            or "order is not marketable" in err_lower
        ):
            return False
        if float(live_exec.get("total_spent_usdc") or 0.0) > 0.0:
            return False
        try:
            if to_decimal(live_exec.get("filled_quantity", "0") or "0", "filled_quantity") > Decimal("0"):
                return False
        except ValueError:
            return False
        return True

    def process_pipeline_state(
        self,
        state: Mapping[str, Any],
        detected_at_local_utc: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Inspect `pipeline_state.json` and execute any new `signal_id` in `execution.fills`."""
        detected_local = detected_at_local_utc or datetime.now(timezone.utc)
        detected_mono = time.perf_counter()
        candidates = self.observer.extract_candidate_fills(state)
        if not candidates:
            return []

        has_multiple_candidates = len(candidates) > 1
        try:
            paper_caps = self._paper_risk_caps()
            scale_dec = to_decimal(self.live_scale_factor, "live_scale_factor")
            daily_loss_cap, max_consec_losses, cooldown_mins = self._live_loss_limits(
                scale_dec, now_utc=detected_local
            )
        except PolicyVerificationError as pol_exc:
            self._last_verified_policy_sha256 = None
            halt_info = {
                "verified": False,
                "reason_code": pol_exc.reason_code,
                "error": str(pol_exc),
                "timestamp_utc": format_utc_iso(detected_local),
            }
            try:
                self.audit.set_kv("policy_verification_status", json.dumps(halt_info, ensure_ascii=False))
            except Exception:
                pass
            logger.error("Policy verification failed; halting new orders fail-closed: %s", pol_exc)
            return []

        diag_sha = self.observer.get_diagnostic_policy_state().get("sha256")
        if diag_sha != self._last_verified_policy_sha256:
            try:
                self.audit.set_kv(
                    "policy_verification_status",
                    json.dumps(
                        {
                            "verified": True,
                            "sha256": diag_sha,
                            "timestamp_utc": format_utc_iso(detected_local),
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception:
                pass
            self._last_verified_policy_sha256 = diag_sha

        min_buy_price = paper_caps.get("min_buy_price", 0.37)
        max_buy_price = paper_caps.get("max_buy_price", 0.515)

        cohort_obj = (state.get("execution") or {}).get("cohort") if isinstance(state.get("execution"), Mapping) else {}
        paper_cohort_budget = (
            cohort_obj.get("budget_usd") if isinstance(cohort_obj, Mapping) else paper_caps["max_tick_budget_usd"]
        )

        # Preserve exact candidate rank order across immediate finalizations and Phase-2 FAK retries
        ordered_slots: list[dict[str, Any] | None] = []
        deferred_snapshots: list[tuple[str, datetime]] = []
        retry_queue: list[dict[str, Any]] = []

        def _finalize_exec_bundle(ctx: Mapping[str, Any], exec_bundle: Mapping[str, Any], retry_count: int = 0) -> None:
            slot_idx = int(ctx["slot_idx"])
            sig_id = str(ctx["signal_id"])
            live_exec = dict(exec_bundle["live_execution"])
            bundle_sizing = exec_bundle.get("sizing") if isinstance(exec_bundle.get("sizing"), Mapping) else {}
            if isinstance(bundle_sizing, Mapping):
                for b_key in ("free_usdc_before_order", "auth_balance_seq_before_order", "local_spent_before_order"):
                    if bundle_sizing.get(b_key) is not None and live_exec.get(b_key) is None:
                        live_exec[b_key] = bundle_sizing[b_key]
            if retry_count > 0:
                live_exec["fak_retry_count"] = int(retry_count)
            retry_notes = ctx.get("retry_notes")
            if retry_notes and str(live_exec.get("clob_status") or "").upper() == "CANCELED":
                base_err = str(live_exec.get("error_code") or "FAK_KILLED_NO_MATCH")
                notes_str = ";".join(str(n) for n in retry_notes)
                live_exec["error_code"] = f"{base_err}|retries={notes_str}"[:360]

            if float(live_exec.get("total_spent_usdc") or 0.0) > 0.0:
                self.clob._deducted_order_keys.add(f"sig:{sig_id}")

            order_record = {
                "cohort_id": ctx["cohort_id"],
                "stream": ctx["stream"],
                "rank": ctx["rank"],
                "signal_id": sig_id,
                "paper_order_id": ctx["paper_order_id"],
                "market_id": ctx["market_id"],
                "condition_id": ctx["condition_id"],
                "token_id": ctx["token_id"],
                "outcome": ctx["outcome"],
                "market_protocol": ctx["market_protocol"],
                "neg_risk": ctx["neg_risk"],
                "fee_rate": float(ctx["fee_rate"]),
                "timing": {
                    "decision_at": ctx["decision_at_str"],
                    "paper_fill_at": ctx["paper_fill_at_str"],
                    "live_detected_at": format_utc_iso(ctx["live_detected_cal"]),
                    "live_signed_at": exec_bundle["live_signed_at"],
                    "live_http_sent_at": exec_bundle["live_http_sent_at"],
                    "live_http_ack_at": exec_bundle["live_http_ack_at"],
                    "detection_lag_ms": ctx["detection_lag_ms"],
                    "clob_rtt_ms": exec_bundle["clob_rtt_ms"],
                },
                "paper_benchmark": ctx["paper_benchmark"],
                "live_execution": live_exec,
                "diff_vs_paper": compute_diff_vs_paper(ctx["paper_benchmark"], live_exec, paper_status="OPEN"),
            }
            self.audit.record_order_execution(
                order_record,
                underlying=ctx["underlying"],
                window_start=ctx["window_start"],
                window_end=ctx["window_end"],
            )
            ordered_slots[slot_idx] = order_record
            deferred_snapshots.append((ctx["cohort_id"], parse_utc_timestamp(exec_bundle["live_http_ack_at"])))

        # Phase 1: Immediate Attempt-0 dispatch for all candidates in rank order (zero mid-batch snapshot/RPC blocking)
        for item in candidates:
            signal_id = item["signal_id"]
            if self.audit.is_signal_claimed(signal_id):
                continue

            fill = item["fill"]
            sig = item["signal"]
            cohort_id = item["cohort_id"]
            stream = item["stream"]
            observed_best_ask = item.get("observed_best_ask")
            book_observed_at = item.get("book_observed_at") or fill.get("book_observed_at")
            market_protocol = str(item.get("market_protocol") or "v1")
            neg_risk = bool(item.get("neg_risk") or False)

            paper_order_id = str(fill.get("order_id") or f"paper:{signal_id}")
            auth = fill.get("allocation_authorization") or {}
            rank = int(auth.get("rank") or 1)
            market_id = str(fill.get("market_id") or sig.get("market_id") or "")
            outcome = str(fill.get("outcome") or sig.get("outcome") or "").upper()
            underlying = str(fill.get("underlying") or sig.get("underlying") or "").upper()
            venue_mkt = fill.get("venue_market") or {}
            condition_id = str(venue_mkt.get("condition_id") or "")
            outcome_ids = venue_mkt.get("outcome_ids") or {}
            token_id = str(outcome_ids.get(outcome) or "")
            paper_benchmark = compute_paper_benchmark_metrics(fill)
            approved_stake_usd = to_decimal(auth.get("approved_stake_usd", 0), "approved_stake_usd")
            desired_stake_usd = (scale_dec * approved_stake_usd).quantize(USD_MICRO, rounding=ROUND_DOWN)
            fee_quote = fill.get("fee_quote") or {}
            fee_rate = to_decimal(fee_quote.get("rate", 0.075), "fee_rate")

            eff_clock_proj = fill.get("clock_projection") if fill.get("clock_projection") is not None else state.get("clock")
            try:
                live_detected_cal = validate_and_project_utc(
                    detected_local,
                    fill.get("clock_projection"),
                    state_clock=state.get("clock"),
                )
                deadline_info = derive_effective_order_deadline(
                    cohort_id=cohort_id,
                    sig=sig,
                    fill=fill,
                    signal_id=signal_id,
                    state_last_cohort=state.get("last_cohort"),
                    hard_cap_seconds=self.hard_deadline_after_cohort_seconds,
                )
            except (ValueError, TypeError) as time_exc:
                fallback_ws = str((sig or {}).get("window_start") or venue_mkt.get("window_start") or cohort_id or "")
                fallback_we = str((sig or {}).get("window_end") or venue_mkt.get("window_end") or "")
                if not self.audit.claim_signal(
                    signal_id=signal_id,
                    cohort_id=str(cohort_id or ""),
                    stream=str(stream or ""),
                    paper_order_id=paper_order_id,
                    claimed_at_utc=detected_local,
                ):
                    continue
                slot_idx = len(ordered_slots)
                ordered_slots.append(None)
                now_iso = format_utc_iso(detected_local)
                dec_fallback = str((sig or {}).get("decision_at") or auth.get("planned_at") or now_iso)
                pfill_fallback = str(fill.get("fill_at") or now_iso)
                live_exec = {
                    "live_scale_factor": float(scale_dec),
                    "order_type": "FAK",
                    "limit_price": self._default_limit_price(),
                    "requested_stake_usd": float(desired_stake_usd),
                    "requested_quantity": "0.00",
                    "clob_order_id": None,
                    "clob_status": "ABORTED_INVALID_DEADLINE",
                    "error_code": f"INVALID_TIMING_OR_DEADLINE:{time_exc}",
                    "filled_quantity": "0.00",
                    "avg_fill_price": 0.0,
                    "cost_usdc": 0.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 0.0,
                    "unfilled_stake_usdc": float(desired_stake_usd),
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                }
                order_record = {
                    "cohort_id": cohort_id,
                    "stream": stream,
                    "rank": rank,
                    "signal_id": signal_id,
                    "paper_order_id": paper_order_id,
                    "market_id": market_id,
                    "condition_id": condition_id,
                    "token_id": token_id,
                    "outcome": outcome,
                    "market_protocol": market_protocol,
                    "neg_risk": neg_risk,
                    "timing": {
                        "decision_at": dec_fallback,
                        "paper_fill_at": pfill_fallback,
                        "live_detected_at": now_iso,
                        "live_signed_at": now_iso,
                        "live_http_sent_at": now_iso,
                        "live_http_ack_at": now_iso,
                        "detection_lag_ms": 0.0,
                        "clob_rtt_ms": 0.0,
                    },
                    "paper_benchmark": paper_benchmark,
                    "live_execution": live_exec,
                    "diff_vs_paper": compute_diff_vs_paper(paper_benchmark, live_exec, paper_status="OPEN"),
                }
                self.audit.record_order_execution(
                    order_record,
                    underlying=underlying,
                    window_start=fallback_ws,
                    window_end=fallback_we,
                )
                ordered_slots[slot_idx] = order_record
                deferred_snapshots.append((str(cohort_id or ""), detected_local))
                continue

            window_start = format_utc_iso(deadline_info["window_start_dt"])
            window_end = format_utc_iso(deadline_info["window_end_dt"])
            decision_at_str = format_utc_iso(deadline_info["decision_dt"])
            paper_fill_at_str = format_utc_iso(deadline_info["fill_dt"])
            valid_until_str = deadline_info["effective_valid_iso"]
            detection_lag_ms = round(
                max(0.0, (live_detected_cal - deadline_info["fill_dt"]).total_seconds() * 1000.0),
                2,
            )

            free_usdc, allowance_usdc = self.clob.query_wallet_balance_and_allowance(
                refresh_network=False,
                market_protocol=market_protocol,
                neg_risk=neg_risk,
            )

            # Check initial hard deadline `valid_until` before DB reservation
            within_deadline, _ = check_hard_deadline(live_detected_cal, valid_until_str)
            if not within_deadline:
                    if not self.audit.claim_signal(
                        signal_id=signal_id,
                        cohort_id=cohort_id,
                        stream=stream,
                        paper_order_id=paper_order_id,
                        claimed_at_utc=live_detected_cal,
                    ):
                        continue
                    slot_idx = len(ordered_slots)
                    ordered_slots.append(None)
                    now_iso = format_utc_iso(live_detected_cal)
                    live_exec = {
                        "live_scale_factor": float(scale_dec),
                        "order_type": "FAK",
                        "limit_price": self._default_limit_price(),
                        "requested_stake_usd": float(desired_stake_usd),
                        "requested_quantity": "0.00",
                        "clob_order_id": None,
                        "clob_status": "ABORTED_PAST_VALID_UNTIL",
                        "error_code": f"HARD_DEADLINE_EXCEEDED:live_detected_at={now_iso}>=valid_until={valid_until_str}",
                        "filled_quantity": "0.00",
                        "avg_fill_price": 0.0,
                        "cost_usdc": 0.0,
                        "fee_usdc": 0.0,
                        "total_spent_usdc": 0.0,
                        "unfilled_stake_usdc": float(desired_stake_usd),
                        "polygon_tx_hashes": [],
                        "onchain_confirmed": False,
                        "onchain_block_number": None,
                    }
                    order_record = {
                        "cohort_id": cohort_id,
                        "stream": stream,
                        "rank": rank,
                        "signal_id": signal_id,
                        "paper_order_id": paper_order_id,
                        "market_id": market_id,
                        "condition_id": condition_id,
                        "token_id": token_id,
                        "outcome": outcome,
                        "timing": {
                            "decision_at": decision_at_str,
                            "paper_fill_at": paper_fill_at_str,
                            "live_detected_at": now_iso,
                            "live_signed_at": now_iso,
                            "live_http_sent_at": now_iso,
                            "live_http_ack_at": now_iso,
                            "detection_lag_ms": detection_lag_ms,
                            "clob_rtt_ms": 0.0,
                        },
                        "paper_benchmark": paper_benchmark,
                        "live_execution": live_exec,
                        "diff_vs_paper": compute_diff_vs_paper(paper_benchmark, live_exec, paper_status="OPEN"),
                    }
                    self.audit.record_order_execution(
                        order_record,
                        underlying=underlying,
                        window_start=window_start,
                        window_end=window_end,
                    )
                    ordered_slots[slot_idx] = order_record
                    deferred_snapshots.append((cohort_id, live_detected_cal))
                    continue

            # [P1-6], [P1-7], Design Gap #1: Atomically claim signal and reserve risk quota under BEGIN IMMEDIATE
            stream_min_map = paper_caps.get("stream_min_buy_price") or {}
            effective_min_buy_price = float(
                stream_min_map.get(stream, min_buy_price) if isinstance(stream_min_map, Mapping) else min_buy_price
            )
            stream_max_map = paper_caps.get("stream_max_order_usd") or {}
            effective_stream_max_order = float(
                min(paper_caps["max_order_usd"], float(stream_max_map[stream]))
                if isinstance(stream_max_map, Mapping) and stream in stream_max_map
                else paper_caps["max_order_usd"]
            )
            effective_paper_caps = {**paper_caps, "max_order_usd": effective_stream_max_order}

            eff_approved_stake = approved_stake_usd
            eff_scale_dec = scale_dec
            eff_cohort_budget = paper_cohort_budget
            eff_daily_loss_cap = daily_loss_cap
            auto_sz = self.resolve_auto_equity_sizing(
                onchain_free_usdc=free_usdc,
                balance_verified=bool(self.clob.balance_verified),
                stream=stream,
                scale_dec=scale_dec,
                now_utc=live_detected_cal,
            )
            if auto_sz is not None:
                eff_approved_stake = auto_sz["target_stake_dec"]
                eff_scale_dec = Decimal("1.0")
                desired_stake_usd = eff_approved_stake
                effective_paper_caps = {**effective_paper_caps, **auto_sz["caps_override"]}
                eff_cohort_budget = auto_sz["cohort_budget_usd"]
                eff_daily_loss_cap = auto_sz["daily_loss_cap_dec"]

            reservation = self.audit.claim_and_reserve_signal(
                signal_id=signal_id,
                cohort_id=cohort_id,
                stream=stream,
                rank=rank,
                paper_order_id=paper_order_id,
                market_id=market_id,
                condition_id=condition_id,
                token_id=token_id,
                outcome=outcome,
                underlying=underlying,
                window_start=window_start,
                window_end=window_end,
                claimed_at_utc=live_detected_cal,
                approved_stake_usd=eff_approved_stake,
                live_scale_factor=eff_scale_dec,
                paper_caps=effective_paper_caps,
                proxy_wallet_address=self.clob.funder_address,
                onchain_free_usdc_balance=free_usdc,
                clob_allowance_usdc=allowance_usdc,
                balance_verified=bool(self.clob.balance_verified),
                cohort_single_order_cap_usd=eff_cohort_budget,
                max_daily_realized_loss_usd=eff_daily_loss_cap,
                max_consecutive_losses=max_consec_losses,
                cooldown_minutes_after_consecutive_losses=cooldown_mins,
                market_protocol=market_protocol,
                neg_risk=neg_risk,
                timing={
                    "decision_at": decision_at_str,
                    "paper_fill_at": paper_fill_at_str,
                    "live_detected_at": format_utc_iso(live_detected_cal),
                    "detection_lag_ms": detection_lag_ms,
                },
                paper_benchmark=paper_benchmark,
                fee_rate=float(fee_rate),
                valid_until=valid_until_str,
                hard_deadline_seconds=self.hard_deadline_after_cohort_seconds,
                calibrated_now_fn=lambda _p=eff_clock_proj: validate_and_project_utc(
                    detected_local + timedelta(seconds=max(0.0, time.perf_counter() - detected_mono)),
                    _p,
                ),
            )

            if not reservation.get("claimed"):
                continue

            slot_idx = len(ordered_slots)
            ordered_slots.append(None)

            if reservation.get("aborted"):
                reason = str(reservation.get("abort_status") or "ABORTED_RISK_GUARD")
                now_iso = format_utc_iso(live_detected_cal)
                live_exec = {
                    "live_scale_factor": float(eff_scale_dec),
                    "order_type": "FAK",
                    "limit_price": self._default_limit_price(),
                    "requested_stake_usd": float(desired_stake_usd),
                    "requested_quantity": "0.00",
                    "clob_order_id": None,
                    "clob_status": reason,
                    "error_code": str(reservation.get("abort_error") or reason),
                    "filled_quantity": "0.00",
                    "avg_fill_price": 0.0,
                    "cost_usdc": 0.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 0.0,
                    "unfilled_stake_usdc": float(desired_stake_usd),
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                }
                order_record = {
                    "cohort_id": cohort_id,
                    "stream": stream,
                    "rank": rank,
                    "signal_id": signal_id,
                    "paper_order_id": paper_order_id,
                    "market_id": market_id,
                    "condition_id": condition_id,
                    "token_id": token_id,
                    "outcome": outcome,
                    "market_protocol": market_protocol,
                    "neg_risk": neg_risk,
                    "timing": {
                        "decision_at": decision_at_str,
                        "paper_fill_at": paper_fill_at_str,
                        "live_detected_at": now_iso,
                        "live_signed_at": now_iso,
                        "live_http_sent_at": now_iso,
                        "live_http_ack_at": now_iso,
                        "detection_lag_ms": detection_lag_ms,
                        "clob_rtt_ms": 0.0,
                    },
                    "paper_benchmark": paper_benchmark,
                    "live_execution": live_exec,
                    "diff_vs_paper": compute_diff_vs_paper(paper_benchmark, live_exec, paper_status="OPEN"),
                }
                self.audit.record_order_execution(
                    order_record,
                    underlying=underlying,
                    window_start=window_start,
                    window_end=window_end,
                )
                ordered_slots[slot_idx] = order_record
                deferred_snapshots.append((cohort_id, live_detected_cal))
                continue

            quota_cap = to_decimal(reservation["reserved_stake_usd"], "reserved_stake_usd")
            pre_send_state = {"invoked": False}

            def _pre_send_persist(
                local_oid: str | None,
                sizing_info: Mapping[str, Any],
                signed_iso: str,
                sent_iso: str,
                _state_box: dict[str, bool] = pre_send_state,
                _sig_id: str = signal_id,
                _cohort: str = cohort_id,
                _stream: str = stream,
                _rank: int = rank,
                _p_oid: str = paper_order_id,
                _mid: str = market_id,
                _cid: str = condition_id,
                _tid: str = token_id,
                _out: str = outcome,
                _proto: str = market_protocol,
                _neg: bool = neg_risk,
                _dec_at: str = decision_at_str,
                _pfill_at: str = paper_fill_at_str,
                _det_at: str = format_utc_iso(live_detected_cal),
                _det_lag: float = detection_lag_ms,
                _pbench: Mapping[str, Any] = paper_benchmark,
                _fee_r: float = float(fee_rate),
            ) -> None:
                _state_box["invoked"] = True
                draft = {
                    "cohort_id": _cohort,
                    "stream": _stream,
                    "rank": _rank,
                    "signal_id": _sig_id,
                    "paper_order_id": _p_oid,
                    "market_id": _mid,
                    "condition_id": _cid,
                    "token_id": _tid,
                    "outcome": _out,
                    "market_protocol": _proto,
                    "neg_risk": _neg,
                    "decision_at": _dec_at,
                    "paper_fill_at": _pfill_at,
                    "live_detected_at": _det_at,
                    "live_signed_at": signed_iso,
                    "live_http_sent_at": sent_iso,
                    "detection_lag_ms": _det_lag,
                    "fee_rate": _fee_r,
                    "free_usdc_before_order": sizing_info.get("free_usdc_before_order"),
                    "auth_balance_seq_before_order": sizing_info.get("auth_balance_seq_before_order"),
                    "local_spent_before_order": sizing_info.get("local_spent_before_order"),
                    "paper_benchmark": dict(_pbench),
                    "live_execution": {
                        "live_scale_factor": sizing_info.get("live_scale_factor"),
                        "order_type": "FAK",
                        "limit_price": sizing_info.get("limit_price"),
                        "requested_stake_usd": sizing_info.get("requested_stake_usd"),
                        "requested_quantity": sizing_info.get("requested_quantity"),
                        "clob_order_id": local_oid,
                        "clob_status": "ORDER_STATE_UNKNOWN",
                        "fee_rate": _fee_r,
                        "free_usdc_before_order": sizing_info.get("free_usdc_before_order"),
                        "auth_balance_seq_before_order": sizing_info.get("auth_balance_seq_before_order"),
                        "local_spent_before_order": sizing_info.get("local_spent_before_order"),
                    },
                }
                self.audit.record_order_pre_send(
                    signal_id=_sig_id,
                    clob_order_id=local_oid,
                    order_record_draft=draft,
                )

            try:
                exec_bundle = self.clob.sign_and_execute_fak(
                    token_id=token_id,
                    approved_stake_usd=eff_approved_stake,
                    live_scale_factor=eff_scale_dec,
                    fee_rate=fee_rate,
                    clock_projection=eff_clock_proj,
                    quota_cap_usd=quota_cap,
                    paper_benchmark=paper_benchmark,
                    market_protocol=market_protocol,
                    neg_risk=neg_risk,
                    valid_until=valid_until_str,
                    window_start=window_start,
                    hard_deadline_seconds=self.hard_deadline_after_cohort_seconds,
                    observed_best_ask=observed_best_ask,
                    book_observed_at=book_observed_at,
                    min_buy_price=effective_min_buy_price,
                    max_buy_price=max_buy_price,
                    reference_now_utc=detected_local,
                    reference_mono=detected_mono,
                    pre_send_callback=_pre_send_persist,
                    defer_tx_receipt_check=has_multiple_candidates,
                )
            except Exception as pre_exc:
                if not pre_send_state["invoked"]:
                    # [P1-R4-1]: Pre-send failure — order was NEVER transmitted over HTTP!
                    now_abort_iso = format_utc_iso(live_detected_cal)
                    exec_bundle = {
                        "live_signed_at": now_abort_iso,
                        "live_http_sent_at": now_abort_iso,
                        "live_http_ack_at": now_abort_iso,
                        "clob_rtt_ms": 0.0,
                        "live_execution": {
                            "live_scale_factor": float(eff_scale_dec),
                            "order_type": "FAK",
                            "limit_price": self._default_limit_price(),
                            "requested_stake_usd": float(quota_cap),
                            "requested_quantity": "0.00",
                            "clob_order_id": None,
                            "clob_status": "ABORTED_PRE_SEND_FAILED",
                            "error_code": f"PRE_SEND_EXCEPTION:{type(pre_exc).__name__}:{str(pre_exc)[:160]}",
                            "filled_quantity": "0.00",
                            "avg_fill_price": 0.0,
                            "cost_usdc": 0.0,
                            "fee_usdc": 0.0,
                            "total_spent_usdc": 0.0,
                            "unfilled_stake_usdc": float(quota_cap),
                            "polygon_tx_hashes": [],
                            "onchain_confirmed": False,
                            "onchain_block_number": None,
                            "fee_rate": float(fee_rate),
                        },
                    }
                else:
                    raise

            item_ctx: dict[str, Any] = {
                "slot_idx": slot_idx,
                "signal_id": signal_id,
                "cohort_id": cohort_id,
                "stream": stream,
                "rank": rank,
                "paper_order_id": paper_order_id,
                "market_id": market_id,
                "condition_id": condition_id,
                "token_id": token_id,
                "outcome": outcome,
                "underlying": underlying,
                "window_start": window_start,
                "window_end": window_end,
                "market_protocol": market_protocol,
                "neg_risk": neg_risk,
                "fee_rate": fee_rate,
                "approved_stake_usd": eff_approved_stake,
                "eff_scale_dec": eff_scale_dec,
                "quota_cap": quota_cap,
                "clock_proj": eff_clock_proj,
                "valid_until_str": valid_until_str,
                "decision_at_str": decision_at_str,
                "paper_fill_at_str": paper_fill_at_str,
                "live_detected_cal": live_detected_cal,
                "detection_lag_ms": detection_lag_ms,
                "paper_benchmark": paper_benchmark,
                "effective_min_buy_price": effective_min_buy_price,
                "pre_send_state": pre_send_state,
                "pre_send_persist": _pre_send_persist,
                "exec_bundle": exec_bundle,
                "retry_count": 0,
                "retry_notes": [],
            }

            can_schedule_retry = False
            if (
                self.clob.mode == "live"
                and self.fak_no_match_max_retries > 0
                and valid_until_str
                and self._is_retryable_fak_no_match(exec_bundle["live_execution"])
            ):
                try:
                    valid_until_dt = parse_utc_timestamp(valid_until_str)
                    ack_dt = parse_utc_timestamp(exec_bundle["live_http_ack_at"])
                    if (valid_until_dt - ack_dt).total_seconds() >= (
                        self.fak_no_match_retry_interval_seconds + self.fak_no_match_min_remaining_seconds
                    ):
                        can_schedule_retry = True
                except ValueError:
                    can_schedule_retry = False

            if can_schedule_retry:
                # Keep `quota_cap` reserved in SQLite (`RESERVED`, `0` fill) while lower-rank candidates run Attempt 0
                self.audit.reset_reservation_for_retry(
                    signal_id=signal_id,
                    reserved_stake_usd=float(quota_cap),
                    updated_at_utc=exec_bundle["live_http_ack_at"],
                )
                retry_queue.append(item_ctx)
            else:
                _finalize_exec_bundle(item_ctx, exec_bundle, retry_count=0)

        # Phase 2: Bounded multi-look FAK retry inside `valid_until` (`T+20s`) for `FAK_KILLED_NO_MATCH`
        virtual_wait_accum = 0.0
        for retry_idx in range(1, self.fak_no_match_max_retries + 1):
            if not retry_queue:
                break

            eligible_this_round: list[dict[str, Any]] = []
            for r_ctx in retry_queue:
                now_chk_local = detected_local + timedelta(
                    seconds=max(virtual_wait_accum, time.perf_counter() - detected_mono)
                )
                now_chk_cal = validate_and_project_utc(now_chk_local, r_ctx["clock_proj"])
                valid_dt = parse_utc_timestamp(r_ctx["valid_until_str"])
                if (valid_dt - now_chk_cal).total_seconds() >= (
                    self.fak_no_match_retry_interval_seconds + self.fak_no_match_min_remaining_seconds
                ):
                    eligible_this_round.append(r_ctx)
                else:
                    _finalize_exec_bundle(r_ctx, r_ctx["exec_bundle"], retry_count=r_ctx["retry_count"])

            if not eligible_this_round:
                retry_queue = []
                break

            if self._wait_retry_interval(self.fak_no_match_retry_interval_seconds):
                for r_ctx in eligible_this_round:
                    _finalize_exec_bundle(r_ctx, r_ctx["exec_bundle"], retry_count=r_ctx["retry_count"])
                retry_queue = []
                break
            virtual_wait_accum += self.fak_no_match_retry_interval_seconds

            still_retrying: list[dict[str, Any]] = []
            for r_ctx in eligible_this_round:
                retry_ref_mono = time.perf_counter()
                retry_ref_utc = detected_local + timedelta(
                    seconds=max(virtual_wait_accum, retry_ref_mono - detected_mono)
                )
                now_retry_cal = validate_and_project_utc(retry_ref_utc, r_ctx["clock_proj"])
                valid_dt = parse_utc_timestamp(r_ctx["valid_until_str"])
                if (valid_dt - now_retry_cal).total_seconds() < self.fak_no_match_min_remaining_seconds:
                    _finalize_exec_bundle(r_ctx, r_ctx["exec_bundle"], retry_count=r_ctx["retry_count"])
                    continue

                tid_raw = str(r_ctx["token_id"]).strip()
                tid_norm = str(int(tid_raw, 16)) if tid_raw.startswith(("0x", "0X")) else tid_raw
                live_ask, live_book_err = self.clob._fetch_live_best_ask(
                    tid_norm,
                    tid_raw,
                    now_dt=now_retry_cal,
                    max_book_age_seconds=self.clob.max_book_age_seconds,
                )
                min_p_dec = to_decimal(r_ctx["effective_min_buy_price"], "effective_min_buy_price")
                prev_sizing = r_ctx["exec_bundle"].get("sizing") or {}
                limit_p_dec = to_decimal(
                    prev_sizing.get("limit_price_dec")
                    or r_ctx["exec_bundle"]["live_execution"].get("limit_price")
                    or 0.51,
                    "limit_price",
                )

                if live_ask is not None and Decimal("0") < live_ask < min_p_dec:
                    now_abort_iso = format_utc_iso(now_retry_cal)
                    prev_exec = r_ctx["exec_bundle"]["live_execution"]
                    abort_bundle = {
                        "sizing": prev_sizing,
                        "live_signed_at": now_abort_iso,
                        "live_http_sent_at": now_abort_iso,
                        "live_http_ack_at": now_abort_iso,
                        "clob_rtt_ms": 0.0,
                        "live_execution": {
                            "live_scale_factor": prev_exec.get("live_scale_factor", float(scale_dec)),
                            "order_type": "FAK",
                            "limit_price": float(limit_p_dec),
                            "requested_stake_usd": prev_exec.get("requested_stake_usd", float(r_ctx["quota_cap"])),
                            "requested_quantity": prev_exec.get("requested_quantity", "0.00"),
                            "clob_order_id": None,
                            "clob_status": "ABORTED_BELOW_MIN_BUY_PRICE",
                            "error_code": (
                                f"LIVE_ORDERBOOK_BEST_ASK_BELOW_MIN_BUY_PRICE_ON_RETRY:"
                                f"{live_ask}<{min_p_dec}:retry={retry_idx}"
                            ),
                            "filled_quantity": "0.00",
                            "avg_fill_price": 0.0,
                            "cost_usdc": 0.0,
                            "fee_usdc": 0.0,
                            "total_spent_usdc": 0.0,
                            "unfilled_stake_usdc": prev_exec.get("unfilled_stake_usdc", float(r_ctx["quota_cap"])),
                            "polygon_tx_hashes": [],
                            "onchain_confirmed": False,
                            "onchain_block_number": None,
                            "fee_rate": float(r_ctx["fee_rate"]),
                        },
                    }
                    _finalize_exec_bundle(r_ctx, abort_bundle, retry_count=retry_idx)
                    continue

                if live_ask is None or live_ask <= Decimal("0") or live_ask > limit_p_dec:
                    r_ctx["retry_notes"].append(
                        f"r{retry_idx}:ask={live_ask if live_ask is not None else (live_book_err or 'NA')}>{limit_p_dec}"
                    )
                    if retry_idx < self.fak_no_match_max_retries:
                        still_retrying.append(r_ctx)
                    else:
                        _finalize_exec_bundle(r_ctx, r_ctx["exec_bundle"], retry_count=r_ctx["retry_count"])
                    continue

                # Executable ask replenished inside `[min_buy_price, limit_price]`: sign & send fresh FAK order
                r_ctx["pre_send_state"]["invoked"] = False
                try:
                    retry_bundle = self.clob.sign_and_execute_fak(
                        token_id=tid_raw,
                        approved_stake_usd=r_ctx["approved_stake_usd"],
                        live_scale_factor=r_ctx["eff_scale_dec"],
                        fee_rate=r_ctx["fee_rate"],
                        clock_projection=r_ctx["clock_proj"],
                        quota_cap_usd=r_ctx["quota_cap"],
                        paper_benchmark=r_ctx["paper_benchmark"],
                        market_protocol=r_ctx["market_protocol"],
                        neg_risk=r_ctx["neg_risk"],
                        valid_until=r_ctx["valid_until_str"],
                        window_start=r_ctx["window_start"],
                        hard_deadline_seconds=self.hard_deadline_after_cohort_seconds,
                        observed_best_ask=live_ask,
                        book_observed_at=format_utc_iso(now_retry_cal),
                        min_buy_price=r_ctx["effective_min_buy_price"],
                        max_buy_price=max_buy_price,
                        reference_now_utc=retry_ref_utc,
                        reference_mono=retry_ref_mono,
                        pre_send_callback=r_ctx["pre_send_persist"],
                        skip_preflight_book_fetch=True,
                        defer_tx_receipt_check=(len(eligible_this_round) > 1),
                    )
                except Exception as retry_exc:
                    if not r_ctx["pre_send_state"]["invoked"]:
                        r_ctx["retry_notes"].append(f"r{retry_idx}:pre_send_exc={type(retry_exc).__name__}")
                        _finalize_exec_bundle(r_ctx, r_ctx["exec_bundle"], retry_count=r_ctx["retry_count"])
                        continue
                    raise

                r_ctx["exec_bundle"] = retry_bundle
                r_ctx["retry_count"] = retry_idx
                if (
                    retry_idx < self.fak_no_match_max_retries
                    and self._is_retryable_fak_no_match(retry_bundle["live_execution"])
                ):
                    r_ctx["retry_notes"].append(f"r{retry_idx}:no_match@{live_ask}")
                    self.audit.reset_reservation_for_retry(
                        signal_id=r_ctx["signal_id"],
                        reserved_stake_usd=float(r_ctx["quota_cap"]),
                        updated_at_utc=retry_bundle["live_http_ack_at"],
                    )
                    still_retrying.append(r_ctx)
                else:
                    _finalize_exec_bundle(r_ctx, retry_bundle, retry_count=retry_idx)

            retry_queue = still_retrying

        for leftover_ctx in retry_queue:
            _finalize_exec_bundle(
                leftover_ctx,
                leftover_ctx["exec_bundle"],
                retry_count=leftover_ctx["retry_count"],
            )

        # Phase 3: Persist quota snapshots after all cohort orders have completed dispatch
        for snap_cohort_id, snap_dt in deferred_snapshots:
            self._write_quota_snapshot(
                trigger_event="POST_ORDER_EXECUTION",
                cohort_id=snap_cohort_id,
                now_utc=snap_dt,
                pipeline_state=state,
            )

        return [rec for rec in ordered_slots if rec is not None]

    def reconcile_pending_orders(self, force: bool = False) -> list[dict[str, Any]]:
        """Reconcile any `ORDER_STATE_UNKNOWN` / `DELAYED` / `RESERVED` orders with Polymarket CLOB ([P1-2], [P1-3], [P1-A], [P1-R4-1], [P2-R4-5])."""
        if not force and self.observer.is_in_critical_window():
            return []
        pending = self.audit.get_orders_needing_reconciliation()
        if not pending:
            return []

        reconciled: list[dict[str, Any]] = []
        for row in pending:
            # [P2-R4-5]: Preserve legitimate `0.0` fee_rate (never use `or 0.075` which turns 0.0 into 0.075!)
            raw_fr = row.get("fee_rate")
            fee_r = to_decimal(0.075 if raw_fr is None else raw_fr, "fee_rate")
            res_exec = self.clob.reconcile_order(row, fee_rate=fee_r)
            if not isinstance(res_exec, Mapping):
                continue
            now_iso = format_utc_iso(datetime.now(timezone.utc))
            saved_bench = row.get("paper_benchmark")
            if isinstance(saved_bench, Mapping) and saved_bench.get("approved_stake_usd") is not None:
                paper_bench = dict(saved_bench)
            else:
                paper_bench = {
                    "approved_stake_usd": float(row.get("requested_stake_usd") or 0.0),
                    "paper_filled_quantity": float(row.get("requested_quantity") or 0.0),
                    "avg_fill_price": float(row.get("limit_price") or self._default_limit_price()),
                    "cost_usd": 0.0,
                    "fee_usd": 0.0,
                    "total_spent_usd": float(row.get("requested_stake_usd") or 0.0),
                }
            rec_iso = str(row.get("recorded_at_utc") or now_iso)
            decision_at = str(row.get("decision_at") or row.get("window_start") or rec_iso)
            paper_fill_at = str(row.get("paper_fill_at") or decision_at)
            live_detected_at = str(row.get("live_detected_at") or rec_iso)
            live_signed_at = str(row.get("live_signed_at") or live_detected_at)
            live_http_sent_at = str(row.get("live_http_sent_at") or live_signed_at)

            order_record = {
                "cohort_id": str(row.get("cohort_id") or ""),
                "stream": str(row.get("stream") or ""),
                "rank": int(row.get("rank") or 1),
                "signal_id": str(row["signal_id"]),
                "paper_order_id": str(row.get("paper_order_id") or f"paper:{row['signal_id']}"),
                "market_id": str(row.get("market_id") or ""),
                "condition_id": str(row.get("condition_id") or ""),
                "token_id": str(row.get("token_id") or ""),
                "outcome": str(row.get("outcome") or ""),
                "market_protocol": str(row.get("market_protocol") or "v1"),
                "neg_risk": bool(row.get("neg_risk") or False),
                "fee_rate": float(fee_r),
                "timing": {
                    "decision_at": decision_at,
                    "paper_fill_at": paper_fill_at,
                    "live_detected_at": live_detected_at,
                    "live_signed_at": live_signed_at,
                    "live_http_sent_at": live_http_sent_at,
                    "live_http_ack_at": now_iso,
                    "detection_lag_ms": float(row.get("detection_lag_ms") or 0.0),
                    "clob_rtt_ms": float(row.get("clob_rtt_ms") or 0.0),
                },
                "paper_benchmark": paper_bench,
                "live_execution": dict(res_exec),
                "diff_vs_paper": compute_diff_vs_paper(paper_bench, res_exec, paper_status="OPEN"),
            }
            # [P2-R6-2]: Synchronize/deduct confirmed order spend from `self.clob._cached_free_usdc`
            # BEFORE releasing the reserved stake in `record_order_execution`!
            self.clob.apply_reconciled_order_spend(row, res_exec)
            self.audit.record_order_execution(
                order_record,
                underlying=str(row.get("underlying") or ""),
                window_start=row.get("window_start"),
                window_end=row.get("window_end"),
            )
            reconciled.append(order_record)
        return reconciled

    def _write_quota_snapshot(
        self,
        trigger_event: str,
        cohort_id: str | None,
        now_utc: datetime | None = None,
        pipeline_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build and persist a Spec Section 5.2 `live_quota_snapshots.jsonl` entry."""
        stamp = now_utc or datetime.now(timezone.utc)
        try:
            paper_caps = self._paper_risk_caps()
        except PolicyVerificationError as pol_exc:
            logger.warning("Skipping quota snapshot while policy is unverified: %s", pol_exc)
            return {
                "timestamp_utc": format_utc_iso(stamp),
                "trigger_event": trigger_event,
                "cohort_id": cohort_id,
                "policy_verified": False,
                "policy_error": str(pol_exc),
            }
        free_usdc, allowance_usdc = self.clob.query_wallet_balance_and_allowance(refresh_network=False)

        cohort_budget_usd = None
        if isinstance(pipeline_state, Mapping) and isinstance(pipeline_state.get("execution"), Mapping):
            c_obj = pipeline_state["execution"].get("cohort")
            if isinstance(c_obj, Mapping) and c_obj.get("budget_usd") is not None:
                cohort_budget_usd = c_obj["budget_usd"]

        live_quotas = self.audit.compute_live_quotas(
            now_utc=stamp,
            cohort_id=cohort_id,
            live_scale_factor=self.live_scale_factor,
            paper_caps=paper_caps,
            proxy_wallet_address=self.clob.funder_address,
            onchain_free_usdc_balance=free_usdc,
            clob_allowance_usdc=allowance_usdc,
            cohort_single_order_cap_usd=cohort_budget_usd,
        )
        status_payload = self.observer.fetch_status(allow_in_critical_window=False)
        paper_mirror = self.observer.extract_paper_mirror_snapshot(status_payload, pipeline_state=pipeline_state)

        snapshot = {
            "timestamp_utc": format_utc_iso(stamp),
            "trigger_event": trigger_event,
            "cohort_id": cohort_id,
            "live_wallet_and_quotas": live_quotas,
            "paper_account_mirror": paper_mirror,
        }
        self.audit.record_quota_snapshot(snapshot)
        return snapshot

    def _execute_and_record_settlement(
        self,
        market_id: str,
        condition_id: str,
        official_outcome: str,
        paper_payout_usd: float,
        paper_realized_pnl_usd: float,
        paper_settled_at: str | None,
        market_protocol: str | None = None,
        neg_risk: bool | None = None,
        payout_ratios: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        live_orders = self.audit.get_live_orders_for_market(market_id)
        if any(
            str(o.get("position_status") or "").upper() in ("RESERVED", "ORDER_STATE_UNKNOWN", "DELAYED", "SUBMITTING")
            for o in live_orders
        ):
            self.reconcile_pending_orders(force=True)
            live_orders = self.audit.get_live_orders_for_market(market_id)

        resolved_cond = (
            str(live_orders[0]["condition_id"])
            if live_orders and live_orders[0].get("condition_id")
            else condition_id
        )
        resolved_proto = str(
            market_protocol
            or (
                "v2"
                if any(str(o.get("market_protocol") or "").lower() == "v2" for o in live_orders)
                else (live_orders[0].get("market_protocol") if live_orders else "v1")
            )
            or "v1"
        ).strip().lower()
        resolved_neg_risk = bool(
            neg_risk
            if neg_risk is not None
            else any(bool(o.get("neg_risk")) for o in live_orders)
        )
        existing_settlement = self.audit.get_settlement_record(market_id)
        eff_payout_ratios: Mapping[str, Any] | None = None
        if payout_ratios is not None:
            if resolve_verified_payout_ratios(official_outcome, payout_ratios) is not None:
                eff_payout_ratios = payout_ratios
        else:
            if isinstance(existing_settlement, Mapping):
                existing_pr = existing_settlement.get("payout_ratios")
                if resolve_verified_payout_ratios(official_outcome, existing_pr) is not None:
                    eff_payout_ratios = existing_pr
            if eff_payout_ratios is None and str(official_outcome).upper().strip() not in ("UP", "DOWN"):
                fetched_pr = self.observer.fetch_official_market_payout_ratios(market_id, resolved_cond)
                if resolve_verified_payout_ratios(official_outcome, fetched_pr) is not None:
                    eff_payout_ratios = fetched_pr
        before_redeemed = float((existing_settlement or {}).get("redeemed_usdc_amount") or 0.0)
        was_pending_redemption = (
            isinstance(existing_settlement, Mapping)
            and str(existing_settlement.get("redeem_status") or "").upper() in ("REDEEM_PENDING", "REDEEM_SUBMIT_UNKNOWN")
            and bool(
                existing_settlement.get("redeem_tx_id")
                or existing_settlement.get("redeem_tx_hash")
                or existing_settlement.get("pending_batch_order_keys")
                or existing_settlement.get("redeem_request_id")
                or existing_settlement.get("submit_state")
            )
        )
        if was_pending_redemption and isinstance(existing_settlement, Mapping):
            submit_boot_id = (
                str(existing_settlement["submit_boot_id"])
                if existing_settlement.get("submit_boot_id")
                else None
            )
            submit_balance_seq = (
                int(existing_settlement["balance_snapshot_seq_at_submit"])
                if existing_settlement.get("balance_snapshot_seq_at_submit") is not None
                else None
            )
            submit_free_usdc = (
                float(existing_settlement["free_usdc_at_submit"])
                if existing_settlement.get("free_usdc_at_submit") is not None
                else None
            )
            submit_local_spent = (
                float(existing_settlement["local_spent_at_submit"])
                if existing_settlement.get("local_spent_at_submit") is not None
                else None
            )
            submit_ts_utc = str(
                existing_settlement.get("tx_submitted_at_utc")
                or existing_settlement.get("settled_at_utc")
                or existing_settlement.get("timestamp_utc")
                or format_utc_iso(datetime.now(timezone.utc))
            )
        else:
            submit_boot_id = self.clob._boot_id
            submit_balance_seq = self.clob._last_balance_snapshot_seq
            submit_free_usdc = self.clob._cached_free_usdc
            submit_local_spent = self.clob._cumulative_local_spent_usdc
            submit_ts_utc = format_utc_iso(datetime.now(timezone.utc))

        def _build_settlement_dict(res_dict: Mapping[str, Any], ts_iso: str) -> dict[str, Any]:
            return {
                "timestamp_utc": ts_iso,
                "market_id": market_id,
                "condition_id": resolved_cond,
                "official_outcome": official_outcome,
                "market_protocol": resolved_proto,
                "neg_risk": resolved_neg_risk,
                "paper_payout_usd": paper_payout_usd,
                "paper_realized_pnl_usd": paper_realized_pnl_usd,
                "live_expected_payout_usdc": res_dict["live_expected_payout_usdc"],
                "redeem_tx_id": res_dict.get("redeem_tx_id"),
                "redeem_tx_hash": res_dict.get("redeem_tx_hash"),
                "redeem_status": res_dict["redeem_status"],
                "submit_state": res_dict.get("submit_state"),
                "redeem_request_id": res_dict.get("redeem_request_id"),
                "redeem_target_to": res_dict.get("redeem_target_to"),
                "redeem_calldata_hex": res_dict.get("redeem_calldata_hex"),
                "redeemed_usdc_amount": res_dict["redeemed_usdc_amount"],
                "newly_redeemed_usdc_amount": res_dict.get("newly_redeemed_usdc_amount"),
                "redeemed_order_keys": res_dict.get("redeemed_order_keys"),
                "newly_redeemed_order_keys": res_dict.get("newly_redeemed_order_keys"),
                "pending_batch_order_keys": res_dict.get("pending_batch_order_keys"),
                "pending_batch_shares": res_dict.get("pending_batch_shares"),
                "pending_batch_payout_usdc": res_dict.get("pending_batch_payout_usdc"),
                "has_remaining_unredeemed": res_dict.get("has_remaining_unredeemed"),
                "remaining_unredeemed_order_keys": res_dict.get("remaining_unredeemed_order_keys"),
                "remaining_unredeemed_payout_usdc": res_dict.get("remaining_unredeemed_payout_usdc"),
                "redeem_latency_seconds": res_dict.get("redeem_latency_seconds"),
                "paper_settled_at": paper_settled_at,
                "live_order_count": len(live_orders),
                "submit_boot_id": submit_boot_id,
                "balance_snapshot_seq_at_submit": submit_balance_seq,
                "free_usdc_at_submit": submit_free_usdc,
                "local_spent_at_submit": submit_local_spent,
                "tx_submitted_at_utc": submit_ts_utc,
                "payout_ratios": res_dict.get("payout_ratios") if res_dict.get("payout_ratios") is not None else eff_payout_ratios,
                "error": res_dict.get("error"),
            }

        def _on_redeem_progress(progress_res: dict[str, Any]) -> None:
            progress_ts = format_utc_iso(datetime.now(timezone.utc))
            progress_rec = _build_settlement_dict(progress_res, progress_ts)
            self.audit.record_settlement_and_redeem(progress_rec, append_jsonl=False)

        redeem_res = self.redeemer.redeem_settled_market(
            market_id=market_id,
            condition_id=resolved_cond,
            official_outcome=official_outcome,
            live_orders=live_orders,
            market_protocol=resolved_proto,
            neg_risk=resolved_neg_risk,
            existing_settlement=existing_settlement,
            on_progress_callback=_on_redeem_progress,
            payout_ratios=eff_payout_ratios,
        )

        now_utc = datetime.now(timezone.utc)
        settlement_record = _build_settlement_dict(redeem_res, format_utc_iso(now_utc))
        self.audit.record_settlement_and_redeem(settlement_record)
        saved_settlement = self.audit.get_settlement_record(market_id)
        after_redeemed = before_redeemed
        if isinstance(saved_settlement, Mapping):
            settlement_record["redeem_status"] = str(saved_settlement.get("redeem_status") or settlement_record["redeem_status"])
            settlement_record["redeemed_usdc_amount"] = float(
                saved_settlement.get("redeemed_usdc_amount")
                if saved_settlement.get("redeemed_usdc_amount") is not None
                else settlement_record["redeemed_usdc_amount"]
            )
            after_redeemed = float(settlement_record["redeemed_usdc_amount"] or 0.0)
            settlement_record["live_realized_pnl_usdc"] = float(saved_settlement.get("live_realized_pnl_usdc") or 0.0)

        # [P1-4] / [P1-E] / [P1-R3-1] / [P2-R4-4] Only credit newly redeemed pUSD for the batch confirmed in this pass,
        # deduplicated against any wallet balance refresh and idempotent against replay in `audit_store`.
        raw_redeem_status = str(redeem_res.get("redeem_status") or "").upper()
        newly_credited = (
            float(redeem_res.get("newly_redeemed_usdc_amount"))
            if redeem_res.get("newly_redeemed_usdc_amount") is not None
            else (
                float(redeem_res.get("redeemed_usdc_amount") or 0.0)
                if raw_redeem_status in ("REDEEMED", "SHADOW_REDEEMED")
                else 0.0
            )
        )
        store_increased = (saved_settlement is None) or ((after_redeemed - before_redeemed) > 1e-9)
        if newly_credited > 0 and raw_redeem_status in ("REDEEMED", "SHADOW_REDEEMED") and store_increased:
            batch_keys_str = ",".join(
                sorted(str(k) for k in (redeem_res.get("newly_redeemed_order_keys") or redeem_res.get("redeemed_order_keys") or []))
            )
            batch_credit_key = str(
                redeem_res.get("redeem_tx_id")
                or redeem_res.get("redeem_tx_hash")
                or f"{market_id}:{batch_keys_str}:{newly_credited:.6f}"
            )
            self.clob.apply_confirmed_redemption_credit(
                newly_credited,
                batch_credit_key=batch_credit_key,
                was_pending_redemption=was_pending_redemption,
                submit_boot_id=submit_boot_id,
                balance_snapshot_seq_at_submit=submit_balance_seq,
                tx_submitted_at_utc=submit_ts_utc,
                free_usdc_at_submit=submit_free_usdc,
                local_spent_at_submit=submit_local_spent,
            )
        if live_orders:
            self._write_quota_snapshot(
                trigger_event="POST_SETTLEMENT_REDEEM",
                cohort_id=live_orders[-1].get("cohort_id"),
                now_utc=now_utc,
            )
        return settlement_record

    def poll_settlements_and_redeem(self, force: bool = False) -> list[dict[str, Any]]:
        """Check `/api/ledger` AND unresolved local live positions for official resolution and auto-redeem."""
        if not force and self.observer.is_in_critical_window():
            return []

        # [P1-A] / [P1-B] Always reconcile any pending/unknown orders before evaluating market settlements!
        self.reconcile_pending_orders(force=force)

        settled_records: list[dict[str, Any]] = []
        processed_markets_this_pass: set[str] = set()
        ledger_events = self.observer.fetch_ledger(allow_in_critical_window=force)

        market_condition_map: dict[str, str] = {}
        market_protocol_map: dict[str, str] = {}
        market_negrisk_map: dict[str, bool] = {}
        max_seq_seen: int | None = None
        for ev in ledger_events:
            if not isinstance(ev, Mapping):
                continue
            seq_val = ev.get("seq") or ev.get("id")
            if isinstance(seq_val, int):
                max_seq_seen = seq_val if max_seq_seen is None else max(max_seq_seen, seq_val)
            if ev.get("event_type") == "PAPER_IOC":
                pl = ev.get("payload")
                if isinstance(pl, Mapping):
                    mid = str(pl.get("market_id") or "")
                    vm = pl.get("venue_market")
                    if mid and isinstance(vm, Mapping):
                        if vm.get("condition_id"):
                            market_condition_map[mid] = str(vm["condition_id"])
                        if vm.get("protocol"):
                            market_protocol_map[mid] = str(vm["protocol"]).lower()
                        if vm.get("neg_risk") is not None:
                            market_negrisk_map[mid] = bool(vm["neg_risk"])

        for ev in reversed(ledger_events):
            if not isinstance(ev, Mapping) or ev.get("event_type") != "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER":
                continue
            payload = ev.get("payload")
            if not isinstance(payload, Mapping):
                continue
            market_id = str(payload.get("market_id") or "").strip()
            if not market_id or market_id in processed_markets_this_pass or self.audit.is_market_settled(market_id):
                continue

            res_obj = payload.get("resolution") if isinstance(payload.get("resolution"), Mapping) else {}
            official_outcome = str(res_obj.get("outcome") or "").upper().strip()
            if official_outcome not in ("UP", "DOWN", "VOID"):
                continue

            settlement_record = self._execute_and_record_settlement(
                market_id=market_id,
                condition_id=market_condition_map.get(market_id, ""),
                official_outcome=official_outcome,
                paper_payout_usd=float(payload.get("payout_usd") or 0.0),
                paper_realized_pnl_usd=float(payload.get("realized_pnl_usd") or 0.0),
                paper_settled_at=str(ev.get("utc")) if ev.get("utc") else None,
                market_protocol=market_protocol_map.get(market_id),
                neg_risk=market_negrisk_map.get(market_id),
                payout_ratios=extract_payout_ratios_from_payload(res_obj) or extract_payout_ratios_from_payload(payload),
            )
            processed_markets_this_pass.add(market_id)
            settled_records.append(settlement_record)

        if max_seq_seen is not None:
            self.audit.set_kv("last_ledger_seq", str(max_seq_seen))

        # [P1-B] / [P1-E]: Resume any pending/deferred settlements from `live_settlements`
        for pend in self.audit.get_pending_settlements():
            mid = str(pend.get("market_id") or "").strip()
            if not mid or mid in processed_markets_this_pass or self.audit.is_market_settled(mid):
                continue
            out = str(pend.get("official_outcome") or "").upper().strip()
            if out not in ("UP", "DOWN", "VOID"):
                continue
            settlement_record = self._execute_and_record_settlement(
                market_id=mid,
                condition_id=str(pend.get("condition_id") or market_condition_map.get(mid, "")),
                official_outcome=out,
                paper_payout_usd=float(pend.get("paper_payout_usd") or 0.0),
                paper_realized_pnl_usd=float(pend.get("paper_realized_pnl_usd") or 0.0),
                paper_settled_at=pend.get("paper_settled_at") or pend.get("settled_at_utc"),
                market_protocol=pend.get("market_protocol") or market_protocol_map.get(mid),
                neg_risk=pend.get("neg_risk") if pend.get("neg_risk") is not None else market_negrisk_map.get(mid),
                payout_ratios=pend.get("payout_ratios") if isinstance(pend.get("payout_ratios"), Mapping) else None,
            )
            processed_markets_this_pass.add(mid)
            settled_records.append(settlement_record)

        # Design Gap #3: Catch up any unresolved local live markets that aged out of /api/ledger during downtime
        now_utc = datetime.now(timezone.utc)
        unresolved = self.audit.get_unresolved_live_markets()
        for item in unresolved:
            mid = str(item.get("market_id") or "").strip()
            if not mid or mid in processed_markets_this_pass or self.audit.is_market_settled(mid):
                continue
            w_end = item.get("window_end")
            if w_end and not force:
                try:
                    if parse_utc_timestamp(w_end) > now_utc:
                        continue
                except ValueError:
                    pass
            fallback = self.observer.resolve_unsettled_market_fallback(
                market_id=mid,
                condition_id=str(item.get("condition_id") or ""),
                token_id=str(item.get("token_id") or ""),
                expected_outcome=str(item.get("outcome") or ""),
                allow_in_critical_window=force,
            )
            if isinstance(fallback, Mapping) and fallback.get("official_outcome") in ("UP", "DOWN", "VOID"):
                settlement_record = self._execute_and_record_settlement(
                    market_id=mid,
                    condition_id=str(fallback.get("condition_id") or item.get("condition_id") or ""),
                    official_outcome=str(fallback["official_outcome"]),
                    paper_payout_usd=float(fallback.get("paper_payout_usd") or 0.0),
                    paper_realized_pnl_usd=float(fallback.get("paper_realized_pnl_usd") or 0.0),
                    paper_settled_at=fallback.get("settled_at"),
                    market_protocol=item.get("market_protocol"),
                    neg_risk=bool(item.get("neg_risk")) if item.get("neg_risk") is not None else None,
                    payout_ratios=fallback.get("payout_ratios") if isinstance(fallback.get("payout_ratios"), Mapping) else None,
                )
                processed_markets_this_pass.add(mid)
                settled_records.append(settlement_record)

        return settled_records

    def prewarm_upcoming_tokens(self) -> int:
        """Pre-warm `tick_size` and `fee_rate_bps` for current & next window tokens outside `T+0s..T+20s`."""
        if self.observer.is_in_critical_window():
            return 0
        status = self.observer.fetch_status(allow_in_critical_window=False)
        if not isinstance(status, Mapping):
            return 0
        markets = status.get("markets")
        if not isinstance(markets, Mapping) or not isinstance(markets.get("streams"), Mapping):
            return 0

        token_ids: list[str] = []
        for stream_state in markets["streams"].values():
            if not isinstance(stream_state, Mapping):
                continue
            for slot in ("current", "next"):
                mkt = stream_state.get(slot)
                if isinstance(mkt, Mapping) and isinstance(mkt.get("outcome_ids"), Mapping):
                    for tid in mkt["outcome_ids"].values():
                        if tid:
                            token_ids.append(str(tid))
        if not token_ids:
            return 0
        self.clob.prewarm_token_metadata(token_ids, fees_enabled=True)
        return len(token_ids)

    def run_once(self, force_state_poll: bool = False) -> dict[str, Any]:
        """Run one non-blocking iteration of the live executor loop."""
        changed, state, detected_at = self.observer.poll_pipeline_state(force=force_state_poll)
        orders_executed: list[dict[str, Any]] = []
        if changed and isinstance(state, Mapping):
            orders_executed = self.process_pipeline_state(state, detected_at_local_utc=detected_at)

        now_mono = time.monotonic()
        settlements_processed: list[dict[str, Any]] = []
        orders_reconciled: list[dict[str, Any]] = []
        if not self.observer.is_in_critical_window():
            if (now_mono - self._last_reconcile_mono) >= self.reconcile_interval:
                self._last_reconcile_mono = now_mono
                orders_reconciled = self.reconcile_pending_orders(force=False)
            if (now_mono - self._last_settlement_poll_mono) >= self.settlement_poll_interval:
                self._last_settlement_poll_mono = now_mono
                settlements_processed = self.poll_settlements_and_redeem(force=False)
            if (now_mono - self._last_prewarm_mono) >= self.prewarm_interval:
                self._last_prewarm_mono = now_mono
                self.clob.query_wallet_balance_and_allowance(refresh_network=True)
                self.prewarm_upcoming_tokens()

        return {
            "state_changed": changed,
            "orders_executed": orders_executed,
            "orders_reconciled": orders_reconciled,
            "settlements_processed": settlements_processed,
        }

    def run_forever(self) -> None:
        """Start the 50ms continuous polling loop until `stop_event` is set."""
        logger.info(
            "Starting ARGON4 External Live Executor (mode=%s, wallet=%s/%s, scale=%.2f, poll=%.0fms)",
            self.clob.mode,
            self.clob.wallet_type,
            self.clob.funder_address,
            self.live_scale_factor,
            self.poll_interval_seconds * 1000.0,
        )
        # Initial balance check, order reconciliation, & startup quota snapshot
        self.clob.query_wallet_balance_and_allowance(refresh_network=True)
        self.reconcile_pending_orders(force=True)
        self.prewarm_upcoming_tokens()
        _, initial_state, _ = self.observer.poll_pipeline_state(force=True)
        self._write_quota_snapshot(
            trigger_event="STARTUP_SNAPSHOT",
            cohort_id=(initial_state or {}).get("last_cohort") if isinstance(initial_state, Mapping) else None,
            pipeline_state=initial_state if isinstance(initial_state, Mapping) else None,
        )

        try:
            while not self.stop_event.is_set():
                try:
                    self.run_once(force_state_poll=False)
                except Exception as exc:
                    logger.error("LiveExecutionEngine iteration error: %s: %s", type(exc).__name__, exc)
                self.stop_event.wait(self.poll_interval_seconds)
        finally:
            self.close()
            logger.info("ARGON4 External Live Executor stopped cleanly.")
