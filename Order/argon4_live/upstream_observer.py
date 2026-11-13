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
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
import requests

from .math_utils import parse_utc_timestamp
from .safety import assert_safe_read_path


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

    def read_policy(self, force: bool = False) -> dict[str, Any]:
        """Read `config/argon4_forward_policy.json` (read-only JSON file, hot-reloads on mtime/size change)."""
        assert_safe_read_path(self.policy_path, self.upstream_root)
        if not self.policy_path.is_file():
            if self._cached_policy is not None and not force:
                return self._cached_policy
            return {
                "initial_capital_usd": 1269.0,
                "max_tick_budget_usd": 380.0,
                "max_order_usd": 240.0,
                "max_open_risk_usd": 650.0,
                "max_underlying_open_risk_usd": 450.0,
                "max_buy_price": 0.515,
                "min_buy_price": 0.37,
            }
        try:
            st = os.stat(self.policy_path)
            if (
                not force
                and self._cached_policy is not None
                and st.st_mtime_ns == self._last_policy_mtime_ns
                and st.st_size == self._last_policy_size
            ):
                return self._cached_policy
            raw = self.policy_path.read_text(encoding="utf-8")
            self._cached_policy = json.loads(raw)
            self._last_policy_mtime_ns = st.st_mtime_ns
            self._last_policy_size = st.st_size
            return self._cached_policy
        except (OSError, json.JSONDecodeError):
            if self._cached_policy is not None:
                return self._cached_policy
            raise

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
                                return {
                                    "market_id": str(market_id),
                                    "condition_id": str(condition_id or item.get("condition_id") or ""),
                                    "official_outcome": out,
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
                    if isinstance(mkt, Mapping) and mkt.get("closed") is True:
                        uma_st = str(
                            mkt.get("umaResolutionStatus") or mkt.get("uma_resolution_status") or ""
                        ).strip().lower()
                        res_st = str(
                            mkt.get("resolutionStatus") or mkt.get("resolution_status") or ""
                        ).strip().lower()
                        non_final_statuses = {
                            "disputed",
                            "proposed",
                            "requested",
                            "pending",
                            "open",
                            "in_dispute",
                            "active",
                            "inactive",
                            "unresolved",
                        }
                        if (
                            uma_st not in non_final_statuses
                            and res_st not in non_final_statuses
                            and mkt.get("disputed") is not True
                        ):
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
                                        return {
                                            "market_id": str(market_id),
                                            "condition_id": str(condition_id),
                                            "official_outcome": winner,
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
                    if isinstance(gm, Mapping) and gm.get("closed") is True:
                        res_block = gm.get("resolution") if isinstance(gm.get("resolution"), Mapping) else {}
                        uma_st = str(
                            gm.get("umaResolutionStatus")
                            or gm.get("uma_resolution_status")
                            or res_block.get("umaResolutionStatus")
                            or res_block.get("uma_resolution_status")
                            or ""
                        ).strip().lower()
                        res_st = str(
                            gm.get("resolutionStatus")
                            or gm.get("resolution_status")
                            or res_block.get("status")
                            or res_block.get("resolutionStatus")
                            or ""
                        ).strip().lower()
                        non_final_statuses = {
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
                        }
                        final_statuses = {"resolved", "settled", "finalized"}
                        is_officially_resolved = (
                            uma_st not in non_final_statuses
                            and res_st not in non_final_statuses
                            and gm.get("disputed") is not True
                            and (
                                uma_st in final_statuses
                                or res_st in final_statuses
                                or gm.get("resolved") is True
                            )
                        )
                        if is_officially_resolved:
                            # 3a. Check explicit winner / resolution outcome field first
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
                                return {
                                    "market_id": str(market_id),
                                    "condition_id": str(condition_id or gm.get("conditionId") or ""),
                                    "official_outcome": winner,
                                    "paper_payout_usd": 0.0,
                                    "paper_realized_pnl_usd": 0.0,
                                    "settled_at": gm.get("resolvedAt") or gm.get("closedTime"),
                                    "source": "polymarket_gamma",
                                }

                            # 3b. Check exact settled payout vector [1, 0] or [0, 1] (NEVER price >= 0.99!)
                            raw_outcomes = gm.get("outcomes")
                            raw_prices = gm.get("outcomePrices")
                            outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
                            prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
                            if (
                                isinstance(outcomes, list)
                                and isinstance(prices, list)
                                and len(outcomes) == len(prices)
                                and len(outcomes) >= 2
                            ):
                                parsed_payouts: list[tuple[str, float]] = []
                                for out_label, pr_val in zip(outcomes, prices):
                                    try:
                                        parsed_payouts.append((str(out_label), float(pr_val)))
                                    except (ValueError, TypeError):
                                        parsed_payouts = []
                                        break
                                exact_winners = [lbl for lbl, p_num in parsed_payouts if p_num == 1.0]
                                exact_losers = [lbl for lbl, p_num in parsed_payouts if p_num == 0.0]
                                if len(exact_winners) == 1 and len(exact_losers) == len(parsed_payouts) - 1:
                                    norm = exact_winners[0].upper().strip()
                                    winner = "UP" if norm in ("UP", "YES") else ("DOWN" if norm in ("DOWN", "NO") else norm)
                                    if winner in ("UP", "DOWN", "VOID"):
                                        return {
                                            "market_id": str(market_id),
                                            "condition_id": str(condition_id or gm.get("conditionId") or ""),
                                            "official_outcome": winner,
                                            "paper_payout_usd": 0.0,
                                            "paper_realized_pnl_usd": 0.0,
                                            "settled_at": gm.get("resolvedAt") or gm.get("closedTime"),
                                            "source": "polymarket_gamma",
                                        }
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
