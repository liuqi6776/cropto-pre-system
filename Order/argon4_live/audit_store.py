"""Dual-storage audit persistence (`live_audit.sqlite3` + append-only `.jsonl` logs).

Implements Section 5 (5.1, 5.2, 5.3) and durable `signal_id` idempotency from the
ARGON4 External Live Executor Requirement Specification, plus all P1 audit fixes:
  - [P1-2] / [P1-3]: Tracks `RESERVED`, `ORDER_STATE_UNKNOWN`, and `DELAYED` orders as
    locked open risk until reconciled via CLOB `/data/order` and `/data/trades`.
  - [P1-4]: Only marks positions `WON_REDEEMED` and releases open risk when redemption
    reaches confirmed terminal state (`REDEEMED` / `SHADOW_REDEEMED`).
  - [P1-6]: Atomic `BEGIN IMMEDIATE` cross-process transaction (`claim_and_reserve_signal`)
    that checks all risk caps, wallet balance/allowance caps, daily loss stop, and
    consecutive-loss cooldown, and inserts a `RESERVED` row before releasing the DB lock.
  - Design Gaps #1, #2, #3: Tracks independent live daily realized PnL & consecutive-loss
    streaks, persistent settlement cursor (`runtime_kv`), and `get_unresolved_live_markets()`.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator, Mapping

from .math_utils import (
    HARD_WINDOW_DEADLINE_SECONDS,
    USD_MICRO,
    check_hard_deadline,
    format_utc_iso,
    parse_utc_timestamp,
    to_decimal,
)
from .redeemer import _norm_outcome_key, resolve_verified_payout_ratios
from .safety import assert_safe_write_path

# Statuses that lock capital and risk quota until resolved or reconciled
RISK_LOCKING_POSITION_STATUSES = (
    "OPEN",
    "WON_UNREDEEMED",
    "RESERVED",
    "ORDER_STATE_UNKNOWN",
    "DELAYED",
)

# Normalized CLOB execution statuses representing an actual fill when filled_quantity > 0
FILLED_CLOB_STATUSES = ("MATCHED", "FILLED", "SHADOW_MATCHED")

IN_FLIGHT_POSITION_STATUSES = (
    "RESERVED",
    "ORDER_STATE_UNKNOWN",
    "DELAYED",
)

TERMINAL_SETTLED_REDEEM_STATUSES = (
    "REDEEMED",
    "LOST_NO_REDEEM",
    "SHADOW_REDEEMED",
    "NO_LIVE_POSITION",
)


class AuditStore:
    """Thread-safe and cross-process atomic SQLite + JSONL audit store and risk coordinator."""

    def __init__(
        self,
        sqlite_path: str | Path,
        orders_jsonl_path: str | Path,
        quota_jsonl_path: str | Path,
        settlements_jsonl_path: str | Path,
        local_root: str | Path = r"C:\Users\liuqi\Order",
        upstream_root: str | Path = r"C:\Users\liuqi\crypto-predict_system",
    ) -> None:
        self.sqlite_path = assert_safe_write_path(sqlite_path, local_root, upstream_root)
        self.orders_jsonl_path = assert_safe_write_path(orders_jsonl_path, local_root, upstream_root)
        self.quota_jsonl_path = assert_safe_write_path(quota_jsonl_path, local_root, upstream_root)
        self.settlements_jsonl_path = assert_safe_write_path(settlements_jsonl_path, local_root, upstream_root)

        for p in (self.sqlite_path, self.orders_jsonl_path, self.quota_jsonl_path, self.settlements_jsonl_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._claimed_signals: set[str] = set()
        self._settled_markets: set[str] = set()
        self._init_db()

    @contextmanager
    def _connection(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.sqlite_path, timeout=30.0, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        try:
            con.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield con
                con.execute("COMMIT")
            except Exception:
                if con.in_transaction:
                    con.execute("ROLLBACK")
                raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._lock, self._connection(immediate=True) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_kv (
                    key TEXT PRIMARY KEY,
                    value_text TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_claims (
                    signal_id TEXT PRIMARY KEY,
                    cohort_id TEXT NOT NULL,
                    stream TEXT NOT NULL,
                    paper_order_id TEXT NOT NULL,
                    claimed_at_utc TEXT NOT NULL,
                    state TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS live_orders (
                    signal_id TEXT PRIMARY KEY,
                    cohort_id TEXT NOT NULL,
                    stream TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    paper_order_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    condition_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    window_start TEXT,
                    window_end TEXT,
                    underlying TEXT NOT NULL,
                    market_protocol TEXT NOT NULL DEFAULT 'v1',
                    neg_risk INTEGER NOT NULL DEFAULT 0,
                    clob_order_id TEXT,
                    clob_status TEXT NOT NULL,
                    position_status TEXT NOT NULL,
                    filled_quantity TEXT NOT NULL,
                    avg_fill_price REAL NOT NULL,
                    cost_usdc REAL NOT NULL,
                    fee_usdc REAL NOT NULL,
                    total_spent_usdc REAL NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_live_orders_market ON live_orders(market_id, position_status)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_live_orders_cohort ON live_orders(cohort_id)")
            order_cols = {
                str(r["name"]) for r in con.execute("PRAGMA table_info(live_orders)").fetchall()
            }
            if "market_protocol" not in order_cols:
                con.execute("ALTER TABLE live_orders ADD COLUMN market_protocol TEXT NOT NULL DEFAULT 'v1'")
            if "neg_risk" not in order_cols:
                con.execute("ALTER TABLE live_orders ADD COLUMN neg_risk INTEGER NOT NULL DEFAULT 0")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS live_quota_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    trigger_event TEXT NOT NULL,
                    cohort_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS live_settlements (
                    market_id TEXT PRIMARY KEY,
                    condition_id TEXT NOT NULL,
                    official_outcome TEXT NOT NULL,
                    paper_payout_usd REAL NOT NULL,
                    live_expected_payout_usdc REAL NOT NULL,
                    redeem_tx_hash TEXT,
                    redeem_status TEXT NOT NULL,
                    redeemed_usdc_amount REAL NOT NULL,
                    live_cost_usdc REAL NOT NULL DEFAULT 0.0,
                    live_realized_pnl_usdc REAL NOT NULL DEFAULT 0.0,
                    redeem_latency_seconds REAL,
                    settled_at_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            # Ensure new columns exist if migrating an older DB
            existing_cols = {
                str(r["name"]) for r in con.execute("PRAGMA table_info(live_settlements)").fetchall()
            }
            if "live_cost_usdc" not in existing_cols:
                con.execute("ALTER TABLE live_settlements ADD COLUMN live_cost_usdc REAL NOT NULL DEFAULT 0.0")
            if "live_realized_pnl_usdc" not in existing_cols:
                con.execute("ALTER TABLE live_settlements ADD COLUMN live_realized_pnl_usdc REAL NOT NULL DEFAULT 0.0")

            for row in con.execute("SELECT signal_id FROM signal_claims"):
                self._claimed_signals.add(str(row["signal_id"]))
            for row in con.execute("SELECT signal_id FROM live_orders"):
                self._claimed_signals.add(str(row["signal_id"]))
            active_market_rows = con.execute(
                """
                SELECT DISTINCT market_id FROM live_orders
                WHERE position_status IN ('OPEN', 'WON_UNREDEEMED', 'RESERVED', 'ORDER_STATE_UNKNOWN', 'DELAYED')
                """
            ).fetchall()
            markets_with_active_orders = {str(r["market_id"]) for r in active_market_rows}
            for row in con.execute(
                "SELECT market_id FROM live_settlements WHERE redeem_status IN ('REDEEMED', 'LOST_NO_REDEEM', 'SHADOW_REDEEMED', 'NO_LIVE_POSITION')"
            ):
                m_id = str(row["market_id"])
                if m_id not in markets_with_active_orders:
                    self._settled_markets.add(m_id)

    @staticmethod
    def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()

    def get_kv(self, key: str, default: str | None = None) -> str | None:
        with self._lock, self._connection(immediate=False) as con:
            row = con.execute("SELECT value_text FROM runtime_kv WHERE key=?", (str(key),)).fetchone()
            return str(row["value_text"]) if row else default

    def set_kv(self, key: str, value: str, now_utc: datetime | None = None) -> None:
        stamp = format_utc_iso(now_utc or datetime.now(timezone.utc))
        with self._lock, self._connection(immediate=True) as con:
            con.execute(
                """
                INSERT INTO runtime_kv(key, value_text, updated_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_text=excluded.value_text, updated_at_utc=excluded.updated_at_utc
                """,
                (str(key), str(value), stamp),
            )

    def is_signal_claimed(self, signal_id: str) -> bool:
        sig = str(signal_id).strip()
        with self._lock:
            if sig in self._claimed_signals:
                return True
            with self._connection(immediate=False) as con:
                row = con.execute("SELECT 1 FROM signal_claims WHERE signal_id=?", (sig,)).fetchone()
                if row is not None:
                    self._claimed_signals.add(sig)
                    return True
            return False

    def claim_signal(
        self,
        signal_id: str,
        cohort_id: str,
        stream: str,
        paper_order_id: str,
        claimed_at_utc: str | datetime | None = None,
    ) -> bool:
        """Atomically claim `signal_id` BEFORE signing/submitting a live order."""
        sig = str(signal_id).strip()
        if not sig:
            raise ValueError("signal_id required for idempotency claim")
        stamp = format_utc_iso(parse_utc_timestamp(claimed_at_utc) if claimed_at_utc else datetime.now(timezone.utc))
        with self._lock:
            if sig in self._claimed_signals:
                return False
            with self._connection(immediate=True) as con:
                cur = con.execute(
                    """
                    INSERT OR IGNORE INTO signal_claims(
                        signal_id, cohort_id, stream, paper_order_id, claimed_at_utc, state, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, 'CLAIMED', ?)
                    """,
                    (sig, str(cohort_id), str(stream), str(paper_order_id), stamp, stamp),
                )
                if cur.rowcount == 0:
                    self._claimed_signals.add(sig)
                    return False
            self._claimed_signals.add(sig)
            return True

    def _compute_quotas_in_tx(
        self,
        con: sqlite3.Connection,
        now_utc: datetime,
        cohort_id: str | None,
        live_scale_factor: float | Decimal,
        paper_caps: Mapping[str, Any],
        proxy_wallet_address: str,
        onchain_free_usdc_balance: float,
        clob_allowance_usdc: float,
        cohort_single_order_cap_usd: float | Decimal | None = None,
    ) -> dict[str, Any]:
        scale = to_decimal(live_scale_factor, "live_scale_factor")
        max_active_window_risk = (scale * to_decimal(paper_caps.get("max_active_window_risk_usd", 450.0))).quantize(
            USD_MICRO, rounding=ROUND_DOWN
        )
        max_open_risk = (scale * to_decimal(paper_caps.get("max_open_risk_usd", 1000.0))).quantize(
            USD_MICRO, rounding=ROUND_DOWN
        )
        max_underlying_active_risk = (
            scale * to_decimal(paper_caps.get("max_underlying_active_risk_usd", 300.0))
        ).quantize(USD_MICRO, rounding=ROUND_DOWN)
        max_underlying_risk = (scale * to_decimal(paper_caps.get("max_underlying_open_risk_usd", 500.0))).quantize(
            USD_MICRO, rounding=ROUND_DOWN
        )
        max_tick_budget = (scale * to_decimal(paper_caps.get("max_tick_budget_usd", 1200.0))).quantize(
            USD_MICRO, rounding=ROUND_DOWN
        )
        if cohort_single_order_cap_usd is not None:
            cohort_budget_cap = min(
                max_tick_budget,
                (scale * to_decimal(cohort_single_order_cap_usd)).quantize(USD_MICRO, rounding=ROUND_DOWN),
            )
        else:
            cohort_budget_cap = max_tick_budget

        placeholders = ",".join("?" for _ in RISK_LOCKING_POSITION_STATUSES)
        rows = con.execute(
            f"SELECT * FROM live_orders WHERE position_status IN ({placeholders})",
            RISK_LOCKING_POSITION_STATUSES,
        ).fetchall()

        active_locked = Decimal("0")
        expired_awaiting = Decimal("0")
        unredeemed_winning = Decimal("0")
        in_flight_reserved = Decimal("0")
        underlying_locked: dict[str, Decimal] = {
            "BTC": Decimal("0"),
            "ETH": Decimal("0"),
            "SOL": Decimal("0"),
            "BNB": Decimal("0"),
        }
        underlying_active_locked: dict[str, Decimal] = {
            "BTC": Decimal("0"),
            "ETH": Decimal("0"),
            "SOL": Decimal("0"),
            "BNB": Decimal("0"),
        }

        for row in rows:
            spent = to_decimal(row["total_spent_usdc"] or 0.0, "total_spent_usdc")
            coin = str(row["underlying"] or "").upper()
            pos_status = str(row["position_status"] or "OPEN")
            row_counts_active = False
            if pos_status == "WON_UNREDEEMED":
                unredeemed_winning += spent
            elif pos_status in IN_FLIGHT_POSITION_STATUSES:
                in_flight_reserved += spent
                active_locked += spent
                row_counts_active = True
            else:
                w_end_raw = row["window_end"]
                is_active = True
                if w_end_raw:
                    try:
                        is_active = parse_utc_timestamp(w_end_raw) > now_utc
                    except ValueError:
                        is_active = True
                if is_active:
                    active_locked += spent
                    row_counts_active = True
                else:
                    expired_awaiting += spent
            if coin in underlying_locked:
                underlying_locked[coin] += spent
            elif coin:
                underlying_locked[coin] = underlying_locked.get(coin, Decimal("0")) + spent
            if row_counts_active and coin:
                underlying_active_locked[coin] = underlying_active_locked.get(coin, Decimal("0")) + spent

        total_locked = (active_locked + expired_awaiting + unredeemed_winning).quantize(USD_MICRO, rounding=ROUND_DOWN)
        rem_total_open = max(Decimal("0"), (max_open_risk - total_locked).quantize(USD_MICRO, rounding=ROUND_DOWN))
        rem_active_window = max(
            Decimal("0"), (max_active_window_risk - active_locked).quantize(USD_MICRO, rounding=ROUND_DOWN)
        )

        if cohort_id:
            c_row = con.execute(
                "SELECT COALESCE(SUM(total_spent_usdc), 0.0) FROM live_orders WHERE cohort_id=? AND position_status != 'NO_FILL'",
                (str(cohort_id),),
            ).fetchone()
            cohort_spent = to_decimal(c_row[0] if c_row else 0, "cohort_spent_usdc").quantize(
                USD_MICRO, rounding=ROUND_DOWN
            )
        else:
            cohort_spent = Decimal("0")

        rem_cohort_budget = max(
            Decimal("0"), (cohort_budget_cap - cohort_spent).quantize(USD_MICRO, rounding=ROUND_DOWN)
        )

        rem_underlying = {
            coin: float(max(Decimal("0"), (max_underlying_risk - locked).quantize(USD_MICRO, rounding=ROUND_DOWN)))
            for coin, locked in underlying_locked.items()
        }
        rem_underlying_active = {
            coin: float(
                max(Decimal("0"), (max_underlying_active_risk - act_locked).quantize(USD_MICRO, rounding=ROUND_DOWN))
            )
            for coin, act_locked in underlying_active_locked.items()
        }

        free_bal_dec = max(
            Decimal("0"),
            to_decimal(onchain_free_usdc_balance, "onchain_free_usdc_balance").quantize(
                USD_MICRO, rounding=ROUND_DOWN
            ),
        )
        allowance_dec = max(
            Decimal("0"),
            to_decimal(clob_allowance_usdc, "clob_allowance_usdc").quantize(USD_MICRO, rounding=ROUND_DOWN),
        )

        return {
            "proxy_wallet_address": str(proxy_wallet_address or "0x0000000000000000000000000000000000000000"),
            "onchain_free_usdc_balance": float(free_bal_dec),
            "clob_allowance_usdc": float(allowance_dec),
            "in_flight_reserved_usdc": float(in_flight_reserved.quantize(USD_MICRO, rounding=ROUND_DOWN)),
            "active_window_locked_usdc": float(active_locked.quantize(USD_MICRO, rounding=ROUND_DOWN)),
            "expired_awaiting_resolution_usdc": float(expired_awaiting.quantize(USD_MICRO, rounding=ROUND_DOWN)),
            "resolved_unredeemed_winning_usdc": float(unredeemed_winning.quantize(USD_MICRO, rounding=ROUND_DOWN)),
            "total_locked_open_risk_usdc": float(total_locked),
            "remaining_total_open_risk_quota_usd": float(rem_total_open),
            "remaining_active_window_quota_usd": float(rem_active_window),
            "remaining_cohort_budget_quota_usd": float(rem_cohort_budget),
            "remaining_underlying_quota_usd": rem_underlying,
            "remaining_underlying_active_quota_usd": rem_underlying_active,
        }

    def _compute_loss_guard_in_tx(
        self,
        con: sqlite3.Connection,
        now_utc: datetime,
    ) -> dict[str, Any]:
        """Compute live daily realized PnL and consecutive loss streak from `live_settlements` (`[P1-R3-2]`, `[P1-R4-2]`).

        Enforces:
          1. Includes confirmed realized losses from markets that still have in-flight/unknown orders
             (`AWAITING_ORDER_RECONCILIATION` / `REDEEM_PENDING`) (`[P1-R3-2]`).
          2. Evaluates daily PnL across all PnL events (`pnl_history`) without pre-filtering out markets
             whose cumulative `live_realized_pnl_usdc` nets to zero (`[P1-R5-1]`).
          3. Distinguishes repeated queries (`pnl_delta == 0`) from newly confirmed losses (`pnl_delta < 0`):
             repeated queries preserve initial timestamps (`[P1-R4-2]`), whereas newly confirmed late losses
             advance ordering and consecutive-loss cooldown to the new loss confirmation timestamp (`[P1-R5-2]`).
        """
        day_prefix = now_utc.astimezone(timezone.utc).strftime("%Y-%m-%d")
        all_rows = con.execute(
            """
            SELECT redeem_status, live_realized_pnl_usdc, live_cost_usdc, settled_at_utc, payload_json
            FROM live_settlements
            ORDER BY settled_at_utc DESC
            """,
        ).fetchall()

        daily_pnl_sum = Decimal("0")
        parsed_rows: list[tuple[str, sqlite3.Row, dict[str, Any], list[dict[str, Any]]]] = []
        for s_row in all_rows:
            used_history = False
            parsed_pj: dict[str, Any] = {}
            hist_entries: list[dict[str, Any]] = []
            raw_pj = s_row["payload_json"]
            if isinstance(raw_pj, str) and raw_pj.strip():
                try:
                    loaded = json.loads(raw_pj)
                    if isinstance(loaded, dict):
                        parsed_pj = loaded
                        hist = parsed_pj.get("pnl_history")
                        if isinstance(hist, list) and hist:
                            used_history = True
                            for entry in hist:
                                if isinstance(entry, Mapping):
                                    entry_dict = dict(entry)
                                    hist_entries.append(entry_dict)
                                    c_at = str(entry_dict.get("confirmed_at_utc") or "")
                                    if c_at.startswith(day_prefix):
                                        daily_pnl_sum += to_decimal(
                                            entry_dict.get("pnl_delta_usdc") or 0.0, "pnl_delta_usdc"
                                        )
                except Exception:
                    used_history = False
            if not used_history:
                s_at = str(s_row["settled_at_utc"] or "")
                if s_at.startswith(day_prefix):
                    daily_pnl_sum += to_decimal(s_row["live_realized_pnl_usdc"] or 0.0, "live_realized_pnl_usdc")

            latest_event_ts = str(
                (hist_entries[-1].get("confirmed_at_utc") if hist_entries else None)
                or parsed_pj.get("last_loss_confirmed_at_utc")
                or parsed_pj.get("first_loss_confirmed_at_utc")
                or s_row["settled_at_utc"]
                or ""
            )
            parsed_rows.append((latest_event_ts, s_row, parsed_pj, hist_entries))

        daily_realized_pnl = daily_pnl_sum.quantize(USD_MICRO, rounding=ROUND_DOWN)

        parsed_rows.sort(key=lambda item: item[0], reverse=True)

        consecutive_losses = 0
        last_loss_settled_at: datetime | None = None
        for _, s_row, parsed_pj, hist_entries in parsed_rows[:50]:
            if float(s_row["live_cost_usdc"] or 0.0) <= 0.0 and not hist_entries:
                continue
            pnl = float(s_row["live_realized_pnl_usdc"] or 0.0)
            latest_delta = (
                float(hist_entries[-1].get("pnl_delta_usdc") or 0.0)
                if hist_entries
                else None
            )
            is_loss = (
                (latest_delta is not None and latest_delta < -1e-9)
                or (
                    latest_delta is None
                    and (pnl < -1e-9 or (str(s_row["redeem_status"]) == "LOST_NO_REDEEM" and pnl <= 0))
                )
            )
            is_win = (
                (latest_delta is not None and latest_delta > 1e-9)
                or (
                    latest_delta is None
                    and (pnl > 1e-9 or str(s_row["redeem_status"]) in ("REDEEMED", "SHADOW_REDEEMED"))
                )
            )
            if is_loss:
                consecutive_losses += 1
                if last_loss_settled_at is None:
                    loss_ts_str = str(
                        parsed_pj.get("last_loss_confirmed_at_utc")
                        or (hist_entries[-1].get("confirmed_at_utc") if hist_entries else None)
                        or parsed_pj.get("first_loss_confirmed_at_utc")
                        or s_row["settled_at_utc"]
                        or ""
                    )
                    if loss_ts_str:
                        try:
                            last_loss_settled_at = parse_utc_timestamp(loss_ts_str)
                        except ValueError:
                            pass
            elif is_win:
                break

        return {
            "daily_realized_pnl_usdc": float(daily_realized_pnl),
            "daily_realized_pnl_dec": daily_realized_pnl,
            "consecutive_losses": consecutive_losses,
            "last_loss_settled_at_utc": last_loss_settled_at,
        }

    def get_prior_days_realized_pnl_usdc(self, now_utc: datetime | None = None) -> Decimal:
        """Return cumulative realized PnL in USDC from settlements strictly prior to today UTC (`day_prefix`).

        Mirrors upstream `account.py` `day_start_equity = initial_capital + realized_pnl - today_pnl`
        so that daily loss limits scale dynamically as live account equity compounds.
        """
        stamp = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        day_prefix = stamp.strftime("%Y-%m-%d")
        if not self.sqlite_path.is_file():
            return Decimal("0")
        prior_sum = Decimal("0")
        with self._connection() as con:
            rows = con.execute(
                "SELECT live_realized_pnl_usdc, settled_at_utc, payload_json FROM live_settlements"
            ).fetchall()
        for s_row in rows:
            used_history = False
            raw_pj = s_row["payload_json"]
            if isinstance(raw_pj, str) and raw_pj.strip():
                try:
                    loaded = json.loads(raw_pj)
                    hist = loaded.get("pnl_history") if isinstance(loaded, dict) else None
                    if isinstance(hist, list) and hist:
                        used_history = True
                        for entry in hist:
                            if isinstance(entry, Mapping):
                                c_at = str(entry.get("confirmed_at_utc") or "")
                                if c_at and c_at[:10] < day_prefix:
                                    prior_sum += to_decimal(entry.get("pnl_delta_usdc") or 0.0, "pnl_delta_usdc")
                except Exception:
                    used_history = False
            if not used_history:
                s_at = str(s_row["settled_at_utc"] or "")
                if s_at and s_at[:10] < day_prefix:
                    prior_sum += to_decimal(s_row["live_realized_pnl_usdc"] or 0.0, "live_realized_pnl_usdc")
        return prior_sum.quantize(USD_MICRO, rounding=ROUND_DOWN)

    def get_live_equity_and_today_pnl(
        self,
        onchain_free_usdc_balance: float | Decimal,
        now_utc: datetime | None = None,
    ) -> dict[str, Decimal]:
        """Compute double-count-free true on-chain portfolio equity and today's realized PnL.

        ``true_live_equity = max(0, onchain_free - in_flight_reserved) + total_locked_open_risk``
        so open positions and in-flight reservations are neither lost nor double-counted.
        """
        stamp = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        free_dec = max(
            Decimal("0"),
            to_decimal(onchain_free_usdc_balance or 0.0, "onchain_free_usdc_balance").quantize(
                USD_MICRO, rounding=ROUND_DOWN
            ),
        )
        if not self.sqlite_path.is_file():
            return {
                "true_live_equity_dec": free_dec,
                "day_start_true_equity_dec": free_dec,
                "today_realized_pnl_dec": Decimal("0"),
            }
        placeholders = ",".join("?" for _ in RISK_LOCKING_POSITION_STATUSES)
        in_flight = Decimal("0")
        total_locked = Decimal("0")
        with self._connection() as con:
            rows = con.execute(
                f"SELECT position_status, total_spent_usdc FROM live_orders WHERE position_status IN ({placeholders})",
                RISK_LOCKING_POSITION_STATUSES,
            ).fetchall()
            for r in rows:
                spent = to_decimal(r["total_spent_usdc"] or 0.0, "total_spent_usdc")
                pos_st = str(r["position_status"] or "OPEN")
                if pos_st in IN_FLIGHT_POSITION_STATUSES:
                    in_flight += spent
                total_locked += spent
            loss_stats = self._compute_loss_guard_in_tx(con, stamp)
        avail_free = max(Decimal("0"), free_dec - in_flight)
        true_eq = (avail_free + total_locked).quantize(USD_MICRO, rounding=ROUND_DOWN)
        today_pnl = loss_stats["daily_realized_pnl_dec"]
        day_start_eq = max(Decimal("0"), (true_eq - today_pnl).quantize(USD_MICRO, rounding=ROUND_DOWN))
        return {
            "true_live_equity_dec": true_eq,
            "day_start_true_equity_dec": day_start_eq,
            "today_realized_pnl_dec": today_pnl,
        }

    def claim_and_reserve_signal(
        self,
        *,
        signal_id: str,
        cohort_id: str,
        stream: str,
        rank: int,
        paper_order_id: str,
        market_id: str,
        condition_id: str,
        token_id: str,
        outcome: str,
        underlying: str,
        window_start: str | None,
        window_end: str | None,
        approved_stake_usd: float | Decimal,
        live_scale_factor: float | Decimal,
        paper_caps: Mapping[str, Any],
        proxy_wallet_address: str,
        onchain_free_usdc_balance: float,
        clob_allowance_usdc: float,
        balance_verified: bool = True,
        require_verified_balance: bool = True,
        cohort_single_order_cap_usd: float | Decimal | None = None,
        max_daily_realized_loss_usd: float | Decimal | None = None,
        max_consecutive_losses: int = 0,
        cooldown_minutes_after_consecutive_losses: float = 0.0,
        claimed_at_utc: datetime | None = None,
        market_protocol: str = "v1",
        neg_risk: bool = False,
        timing: Mapping[str, Any] | None = None,
        paper_benchmark: Mapping[str, Any] | None = None,
        fee_rate: float = 0.0,
        valid_until: str | datetime | None = None,
        hard_deadline_seconds: float = HARD_WINDOW_DEADLINE_SECONDS,
        calibrated_now_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Atomically claim `signal_id`, check all risk & balance caps, and reserve budget (`[P1-6]`).

        Executes inside a single `BEGIN IMMEDIATE` SQLite transaction so concurrent processes
        or threads can never double-spend risk quotas or wallet balance.
        """
        sig = str(signal_id).strip()
        if not sig:
            raise ValueError("signal_id required for idempotency claim")
        now_dt = parse_utc_timestamp(claimed_at_utc) if claimed_at_utc else datetime.now(timezone.utc)
        now_iso = format_utc_iso(now_dt)
        coin = str(underlying).upper().strip()

        scale_dec = to_decimal(live_scale_factor, "live_scale_factor")
        approved_dec = to_decimal(approved_stake_usd, "approved_stake_usd")
        scaled_target = (scale_dec * approved_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
        live_max_order = (scale_dec * to_decimal(paper_caps.get("max_order_usd", 200.0))).quantize(
            USD_MICRO, rounding=ROUND_DOWN
        )

        with self._lock:
            if sig in self._claimed_signals:
                return {"claimed": False, "reason": "ALREADY_CLAIMED"}

            with self._connection(immediate=True) as con:
                cur = con.execute(
                    """
                    INSERT OR IGNORE INTO signal_claims(
                        signal_id, cohort_id, stream, paper_order_id, claimed_at_utc, state, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, 'CLAIMED', ?)
                    """,
                    (sig, str(cohort_id), str(stream), str(paper_order_id), now_iso, now_iso),
                )
                if cur.rowcount == 0:
                    self._claimed_signals.add(sig)
                    return {"claimed": False, "reason": "ALREADY_CLAIMED"}

                self._claimed_signals.add(sig)

                lock_now_dt = parse_utc_timestamp(calibrated_now_fn()) if callable(calibrated_now_fn) else now_dt
                lock_now_iso = format_utc_iso(lock_now_dt)

                if valid_until is not None:
                    try:
                        valid_dt = parse_utc_timestamp(valid_until)
                        cap_s = float(hard_deadline_seconds)
                        if not (0.0 < cap_s <= HARD_WINDOW_DEADLINE_SECONDS):
                            raise ValueError(f"hard_deadline_seconds={cap_s} out of bounds")
                        if window_start is not None and str(window_start).strip():
                            w_start_dt = parse_utc_timestamp(window_start)
                            if not (w_start_dt < valid_dt <= (w_start_dt + timedelta(seconds=cap_s))):
                                raise ValueError(
                                    f"valid_until={format_utc_iso(valid_dt)} outside (window_start={format_utc_iso(w_start_dt)}, +{cap_s}s]"
                                )
                    except (ValueError, TypeError) as dl_exc:
                        return {
                            "claimed": True,
                            "aborted": True,
                            "abort_status": "ABORTED_INVALID_DEADLINE",
                            "abort_error": f"INVALID_DEADLINE_AT_RESERVE:{dl_exc}",
                            "quota_cap_usd": Decimal("0"),
                            "reserved_stake_usd": Decimal("0"),
                        }
                    within_deadline, _ = check_hard_deadline(lock_now_dt, valid_dt)
                    if not within_deadline:
                        return {
                            "claimed": True,
                            "aborted": True,
                            "abort_status": "ABORTED_PAST_VALID_UNTIL",
                            "abort_error": (
                                f"HARD_DEADLINE_EXCEEDED_AT_RESERVE:now={lock_now_iso}>=valid_until={format_utc_iso(valid_dt)}"
                            ),
                            "quota_cap_usd": Decimal("0"),
                            "reserved_stake_usd": Decimal("0"),
                        }

                # 1. Check independent live daily loss stop & consecutive loss cooldown (Design Gap #1)
                # Explicit `None` means daily loss limit is disabled; any value `<= 0` means zero remaining loss budget -> STOP!
                loss_stats = self._compute_loss_guard_in_tx(con, lock_now_dt)
                if max_daily_realized_loss_usd is not None:
                    daily_loss_cap = to_decimal(max_daily_realized_loss_usd, "max_daily_realized_loss_usd")
                    if daily_loss_cap <= Decimal("0") or loss_stats["daily_realized_pnl_dec"] <= -daily_loss_cap:
                        return {
                            "claimed": True,
                            "aborted": True,
                            "abort_status": "ABORTED_DAILY_LOSS_STOP",
                            "abort_error": (
                                f"DAILY_LOSS_STOP:daily_loss_cap={float(daily_loss_cap)}<=0"
                                if daily_loss_cap <= Decimal("0")
                                else (
                                    f"DAILY_LOSS_STOP:daily_realized_pnl={loss_stats['daily_realized_pnl_usdc']}"
                                    f"<=-{float(daily_loss_cap)}"
                                )
                            ),
                            "quota_cap_usd": Decimal("0"),
                            "reserved_stake_usd": Decimal("0"),
                            "loss_stats": loss_stats,
                        }

                if max_consecutive_losses > 0 and loss_stats["consecutive_losses"] >= max_consecutive_losses:
                    last_loss_dt = loss_stats["last_loss_settled_at_utc"]
                    if cooldown_minutes_after_consecutive_losses <= 0 or (
                        last_loss_dt is not None
                        and lock_now_dt < (last_loss_dt + timedelta(minutes=float(cooldown_minutes_after_consecutive_losses)))
                    ):
                        return {
                            "claimed": True,
                            "aborted": True,
                            "abort_status": "ABORTED_CONSECUTIVE_LOSS_COOLDOWN",
                            "abort_error": (
                                f"CONSECUTIVE_LOSS_COOLDOWN:streak={loss_stats['consecutive_losses']}"
                                f">=max={max_consecutive_losses}"
                            ),
                            "quota_cap_usd": Decimal("0"),
                            "reserved_stake_usd": Decimal("0"),
                            "loss_stats": loss_stats,
                        }

                # 2. Check wallet balance & allowance verification gate ([P1-7])
                if require_verified_balance and not balance_verified:
                    return {
                        "claimed": True,
                        "aborted": True,
                        "abort_status": "ABORTED_UNVERIFIED_BALANCE_ALLOWANCE",
                        "abort_error": "WALLET_BALANCE_OR_ALLOWANCE_UNVERIFIED",
                        "quota_cap_usd": Decimal("0"),
                        "reserved_stake_usd": Decimal("0"),
                        "loss_stats": loss_stats,
                    }

                # 3. Compute live quotas including any in-flight RESERVED / ORDER_STATE_UNKNOWN / DELAYED orders
                quotas = self._compute_quotas_in_tx(
                    con=con,
                    now_utc=lock_now_dt,
                    cohort_id=cohort_id,
                    live_scale_factor=scale_dec,
                    paper_caps=paper_caps,
                    proxy_wallet_address=proxy_wallet_address,
                    onchain_free_usdc_balance=onchain_free_usdc_balance,
                    clob_allowance_usdc=clob_allowance_usdc,
                    cohort_single_order_cap_usd=cohort_single_order_cap_usd,
                )
                rem_active_window = to_decimal(quotas["remaining_active_window_quota_usd"])
                rem_underlying = to_decimal(
                    quotas["remaining_underlying_quota_usd"].get(
                        coin,
                        (scale_dec * to_decimal(paper_caps.get("max_underlying_open_risk_usd", 500.0))).quantize(
                            USD_MICRO, rounding=ROUND_DOWN
                        ),
                    )
                )
                rem_underlying_active = to_decimal(
                    quotas["remaining_underlying_active_quota_usd"].get(
                        coin,
                        (scale_dec * to_decimal(paper_caps.get("max_underlying_active_risk_usd", 300.0))).quantize(
                            USD_MICRO, rounding=ROUND_DOWN
                        ),
                    )
                )
                in_flight_dec = to_decimal(quotas["in_flight_reserved_usdc"])
                avail_free_usdc = max(
                    Decimal("0"),
                    to_decimal(quotas["onchain_free_usdc_balance"]) - in_flight_dec,
                )
                avail_allowance_usdc = max(
                    Decimal("0"),
                    to_decimal(quotas["clob_allowance_usdc"]) - in_flight_dec,
                )

                quota_cap = min(
                    live_max_order,
                    to_decimal(quotas["remaining_total_open_risk_quota_usd"]),
                    rem_active_window,
                    rem_underlying,
                    rem_underlying_active,
                    to_decimal(quotas["remaining_cohort_budget_quota_usd"]),
                    avail_free_usdc,
                    avail_allowance_usdc,
                ).quantize(USD_MICRO, rounding=ROUND_DOWN)

                reserved_stake = min(scaled_target, max(Decimal("0"), quota_cap)).quantize(
                    USD_MICRO, rounding=ROUND_DOWN
                )

                if reserved_stake > Decimal("0"):
                    timing_dict = dict(timing) if isinstance(timing, Mapping) else {}
                    bench_dict = dict(paper_benchmark) if isinstance(paper_benchmark, Mapping) else {}
                    proto_str = str(market_protocol or "v1").lower().strip() or "v1"
                    neg_flag = bool(neg_risk)
                    placeholder_payload = json.dumps(
                        {
                            "signal_id": sig,
                            "cohort_id": cohort_id,
                            "stream": stream,
                            "rank": int(rank),
                            "paper_order_id": str(paper_order_id),
                            "market_id": str(market_id),
                            "condition_id": str(condition_id),
                            "token_id": str(token_id),
                            "outcome": str(outcome).upper(),
                            "underlying": coin,
                            "market_protocol": proto_str,
                            "neg_risk": neg_flag,
                            "status": "RESERVED",
                            "reserved_stake_usd": float(reserved_stake),
                            "reserved_at_utc": lock_now_iso,
                            "free_usdc_at_reserve": float(onchain_free_usdc_balance),
                            "live_scale_factor": float(scale_dec),
                            "fee_rate": float(fee_rate),
                            "decision_at": timing_dict.get("decision_at") or window_start or now_iso,
                            "paper_fill_at": timing_dict.get("paper_fill_at") or timing_dict.get("decision_at") or now_iso,
                            "live_detected_at": timing_dict.get("live_detected_at") or now_iso,
                            "detection_lag_ms": float(timing_dict.get("detection_lag_ms") or 0.0),
                            "paper_benchmark": bench_dict,
                        },
                        ensure_ascii=False,
                    )
                    con.execute(
                        """
                        INSERT OR REPLACE INTO live_orders(
                            signal_id, cohort_id, stream, rank, paper_order_id, market_id,
                            condition_id, token_id, outcome, window_start, window_end, underlying,
                            market_protocol, neg_risk,
                            clob_order_id, clob_status, position_status, filled_quantity,
                            avg_fill_price, cost_usdc, fee_usdc, total_spent_usdc,
                            recorded_at_utc, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'RESERVED', 'RESERVED', '0.00', 0.0, 0.0, 0.0, ?, ?, ?)
                        """,
                        (
                            sig,
                            str(cohort_id),
                            str(stream),
                            int(rank),
                            str(paper_order_id),
                            str(market_id),
                            str(condition_id),
                            str(token_id),
                            str(outcome).upper(),
                            window_start,
                            window_end,
                            coin,
                            proto_str,
                            1 if neg_flag else 0,
                            float(reserved_stake),
                            now_iso,
                            placeholder_payload,
                        ),
                    )
                    con.execute(
                        "UPDATE signal_claims SET state='RESERVED', updated_at_utc=? WHERE signal_id=?",
                        (now_iso, sig),
                    )

                return {
                    "claimed": True,
                    "aborted": False,
                    "abort_status": None,
                    "abort_error": None,
                    "quota_cap_usd": quota_cap,
                    "reserved_stake_usd": reserved_stake,
                    "quotas": quotas,
                    "loss_stats": loss_stats,
                }

    def update_claim_state(self, signal_id: str, state: str, updated_at_utc: str | datetime | None = None) -> None:
        stamp = format_utc_iso(parse_utc_timestamp(updated_at_utc) if updated_at_utc else datetime.now(timezone.utc))
        with self._lock, self._connection(immediate=True) as con:
            con.execute(
                "UPDATE signal_claims SET state=?, updated_at_utc=? WHERE signal_id=?",
                (str(state), stamp, str(signal_id)),
            )

    def reset_reservation_for_retry(
        self,
        signal_id: str,
        reserved_stake_usd: float,
        updated_at_utc: str | datetime | None = None,
    ) -> None:
        """Reset a `0`-fill `FAK_KILLED_NO_MATCH` order row back to `RESERVED` while awaiting bounded multi-look retry."""
        sig = str(signal_id).strip()
        if not sig:
            return
        stamp = format_utc_iso(parse_utc_timestamp(updated_at_utc) if updated_at_utc else datetime.now(timezone.utc))
        with self._lock, self._connection(immediate=True) as con:
            con.execute(
                """
                UPDATE live_orders
                SET clob_order_id = NULL,
                    clob_status = 'RESERVED',
                    position_status = 'RESERVED',
                    filled_quantity = '0.00',
                    avg_fill_price = 0.0,
                    cost_usdc = 0.0,
                    fee_usdc = 0.0,
                    total_spent_usdc = ?,
                    recorded_at_utc = ?
                WHERE signal_id = ?
                """,
                (float(reserved_stake_usd), stamp, sig),
            )
            con.execute(
                "UPDATE signal_claims SET state='RESERVED', updated_at_utc=? WHERE signal_id=?",
                (stamp, sig),
            )

    def record_order_pre_send(
        self,
        *,
        signal_id: str,
        clob_order_id: str | None,
        order_record_draft: Mapping[str, Any],
    ) -> None:
        """Persist `clob_order_id` (`local_order_hash`) and draft metadata BEFORE `post_order` is sent (`[P1-A]`).

        Ensures that if the process crashes or loses power after the exchange receives the HTTP
        request but before `record_order_execution` runs, startup reconciliation can recover the
        order by `clob_order_id` and `token_id` with full timing/benchmark metadata.
        """
        sig = str(signal_id).strip()
        if not sig:
            return
        now_iso = format_utc_iso(datetime.now(timezone.utc))
        live_exec = order_record_draft.get("live_execution")
        live_exec_map = dict(live_exec) if isinstance(live_exec, Mapping) else {}
        if clob_order_id and not live_exec_map.get("clob_order_id"):
            live_exec_map["clob_order_id"] = str(clob_order_id)
        if not live_exec_map.get("clob_status"):
            live_exec_map["clob_status"] = "ORDER_STATE_UNKNOWN"
        draft_copy = dict(order_record_draft)
        draft_copy["live_execution"] = live_exec_map
        payload_text = json.dumps(draft_copy, ensure_ascii=False, allow_nan=False)

        with self._lock, self._connection(immediate=True) as con:
            con.execute(
                """
                UPDATE live_orders
                SET clob_order_id = COALESCE(?, clob_order_id),
                    clob_status = 'ORDER_STATE_UNKNOWN',
                    position_status = 'ORDER_STATE_UNKNOWN',
                    recorded_at_utc = ?,
                    payload_json = ?
                WHERE signal_id = ?
                """,
                (str(clob_order_id) if clob_order_id else None, now_iso, payload_text, sig),
            )
            con.execute(
                "UPDATE signal_claims SET state='SUBMITTING', updated_at_utc=? WHERE signal_id=?",
                (now_iso, sig),
            )

    @staticmethod
    def _hydrate_order_row(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        """Unpack `payload_json` fields into top-level keys with safe defaults (`[P1-A]`, `[P1-D]`)."""
        d = dict(row)
        payload: dict[str, Any] = {}
        raw_payload = d.get("payload_json")
        if isinstance(raw_payload, str) and raw_payload.strip():
            try:
                parsed = json.loads(raw_payload)
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = {}

        live_exec = payload.get("live_execution")
        live_exec_map = dict(live_exec) if isinstance(live_exec, Mapping) else {}
        paper_bench = payload.get("paper_benchmark")
        paper_bench_map = dict(paper_bench) if isinstance(paper_bench, Mapping) else {}

        rec_at = str(d.get("recorded_at_utc") or format_utc_iso(datetime.now(timezone.utc)))
        decision_at = str(payload.get("decision_at") or d.get("window_start") or rec_at)
        paper_fill_at = str(payload.get("paper_fill_at") or decision_at)
        live_detected_at = str(payload.get("live_detected_at") or payload.get("reserved_at_utc") or rec_at)
        live_signed_at = str(payload.get("live_signed_at") or live_detected_at)
        live_http_sent_at = str(payload.get("live_http_sent_at") or live_signed_at)
        live_http_ack_at = str(payload.get("live_http_ack_at") or rec_at)

        d["decision_at"] = decision_at
        d["paper_fill_at"] = paper_fill_at
        d["live_detected_at"] = live_detected_at
        d["live_signed_at"] = live_signed_at
        d["live_http_sent_at"] = live_http_sent_at
        d["live_http_ack_at"] = live_http_ack_at
        d["detection_lag_ms"] = float(payload.get("detection_lag_ms") or 0.0)
        d["clob_rtt_ms"] = float(payload.get("clob_rtt_ms") or 0.0)
        d["paper_benchmark"] = paper_bench_map
        d["live_execution"] = live_exec_map

        if not d.get("clob_order_id") and live_exec_map.get("clob_order_id"):
            d["clob_order_id"] = str(live_exec_map["clob_order_id"])

        d["requested_stake_usd"] = float(
            live_exec_map.get("requested_stake_usd")
            or payload.get("reserved_stake_usd")
            or d.get("total_spent_usdc")
            or 0.0
        )
        d["requested_quantity"] = str(
            live_exec_map.get("requested_quantity")
            or (d.get("filled_quantity") if str(d.get("filled_quantity", "0.00")) not in ("0", "0.0", "0.00") else None)
            or "0.00"
        )
        d["limit_price"] = float(
            live_exec_map.get("limit_price")
            or paper_bench_map.get("limit_price")
            or paper_bench_map.get("avg_fill_price")
            or d.get("avg_fill_price")
            or 0.50
        )
        d["live_scale_factor"] = float(
            live_exec_map.get("live_scale_factor")
            or payload.get("live_scale_factor")
            or 0.10
        )
        raw_fee_rate = None
        for fr_cand in (
            d.get("fee_rate"),
            live_exec_map.get("fee_rate"),
            payload.get("fee_rate"),
            paper_bench_map.get("fee_rate"),
        ):
            if fr_cand is not None:
                raw_fee_rate = fr_cand
                break
        if raw_fee_rate is None:
            p_fee = paper_bench_map.get("fee_usd")
            p_cost = paper_bench_map.get("cost_usd")
            p_price = paper_bench_map.get("avg_fill_price")
            if p_fee is not None and float(p_fee) == 0.0 and p_cost is not None and float(p_cost) > 0.0:
                raw_fee_rate = 0.0
            elif (
                p_fee is not None
                and p_cost is not None
                and p_price is not None
                and 0.0 < float(p_price) < 1.0
                and float(p_cost) > 0.0
            ):
                p_qty = float(paper_bench_map.get("paper_filled_quantity") or (float(p_cost) / float(p_price)))
                denom = p_qty * float(p_price) * (1.0 - float(p_price))
                raw_fee_rate = round(float(p_fee) / denom, 6) if denom > 0 else 0.075
            else:
                raw_fee_rate = 0.0
        d["fee_rate"] = float(raw_fee_rate)
        proto_val = str(
            d.get("market_protocol")
            or payload.get("market_protocol")
            or live_exec_map.get("market_protocol")
            or "v1"
        ).lower().strip()
        d["market_protocol"] = proto_val if proto_val else "v1"
        d["neg_risk"] = bool(
            d.get("neg_risk")
            or payload.get("neg_risk")
            or live_exec_map.get("neg_risk")
            or False
        )
        for b_key in (
            "free_usdc_at_reserve",
            "free_usdc_before_order",
            "auth_balance_seq_before_order",
            "local_spent_before_order",
        ):
            if d.get(b_key) is None:
                if payload.get(b_key) is not None:
                    d[b_key] = payload[b_key]
                elif live_exec_map.get(b_key) is not None:
                    d[b_key] = live_exec_map[b_key]
        return d

    @staticmethod
    def _derive_position_status_and_locked_spend(
        live_exec: Mapping[str, Any],
    ) -> tuple[str, float]:
        clob_status = str(live_exec.get("clob_status") or "UNKNOWN").upper()
        filled_qty = to_decimal(live_exec.get("filled_quantity", "0") or "0", "filled_quantity")
        spent_val = float(live_exec.get("total_spent_usdc") or 0.0)
        req_stake = float(live_exec.get("requested_stake_usd") or 0.0)

        # [P1-2]: Unknown submission outcome keeps full requested stake locked until reconciled!
        if clob_status in ("ORDER_STATE_UNKNOWN", "PENDING_RECONCILIATION", "SUBMITTING"):
            return "ORDER_STATE_UNKNOWN", max(spent_val, req_stake)

        # [P1-3]: Delayed order or unsettled fill keeps stake locked until reconciled!
        if clob_status in ("DELAYED", "LIVE", "MATCHED_PENDING_SETTLEMENT"):
            return "DELAYED", max(spent_val, req_stake)

        if clob_status in FILLED_CLOB_STATUSES and filled_qty > Decimal("0"):
            return "OPEN", spent_val

        return "NO_FILL", 0.0

    def record_order_execution(
        self,
        order_record: Mapping[str, Any],
        underlying: str,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> None:
        """Persist a per-order execution record to both `live_audit.sqlite3` and `live_orders.jsonl`."""
        sig = str(order_record["signal_id"])
        market_id = str(order_record["market_id"])
        live_exec = order_record["live_execution"]
        clob_status = str(live_exec.get("clob_status") or "UNKNOWN").upper()
        position_status, effective_spent_usdc = self._derive_position_status_and_locked_spend(live_exec)
        now_iso = format_utc_iso(datetime.now(timezone.utc))

        with self._lock:
            with self._connection(immediate=True) as con:
                existing_row = con.execute(
                    "SELECT market_protocol, neg_risk, window_start, window_end, payload_json FROM live_orders WHERE signal_id=?",
                    (sig,),
                ).fetchone()
                prior_fee_rate = None
                prior_balance_meta: dict[str, Any] = {}
                if existing_row and isinstance(existing_row["payload_json"], str) and existing_row["payload_json"].strip():
                    try:
                        p_old = json.loads(existing_row["payload_json"])
                        if isinstance(p_old, dict):
                            if p_old.get("fee_rate") is not None:
                                prior_fee_rate = float(p_old["fee_rate"])
                            elif isinstance(p_old.get("live_execution"), Mapping) and p_old["live_execution"].get("fee_rate") is not None:
                                prior_fee_rate = float(p_old["live_execution"]["fee_rate"])
                            for b_key in (
                                "free_usdc_at_reserve",
                                "free_usdc_before_order",
                                "auth_balance_seq_before_order",
                                "local_spent_before_order",
                            ):
                                if p_old.get(b_key) is not None:
                                    prior_balance_meta[b_key] = p_old[b_key]
                                elif isinstance(p_old.get("live_execution"), Mapping) and p_old["live_execution"].get(b_key) is not None:
                                    prior_balance_meta[b_key] = p_old["live_execution"][b_key]
                    except Exception:
                        prior_fee_rate = None
                proto_str = str(
                    order_record.get("market_protocol")
                    or live_exec.get("market_protocol")
                    or (existing_row["market_protocol"] if existing_row else None)
                    or "v1"
                ).lower().strip() or "v1"
                neg_flag = bool(
                    order_record.get("neg_risk")
                    if order_record.get("neg_risk") is not None
                    else (
                        live_exec.get("neg_risk")
                        if live_exec.get("neg_risk") is not None
                        else (bool(existing_row["neg_risk"]) if existing_row else False)
                    )
                )
                eff_w_start = window_start if window_start is not None else (existing_row["window_start"] if existing_row else None)
                eff_w_end = window_end if window_end is not None else (existing_row["window_end"] if existing_row else None)

                enriched_record = dict(order_record)
                enriched_record.setdefault("market_protocol", proto_str)
                enriched_record.setdefault("neg_risk", neg_flag)
                if enriched_record.get("fee_rate") is None:
                    if live_exec.get("fee_rate") is not None:
                        enriched_record["fee_rate"] = float(live_exec["fee_rate"])
                    elif prior_fee_rate is not None:
                        enriched_record["fee_rate"] = prior_fee_rate
                for b_key, b_val in prior_balance_meta.items():
                    if enriched_record.get(b_key) is None:
                        enriched_record[b_key] = b_val
                payload_text = json.dumps(enriched_record, ensure_ascii=False, allow_nan=False)

                con.execute(
                    """
                    INSERT OR REPLACE INTO live_orders(
                        signal_id, cohort_id, stream, rank, paper_order_id, market_id,
                        condition_id, token_id, outcome, window_start, window_end, underlying,
                        market_protocol, neg_risk,
                        clob_order_id, clob_status, position_status, filled_quantity,
                        avg_fill_price, cost_usdc, fee_usdc, total_spent_usdc,
                        recorded_at_utc, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sig,
                        str(order_record["cohort_id"]),
                        str(order_record["stream"]),
                        int(order_record["rank"]),
                        str(order_record["paper_order_id"]),
                        market_id,
                        str(order_record["condition_id"]),
                        str(order_record["token_id"]),
                        str(order_record["outcome"]).upper(),
                        eff_w_start,
                        eff_w_end,
                        str(underlying).upper(),
                        proto_str,
                        1 if neg_flag else 0,
                        live_exec.get("clob_order_id"),
                        clob_status,
                        position_status,
                        str(live_exec.get("filled_quantity", "0.00")),
                        float(live_exec.get("avg_fill_price") or 0.0),
                        float(live_exec.get("cost_usdc") or 0.0),
                        float(live_exec.get("fee_usdc") or 0.0),
                        effective_spent_usdc,
                        now_iso,
                        payload_text,
                    ),
                )
                con.execute(
                    "UPDATE signal_claims SET state=?, updated_at_utc=? WHERE signal_id=?",
                    (clob_status, now_iso, sig),
                )
                # [P1-B] & [P1-R3-1]: If an order transitions to OPEN (e.g. via late reconciliation) in a market
                # that already had a settlement record (even if an earlier batch was already REDEEMED!),
                # ensure the market is NOT marked settled so the newly confirmed position can be evaluated and redeemed!
                if position_status in ("OPEN", "WON_UNREDEEMED", *IN_FLIGHT_POSITION_STATUSES):
                    self._settled_markets.discard(market_id)
                    if position_status == "OPEN":
                        existing_settle = con.execute(
                            "SELECT redeem_status, payload_json FROM live_settlements WHERE market_id=?",
                            (market_id,),
                        ).fetchone()
                        if existing_settle is not None:
                            prev_st = str(existing_settle["redeem_status"] or "").upper()
                            if prev_st in ("REDEEMED", "SHADOW_REDEEMED"):
                                # Clear completed batch's tx_id so the next poll submits a new tx for the newly reconciled order
                                updated_payload_text = existing_settle["payload_json"]
                                if isinstance(updated_payload_text, str) and updated_payload_text.strip():
                                    try:
                                        p_dict = json.loads(updated_payload_text)
                                        if isinstance(p_dict, dict):
                                            p_dict["redeem_status"] = "REDEEM_PENDING"
                                            p_dict["redeem_tx_id"] = None
                                            p_dict["redeem_tx_hash"] = None
                                            p_dict["pending_batch_order_keys"] = []
                                            p_dict["pending_batch_payout_usdc"] = 0.0
                                            updated_payload_text = json.dumps(p_dict, ensure_ascii=False, allow_nan=False)
                                    except Exception:
                                        pass
                                con.execute(
                                    """
                                    UPDATE live_settlements
                                    SET redeem_status = 'REDEEM_PENDING',
                                        redeem_tx_hash = NULL,
                                        payload_json = ?
                                    WHERE market_id = ?
                                    """,
                                    (updated_payload_text, market_id),
                                )
                            elif prev_st in ("LOST_NO_REDEEM", "NO_LIVE_POSITION", "AWAITING_ORDER_RECONCILIATION"):
                                con.execute(
                                    """
                                    UPDATE live_settlements
                                    SET redeem_status = 'REDEEM_PENDING'
                                    WHERE market_id = ?
                                    """,
                                    (market_id,),
                                )
            self._claimed_signals.add(sig)
            self._append_jsonl(self.orders_jsonl_path, enriched_record)

    def get_orders_needing_reconciliation(self) -> list[dict[str, Any]]:
        """Return orders in `RESERVED`, `ORDER_STATE_UNKNOWN`, or `DELAYED` state (`[P1-2]`, `[P1-3]`, `[P1-A]`)."""
        placeholders = ",".join("?" for _ in IN_FLIGHT_POSITION_STATUSES)
        with self._lock, self._connection(immediate=False) as con:
            rows = con.execute(
                f"SELECT * FROM live_orders WHERE position_status IN ({placeholders}) ORDER BY recorded_at_utc ASC",
                IN_FLIGHT_POSITION_STATUSES,
            ).fetchall()
        return [self._hydrate_order_row(r) for r in rows]

    def get_open_live_orders(self, market_id: str | None = None) -> list[dict[str, Any]]:
        """Return all risk-locking live positions (`OPEN`, `WON_UNREDEEMED`, `RESERVED`, `ORDER_STATE_UNKNOWN`, `DELAYED`)."""
        placeholders = ",".join("?" for _ in RISK_LOCKING_POSITION_STATUSES)
        with self._lock, self._connection(immediate=False) as con:
            if market_id is not None:
                rows = con.execute(
                    f"SELECT * FROM live_orders WHERE market_id=? AND position_status IN ({placeholders})",
                    (str(market_id), *RISK_LOCKING_POSITION_STATUSES),
                ).fetchall()
            else:
                rows = con.execute(
                    f"SELECT * FROM live_orders WHERE position_status IN ({placeholders})",
                    RISK_LOCKING_POSITION_STATUSES,
                ).fetchall()
        return [self._hydrate_order_row(r) for r in rows]

    def get_unresolved_live_markets(self) -> list[dict[str, Any]]:
        """Return distinct markets that still have open, unredeemed, or in-flight live positions (Design Gap #3, `[P1-B]`)."""
        placeholders = ",".join("?" for _ in RISK_LOCKING_POSITION_STATUSES)
        with self._lock, self._connection(immediate=False) as con:
            rows = con.execute(
                f"""
                SELECT market_id, condition_id,
                       MIN(window_start) AS window_start,
                       MAX(window_end) AS window_end,
                       MAX(market_protocol) AS market_protocol,
                       MAX(neg_risk) AS neg_risk
                FROM live_orders
                WHERE position_status IN ({placeholders})
                GROUP BY market_id, condition_id
                ORDER BY window_end ASC
                """,
                RISK_LOCKING_POSITION_STATUSES,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_live_orders_for_market(self, market_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection(immediate=False) as con:
            rows = con.execute(
                "SELECT * FROM live_orders WHERE market_id=? ORDER BY rank ASC",
                (str(market_id),),
            ).fetchall()
        return [self._hydrate_order_row(r) for r in rows]

    def get_cohort_spent_usdc(self, cohort_id: str) -> Decimal:
        with self._lock, self._connection(immediate=False) as con:
            row = con.execute(
                "SELECT COALESCE(SUM(total_spent_usdc), 0.0) FROM live_orders WHERE cohort_id=? AND position_status != 'NO_FILL'",
                (str(cohort_id),),
            ).fetchone()
        return to_decimal(row[0] if row else 0, "cohort_spent_usdc").quantize(USD_MICRO, rounding=ROUND_DOWN)

    def compute_live_quotas(
        self,
        now_utc: datetime,
        cohort_id: str | None,
        live_scale_factor: float | Decimal,
        paper_caps: Mapping[str, Any],
        proxy_wallet_address: str,
        onchain_free_usdc_balance: float,
        clob_allowance_usdc: float,
        cohort_single_order_cap_usd: float | Decimal | None = None,
    ) -> dict[str, Any]:
        """Compute live wallet exposure and remaining risk quotas per Spec Section 5.2."""
        with self._lock, self._connection(immediate=False) as con:
            return self._compute_quotas_in_tx(
                con=con,
                now_utc=now_utc,
                cohort_id=cohort_id,
                live_scale_factor=live_scale_factor,
                paper_caps=paper_caps,
                proxy_wallet_address=proxy_wallet_address,
                onchain_free_usdc_balance=onchain_free_usdc_balance,
                clob_allowance_usdc=clob_allowance_usdc,
                cohort_single_order_cap_usd=cohort_single_order_cap_usd,
            )

    def record_quota_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Persist a remaining quota & capital snapshot to `live_audit.sqlite3` and `live_quota_snapshots.jsonl`."""
        ts = str(snapshot.get("timestamp_utc") or format_utc_iso(datetime.now(timezone.utc)))
        trigger = str(snapshot.get("trigger_event") or "SNAPSHOT")
        cohort_id = snapshot.get("cohort_id")
        payload_text = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
        with self._lock:
            with self._connection(immediate=True) as con:
                con.execute(
                    "INSERT INTO live_quota_snapshots(timestamp_utc, trigger_event, cohort_id, payload_json) VALUES (?, ?, ?, ?)",
                    (ts, trigger, str(cohort_id) if cohort_id else None, payload_text),
                )
            self._append_jsonl(self.quota_jsonl_path, snapshot)

    def is_market_settled(self, market_id: str) -> bool:
        m_id = str(market_id)
        with self._lock:
            if m_id not in self._settled_markets:
                return False
            placeholders = ",".join("?" for _ in RISK_LOCKING_POSITION_STATUSES)
            with self._connection(immediate=False) as con:
                active = con.execute(
                    f"SELECT 1 FROM live_orders WHERE market_id=? AND position_status IN ({placeholders}) LIMIT 1",
                    (m_id, *RISK_LOCKING_POSITION_STATUSES),
                ).fetchone()
                if active is not None:
                    self._settled_markets.discard(m_id)
                    return False
            return True

    def get_settlement_record(self, market_id: str) -> dict[str, Any] | None:
        """Return existing `live_settlements` record (unpacked with `payload_json`) for `market_id` (`[P1-E]`)."""
        with self._lock, self._connection(immediate=False) as con:
            row = con.execute(
                "SELECT * FROM live_settlements WHERE market_id=?",
                (str(market_id),),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        raw_payload = d.get("payload_json")
        if isinstance(raw_payload, str) and raw_payload.strip():
            try:
                parsed = json.loads(raw_payload)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        d.setdefault(k, v)
            except Exception:
                pass
        return d

    def get_pending_settlements(self) -> list[dict[str, Any]]:
        """Return settlements that are awaiting order reconciliation or redemption confirmation (`[P1-B]`, `[P1-E]`)."""
        placeholders = ",".join("?" for _ in TERMINAL_SETTLED_REDEEM_STATUSES)
        with self._lock, self._connection(immediate=False) as con:
            rows = con.execute(
                f"SELECT * FROM live_settlements WHERE redeem_status NOT IN ({placeholders}) ORDER BY settled_at_utc ASC",
                TERMINAL_SETTLED_REDEEM_STATUSES,
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            raw_payload = d.get("payload_json")
            if isinstance(raw_payload, str) and raw_payload.strip():
                try:
                    parsed = json.loads(raw_payload)
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            d.setdefault(k, v)
                except Exception:
                    pass
            out.append(d)
        return out

    def record_settlement_and_redeem(
        self,
        settlement_record: Mapping[str, Any],
        *,
        append_jsonl: bool = True,
    ) -> None:
        """Persist a settlement & redemption comparison record per Spec Section 5.3 (`[P1-4]`, `[P1-B]`, `[P1-R3-1]`, `[P1-R3-2]`, `[P1-R4-2]`, `[P1-R4-3]`, `[P2-R4-6]`).

        Crucially:
          1. If any order in `market_id` is still in `RESERVED`, `ORDER_STATE_UNKNOWN`, or `DELAYED`,
             the market MUST NOT be finalized as `LOST_NO_REDEEM` or added to `_settled_markets`.
          2. Only marks `WON_REDEEMED` for the specific orders included in the confirmed redemption batch
             (`redeemed_order_keys` / `newly_redeemed_order_keys`), leaving any newly arrived winning orders
             as `WON_UNREDEEMED` so they are redeemed in their own transaction (`[P1-R3-1]`, `[P2-R4-6]`).
          3. Immediately accounts for confirmed `LOST` orders in `live_realized_pnl_usdc` even while
             other orders in the same market are still in-flight or awaiting redemption (`[P1-R3-2]`).
          4. Preserves the first confirmation timestamp (`settled_at_utc`, `pnl_history`, and
             `first_loss_confirmed_at_utc`) so re-polling pending settlements can NEVER shift PnL
             across days or postpone consecutive-loss cooldown (`[P1-R4-2]`).
        """
        market_id = str(settlement_record["market_id"])
        official_outcome = str(settlement_record["official_outcome"]).upper()
        redeem_status = str(settlement_record["redeem_status"]).upper()
        incoming_ts = str(settlement_record.get("timestamp_utc") or format_utc_iso(datetime.now(timezone.utc)))
        batch_redeemed_amount = float(settlement_record.get("redeemed_usdc_amount") or 0.0)
        newly_redeemed_raw = settlement_record.get("newly_redeemed_usdc_amount")

        with self._lock:
            with self._connection(immediate=True) as con:
                existing_settle = con.execute(
                    "SELECT redeemed_usdc_amount, live_realized_pnl_usdc, settled_at_utc, redeem_tx_hash, payload_json FROM live_settlements WHERE market_id=?",
                    (market_id,),
                ).fetchone()
                prior_redeemed_amount = 0.0
                prior_pnl = 0.0
                prior_settled_at: str | None = None
                prior_tx_hash: str | None = None
                prior_tx_id: str | None = None
                prior_pending_keys: list[str] = []
                prior_pending_shares: str | None = None
                prior_pending_payout: float = 0.0
                prior_tx_submitted_at: str | None = None
                prior_submit_boot_id: str | None = None
                prior_balance_seq: int | None = None
                prior_free_usdc_at_submit: float | None = None
                prior_local_spent_at_submit: float | None = None
                prior_submit_state: str | None = None
                prior_redeem_request_id: str | None = None
                prior_redeem_target_to: str | None = None
                prior_redeem_calldata_hex: str | None = None
                prior_first_settled_at: str | None = None
                prior_first_loss_at: str | None = None
                prior_last_loss_at: str | None = None
                prior_pnl_history: list[dict[str, Any]] = []
                prior_redeemed_keys: set[str] = set()
                prior_processed_batches: set[str] = set()
                prior_payout_ratios: Mapping[str, Any] | None = None
                prior_payout_conflict: bool = False
                prior_unverified_payout_ratios: Any = None

                if existing_settle is not None:
                    prior_redeemed_amount = float(existing_settle["redeemed_usdc_amount"] or 0.0)
                    prior_pnl = float(existing_settle["live_realized_pnl_usdc"] or 0.0)
                    prior_settled_at = (
                        str(existing_settle["settled_at_utc"])
                        if existing_settle["settled_at_utc"]
                        else None
                    )
                    prior_tx_hash = (
                        str(existing_settle["redeem_tx_hash"])
                        if existing_settle["redeem_tx_hash"]
                        else None
                    )
                    raw_pj = existing_settle["payload_json"]
                    if isinstance(raw_pj, str) and raw_pj.strip():
                        try:
                            parsed_pj = json.loads(raw_pj)
                            if isinstance(parsed_pj, dict):
                                for k in parsed_pj.get("redeemed_order_keys") or []:
                                    if k is not None:
                                        prior_redeemed_keys.add(str(k))
                                for b in parsed_pj.get("processed_redemption_batches") or []:
                                    if b is not None:
                                        prior_processed_batches.add(str(b))
                                if isinstance(parsed_pj.get("payout_ratios"), Mapping):
                                    prior_payout_ratios = dict(parsed_pj["payout_ratios"])
                                if parsed_pj.get("payout_conflict") is True:
                                    prior_payout_conflict = True
                                if parsed_pj.get("unverified_payout_ratios") is not None:
                                    prior_unverified_payout_ratios = parsed_pj.get("unverified_payout_ratios")
                                if parsed_pj.get("redeem_tx_id"):
                                    prior_tx_id = str(parsed_pj["redeem_tx_id"])
                                if not prior_tx_hash and parsed_pj.get("redeem_tx_hash"):
                                    prior_tx_hash = str(parsed_pj["redeem_tx_hash"])
                                if isinstance(parsed_pj.get("pending_batch_order_keys"), list):
                                    prior_pending_keys = [
                                        str(k) for k in parsed_pj["pending_batch_order_keys"] if k is not None
                                    ]
                                if parsed_pj.get("pending_batch_shares") is not None:
                                    prior_pending_shares = str(parsed_pj["pending_batch_shares"])
                                if parsed_pj.get("pending_batch_payout_usdc") is not None:
                                    prior_pending_payout = float(parsed_pj["pending_batch_payout_usdc"] or 0.0)
                                if parsed_pj.get("tx_submitted_at_utc"):
                                    prior_tx_submitted_at = str(parsed_pj["tx_submitted_at_utc"])
                                if parsed_pj.get("submit_boot_id"):
                                    prior_submit_boot_id = str(parsed_pj["submit_boot_id"])
                                if parsed_pj.get("balance_snapshot_seq_at_submit") is not None:
                                    prior_balance_seq = int(parsed_pj["balance_snapshot_seq_at_submit"])
                                if parsed_pj.get("free_usdc_at_submit") is not None:
                                    prior_free_usdc_at_submit = float(parsed_pj["free_usdc_at_submit"])
                                if parsed_pj.get("local_spent_at_submit") is not None:
                                    prior_local_spent_at_submit = float(parsed_pj["local_spent_at_submit"])
                                if parsed_pj.get("submit_state"):
                                    prior_submit_state = str(parsed_pj["submit_state"])
                                if parsed_pj.get("redeem_request_id"):
                                    prior_redeem_request_id = str(parsed_pj["redeem_request_id"])
                                if parsed_pj.get("redeem_target_to"):
                                    prior_redeem_target_to = str(parsed_pj["redeem_target_to"])
                                if parsed_pj.get("redeem_calldata_hex"):
                                    prior_redeem_calldata_hex = str(parsed_pj["redeem_calldata_hex"])
                                if parsed_pj.get("first_settled_at_utc"):
                                    prior_first_settled_at = str(parsed_pj["first_settled_at_utc"])
                                if parsed_pj.get("first_loss_confirmed_at_utc"):
                                    prior_first_loss_at = str(parsed_pj["first_loss_confirmed_at_utc"])
                                if parsed_pj.get("last_loss_confirmed_at_utc"):
                                    prior_last_loss_at = str(parsed_pj["last_loss_confirmed_at_utc"])
                                if isinstance(parsed_pj.get("pnl_history"), list):
                                    prior_pnl_history = [
                                        dict(item)
                                        for item in parsed_pj["pnl_history"]
                                        if isinstance(item, Mapping)
                                    ]
                        except Exception:
                            pass
                    if not prior_pnl_history and abs(prior_pnl) > 1e-9 and prior_settled_at:
                        prior_pnl_history = [
                            {"pnl_delta_usdc": prior_pnl, "confirmed_at_utc": prior_settled_at}
                        ]
                    if not prior_first_loss_at and prior_pnl < -1e-9 and prior_settled_at:
                        prior_first_loss_at = prior_settled_at
                    if not prior_last_loss_at and prior_first_loss_at:
                        prior_last_loss_at = prior_first_loss_at

                orders = con.execute(
                    "SELECT signal_id, clob_order_id, paper_order_id, outcome, position_status, filled_quantity, total_spent_usdc FROM live_orders WHERE market_id=? ORDER BY rank ASC",
                    (market_id,),
                ).fetchall()

                incoming_ratios = settlement_record.get("payout_ratios")
                incoming_conflict = bool(settlement_record.get("payout_conflict")) or (
                    incoming_ratios is not None
                    and resolve_verified_payout_ratios(official_outcome, incoming_ratios) is None
                )
                incoming_unverified = settlement_record.get("unverified_payout_ratios")
                if incoming_unverified is None and incoming_conflict and incoming_ratios is not None:
                    incoming_unverified = (
                        dict(incoming_ratios) if isinstance(incoming_ratios, Mapping) else incoming_ratios
                    )
                verified_prior_ratios = (
                    resolve_verified_payout_ratios(official_outcome, prior_payout_ratios)
                    if prior_payout_ratios is not None
                    else None
                )
                if incoming_conflict:
                    ratio_map = None
                elif incoming_ratios is not None:
                    ratio_map = resolve_verified_payout_ratios(official_outcome, incoming_ratios)
                elif prior_payout_conflict:
                    ratio_map = None
                else:
                    candidate_ratios = prior_payout_ratios
                    ratio_map = resolve_verified_payout_ratios(official_outcome, candidate_ratios)
                if ratio_map is not None:
                    for order in orders:
                        pos_stat = str(order["position_status"])
                        if pos_stat in ("NO_FILL", *IN_FLIGHT_POSITION_STATUSES):
                            continue
                        if _norm_outcome_key(order["outcome"]) not in ratio_map:
                            ratio_map = None
                            break
                if redeem_status == "AWAITING_OFFICIAL_PAYOUT_VECTOR" or ratio_map is None:
                    live_cost_float = round(
                        float(
                            sum(
                                (
                                    to_decimal(o["total_spent_usdc"] or 0.0, "total_spent_usdc")
                                    for o in orders
                                    if str(o["position_status"]) not in ("NO_FILL", *IN_FLIGHT_POSITION_STATUSES)
                                ),
                                Decimal("0"),
                            )
                        ),
                        6,
                    )
                    cumulative_redeemed_amount = round(prior_redeemed_amount, 6)
                    live_realized_pnl = round(prior_pnl, 6)
                    settled_at = prior_settled_at or incoming_ts
                    effective_conflict = bool(incoming_conflict or (incoming_ratios is None and prior_payout_conflict))
                    rec_to_save = dict(settlement_record)
                    rec_to_save["timestamp_utc"] = settled_at
                    rec_to_save["first_settled_at_utc"] = prior_first_settled_at or settled_at
                    rec_to_save["first_loss_confirmed_at_utc"] = prior_first_loss_at
                    rec_to_save["last_loss_confirmed_at_utc"] = prior_last_loss_at
                    rec_to_save["pnl_history"] = list(prior_pnl_history)
                    rec_to_save["redeem_status"] = "AWAITING_OFFICIAL_PAYOUT_VECTOR"
                    rec_to_save["redeemed_usdc_amount"] = cumulative_redeemed_amount
                    rec_to_save["live_cost_usdc"] = live_cost_float
                    rec_to_save["live_realized_pnl_usdc"] = live_realized_pnl
                    rec_to_save["payout_conflict"] = effective_conflict
                    rec_to_save["unverified_payout_ratios"] = (
                        incoming_unverified
                        if incoming_unverified is not None
                        else (prior_unverified_payout_ratios if effective_conflict else None)
                    )
                    rec_to_save["payout_ratios"] = (
                        {k: float(v) for k, v in verified_prior_ratios.items()}
                        if (verified_prior_ratios is not None and not effective_conflict)
                        else None
                    )
                    rec_to_save["redeemed_order_keys"] = sorted(prior_redeemed_keys)
                    rec_to_save["processed_redemption_batches"] = sorted(prior_processed_batches)
                    payload_text = json.dumps(rec_to_save, ensure_ascii=False, allow_nan=False)
                    con.execute(
                        """
                        INSERT OR REPLACE INTO live_settlements(
                            market_id, condition_id, official_outcome, paper_payout_usd,
                            live_expected_payout_usdc, redeem_tx_hash, redeem_status,
                            redeemed_usdc_amount, live_cost_usdc, live_realized_pnl_usdc,
                            redeem_latency_seconds, settled_at_utc, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            market_id,
                            str(settlement_record.get("condition_id") or ""),
                            official_outcome,
                            float(settlement_record.get("paper_payout_usd") or 0.0),
                            float(settlement_record.get("live_expected_payout_usdc") or 0.0),
                            None,
                            "AWAITING_OFFICIAL_PAYOUT_VECTOR",
                            cumulative_redeemed_amount,
                            live_cost_float,
                            live_realized_pnl,
                            (
                                float(settlement_record["redeem_latency_seconds"])
                                if settlement_record.get("redeem_latency_seconds") is not None
                                else None
                            ),
                            settled_at,
                            payload_text,
                        ),
                    )
                    self._settled_markets.discard(market_id)
                    if append_jsonl:
                        self._append_jsonl(self.settlements_jsonl_path, rec_to_save)
                    return

                float_ratios = {k: float(v) for k, v in ratio_map.items()}
                explicit_redeemed_keys_list = settlement_record.get("redeemed_order_keys")
                explicit_new_keys_list = settlement_record.get("newly_redeemed_order_keys")
                has_explicit_batch_keys = (
                    isinstance(explicit_redeemed_keys_list, list) or isinstance(explicit_new_keys_list, list)
                )
                redeemed_order_keys: set[str] = set(prior_redeemed_keys)
                if isinstance(explicit_redeemed_keys_list, list):
                    redeemed_order_keys.update(str(k) for k in explicit_redeemed_keys_list if k is not None)
                batch_new_keys = (
                    [str(k) for k in explicit_new_keys_list if k is not None]
                    if isinstance(explicit_new_keys_list, list)
                    else []
                )
                redeemed_order_keys.update(batch_new_keys)

                processed_batches: set[str] = set(prior_processed_batches)
                batch_id = str(
                    settlement_record.get("redeem_tx_hash")
                    or settlement_record.get("redeem_tx_id")
                    or (f"keys:{','.join(sorted(batch_new_keys))}:{batch_redeemed_amount:.6f}" if batch_new_keys else "")
                )
                is_duplicate_batch = (
                    (bool(batch_new_keys) and set(batch_new_keys) <= prior_redeemed_keys)
                    or (bool(batch_id) and batch_id in prior_processed_batches)
                )

                if is_duplicate_batch:
                    cumulative_redeemed_amount = round(prior_redeemed_amount, 6)
                elif newly_redeemed_raw is not None:
                    cumulative_redeemed_amount = round(prior_redeemed_amount + float(newly_redeemed_raw or 0.0), 6)
                elif redeem_status in ("REDEEMED", "SHADOW_REDEEMED"):
                    cumulative_redeemed_amount = round(max(prior_redeemed_amount, batch_redeemed_amount), 6)
                else:
                    cumulative_redeemed_amount = round(prior_redeemed_amount, 6)

                if not is_duplicate_batch and batch_id and redeem_status in ("REDEEMED", "SHADOW_REDEEMED"):
                    processed_batches.add(batch_id)

                has_in_flight = any(str(o["position_status"]) in IN_FLIGHT_POSITION_STATUSES for o in orders)
                if has_in_flight and redeem_status in ("LOST_NO_REDEEM", "NO_LIVE_POSITION"):
                    redeem_status = "AWAITING_ORDER_RECONCILIATION"

                # [P2-R4-6]: When `has_explicit_batch_keys` is False, only allow implicit attribution
                # if there are no in-flight orders AND the unredeemed expected payout total in the market
                # exactly matches `batch_redeemed_amount`!
                unredeemed_expected_payout_total = Decimal("0")
                for idx, order in enumerate(orders):
                    pos_stat = str(order["position_status"])
                    if pos_stat in ("NO_FILL", *IN_FLIGHT_POSITION_STATUSES):
                        continue
                    ord_out = _norm_outcome_key(order["outcome"])
                    ord_ratio = ratio_map[ord_out]
                    qty_dec = to_decimal(order["filled_quantity"] or "0", "filled_quantity")
                    ord_expected_payout = (qty_dec * ord_ratio).quantize(USD_MICRO, rounding=ROUND_DOWN)
                    if ord_ratio > Decimal("0") and ord_expected_payout > Decimal("0"):
                        sig_id = str(order["signal_id"])
                        c_keys = {
                            sig_id,
                            f"clob:{order['clob_order_id']}" if order["clob_order_id"] else "",
                            f"paper:{order['paper_order_id']}" if order["paper_order_id"] else "",
                            f"idx:{idx}:{str(order['outcome']).upper()}:{order['filled_quantity']}",
                        } - {""}
                        if pos_stat != "WON_REDEEMED" and not bool(c_keys & prior_redeemed_keys):
                            unredeemed_expected_payout_total += ord_expected_payout
                allow_implicit_all_redeemed = (
                    not has_explicit_batch_keys
                    and not has_in_flight
                    and batch_redeemed_amount > 0
                    and unredeemed_expected_payout_total.quantize(USD_MICRO, rounding=ROUND_DOWN)
                    == to_decimal(batch_redeemed_amount, "batch_redeemed_amount").quantize(USD_MICRO, rounding=ROUND_DOWN)
                )

                confirmed_lost_cost = Decimal("0")
                won_redeemed_cost = Decimal("0")
                won_unredeemed_cost = Decimal("0")
                has_unredeemed_winner = False

                for idx, order in enumerate(orders):
                    pos_stat = str(order["position_status"])
                    if pos_stat in ("NO_FILL", *IN_FLIGHT_POSITION_STATUSES):
                        continue
                    spent_dec = to_decimal(order["total_spent_usdc"] or 0.0, "total_spent_usdc")
                    sig_id = str(order["signal_id"])
                    candidate_keys = {
                        sig_id,
                        f"clob:{order['clob_order_id']}" if order["clob_order_id"] else "",
                        f"paper:{order['paper_order_id']}" if order["paper_order_id"] else "",
                        f"idx:{idx}:{str(order['outcome']).upper()}:{order['filled_quantity']}",
                    } - {""}

                    ord_out = _norm_outcome_key(order["outcome"])
                    ord_ratio = ratio_map[ord_out]
                    qty_dec = to_decimal(order["filled_quantity"] or "0", "filled_quantity")
                    ord_expected_payout = (qty_dec * ord_ratio).quantize(USD_MICRO, rounding=ROUND_DOWN)

                    if ord_ratio > Decimal("0") and ord_expected_payout > Decimal("0"):
                        if pos_stat == "WON_REDEEMED" or bool(candidate_keys & redeemed_order_keys):
                            new_status = "WON_REDEEMED"
                            redeemed_order_keys.add(sig_id)
                        elif (
                            redeem_status in ("REDEEMED", "SHADOW_REDEEMED")
                            and cumulative_redeemed_amount > 0
                            and allow_implicit_all_redeemed
                        ):
                            new_status = "WON_REDEEMED"
                            redeemed_order_keys.add(sig_id)
                        else:
                            new_status = "WON_UNREDEEMED"
                            has_unredeemed_winner = True

                        if new_status == "WON_REDEEMED":
                            won_redeemed_cost += spent_dec
                        else:
                            won_unredeemed_cost += spent_dec
                    else:
                        new_status = "LOST"
                        confirmed_lost_cost += spent_dec

                    con.execute(
                        "UPDATE live_orders SET position_status=? WHERE signal_id=?",
                        (new_status, sig_id),
                    )

                # [P1-R3-1]: If a batch redemption just confirmed (`REDEEMED`) while additional winning orders
                # remain `WON_UNREDEEMED` (or in-flight orders exist), keep `live_settlements` in a pending
                # state (with cleared pending tx_id) so the remaining batch will be redeemed next!
                effective_redeem_status = redeem_status
                clear_pending_tx_for_next_batch = False
                if redeem_status in ("REDEEMED", "SHADOW_REDEEMED"):
                    if has_unredeemed_winner or bool(settlement_record.get("has_remaining_unredeemed")):
                        effective_redeem_status = "REDEEM_PENDING"
                        clear_pending_tx_for_next_batch = True
                    elif has_in_flight:
                        effective_redeem_status = "AWAITING_ORDER_RECONCILIATION"
                        clear_pending_tx_for_next_batch = True

                total_live_cost = (confirmed_lost_cost + won_redeemed_cost + won_unredeemed_cost).quantize(
                    USD_MICRO, rounding=ROUND_DOWN
                )
                realized_cost_dec = (confirmed_lost_cost + won_redeemed_cost).quantize(
                    USD_MICRO, rounding=ROUND_DOWN
                )
                live_cost_float = float(total_live_cost)

                # [P1-R3-2]: Immediately account for confirmed losses (`confirmed_lost_cost`) and confirmed
                # redemptions (`won_redeemed_cost`) even when other orders in the same market are still
                # `ORDER_STATE_UNKNOWN` (`has_in_flight`) or `WON_UNREDEEMED`!
                if won_redeemed_cost > Decimal("0") or confirmed_lost_cost > Decimal("0"):
                    live_realized_pnl = round(cumulative_redeemed_amount - float(realized_cost_dec), 6)
                elif effective_redeem_status in ("REDEEMED", "SHADOW_REDEEMED"):
                    live_realized_pnl = round(cumulative_redeemed_amount - live_cost_float, 6)
                elif effective_redeem_status == "LOST_NO_REDEEM" and not has_in_flight:
                    live_realized_pnl = round(-live_cost_float, 6)
                else:
                    live_realized_pnl = 0.0

                # [P1-R4-2] & [P1-R5-2]:
                #   - On repeated queries (`abs(pnl_delta) <= 1e-9`), preserve `prior_settled_at`
                #     and `prior_last_loss_at` so re-polling never shifts PnL dates or cooldowns.
                #   - On newly confirmed PnL (`abs(pnl_delta) > 1e-9`), append to `pnl_history`;
                #     specifically when a new loss is confirmed (`pnl_delta < -1e-9`), update
                #     `settled_at` and `last_loss_confirmed_at_utc` to `incoming_ts` so late losses
                #     properly re-order consecutive losses and re-trigger cooldown!
                pnl_delta = round(live_realized_pnl - prior_pnl, 6)
                pnl_history = list(prior_pnl_history)
                first_settled_at = prior_first_settled_at or prior_settled_at or incoming_ts
                first_loss_at = prior_first_loss_at
                last_loss_at = prior_last_loss_at

                if abs(pnl_delta) > 1e-9:
                    pnl_history.append({"pnl_delta_usdc": pnl_delta, "confirmed_at_utc": incoming_ts})
                    settled_at = incoming_ts
                    if pnl_delta < -1e-9:
                        if not first_loss_at:
                            first_loss_at = incoming_ts
                        last_loss_at = incoming_ts
                else:
                    settled_at = prior_settled_at or incoming_ts
                    if live_realized_pnl < -1e-9 and not first_loss_at:
                        first_loss_at = settled_at
                    if live_realized_pnl < -1e-9 and not last_loss_at:
                        last_loss_at = first_loss_at or settled_at

                rec_to_save = dict(settlement_record)
                rec_to_save["timestamp_utc"] = settled_at
                rec_to_save["first_settled_at_utc"] = first_settled_at
                rec_to_save["first_loss_confirmed_at_utc"] = first_loss_at
                rec_to_save["last_loss_confirmed_at_utc"] = last_loss_at
                rec_to_save["pnl_history"] = pnl_history
                rec_to_save["redeem_status"] = effective_redeem_status
                rec_to_save["redeemed_usdc_amount"] = cumulative_redeemed_amount
                rec_to_save["live_cost_usdc"] = live_cost_float
                rec_to_save["live_realized_pnl_usdc"] = live_realized_pnl
                rec_to_save["payout_ratios"] = float_ratios
                rec_to_save["payout_conflict"] = False
                rec_to_save["unverified_payout_ratios"] = None
                rec_to_save["redeemed_order_keys"] = sorted(redeemed_order_keys)
                rec_to_save["processed_redemption_batches"] = sorted(processed_batches)

                if clear_pending_tx_for_next_batch:
                    rec_to_save["redeem_tx_id"] = None
                    rec_to_save["redeem_tx_hash"] = None
                    rec_to_save["pending_batch_order_keys"] = []
                    rec_to_save["pending_batch_payout_usdc"] = 0.0
                    rec_to_save["submit_state"] = None
                    rec_to_save["redeem_request_id"] = None
                    rec_to_save["redeem_target_to"] = None
                    rec_to_save["redeem_calldata_hex"] = None
                    rec_to_save["submit_boot_id"] = None
                    rec_to_save["balance_snapshot_seq_at_submit"] = None
                    rec_to_save["free_usdc_at_submit"] = None
                    rec_to_save["local_spent_at_submit"] = None
                    rec_to_save["tx_submitted_at_utc"] = None
                elif effective_redeem_status == "REDEEM_PENDING" and str(rec_to_save.get("submit_state") or "").upper() != "SUBMIT_HTTP_400_REJECTED":
                    # [P1-R4-3] & [P1-R5-3]: Never lose an already-persisted pending `redeem_tx_id` /
                    # `redeem_tx_hash`, `submit_state`, `redeem_request_id`, or batch metadata if a
                    # subsequent progress update lacks them!
                    if not rec_to_save.get("redeem_tx_id") and prior_tx_id:
                        rec_to_save["redeem_tx_id"] = prior_tx_id
                    if not rec_to_save.get("redeem_tx_hash") and prior_tx_hash:
                        rec_to_save["redeem_tx_hash"] = prior_tx_hash
                    if not rec_to_save.get("pending_batch_order_keys") and prior_pending_keys:
                        rec_to_save["pending_batch_order_keys"] = prior_pending_keys
                    if not rec_to_save.get("pending_batch_shares") and prior_pending_shares:
                        rec_to_save["pending_batch_shares"] = prior_pending_shares
                    if not float(rec_to_save.get("pending_batch_payout_usdc") or 0.0) and prior_pending_payout > 0:
                        rec_to_save["pending_batch_payout_usdc"] = prior_pending_payout
                    if not rec_to_save.get("submit_state") and prior_submit_state:
                        rec_to_save["submit_state"] = prior_submit_state
                    if not rec_to_save.get("redeem_request_id") and prior_redeem_request_id:
                        rec_to_save["redeem_request_id"] = prior_redeem_request_id
                    if not rec_to_save.get("redeem_target_to") and prior_redeem_target_to:
                        rec_to_save["redeem_target_to"] = prior_redeem_target_to
                    if not rec_to_save.get("redeem_calldata_hex") and prior_redeem_calldata_hex:
                        rec_to_save["redeem_calldata_hex"] = prior_redeem_calldata_hex
                    if not rec_to_save.get("tx_submitted_at_utc") and prior_tx_submitted_at:
                        rec_to_save["tx_submitted_at_utc"] = prior_tx_submitted_at
                    if not rec_to_save.get("submit_boot_id") and prior_submit_boot_id:
                        rec_to_save["submit_boot_id"] = prior_submit_boot_id
                    if rec_to_save.get("balance_snapshot_seq_at_submit") is None and prior_balance_seq is not None:
                        rec_to_save["balance_snapshot_seq_at_submit"] = prior_balance_seq
                    if rec_to_save.get("free_usdc_at_submit") is None and prior_free_usdc_at_submit is not None:
                        rec_to_save["free_usdc_at_submit"] = prior_free_usdc_at_submit
                    if rec_to_save.get("local_spent_at_submit") is None and prior_local_spent_at_submit is not None:
                        rec_to_save["local_spent_at_submit"] = prior_local_spent_at_submit

                payload_text = json.dumps(rec_to_save, ensure_ascii=False, allow_nan=False)

                con.execute(
                    """
                    INSERT OR REPLACE INTO live_settlements(
                        market_id, condition_id, official_outcome, paper_payout_usd,
                        live_expected_payout_usdc, redeem_tx_hash, redeem_status,
                        redeemed_usdc_amount, live_cost_usdc, live_realized_pnl_usdc,
                        redeem_latency_seconds, settled_at_utc, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        market_id,
                        str(settlement_record.get("condition_id") or ""),
                        official_outcome,
                        float(settlement_record.get("paper_payout_usd") or 0.0),
                        float(settlement_record.get("live_expected_payout_usdc") or 0.0),
                        None if clear_pending_tx_for_next_batch else rec_to_save.get("redeem_tx_hash"),
                        effective_redeem_status,
                        cumulative_redeemed_amount,
                        live_cost_float,
                        live_realized_pnl,
                        (
                            float(settlement_record["redeem_latency_seconds"])
                            if settlement_record.get("redeem_latency_seconds") is not None
                            else None
                        ),
                        settled_at,
                        payload_text,
                    ),
                )
            if (
                effective_redeem_status in TERMINAL_SETTLED_REDEEM_STATUSES
                and not has_in_flight
                and not has_unredeemed_winner
            ):
                self._settled_markets.add(market_id)
            else:
                self._settled_markets.discard(market_id)
            if append_jsonl:
                self._append_jsonl(self.settlements_jsonl_path, rec_to_save)
