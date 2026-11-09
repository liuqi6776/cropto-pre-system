"""Local-only binary-contract paper accounting with observed-input admission.

The caller must verify model artifacts and collect public venue observations.
INPUTS_VALIDATED is that caller's attestation, not a strategy effectiveness gate.
No network, signing, wallet, or live-order capability exists in this module.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

from .stage import DISABLED, StageRejected, stage_profile


STREAMS = {
    "BTC_15m": ("BTC", 900), "ETH_15m": ("ETH", 900),
    "BTC_5m": ("BTC", 300), "ETH_5m": ("ETH", 300),
    "SOL_15m": ("SOL", 900), "SOL_5m": ("SOL", 300),
    "BNB_15m": ("BNB", 900),
}
ALL_STREAMS = {**STREAMS, "XRP_15m": ("XRP", 900)}
MICRO = Decimal("1000000")
QTY_STEP = Decimal("0.00000001")
V5_SEMANTICS = "ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1"
V5_UNLOCKED = ("FILLED_BUDGET", "NO_SIGNAL", "NOT_DUE", "PRICE_LIMIT_NOT_MET", DISABLED)
LEGACY_UNLOCKED = ("FILLED_CAPACITY", "NO_SIGNAL", "NOT_DUE")
FORBIDDEN = re.compile(r"private[_ -]?key|mnemonic|secret|seed[_ -]?phrase|api[_ -]?key|password", re.I)


class AccountRejected(ValueError):
    """An input failed admission; error text never includes the rejected value."""


def _safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _safe(str(k))
            _safe(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _safe(v)
    elif isinstance(value, str) and FORBIDDEN.search(value):
        raise AccountRejected("credential-like input refused")


def _ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError) as exc:
            raise AccountRejected("invalid UTC timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise AccountRejected("timezone-aware timestamp required")
    return stamp.astimezone(timezone.utc)


def _now(value=None) -> datetime:
    return _ts(value) if value is not None else datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _d(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise AccountRejected("invalid numeric input") from exc
    if not result.is_finite():
        raise AccountRejected("nonfinite numeric input")
    return result


def _atoms(value: Decimal) -> int:
    return int((value * MICRO).to_integral_value(rounding=ROUND_CEILING))


def _usd(value: int) -> float:
    return float(Decimal(value) / MICRO)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stage(policy, boundary):
    try:
        return stage_profile(policy, boundary)
    except StageRejected as exc:
        raise AccountRejected(str(exc)) from exc


def _policy_risk_limits(policy=None):
    """Read and validate configurable risk limits from policy (with v13r2 defaults)."""
    src = policy if isinstance(policy, Mapping) else {}
    risk_block = src.get("risk") if isinstance(src.get("risk"), Mapping) else {}

    def dec_field(key, default):
        raw = risk_block.get(key) if key in risk_block else src.get(key)
        if raw is None:
            return Decimal(str(default))
        if isinstance(raw, bool):
            raise AccountRejected("invalid policy risk limits")
        return _d(raw)

    def has_field(key):
        return key in risk_block or key in src

    single_frac = dec_field("single_stream_equity_fraction", "0.015")
    tick_frac = dec_field("tick_equity_fraction", "0.022")
    max_order = dec_field("max_order_usd", "200")
    max_open = dec_field("max_open_risk_usd", "1000")
    default_tick = max(Decimal("1200"), max_order) if has_field("max_order_usd") and not has_field("max_tick_budget_usd") else "1200"
    max_tick = dec_field("max_tick_budget_usd", default_tick)
    default_underlying = min(Decimal("500"), max_open) if has_field("max_open_risk_usd") and not has_field("max_underlying_open_risk_usd") else "500"
    max_underlying = dec_field("max_underlying_open_risk_usd", default_underlying)
    max_active_open = dec_field("max_active_window_risk_usd", max_open)
    max_active_underlying = dec_field("max_underlying_active_risk_usd", min(max_underlying, max_active_open))
    daily_loss = dec_field("daily_loss_fraction", "0.055")
    min_buy_price = dec_field("min_buy_price", "0")
    raw_stream_min = risk_block.get("stream_min_buy_price") if "stream_min_buy_price" in risk_block else src.get("stream_min_buy_price")
    stream_min_buy_price = {}
    if raw_stream_min is not None:
        if not isinstance(raw_stream_min, Mapping):
            raise AccountRejected("invalid policy risk limits")
        for s_key, s_val in raw_stream_min.items():
            if s_key not in ALL_STREAMS or isinstance(s_val, bool):
                raise AccountRejected("invalid policy risk limits")
            d_val = _d(s_val)
            if not Decimal("0") <= d_val <= Decimal("0.500"):
                raise AccountRejected("invalid policy risk limits")
            stream_min_buy_price[str(s_key)] = d_val
    if (not Decimal("0.005") <= single_frac <= Decimal("0.030")
            or not Decimal("0.010") <= tick_frac <= Decimal("0.050")
            or not Decimal("0.010") <= daily_loss <= Decimal("0.100")
            or not Decimal("0") <= min_buy_price <= Decimal("0.500")
            or max_tick <= 0 or max_order <= 0 or max_open <= 0 or max_underlying <= 0
            or max_active_open <= 0 or max_active_underlying <= 0
            or max_order > max_tick or max_underlying > max_open
            or max_active_open > max_open or max_active_underlying > max_underlying
            or max_active_underlying > max_active_open):
        raise AccountRejected("invalid policy risk limits")
    return {
        "single_stream_equity_fraction": single_frac,
        "tick_equity_fraction": tick_frac,
        "max_tick_budget_usd": max_tick,
        "max_order_usd": max_order,
        "max_open_risk_usd": max_open,
        "max_underlying_open_risk_usd": max_underlying,
        "max_active_window_risk_usd": max_active_open,
        "max_underlying_active_risk_usd": max_active_underlying,
        "daily_loss_fraction": daily_loss,
        "min_buy_price": min_buy_price,
        "stream_min_buy_price": stream_min_buy_price,
        "max_tick_budget_atoms": _atoms(max_tick),
        "max_order_atoms": _atoms(max_order),
        "max_open_risk_atoms": _atoms(max_open),
        "max_underlying_open_risk_atoms": _atoms(max_underlying),
        "max_active_window_risk_atoms": _atoms(max_active_open),
        "max_underlying_active_risk_atoms": _atoms(max_active_underlying),
    }


def multi_look_settings(policy):
    """Explicit v12-only admission; older policies retain one-look semantics."""
    if policy.get("engineering", {}).get("profile") != "argon4-latency-v12":
        return None
    settings = policy.get("execution", {})
    if not isinstance(settings, Mapping) or not isinstance(settings.get("multi_look_ioc", False), bool):
        raise AccountRejected("invalid multi-look execution settings")
    if settings.get("multi_look_ioc") is not True:
        return None
    count = settings.get("max_price_rechecks")
    if (not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 3
            or _d(settings.get("price_recheck_interval_seconds")) != Decimal("2.5")
            or _d(settings.get("minimum_recheck_remaining_seconds")) != Decimal("3")
            or _d(settings.get("last_recheck_start_after_boundary_seconds")) != Decimal("18.5")):
        raise AccountRejected("multi-look bounds differ from the approved execution policy")
    return dict(settings)


class Account:
    """One restart-safe USD account, stored separately from every historical run."""

    def __init__(self, db_path, initial_capital=10000, currency="USD", started_at=None):
        if currency != "USD":
            raise AccountRejected("this binary paper account uses USD only")
        capital = _atoms(_d(initial_capital))
        if capital <= 0:
            raise AccountRejected("positive initial capital required")
        start = _now(started_at)
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK(id=1), currency TEXT NOT NULL,
                    initial_atoms INTEGER NOT NULL, cash_atoms INTEGER NOT NULL,
                    started_at TEXT NOT NULL, realized_atoms INTEGER NOT NULL DEFAULT 0,
                    peak_realized_atoms INTEGER NOT NULL, max_realized_dd REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS runtime (
                    id INTEGER PRIMARY KEY CHECK(id=1), status TEXT NOT NULL,
                    blocked_reason TEXT, heartbeat_at TEXT NOT NULL, details TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY, payload TEXT NOT NULL, received_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY, signal_id TEXT UNIQUE NOT NULL,
                    market_id TEXT NOT NULL, underlying TEXT NOT NULL, outcome TEXT NOT NULL,
                    quantity TEXT NOT NULL, cost_atoms INTEGER NOT NULL, fee_atoms INTEGER NOT NULL,
                    status TEXT NOT NULL, fill_at TEXT NOT NULL, window_end TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS resolutions (
                    market_id TEXT PRIMARY KEY, payload TEXT NOT NULL, settled_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS consumed_book_depth (
                    book_hash TEXT NOT NULL, price TEXT NOT NULL, quantity TEXT NOT NULL,
                    PRIMARY KEY(book_hash,price)
                );
                CREATE TABLE IF NOT EXISTS allocation_consumption (
                    plan_id TEXT NOT NULL, rank INTEGER NOT NULL, signal_id TEXT UNIQUE NOT NULL,
                    approved_atoms INTEGER NOT NULL, spent_atoms INTEGER NOT NULL,
                    order_id TEXT UNIQUE NOT NULL, PRIMARY KEY(plan_id,rank)
                );
                CREATE TABLE IF NOT EXISTS decision_cohorts (
                    cohort_id TEXT PRIMARY KEY, policy_hash TEXT NOT NULL, policy_version TEXT NOT NULL,
                    plan_id TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                    equity_basis_atoms INTEGER NOT NULL, budget_atoms INTEGER NOT NULL,
                    single_atoms INTEGER NOT NULL, used_atoms INTEGER NOT NULL DEFAULT 0,
                    blocked_reason TEXT, policy_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cohort_streams (
                    cohort_id TEXT NOT NULL, stream TEXT NOT NULL, rank INTEGER NOT NULL,
                    state TEXT NOT NULL, capacity_atoms INTEGER NOT NULL, approved_atoms INTEGER NOT NULL,
                    spent_atoms INTEGER NOT NULL, signal_id TEXT, receipt_json TEXT NOT NULL,
                    PRIMARY KEY(cohort_id,stream), UNIQUE(cohort_id,rank)
                );
                CREATE TABLE IF NOT EXISTS cohort_book_looks (
                    cohort_id TEXT NOT NULL, stream TEXT NOT NULL, attempt INTEGER NOT NULL,
                    requested_at TEXT NOT NULL, completed_at TEXT, state TEXT NOT NULL,
                    next_retry_at TEXT, signal_sha256 TEXT NOT NULL, receipt_json TEXT NOT NULL,
                    PRIMARY KEY(cohort_id,stream,attempt)
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, utc TEXT NOT NULL, event_type TEXT NOT NULL,
                    payload TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS equity_observations (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, utc TEXT NOT NULL,
                    cash_atoms INTEGER NOT NULL, locked_atoms INTEGER NOT NULL,
                    realized_atoms INTEGER NOT NULL, equity_atoms INTEGER, valuation_status TEXT NOT NULL
                );
            """)
            if con.execute("SELECT 1 FROM account WHERE id=1").fetchone() is None:
                con.execute("INSERT INTO account VALUES(1,?,?,?,?,0,?,0)",
                            (currency, capital, capital, _iso(start), capital))
                con.execute("INSERT INTO runtime VALUES(1,?,?,?,?)", (
                    "WAITING_FOR_MODEL", "Seven-stream inference models have not been installed",
                    _iso(start), "{}"))
                self._event(con, "ACCOUNT_CREATED", {"initial_capital": _usd(capital), "currency": currency}, start)
                self._snapshot(con, start)

    @contextmanager
    def _connection(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    @staticmethod
    def _event(con, kind, payload, now):
        text = _json(payload)
        prev = con.execute("SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        previous = prev[0] if prev else "0" * 64
        digest = hashlib.sha256((previous + _iso(now) + kind + text).encode()).hexdigest()
        con.execute("INSERT INTO events(utc,event_type,payload,previous_hash,event_hash) VALUES(?,?,?,?,?)",
                    (_iso(now), kind, text, previous, digest))

    @staticmethod
    def _snapshot(con, now):
        account = con.execute("SELECT * FROM account WHERE id=1").fetchone()
        locked = con.execute("SELECT COALESCE(SUM(cost_atoms+fee_atoms),0) FROM orders WHERE status='OPEN'").fetchone()[0]
        status = "UNVALUED_OPEN_POSITIONS" if locked else "CASH_ONLY"
        equity = None if locked else account["cash_atoms"]
        con.execute("INSERT INTO equity_observations(utc,cash_atoms,locked_atoms,realized_atoms,equity_atoms,valuation_status) VALUES(?,?,?,?,?,?)",
                    (_iso(now), account["cash_atoms"], locked, account["realized_atoms"], equity, status))

    def set_runtime_status(self, status, blocked_reason=None, details=None, now=None):
        value = {"status": status, "blocked_reason": blocked_reason, "details": details or {}}
        _safe(value)
        if not isinstance(status, str) or not status:
            raise AccountRejected("runtime status required")
        stamp = _now(now)
        with self._connection() as con:
            con.execute("UPDATE runtime SET status=?,blocked_reason=?,heartbeat_at=?,details=? WHERE id=1",
                        (status, blocked_reason, _iso(stamp), _json(details or {})))
        return self.overview(now=stamp)

    heartbeat = set_runtime_status

    def cohort_state(self, cohort_id):
        cohort_id = _iso(_ts(cohort_id))
        with self._connection() as con:
            row = con.execute("SELECT * FROM decision_cohorts WHERE cohort_id=?", (cohort_id,)).fetchone()
            if row is None:
                return None
            states = [dict(r) for r in con.execute("SELECT * FROM cohort_streams WHERE cohort_id=? ORDER BY rank", (cohort_id,))]
        result = dict(row)
        result["policy"] = json.loads(result.pop("policy_json"))
        for key in ("equity_basis", "budget", "single", "used"):
            result[key + "_usd"] = _usd(result.pop(key + "_atoms"))
        result["remaining_usd"] = max(0.0, result["budget_usd"] - result["used_usd"])
        result["budget_basis"] = "REALIZED_BALANCE_NOT_MARK_TO_MARKET"
        for state in states:
            for key in ("capacity", "approved", "spent"):
                state[key + "_usd"] = _usd(state.pop(key + "_atoms"))
            state["receipt"] = json.loads(state.pop("receipt_json"))
        result["streams"] = states
        return result

    def ensure_cohort(self, cohort_id, policy, now=None):
        """Freeze one five-minute decision cohort's risk budget across polling/restarts."""
        stamp, boundary = _now(now), _ts(cohort_id)
        _safe(policy)
        limits = _policy_risk_limits(policy)
        if boundary.timestamp() % 300 or not boundary <= stamp < boundary + timedelta(seconds=20):
            raise AccountRejected("cohort must be a current five-minute decision boundary")
        order = policy.get("order", [])
        valid_streams = set(STREAMS) if isinstance(order, (tuple, list)) and len(order) == 7 else (
            set(ALL_STREAMS) if isinstance(order, (tuple, list)) and len(order) == 8 else None)
        if (policy.get("status") != "CONFIRMED" or valid_streams is None or set(order) != valid_streams
                or not policy.get("version") or not policy.get("ranking_source")):
            raise AccountRejected("confirmed complete ranking policy required")
        if policy.get("waterfall_semantics") not in (None, V5_SEMANTICS):
            raise AccountRejected("unsupported waterfall semantics")
        stage = _stage(policy, boundary)
        key = _iso(boundary)
        canonical = _json(dict(policy))
        policy_hash = hashlib.sha256(canonical.encode()).hexdigest()
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT * FROM decision_cohorts WHERE cohort_id=?", (key,)).fetchone()
            if old:
                if old["policy_hash"] != policy_hash:
                    raise AccountRejected("policy cannot change inside a decision cohort")
            else:
                a = con.execute("SELECT * FROM account WHERE id=1").fetchone()
                if stamp < _ts(a["started_at"]):
                    raise AccountRejected("cohort precedes account activation")
                # Known cash plus paid cost of unresolved contracts is a risk-budget
                # basis, not a market valuation. Unresolved contracts also consume
                # their full at-risk paid cost under the separate $1,000 limit.
                equity = a["initial_atoms"] + a["realized_atoms"]
                open_rows = con.execute("SELECT cost_atoms,fee_atoms,window_end FROM orders WHERE status='OPEN'").fetchall()
                risk = sum(r["cost_atoms"] + r["fee_atoms"] for r in open_rows)
                active_risk = sum(r["cost_atoms"] + r["fee_atoms"] for r in open_rows if _ts(r["window_end"]) > stamp)
                daily = {}
                for event in con.execute("SELECT utc,payload FROM events WHERE event_type='OFFICIAL_RESOLUTION_APPLIED_TO_PAPER'"):
                    date = _ts(event["utc"]).date()
                    amount = _atoms(_d(json.loads(event["payload"])["realized_pnl_usd"]))
                    daily[date] = daily.get(date, 0) + amount
                today_pnl = daily.get(stamp.date(), 0)
                day_start_equity = equity - today_pnl
                running_today = 0
                peak_today_pnl = 0
                for event in con.execute("SELECT utc,payload FROM events WHERE event_type='OFFICIAL_RESOLUTION_APPLIED_TO_PAPER' ORDER BY seq ASC"):
                    if _ts(event["utc"]).date() == stamp.date():
                        running_today += _atoms(_d(json.loads(event["payload"])["realized_pnl_usd"]))
                        if running_today > peak_today_pnl:
                            peak_today_pnl = running_today
                trailing_enabled = bool(policy.get("trailing_daily_loss_limit", False))
                ref_equity = (day_start_equity + peak_today_pnl) if trailing_enabled else day_start_equity
                drawdown_from_ref = (today_pnl - peak_today_pnl) if trailing_enabled else today_pnl
                max_daily_loss_atoms = int((Decimal(max(0, ref_equity)) * limits["daily_loss_fraction"]).to_integral_value(rounding=ROUND_DOWN))
                until_raw = policy.get("cooldown_until")
                until_ts = _ts(until_raw) if until_raw else None
                known_untils = [until_ts] if until_ts is not None else []
                for (p_json,) in con.execute("SELECT policy_json FROM decision_cohorts"):
                    try:
                        p_u = json.loads(p_json).get("cooldown_until")
                        if p_u:
                            known_untils.append(_ts(p_u))
                    except Exception:
                        pass
                losing_dates = sorted(d for d, amt in daily.items() if d < stamp.date() and amt < 0)
                run_len = 0
                prev_d = None
                unsettled_trigger_ts = None
                last_cleared_loss_date = None
                for d in losing_dates:
                    if prev_d is not None and (d - prev_d).days == 1:
                        run_len += 1
                    else:
                        run_len = 1
                    prev_d = d
                    if run_len >= 5:
                        # Real trigger time is the end of the 5th losing day (00:00:00Z of the next day),
                        # never overwritten by subsequent blocked cohort dates.
                        trigger_ts = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=1)
                        matching_untils = [u for u in known_untils if u > trigger_ts]
                        if matching_untils and any(u <= stamp for u in matching_untils):
                            last_cleared_loss_date = d
                            run_len = 0
                            prev_d = None
                            unsettled_trigger_ts = None
                        else:
                            unsettled_trigger_ts = trigger_ts
                if not losing_dates and unsettled_trigger_ts is None:
                    prior_req_row = con.execute(
                        "SELECT cohort_id FROM decision_cohorts "
                        "WHERE blocked_reason='COOLDOWN_STATE_REQUIRED' "
                        "ORDER BY cohort_id ASC LIMIT 1"
                    ).fetchone()
                    if prior_req_row is not None:
                        req_ts = _ts(prior_req_row[0])
                        matching_untils = [u for u in known_untils if u > req_ts]
                        if not (matching_untils and any(u <= stamp for u in matching_untils)):
                            unsettled_trigger_ts = req_ts
                streak, day = 0, stamp.date() - timedelta(days=1)
                while (last_cleared_loss_date is None or day > last_cleared_loss_date) and daily.get(day, 0) < 0:
                    streak += 1
                    day -= timedelta(days=1)
                cooldown_latched = unsettled_trigger_ts is not None
                if cooldown_latched:
                    streak = max(streak, 5)
                blocked = None
                if ref_equity <= 0 or -drawdown_from_ref >= max_daily_loss_atoms:
                    blocked = "DAILY_LOSS_LIMIT"
                if until_ts is not None and stamp < until_ts:
                    blocked = "COOLDOWN_ACTIVE"
                elif streak >= 5 or cooldown_latched:
                    if until_ts is None or (unsettled_trigger_ts is not None and until_ts <= unsettled_trigger_ts):
                        blocked = "COOLDOWN_STATE_REQUIRED"
                    elif stamp < until_ts:
                        blocked = "COOLDOWN_ACTIVE"
                budget = max(0, min(int(Decimal(equity)*limits["tick_equity_fraction"]), limits["max_tick_budget_atoms"],
                                    a["cash_atoms"], limits["max_open_risk_atoms"]-risk,
                                    limits["max_active_window_risk_atoms"]-active_risk))
                single = max(0, min(limits["max_order_atoms"], int(Decimal(equity)*limits["single_stream_equity_fraction"]*(Decimal(".5") if streak >= 4 else 1))))
                remaining_loss_room = max(0, max_daily_loss_atoms + drawdown_from_ref - active_risk)
                if (drawdown_from_ref < 0 and policy.get("cap_budget_by_daily_loss_room", True)) or bool(
                    policy.get("cap_order_by_remaining_daily_loss", False)
                ):
                    budget = min(budget, remaining_loss_room)
                    single = min(single, remaining_loss_room)
                if not budget:
                    blocked = blocked or (
                        "DAILY_LOSS_LIMIT"
                        if (drawdown_from_ref < 0 and max_daily_loss_atoms + drawdown_from_ref <= 0)
                        else "NO_RISK_BUDGET"
                    )
                plan_id = "cohort:" + hashlib.sha256((key+policy_hash).encode()).hexdigest()
                con.execute("INSERT INTO decision_cohorts VALUES(?,?,?,?,?,?,?,?,?,0,?,?)", (
                    key, policy_hash, policy["version"], plan_id, _iso(stamp), _iso(boundary+timedelta(seconds=20)),
                    equity, budget, single, blocked, canonical))
                self._event(con, "DECISION_COHORT_CREATED", {"cohort_id": key, "plan_id": plan_id,
                    "budget_usd": _usd(budget), "single_stream_limit_usd": _usd(single),
                    "equity_budget_basis": "REALIZED_BALANCE_NOT_MARK_TO_MARKET", "blocked_reason": blocked}, stamp)
                # The policy, every disabled tier and their audit events commit
                # together. No model output or market receipt is invented.
                for stream in stage["disabled_streams"]:
                    receipt = {"source": "CONFIRMED_EXECUTION_STAGE", "stage_id": stage["id"],
                               "effective_from": stage["effective_from"], "policy_sha256": policy_hash,
                               "enabled_streams": stage["enabled_streams"],
                               "book_checked": False, "model_evaluated": False}
                    encoded = _json(receipt)
                    con.execute("INSERT INTO cohort_streams VALUES(?,?,?,?,0,0,0,NULL,?)", (
                        key, stream, policy["order"].index(stream)+1, DISABLED, encoded))
                    self._event(con, "COHORT_STREAM_SKIPPED", {"cohort_id": key, "stream": stream,
                        "state": DISABLED, "receipt": receipt,
                        "receipt_sha256": hashlib.sha256(encoded.encode()).hexdigest()}, stamp)
        return self.cohort_state(key)

    @staticmethod
    def _higher_ranks_unlocked(con, cohort_key, rank, cohort_policy=None, v5=True):
        higher = con.execute("SELECT rank,state FROM cohort_streams WHERE cohort_id=? AND rank<?",
                             (cohort_key, rank)).fetchall()
        return len(higher) == rank - 1 and all(r["state"] in (V5_UNLOCKED if v5 else LEGACY_UNLOCKED) for r in higher)

    def record_cohort_gate(self, cohort_id, stream, decision_status, capacity_usd, receipt, now=None):
        """Persist an observed upper-layer capacity gate for an explicitly absent signal."""
        stamp = _now(now)
        key = _iso(_ts(cohort_id))
        _safe(receipt)
        if decision_status not in ("NO_SIGNAL", "NOT_DUE") or stream not in ALL_STREAMS:
            raise AccountRejected("explicit no-signal or not-due state required")
        capacity = _atoms(_d(capacity_usd))
        if capacity <= 0 or not receipt.get("source") or not receipt.get("sha256"):
            raise AccountRejected("positive observed capacity with provenance required")
        age = (stamp - _ts(receipt.get("observed_at"))).total_seconds()
        if not 0 <= age <= 2:
            raise AccountRejected("capacity observation must be current")
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            cohort = con.execute("SELECT * FROM decision_cohorts WHERE cohort_id=?", (key,)).fetchone()
            if not cohort or not stamp < _ts(cohort["expires_at"]) or cohort["blocked_reason"]:
                raise AccountRejected("decision cohort unavailable")
            policy = json.loads(cohort["policy_json"])
            if policy.get("waterfall_semantics") == V5_SEMANTICS:
                raise AccountRejected("eligible-signal policy requires an explicit skip receipt")
            order = policy["order"]
            rank = order.index(stream)+1
            if decision_status == "NOT_DUE" and (ALL_STREAMS[stream][1] != 900 or _ts(key).timestamp()%900 == 0):
                raise AccountRejected("stream is actually due at this boundary")
            existing = con.execute("SELECT * FROM cohort_streams WHERE cohort_id=? AND stream=?", (key,stream)).fetchone()
            if existing:
                return dict(existing)
            preceding = con.execute("SELECT rank,state FROM cohort_streams WHERE cohort_id=? AND rank<?", (key,rank)).fetchall()
            if len(preceding) != rank-1 or any(r["state"] not in ("FILLED_CAPACITY", "NO_SIGNAL", "NOT_DUE") for r in preceding):
                raise AccountRejected("higher cohort ranks are not unlocked")
            state = decision_status if capacity <= cohort["single_atoms"] else "CAPACITY_LOCKED"
            con.execute("INSERT INTO cohort_streams VALUES(?,?,?,?,?,0,0,NULL,?)", (key,stream,rank,state,capacity,_json(dict(receipt))))
            self._event(con, "COHORT_CAPACITY_GATE", {"cohort_id": key, "stream": stream,
                "state": state, "capacity_usd": _usd(capacity), "receipt": dict(receipt)}, stamp)
        return self.cohort_state(key)

    def record_cohort_skip(self, cohort_id, stream, decision_status, receipt, now=None):
        """Persist a v5 ineligible stream without granting fabricated fill credit.

        NO_SIGNAL carries the validated model prediction; NOT_DUE is verified
        from the UTC schedule. A price skip carries the full observed own-side
        book, bound fee quote and validated signal. Missing data is never a skip.
        """
        current_time = lambda: _now(now() if callable(now) else now)
        stamp, boundary = current_time(), _ts(cohort_id)
        key = _iso(boundary)
        _safe(receipt)
        if (stream not in ALL_STREAMS or decision_status not in ("NO_SIGNAL", "NOT_DUE", "PRICE_LIMIT_NOT_MET")
                or not isinstance(receipt, Mapping)):
            raise AccountRejected("explicit validated skip receipt required")
        clean = None
        if decision_status == "PRICE_LIMIT_NOT_MET":
            clean = self._validate_price_skip_receipt(stream, boundary, receipt, stamp)
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            stamp = current_time()
            cohort = con.execute("SELECT * FROM decision_cohorts WHERE cohort_id=?", (key,)).fetchone()
            if (not cohort or not boundary <= stamp < _ts(cohort["expires_at"]) or cohort["blocked_reason"]):
                raise AccountRejected("decision cohort unavailable")
            policy = json.loads(cohort["policy_json"])
            if decision_status == "PRICE_LIMIT_NOT_MET":
                clean = self._validate_price_skip_receipt(stream, boundary, receipt, stamp, cohort_policy=policy)
            if policy.get("waterfall_semantics") != V5_SEMANTICS:
                raise AccountRejected("eligible-signal skip is not valid under this cohort policy")
            if stream in _stage(policy, boundary)["disabled_streams"]:
                raise AccountRejected("stream is disabled by the frozen execution stage")
            rank = policy["order"].index(stream)+1
            due = int(boundary.timestamp()) % ALL_STREAMS[stream][1] == 0
            if decision_status == "NOT_DUE":
                if due or receipt.get("source") != "UTC_COHORT_SCHEDULE":
                    raise AccountRejected("stream is actually due or schedule proof is missing")
            elif not due:
                raise AccountRejected("signal decision stream is not due")
            if decision_status == "NO_SIGNAL":
                prediction = receipt.get("prediction")
                if (receipt.get("source") != "VALIDATED_MODEL_DECISION" or not isinstance(prediction, Mapping)
                        or prediction.get("stream") != stream or prediction.get("selected") is not False
                        or not prediction.get("model_version")
                        or any(not re.fullmatch(r"[0-9a-fA-F]{64}", str(prediction.get(k, "")))
                               for k in ("model_sha256", "manifest_sha256"))):
                    raise AccountRejected("validated no-signal model prediction required")
                expected = policy.get("deployment_manifest_sha256", {}).get(stream)
                if expected is not None and prediction["manifest_sha256"] != expected:
                    raise AccountRejected("no-signal model pin mismatch")
                if (not boundary <= _ts(prediction.get("decision_time")) <= stamp
                        or _ts(prediction.get("feature_bar_open"))+timedelta(minutes=5) != boundary):
                    raise AccountRejected("no-signal prediction time mismatch")
                score, threshold = _d(prediction.get("z_v4")), _d(prediction.get("threshold_frozen"))
                vol_guarded = prediction.get("volatility_regime_guarded") is True
                if threshold <= 0 or (abs(score) >= threshold and not vol_guarded) or prediction.get("side") != ("UP" if score > 0 else "DOWN"):
                    raise AccountRejected("no-signal selection does not match the frozen threshold")
            encoded = _json(dict(receipt))
            if multi_look_settings(policy) is not None:
                look = con.execute("SELECT * FROM cohort_book_looks WHERE cohort_id=? AND stream=? ORDER BY attempt DESC LIMIT 1",
                                   (key, stream)).fetchone()
                if decision_status == "PRICE_LIMIT_NOT_MET":
                    if (look is None or look["state"] != decision_status or look["next_retry_at"] is not None
                            or look["receipt_json"] != encoded):
                        raise AccountRejected("terminal price skip requires its completed final book look")
                elif look is not None:
                    raise AccountRejected("cannot replace a book look with an absent signal")
            previous = con.execute("SELECT * FROM cohort_streams WHERE cohort_id=? AND stream=?", (key,stream)).fetchone()
            if previous:
                if previous["state"] != decision_status or previous["receipt_json"] != encoded:
                    raise AccountRejected("conflicting decision cohort skip")
            else:
                if not self._higher_ranks_unlocked(con, key, rank, policy, v5=True):
                    raise AccountRejected("higher cohort ranks are not unlocked")
                con.execute("INSERT INTO cohort_streams VALUES(?,?,?,?,0,0,0,?,?)", (
                    key,stream,rank,decision_status,clean["signal_id"] if clean else None,encoded))
                self._event(con, "COHORT_STREAM_SKIPPED", {"cohort_id":key,"stream":stream,"state":decision_status,
                    "receipt":dict(receipt),"receipt_sha256":hashlib.sha256(encoded.encode()).hexdigest()}, stamp)
        return self.cohort_state(key)

    def book_look_state(self, cohort_id, stream):
        """Last durable observation attempt; waiting never grants rank credit."""
        with self._connection() as con:
            row = con.execute("SELECT * FROM cohort_book_looks WHERE cohort_id=? AND stream=? ORDER BY attempt DESC LIMIT 1",
                              (_iso(_ts(cohort_id)), stream)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["receipt"] = json.loads(result.pop("receipt_json"))
        return result

    @staticmethod
    def _book_look_context(con, key, stream, stamp):
        cohort = con.execute("SELECT * FROM decision_cohorts WHERE cohort_id=?", (key,)).fetchone()
        if (not cohort or cohort["blocked_reason"] or not _ts(key) <= stamp < _ts(cohort["expires_at"])):
            raise AccountRejected("decision cohort unavailable for a book look")
        policy = json.loads(cohort["policy_json"])
        settings = multi_look_settings(policy)
        if settings is None or policy.get("waterfall_semantics") != V5_SEMANTICS:
            raise AccountRejected("multi-look is not enabled for this cohort")
        if stream not in ALL_STREAMS or stream in _stage(policy, _ts(key))["disabled_streams"]:
            raise AccountRejected("book look stream is unavailable")
        if con.execute("SELECT 1 FROM cohort_streams WHERE cohort_id=? AND stream=?", (key, stream)).fetchone():
            raise AccountRejected("book look stream is already terminal")
        rank = policy["order"].index(stream) + 1
        if not Account._higher_ranks_unlocked(con, key, rank, policy, v5=True):
            raise AccountRejected("higher cohort ranks are not unlocked")
        return cohort, settings

    def begin_book_look(self, cohort_id, stream, signal, now=None):
        """Claim before HTTP. An interrupted claim is never retried or replayed."""
        current_time = lambda: _now(now() if callable(now) else now)
        stamp, key = current_time(), _iso(_ts(cohort_id))
        clean = self._validate_signal(signal, stamp)
        if clean["stream"] != stream or _ts(clean["window_start"]) != _ts(key):
            raise AccountRejected("book look signal cohort mismatch")
        digest = hashlib.sha256(_json(clean).encode()).hexdigest()
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            stamp = current_time()
            self._validate_signal(clean, stamp)
            cohort, settings = self._book_look_context(con, key, stream, stamp)
            old = con.execute("SELECT * FROM cohort_book_looks WHERE cohort_id=? AND stream=? ORDER BY attempt DESC LIMIT 1",
                              (key, stream)).fetchone()
            attempt = 0
            if old is not None:
                if old["signal_sha256"] != digest:
                    raise AccountRejected("book look signal changed within the cohort")
                if old["state"] != "PRICE_LIMIT_NOT_MET" or not old["next_retry_at"]:
                    raise AccountRejected("previous book look is not retryable")
                if stamp < _ts(old["next_retry_at"]):
                    raise AccountRejected("price recheck is not yet due")
                if stamp > _ts(key) + timedelta(seconds=18.5):
                    raise AccountRejected("price recheck start deadline expired")
                attempt = old["attempt"] + 1
                if attempt > settings["max_price_rechecks"]:
                    raise AccountRejected("price recheck count exhausted")
            con.execute("INSERT INTO cohort_book_looks VALUES(?,?,?,?,NULL,'REQUESTED',NULL,?,?)",
                        (key, stream, attempt, _iso(stamp), digest, _json({"signal": clean})))
            self._event(con, "COHORT_BOOK_LOOK_REQUESTED", {"cohort_id": key, "stream": stream,
                        "attempt": attempt, "signal_sha256": digest}, stamp)
        return self.book_look_state(key, stream)

    def finish_book_look(self, cohort_id, stream, attempt, outcome, receipt, now=None):
        """Record a real validated response, with no cashflow or invented fill."""
        current_time = lambda: _now(now() if callable(now) else now)
        stamp, key = current_time(), _iso(_ts(cohort_id))
        _safe(receipt)
        if outcome not in ("PRICE_LIMIT_NOT_MET", "EXECUTABLE_BOOK"):
            raise AccountRejected("unsupported book look outcome")
        clean = self._validate_signal(receipt.get("signal"), stamp)
        if clean["stream"] != stream or _ts(clean["window_start"]) != _ts(key):
            raise AccountRejected("book look signal cohort mismatch")
        if outcome == "PRICE_LIMIT_NOT_MET":
            self._validate_price_skip_receipt(stream, _ts(key), receipt, stamp)
        digest = hashlib.sha256(_json(clean).encode()).hexdigest()
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            stamp = current_time()
            self._validate_signal(clean, stamp)
            cohort, settings = self._book_look_context(con, key, stream, stamp)
            cohort_policy = json.loads(cohort["policy_json"])
            if outcome == "PRICE_LIMIT_NOT_MET":
                self._validate_price_skip_receipt(stream, _ts(key), receipt, stamp, cohort_policy=cohort_policy)
            row = con.execute("SELECT * FROM cohort_book_looks WHERE cohort_id=? AND stream=? AND attempt=?",
                              (key, stream, attempt)).fetchone()
            if (row is None or row["state"] != "REQUESTED" or row["signal_sha256"] != digest
                    or stamp < _ts(row["requested_at"])):
                raise AccountRejected("book look claim changed or already completed")
            # A response received before this claim is cached evidence, not a
            # new look. Its venue source may be earlier but must pass freshness.
            received = _ts(receipt.get("orderbook", {}).get("received_at"))
            if not _ts(row["requested_at"]) <= received <= stamp:
                raise AccountRejected("book look response precedes its request")
            retry_at = None
            expiry = min(_ts(cohort["expires_at"]), _ts(clean["valid_until"]))
            proposed = stamp + timedelta(seconds=2.5)
            limits = _policy_risk_limits(cohort_policy)
            expected_min_buy = (limits.get("stream_min_buy_price") or {}).get(stream, limits["min_buy_price"])
            min_limit = _d(receipt.get("min_buy_price", str(expected_min_buy)))
            positive_asks = [_d(a["price"]) for a in receipt.get("orderbook", {}).get("asks", []) if _d(a.get("size", 0)) > 0]
            below_min_buy = bool(positive_asks and min(positive_asks) < min_limit)
            if (outcome == "PRICE_LIMIT_NOT_MET" and not below_min_buy
                    and attempt < settings["max_price_rechecks"]
                    and (expiry-stamp).total_seconds() >= 3
                    and proposed <= _ts(key)+timedelta(seconds=18.5)):
                retry_at = _iso(proposed)
            encoded = _json(dict(receipt))
            con.execute("UPDATE cohort_book_looks SET completed_at=?,state=?,next_retry_at=?,receipt_json=? "
                        "WHERE cohort_id=? AND stream=? AND attempt=?",
                        (_iso(stamp), outcome, retry_at, encoded, key, stream, attempt))
            self._event(con, "COHORT_BOOK_LOOK_COMPLETED", {"cohort_id": key, "stream": stream,
                        "attempt": attempt, "state": outcome, "next_retry_at": retry_at,
                        "receipt_sha256": hashlib.sha256(encoded.encode()).hexdigest()}, stamp)
        return self.book_look_state(key, stream)

    def fail_book_look(self, cohort_id, stream, attempt, reason, now=None):
        """Seal an interrupted, invalid or late attempt; never make it retryable."""
        current_time = lambda: _now(now() if callable(now) else now)
        stamp, key = current_time(), _iso(_ts(cohort_id))
        _safe(reason)
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            stamp = current_time()
            row = con.execute("SELECT * FROM cohort_book_looks WHERE cohort_id=? AND stream=? AND attempt=?",
                              (key, stream, attempt)).fetchone()
            if row is None or (row["state"] not in ("REQUESTED", "EXECUTABLE_BOOK")
                               and not (row["state"] == "PRICE_LIMIT_NOT_MET" and row["next_retry_at"])):
                return False
            if stamp < _ts(row["requested_at"]):
                raise AccountRejected("book look failure time precedes request")
            encoded = _json({"reason": reason, "prior_receipt": json.loads(row["receipt_json"])})
            con.execute("UPDATE cohort_book_looks SET state='FAILED',completed_at=?,next_retry_at=NULL,receipt_json=? "
                        "WHERE cohort_id=? AND stream=? AND attempt=?", (_iso(stamp), encoded, key, stream, attempt))
            self._event(con, "COHORT_BOOK_LOOK_FAILED", {"cohort_id": key, "stream": stream,
                        "attempt": attempt, "reason": reason}, stamp)
        return True

    def _validate_price_skip_receipt(self, stream, boundary, receipt, stamp, cohort_policy=None):
        """One admission path for terminal skips and retryable price misses."""
        clean = self._validate_signal(receipt.get("signal"), stamp)
        if clean["stream"] != stream or _ts(clean["window_start"]) != boundary:
            raise AccountRejected("price skip signal cohort mismatch")
        book, fee = receipt.get("orderbook"), receipt.get("fee_quote")
        if (receipt.get("source") != "OBSERVED_PRICE_LIMIT_CHECK" or not isinstance(book, Mapping)
                or book.get("kind") != "OBSERVED_PUBLIC_ORDERBOOK" or not book.get("source")
                or not isinstance(fee, Mapping)):
            raise AccountRejected("observed price and fee evidence required")
        if any(book.get(k) != clean[k] for k in ("market_id", "underlying", "outcome")):
            raise AccountRejected("price skip book identity mismatch")
        if any(_ts(book.get(k)) != _ts(clean[k]) for k in ("window_start", "window_end")):
            raise AccountRejected("price skip contract window mismatch")
        source, received = _ts(book.get("observed_at")), _ts(book.get("received_at"))
        if not _ts(clean["decision_at"]) <= source <= received <= stamp or (stamp-source).total_seconds() > 2:
            raise AccountRejected("price skip book is stale or future")
        skip_limit = _d(receipt.get("limit_price", ".515"))
        if skip_limit not in (Decimal(".515"), Decimal(".520")):
            raise AccountRejected("unsupported price skip limit")
        min_limit = _d(receipt.get("min_buy_price", "0"))
        if not Decimal("0") <= min_limit <= Decimal(".500"):
            raise AccountRejected("unsupported min buy price limit")
        self._fee_per_share(fee, skip_limit, stamp, clean["market_id"], clean["outcome"])
        asks = book.get("asks")
        if not isinstance(asks, list) or not asks:
            raise AccountRejected("nonempty observed asks required for price skip")
        positive = []
        for ask in asks:
            price, size = _d(ask.get("price")), _d(ask.get("size"))
            if not 0 < price < 1 or size < 0:
                raise AccountRejected("invalid price skip ask level")
            if size > 0:
                positive.append(price)
        if not positive or (min_limit <= min(positive) <= skip_limit):
            raise AccountRejected("price skip requires all positive asks above the limit")
        if cohort_policy is not None:
            limits = _policy_risk_limits(cohort_policy)
            expected_min_buy = (limits.get("stream_min_buy_price") or {}).get(stream, limits["min_buy_price"])
            if "min_buy_price" in receipt or (expected_min_buy > 0 and min(positive) < skip_limit):
                if min_limit != expected_min_buy:
                    raise AccountRejected("price skip min_buy_price does not match cohort policy")
        return clean

    def _validate_signal(self, signal, now):
        _safe(signal)
        if not isinstance(signal, Mapping):
            raise AccountRejected("signal object required")
        required = ("signal_id", "stream", "underlying", "outcome", "market_id", "model_id", "model_sha256",
                    "feature_available_at", "decision_at", "valid_until", "window_start", "window_end")
        if any(k not in signal or signal[k] in (None, "") for k in required):
            raise AccountRejected("incomplete signal or model metadata")
        if signal.get("validation_status") != "INPUTS_VALIDATED":
            raise AccountRejected("validated model inputs required")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(signal["model_sha256"])):
            raise AccountRejected("model SHA256 required")
        spec = ALL_STREAMS.get(signal["stream"])
        if spec is None or signal["underlying"] != spec[0] or signal["outcome"] not in ("UP", "DOWN"):
            raise AccountRejected("unknown stream, underlying or outcome")
        available, decision, expiry, start, end = (_ts(signal[k]) for k in (
            "feature_available_at", "decision_at", "valid_until", "window_start", "window_end"))
        with self._connection() as con:
            account_start = _ts(con.execute("SELECT started_at FROM account").fetchone()[0])
        if not (account_start <= decision <= now and available <= decision and start <= decision < end):
            raise AccountRejected("signal time or feature availability invalid")
        if not decision < expiry <= end or not now < expiry:
            raise AccountRejected("signal expired or expiry invalid")
        if (end - start).total_seconds() != spec[1] or start.timestamp() % spec[1] != 0:
            raise AccountRejected("incorrect contract window")
        if (now - decision).total_seconds() > 20:
            raise AccountRejected("stale decision")
        # Normalize timestamps before deduplication; never store arbitrary extras.
        clean = {k: signal[k] for k in required}
        clean["validation_status"] = "INPUTS_VALIDATED"
        for k in ("feature_available_at", "decision_at", "valid_until", "window_start", "window_end"):
            clean[k] = _iso(_ts(clean[k]))
        return clean

    def record_signal(self, signal, now=None):
        stamp = _now(now)
        clean = self._validate_signal(signal, stamp)
        payload = _json(clean)
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT payload FROM signals WHERE signal_id=?", (clean["signal_id"],)).fetchone()
            if old:
                if old[0] != payload:
                    raise AccountRejected("signal identifier reused with different content")
                return {"signal_id": clean["signal_id"], "status": "DUPLICATE"}
            con.execute("INSERT INTO signals VALUES(?,?,?)", (clean["signal_id"], payload, _iso(stamp)))
            self._event(con, "SIGNAL_ADMITTED", clean, stamp)
        return {"signal_id": clean["signal_id"], "status": "ADMITTED"}

    @staticmethod
    def _fee_per_share(quote, price, now, market_id=None, outcome=None):
        _safe(quote)
        if not quote.get("source") or "observed_at" not in quote:
            raise AccountRejected("fee provenance required")
        age = (now - _ts(quote["observed_at"])).total_seconds()
        if not 0 <= age <= 300:
            raise AccountRejected("fee quote is future or stale")
        model = quote.get("fee_model", quote.get("model"))
        if model == "fixed_bps" and quote.get("label") == "SIMULATED_ASSUMPTION":
            bps = _d(quote.get("bps"))
            if not 0 <= bps <= 10000:
                raise AccountRejected("invalid fee bps")
            return price * bps / 10000
        if model == "polymarket_crypto":
            if (quote.get("market_id") != market_id or quote.get("outcome") != outcome
                    or quote.get("exponent") != 1 or quote.get("taker_only") is not True
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", str(quote.get("source_sha256", "")))):
                raise AccountRejected("unsupported or unbound Polymarket fee schedule")
            rate = _d(quote.get("rate"))
            if not 0 <= rate <= 1:
                raise AccountRejected("invalid Polymarket rate")
            return rate * price * (1 - price)
        raise AccountRejected("unsupported fee model or undisclosed fee assumption")

    def place_paper_buy(self, signal, orderbook, fee_quote, now=None, stake=150, limit_price=.515, order_id=None,
                        allocation_authorization=None):
        _safe({"signal": signal, "orderbook": orderbook, "fee_quote": fee_quote, "order_id": order_id,
               "allocation_authorization": allocation_authorization})
        if isinstance(signal, str):
            _safe(signal)
            with self._connection() as con:
                row = con.execute("SELECT payload FROM signals WHERE signal_id=?", (signal,)).fetchone()
            if row is None:
                raise AccountRejected("unknown signal")
            signal = json.loads(row[0])
        if not isinstance(signal, Mapping) or not isinstance(signal.get("signal_id"), str) or not signal["signal_id"]:
            raise AccountRejected("signal identifier required")
        identifier = str(order_id or ("paper:" + signal["signal_id"]))
        request_hash = hashlib.sha256(_json({"signal": dict(signal), "orderbook": orderbook,
                                             "fee_quote": fee_quote, "order_id": identifier,
                                             "stake": str(_d(stake)), "limit_price": str(_d(limit_price)),
                                             "allocation_authorization": allocation_authorization}).encode()).hexdigest()
        # An identical retry returns its old result even after the quote expires.
        # A changed request is a conflict, never a new use of the same order ID.
        with self._connection() as con:
            old = con.execute("SELECT * FROM orders WHERE signal_id=? OR order_id=?",
                              (signal["signal_id"], identifier)).fetchone()
            if old:
                previous = json.loads(old["payload"])
                if old["signal_id"] != signal["signal_id"] or old["order_id"] != identifier or previous.get("request_sha256") != request_hash:
                    raise AccountRejected("conflicting reuse of signal or order identifier")
                return {**previous, "status": old["status"], "duplicate": True}
        # A fixed timestamp remains available to explicit offline fixtures.
        # Real-time callers pass their calibrated UTC upper-bound clock itself,
        # not a timestamp captured before database waits or fill computation.
        def current_time():
            return _now(now() if callable(now) else now)
        stamp = current_time()
        clean = self._validate_signal(signal, stamp)
        authorization = allocation_authorization or {}
        _safe(authorization)
        if authorization.get("policy") != "RANKED_CAPACITY_WATERFALL" or not authorization.get("plan_id"):
            raise AccountRejected("ranked capacity allocation authorization required")
        if authorization.get("signal_id") != clean["signal_id"] or authorization.get("market_id") != clean["market_id"]:
            raise AccountRejected("allocation authorization does not match signal")
        if not isinstance(authorization.get("rank"), int) or not 1 <= authorization["rank"] <= len(ALL_STREAMS):
            raise AccountRejected("allocation rank required")
        if not _ts(authorization.get("planned_at")) <= stamp < _ts(authorization.get("valid_until")) <= _ts(clean["valid_until"]):
            raise AccountRejected("allocation authorization is future or expired")
        if _d(stake) > _d(authorization.get("approved_stake_usd")):
            raise AccountRejected("stake exceeds ranked allocation")
        if not _ts(clean["decision_at"]) < stamp:
            raise AccountRejected("fill must be strictly after decision")
        for key in ("market_id", "underlying", "outcome"):
            if orderbook.get(key) != clean[key]:
                raise AccountRejected("orderbook does not match signal market")
        for key in ("window_start", "window_end"):
            if _ts(orderbook.get(key)) != _ts(clean[key]):
                raise AccountRejected("orderbook contract window mismatch")
        if orderbook.get("kind") != "OBSERVED_PUBLIC_ORDERBOOK" or not orderbook.get("source"):
            raise AccountRejected("observed public orderbook provenance required")
        observed, received = _ts(orderbook.get("observed_at")), _ts(orderbook.get("received_at"))
        if not _ts(clean["decision_at"]) <= observed <= received <= stamp:
            raise AccountRejected("orderbook is future or predates decision")
        if (stamp - observed).total_seconds() > 5:
            raise AccountRejected("stale orderbook")
        limit = _d(limit_price)
        budget = _atoms(_d(stake))
        pre_limits = _policy_risk_limits(getattr(self, "policy", None))
        if not 0 < limit <= Decimal(".515") or budget <= 0 or (authorization.get("cohort_id") is None and budget > pre_limits["max_order_atoms"]):
            raise AccountRejected("order price or stake exceeds limit")
        asks = orderbook.get("asks")
        if not isinstance(asks, list):
            raise AccountRejected("ask levels required")
        levels = []
        for ask in asks:
            price, size = _d(ask.get("price")), _d(ask.get("size"))
            if not 0 < price < 1 or size < 0:
                raise AccountRejected("invalid orderbook level")
            levels.append((price, size))
        # Fee input must be valid even if the book has no eligible liquidity.
        self._fee_per_share(fee_quote, limit, stamp, clean["market_id"], clean["outcome"])
        aggregated = {}
        for price, available in levels:
            aggregated[price] = aggregated.get(price, Decimal(0)) + available
        book_hash = hashlib.sha256(_json({"market": clean["market_id"], "outcome": clean["outcome"],
                                         "source": orderbook["source"], "observed_at": _iso(observed),
                                         "asks": [[str(p), str(v)] for p, v in sorted(aggregated.items())]}).encode()).hexdigest()
        def recheck_admission():
            nonlocal stamp
            current = current_time()
            if current < stamp:
                raise AccountRejected("admission clock moved backwards")
            decision, expiry = _ts(clean["decision_at"]), _ts(clean["valid_until"])
            if not decision < current < expiry or (current-decision).total_seconds() > 20:
                raise AccountRejected("signal expired before paper acceptance")
            if not _ts(authorization["planned_at"]) <= current < _ts(authorization["valid_until"]):
                raise AccountRejected("allocation authorization expired before paper acceptance")
            if authorization.get("cohort_id") is not None:
                boundary = _ts(authorization["cohort_id"])
                if not boundary <= current < boundary+timedelta(seconds=20):
                    raise AccountRejected("decision cohort expired before paper acceptance")
            if not decision <= observed <= received <= current or (current-observed).total_seconds() > 5:
                raise AccountRejected("orderbook stale before paper acceptance")
            self._fee_per_share(fee_quote, limit, current, clean["market_id"], clean["outcome"])
            stamp = current
            return current
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT * FROM orders WHERE signal_id=? OR order_id=?",
                              (clean["signal_id"], identifier)).fetchone()
            if old:
                if old["signal_id"] != clean["signal_id"] or old["order_id"] != identifier or json.loads(old["payload"]).get("request_sha256") != request_hash:
                    raise AccountRejected("order or signal identifier already used")
                return {**json.loads(old["payload"]), "status": old["status"], "duplicate": True}
            recheck_admission()  # BEGIN IMMEDIATE may have waited for another writer.
            cohort = None
            cohort_policy = None
            cohort_key = authorization.get("cohort_id")
            # A caller cannot strip cohort_id to escape an already recorded
            # stage through the legacy standalone-preview entry point. Only
            # policies effective at this signal's window affect new purchases;
            # old orders and their official settlements remain untouched.
            latest = con.execute("SELECT policy_json FROM decision_cohorts WHERE cohort_id<=? ORDER BY cohort_id DESC LIMIT 1",
                                 (clean["window_start"],)).fetchone()
            active_stage = _stage(json.loads(latest["policy_json"]), _ts(clean["window_start"])) if latest else None
            if active_stage and active_stage["configured"]:
                if clean["stream"] in active_stage["disabled_streams"]:
                    raise AccountRejected("stream is disabled by the frozen execution stage")
                if cohort_key is None:
                    raise AccountRejected("staged execution requires a persisted decision cohort")
            if authorization.get("execution_stage") is not None and cohort_key is None:
                raise AccountRejected("staged execution requires a persisted decision cohort")
            if cohort_key is not None:
                cohort_key = _iso(_ts(cohort_key))
                cohort = con.execute("SELECT * FROM decision_cohorts WHERE cohort_id=?", (cohort_key,)).fetchone()
                if (cohort is None or cohort["plan_id"] != authorization["plan_id"]
                        or cohort["policy_version"] != authorization.get("policy_version")
                        or cohort["blocked_reason"] or not stamp < _ts(cohort["expires_at"])):
                    raise AccountRejected("cohort authorization unavailable or expired")
                cohort_policy = json.loads(cohort["policy_json"])
                if clean["stream"] in _stage(cohort_policy, _ts(cohort_key))["disabled_streams"]:
                    raise AccountRejected("stream is disabled by the frozen execution stage")
                v5 = cohort_policy.get("waterfall_semantics") == V5_SEMANTICS
                ranking = cohort_policy["order"]
                if v5 and (authorization.get("waterfall_semantics") != V5_SEMANTICS
                           or budget != _atoms(_d(authorization["approved_stake_usd"]))):
                    raise AccountRejected("approved-budget cohort authorization mismatch")
                if multi_look_settings(cohort_policy) is not None:
                    look = con.execute("SELECT * FROM cohort_book_looks WHERE cohort_id=? AND stream=? ORDER BY attempt DESC LIMIT 1",
                                       (cohort_key, clean["stream"])).fetchone()
                    proof = json.loads(look["receipt_json"]) if look is not None else {}
                    if (look is None or look["state"] != "EXECUTABLE_BOOK"
                            or look["attempt"] != authorization.get("book_look_attempt")
                            or look["signal_sha256"] != hashlib.sha256(_json(clean).encode()).hexdigest()
                            or proof.get("orderbook") != orderbook or proof.get("fee_quote") != fee_quote):
                        raise AccountRejected("paper acceptance does not match the claimed book look")
                if ranking[authorization["rank"]-1] != clean["stream"]:
                    raise AccountRejected("cohort ranking mismatch")
                if _ts(clean["window_start"]) != _ts(cohort_key):
                    raise AccountRejected("signal window is outside its decision cohort")
                if budget > min(cohort["single_atoms"], cohort["budget_atoms"]-cohort["used_atoms"]):
                    raise AccountRejected("persistent cohort budget exceeded")
                if con.execute("SELECT 1 FROM cohort_streams WHERE cohort_id=? AND stream=?", (cohort_key,clean["stream"])).fetchone():
                    raise AccountRejected("stream already consumed this decision cohort")
                if not self._higher_ranks_unlocked(con, cohort_key, authorization["rank"], cohort_policy, v5=v5):
                    raise AccountRejected("higher cohort capacity is not fully satisfied")
                cohort_capacity = _atoms(_d(authorization.get("cohort_capacity_usd")))
                if cohort_capacity <= 0 or budget > cohort_capacity:
                    raise AccountRejected("cohort observed capacity invalid")
            limits = _policy_risk_limits(cohort_policy if cohort_policy is not None else (json.loads(latest["policy_json"]) if latest else getattr(self, "policy", None)))
            if budget > limits["max_order_atoms"]:
                raise AccountRejected("order price or stake exceeds limit")
            rank_use = con.execute("SELECT 1 FROM allocation_consumption WHERE plan_id=? AND rank=?",
                                   (authorization["plan_id"], authorization["rank"])).fetchone()
            if rank_use:
                raise AccountRejected("allocation rank already consumed")
            predecessors = authorization.get("preceding_allocations", [])
            if not isinstance(predecessors, list):
                raise AccountRejected("preceding allocation requirements invalid")
            seen_predecessors = set()
            for required in predecessors:
                if (not isinstance(required, Mapping) or not isinstance(required.get("signal_id"), str)
                        or required["signal_id"] in seen_predecessors
                        or not isinstance(required.get("rank"), int) or not 1 <= required["rank"] < authorization["rank"]):
                    raise AccountRejected("preceding allocation requirements invalid")
                seen_predecessors.add(required["signal_id"])
                threshold = _atoms(_d(required.get("required_cost_usd")))
                if threshold <= 0:
                    raise AccountRejected("preceding allocation amount invalid")
                consumed = con.execute("SELECT * FROM allocation_consumption WHERE plan_id=? AND signal_id=? AND rank=?",
                                       (authorization["plan_id"], required["signal_id"], required["rank"])).fetchone()
                if consumed is None or consumed["approved_atoms"] != threshold or consumed["spent_atoms"] + 2 < threshold:
                    raise AccountRejected("higher ranked capacity has not actually filled in this plan")
            previous = con.execute("SELECT payload FROM signals WHERE signal_id=?", (clean["signal_id"],)).fetchone()
            if previous is not None and previous[0] != _json(clean):
                raise AccountRejected("signal identifier reused with different content")
            if previous is None:
                con.execute("INSERT INTO signals VALUES(?,?,?)", (clean["signal_id"], _json(clean), _iso(stamp)))
                self._event(con, "SIGNAL_ADMITTED", clean, stamp)
            cash = con.execute("SELECT cash_atoms FROM account WHERE id=1").fetchone()[0]
            open_rows = con.execute("SELECT underlying,cost_atoms,fee_atoms,window_end FROM orders WHERE status='OPEN'").fetchall()
            open_risk = sum(r["cost_atoms"] + r["fee_atoms"] for r in open_rows)
            coin_risk = sum(r["cost_atoms"] + r["fee_atoms"] for r in open_rows if r["underlying"] == clean["underlying"])
            active_risk = sum(r["cost_atoms"] + r["fee_atoms"] for r in open_rows if _ts(r["window_end"]) > stamp)
            active_coin_risk = sum(r["cost_atoms"] + r["fee_atoms"] for r in open_rows
                                   if r["underlying"] == clean["underlying"] and _ts(r["window_end"]) > stamp)
            if budget > min(cash, limits["max_open_risk_atoms"] - open_risk,
                            limits["max_underlying_open_risk_atoms"] - coin_risk,
                            limits["max_active_window_risk_atoms"] - active_risk,
                            limits["max_underlying_active_risk_atoms"] - active_coin_risk):
                raise AccountRejected("cash or concurrent exposure limit exceeded")
            remaining = budget
            stream_mins = limits.get("stream_min_buy_price") or {}
            min_buy = stream_mins.get(clean["stream"], limits.get("min_buy_price", Decimal("0")))
            positive_available_prices = [
                price for price, quoted_available in aggregated.items()
                if quoted_available - (_d((con.execute("SELECT quantity FROM consumed_book_depth WHERE book_hash=? AND price=?",
                                                       (book_hash, str(price))).fetchone() or [0])[0])) > 0
            ]
            if positive_available_prices and min(positive_available_prices) < min_buy:
                raise AccountRejected("order price or stake exceeds limit")
            fills, cost, fees, quantity = [], 0, 0, Decimal(0)
            for price, quoted_available in sorted(aggregated.items()):
                consumed = con.execute("SELECT quantity FROM consumed_book_depth WHERE book_hash=? AND price=?",
                                       (book_hash, str(price))).fetchone()
                previous_quantity = _d(consumed[0]) if consumed else Decimal(0)
                available = max(Decimal(0), quoted_available - previous_quantity)
                if price > limit or price < min_buy or remaining <= 0 or available == 0:
                    continue
                fee_ps = self._fee_per_share(fee_quote, price, stamp, clean["market_id"], clean["outcome"])
                qty = min(available, Decimal(remaining) / MICRO / (price + fee_ps)).quantize(QTY_STEP, rounding=ROUND_DOWN)
                # Reserve conservatively rounded currency atoms for each fill.
                while qty > 0:
                    c, f = _atoms(qty * price), _atoms(qty * fee_ps)
                    if c + f <= remaining:
                        break
                    qty -= QTY_STEP
                if qty <= 0:
                    continue
                fills.append({"price": str(price), "quantity": str(qty), "cost_usd": _usd(c), "fee_usd": _usd(f)})
                cost += c
                fees += f
                quantity += qty
                remaining -= c + f
                con.execute("INSERT INTO consumed_book_depth VALUES(?,?,?) ON CONFLICT(book_hash,price) DO UPDATE SET quantity=excluded.quantity",
                            (book_hash, str(price), str(previous_quantity + qty)))
            status = "OPEN" if quantity else "NO_FILL"
            stream_state = None
            if cohort is not None:
                if v5:
                    stream_state = "FILLED_BUDGET" if quantity > 0 and cost+fees+2 >= budget else "PARTIAL"
                else:
                    stream_state = "FILLED_CAPACITY" if cost+fees+2 >= cohort_capacity else ("PARTIAL_CAPACITY" if quantity else "NO_FILL")
            # All earlier writes are still uncommitted. A computation overrun
            # rolls back the signal, consumed depth and every ledger mutation.
            recheck_admission()
            result = {"order_id": identifier, "signal_id": clean["signal_id"], "market_id": clean["market_id"],
                      "underlying": clean["underlying"], "outcome": clean["outcome"], "status": status,
                      "quantity": str(quantity), "cost_usd": _usd(cost), "fee_usd": _usd(fees),
                      "unfilled_budget_usd": _usd(remaining), "fills": fills, "fill_at": _iso(stamp),
                      "execution": "SIMULATED_IOC_FROM_OBSERVED_ASKS", "live_enabled": False,
                      "book_source": orderbook["source"], "book_observed_at": _iso(observed), "fee_quote": dict(fee_quote),
                      "allocation_authorization": dict(authorization), "request_sha256": request_hash,
                      "book_sha256": book_hash,
                      "fee_quote_sha256": hashlib.sha256(_json(dict(fee_quote)).encode()).hexdigest(),
                      "venue_market": orderbook.get("venue_market"),
                      "book_received_at_local": orderbook.get("received_at_local"),
                      "clock_projection": orderbook.get("clock_projection"),
                      "cohort_stream_state": stream_state}
            con.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                identifier, clean["signal_id"], clean["market_id"], clean["underlying"], clean["outcome"],
                str(quantity), cost, fees, status, _iso(stamp), clean["window_end"], _json(result)))
            con.execute("UPDATE account SET cash_atoms=cash_atoms-? WHERE id=1", (cost + fees,))
            con.execute("INSERT INTO allocation_consumption VALUES(?,?,?,?,?,?)", (
                authorization["plan_id"], authorization["rank"], clean["signal_id"],
                _atoms(_d(authorization["approved_stake_usd"])), cost+fees, identifier))
            if cohort is not None:
                con.execute("INSERT INTO cohort_streams VALUES(?,?,?,?,?,?,?,?,?)", (
                    cohort_key, clean["stream"], authorization["rank"], stream_state, cohort_capacity,
                    _atoms(_d(authorization["approved_stake_usd"])), cost+fees, clean["signal_id"],
                    _json({"book_sha256": book_hash, "fee_quote_sha256": result["fee_quote_sha256"], "order_id": identifier})))
                con.execute("UPDATE decision_cohorts SET used_atoms=used_atoms+? WHERE cohort_id=?", (cost+fees,cohort_key))
            self._event(con, "PAPER_IOC", result, stamp)
            self._snapshot(con, stamp)
            # Last application-level guard before SQLite commit. Physical disk
            # commit duration is not a claimed execution-time guarantee.
            recheck_admission()
        return result

    def settle(self, market_id, resolution, now=None):
        stamp = _now(now)
        _safe({"market_id": market_id, "resolution": resolution})
        if resolution.get("market_id") != market_id or resolution.get("kind") != "OBSERVED_OFFICIAL_RESOLUTION":
            raise AccountRejected("official market resolution required")
        if not resolution.get("source") or not resolution.get("resolution_id"):
            raise AccountRejected("resolution source and identifier required")
        outcome = resolution.get("outcome")
        if outcome not in ("UP", "DOWN", "VOID"):
            raise AccountRejected("unsupported resolution outcome")
        if outcome == "VOID" and resolution.get("refund_rule") != "COST_AND_FEES":
            raise AccountRejected("explicit cost-and-fees refund rule required")
        resolved, received = _ts(resolution.get("resolved_at")), _ts(resolution.get("received_at"))
        if not resolved <= received <= stamp:
            raise AccountRejected("resolution cannot be from the future")
        canonical = _json(dict(resolution))
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT payload FROM resolutions WHERE market_id=?", (market_id,)).fetchone()
            if old:
                stable_fields = ("market_id", "kind", "source", "resolution_id", "outcome", "resolved_at", "refund_rule")
                old_payload = json.loads(old[0])
                if any(old_payload.get(k) != resolution.get(k) for k in stable_fields):
                    raise AccountRejected("conflicting market resolution")
                return {"market_id": market_id, "status": "ALREADY_SETTLED"}
            orders = con.execute("SELECT * FROM orders WHERE market_id=? AND status='OPEN'", (market_id,)).fetchall()
            if not orders:
                raise AccountRejected("no open paper positions for market")
            if any(resolved < _ts(o["window_end"]) for o in orders):
                raise AccountRejected("resolution precedes contract close")
            returned = pnl = 0
            settlements = []
            for order in orders:
                invested = order["cost_atoms"] + order["fee_atoms"]
                if outcome == "VOID":
                    payout, status = invested, "VOID"
                elif order["outcome"] == outcome:
                    payout = int((_d(order["quantity"]) * MICRO).to_integral_value(rounding=ROUND_DOWN))
                    status = "WON"
                else:
                    payout, status = 0, "LOST"
                returned += payout
                pnl += payout - invested
                con.execute("UPDATE orders SET status=? WHERE order_id=?", (status, order["order_id"]))
                settlements.append({"order_id": order["order_id"], "status": status, "payout_usd": _usd(payout), "realized_pnl_usd": _usd(payout-invested)})
            con.execute("UPDATE account SET cash_atoms=cash_atoms+?,realized_atoms=realized_atoms+? WHERE id=1", (returned, pnl))
            a = con.execute("SELECT * FROM account WHERE id=1").fetchone()
            balance = a["initial_atoms"] + a["realized_atoms"]
            peak = max(a["peak_realized_atoms"], balance)
            drawdown = min(a["max_realized_dd"], (balance - peak) / peak if peak else 0.0)
            con.execute("UPDATE account SET peak_realized_atoms=?,max_realized_dd=? WHERE id=1", (peak, drawdown))
            con.execute("INSERT INTO resolutions VALUES(?,?,?)", (market_id, canonical, _iso(stamp)))
            result = {"market_id": market_id, "status": "SETTLED", "resolution": dict(resolution),
                      "payout_usd": _usd(returned), "realized_pnl_usd": _usd(pnl), "orders": settlements}
            self._event(con, "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER", result, stamp)
            self._snapshot(con, stamp)
        return result

    def overview(self, now=None, policy=None):
        stamp = _now(now)
        with self._connection() as con:
            a = dict(con.execute("SELECT * FROM account WHERE id=1").fetchone())
            runtime = dict(con.execute("SELECT * FROM runtime WHERE id=1").fetchone())
            counts = dict(con.execute("SELECT status,COUNT(*) FROM orders GROUP BY status").fetchall())
            positions = [dict(o) for o in con.execute("SELECT order_id,signal_id,market_id,underlying,outcome,quantity,cost_atoms,fee_atoms,fill_at,window_end FROM orders WHERE status='OPEN'")]
            signal_count = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            latest_policy_row = con.execute("SELECT policy_json FROM decision_cohorts ORDER BY cohort_id DESC LIMIT 1").fetchone() if policy is None else None
        effective_policy = policy if policy is not None else (
            json.loads(latest_policy_row[0]) if latest_policy_row else getattr(self, "policy", None)
        )
        try:
            limits = _policy_risk_limits(effective_policy)
        except AccountRejected:
            limits = _policy_risk_limits(None)

        def _num(d):
            return int(d) if d == int(d) else float(d)

        default_order = min(limits["max_order_usd"], (_d(_usd(a["initial_atoms"])) * limits["single_stream_equity_fraction"]).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
        locked = sum(p["cost_atoms"] + p["fee_atoms"] for p in positions)
        for p in positions:
            p["cost_usd"] = _usd(p.pop("cost_atoms"))
            p["fee_usd"] = _usd(p.pop("fee_atoms"))
        wins, losses = counts.get("WON", 0), counts.get("LOST", 0)
        return {"classification": "EXPLORATORY_FORWARD_PAPER", "live_enabled": False,
                "currency": a["currency"], "started_at": a["started_at"], "as_of": _iso(stamp),
                "initial_capital": _usd(a["initial_atoms"]), "cash": _usd(a["cash_atoms"]),
                "locked_cost": _usd(locked), "realized_pnl": _usd(a["realized_atoms"]),
                "realized_return_pct": 100*a["realized_atoms"]/a["initial_atoms"],
                "realized_balance": _usd(a["initial_atoms"] + a["realized_atoms"]),
                "equity": None if positions else _usd(a["cash_atoms"]),
                "unrealized_pnl": None if positions else 0.0,
                "valuation_status": "UNVALUED_OPEN_POSITIONS" if positions else "CASH_ONLY",
                "max_realized_drawdown_pct": 100*a["max_realized_dd"],
                "mark_to_market_drawdown_pct": None,
                "wins": wins, "losses": losses, "voids": counts.get("VOID", 0),
                "win_rate_pct": 100*wins/(wins+losses) if wins+losses else None,
                "signal_count": signal_count, "order_count": sum(counts.values()),
                "filled_order_count": sum(v for k,v in counts.items() if k != "NO_FILL"),
                "positions": positions, "runtime_status": runtime["status"], "blocked_reason": runtime["blocked_reason"],
                "heartbeat_at": runtime["heartbeat_at"],
                "heartbeat_age_seconds": max(0.0, (stamp-_ts(runtime["heartbeat_at"])).total_seconds()),
                "runtime_details": json.loads(runtime["details"]),
                "ready_streams": [],
                "risk_limits": {"default_order_usd": _num(default_order),
                                "max_order_usd": _num(limits["max_order_usd"]),
                                "max_stream_cohort_usd": _num(limits["max_order_usd"]),
                                "max_total_open_cost_usd": _num(limits["max_open_risk_usd"]),
                                "max_underlying_open_cost_usd": _num(limits["max_underlying_open_risk_usd"]),
                                "max_active_window_cost_usd": _num(limits["max_active_window_risk_usd"]),
                                "max_underlying_active_cost_usd": _num(limits["max_underlying_active_risk_usd"]),
                                "max_buy_price": .515},
                "limitations": ["No real orders or signed transactions", "Model attestation must be verified by the upstream signal adapter", "Paper IOC fills use observed ask depth; they are not actual fills", "Open positions have no mark-to-market valuation"]}

    def ledger(self, limit=100):
        limit = max(1, min(int(limit), 10000))
        with self._connection() as con:
            rows = con.execute("SELECT * FROM events ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def equity_history(self, limit=1000):
        limit = max(1, min(int(limit), 10000))
        with self._connection() as con:
            rows = con.execute("SELECT * FROM equity_observations ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [{"utc": r["utc"], "cash": _usd(r["cash_atoms"]), "locked_cost": _usd(r["locked_atoms"]),
                 "realized_pnl": _usd(r["realized_atoms"]), "equity": None if r["equity_atoms"] is None else _usd(r["equity_atoms"]),
                 "valuation_status": r["valuation_status"]} for r in reversed(rows)]


ForwardAccount = Account
