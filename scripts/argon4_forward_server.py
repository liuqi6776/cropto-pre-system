"""Independent 10,000 USD paper account and GET-only local observation panel.

Saved local models can be enabled for isolated paper execution explicitly.
This entrypoint has no signing, exchange-order, or HTTP order mutation route.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
_VENV_SITE = ROOT / ".venv/Lib/site-packages"
if sys.version_info >= (3, 10) and _VENV_SITE.is_dir() and str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))
sys.path.insert(0, str(ROOT))
from tools.argon4_forward.account import Account
from tools.argon4_forward.monitor import ObservationStore, collect_reference_snapshot, utcnow
from tools.argon4_forward.stage import stage_profile, StageRejected, DISABLED
from tools.argon4_forward.runtime_profile import engineering_profile
from tools.argon4_forward.live_preflight import preparation_status
from tools.argon4_forward.risk_view import split_open_risk, RiskViewError

STREAMS = ["BTC_15m", "ETH_5m", "SOL_15m", "ETH_15m", "SOL_5m", "BTC_5m", "BNB_15m"]
BLOCKERS = [
    {"code": "LOCAL_MODEL_TRAINING", "message": "正在准备本地 V4 独立专家训练；回测与模拟将使用已保存的同版本模型"},
    {"code": "EXECUTION_WIRING_PENDING", "message": "市场窗口、费用与官方结算尚未接入自动模拟执行"},
]


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default
    except Exception:
        return default


def _is_pid_running(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == 259
                return True
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_jsonl_tail(path, max_lines=200):
    p = Path(path)
    if not p.is_file():
        return []
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    out = []
    for raw in lines[-max_lines:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except Exception:
            continue
    return out


def _read_log_tail(path, max_lines=25):
    p = Path(path)
    if not p.is_file():
        return []
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return [ln[-280:] for ln in lines if ln.strip()][-max_lines:]
    except Exception:
        return []


def binance_backoff(runtime):
    state = load_json(Path(runtime) / "features/binance_backoff.json", {})
    until = state.get("blocked_until_utc")
    if until and datetime.now(timezone.utc) < datetime.fromisoformat(until.replace("Z", "+00:00")):
        return state
    return None


def _floor_cohort_iso(ts_str):
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        floored = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
        return floored.isoformat()
    except Exception:
        return None


def _classify_stream_guard(stream_name, info):
    """Return a dict describing the risk guard triggered on a stream, or None if normal no-signal/not-due."""
    if not isinstance(info, dict) or info.get("traded"):
        return None
    ntr = str(info.get("no_trade_reason") or "")
    exec_st = str(info.get("execution_status") or "")
    dec_st = str(info.get("decision_status") or "")
    side = info.get("side")
    abs_z = info.get("abs_z_v4")
    thr = info.get("threshold")
    z_str = f"|z|={float(abs_z):.2f}/{float(thr):.2f}" if isinstance(abs_z, (int, float)) and isinstance(thr, (int, float)) else ""

    if "ADVERSE_SELECTION_GUARD" in ntr:
        detail = ntr.split("(", 1)[1].rstrip(")") if "(" in ntr else ntr
        return {
            "stream": stream_name,
            "side": side,
            "guard_code": "ADVERSE_SELECTION_GUARD",
            "guard_tag": "【风控·折价飞刀护栏】",
            "guard_type": "RISK_GUARD",
            "summary": f"{stream_name} {side or ''} ({z_str}) · {detail}".strip(),
        }
    if any(k in ntr for k in ("INSUFFICIENT_EXECUTABLE_DEPTH", "WAITING_DEPTH_RECHECK", "THIN_BOOK_DEPTH_GUARD", "WAITING_PRICE_RECHECK_THIN_BOOK")):
        detail = ntr.split("(", 1)[1].rstrip(")") if "(" in ntr else ntr
        return {
            "stream": stream_name,
            "side": side,
            "guard_code": "INSUFFICIENT_EXECUTABLE_DEPTH",
            "guard_tag": "【风控·盘口深度不足】",
            "guard_type": "RISK_GUARD",
            "summary": f"{stream_name} {side or ''} ({z_str}) · {detail}".strip(),
        }
    if "CONFIDENCE_TIERED_PRICE_CAP" in ntr:
        detail = ntr.split("(", 1)[1].rstrip(")") if "(" in ntr else ntr
        return {
            "stream": stream_name,
            "side": side,
            "guard_code": "CONFIDENCE_TIERED_PRICE_CAP",
            "guard_tag": "【风控·微弱信号限价】",
            "guard_type": "RISK_GUARD",
            "summary": f"{stream_name} {side or ''} ({z_str}) · {detail}".strip(),
        }
    if "PRICE_LIMIT_NOT_MET" in ntr or exec_st == "PRICE_LIMIT_NOT_MET":
        detail = ntr.split("(", 1)[1].rstrip(")") if "(" in ntr else ntr
        return {
            "stream": stream_name,
            "side": side,
            "guard_code": "PRICE_LIMIT_NOT_MET",
            "guard_tag": "【风控·超限价保护】",
            "guard_type": "RISK_GUARD",
            "summary": f"{stream_name} {side or ''} ({z_str}) · {detail}".strip(),
        }
    if "WAITING_PRICE_RECHECK" in ntr or exec_st == "WAITING_PRICE_RECHECK":
        detail = ntr.split("(", 1)[1].rstrip(")") if "(" in ntr else ntr
        return {
            "stream": stream_name,
            "side": side,
            "guard_code": "WAITING_PRICE_RECHECK",
            "guard_tag": "【风控·超限价重检】",
            "guard_type": "RISK_GUARD",
            "summary": f"{stream_name} {side or ''} ({z_str}) · {detail}".strip(),
        }
    if "BLEED_WINDOW_SINGLE_ORDER_CAP" in ntr or exec_st == "BLEED_WINDOW_SINGLE_ORDER_CAP":
        return {
            "stream": stream_name,
            "side": side,
            "guard_code": "BLEED_WINDOW_SINGLE_ORDER_CAP",
            "guard_tag": "【风控·2h阴跌同窗限1单】",
            "guard_type": "RISK_GUARD",
            "summary": f"{stream_name} {side or ''} ({z_str}) · 已成交第1顺位主力单，拦截同窗第2单防双杀",
        }
    if "VOLATILITY_REGIME_GUARD" in ntr:
        detail = ntr.split("(", 1)[1].rstrip(")") if "(" in ntr else ntr
        if "US_OPEN_BLACKOUT" in ntr:
            tag = "【风控·美股开盘避险】"
        elif "US_CLOSE_BLACKOUT" in ntr:
            tag = "【风控·美股收盘1h避险】"
        elif "SUNDAY_REOPEN_BLACKOUT" in ntr:
            tag = "【风控·周一美盘重开避险】"
        elif "OWN_ASSET_1H_WATERFALL_GUARD" in ntr:
            tag = "【风控·单币1h瀑布拦截】"
        elif "OWN_ASSET_15M_WATERFALL_GUARD" in ntr:
            tag = "【风控·单币15m急跌拦截】"
        elif "BTC_4H_MACRO_BREAKOUT" in ntr:
            tag = "【风控·宏观4h破位熔断】"
        elif "BTC_1H_MOMENTUM_CHASE" in ntr:
            tag = "【风控·1h顺势追单拦截】"
        else:
            tag = "【风控·波动率过滤】"
        return {
            "stream": stream_name,
            "side": side,
            "guard_code": "VOLATILITY_REGIME_GUARD",
            "guard_tag": tag,
            "guard_type": "RISK_GUARD",
            "summary": f"{stream_name} {side or ''} · {detail}".strip(),
        }
    if "UNREACHED" in ntr or exec_st == "UNREACHED":
        return {
            "stream": stream_name,
            "side": side,
            "guard_code": "COHORT_BUDGET_CAP",
            "guard_tag": "【风控·窗口预算上限】",
            "guard_type": "RISK_GUARD",
            "summary": f"{stream_name} {side or ''} ({z_str}) · 上层流已用满本窗口预算额度",
        }
    if "EXECUTION_REJECTED" in ntr or "BLOCKED_BEFORE_STREAM" in ntr or dec_st == "SIGNAL":
        detail = ntr.split("(", 1)[1].rstrip(")") if "(" in ntr else ntr
        return {
            "stream": stream_name,
            "side": side,
            "guard_code": "EXECUTION_GUARD",
            "guard_tag": "【风控·敞口/额度护栏】",
            "guard_type": "RISK_GUARD",
            "summary": f"{stream_name} {side or ''} ({z_str}) · {detail}".strip(),
        }
    return None


def _format_input_mismatch_detail(details):
    if not isinstance(details, dict):
        return ""
    parts = []
    for key in ("errors", "gaps"):
        items = details.get(key)
        if not isinstance(items, list):
            continue
        for item in items[:4]:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source") or "数据源")
            err = str(item.get("error") or "")
            if "LATEST_CLOSED_BAR_MISSING" in err:
                parts.append(f"{src} 最新收盘K线延迟/未闭合")
            elif "CONFLICTING_CLOSED_OBSERVATION" in err:
                parts.append(f"{src} K线收盘价观测冲突(待核验)")
            elif "PUBLIC_HTTP_STATUS_418" in err or "PUBLIC_HTTP_STATUS_429" in err or "PUBLIC_RATE_LIMIT" in err:
                parts.append(f"{src} Binance接口限流(418/429)")
            elif err:
                parts.append(f"{src}({err[:48]})")
    dedup = list(dict.fromkeys(parts))
    return "；".join(dedup)


def _classify_block_guard(block_code, reason_text, details=None):
    code = str(block_code or "BLOCKED")
    reason = str(reason_text or "")
    det = details if isinstance(details, dict) else {}
    mapping = {
        "DAILY_LOSS_LIMIT": ("【风控·单日止损熔断】", "RISK_GUARD"),
        "NO_RISK_BUDGET": ("【风控·总敞口上限保护】", "RISK_GUARD"),
        "INPUTS_NOT_VALIDATED": ("【数据对不上·K线延迟/冲突】", "DATA_MISMATCH"),
        "REVISION_CUTOVER_WINDOW": ("【数据对不上·修订切换保护窗】", "DATA_MISMATCH"),
        "PAPER_WARMUP": ("【启动热身核验】", "WARMUP"),
        "WARMUP_COMPLETED": ("【启动热身完成】", "WARMUP"),
        "STAGE_WAITING_ACTIVATION": ("【等待阶段启用】", "WARMUP"),
        "VERIFICATION_PAUSE": ("【校验维护暂停】", "WARMUP"),
        "PUBLIC_RATE_LIMIT_BACKOFF": ("【限流异常·API退避等待】", "STUCK_OR_ERROR"),
        "PIPELINE_CYCLE_EXCEPTION": ("【异常·时钟校准/循环拦截】", "STUCK_OR_ERROR"),
        "THREAD_STALL_DETECTED": ("【卡死告警·线程阻塞】", "STUCK_OR_ERROR"),
        "WORKER_HEARTBEAT_STALL": ("【卡死告警·心跳超时】", "STUCK_OR_ERROR"),
    }
    tag, gtype = mapping.get(code, (f"【拦截·{code}】", "DATA_MISMATCH"))
    extra = ""
    if code == "INPUTS_NOT_VALIDATED":
        extra = _format_input_mismatch_detail(det)
    elif code == "PIPELINE_CYCLE_EXCEPTION":
        err_str = str(det.get("error") or reason)
        if "clock uncertainty" in err_str.lower():
            extra = "Polymarket时钟校准误差 > 0.5s，本轮拒绝下单并自动重采"
        elif err_str:
            extra = err_str[:120]
    elif code == "PUBLIC_RATE_LIMIT_BACKOFF":
        bo = det.get("backoff") if isinstance(det.get("backoff"), dict) else {}
        if bo.get("blocked_until_utc"):
            extra = f"原因={bo.get('reason', '418/429')}，退避至 {bo.get('blocked_until_utc')}"
    summary_body = extra or reason
    return {
        "guard_code": code,
        "guard_tag": tag,
        "guard_type": gtype,
        "summary": f"{tag} {summary_body}".strip(),
    }


class Application:
    def __init__(self, runtime, capital=10000, enable_paper=False):
        self.runtime = Path(runtime)
        self.account = Account(self.runtime / "account.sqlite3", initial_capital=capital, currency="USD")
        self.observations = ObservationStore(self.runtime / "observations.sqlite3")
        from tools.argon4_forward.monitor_logger import MonitorAuditLogger
        self.monitor_log = MonitorAuditLogger(self.runtime)
        self.stop = threading.Event()
        self.started_at = utcnow()
        self.last_cycle = None
        self.last_error = None
        from tools.argon4_forward.polymarket_transport import PublicPolymarketTransport
        from tools.argon4_forward.storage_janitor import StorageJanitor
        self.market_transport = PublicPolymarketTransport(pool_size=4)
        self.janitor = StorageJanitor(
            self.runtime,
            policy_provider=lambda: load_json(ROOT / "config/argon4_forward_policy.json", {}),
            monitor_log=self.monitor_log,
        )
        self.pipeline = None
        self.telemetry = None
        if enable_paper:
            from tools.argon4_forward.pipeline import ForwardPipeline
            self.pipeline = ForwardPipeline(self.account, self.runtime, self.observations, monitor_log=self.monitor_log)
            self.pipeline.janitor = self.janitor
            policy_provider = lambda: load_json(ROOT / "config/argon4_forward_policy.json", {})
            if engineering_profile(policy_provider())["book_telemetry"]:
                from tools.argon4_forward.book_telemetry import BookTelemetry
                self.telemetry = BookTelemetry(self.runtime, policy_provider=policy_provider,
                    market_provider=lambda: self.observations.latest("polymarket"),
                    clock_provider=lambda: dict(self.pipeline.clock or {}),
                    decision_provider=self.pipeline.status,
                    sampling_profile=engineering_profile(policy_provider()).get("telemetry_sampling_profile", "legacy_v11"))
        self.account.heartbeat("WAITING_FOR_MODEL", BLOCKERS[0]["message"])

    def status(self):
        allocation = load_json(ROOT / "config/argon4_forward_policy.json", {"status": "AWAITING_RANKING_CONFIRMATION", "order": STREAMS})
        training = load_json(ROOT / "reports/argon4_local_v4_20261106/progress.json", {"status": "PREPARING", "phase": "本地训练准备", "completed_streams": []})
        live = load_json(ROOT / "config/argon4_live_preflight.json", {})
        pipeline = self.pipeline.status() if self.pipeline else {"active": False, "models": {}, "status": "NOT_ENABLED"}
        models = pipeline.get("models", {})
        overview = self.account.overview(policy=allocation)
        try:
            open_risk = split_open_risk(overview, clock=dict(getattr(self.pipeline, "clock", None) or {}))
        except RiskViewError:
            open_risk = {"status": "UNAVAILABLE", "reconciled": False,
                         "total_locked_cost_usd": overview.get("locked_cost"),
                         "risk_released_usd": 0., "reusable_cash_from_unsettled_usd": 0.}
        blockers = ([{"code": pipeline["status"], "message": pipeline.get("reason")}]
                    if self.pipeline and pipeline.get("reason") else ([] if pipeline.get("active") else BLOCKERS))
        try:
            stage = stage_profile(allocation)
            engineering = engineering_profile(allocation)
            live_preparation = preparation_status(allocation, live)
        except StageRejected:
            stage = {"configured": True, "status": "INVALID", "enabled_streams": [],
                     "disabled_streams": list(STREAMS), "live_enabled": False}
            blockers = [{"code": "INVALID_EXECUTION_STAGE", "message": "阶段配置未通过检查，暂停新模拟单"}]
            engineering = {"enabled": False, "profile": "INVALID"}
            live_preparation = {"ready_for_live": False, "configuration_safe": False}
        except ValueError:
            engineering = {"enabled": False, "profile": "INVALID"}
            live_preparation = {"ready_for_live": False, "configuration_safe": False}
            blockers = [{"code": "INVALID_ENGINEERING_PROFILE", "message": "采集配置未通过检查，暂停新模拟单"}]
        monitor_audit_summary = self.monitor_log.summary()
        timeline, timeline_summary = self._build_cohort_timeline(overview=overview)
        status_5m_live = self._build_live_5m_diagnostics(
            pipeline=pipeline,
            monitor_audit=monitor_audit_summary,
            timeline=timeline,
            timeline_summary=timeline_summary,
        )
        return {"schema_version": 1, "now": utcnow(), "service_started_at": self.started_at,
                "service_status": pipeline["status"] if self.pipeline else "OBSERVING_ONLY",
                "paper_trading_active": pipeline.get("active", False) and stage["status"] != "INVALID" and engineering["profile"] != "INVALID", "live_enabled": False,
                "strategy": "ARGON4_SEVEN_STREAM_PRIORITY_WATERFALL",
                "account": overview, "open_risk": open_risk, "blockers": blockers,
                "status_5m_live": status_5m_live,
                "cohort_timeline": timeline,
                "cohort_summary": timeline_summary,
                "last_cycle": self.last_cycle, "last_error": self.last_error,
                "reference": self.observations.latest("reference"),
                "markets": self.observations.latest("polymarket"),
                "allocation": allocation,
                "stage": stage,
                "training": training,
                "pipeline": pipeline,
                "engineering": engineering,
                "book_telemetry": self.telemetry.status() if self.telemetry else {"status": "DISABLED"},
                "storage_janitor": self.janitor.status(),
                "monitor_audit": monitor_audit_summary,
                "backtest": load_json(ROOT / "reports/argon4_local_v4_20261106/summary.json", {}),
                "streams": [{"stream": s, "status": DISABLED if s in stage["disabled_streams"] else "LOCAL_MODEL_LOADED" if s in models else "WAITING_FOR_MODEL",
                             "stage_enabled": s in stage["enabled_streams"], "model": models.get(s)} for s in allocation.get("order", STREAMS)],
                "live_preflight": {"venue": "polymarket", "status": "DISABLED", "signer_connected": False,
                                   "preparation": live_preparation,
                                   "execution_stage": stage, "live_enabled": False,
                                   "wallet_configured": False, "residency_confirmed": live.get("residency_confirmed", False),
                                   "residency_country": live.get("residency_country"),
                                   "planned_total_capital_usd": live.get("planned_total_capital_usd"),
                                   "eligibility_status": live.get("eligibility_status", "UNVERIFIED")},
                "limitations": ["账户从创建时刻起算，历史研究收益不计入", "本地模型与原作者模型不等同；以实际加载版本为准",
                                "盘口观察不证明挂单成交；模拟结算须来自平台官方结果", "本机运行时采集，关机或休眠期间不补造观测"]}

    def _build_cohort_timeline(self, overview=None, live_orders=None, settlements_by_mkt=None, max_cohorts=144):
        if overview is None:
            allocation = load_json(ROOT / "config/argon4_forward_policy.json", {"order": STREAMS})
            overview = self.account.overview(policy=allocation)
        initial_cap = float(overview.get("initial_capital") or 1269.0)
        paper_orders = list(overview.get("recent_orders") or [])
        paper_by_cohort = {}
        paper_by_id = {}
        for po in reversed(paper_orders):
            oid = po.get("order_id")
            if oid:
                paper_by_id[str(oid)] = po
            c_iso = _floor_cohort_iso(po.get("fill_at"))
            if c_iso:
                paper_by_cohort.setdefault(c_iso, []).append(po)

        settlements_by_mkt = settlements_by_mkt or {}
        live_by_cohort = {}
        if live_orders:
            for lo in reversed(live_orders):
                c_iso = lo.get("cohort_id") or _floor_cohort_iso((lo.get("timing") or {}).get("paper_fill_at"))
                if c_iso:
                    lo_copy = dict(lo)
                    mkt = lo_copy.get("market_id")
                    if mkt and str(mkt) in settlements_by_mkt:
                        lo_copy["settlement"] = settlements_by_mkt[str(mkt)]
                    live_by_cohort.setdefault(str(c_iso), []).append(lo_copy)

        audit_records = _read_jsonl_tail(self.runtime / "logs/monitor_audit.jsonl", max_lines=2500)
        stream_audits = {}
        block_audits = {}
        warmup_cohorts = set()
        for rec in audit_records:
            ev = rec.get("event")
            det = rec.get("details") or {}
            c_iso = det.get("cohort") or _floor_cohort_iso(rec.get("ts"))
            if ev == "WARMUP_COMPLETED":
                msg = str(rec.get("message") or "")
                if "当前窗口=" in msg:
                    part = msg.split("当前窗口=", 1)[1].split(" ", 1)[0].strip()
                    if part and part != "N/A":
                        warmup_cohorts.add(part)
                if c_iso:
                    warmup_cohorts.add(str(c_iso))
                continue
            if ev in ("THREAD_STALL_DETECTED", "WORKER_HEARTBEAT_STALL") and c_iso:
                block_audits.setdefault(str(c_iso), []).append({
                    "event": "COHORT_NO_TRADE_BLOCKED",
                    "details": {
                        "cohort": str(c_iso),
                        "block_code": ev,
                        "reason": str(rec.get("message") or ""),
                    },
                })
                continue
            raw_cohort = det.get("cohort")
            if not raw_cohort and ev == "COHORT_NO_TRADE_BLOCKED":
                raw_cohort = _floor_cohort_iso(rec.get("ts"))
            if not raw_cohort:
                continue
            raw_cohort = str(raw_cohort)
            if ev == "COHORT_TRADED_WITH_STREAM_AUDIT":
                stream_audits[raw_cohort] = rec
            elif ev == "COHORT_NO_TRADE_AUDIT":
                prev = stream_audits.get(raw_cohort)
                if not prev or prev.get("event") != "COHORT_TRADED_WITH_STREAM_AUDIT":
                    if not prev or det.get("overall_status") == "PROCESSED" or prev.get("details", {}).get("overall_status") != "PROCESSED":
                        stream_audits[raw_cohort] = rec
            elif ev == "COHORT_NO_TRADE_BLOCKED":
                block_audits.setdefault(raw_cohort, []).append(rec)

        all_known = set(stream_audits.keys()) | set(block_audits.keys()) | set(paper_by_cohort.keys()) | set(live_by_cohort.keys()) | warmup_cohorts
        now_utc = datetime.now(timezone.utc)
        current_floor_dt = now_utc.replace(minute=(now_utc.minute // 5) * 5, second=0, microsecond=0)
        current_floor_iso = current_floor_dt.isoformat()
        elapsed_now = (now_utc - current_floor_dt).total_seconds()
        service_started_floor_iso = _floor_cohort_iso(self.started_at)
        latest_floor = current_floor_dt
        if all_known:
            try:
                max_known = max(datetime.fromisoformat(k.replace("Z", "+00:00")).astimezone(timezone.utc) for k in all_known)
                if max_known > latest_floor:
                    latest_floor = max_known
            except Exception:
                pass
        started_iso = _floor_cohort_iso(overview.get("started_at"))
        try:
            started_dt = datetime.fromisoformat(started_iso).astimezone(timezone.utc) if started_iso else (latest_floor - timedelta(minutes=5 * (max_cohorts - 1)))
        except Exception:
            started_dt = latest_floor - timedelta(minutes=5 * (max_cohorts - 1))
        earliest_dt = max(started_dt, latest_floor - timedelta(minutes=5 * (max_cohorts - 1)))

        grid = []
        cur = earliest_dt
        while cur <= latest_floor:
            grid.append(cur.isoformat())
            cur += timedelta(minutes=5)

        settlements_by_mkt = settlements_by_mkt or {}
        running_paper_pnl = 0.0
        running_live_pnl = 0.0
        # Pre-accumulate any orders prior to earliest_dt
        for po in reversed(paper_orders):
            c_iso = _floor_cohort_iso(po.get("fill_at"))
            if c_iso and c_iso < earliest_dt.isoformat() and po.get("realized_pnl_usd") is not None:
                running_paper_pnl += float(po.get("realized_pnl_usd") or 0.0)
        if live_orders:
            for lo in reversed(live_orders):
                c_iso = lo.get("cohort_id") or _floor_cohort_iso((lo.get("timing") or {}).get("paper_fill_at"))
                mkt = lo.get("market_id")
                st = settlements_by_mkt.get(str(mkt)) if mkt else None
                if c_iso and str(c_iso) < earliest_dt.isoformat() and st and st.get("live_realized_pnl_usdc") is not None:
                    running_live_pnl += float(st.get("live_realized_pnl_usdc") or 0.0)

        timeline = []
        traded_count = 0
        risk_guard_count = 0
        data_mismatch_count = 0
        stuck_error_count = 0
        warmup_count = 0
        no_signal_count = 0

        for c_iso in grid:
            p_ords = list(paper_by_cohort.get(c_iso, []))
            l_ords = list(live_by_cohort.get(c_iso, []))
            s_rec = stream_audits.get(c_iso)
            b_recs = block_audits.get(c_iso, [])

            for po in p_ords:
                if po.get("realized_pnl_usd") is not None:
                    running_paper_pnl += float(po.get("realized_pnl_usd") or 0.0)
            for lo in l_ords:
                mkt = lo.get("market_id")
                st = settlements_by_mkt.get(str(mkt)) if mkt else None
                if st and st.get("live_realized_pnl_usdc") is not None:
                    running_live_pnl += float(st.get("live_realized_pnl_usdc") or 0.0)

            guards = []
            due_evals = []
            duration_sec = None
            if s_rec:
                s_det = s_rec.get("details") or {}
                if isinstance(s_det.get("duration_seconds"), (int, float)):
                    duration_sec = round(float(s_det["duration_seconds"]), 2)
                streams_map = s_det.get("streams") or {}
                for s_name, s_info in streams_map.items():
                    if not isinstance(s_info, dict):
                        continue
                    dec_st = s_info.get("decision_status")
                    if dec_st in ("DISABLED_BY_STAGE", "NOT_DUE"):
                        continue
                    abs_z = s_info.get("abs_z_v4")
                    thr = s_info.get("threshold")
                    side = s_info.get("side")
                    g_info = _classify_stream_guard(s_name, s_info)
                    if g_info:
                        guards.append(g_info)
                    due_evals.append({
                        "stream": s_name,
                        "side": side,
                        "abs_z_v4": abs_z,
                        "threshold": thr,
                        "traded": bool(s_info.get("traded")),
                        "decision_status": dec_st,
                        "execution_status": s_info.get("execution_status"),
                        "guard_tag": g_info["guard_tag"] if g_info else None,
                    })
                for blk_code in (s_det.get("blocked") or []):
                    guards.append(_classify_block_guard(blk_code, "执行前账户/风控额度拦截"))

            if not s_rec and b_recs:
                seen_codes = set()
                for br in b_recs:
                    b_det = br.get("details") or {}
                    b_code = b_det.get("block_code") or "BLOCKED"
                    if b_code not in seen_codes:
                        seen_codes.add(b_code)
                        guards.append(_classify_block_guard(b_code, b_det.get("reason") or "", b_det))
            elif not s_rec and c_iso in warmup_cohorts:
                guards.append(_classify_block_guard("WARMUP_COMPLETED", "服务启动热身核验完成，当窗跳过不补单"))

            has_trade = bool(p_ords or l_ords or (s_rec and (s_rec.get("details") or {}).get("traded_streams")))
            has_risk_guard = any(g.get("guard_type") == "RISK_GUARD" for g in guards)
            has_data_mismatch = any(g.get("guard_type") == "DATA_MISMATCH" for g in guards)
            has_stuck_or_error = any(g.get("guard_type") == "STUCK_OR_ERROR" for g in guards)
            has_warmup = any(g.get("guard_type") == "WARMUP" for g in guards)

            custom_summary = None
            if has_trade:
                category = "TRADED"
                traded_count += 1
                if has_risk_guard:
                    risk_guard_count += 1
                badge_text = "🟢 已下单成交" + (f" + {guards[0]['guard_tag']}" if guards else "")
            elif has_risk_guard:
                category = "RISK_GUARD"
                risk_guard_count += 1
                badge_text = "🛡️ " + " / ".join(dict.fromkeys(g["guard_tag"] for g in guards if g.get("guard_type") == "RISK_GUARD"))
            elif has_data_mismatch:
                category = "DATA_MISMATCH"
                data_mismatch_count += 1
                badge_text = "🟠 " + " / ".join(dict.fromkeys(g["guard_tag"] for g in guards if g.get("guard_type") == "DATA_MISMATCH"))
            elif has_stuck_or_error:
                category = "STUCK_OR_ERROR"
                stuck_error_count += 1
                badge_text = "🔴 " + " / ".join(dict.fromkeys(g["guard_tag"] for g in guards if g.get("guard_type") == "STUCK_OR_ERROR"))
            elif s_rec:
                category = "NO_SIGNAL"
                no_signal_count += 1
                badge_text = f"🔵 正常无信号 ({len(due_evals)}流已检)"
            elif has_warmup:
                category = "WARMUP"
                warmup_count += 1
                badge_text = "⚪ " + " / ".join(dict.fromkeys(g["guard_tag"] for g in guards if g.get("guard_type") == "WARMUP"))
            elif c_iso == current_floor_iso and elapsed_now < 22.0:
                if self.pipeline and not getattr(self.pipeline, "warmup_complete", True):
                    category = "WARMUP"
                    warmup_count += 1
                    badge_text = "⚪ 启动热身核验中"
                    custom_summary = "服务刚启动，正在补齐并核验四币现货/合约历史 K 线输入，热身窗口不补单"
                else:
                    category = "IN_PROGRESS"
                    badge_text = f"⏳ 当前窗口定检中 (T+{int(max(0.0, elapsed_now))}s)"
                    custom_summary = f"已进入 {c_iso[11:16]} UTC 窗口 T+{elapsed_now:.1f}s，正在等待 K 线闭合与七流实时定检（20s 截止线内）"
            elif service_started_floor_iso and c_iso == service_started_floor_iso:
                category = "WARMUP"
                warmup_count += 1
                badge_text = "⚪ 启动热身跳过"
                custom_summary = "该 5m 窗口为本次服务启动所在窗口，热身核验后按安全规则跳过不补单"
            elif service_started_floor_iso and c_iso < service_started_floor_iso:
                category = "WARMUP"
                warmup_count += 1
                badge_text = "⚪ 历史重启/停机空档"
                custom_summary = "该 5m 窗口期间本地进程处于重启或停机状态，未执行实时采样，绝不伪造补单"
            else:
                category = "STUCK_OR_ERROR"
                stuck_error_count += 1
                badge_text = "🔴 疑似卡死 / 漏窗未定检"
                custom_summary = "告警：服务运行期间该 5m 窗口已过 T+22s 截止线仍无定检或风控记录，可能发生代理断连、时钟超时或线程阻塞！"

            eval_parts = []
            for ev_item in due_evals:
                z_v = ev_item.get("abs_z_v4")
                t_v = ev_item.get("threshold")
                sd = ev_item.get("side") or ""
                if ev_item.get("traded"):
                    eval_parts.append(f"{ev_item['stream']} {sd} |z|={float(z_v):.2f}>={float(t_v):.2f} [成交]")
                elif ev_item.get("guard_tag"):
                    z_txt = f"|z|={float(z_v):.2f}/{float(t_v):.2f}" if isinstance(z_v, (int, float)) and isinstance(t_v, (int, float)) else ""
                    eval_parts.append(f"{ev_item['stream']} {sd} {z_txt} {ev_item['guard_tag']}".strip())
                elif isinstance(z_v, (int, float)) and isinstance(t_v, (int, float)):
                    eval_parts.append(f"{ev_item['stream']} {sd} |z|={float(z_v):.2f}<{float(t_v):.2f}")
                else:
                    eval_parts.append(f"{ev_item['stream']} 未达阈值")

            if custom_summary:
                summary_text = custom_summary
            elif guards and not eval_parts:
                summary_text = "；".join(g["summary"] for g in guards)
            elif guards:
                guard_str = "；".join(g["summary"] for g in guards)
                other_str = " · ".join(p for p in eval_parts if "【" not in p)
                summary_text = guard_str + (f" | 其余定检: {other_str}" if other_str else "")
            elif eval_parts:
                prefix = f"模型完成 {len(due_evals)} 流定检并执行成交：" if has_trade else f"模型准时完成 {len(due_evals)} 流定检，均未达开仓阈值："
                summary_text = prefix + " · ".join(eval_parts)
            else:
                summary_text = "该 5m 窗口处于服务重启/热身期，按安全规则跳过不补单"

            timeline.append({
                "cohort": c_iso,
                "category": category,
                "badge_text": badge_text,
                "duration_seconds": duration_sec,
                "due_stream_count": len(due_evals),
                "guards": guards,
                "evaluated_streams": due_evals,
                "summary_text": summary_text,
                "paper_orders": p_ords,
                "live_orders": l_ords,
                "paper_balance_usd": round(initial_cap + running_paper_pnl, 4),
                "paper_pnl_usd": round(running_paper_pnl, 4),
                "live_pnl_usdc": round(running_live_pnl, 4),
            })

        timeline.reverse()
        summary = {
            "total_cohorts": len(timeline),
            "traded_cohorts": traded_count,
            "risk_guard_cohorts": risk_guard_count,
            "data_mismatch_cohorts": data_mismatch_count,
            "stuck_error_cohorts": stuck_error_count,
            "warmup_cohorts": warmup_count,
            "data_guard_cohorts": data_mismatch_count + stuck_error_count + warmup_count,
            "no_signal_cohorts": no_signal_count,
        }
        return timeline, summary

    def _build_live_5m_diagnostics(self, pipeline, monitor_audit, timeline, timeline_summary):
        now_utc = datetime.now(timezone.utc)
        current_floor = now_utc.replace(minute=(now_utc.minute // 5) * 5, second=0, microsecond=0)
        elapsed_sec = round((now_utc - current_floor).total_seconds(), 1)
        until_next_sec = round(max(0.0, 300.0 - elapsed_sec), 1)

        hb_map = (monitor_audit or {}).get("worker_heartbeats") or {}
        pipe_hb = hb_map.get("pipeline") or {}
        obs_hb = hb_map.get("market_observer") or {}
        pipe_hb_age = pipe_hb.get("age_seconds")
        obs_hb_age = obs_hb.get("age_seconds")
        active_phases = list((monitor_audit or {}).get("active_phases") or [])

        stalled_phase = next((ph for ph in active_phases if float(ph.get("elapsed_seconds") or 0) >= 20.0), None)
        hb_stalled = isinstance(pipe_hb_age, (int, float)) and float(pipe_hb_age) >= 35.0

        clk_raw = dict(getattr(self.pipeline, "clock", None) or {}) if self.pipeline else {}
        clk_status = clk_raw.get("status") or ("DISABLED" if not self.pipeline else "WAITING")
        clk_unc = clk_raw.get("uncertainty_seconds")
        clk_age = (
            round(max(0.0, time.monotonic() - float(clk_raw["calibrated_monotonic"])), 1)
            if isinstance(clk_raw.get("calibrated_monotonic"), (int, float))
            else None
        )

        backoff = binance_backoff(self.runtime)
        p_status = str((pipeline or {}).get("status") or "OBSERVING_ONLY")
        p_inputs = (pipeline or {}).get("inputs") or {}
        mismatch_detail = _format_input_mismatch_detail(p_inputs)

        if backoff:
            data_state = "RATE_LIMIT"
            data_label = f"🟠 API限流退避中 ({backoff.get('reason', '418/429')})"
            data_detail = f"退避至 {backoff.get('blocked_until_utc', '未知')}，新连接自动探测恢复中"
        elif p_status == "PAPER_WARMUP":
            data_state = "WARMUP"
            data_label = "⚪ 启动热身补齐中"
            data_detail = mismatch_detail or "正在核验四币种现货/合约历史 K 线完整性"
        elif p_status == "REVISION_WAITING_FUTURE_WINDOW":
            data_state = "MISMATCH"
            data_label = "🟠 K线修订切换保护"
            data_detail = "检测到历史 K 线修订并已完成归档核验，当前窗口安全跳过，下一 5m 窗口恢复"
        elif p_status == "PAPER_DATA_BLOCKED" or p_inputs.get("status") == "BLOCKED":
            data_state = "MISMATCH"
            data_label = "🟠 数据对不上 (K线缺口/冲突)"
            data_detail = mismatch_detail or str((pipeline or {}).get("reason") or "最新闭合 K 线未就绪或收盘价冲突")
        else:
            data_state = "ALIGNED"
            data_label = "🟢 四币现货/合约 K 线已对齐"
            data_detail = "BTC / ETH / SOL / BNB 闭合 K 线与标记价格校验通过"

        latest_finished = next((item for item in (timeline or []) if item.get("category") != "IN_PROGRESS"), None)
        current_item = timeline[0] if timeline and timeline[0].get("cohort") == current_floor.isoformat() else None

        if stalled_phase or hb_stalled:
            overall_state = "STUCK"
            overall_badge = "🔴 疑似卡死 / 线程阻塞告警"
            if stalled_phase:
                headline = (f"告警：工作线程 [{stalled_phase.get('worker')}:{stalled_phase.get('phase')}] "
                            f"已阻塞 {stalled_phase.get('elapsed_seconds')}s，请检查网络/代理连接！")
            else:
                headline = f"告警：模拟引擎 pipeline 心跳已 {pipe_hb_age}s 未更新，疑似发生阻塞！"
        elif p_status == "PAPER_BLOCKED" or (clk_status not in ("CALIBRATED", "WAITING", "DISABLED") and p_status != "PAPER_WAITING_WINDOW"):
            overall_state = "STUCK"
            overall_badge = "🔴 时钟/循环异常拦截"
            headline = f"当前引擎受阻：{str((pipeline or {}).get('reason') or clk_status)}（正在每 1.5s 自动重采恢复）"
        elif current_item and current_item.get("category") == "STUCK_OR_ERROR":
            overall_state = "STUCK"
            overall_badge = "🔴 当前 5m 窗口异常漏检"
            headline = current_item.get("summary_text") or "当前 5m 窗口未按时完成定检"
        elif data_state == "RATE_LIMIT":
            overall_state = "RATE_LIMIT"
            overall_badge = "🟠 Binance 接口限流退避中"
            headline = f"{data_label} · {data_detail}"
        elif data_state == "MISMATCH" and current_item and current_item.get("category") == "DATA_MISMATCH":
            overall_state = "DATA_MISMATCH"
            overall_badge = "🟠 当前窗口数据对不上（安全跳过）"
            headline = current_item.get("summary_text") or data_detail
        elif data_state == "WARMUP":
            overall_state = "WARMUP"
            overall_badge = "⚪ 服务启动热身核验中"
            headline = data_detail
        elif current_item and current_item.get("category") == "IN_PROGRESS":
            overall_state = "IN_WINDOW"
            overall_badge = f"⚡ 正在执行 {current_floor.strftime('%H:%M')} UTC 窗口定检"
            headline = current_item.get("summary_text") or "正在采集闭合 K 线并计算七流预测分数…"
        else:
            overall_state = "HEALTHY"
            overall_badge = "🟢 引擎正常 · 无卡死 · 数据对齐"
            if latest_finished:
                dur_txt = f"（耗时 {latest_finished['duration_seconds']}s）" if latest_finished.get("duration_seconds") is not None else ""
                headline = (f"系统运行正常（心跳 {pipe_hb_age if pipe_hb_age is not None else 0:.1f}s 前）："
                            f"最近 {latest_finished['cohort'][11:16]} UTC 窗口 → {latest_finished['badge_text']}{dur_txt} · "
                            f"{latest_finished['summary_text']}")
            else:
                headline = "系统运行正常，等待首个 5 分钟整点窗口定检"

        return {
            "now_utc": now_utc.isoformat(),
            "current_cohort_utc": current_floor.isoformat(),
            "elapsed_in_window_seconds": elapsed_sec,
            "seconds_until_next_window": until_next_sec,
            "overall_state": overall_state,
            "overall_badge": overall_badge,
            "headline": headline,
            "heartbeat": {
                "pipeline_age_seconds": pipe_hb_age,
                "observer_age_seconds": obs_hb_age,
                "active_phases": active_phases,
                "is_stalled": bool(stalled_phase or hb_stalled),
            },
            "clock": {
                "status": clk_status,
                "uncertainty_seconds": round(float(clk_unc), 3) if isinstance(clk_unc, (int, float)) else None,
                "age_seconds": clk_age,
            },
            "data_alignment": {
                "state": data_state,
                "label": data_label,
                "detail": data_detail,
            },
            "latest_finished_window": latest_finished,
        }

    def live_dashboard(self):
        order_root = Path(os.environ.get("ARGON4_ORDER_ROOT", str(ROOT.parent / "Order")))
        live_cfg = load_json(order_root / "config/live_config.json", {})
        runtime_base = order_root / "runtime"
        candidate_services = [
            runtime_base / "live/live_service.json",
            runtime_base / "live_service.json",
            runtime_base / "shadow/live_service.json",
        ]
        service = {}
        active_mode_dir = runtime_base / "live"
        for sp in candidate_services:
            if sp.is_file():
                s_data = load_json(sp, {})
                if s_data:
                    service = s_data
                    if sp.parent.name in ("live", "shadow"):
                        active_mode_dir = sp.parent
                    elif s_data.get("mode") in ("live", "shadow"):
                        active_mode_dir = runtime_base / str(s_data["mode"])
                    break
        if not active_mode_dir.is_dir() and (runtime_base / "shadow").is_dir():
            active_mode_dir = runtime_base / "shadow"

        pid = service.get("pid")
        process_running = _is_pid_running(pid) if isinstance(pid, int) else False

        raw_orders = _read_jsonl_tail(active_mode_dir / "live_orders.jsonl", max_lines=10000)
        orders_by_sig = {}
        orders_by_mkt = {}
        for o in raw_orders:
            sig = o.get("signal_id")
            if sig:
                orders_by_sig[str(sig)] = o
            mkt_id = o.get("market_id")
            if mkt_id:
                orders_by_mkt[str(mkt_id)] = o
        dedup_orders = list(orders_by_sig.values())
        dedup_orders.reverse()

        raw_quotas = _read_jsonl_tail(active_mode_dir / "live_quota_snapshots.jsonl", max_lines=120)
        latest_quota = raw_quotas[-1] if raw_quotas else {}

        raw_settlements = _read_jsonl_tail(active_mode_dir / "live_settlements.jsonl", max_lines=20000)
        settlements_by_mkt = {}
        for st in raw_settlements:
            mkt = st.get("market_id")
            if mkt:
                if st.get("live_realized_pnl_usdc") is None:
                    hist = st.get("pnl_history")
                    if isinstance(hist, list) and hist:
                        st["live_realized_pnl_usdc"] = round(
                            sum(float(h.get("pnl_delta_usdc") or 0.0) for h in hist if isinstance(h, dict)),
                            6,
                        )
                    else:
                        st["live_realized_pnl_usdc"] = 0.0
                matched_order = orders_by_mkt.get(str(mkt))
                if matched_order:
                    st.setdefault("stream", matched_order.get("stream"))
                    st.setdefault("order_outcome", matched_order.get("outcome"))
                    le_info = matched_order.get("live_execution") or {}
                    st.setdefault("live_cost_usdc", float(le_info.get("total_spent_usdc") or 0.0))
                    st.setdefault("live_filled_quantity", le_info.get("filled_quantity"))
                    st.setdefault("clob_status", le_info.get("clob_status"))
                settlements_by_mkt[str(mkt)] = st
        dedup_settlements = list(settlements_by_mkt.values())
        dedup_settlements.reverse()

        filled_orders = 0
        total_spent_usdc = 0.0
        detection_lags = []
        rtts = []
        slippages = []
        for o in dedup_orders:
            le = o.get("live_execution") or {}
            qty = 0.0
            try:
                qty = float(le.get("filled_quantity") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            spent = float(le.get("total_spent_usdc") or 0.0)
            total_spent_usdc += spent
            if qty > 0:
                filled_orders += 1
                diff = o.get("diff_vs_paper") or {}
                slip_val = diff.get("avg_price_slippage_bps", diff.get("price_slippage_bps"))
                if slip_val is not None:
                    slippages.append(float(slip_val))
            tm = o.get("timing") or {}
            if tm.get("detection_lag_ms") is not None:
                detection_lags.append(float(tm["detection_lag_ms"]))
            if tm.get("clob_rtt_ms") is not None and float(tm["clob_rtt_ms"]) > 0:
                rtts.append(float(tm["clob_rtt_ms"]))

        wins = 0
        losses = 0
        realized_pnl_usdc = 0.0
        redeemed_usdc_total = 0.0
        for st in dedup_settlements:
            pnl = float(st.get("live_realized_pnl_usdc") or 0.0)
            realized_pnl_usdc += pnl
            redeemed_usdc_total += float(st.get("redeemed_usdc_amount") or 0.0)
            if pnl > 1e-6:
                wins += 1
            elif pnl < -1e-6:
                losses += 1

        stderr_tail = _read_log_tail(runtime_base / "live_executor.stderr.log", max_lines=20)
        stdout_tail = _read_log_tail(runtime_base / "live_executor.stdout.log", max_lines=10)
        policy = load_json(ROOT / "config/argon4_forward_policy.json", {})
        scale = float(service.get("live_scale_factor") or live_cfg.get("capital_and_scaling", {}).get("live_scale_factor") or 1.0)
        overview = self.account.overview(policy=policy)
        timeline, timeline_summary = self._build_cohort_timeline(
            overview=overview,
            live_orders=dedup_orders,
            settlements_by_mkt=settlements_by_mkt,
        )

        paper_caps = {
            "initial_capital_usd": policy.get("initial_capital_usd", 1269),
            "single_stream_equity_fraction": policy.get("single_stream_equity_fraction", 0.08),
            "tick_equity_fraction": policy.get("tick_equity_fraction", 0.15),
            "max_order_usd": policy.get("max_order_usd", 240),
            "stream_max_order_usd": policy.get("stream_max_order_usd", {
                "BTC_15m": 240, "BTC_5m": 220, "ETH_15m": 180, "ETH_5m": 180, "SOL_15m": 50, "SOL_5m": 45
            }),
            "max_tick_budget_usd": policy.get("max_tick_budget_usd", 380),
            "max_active_window_risk_usd": policy.get("max_active_window_risk_usd", 450),
            "max_open_risk_usd": policy.get("max_open_risk_usd", 650),
            "max_underlying_active_risk_usd": policy.get("max_underlying_active_risk_usd", 300),
            "max_underlying_open_risk_usd": policy.get("max_underlying_open_risk_usd", 450),
            "daily_loss_fraction": policy.get("daily_loss_fraction", 0.15),
        }
        scaled_caps = {
            "max_order_usd": round(float(paper_caps["max_order_usd"]) * scale, 2),
            "max_tick_budget_usd": round(float(paper_caps["max_tick_budget_usd"]) * scale, 2),
            "max_active_window_risk_usd": round(float(paper_caps["max_active_window_risk_usd"]) * scale, 2),
            "max_open_risk_usd": round(float(paper_caps["max_open_risk_usd"]) * scale, 2),
            "max_underlying_active_risk_usd": round(float(paper_caps["max_underlying_active_risk_usd"]) * scale, 2),
            "max_underlying_open_risk_usd": round(float(paper_caps["max_underlying_open_risk_usd"]) * scale, 2),
        }

        latest_quota = dict(latest_quota) if latest_quota else {}
        lwq = dict(latest_quota.get("live_wallet_and_quotas") or {})
        tot_locked = float(lwq.get("total_locked_open_risk_usdc") or 0.0)
        act_locked = float(lwq.get("active_window_locked_usdc") or 0.0)
        lwq["remaining_total_open_risk_quota_usd"] = round(max(0.0, scaled_caps["max_open_risk_usd"] - tot_locked), 4)
        lwq["remaining_active_window_quota_usd"] = round(max(0.0, scaled_caps["max_active_window_risk_usd"] - act_locked), 4)
        lwq["remaining_cohort_budget_quota_usd"] = round(max(0.0, scaled_caps["max_tick_budget_usd"] - act_locked), 4)
        rem_u = dict(lwq.get("remaining_underlying_quota_usd") or {})
        for coin in ("BTC", "ETH", "SOL", "BNB"):
            rem_u[coin] = round(max(0.0, scaled_caps["max_underlying_open_risk_usd"]), 4) if tot_locked <= 1e-6 else round(
                max(0.0, float(rem_u.get(coin, scaled_caps["max_underlying_open_risk_usd"]))), 4
            )
        lwq["remaining_underlying_quota_usd"] = rem_u
        init_cap_val = float(paper_caps["initial_capital_usd"] or 1269.0)
        q_ts = str(latest_quota.get("timestamp_utc") or "")
        acc_started = str(overview.get("started_at") or "")
        if (not q_ts or (acc_started and q_ts < acc_started)) and float(lwq.get("onchain_free_usdc_balance") or 0.0) < init_cap_val:
            old_bal = float(lwq.get("onchain_free_usdc_balance") or 0.0)
            lwq["onchain_free_usdc_balance"] = round(old_bal + 800.0 if 400.0 <= old_bal < init_cap_val else init_cap_val, 6)
        latest_quota["live_wallet_and_quotas"] = lwq
        latest_quota["paper_account_mirror"] = {
            "paper_cash_usd": float(overview.get("cash") or init_cap_val),
            "paper_locked_cost_usd": float(overview.get("locked_cost") or 0.0),
            "paper_realized_pnl_usd": float(overview.get("realized_pnl") or 0.0),
        }

        pipeline_obj = self.pipeline.status() if self.pipeline else {"active": False, "models": {}, "status": "NOT_ENABLED"}
        status_5m_live = self._build_live_5m_diagnostics(
            pipeline=pipeline_obj,
            monitor_audit=self.monitor_log.summary(),
            timeline=timeline,
            timeline_summary=timeline_summary,
        )
        return {
            "now": utcnow(),
            "order_repo_exists": order_root.is_dir(),
            "process_running": process_running,
            "service": service,
            "active_mode": service.get("mode") or live_cfg.get("execution_mode") or "live",
            "live_scale_factor": scale,
            "paper_policy_caps": paper_caps,
            "scaled_live_caps": scaled_caps,
            "latest_quota": latest_quota,
            "quota_history": raw_quotas[-40:],
            "summary": {
                "total_signals": len(dedup_orders),
                "filled_orders": filled_orders,
                "total_spent_usdc": round(total_spent_usdc, 4),
                "settled_markets": len(dedup_settlements),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round(100.0 * wins / (wins + losses), 2) if (wins + losses) > 0 else None,
                "realized_pnl_usdc": round(realized_pnl_usdc, 4),
                "redeemed_usdc_total": round(redeemed_usdc_total, 4),
                "avg_detection_lag_ms": round(sum(detection_lags) / len(detection_lags), 2) if detection_lags else None,
                "avg_clob_rtt_ms": round(sum(rtts) / len(rtts), 2) if rtts else None,
                "avg_slippage_bps": round(sum(slippages) / len(slippages), 2) if slippages else None,
            },
            "status_5m_live": status_5m_live,
            "cohort_timeline": timeline,
            "cohort_summary": timeline_summary,
            "orders": dedup_orders[:50],
            "settlements": dedup_settlements[:50],
            "recent_logs": (stdout_tail + stderr_tail)[-25:],
        }

    def cycle(self):
        self.monitor_log.heartbeat("market_observer")
        try:
            backoff = binance_backoff(self.runtime)
            if backoff:
                reference = {"status": "RATE_LIMIT_BACKOFF", "quotes": [], "backoff": backoff, "observed_at": utcnow()}
            elif self.pipeline is not None:
                # The model worker owns the Binance input stream and its shared
                # backoff. Avoid a second, independent requester for optional
                # reference quotes; Polymarket executable books remain separate.
                reference = {"status": "MODEL_INPUT_FEED_ONLY", "quotes": [], "observed_at": utcnow()}
            else:
                with self.monitor_log.phase("market_observer", "collect_reference_snapshot", slow_warn_seconds=12.0):
                    reference = collect_reference_snapshot()
            self.observations.save("reference", reference)
            try:
                from tools.argon4_forward.polymarket import ALL_STREAMS, STREAMS as PM_STREAMS, collect_market_snapshot
                policy_order = load_json(ROOT / "config/argon4_forward_policy.json", {}).get("order", ())
                target_streams = ALL_STREAMS if "XRP_15m" in policy_order else PM_STREAMS
                with self.monitor_log.phase("market_observer", "collect_market_snapshot", slow_warn_seconds=15.0):
                    markets = collect_market_snapshot(streams=target_streams, transport=self.market_transport)
            except Exception as exc:
                self.monitor_log.record("WARN", "MARKET_SNAPSHOT_FAILED",
                    f"Polymarket盘口采集失败: {type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__, error=str(exc)[:200])
                markets = {"status": "UNAVAILABLE", "observed_at": utcnow(), "streams": {},
                           "errors": [type(exc).__name__]}
            self.observations.save("polymarket", markets)
            self.last_cycle = utcnow()
            self.last_error = None
            if self.pipeline is None:
                self.account.heartbeat("WAITING_FOR_MODEL", BLOCKERS[0]["message"], details={"last_observation": self.last_cycle})
        except Exception as exc:
            self.last_error = type(exc).__name__
            self.monitor_log.record("ERROR", "OBSERVER_CYCLE_ERROR",
                f"监控主循环异常: {type(exc).__name__}: {exc}",
                error_type=type(exc).__name__, error=str(exc)[:200])

    def worker(self, interval):
        self.monitor_log.start_watchdog()
        try:
            while not self.stop.is_set():
                self.cycle()
                self.stop.wait(interval)
        finally:
            self.monitor_log.close()
            self.market_transport.close()


def handler_for(app, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Never log URLs/query strings, request headers, or account payloads.
            return

        def send_bytes(self, status, data, mime):
            try:
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
                self.end_headers()
                self.wfile.write(data)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                return

        def do_GET(self):
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed_hosts:
                return self.send_bytes(403, b"Local host only", "text/plain")
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in allowed_hosts}:
                return self.send_bytes(403, b"Cross origin denied", "text/plain")
            path = urlsplit(self.path).path
            routes = {"/api/status": app.status, "/api/ledger": app.account.ledger,
                      "/api/equity": app.account.equity_history,
                      "/api/live_dashboard": app.live_dashboard}
            if path in routes:
                try:
                    payload = json.dumps(routes[path](), ensure_ascii=False, allow_nan=False).encode("utf-8")
                except Exception:
                    return self.send_bytes(503, b'{"error":"STATE_UNAVAILABLE"}', "application/json")
                return self.send_bytes(200, payload, "application/json; charset=utf-8")
            assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "application/javascript"),
                      "/style.css": ("style.css", "text/css")}
            if path in assets:
                filename, mime = assets[path]
                return self.send_bytes(200, (ROOT / "dashboard/argon4_forward" / filename).read_bytes(), mime + "; charset=utf-8")
            self.send_bytes(404, b"Not found", "text/plain")

        def do_POST(self):
            self.send_bytes(405, b"Read only", "text/plain")

        do_PUT = do_DELETE = do_PATCH = do_POST
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8897)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "paper_runtime/argon4_forward_20261105")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--enable-paper", action="store_true", help="Enable isolated local-model paper worker; never real trading")
    args = parser.parse_args()
    runtime = args.runtime_dir.resolve()
    if not runtime.is_relative_to((ROOT / "paper_runtime").resolve()):
        parser.error("Runtime must stay inside this workspace's paper_runtime")
    if not 1024 <= args.port <= 65535 or args.interval < 30:
        parser.error("Invalid port or polling interval")
    capital = (
        args.capital
        if args.capital is not None
        else float(load_json(ROOT / "config/argon4_forward_policy.json", {}).get("initial_capital_usd", 10000))
    )
    # Reserve the listener before constructing any account/worker. A duplicate
    # startup must fail before it can touch the paper account or input files.
    server = None if args.once else ThreadingHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    try:
        app = Application(runtime, capital, enable_paper=args.enable_paper)
    except BaseException:
        if server is not None:
            server.server_close()
        raise
    if args.once:
        try:
            app.cycle()
            print(json.dumps(app.status(), ensure_ascii=False, allow_nan=False))
        finally:
            app.market_transport.close()
            if app.pipeline:
                app.pipeline.close()
            if app.telemetry:
                app.telemetry.close(wait=False)
        return
    server.RequestHandlerClass = handler_for(app, args.port)
    (runtime / "service.json").write_text(json.dumps({"pid": os.getpid(), "started_at": app.started_at,
        "url": f"http://127.0.0.1:{args.port}", "mode": "LOCAL_MODEL_PAPER" if args.enable_paper else "OBSERVING_ONLY", "live_enabled": False}), encoding="utf-8")
    app.monitor_log.record(
        "INFO",
        "SERVICE_STARTED",
        f"ARGON-4 本地监控与模拟服务已启动: http://127.0.0.1:{args.port} (mode={'LOCAL_MODEL_PAPER' if args.enable_paper else 'OBSERVING_ONLY'})",
        pid=os.getpid(),
        port=args.port,
        enable_paper=bool(args.enable_paper),
    )
    worker = threading.Thread(target=app.worker, args=(args.interval,), daemon=True)
    worker.start()
    pipeline_worker = None
    if app.pipeline:
        pipeline_worker = threading.Thread(target=app.pipeline.worker, daemon=True)
        pipeline_worker.start()
    telemetry_worker = None
    if app.telemetry:
        telemetry_worker = threading.Thread(target=app.telemetry.worker, args=(app.stop,), daemon=True)
        telemetry_worker.start()
    print(f"Local paper panel: http://127.0.0.1:{args.port}; live trading disabled", flush=True)
    try:
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        if app.pipeline:
            app.pipeline.stop.set()
        worker.join(timeout=15)
        if pipeline_worker:
            pipeline_worker.join(timeout=15)
        if app.telemetry:
            app.telemetry.close(wait=False)
        if telemetry_worker:
            telemetry_worker.join(timeout=2)
        server.server_close()


if __name__ == "__main__":
    main()
