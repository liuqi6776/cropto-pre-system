"""Local persistent audit & stall watchdog logger for ARGON-4 forward monitor.

Records to `<runtime>/logs/monitor_audit.log` (human-readable) and
`<runtime>/logs/monitor_audit.jsonl` (structured JSONL):
1. Every cohort evaluation and every per-stream reason for not trading.
2. Every cohort-level block (rate limit, input gap, price revision, clock skew,
   abandoned recheck, or cycle exception).
3. Slow execution steps and real-time thread hang/stall detection with live
   Python stack traces via `sys._current_frames()`.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
import traceback


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MonitorAuditLogger:
    def __init__(self, runtime: Path | str, *, max_recent: int = 100,
                 stall_threshold_seconds: float = 25.0,
                 heartbeat_stall_seconds: float = 65.0):
        self.runtime = Path(runtime)
        self.log_dir = self.runtime / "logs"
        self.text_log_path = self.log_dir / "monitor_audit.log"
        self.jsonl_log_path = self.log_dir / "monitor_audit.jsonl"
        self.stall_threshold_seconds = float(stall_threshold_seconds)
        self.heartbeat_stall_seconds = float(heartbeat_stall_seconds)
        self._lock = threading.RLock()
        self._recent: deque[dict] = deque(maxlen=max_recent)
        self._active_phases: dict[int, dict] = {}
        self._worker_heartbeats: dict[str, dict] = {}
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    def record(self, level: str, event: str, message: str, **details) -> dict:
        entry = {
            "ts": _utc_iso(),
            "level": level.upper(),
            "event": event,
            "message": message,
            "details": details,
        }
        line_text = f"{entry['ts']} [{entry['level']}] [{entry['event']}] {message}"
        if details:
            try:
                compact = json.dumps(details, ensure_ascii=False, sort_keys=True, default=str)
                line_text = f"{line_text} | {compact}"
            except Exception:
                pass
        with self._lock:
            self._recent.append(entry)
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with self.text_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line_text + "\n")
                with self.jsonl_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            except OSError:
                pass
        return entry

    def recent_events(self, limit: int = 40) -> list[dict]:
        with self._lock:
            items = list(self._recent)
        return items[-limit:]

    def summary(self) -> dict:
        with self._lock:
            active = [
                {
                    "worker": info["worker"],
                    "phase": info["phase"],
                    "elapsed_seconds": round(time.monotonic() - info["started_monotonic"], 3),
                    "cohort": info.get("cohort"),
                }
                for info in self._active_phases.values()
            ]
            heartbeats = {
                name: {
                    "last_seen_utc": hb["last_seen_utc"],
                    "age_seconds": round(time.monotonic() - hb["last_monotonic"], 3),
                    "thread_id": hb["thread_id"],
                }
                for name, hb in self._worker_heartbeats.items()
            }
        return {
            "text_log_path": str(self.text_log_path),
            "jsonl_log_path": str(self.jsonl_log_path),
            "active_phases": active,
            "worker_heartbeats": heartbeats,
            "recent_events": self.recent_events(25),
        }

    def heartbeat(self, worker: str) -> None:
        tid = threading.get_ident()
        with self._lock:
            prev = self._worker_heartbeats.get(worker)
            was_stalled = bool(prev and prev.get("stalled_logged"))
            self._worker_heartbeats[worker] = {
                "last_monotonic": time.monotonic(),
                "last_seen_utc": _utc_iso(),
                "thread_id": tid,
                "stalled_logged": False,
            }
        if was_stalled:
            self.record("INFO", "WORKER_HEARTBEAT_RECOVERED",
                        f"工作线程 {worker} 心跳已恢复 / Worker {worker} heartbeat recovered",
                        worker=worker, thread_id=tid)

    @contextmanager
    def phase(self, worker: str, phase_name: str, *, cohort: str | None = None,
              slow_warn_seconds: float = 8.0):
        tid = threading.get_ident()
        started = time.monotonic()
        with self._lock:
            self._active_phases[tid] = {
                "worker": worker,
                "phase": phase_name,
                "cohort": cohort,
                "started_monotonic": started,
                "started_utc": _utc_iso(),
                "slow_warn_seconds": float(slow_warn_seconds),
                "stalled_logged": False,
            }
            self._worker_heartbeats[worker] = {
                "last_monotonic": started,
                "last_seen_utc": _utc_iso(),
                "thread_id": tid,
                "stalled_logged": False,
            }
        try:
            yield
        except Exception as exc:
            elapsed = round(time.monotonic() - started, 3)
            self.record(
                "ERROR",
                "PHASE_EXCEPTION",
                f"阶段异常 [{worker}:{phase_name}] 耗时 {elapsed}s: {type(exc).__name__}: {exc}",
                worker=worker,
                phase=phase_name,
                cohort=cohort,
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error=str(exc)[:240],
            )
            raise
        finally:
            elapsed = round(time.monotonic() - started, 3)
            with self._lock:
                info = self._active_phases.pop(tid, None)
                if worker in self._worker_heartbeats:
                    self._worker_heartbeats[worker]["last_monotonic"] = time.monotonic()
                    self._worker_heartbeats[worker]["last_seen_utc"] = _utc_iso()
            if info and info.get("stalled_logged"):
                self.record(
                    "INFO",
                    "STALL_RECOVERED",
                    f"卡顿阶段已恢复 [{worker}:{phase_name}] 总耗时 {elapsed}s",
                    worker=worker,
                    phase=phase_name,
                    cohort=cohort,
                    elapsed_seconds=elapsed,
                )
            elif elapsed >= slow_warn_seconds:
                self.record(
                    "WARN",
                    "SLOW_PHASE",
                    f"阶段耗时偏慢 [{worker}:{phase_name}] 耗时 {elapsed}s (阈值 {slow_warn_seconds}s)",
                    worker=worker,
                    phase=phase_name,
                    cohort=cohort,
                    elapsed_seconds=elapsed,
                    threshold_seconds=slow_warn_seconds,
                )

    def _capture_thread_stack(self, thread_id: int) -> list[str]:
        frames = sys._current_frames()
        frame = frames.get(thread_id)
        if frame is None:
            return []
        extracted = traceback.extract_stack(frame)
        return [f"{item.filename}:{item.lineno} in {item.name}" for item in extracted[-8:]]

    def check_for_stalls(self) -> list[dict]:
        now_mono = time.monotonic()
        alerts = []
        with self._lock:
            phases_snapshot = list(self._active_phases.items())
            hb_snapshot = list(self._worker_heartbeats.items())
        for tid, info in phases_snapshot:
            elapsed = round(now_mono - info["started_monotonic"], 3)
            if elapsed >= self.stall_threshold_seconds and not info.get("stalled_logged"):
                with self._lock:
                    if tid in self._active_phases:
                        self._active_phases[tid]["stalled_logged"] = True
                stack = self._capture_thread_stack(tid)
                alert = self.record(
                    "WARN",
                    "THREAD_STALL_DETECTED",
                    f"检测到潜在卡死/阻塞 [{info['worker']}:{info['phase']}] 已持续 {elapsed}s",
                    worker=info["worker"],
                    phase=info["phase"],
                    cohort=info.get("cohort"),
                    thread_id=tid,
                    elapsed_seconds=elapsed,
                    stack_tail=stack,
                )
                alerts.append(alert)
        for worker, hb in hb_snapshot:
            age = round(now_mono - hb["last_monotonic"], 3)
            if age >= self.heartbeat_stall_seconds and not hb.get("stalled_logged"):
                with self._lock:
                    if worker in self._worker_heartbeats:
                        self._worker_heartbeats[worker]["stalled_logged"] = True
                stack = self._capture_thread_stack(hb["thread_id"])
                alert = self.record(
                    "WARN",
                    "WORKER_HEARTBEAT_STALL",
                    f"检测到工作线程心跳超时 [{worker}] 已 {age}s 未更新",
                    worker=worker,
                    thread_id=hb["thread_id"],
                    age_seconds=age,
                    stack_tail=stack,
                )
                alerts.append(alert)
        return alerts

    def start_watchdog(self, poll_interval_seconds: float = 5.0) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()

        def _loop():
            while not self._watchdog_stop.wait(poll_interval_seconds):
                try:
                    self.check_for_stalls()
                except Exception:
                    pass

        self._watchdog_thread = threading.Thread(
            target=_loop, name="argon4-stall-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def close(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=1.5)

    def log_cohort_blocked(self, cohort: str | None, code: str, reason: str, **extra) -> dict:
        return self.record(
            "WARN",
            "COHORT_NO_TRADE_BLOCKED",
            f"窗口未交易 [{cohort or 'N/A'}] 原因={code}: {reason}",
            cohort=cohort,
            block_code=code,
            reason=reason,
            **extra,
        )

    def log_cohort_execution(self, cohort: str, policy: dict, decisions: dict,
                             outcome: dict, timing: dict | None = None) -> dict:
        order = policy.get("order") or list(decisions.keys())
        stream_outcomes = outcome.get("streams") or outcome.get("stream_results") or {}
        fills = outcome.get("fills") or []
        fills_by_stream = {f.get("stream"): f for f in fills if isinstance(f, dict) and f.get("stream")}
        fills_by_signal = {f.get("signal_id"): f for f in fills if isinstance(f, dict) and f.get("signal_id")}
        blocked_list = outcome.get("blocked") or []
        overall_status = outcome.get("status", "UNKNOWN")

        per_stream_reasons = {}
        traded_streams = []
        no_trade_streams = {}

        for stream in order:
            dec = decisions.get(stream) or {}
            dec_status = dec.get("status", "UNKNOWN")
            pred = dec.get("prediction") or {}
            st_out = stream_outcomes.get(stream) or {}
            st_status = st_out.get("status")
            sig_id = (dec.get("signal") or {}).get("signal_id")
            fill = fills_by_stream.get(stream) or (fills_by_signal.get(sig_id) if sig_id else None)

            z_v4 = pred.get("z_v4")
            abs_z = abs(float(z_v4)) if isinstance(z_v4, (int, float)) else None
            threshold = pred.get("threshold_frozen")
            side = pred.get("side") or dec.get("outcome")

            if fill or st_status in ("FILLED_BUDGET", "PARTIAL") and float(st_out.get("spent_usd", 0) or 0) > 0:
                spent = float(st_out.get("spent_usd") or (float(fill.get("cost_usd", 0)) + float(fill.get("fee_usd", 0))) if fill else 0.0)
                traded_streams.append(stream)
                per_stream_reasons[stream] = {
                    "traded": True,
                    "status": st_status or "FILLED",
                    "side": side,
                    "spent_usd": round(spent, 4),
                    "approved_usd": st_out.get("approved_usd"),
                    "order_id": st_out.get("order_id") or (fill.get("order_id") if fill else None),
                    "abs_z_v4": round(abs_z, 6) if abs_z is not None else None,
                    "threshold": round(float(threshold), 6) if isinstance(threshold, (int, float)) else None,
                }
                continue

            if dec_status == "NOT_DUE":
                why = "NOT_DUE (当前5m窗口非该15m流整点决策时刻)"
            elif dec_status == "DISABLED_BY_STAGE" or dec.get("stage_disabled"):
                why = "DISABLED_BY_STAGE (当前执行阶段配置禁用该流)"
            elif dec_status == "NO_SIGNAL":
                if pred.get("volatility_regime_guarded") is True and (
                    abs_z is None or not isinstance(threshold, (int, float)) or abs_z >= float(threshold)
                ):
                    v_reason = pred.get("volatility_regime_reason", "VOLATILITY_REGIME_GUARD")
                    rv12 = pred.get("ex_ante_rv12_bps")
                    why = (f"VOLATILITY_REGIME_GUARD ({v_reason}: ex_ante_rv12_bps={rv12}, "
                           f"|z_v4|={abs_z:.4f} >= 阈值={float(threshold):.4f}, 方向={side})"
                           if abs_z is not None and isinstance(threshold, (int, float))
                           else f"VOLATILITY_REGIME_GUARD ({v_reason}: ex_ante_rv12_bps={rv12}, 方向={side})")
                elif abs_z is not None and isinstance(threshold, (int, float)):
                    why = f"BELOW_THRESHOLD (模型置信度 |z_v4|={abs_z:.4f} < 阈值={float(threshold):.4f}, 方向={side})"
                else:
                    why = "BELOW_THRESHOLD (模型分数未达阈值)"
            elif dec_status == "SIGNAL":
                best_ask_val = st_out.get("best_ask")
                min_buy_val = st_out.get("min_buy_price")
                is_below_min_buy = (
                    isinstance(best_ask_val, (int, float))
                    and isinstance(min_buy_val, (int, float))
                    and float(best_ask_val) < float(min_buy_val)
                )
                if st_status == "PRICE_LIMIT_NOT_MET":
                    if is_below_min_buy:
                        why = (f"ADVERSE_SELECTION_GUARD (信号触发 方向={side}, 但盘口最优卖价 best_ask={best_ask_val} "
                               f"< 逆向选择保护下限 min_buy_price={min_buy_val})")
                    else:
                        why = (f"PRICE_LIMIT_NOT_MET (信号触发 方向={side}, 但盘口最优卖价 best_ask={best_ask_val} "
                               f"> 限价上限 limit_price={st_out.get('limit_price')})")
                elif st_status == "WAITING_PRICE_RECHECK":
                    if is_below_min_buy:
                        why = (f"WAITING_PRICE_RECHECK (信号触发 方向={side}, best_ask={best_ask_val} < "
                               f"min_buy_price={min_buy_val}, 等待窗口内第 {st_out.get('attempt')} 次重检 @ {st_out.get('next_retry_at')})")
                    else:
                        why = (f"WAITING_PRICE_RECHECK (信号触发 方向={side}, best_ask={best_ask_val} > "
                               f"limit_price={st_out.get('limit_price')}, 等待窗口内第 {st_out.get('attempt')} 次重检 @ {st_out.get('next_retry_at')})")
                elif st_status == "UNREACHED":
                    why = f"UNREACHED ({st_out.get('reason', 'COHORT_BUDGET_EXHAUSTED')}: 高优先级流已用尽本窗口预算)"
                elif st_status == "BLEED_WINDOW_SINGLE_ORDER_CAP":
                    why = "BLEED_WINDOW_SINGLE_ORDER_CAP (2小时阴跌/逼空期同窗口限1单护栏触发: 已成交第1顺位主力单，跳过第2单防同窗双杀)"
                elif st_status == "ERROR":
                    why = f"EXECUTION_REJECTED ({st_out.get('reason', 'UNKNOWN_ERROR')})"
                elif st_status is None:
                    if overall_status == "WAITING_PRICE_RECHECK":
                        why = "DEFERRED_BY_HIGHER_PRIORITY_RECHECK (上层高优先级流正在等待价格重检，瀑布流暂未下放)"
                    elif blocked_list:
                        why = f"BLOCKED_BEFORE_STREAM ({', '.join(str(b) for b in blocked_list)})"
                    else:
                        why = "UNREACHED_IN_WATERFALL (上层瀑布流已终止或预算耗尽)"
                else:
                    why = f"NO_FILL ({st_status})"
            else:
                why = f"DECISION_{dec_status}"

            no_trade_streams[stream] = why
            per_stream_reasons[stream] = {
                "traded": False,
                "decision_status": dec_status,
                "execution_status": st_status,
                "side": side,
                "abs_z_v4": round(abs_z, 6) if abs_z is not None else None,
                "threshold": round(float(threshold), 6) if isinstance(threshold, (int, float)) else None,
                "no_trade_reason": why,
            }

        total_spent = round(
            sum(float(f.get("cost_usd", 0)) + float(f.get("fee_usd", 0)) for f in fills if isinstance(f, dict)),
            4,
        )
        if traded_streams:
            msg = (f"窗口执行完成 [{cohort}] 状态={overall_status} 成交流={traded_streams} "
                   f"总成交=${total_spent:.2f} | 未交易流原因: "
                   + "; ".join(f"{k}: {v}" for k, v in no_trade_streams.items()))
            level = "INFO"
            event = "COHORT_TRADED_WITH_STREAM_AUDIT"
        else:
            msg = (f"窗口无成交 [{cohort}] 状态={overall_status} 阻断={blocked_list} | 各流未交易原因: "
                   + "; ".join(f"{k}: {v}" for k, v in no_trade_streams.items()))
            level = "INFO" if overall_status in ("PROCESSED", "WAITING_PRICE_RECHECK") else "WARN"
            event = "COHORT_NO_TRADE_AUDIT"

        return self.record(
            level,
            event,
            msg,
            cohort=cohort,
            overall_status=overall_status,
            traded_streams=traded_streams,
            total_spent_usd=total_spent,
            blocked=blocked_list,
            streams=per_stream_reasons,
            duration_seconds=timing.get("duration_seconds") if isinstance(timing, dict) else None,
        )
