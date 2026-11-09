"""Independent 10,000 USD paper account and GET-only local observation panel.

Saved local models can be enabled for isolated paper execution explicitly.
This entrypoint has no signing, exchange-order, or HTTP order mutation route.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
_VENV_SITE = ROOT / ".venv/Lib/site-packages"
if _VENV_SITE.is_dir() and str(_VENV_SITE) not in sys.path:
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
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def binance_backoff(runtime):
    state = load_json(Path(runtime) / "features/binance_backoff.json", {})
    until = state.get("blocked_until_utc")
    if until and datetime.now(timezone.utc) < datetime.fromisoformat(until.replace("Z", "+00:00")):
        return state
    return None


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
        return {"schema_version": 1, "now": utcnow(), "service_started_at": self.started_at,
                "service_status": pipeline["status"] if self.pipeline else "OBSERVING_ONLY",
                "paper_trading_active": pipeline.get("active", False) and stage["status"] != "INVALID" and engineering["profile"] != "INVALID", "live_enabled": False,
                "strategy": "ARGON4_SEVEN_STREAM_PRIORITY_WATERFALL",
                "account": overview, "open_risk": open_risk, "blockers": blockers,
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
                "monitor_audit": self.monitor_log.summary(),
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
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed_hosts:
                return self.send_bytes(403, b"Local host only", "text/plain")
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in allowed_hosts}:
                return self.send_bytes(403, b"Cross origin denied", "text/plain")
            path = urlsplit(self.path).path
            routes = {"/api/status": app.status, "/api/ledger": app.account.ledger,
                      "/api/equity": app.account.equity_history}
            if path in routes:
                try:
                    payload = json.dumps(routes[path](), ensure_ascii=False, allow_nan=False).encode("utf-8")
                    return self.send_bytes(200, payload, "application/json; charset=utf-8")
                except Exception:
                    return self.send_bytes(503, b'{"error":"STATE_UNAVAILABLE"}', "application/json")
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
