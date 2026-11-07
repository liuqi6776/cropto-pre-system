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
sys.path.insert(0, str(ROOT))
from tools.argon4_forward.account import Account
from tools.argon4_forward.monitor import ObservationStore, collect_reference_snapshot, utcnow
from tools.argon4_forward.stage import stage_profile, StageRejected, DISABLED

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
        self.stop = threading.Event()
        self.started_at = utcnow()
        self.last_cycle = None
        self.last_error = None
        from tools.argon4_forward.polymarket_transport import PublicPolymarketTransport
        self.market_transport = PublicPolymarketTransport(pool_size=4)
        self.pipeline = None
        if enable_paper:
            from tools.argon4_forward.pipeline import ForwardPipeline
            self.pipeline = ForwardPipeline(self.account, self.runtime, self.observations)
        self.account.heartbeat("WAITING_FOR_MODEL", BLOCKERS[0]["message"])

    def status(self):
        allocation = load_json(ROOT / "config/argon4_forward_policy.json", {"status": "AWAITING_RANKING_CONFIRMATION", "order": STREAMS})
        training = load_json(ROOT / "reports/argon4_local_v4_20261106/progress.json", {"status": "PREPARING", "phase": "本地训练准备", "completed_streams": []})
        live = load_json(ROOT / "config/argon4_live_preflight.json", {})
        pipeline = self.pipeline.status() if self.pipeline else {"active": False, "models": {}, "status": "NOT_ENABLED"}
        models = pipeline.get("models", {})
        blockers = ([{"code": pipeline["status"], "message": pipeline.get("reason")}]
                    if self.pipeline and pipeline.get("reason") else ([] if pipeline.get("active") else BLOCKERS))
        try:
            stage = stage_profile(allocation)
        except StageRejected:
            stage = {"configured": True, "status": "INVALID", "enabled_streams": [],
                     "disabled_streams": list(STREAMS), "live_enabled": False}
            blockers = [{"code": "INVALID_EXECUTION_STAGE", "message": "阶段配置未通过检查，暂停新模拟单"}]
        return {"schema_version": 1, "now": utcnow(), "service_started_at": self.started_at,
                "service_status": pipeline["status"] if self.pipeline else "OBSERVING_ONLY",
                "paper_trading_active": pipeline.get("active", False) and stage["status"] != "INVALID", "live_enabled": False,
                "strategy": "ARGON4_SEVEN_STREAM_PRIORITY_WATERFALL",
                "account": self.account.overview(), "blockers": blockers,
                "last_cycle": self.last_cycle, "last_error": self.last_error,
                "reference": self.observations.latest("reference"),
                "markets": self.observations.latest("polymarket"),
                "allocation": allocation,
                "stage": stage,
                "training": training,
                "pipeline": pipeline,
                "backtest": load_json(ROOT / "reports/argon4_local_v4_20261106/summary.json", {}),
                "streams": [{"stream": s, "status": DISABLED if s in stage["disabled_streams"] else "LOCAL_MODEL_LOADED" if s in models else "WAITING_FOR_MODEL",
                             "stage_enabled": s in stage["enabled_streams"], "model": models.get(s)} for s in allocation.get("order", STREAMS)],
                "live_preflight": {"venue": "polymarket", "status": "DISABLED", "signer_connected": False,
                                   "execution_stage": stage, "live_enabled": False,
                                   "wallet_configured": False, "residency_confirmed": live.get("residency_confirmed", False),
                                   "residency_country": live.get("residency_country"),
                                   "planned_total_capital_usd": live.get("planned_total_capital_usd"),
                                   "eligibility_status": live.get("eligibility_status", "UNVERIFIED")},
                "limitations": ["账户从创建时刻起算，历史研究收益不计入", "本地模型与原作者模型不等同；以实际加载版本为准",
                                "盘口观察不证明挂单成交；模拟结算须来自平台官方结果", "本机运行时采集，关机或休眠期间不补造观测"]}

    def cycle(self):
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
                reference = collect_reference_snapshot()
            self.observations.save("reference", reference)
            try:
                from tools.argon4_forward.polymarket import collect_market_snapshot
                markets = collect_market_snapshot(transport=self.market_transport)
            except Exception as exc:
                markets = {"status": "UNAVAILABLE", "observed_at": utcnow(), "streams": {},
                           "errors": [type(exc).__name__]}
            self.observations.save("polymarket", markets)
            self.last_cycle = utcnow()
            self.last_error = None
            if self.pipeline is None:
                self.account.heartbeat("WAITING_FOR_MODEL", BLOCKERS[0]["message"], details={"last_observation": self.last_cycle})
        except Exception as exc:
            self.last_error = type(exc).__name__

    def worker(self, interval):
        try:
            while not self.stop.is_set():
                self.cycle()
                self.stop.wait(interval)
        finally:
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
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--enable-paper", action="store_true", help="Enable isolated local-model paper worker; never real trading")
    args = parser.parse_args()
    runtime = args.runtime_dir.resolve()
    if not runtime.is_relative_to((ROOT / "paper_runtime").resolve()):
        parser.error("Runtime must stay inside this workspace's paper_runtime")
    if not 1024 <= args.port <= 65535 or args.interval < 30:
        parser.error("Invalid port or polling interval")
    app = Application(runtime, args.capital, enable_paper=args.enable_paper)
    if args.once:
        try:
            app.cycle()
            print(json.dumps(app.status(), ensure_ascii=False, allow_nan=False))
        finally:
            app.market_transport.close()
            if app.pipeline:
                app.pipeline.close()
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(app, args.port))
    (runtime / "service.json").write_text(json.dumps({"pid": os.getpid(), "started_at": app.started_at,
        "url": f"http://127.0.0.1:{args.port}", "mode": "LOCAL_MODEL_PAPER" if args.enable_paper else "OBSERVING_ONLY", "live_enabled": False}), encoding="utf-8")
    worker = threading.Thread(target=app.worker, args=(args.interval,), daemon=True)
    worker.start()
    pipeline_worker = None
    if app.pipeline:
        pipeline_worker = threading.Thread(target=app.pipeline.worker, daemon=True)
        pipeline_worker.start()
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
        server.server_close()


if __name__ == "__main__":
    main()
