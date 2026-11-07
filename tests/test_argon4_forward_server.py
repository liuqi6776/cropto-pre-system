from datetime import datetime, timezone, timedelta
import http.client
import json
from threading import Thread
from http.server import ThreadingHTTPServer

import pytest

from scripts.argon4_forward_server import Application, handler_for
from tools.argon4_forward.monitor import ObservationStore, public_get
from tools.argon4_forward import monitor


def test_duplicate_listener_fails_before_creating_account(tmp_path, monkeypatch):
    from scripts import argon4_forward_server as server_module
    monkeypatch.setattr(server_module, "ROOT", tmp_path)
    monkeypatch.setattr(server_module.sys, "argv", ["argon4_forward_server.py", "--runtime-dir",
        str(tmp_path / "paper_runtime/test"), "--enable-paper"])
    def occupied(*args, **kwargs):
        raise OSError("address already in use")
    monkeypatch.setattr(server_module, "ThreadingHTTPServer", occupied)
    monkeypatch.setattr(server_module, "Application", lambda *args, **kwargs: pytest.fail("duplicate process touched account"))
    with pytest.raises(OSError, match="address already in use"):
        server_module.main()
    assert not (tmp_path / "paper_runtime").exists()


def test_invalid_engineering_cannot_report_stale_active_status(tmp_path, monkeypatch):
    from scripts import argon4_forward_server as server_module
    app = Application(tmp_path / "run", enable_paper=True)
    original = server_module.load_json
    def invalid(path, default):
        value = original(path, default)
        if path.name == "argon4_forward_policy.json":
            value["engineering"] = {"profile": "invalid"}
        return value
    monkeypatch.setattr(server_module, "load_json", invalid)
    app.pipeline.state["active"] = True
    try:
        status = app.status()
        assert status["paper_trading_active"] is False
        assert status["engineering"]["profile"] == "INVALID"
    finally:
        app.pipeline.close()
        app.market_transport.close()
        if app.telemetry:
            app.telemetry.close()


def test_initial_account_is_new_and_no_order_adapter(tmp_path):
    app = Application(tmp_path / "run")
    status = app.status()
    assert status["account"]["cash"] == 10000
    assert status["account"]["order_count"] == 0
    assert status["paper_trading_active"] is False
    assert status["live_enabled"] is False
    assert all(s["model"] is None for s in status["streams"])
    assert Application(tmp_path / "run", 99999).status()["account"]["initial_capital"] == 10000


def test_model_worker_owns_binance_requests_while_venue_observation_continues(tmp_path, monkeypatch):
    from scripts import argon4_forward_server as server_module
    from tools.argon4_forward import polymarket
    app = Application(tmp_path / "run", enable_paper=True)
    monkeypatch.setattr(server_module, "collect_reference_snapshot",
                        lambda: pytest.fail("duplicate Binance requester bypassed shared input backoff"))
    monkeypatch.setattr(polymarket, "collect_market_snapshot",
                        lambda **kwargs: {"status": "OBSERVED", "streams": {}})
    app.cycle()
    assert app.last_error is None
    assert app.observations.latest("reference")["status"] == "MODEL_INPUT_FEED_ONLY"
    assert app.observations.latest("polymarket")["status"] == "OBSERVED"
    assert app.account.overview()["order_count"] == 0


def test_failed_public_receipt_has_no_invented_body(tmp_path):
    store = ObservationStore(tmp_path / "observations.sqlite3")
    now = datetime.now(timezone.utc).isoformat()
    store.save("polymarket", {"status": "UNAVAILABLE", "receipts": [{
        "url": "https://gamma-api.polymarket.com/markets?slug=test", "received_at": now,
        "sha256": None, "http_status": None, "raw_text": None}]})
    with store.connect() as conn:
        assert conn.execute("SELECT sha256,raw_text FROM receipts").fetchone() == (None, None)
    assert store.latest("polymarket")["status"] == "UNAVAILABLE"


def test_receipt_persistence_retains_measured_request_timing(tmp_path):
    store = ObservationStore(tmp_path / "observations.sqlite3")
    receipt = {"requested_at": "2026-11-06T12:00:00Z", "received_at": "2026-11-06T12:00:00.3Z",
               "url": "https://clob.polymarket.com/book?token_id=123", "sha256": "a"*64,
               "http_status": 200, "raw_text": "{}", "body_complete": True,
               "transport_timing": {"request_elapsed_seconds": .3}}
    store.save("paper_book_receipt", {"receipts": [receipt]})
    saved = store.latest("paper_book_receipt")["receipt_metadata"][0]
    assert saved["requested_at"] == receipt["requested_at"]
    assert saved["received_at"] == receipt["received_at"]
    assert saved["transport_timing"] == receipt["transport_timing"]
    assert "raw_text" not in saved
    with store.connect() as conn:
        assert conn.execute("SELECT raw_text FROM receipts WHERE id=?", (saved["receipt_id"],)).fetchone() == ("{}",)


def test_background_and_execution_connections_have_separate_owners(tmp_path):
    app = Application(tmp_path / "run", enable_paper=True)
    assert app.market_transport is not app.pipeline.public_transport
    app.market_transport.close()
    app.pipeline.close()


def test_get_only_endpoint_allowlist_rejects_orders():
    with pytest.raises(ValueError):
        public_get("/fapi/v1/order")
    with pytest.raises(ValueError):
        public_get("/fapi/v1/ticker/bookTicker", {"symbol": "BTCUSDT", "apiKey": "not-a-key"})


def test_old_reference_quote_never_becomes_fresh_on_receipt(monkeypatch):
    now = datetime.now(timezone.utc)
    def fake_get(path, params=None):
        receipt = {"received_at": now.isoformat()}
        if path.endswith("/time"):
            return {"serverTime": int(now.timestamp()*1000)}, receipt
        return {"symbol": params["symbol"], "bidPrice": "100", "askPrice": "101",
                "time": int((now-timedelta(days=1)).timestamp()*1000)}, receipt
    monkeypatch.setattr(monitor, "public_get", fake_get)
    snapshot = monitor.collect_reference_snapshot()
    assert snapshot["status"] == "DEGRADED"
    assert all(q["status"] == "STALE" for q in snapshot["quotes"])


def test_saved_fresh_snapshot_does_not_hide_expired_market(tmp_path):
    store = ObservationStore(tmp_path / "observations.sqlite3")
    now = datetime.now(timezone.utc)
    store.save("polymarket", {"status": "WAITING_MODEL", "market_data_status": "COMPLETE", "streams": {
        "BTC_5m": {"market_status": "VERIFIED_MARKETS", "current": {
            "window_end": (now-timedelta(seconds=1)).isoformat(), "books": {
                "UP": {"source_at": (now-timedelta(minutes=5)).isoformat(), "status": "OBSERVED"}}}, "next": None}}})
    latest = store.latest("polymarket")
    assert latest["stale"] is False
    assert latest["market_data_status"] == "PARTIAL"
    assert latest["streams"]["BTC_5m"]["current"]["status"] == "EXPIRED"
    assert latest["streams"]["BTC_5m"]["current"]["books"]["UP"]["status"] == "STALE"


def test_panel_blocks_foreign_hosts_origins_writes_and_paths(tmp_path):
    app = Application(tmp_path / "account")
    # Bind before selecting the host allowlist, so tests do not need a fixed port.
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app, 0))
    port = server.server_port
    server.RequestHandlerClass = handler_for(app, port)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def get(path="/api/status", method="GET", headers=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request(method, path, headers=headers or {})
            response = conn.getresponse()
            result = response.status, response.read(), dict(response.getheaders())
            conn.close()
            return result
        code, body, headers = get()
        assert code == 200
        assert json.loads(body)["live_enabled"] is False
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert get(headers={"Host": "attacker.example"})[0] == 403
        assert get(headers={"Origin": "https://attacker.example"})[0] == 403
        assert get(method="POST")[0] == 405
        assert get("/../config/argon4_live_preflight.json")[0] == 404
        assert get("/api/place_order", method="POST")[0] == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
