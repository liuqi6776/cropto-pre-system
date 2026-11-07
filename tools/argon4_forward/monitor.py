"""GET-only reference feed, kept separate from prediction-market execution."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from urllib.parse import urlencode
import requests

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
BASE = "https://fapi.binance.com"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def public_get(path, params=None):
    if path not in ("/fapi/v1/time", "/fapi/v1/ticker/bookTicker"):
        raise ValueError("Only public reference endpoints are allowed")
    params = params or {}
    if params and (set(params) != {"symbol"} or params["symbol"] not in SYMBOLS):
        raise ValueError("Unknown public symbol")
    url = BASE + path + ("?" + urlencode(params) if params else "")
    with requests.Session() as session:
        session.trust_env = False
        with session.get(url, timeout=(3, 5), allow_redirects=False, stream=True,
                         headers={"User-Agent": "argon4-local-paper-observer/1"}) as response:
            if response.status_code != 200:
                raise ValueError("Public reference unavailable")
            parts = []
            size = 0
            for chunk in response.iter_content(32768):
                size += len(chunk)
                if size > 1_000_000:
                    raise ValueError("Response too large")
                parts.append(chunk)
            raw = b"".join(parts)
            received = utcnow()
    return json.loads(raw), {"url": url, "received_at": received,
        "sha256": hashlib.sha256(raw).hexdigest(), "raw_text": raw.decode("utf-8"),
        "http_status": 200}


def collect_reference_snapshot():
    result = {"observed_at": utcnow(), "source": "BINANCE_USDM_REFERENCE_ONLY",
              "quotes": [], "errors": [], "receipts": []}
    try:
        clock, receipt = public_get("/fapi/v1/time")
        result["receipts"].append(receipt)
        result["server_time"] = datetime.fromtimestamp(clock["serverTime"] / 1000, timezone.utc).isoformat()
        skew = abs(datetime.now(timezone.utc).timestamp() * 1000 - int(clock["serverTime"])) / 1000
        result["clock_skew_seconds"] = round(skew, 3)
        if skew > 10:
            result["errors"].append("REFERENCE_CLOCK_SKEW")
    except Exception as exc:
        result["errors"].append("REFERENCE_CLOCK_UNAVAILABLE:" + type(exc).__name__)

    def one(symbol):
        try:
            data, receipt = public_get("/fapi/v1/ticker/bookTicker", {"symbol": symbol})
            bid, ask = float(data["bidPrice"]), float(data["askPrice"])
            if data["symbol"] != symbol or not all(math.isfinite(v) for v in (bid, ask)) or not (0 < bid <= ask):
                raise ValueError("Invalid book")
            event_at = datetime.fromtimestamp(int(data["time"]) / 1000, timezone.utc).isoformat()
            age = (datetime.fromisoformat(receipt["received_at"]) - datetime.fromisoformat(event_at)).total_seconds()
            return {"symbol": symbol, "bid": bid, "ask": ask, "event_at": event_at,
                    "received_at": receipt["received_at"], "status": "OBSERVED" if -2 <= age <= 90 else "STALE"}, receipt
        except Exception as exc:
            return {"symbol": symbol, "status": "UNAVAILABLE", "error": type(exc).__name__}, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for quote, receipt in pool.map(one, SYMBOLS):
            result["quotes"].append(quote)
            if receipt:
                result["receipts"].append(receipt)
    result["status"] = "OBSERVED" if all(q["status"] == "OBSERVED" for q in result["quotes"]) and not result["errors"] else "DEGRADED"
    return result


class ObservationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS receipts(
                    id INTEGER PRIMARY KEY, received_at TEXT NOT NULL, url TEXT NOT NULL,
                    sha256 TEXT, http_status INTEGER, raw_text TEXT);
                CREATE TABLE IF NOT EXISTS snapshots(
                    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, saved_at TEXT NOT NULL,
                    payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS snapshots_kind ON snapshots(kind,id);
            """)

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save(self, kind, data):
        payload = dict(data)
        receipts = payload.pop("receipts", [])
        with self.connect() as conn:
            receipt_metadata = []
            for receipt in receipts:
                inserted = conn.execute("INSERT INTO receipts(received_at,url,sha256,http_status,raw_text) VALUES(?,?,?,?,?)",
                    (receipt["received_at"], receipt.get("url", receipt.get("source", "")),
                     receipt["sha256"], receipt.get("http_status"), receipt["raw_text"]))
                # Preserve request timing and completion evidence alongside a
                # link to the immutable raw receipt; do not duplicate its body.
                receipt_metadata.append(dict({k: v for k, v in receipt.items() if k != "raw_text"},
                                             receipt_id=inserted.lastrowid))
            if receipts:
                payload["receipt_metadata"] = receipt_metadata
            conn.execute("INSERT INTO snapshots(kind,saved_at,payload) VALUES(?,?,?)",
                         (kind, utcnow(), json.dumps(payload, ensure_ascii=False, allow_nan=False)))

    def latest(self, kind):
        with self.connect() as conn:
            row = conn.execute("SELECT saved_at,payload FROM snapshots WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
        if not row:
            return {"status": "WAITING_FOR_FIRST_OBSERVATION", "streams": {}, "quotes": []}
        payload = json.loads(row[1])
        payload["saved_at"] = row[0]
        age = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(row[0])).total_seconds())
        payload["age_seconds"] = round(age, 1)
        payload["stale"] = age > 120
        now = datetime.now(timezone.utc)
        for state in payload.get("streams", {}).values():
            for slot in ("current", "next"):
                market = state.get(slot)
                if not market:
                    continue
                if datetime.fromisoformat(market["window_end"].replace("Z", "+00:00")) <= now:
                    market["status"] = "EXPIRED"
                    state["market_status"] = "REFRESH_NEEDED"
                for book in market.get("books", {}).values():
                    if not book:
                        continue
                    source_at = datetime.fromisoformat(book["source_at"].replace("Z", "+00:00"))
                    source_age = (now - source_at).total_seconds()
                    book["age_seconds"] = round(source_age, 1)
                    if not -2 <= source_age <= 90:
                        book["status"] = "STALE"
        if payload.get("streams") and any(
                state.get("market_status") == "REFRESH_NEEDED" or
                any(book and book.get("status") != "OBSERVED" for slot in ("current", "next")
                    for book in (state.get(slot) or {}).get("books", {}).values())
                for state in payload["streams"].values()):
            payload["market_data_status"] = "PARTIAL"
        return payload
