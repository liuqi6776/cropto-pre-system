"""One bounded, credential-free comparison; writes only this report subtree."""
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import threading
import time
import uuid

import requests
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(__file__).resolve().parent
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
HTTP_URL = "https://clob.polymarket.com/book"
DURATION = 150
MAX_HTTP = 75
HTTP_INTERVAL = 2.0
MAX_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 50000


def utc():
    return datetime.now(timezone.utc)


def stamp(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stats(values):
    if not values:
        return {"n": 0}
    values = sorted(values)
    def percentile(p):
        index = (len(values)-1)*p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi]-values[lo])*(index-lo)
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "p95": percentile(.95), "max": max(values)}


class NoRedirectConnect(connect):
    def process_redirect(self, exc):
        return exc


class Recorder:
    def __init__(self, folder):
        self.folder = folder
        self.lock = threading.Lock()
        self.raw_bytes = 0
        self.frames = 0
    def save(self, kind, data):
        raw = data.get("raw_text", "").encode("utf-8")
        if raw:
            data["raw_sha256"] = digest(raw)
        with self.lock:
            self.raw_bytes += len(raw)
            self.frames += kind == "ws_receive"
            if self.raw_bytes > MAX_BYTES or self.frames > MAX_FRAMES:
                raise ValueError("REPORT_SIZE_LIMIT")
            with (self.folder / "raw_receipts.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"kind": kind, **data}, ensure_ascii=False, allow_nan=False)+"\n")


def select_tokens():
    path = ROOT / "paper_runtime/argon4_forward_20261105/observations.sqlite3"
    with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as con:
        row = con.execute("SELECT id,saved_at,payload FROM snapshots WHERE kind='polymarket' ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        raise ValueError("NO_LOCAL_MARKET_OBSERVATIONS")
    saved = stamp(row[1])
    now = utc()
    if not -5 <= (now-saved).total_seconds() <= 120:
        raise ValueError("LOCAL_MARKET_OBSERVATIONS_TOO_OLD")
    snapshot = json.loads(row[2])
    selected = []
    for stream in ("ETH_5m", "BTC_5m", "BTC_15m"):
        choices = snapshot["streams"][stream]
        candidates = [choices.get(k) for k in ("current", "next")]
        candidates = [v for v in candidates if isinstance(v, dict)
                      and v.get("status") == "VERIFIED_MARKET"
                      and v.get("active") is True and v.get("closed") is False
                      and stamp(v["window_end"]) > now+timedelta(seconds=DURATION+20)]
        if not candidates:
            raise ValueError("NO_CONTRACT_COVERS_BOUNDED_MEASUREMENT")
        market = min(candidates, key=lambda v: stamp(v["window_start"]))
        for side in ("UP", "DOWN"):
            token = str(market["outcome_ids"][side])
            if not re.fullmatch(r"[0-9]{1,100}", token):
                raise ValueError("INVALID_PUBLIC_OUTCOME_ID")
            selected.append({"stream": stream, "outcome": side, "asset_id": token,
                "condition_id": market["condition_id"], "market_id": market["market_id"],
                "window_start": market["window_start"], "window_end": market["window_end"],
                "active_window_at_selection": stamp(market["window_start"]) <= now < stamp(market["window_end"])})
    return selected, {"snapshot_id": row[0], "saved_at_local_utc": row[1],
                      "snapshot_sha256": digest(row[2].encode())}


def book_fingerprint(payload):
    rows = {}
    for side in ("bids", "asks"):
        if not isinstance(payload.get(side), list):
            return None
        converted = []
        for level in payload[side]:
            price, size = Decimal(str(level["price"])), Decimal(str(level["size"]))
            if not price.is_finite() or not size.is_finite() or not 0 < price < 1 or size < 0:
                raise ValueError("INVALID_PUBLIC_BOOK_LEVEL")
            converted.append((str(price.normalize()), str(size.normalize())))
        rows[side] = sorted(converted)
    return digest(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())


def parse_events(payload, received_mono, received_local, tokens, initial_seen):
    output = []
    for event in payload if isinstance(payload, list) else [payload]:
        if not isinstance(event, dict):
            continue
        kind = event.get("event_type")
        condition = event.get("market")
        source = str(event.get("timestamp", ""))
        if not re.fullmatch(r"[0-9]{13}", source):
            continue
        if kind == "book":
            changes = [event]
        elif kind == "price_change":
            changes = event.get("price_changes", [])
        else:
            continue
        for item in changes:
            token, official_hash = str(item.get("asset_id", "")), item.get("hash")
            if token not in tokens or condition != tokens[token]["condition_id"] or not official_hash:
                continue
            first_for_token = token not in initial_seen
            initial_seen.add(token)
            output.append({"asset_id": token, "condition_id": condition,
                "official_hash": str(official_hash), "source_timestamp_ms": source,
                "received_monotonic": received_mono, "received_at_local_utc": received_local,
                "event_type": kind, "bootstrap_or_first_token_event": first_for_token,
                "canonical_book_sha256": book_fingerprint(item) if kind == "book" else None})
    return output


def get_http(token, recorder):
    requested_local, requested_mono = utc().isoformat(), time.monotonic()
    receipt = {"url": HTTP_URL+"?token_id="+token["asset_id"], "asset_id": token["asset_id"],
        "requested_at_local_utc": requested_local, "requested_monotonic": requested_mono,
        "http_session_mode": "NEW_SESSION_SAME_AS_CURRENT_PRODUCTION", "http_status": None,
        "time_basis": "LOCAL_UTC_PLUS_MONOTONIC_NO_SOURCE_AGE_CALIBRATION"}
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.auth = None
            session.cookies.clear()
            session.proxies.clear()
            with session.get(HTTP_URL, params={"token_id": token["asset_id"]}, timeout=(3, 7),
                             allow_redirects=False, stream=True,
                             headers={"User-Agent": "argon4-public-readonly-comparison/1"}) as response:
                receipt["http_status"] = response.status_code
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > 2_000_000 or time.monotonic()-requested_mono > 10:
                        raise ValueError("HTTP_BODY_SIZE_OR_TIME_LIMIT")
                    chunks.append(chunk)
                received_mono, received_local = time.monotonic(), utc().isoformat()
                raw = b"".join(chunks)
                receipt.update(received_monotonic=received_mono, received_at_local_utc=received_local,
                               round_trip_seconds=received_mono-requested_mono,
                               raw_text=raw.decode("utf-8"), body_complete=True)
        if receipt["http_status"] == 200:
            data = json.loads(receipt["raw_text"])
            if str(data.get("asset_id")) != token["asset_id"] or data.get("market") != token["condition_id"]:
                raise ValueError("HTTP_OUTCOME_OR_CONDITION_MISMATCH")
            receipt.update(condition_id=data["market"], official_hash=data.get("hash"),
                           source_timestamp_ms=str(data.get("timestamp", "")),
                           canonical_book_sha256=book_fingerprint(data))
    except Exception as exc:
        receipt.update(error=type(exc).__name__, received_monotonic=time.monotonic(), received_at_local_utc=utc().isoformat())
    recorder.save("http_receive", receipt)
    return receipt


async def run():
    selected, source = select_tokens()
    folder = OUTPUT_ROOT / (utc().strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:10])
    folder.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(folder)
    manifest = {"classification": "BOUNDED_PUBLIC_DATA_OBSERVATION_NOT_EXECUTION", "duration_seconds": DURATION,
        "max_http_requests": MAX_HTTP, "global_http_min_start_spacing_seconds": HTTP_INTERVAL,
        "streams": selected, "production_snapshot": source, "source_script_sha256": digest(Path(__file__).read_bytes()),
        "source_docs": "https://docs.polymarket.com/market-data/realtime-data",
        "websocket_url": WS_URL, "http_url": HTTP_URL, "credentials": False, "orders": False,
        "production_writes": False, "source_age_status": "UNAVAILABLE_UNCALIBRATED_DO_NOT_SUBTRACT_LOCAL_UTC",
        "comparison_limit": "Observed lead includes HTTP polling cadence (six outcomes round-robin, about 12s/outcome), not pure transport latency. No candidate deployment."}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "STARTING", "folder": str(folder), "outcomes": len(selected)}), flush=True)
    tokens = {t["asset_id"]: t for t in selected}
    stop = asyncio.Event()
    ws_events, http_receipts, errors = [], [], []
    start, until = time.monotonic(), time.monotonic()+DURATION
    initial_seen = set()
    async def receive(ws):
        try:
            while not stop.is_set() and time.monotonic() < until:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(2., max(.01, until-time.monotonic())))
                except asyncio.TimeoutError:
                    continue
                received_mono, received_local = time.monotonic(), utc().isoformat()
                if isinstance(raw, bytes): raw = raw.decode("utf-8")
                recorder.save("ws_receive", {"url": WS_URL, "received_monotonic": received_mono,
                              "received_at_local_utc": received_local, "raw_text": raw})
                if raw == "PONG": continue
                ws_events.extend(parse_events(json.loads(raw), received_mono, received_local, tokens, initial_seen))
        except Exception as exc:
            errors.append({"source": "websocket", "error": type(exc).__name__})
            stop.set()
    async def heartbeat(ws):
        while not stop.is_set() and time.monotonic() < until:
            try:
                await asyncio.wait_for(stop.wait(), timeout=10)
                break
            except asyncio.TimeoutError:
                try:
                    await ws.send("PING")
                    recorder.save("ws_heartbeat", {"sent_at_local_utc": utc().isoformat(), "sent_monotonic": time.monotonic(), "raw_text": "PING"})
                except Exception as exc:
                    errors.append({"source": "heartbeat", "error": type(exc).__name__})
                    stop.set()
    async def poll():
        last = -float("inf")
        for number in range(MAX_HTTP):
            delay = max(0., HTTP_INTERVAL-(time.monotonic()-last))
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(.001, delay))
                break
            except asyncio.TimeoutError:
                pass
            if stop.is_set() or time.monotonic() >= until: break
            token = selected[number % len(selected)]
            if utc()+timedelta(seconds=5) >= stamp(token["window_end"]):
                errors.append({"source": "selection", "error": "CONTRACT_EXPIRY_MARGIN"})
                stop.set()
                break
            last = time.monotonic()
            receipt = await asyncio.to_thread(get_http, token, recorder)
            http_receipts.append(receipt)
            if receipt.get("http_status") in (418, 429) or receipt.get("error"):
                errors.append({"source": "http", "error": receipt.get("error", "HTTP_RATE_LIMIT"), "http_status": receipt.get("http_status")})
                stop.set()
                break
            if len(http_receipts) % 15 == 0:
                print(json.dumps({"status": "OBSERVING", "elapsed_seconds": round(time.monotonic()-start, 2),
                                  "http_requests": len(http_receipts), "ws_events": len(ws_events)}), flush=True)
    handshake_start = time.monotonic()
    try:
        async with NoRedirectConnect(WS_URL, proxy=None, open_timeout=10, close_timeout=3,
                 ping_interval=None, max_size=2_000_000, max_queue=512,
                 user_agent_header="argon4-public-readonly-comparison/1") as ws:
            recorder.save("ws_open", {"url": WS_URL, "requested_monotonic": handshake_start,
                "received_monotonic": time.monotonic(), "received_at_local_utc": utc().isoformat()})
            subscription = json.dumps({"assets_ids": list(tokens), "type": "market"})
            await ws.send(subscription)
            recorder.save("ws_subscription", {"sent_at_local_utc": utc().isoformat(), "sent_monotonic": time.monotonic(), "raw_text": subscription})
            tasks = [asyncio.create_task(receive(ws)), asyncio.create_task(heartbeat(ws)), asyncio.create_task(poll())]
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(.01, until-time.monotonic()))
            except asyncio.TimeoutError:
                pass
            stop.set()
            await asyncio.gather(*tasks)
    except Exception as exc:
        errors.append({"source": "handshake", "error": type(exc).__name__})
        stop.set()
    (folder / "normalized_ws_events.json").write_text(json.dumps(ws_events, indent=2), encoding="utf-8")
    first_ws = {}
    for event in ws_events:
        key = tuple(event.get(k) for k in ("asset_id", "condition_id", "official_hash", "source_timestamp_ms"))
        first_ws.setdefault(key, event)
    seen_http, pairs, excluded = set(), [], Counter()
    for response in http_receipts:
        if response.get("http_status") != 200 or response.get("error"):
            excluded["unsuccessful_http"] += 1
            continue
        key = tuple(response.get(k) for k in ("asset_id", "condition_id", "official_hash", "source_timestamp_ms"))
        if key in seen_http:
            excluded["repeated_same_http_state_not_new_quote"] += 1
            continue
        seen_http.add(key)
        event = first_ws.get(key)
        if event is None:
            excluded["no_exact_same_hash_and_source_timestamp_ws_match"] += 1
            continue
        if event["bootstrap_or_first_token_event"]:
            excluded["bootstrap_snapshot_not_new_update"] += 1
            continue
        if event["canonical_book_sha256"] and event["canonical_book_sha256"] != response["canonical_book_sha256"]:
            excluded["same_official_hash_but_fullbook_content_mismatch"] += 1
            continue
        pairs.append({"asset_id": key[0], "condition_id": key[1], "official_hash": key[2],
            "source_timestamp_ms": key[3], "ws_event_type": event["event_type"],
            "fullbook_content_equivalence_checked": bool(event["canonical_book_sha256"]),
            "ws_received_monotonic": event["received_monotonic"], "http_received_monotonic": response["received_monotonic"],
            "websocket_first_lead_seconds": response["received_monotonic"]-event["received_monotonic"]})
    (folder / "matched_pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    successful = [r for r in http_receipts if r.get("http_status") == 200 and not r.get("error")]
    summary = {"classification": manifest["classification"], "elapsed_seconds": time.monotonic()-start,
        "http_requests": len(http_receipts), "http_status_counts": dict(Counter(str(r.get("http_status")) for r in http_receipts)),
        "http_success_monotonic_rtt_seconds": stats([r["round_trip_seconds"] for r in successful]),
        "http_request_start_spacing_seconds": stats([b["requested_monotonic"]-a["requested_monotonic"] for a,b in zip(http_receipts,http_receipts[1:])]),
        "ws_frames": recorder.frames, "ws_book_and_price_events": len(ws_events),
        "ws_distinct_state_timestamp_keys": len(first_ws), "unique_http_state_timestamp_keys": len(seen_http),
        "matched_nonbootstrap_unique_states": len(pairs), "matched_http_coverage": len(pairs)/len(seen_http) if seen_http else None,
        "websocket_first_lead_seconds": stats([p["websocket_first_lead_seconds"] for p in pairs]),
        "matched_fullbook_count": sum(p["fullbook_content_equivalence_checked"] for p in pairs),
        "ws_first_count": sum(p["websocket_first_lead_seconds"] > 0 for p in pairs),
        "http_first_count": sum(p["websocket_first_lead_seconds"] < 0 for p in pairs),
        "exclusions": dict(excluded), "errors": errors, "source_age_status": manifest["source_age_status"],
        "comparison_limit": manifest["comparison_limit"], "no_trading_or_production_change": True,
        "status": "OBSERVATION_COMPLETE" if not errors else "STOPPED_WITH_RECORDED_ERROR"}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"folder": str(folder), **summary}), flush=True)


if __name__ == "__main__":
    asyncio.run(run())
