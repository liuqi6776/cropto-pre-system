"""Second, separately authorized 75-second public benchmark: memory-only WS receive."""
import asyncio
from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import time
import uuid

import compare_public_book as base

DURATION = 75
MAX_HTTP = 36
INTERVAL = 2.1
RAW_BUDGET = 32 * 1024 * 1024


class MemoryHTTP:
    def __init__(self): self.rows = []
    def save(self, kind, data): self.rows.append({"kind": kind, **data})


async def run():
    candidates, provenance = base.select_tokens()
    selected = [v for v in candidates if v["stream"] == "BTC_15m"]
    if len(selected) != 2 or not all(v["active_window_at_selection"] for v in selected):
        raise ValueError("CURRENT_BTC15_PAIR_UNAVAILABLE")
    folder = base.OUTPUT_ROOT / (base.utc().strftime("%Y%m%dT%H%M%SZ")+"_memory_"+uuid.uuid4().hex[:10])
    folder.mkdir(exist_ok=False)
    manifest = {"classification": "SECOND_INDEPENDENT_BOUNDED_PUBLIC_BENCHMARK_NOT_EXECUTION",
        "duration_seconds": DURATION, "max_http_requests": MAX_HTTP, "actual_http_min_spacing_seconds": INTERVAL,
        "outcomes": selected, "production_snapshot": provenance,
        "receiver": "recv, monotonic/local receive stamps, bounded raw memory append only; parsing, hashing and disk writes occur after capture",
        "raw_memory_conservative_character_budget": RAW_BUDGET,
        "source_script_sha256": base.digest(Path(__file__).read_bytes()),
        "helper_script_sha256": base.digest(Path(base.__file__).read_bytes()),
        "source_age_status": "UNAVAILABLE_UNCALIBRATED; compare monotonic RTT and identical-state arrival order only",
        "source_docs": "https://docs.polymarket.com/market-data/realtime-data",
        "user_requested_api_docs": "https://docs.polymarket.com/api-reference/predictions/overview",
        "no_credentials_orders_or_production_changes": True,
        "first_trial_excluded": "First trial has synchronous per-frame parse/hash/disk overhead, receive backlog, disconnection and a 1.985s minimum HTTP spacing; do not pool results."}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tokens = {t["asset_id"]: t for t in selected}
    stop = asyncio.Event()
    frames, sent, errors = [], [], []
    http = MemoryHTTP()
    byte_bound = 0
    started = time.monotonic()
    until = started + DURATION
    print(json.dumps({"status": "STARTING_MEMORY_RECEIVE", "folder": str(folder), "outcomes": len(tokens)}), flush=True)

    async def receive(ws):
        nonlocal byte_bound
        try:
            while not stop.is_set() and time.monotonic() < until:
                try:
                    raw = await asyncio.wait_for(ws.recv(), min(2., max(.01, until-time.monotonic())))
                except asyncio.TimeoutError:
                    continue
                received_mono, received_wall = time.monotonic(), base.utc().isoformat()
                # Four bytes/character conservatively bounds UTF-8 without
                # encoding/hash/JSON work on the receiver's critical path.
                byte_bound += len(raw) * (4 if isinstance(raw, str) else 1)
                if byte_bound > RAW_BUDGET:
                    errors.append({"source": "receiver", "error": "RAW_MEMORY_BOUND_REACHED"})
                    stop.set()
                    break
                frames.append((received_mono, received_wall, raw))
                if len(frames) % 128 == 0:
                    await asyncio.sleep(0)
        except Exception as exc:
            if not stop.is_set():
                errors.append({"source": "websocket", "error": type(exc).__name__,
                    "received_close_code": getattr(getattr(exc, "rcvd", None), "code", None),
                    "sent_close_code": getattr(getattr(exc, "sent", None), "code", None)})
            stop.set()

    async def heartbeat(ws):
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), 10)
                break
            except asyncio.TimeoutError:
                try:
                    await ws.send("PING")
                    sent.append((time.monotonic(), base.utc().isoformat(), "PING"))
                except Exception as exc:
                    errors.append({"source": "heartbeat", "error": type(exc).__name__})
                    stop.set()

    async def poll():
        last_actual = -float("inf")
        for number in range(MAX_HTTP):
            # Recheck after each wait: timers may wake slightly early on Windows.
            while not stop.is_set() and time.monotonic() < min(until, last_actual+INTERVAL):
                delay = min(until, last_actual+INTERVAL)-time.monotonic()
                try:
                    await asyncio.wait_for(stop.wait(), max(.001, delay))
                except asyncio.TimeoutError:
                    pass
            if stop.is_set() or time.monotonic() >= until: break
            chosen = selected[number % len(selected)]
            if base.utc()+timedelta(seconds=5) >= base.stamp(chosen["window_end"]):
                errors.append({"source": "selection", "error": "EXPIRY_MARGIN"})
                stop.set()
                break
            receipt = await asyncio.to_thread(base.get_http, chosen, http)
            last_actual = receipt["requested_monotonic"]
            if receipt.get("http_status") in (418,429) or receipt.get("error"):
                errors.append({"source": "http", "error": receipt.get("error", "RATE_LIMIT"), "http_status": receipt.get("http_status")})
                stop.set()
                break
            if (number+1) % 12 == 0:
                print(json.dumps({"status": "OBSERVING_MEMORY", "elapsed_seconds": time.monotonic()-started,
                                  "http_requests": number+1, "ws_frames": len(frames)}), flush=True)

    handshake = time.monotonic()
    try:
        async with base.NoRedirectConnect(base.WS_URL, proxy=None, open_timeout=10, close_timeout=3,
                    ping_interval=None, max_size=2_000_000, max_queue=128,
                    user_agent_header="argon4-public-readonly-memory-comparison/1") as ws:
            sent.append((handshake, base.utc().isoformat(), "CONNECTION_OPENED"))
            await ws.send(json.dumps({"assets_ids": list(tokens), "type": "market"}))
            tasks = [asyncio.create_task(receive(ws)), asyncio.create_task(heartbeat(ws)), asyncio.create_task(poll())]
            try:
                await asyncio.wait_for(stop.wait(), max(.01, until-time.monotonic()))
            except asyncio.TimeoutError:
                pass
            stop.set()
            await asyncio.gather(*tasks)
    except Exception as exc:
        errors.append({"source": "handshake", "error": type(exc).__name__})
        stop.set()
    captured_elapsed = time.monotonic()-started

    # All work below is offline, after both network paths have stopped.
    events, initial, pongs = [], set(), []
    with (folder / "raw_receipts.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for mono, wall, raw in frames:
            if isinstance(raw, bytes): raw = raw.decode("utf-8")
            handle.write(json.dumps({"kind": "ws_receive", "received_monotonic": mono,
                "received_at_local_utc": wall, "raw_text": raw, "raw_sha256": base.digest(raw.encode())})+"\n")
            if raw == "PONG":
                pongs.append(mono)
                continue
            try:
                events.extend(base.parse_events(json.loads(raw), mono, wall, tokens, initial))
            except Exception as exc:
                errors.append({"source": "offline_parse", "error": type(exc).__name__})
        for mono, wall, raw in sent:
            handle.write(json.dumps({"kind": "ws_send", "sent_monotonic": mono,
                "sent_at_local_utc": wall, "raw_text": raw})+"\n")
        for row in http.rows:
            if row.get("raw_text") is not None: row["raw_sha256"] = base.digest(row["raw_text"].encode())
            handle.write(json.dumps(row)+"\n")
    first_ws = {}
    for event in events:
        key = tuple(event[k] for k in ("asset_id", "condition_id", "official_hash", "source_timestamp_ms"))
        first_ws.setdefault(key, event)
    seen, matched, exclusions = set(), [], Counter()
    for row in http.rows:
        if row.get("http_status") != 200 or row.get("error"):
            exclusions["unsuccessful_http"] += 1
            continue
        key = tuple(row.get(k) for k in ("asset_id", "condition_id", "official_hash", "source_timestamp_ms"))
        if key in seen:
            exclusions["repeated_static_http_state"] += 1
            continue
        seen.add(key)
        event = first_ws.get(key)
        if event is None:
            exclusions["no_exact_same_hash_and_timestamp_ws_match"] += 1
            continue
        if event["bootstrap_or_first_token_event"]:
            exclusions["bootstrap_not_new_quote"] += 1
            continue
        if event["canonical_book_sha256"] and event["canonical_book_sha256"] != row["canonical_book_sha256"]:
            exclusions["fullbook_content_mismatch"] += 1
            continue
        matched.append({"asset_id": key[0], "condition_id": key[1], "official_hash": key[2], "source_timestamp_ms": key[3],
            "ws_event_type": event["event_type"], "fullbook_equivalence_checked": bool(event["canonical_book_sha256"]),
            "websocket_first_lead_seconds": row["received_monotonic"]-event["received_monotonic"],
            "ws_received_monotonic": event["received_monotonic"], "http_received_monotonic": row["received_monotonic"]})
    (folder / "normalized_ws_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (folder / "matched_pairs.json").write_text(json.dumps(matched, indent=2), encoding="utf-8")
    successes = [r for r in http.rows if r.get("http_status") == 200 and not r.get("error")]
    spacing = [b["requested_monotonic"]-a["requested_monotonic"] for a,b in zip(http.rows,http.rows[1:])]
    if any(value < INTERVAL for value in spacing): errors.append({"source": "schedule", "error": "MINIMUM_SPACING_VIOLATION"})
    summary = {"classification": manifest["classification"], "capture_elapsed_seconds": captured_elapsed,
        "http_requests": len(http.rows), "http_successes": len(successes),
        "http_monotonic_rtt_seconds": base.stats([r["round_trip_seconds"] for r in successes]),
        "http_actual_start_spacing_seconds": base.stats(spacing),
        "ws_frames": len(frames), "ws_book_and_price_events": len(events), "ws_distinct_state_keys": len(first_ws),
        "http_distinct_state_keys": len(seen), "matched_unique_nonbootstrap_states": len(matched),
        "matched_coverage": len(matched)/len(seen) if seen else None,
        "ws_first_lead_seconds": base.stats([r["websocket_first_lead_seconds"] for r in matched]),
        "ws_first_count": sum(r["websocket_first_lead_seconds"]>0 for r in matched),
        "http_first_count": sum(r["websocket_first_lead_seconds"]<0 for r in matched),
        "fullbook_equivalence_pair_count": sum(r["fullbook_equivalence_checked"] for r in matched),
        "pings_sent": sum(raw=="PING" for _,_,raw in sent), "pongs_received": len(pongs),
        "exclusions": dict(exclusions), "errors": errors,
        "source_age_status": manifest["source_age_status"],
        "interpretation_limits": "75-second BTC15 pair only; polling delay is included (2.1s global, ~4.2s/outcome). Equal official hash+source time does not prove full depth equality for price_change-only matches. No deployment/performance claim for other tokens or an unknown API.",
        "status": "COMPLETE" if not errors else "STOPPED_OR_LIMITED", "no_production_or_order_writes": True}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"folder": str(folder), **summary}), flush=True)


if __name__ == "__main__":
    asyncio.run(run())
