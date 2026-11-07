"""Standalone, read-only v5 activation audit. No business-module imports.

Only report artifacts next to this script may be written. All SQLite handles
use mode=ro plus query_only; neither Account nor ObservationStore is created.
Run only after the operator supplies the actual activation and first cohort.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = ROOT / "paper_runtime/argon4_forward_20261105"
SEMANTICS = "ELIGIBLE_SIGNALS_APPROVED_BUDGET_V1"
BASELINE_SHA256 = "ba106275e399b7af0cf70dd8db119f7e27e4e48ab1451e12d231e6bbbc233cd5"
UNLOCKED = {"NO_SIGNAL", "NOT_DUE", "PRICE_LIMIT_NOT_MET", "FILLED_BUDGET"}
SKIPS = UNLOCKED - {"FILLED_BUDGET"}
MICRO = Decimal(1_000_000)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def stamp(value):
    value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("UTC offset required")
    return value.astimezone(timezone.utc)


def atoms(value):
    amount = Decimal(str(value)) * MICRO
    if amount != amount.to_integral_value():
        raise ValueError("non-integral currency atoms")
    return int(amount)


def usd(value):
    return float(Decimal(value) / MICRO)


def read_db(path):
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("BEGIN")
    return con


def rows(con, table, order="rowid"):
    # Callers supply fixed table/column literals; no untrusted SQL identifiers.
    return [dict(row) for row in con.execute(f"SELECT * FROM {table} ORDER BY {order}")]


def audit(activation, first_cohort):
    failures, pending, book_evidence, sequence_evidence = [], [], [], []

    def check(ok, code, **context):
        if not ok:
            failures.append({"code": code, **context})
        return bool(ok)

    baseline_path = HERE / "pre_change_account.sqlite3"
    baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    check(baseline_sha == BASELINE_SHA256, "BASELINE_FILE_CHANGED")
    with read_db(baseline_path) as con:
        baseline = {t: rows(con, t) for t in ("account", "events", "decision_cohorts", "cohort_streams", "orders")}
    with read_db(RUNTIME / "account.sqlite3") as con:
        current = {t: rows(con, t) for t in ("account", "events", "decision_cohorts", "cohort_streams", "orders", "signals", "allocation_consumption", "resolutions")}
    con = read_db(RUNTIME / "observations.sqlite3")
    try:
        observations = [dict(r) for r in con.execute("SELECT * FROM snapshots WHERE kind IN ('prediction_decisions','paper_execution','paper_book_receipt') ORDER BY id")]
        for r in observations:
            r["data"] = json.loads(r["payload"])
        predictions = {r["data"]["cohort"]: r for r in observations if r["kind"] == "prediction_decisions"}
        executions = {r["data"]["cohort"]: r for r in observations if r["kind"] == "paper_execution"}
        baseline_counts = {t: len(baseline[t]) for t in ("events", "decision_cohorts", "cohort_streams", "orders")}
        check(baseline_counts == {"events": 29, "decision_cohorts": 27, "cohort_streams": 1, "orders": 0}, "UNEXPECTED_BASELINE_COUNTS", actual=baseline_counts)
        for table, keys in (("events", ("seq",)), ("decision_cohorts", ("cohort_id",)), ("cohort_streams", ("cohort_id", "stream"))):
            by_key = {tuple(r[k] for k in keys): r for r in current[table]}
            for row in baseline[table]:
                key = tuple(row[k] for k in keys)
                check(by_key.get(key) == row, "OLD_ROW_CHANGED", table=table, key=key)
        account, original = current["account"][0], baseline["account"][0]
        check(all(account[k] == original[k] for k in ("currency", "initial_atoms", "started_at")), "INITIAL_ACCOUNT_CHANGED")
        events = current["events"]
        previous = "0" * 64
        event_by_order, event_by_skip = {}, {}
        for event in events:
            check(event["previous_hash"] == previous and event["event_hash"] == digest(previous + event["utc"] + event["event_type"] + event["payload"]), "EVENT_CHAIN_MISMATCH", seq=event["seq"])
            previous = event["event_hash"]
            event["data"] = json.loads(event["payload"])
            if event["event_type"] == "PAPER_IOC":
                event_by_order[event["data"]["order_id"]] = event
            elif event["event_type"] == "COHORT_STREAM_SKIPPED":
                event_by_skip[(event["data"]["cohort_id"], event["data"]["stream"])] = event
        cohorts = {r["cohort_id"]: r for r in current["decision_cohorts"]}
        tiers = {(r["cohort_id"], r["stream"]): r for r in current["cohort_streams"]}
        signals = {r["signal_id"]: json.loads(r["payload"]) for r in current["signals"]}
        orders = {r["order_id"]: r for r in current["orders"]}
        old_orders = {r["order_id"] for r in baseline["orders"]}
        new_orders = [r for r in current["orders"] if r["order_id"] not in old_orders]
        v5_cohorts = {key: row for key, row in cohorts.items() if json.loads(row["policy_json"]).get("waterfall_semantics") == SEMANTICS}

        def prediction_signal(cohort, stream, signal):
            observed = predictions.get(cohort)
            if not observed:
                pending.append({"code": "PREDICTION_RECEIPT_NOT_CAPTURED", "cohort": cohort, "stream": stream})
                return
            decision = observed["data"]["decisions"].get(stream, {})
            score = decision.get("prediction", {})
            clean = dict(signal)
            clean.pop("market_id", None)
            published = dict(decision.get("signal", {}))
            published.pop("market_id", None)
            check(decision.get("status") == "SIGNAL" and score.get("selected") is True and published == clean,
                  "SIGNAL_DIFFERS_FROM_PUBLISHED_PREDICTION", cohort=cohort, stream=stream)
            check(abs(Decimal(str(score["z_v4"]))) >= Decimal(str(score["threshold_frozen"])) and score["side"] == signal["outcome"],
                  "SELECTED_SCORE_MISMATCH", cohort=cohort, stream=stream)
            check(stamp(signal["feature_available_at"]) <= stamp(signal["decision_at"]) and stamp(signal["decision_at"]) >= activation,
                  "SIGNAL_NOT_FORWARD_AFTER_ACTIVATION", signal_id=signal["signal_id"])

        def book_check(signal, source, source_at, local_received, clock, venue, fee, at, expected_hash=None):
            prior_failures = len(failures)
            market, side = signal["market_id"], signal["outcome"]
            token = venue["outcome_ids"][side]
            check(source == f"https://clob.polymarket.com/book?token_id={token}", "NOT_OWN_SIDE_SOURCE", signal_id=signal["signal_id"])
            received_iso = stamp(local_received).isoformat()
            found = [dict(r) for r in con.execute("SELECT * FROM receipts WHERE url=? AND received_at IN (?,?) ORDER BY id",
                                                 (source, received_iso, received_iso.replace("+00:00", "Z")))]
            if not found:
                pending.append({"code": "RAW_BOOK_RECEIPT_NOT_CAPTURED", "source": source, "received_at_local": local_received})
                return None
            receipt = found[-1]
            raw = json.loads(receipt["raw_text"])
            check(receipt["http_status"] == 200 and digest(receipt["raw_text"]) == receipt["sha256"], "RAW_BOOK_INTEGRITY", receipt_id=receipt["id"])
            check(str(raw.get("asset_id")) == token and raw.get("market") == venue["condition_id"] and venue["market_id"] == market,
                  "RAW_BOOK_CONTRACT_IDENTITY", receipt_id=receipt["id"])
            provider = datetime.fromtimestamp(int(raw["timestamp"]) / 1000, timezone.utc)
            projected = stamp(local_received) + timedelta(seconds=float(Decimal(str(clock["server_minus_local_seconds"])) + Decimal(str(clock["uncertainty_seconds"]))))
            observed = stamp(source_at)
            check(provider == observed and stamp(signal["decision_at"]) <= observed <= projected <= at and 0 <= (at-observed).total_seconds() <= 2,
                  "BOOK_NOT_FRESH_OR_CAUSAL", receipt_id=receipt["id"])
            check(clock["status"] == "CALIBRATED" and 0 <= Decimal(str(clock["uncertainty_seconds"])) <= Decimal(".5")
                  and 0 <= (at-stamp(clock["observed_at"])).total_seconds() <= 60
                  and stamp(signal["decision_at"]) < at-timedelta(seconds=2*float(clock["uncertainty_seconds"])),
                  "CLOCK_EVIDENCE_INVALID", receipt_id=receipt["id"])
            # Mirror the public collector's numeric normalization, not its code.
            aggregated = defaultdict(Decimal)
            for level in raw["asks"]:
                price, size = Decimal(str(float(level["price"]))), Decimal(str(float(level["size"])))
                check(0 < price < 1 and size >= 0, "RAW_ASK_INVALID", receipt_id=receipt["id"])
                if size > 0:
                    aggregated[price] += size
            check(bool(aggregated), "EMPTY_BOOK_TREATED_AS_KNOWN", receipt_id=receipt["id"])
            normalized_hash = digest(canonical({"market": market, "outcome": side, "source": source, "observed_at": observed.isoformat(),
                                               "asks": [[str(p), str(q)] for p, q in sorted(aggregated.items())]}))
            if expected_hash:
                check(normalized_hash == expected_hash, "NORMALIZED_BOOK_HASH_MISMATCH", receipt_id=receipt["id"])
            check(fee["fee_model"] == "polymarket_crypto" and fee["market_id"] == market and fee["outcome"] == side
                  and fee["exponent"] == 1 and fee["taker_only"] is True and 0 <= Decimal(str(fee["rate"])) <= 1
                  and 0 <= (at-stamp(fee["observed_at"])).total_seconds() <= 90,
                  "FEE_BINDING_OR_FRESHNESS", receipt_id=receipt["id"])
            rate = Decimal(str(fee["rate"]))
            capacity = int((sum(q*(p+rate*p*(1-p)) for p,q in aggregated.items() if p <= Decimal(".515"))*MICRO).to_integral_value(rounding=ROUND_DOWN))
            markers = [r["id"] for r in observations if r["kind"] == "paper_book_receipt" and r["data"].get("market_id") == market
                       and r["data"].get("outcome") == side and 0 <= (stamp(r["saved_at"])-stamp(local_received)).total_seconds() <= 1]
            check(bool(markers), "EXECUTION_BOOK_SNAPSHOT_MISSING", receipt_id=receipt["id"])
            item = {"signal_id": signal["signal_id"], "stream": signal["stream"], "outcome": side, "market_id": market,
                    "receipt_sqlite_path": str(RUNTIME / "observations.sqlite3") + f"#receipts.id={receipt['id']}",
                    "paper_book_snapshot_ids": markers, "sha256": receipt["sha256"], "source": source,
                    "received_at_local": local_received, "received_at_calibrated_upper": projected.isoformat(),
                    "provider_source_at": source_at, "checked_at": at.isoformat(), "age_seconds": (at-observed).total_seconds(),
                    "best_positive_ask": float(min(aggregated)) if aggregated else None, "raw_eligible_capacity_usd": usd(capacity),
                    "normalized_book_sha256": normalized_hash, "clock_receipt_sha256": clock["receipt_sha256"],
                    "raw_receipt": receipt, "verified": len(failures) == prior_failures}
            book_evidence.append(item)
            return item

        # A chronological replay audits limits at execution time, even if all
        # positions have subsequently settled and current exposure is zero.
        cash, realized, open_orders = account["initial_atoms"], 0, {}
        max_risk, max_coin = 0, defaultdict(int)
        used = defaultdict(int)
        for event in events:
            payload = event["data"]
            if event["event_type"] == "PAPER_IOC":
                order_id = payload["order_id"]
                row = orders[order_id]
                cost = row["cost_atoms"] + row["fee_atoms"]
                check(cost == atoms(payload["cost_usd"]) + atoms(payload["fee_usd"]), "ORDER_SPEND_PAYLOAD_MISMATCH", order_id=order_id)
                check(order_id not in open_orders, "DUPLICATE_ORDER_EVENT", order_id=order_id)
                auth = payload["allocation_authorization"]
                key = auth["cohort_id"]
                cohort = cohorts[key]
                approved = atoms(auth["approved_stake_usd"])
                risk = sum(x["spent"] for x in open_orders.values())
                coin_risk = sum(x["spent"] for x in open_orders.values() if x["coin"] == row["underlying"])
                check(0 <= cost <= approved <= min(cash, 1_000_000_000-risk, 500_000_000-coin_risk, cohort["single_atoms"], 200_000_000, cohort["budget_atoms"]-used[key]),
                      "AT_EXECUTION_FUNDING_LIMIT", order_id=order_id)
                cash -= cost
                used[key] += cost
                if Decimal(row["quantity"]) > 0:
                    open_orders[order_id] = {"spent": cost, "coin": row["underlying"]}
                if order_id in old_orders:
                    continue
                at, boundary = stamp(row["fill_at"]), stamp(key)
                check(at >= activation and boundary >= first_cohort and boundary <= at < boundary+timedelta(seconds=20), "ORDER_OUTSIDE_ACTIVATION_WINDOW", order_id=order_id)
                check(auth.get("waterfall_semantics") == SEMANTICS and key in v5_cohorts and payload["live_enabled"] is False,
                      "ORDER_NOT_V5_PAPER", order_id=order_id)
                signal = signals[row["signal_id"]]
                stream = signal["stream"]
                tier = tiers[(key, stream)]
                prediction_signal(key, stream, signal)
                check(stamp(signal["window_start"]) == boundary and at <= stamp(signal["valid_until"]) and row["market_id"] == signal["market_id"] and row["outcome"] == signal["outcome"], "ORDER_SIGNAL_BINDING", order_id=order_id)
                check(approved == tier["approved_atoms"] and cost == tier["spent_atoms"] and atoms(auth["cohort_capacity_usd"]) == tier["capacity_atoms"], "TIER_APPROVAL_SPEND_MISMATCH", order_id=order_id)
                satisfied = Decimal(row["quantity"]) > 0 and cost+2 >= approved
                check((tier["state"] == "FILLED_BUDGET") == satisfied, "APPROVED_TARGET_NOT_ACTUALLY_FILLED", order_id=order_id)
                check(payload["fee_quote_sha256"] == digest(canonical(payload["fee_quote"])), "FEE_QUOTE_HASH_MISMATCH", order_id=order_id)
                check(all(0 < Decimal(fill["price"]) <= Decimal(".515") for fill in payload["fills"]), "FILL_PRICE_LIMIT", order_id=order_id)
                evidence = book_check(signal, payload["book_source"], payload["book_observed_at"], payload["book_received_at_local"],
                                      payload["clock_projection"], payload["venue_market"], payload["fee_quote"], at, payload["book_sha256"])
                if evidence:
                    check(atoms(evidence["raw_eligible_capacity_usd"]) == tier["capacity_atoms"], "RAW_CAPACITY_MISMATCH", order_id=order_id)
                    check(approved == min(tier["capacity_atoms"], cohort["single_atoms"], cohort["budget_atoms"]-used[key]+cost,
                                          cash+cost, 200_000_000, 1_000_000_000-risk, 500_000_000-coin_risk),
                          "APPROVED_AMOUNT_NOT_AVAILABLE_TARGET", order_id=order_id)
                    evidence.update(order_id=order_id, state=tier["state"], approved_usd=usd(approved), actual_spent_usd=usd(cost), quantity=row["quantity"],
                                    cost_usd=payload["cost_usd"], fee_usd=payload["fee_usd"], fill_at=payload["fill_at"], fills=payload["fills"],
                                    order_sqlite_path=str(RUNTIME / "account.sqlite3") + "#orders.order_id=" + order_id,
                                    order_payload=payload)
                policy = json.loads(cohort["policy_json"])
                prior = []
                for higher in policy["order"][:tier["rank"]-1]:
                    upper = tiers.get((key, higher))
                    check(upper is not None and upper["state"] in UNLOCKED, "LOWER_WITHOUT_UPPER_CREDIT", order_id=order_id, upper=higher)
                    if upper is None:
                        continue
                    receipt = json.loads(upper["receipt_json"])
                    upper_event = (event_by_skip.get((key, higher)) if upper["state"] in SKIPS else event_by_order.get(receipt.get("order_id")))
                    check(upper_event is not None and upper_event["seq"] < event["seq"], "UPPER_CREDIT_NOT_PRECEDING", order_id=order_id, upper=higher)
                    prior.append({"stream": higher, "state": upper["state"]})
                sequence_evidence.append({"cohort": key, "stream": stream, "state": tier["state"], "order_id": order_id,
                                          "approved_usd": usd(approved), "spent_usd": usd(cost), "raw_capacity_usd": usd(tier["capacity_atoms"]),
                                          "higher_streams": prior, "own_side_raw_receipt_verified": bool(evidence and evidence["verified"])})
            elif event["event_type"] == "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER":
                resolution = payload["resolution"]
                check(resolution["kind"] == "OBSERVED_OFFICIAL_RESOLUTION" and resolution["source"].startswith("https://gamma-api.polymarket.com/markets/"), "NON_OFFICIAL_SETTLEMENT", seq=event["seq"])
                for settled in payload["orders"]:
                    item = open_orders.pop(settled["order_id"], None)
                    check(item is not None, "SETTLEMENT_WITHOUT_OPEN_ORDER", order_id=settled["order_id"])
                    payout, pnl = atoms(settled["payout_usd"]), atoms(settled["realized_pnl_usd"])
                    if item:
                        check(pnl == payout-item["spent"], "SETTLEMENT_CASHFLOW_MISMATCH", order_id=settled["order_id"])
                    cash += payout
                    realized += pnl
            risk = sum(x["spent"] for x in open_orders.values())
            max_risk = max(max_risk, risk)
            for coin in ("BTC", "ETH", "SOL", "BNB"):
                amount = sum(x["spent"] for x in open_orders.values() if x["coin"] == coin)
                max_coin[coin] = max(max_coin[coin], amount)
                check(amount <= 500_000_000, "HISTORICAL_COIN_RISK_LIMIT", seq=event["seq"], coin=coin)
            check(cash >= 0 and risk <= 1_000_000_000 and cash+risk == account["initial_atoms"]+realized,
                  "HISTORICAL_BALANCE_OR_RISK", seq=event["seq"])
        check(cash == account["cash_atoms"] and realized == account["realized_atoms"], "REPLAY_CURRENT_BALANCE_MISMATCH")
        check(set(open_orders) == {r["order_id"] for r in current["orders"] if r["status"] == "OPEN"}, "REPLAY_OPEN_POSITIONS_MISMATCH")
        for key, cohort in v5_cohorts.items():
            boundary = stamp(key)
            check(boundary >= first_cohort and stamp(cohort["created_at"]) >= activation, "COHORT_PREDATES_ACTIVATION", cohort=key)
            check(cohort["used_atoms"] == used[key] <= cohort["budget_atoms"] <= min(int(Decimal(cohort["equity_basis_atoms"])*Decimal(".022")), 1_200_000_000)
                  and cohort["single_atoms"] <= min(200_000_000, int(Decimal(cohort["equity_basis_atoms"])*Decimal(".015"))), "COHORT_BUDGET_MISMATCH", cohort=key)
            if key not in executions:
                pending.append({"code": "EXECUTION_SNAPSHOT_NOT_CAPTURED", "cohort": key})
            for tier in (t for t in current["cohort_streams"] if t["cohort_id"] == key and t["state"] in SKIPS):
                stream, state = tier["stream"], tier["state"]
                event = event_by_skip.get((key, stream))
                check(event is not None, "SKIP_EVENT_MISSING", cohort=key, stream=stream)
                if event is None:
                    continue
                receipt = json.loads(tier["receipt_json"])
                at = stamp(event["utc"])
                policy_order = json.loads(cohort["policy_json"])["order"]
                check(tier["rank"] == policy_order.index(stream)+1 and event["data"]["state"] == state,
                      "SKIP_RANK_OR_STATE_MISMATCH", cohort=key, stream=stream)
                for upper_stream in policy_order[:tier["rank"]-1]:
                    upper = tiers.get((key, upper_stream))
                    upper_receipt = json.loads(upper["receipt_json"]) if upper else {}
                    upper_event = (event_by_skip.get((key, upper_stream)) if upper and upper["state"] in SKIPS
                                   else event_by_order.get(upper_receipt.get("order_id")))
                    check(upper is not None and upper["state"] in UNLOCKED and upper_event is not None and upper_event["seq"] < event["seq"],
                          "LOWER_SKIP_WITHOUT_PRECEDING_CREDIT", cohort=key, stream=stream, upper=upper_stream)
                check(activation <= at and boundary <= at < boundary+timedelta(seconds=20) and tier["spent_atoms"] == tier["approved_atoms"] == 0,
                      "SKIP_TIMING_OR_SPEND", cohort=key, stream=stream)
                check(event["data"]["receipt"] == receipt and event["data"]["receipt_sha256"] == digest(canonical(receipt)), "SKIP_RECEIPT_MISMATCH", cohort=key, stream=stream)
                if state in {"NO_SIGNAL", "NOT_DUE"}:
                    check("orderbook" not in receipt, "ABSENT_SIGNAL_CONSUMED_BOOK", cohort=key, stream=stream)
                    stream_result = executions.get(key, {}).get("data", {}).get("result", {}).get("stream_results", {}).get(stream, {})
                    check(stream_result.get("book_checked") is not True, "ABSENT_SIGNAL_MARKED_BOOK_CHECKED", cohort=key, stream=stream)
                    decision = predictions.get(key, {}).get("data", {}).get("decisions", {}).get(stream, {})
                    check(decision.get("status") == state, "SKIP_DIFFERS_FROM_DECISION", cohort=key, stream=stream)
                    if state == "NO_SIGNAL":
                        check(receipt.get("prediction") == decision.get("prediction") and decision.get("prediction", {}).get("selected") is False,
                              "NO_SIGNAL_SCORE_MISMATCH", cohort=key, stream=stream)
                    else:
                        check(stream.endswith("15m") and int(boundary.timestamp()) % 900 != 0, "FALSE_NOT_DUE", cohort=key, stream=stream)
                else:
                    signal, book = receipt["signal"], receipt["orderbook"]
                    prediction_signal(key, stream, signal)
                    evidence = book_check(signal, book["source"], book["observed_at"], book["received_at_local"], book["clock_projection"], book["venue_market"], receipt["fee_quote"], at)
                    if evidence:
                        check(evidence["best_positive_ask"] > .515 and evidence["raw_eligible_capacity_usd"] == 0, "PRICE_SKIP_NOT_PROVEN", cohort=key, stream=stream)
                        evidence.update(state=state, approved_usd=0, actual_spent_usd=0, quantity="0")
                        higher = [{"stream": s, "state": tiers[(key,s)]["state"]} for s in json.loads(cohort["policy_json"])["order"][:tier["rank"]-1] if (key,s) in tiers]
                        sequence_evidence.append({"cohort": key, "stream": stream, "state": state, "approved_usd": 0, "spent_usd": 0,
                                                  "higher_streams": higher, "own_side_raw_receipt_verified": evidence["verified"]})
        lower_checked = [r for r in sequence_evidence if r["own_side_raw_receipt_verified"] and any(p["state"] in SKIPS for p in r["higher_streams"])]
        lower_filled = [r for r in lower_checked if r["state"] == "FILLED_BUDGET"]
        absent_then_lower = [r for r in lower_checked if any(p["state"] in {"NO_SIGNAL", "NOT_DUE"} for p in r["higher_streams"])]
        if not v5_cohorts:
            pending.append({"code": "NO_POST_ACTIVATION_COHORT_OBSERVED"})
        if not lower_checked:
            pending.append({"code": "NO_LOWER_SIGNAL_WITH_OWN_FRESH_BOOK_OBSERVED_AFTER_SKIP"})
        if not absent_then_lower:
            pending.append({"code": "NO_NO_SIGNAL_OR_NOT_DUE_THEN_LOWER_OWN_FRESH_BOOK_OBSERVED"})
        report = {"status": "FAIL" if failures else ("WAITING_EVIDENCE" if pending else "PASS_OBSERVED"),
                  "scope": "READ_ONLY_PAPER_ACCOUNT_AUDIT_NOT_STRATEGY_EFFECTIVENESS",
                  "activation_utc": activation.isoformat(), "first_allowed_cohort_utc": first_cohort.isoformat(),
                  "audited_at_local_utc": datetime.now(timezone.utc).isoformat(), "live_enabled": False,
                  "baseline": {"path": str(baseline_path), "sha256": baseline_sha, "counts": baseline_counts},
                  "snapshot": {"last_event_seq": events[-1]["seq"] if events else None, "last_event_hash": previous,
                               "last_observation_id": max((r["id"] for r in observations), default=None)},
                  "counts": {"new_orders": len(new_orders), "new_positive_quantity_orders": sum(Decimal(r["quantity"]) > 0 for r in new_orders),
                             "v5_cohorts": len(v5_cohorts), "skip_then_lower_own_side_checked": len(lower_checked), "skip_then_lower_filled_budget": len(lower_filled),
                             "no_signal_or_not_due_then_lower_own_side_checked": len(absent_then_lower)},
                  "funds": {"initial_usd": usd(account["initial_atoms"]), "cash_usd": usd(cash), "locked_cost_usd": usd(sum(x["spent"] for x in open_orders.values())),
                            "realized_pnl_usd": usd(realized), "max_historical_open_risk_usd": usd(max_risk), "max_historical_coin_risk_usd": {k:usd(v) for k,v in max_coin.items()}},
                  "failures": failures, "pending_evidence": pending, "sequences": sequence_evidence, "book_evidence": book_evidence,
                  "execution_results": [{"cohort": key, "snapshot_id": executions[key]["id"],
                                         "status": executions[key]["data"]["result"].get("status"),
                                         "blocked": executions[key]["data"]["result"].get("blocked"),
                                         "stream_results": executions[key]["data"]["result"].get("stream_results")}
                                        for key in v5_cohorts if key in executions],
                  "limitations": ["SQLite paths and receipt ids bind public evidence; local timestamps retain their calibrated projection.",
                                  "Known price failure proves checking only and is never counted as a fill.",
                                  "Account and observation databases are separately consistent snapshots; in-flight receipt publication may require a later read-only audit."]}
        return report
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-utc", required=True)
    parser.add_argument("--first-cohort-utc", required=True)
    args = parser.parse_args()
    activation, first = stamp(args.activation_utc), stamp(args.first_cohort_utc)
    if first.timestamp() % 300 or first <= activation:
        parser.error("first cohort must be a complete UTC five-minute boundary strictly after activation")
    try:
        report = audit(activation, first)
    except Exception as exc:
        report = {"status": "AUDIT_ERROR", "activation_utc": activation.isoformat(), "first_allowed_cohort_utc": first.isoformat(),
                  "counts": {}, "funds": {}, "failures": [{"code": "AUDIT_INPUT_OR_SCHEMA_ERROR", "exception_type": type(exc).__name__}],
                  "pending_evidence": [], "live_enabled": False}
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    evidence = HERE / f"activation_audit_{suffix}.json"
    evidence.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    compact = {key: report[key] for key in ("status", "activation_utc", "first_allowed_cohort_utc", "counts", "funds", "failures", "pending_evidence")}
    compact["evidence_path"] = str(evidence)
    compact["evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    print(canonical(compact))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
