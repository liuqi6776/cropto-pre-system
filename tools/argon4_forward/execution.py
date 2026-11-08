"""Forward-only paper execution; only optional public GETs for final resolution.

No exchange order endpoint, signing, wallet, or trading credential is present.
One persisted UTC five-minute cohort owns its budget and all upper-rank credit.
Depth is an IOC simulation constraint, never a claim that maker orders fill.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import re
import time

from .account import (Account, AccountRejected, ALL_STREAMS, STREAMS, V5_SEMANTICS, V5_UNLOCKED,
                      _d, _iso, _json, _now, _safe, _ts, multi_look_settings)


def _stamp(now):
    return _now(now() if callable(now) else now)


def _array(value):
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise AccountRejected("array required")
    return parsed


def adapt_official_resolution(raw, receipt, expected_market, now=None):
    """Require protocol-specific final status AND exact binary payouts.

    Official schema: docs.polymarket.com/api-reference/markets/get-market-by-id
    and Polymarket/py-sdk MarketResolutionStatus / UmaResolutionStatus. A closed
    market or a price close to one alone is never treated as a final result.
    """
    stamp = _stamp(now)
    market_id = str(expected_market["market_id"])
    source = f"https://gamma-api.polymarket.com/markets/{market_id}"
    if (not re.fullmatch(r"[0-9]+", market_id) or receipt.get("url") != source
            or receipt.get("method") != "GET" or receipt.get("http_status") != 200
            or receipt.get("body_complete") is not True):
        raise AccountRejected("official complete resolution GET receipt required")
    raw_text = receipt.get("raw_text")
    if (not isinstance(raw_text, str) or hashlib.sha256(raw_text.encode()).hexdigest() != receipt.get("sha256")
            or json.loads(raw_text) != raw):
        raise AccountRejected("resolution receipt integrity mismatch")
    received = _ts(receipt.get("received_at"))
    if received > stamp or received < _ts(expected_market["window_end"]):
        raise AccountRejected("resolution observation time invalid")
    if (str(raw.get("id")) != market_id or raw.get("conditionId") != expected_market["condition_id"]
            or raw.get("version") != expected_market["protocol"]
            or _ts(raw.get("eventStartTime")) != _ts(expected_market["window_start"])
            or _ts(raw.get("endDate")) != _ts(expected_market["window_end"])
            or raw.get("resolutionSource") != expected_market["resolution_source"]):
        raise AccountRejected("resolution contract identity mismatch")
    if raw["version"] == "v2":
        if raw.get("resolutionStatus") != "resolved":
            raise AccountRejected("WAITING_OFFICIAL_FINAL_RESOLUTION")
        id_field = "positionIds"
    elif raw["version"] == "v1":
        if raw.get("umaResolutionStatus") not in ("resolved", "settled"):
            raise AccountRejected("WAITING_OFFICIAL_FINAL_RESOLUTION")
        id_field = "clobTokenIds"
    else:
        raise AccountRejected("unsupported resolution protocol")
    labels = _array(raw.get("outcomes"))
    ids = _array(raw.get(id_field))
    prices = [_d(x) for x in _array(raw.get("outcomePrices"))]
    if (len(labels) != 2 or set(labels) != {"Up", "Down"} or len(ids) != 2 or len(prices) != 2
            or {label.upper(): str(token) for label,token in zip(labels,ids)} != expected_market["outcome_ids"]
            or sorted(prices) != [Decimal(0), Decimal(1)]):
        raise AccountRejected("final binary payout or outcome mapping unavailable")
    winner = labels[prices.index(Decimal(1))].upper()
    stable_id = hashlib.sha256((market_id+expected_market["condition_id"]+winner).encode()).hexdigest()
    return {"kind": "OBSERVED_OFFICIAL_RESOLUTION", "market_id": market_id, "outcome": winner,
            "resolution_id": stable_id, "source": source, "source_sha256": receipt["sha256"],
            "resolved_at": _iso(received), "received_at": _iso(received),
            "resolved_at_basis": "FIRST_OBSERVED_FINAL_STATE_NOT_ORACLE_EVENT_TIME",
            "protocol": raw["version"], "final_status": "resolved"}


def fetch_public_resolution(market_id):
    """Bounded unauthenticated GET to the one allowlisted Gamma resource."""
    import requests
    if not re.fullmatch(r"[0-9]+", str(market_id)):
        raise AccountRejected("invalid market identifier")
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    receipt = {"url": url, "method": "GET", "requested_at": _iso(_now()), "body_complete": False}
    with requests.Session() as session:
        session.trust_env = False
        with session.get(url, timeout=(2,3), allow_redirects=False, stream=True,
                         headers={"Accept":"application/json", "User-Agent":"argon4-paper-resolution/1.0"}) as response:
            body = bytearray()
            deadline = time.monotonic()+6
            for part in response.iter_content(32768):
                body.extend(part)
                if len(body)>2_000_000 or time.monotonic()>deadline:
                    raise AccountRejected("resolution response limit")
            receipt.update(http_status=response.status_code, received_at=_iso(_now()), body_complete=True,
                           raw_text=body.decode("utf-8"), sha256=hashlib.sha256(body).hexdigest())
            if response.status_code != 200:
                raise AccountRejected("resolution HTTP unavailable")
    return json.loads(receipt["raw_text"]), receipt


class PaperExecutor:
    def __init__(self, account: Account, policy: dict):
        self.account, self.policy = account, dict(policy)

    @staticmethod
    def _clock(clock, stamp):
        borderline_completed = (isinstance(clock, dict)
                                and isinstance(clock.get("borderline_retry"), dict)
                                and clock["borderline_retry"].get("status") == "COMPLETED"
                                and len(clock.get("receipts", ())) == 8)
        max_uncertainty = Decimal(".6") if borderline_completed else Decimal(".5")
        if (not isinstance(clock, dict) or clock.get("status") != "CALIBRATED"
                or clock.get("source") != "POLYMARKET_CLOB_SERVER_TIME"
                or not 0 <= _d(clock.get("uncertainty_seconds")) <= max_uncertainty
                or not 0 <= (stamp-_ts(clock.get("observed_at"))).total_seconds() <= 60):
            raise AccountRejected("CLOCK_NOT_CALIBRATED")

    @staticmethod
    def _received_upper(local_time, clock):
        local = _ts(local_time)
        if "server_minus_local_seconds" not in clock:
            return local
        digest = clock.get("receipt_sha256",clock.get("calibration_receipt_sha256"))
        if not re.fullmatch(r"[0-9a-fA-F]{64}",str(digest)):
            raise AccountRejected("clock projection receipt hash required")
        shift = _d(clock["server_minus_local_seconds"])+_d(clock["uncertainty_seconds"])
        return local+timedelta(seconds=float(shift))

    @staticmethod
    def _market(markets, stream, start):
        states = markets.get("streams", markets)
        state = states.get(stream)
        if not isinstance(state, dict):
            raise AccountRejected("market is missing")
        choices = [state] if state.get("status") == "VERIFIED_MARKET" else [state.get("current"), state.get("next")]
        for market in choices:
            if (isinstance(market, dict) and market.get("status") == "VERIFIED_MARKET"
                    and _ts(market.get("window_start")) == start):
                return market
        raise AccountRejected("verified contract window is missing")

    def _stream_limit_price(self, decision=None):
        cap = Decimal(".515")
        exec_cfg = self.policy.get("execution")
        if isinstance(exec_cfg, dict) and exec_cfg.get("high_confidence_max_buy_price") is not None:
            hi = _d(exec_cfg["high_confidence_max_buy_price"])
            if not Decimal(".515") <= hi <= Decimal(".520"):
                raise AccountRejected("unsupported high-confidence buy price limit")
            if isinstance(decision, dict) and (
                    decision.get("high_confidence") is True
                    or (decision.get("limit_price") is not None and _d(decision["limit_price"]) == hi)
                    or (isinstance(decision.get("prediction"), dict) and decision["prediction"].get("high_confidence_q97") is True)):
                cap = hi
        return cap

    @staticmethod
    def _adapt(market, side, book, stamp, clock, allow_price_skip=False, price_limit=Decimal(".515")):
        if (market.get("status") != "VERIFIED_MARKET" or market.get("active") is not True
                or market.get("closed") is not False or market.get("accepting_orders") is not True
                or not _ts(market["window_start"]) <= stamp < _ts(market["window_end"])):
            raise AccountRejected("market is not an active current contract")
        meta_local = market.get("metadata_received_at_local",market.get("metadata_received_at"))
        meta_at = PaperExecutor._received_upper(meta_local,clock)
        if not 0 <= (stamp-meta_at).total_seconds() <= 90:
            raise AccountRejected("market metadata is stale or future")
        token = market["outcome_ids"].get(side)
        if (not isinstance(book, dict) or book.get("status") not in ("OBSERVED", "EMPTY_OR_ONE_SIDED")
                or book.get("asset_id") != token or book.get("condition_id") != market["condition_id"]):
            raise AccountRejected("observed outcome book identity mismatch")
        received_local = book.get("received_at_local",book.get("received_at"))
        source, received = _ts(book.get("source_at")), PaperExecutor._received_upper(received_local,clock)
        if not source <= received <= stamp or not 0 <= (stamp-source).total_seconds() <= 2:
            raise AccountRejected("book is future or older than two seconds")
        schedule = market.get("fee", {})
        if schedule.get("status") == "KNOWN_DISABLED" and schedule.get("fees_enabled") is False:
            rate = 0
        elif (schedule.get("status") == "KNOWN_SCHEDULE" and schedule.get("fees_enabled") is True
              and schedule.get("exponent") == 1 and schedule.get("taker_only") is True):
            rate = schedule.get("rate")
        else:
            raise AccountRejected("supported official fee schedule is unavailable")
        fee = {"fee_model":"polymarket_crypto", "rate":rate, "exponent":1, "taker_only":True,
               "market_id":market["market_id"], "outcome":side,
               "source":f"https://gamma-api.polymarket.com/markets/{market['market_id']}",
               "source_sha256":market.get("metadata_sha256"), "observed_at":_iso(meta_at),
               "observed_at_local":_iso(_ts(meta_local))}
        # A bad schedule cannot become a known price failure merely because no
        # ask is inside the limit; validate it independently of executable depth.
        Account._fee_per_share(fee,price_limit,stamp,market["market_id"],side)
        contract = {k:market[k] for k in ("market_id", "condition_id", "protocol", "asset", "window_start", "window_end", "outcome_ids")}
        contract["resolution_source"] = market["resolution"]["source"]
        adapted = {"kind":"OBSERVED_PUBLIC_ORDERBOOK", "source":f"https://clob.polymarket.com/book?token_id={token}",
                   "market_id":market["market_id"], "underlying":market["asset"], "outcome":side,
                   "window_start":market["window_start"], "window_end":market["window_end"],
                   "observed_at":_iso(source), "received_at":_iso(received), "asks":book["asks"],
                   "received_at_local":_iso(_ts(received_local)), "clock_projection":dict(clock),
                   "venue_market":contract}
        if not isinstance(book.get("asks"),list):
            raise AccountRejected("observed ask levels required")
        capacity, positive_prices = Decimal(0), []
        for ask in book["asks"]:
            price, quantity = _d(ask["price"]), _d(ask["size"])
            if not 0 < price < 1 or quantity < 0:
                raise AccountRejected("invalid ask level")
            if quantity > 0:
                positive_prices.append(price)
            if price <= price_limit:
                capacity += quantity * (price + Account._fee_per_share(fee,price,stamp,market["market_id"],side))
        if capacity <= 0:
            if allow_price_skip and positive_prices and min(positive_prices) > price_limit:
                return adapted, fee, Decimal(0)
            raise AccountRejected("zero or unknown capacity cannot unlock a lower rank")
        capacity = capacity.quantize(Decimal(".000001"), rounding=ROUND_DOWN)
        if capacity <= 0:
            raise AccountRejected("observed capacity is below the accounting unit")
        return adapted, fee, capacity

    def _paper_acceptance_clock(self, now, clock, book, fee, signal):
        """Preserve execution freshness while the account waits or computes."""
        def current():
            stamp = _stamp(now)
            self._clock(clock, stamp)
            if not 0 <= (stamp-_ts(book["observed_at"])).total_seconds() <= 2:
                raise AccountRejected("book is future or older than two seconds")
            if not 0 <= (stamp-_ts(fee["observed_at"])).total_seconds() <= 90:
                raise AccountRejected("market metadata is stale or future")
            if not _ts(signal["decision_at"]) < stamp-timedelta(seconds=2*float(_d(clock["uncertainty_seconds"]))):
                raise AccountRejected("fill clock lower bound has not passed the decision")
            return stamp
        return current

    def process_cohort(self, cohort_id, decisions, markets, now=None, book_refresh=None, clock=None, market_refresh=None):
        """Process ranked decisions once per persisted cohort; no history replay.

        book_refresh(market, outcome) may return a newly validated public book.
        Under v5, market_refresh(stream, boundary, existing_or_none) may refresh
        one reached SIGNAL's missing/stale metadata; it never refreshes skips.
        A callable ``now`` is supported for a conservatively calibrated UTC clock.
        Cached 60-second observations cannot be promoted to fresh execution books.
        """
        if self.policy.get("waterfall_semantics") == V5_SEMANTICS:
            return self._process_eligible_cohort(cohort_id,decisions,markets,now,book_refresh,clock,market_refresh)
        result = {"fills":[], "blocked":[], "cohort":None, "live_enabled":False,
                  "execution":"OBSERVED_BOOK_IOC_PAPER_ONLY"}
        try:
            stamp = _stamp(now)
            self._clock(clock,stamp)
            cohort = self.account.ensure_cohort(cohort_id,self.policy,now=stamp)
            result["cohort"] = cohort
            if cohort["blocked_reason"]:
                raise AccountRejected(cohort["blocked_reason"])
            boundary = _ts(cohort["cohort_id"])
            for rank, stream in enumerate(cohort["policy"]["order"],1):
                stamp = _stamp(now)
                self._clock(clock,stamp)
                cohort = self.account.cohort_state(cohort["cohort_id"])
                previous = next((s for s in cohort["streams"] if s["stream"]==stream),None)
                if previous:
                    if previous["state"] in ("FILLED_CAPACITY","NO_SIGNAL","NOT_DUE"):
                        continue
                    raise AccountRejected("upper stream remains unfilled or capacity locked")
                decision = decisions.get(stream)
                if not isinstance(decision,dict) or decision.get("status") not in ("SIGNAL","NO_SIGNAL","NOT_DUE"):
                    raise AccountRejected("higher-ranked decision unavailable")
                due = int(boundary.timestamp()) % ALL_STREAMS[stream][1] == 0
                if (decision["status"]=="NOT_DUE") == due:
                    raise AccountRejected("decision due-state does not match UTC boundary")
                start = datetime.fromtimestamp(int(boundary.timestamp())//ALL_STREAMS[stream][1]*ALL_STREAMS[stream][1],timezone.utc)
                market = self._market(markets,stream,start)
                if market.get("asset") != ALL_STREAMS[stream][0] or market.get("duration_seconds") != ALL_STREAMS[stream][1]:
                    raise AccountRejected("market underlying or horizon mismatch")
                sides = [decision["signal"]["outcome"]] if decision["status"]=="SIGNAL" else ["UP","DOWN"]
                adapted = []
                for side in sides:
                    book = book_refresh(market,side) if book_refresh else market.get("books",{}).get(side)
                    stamp = _stamp(now)
                    self._clock(clock,stamp)
                    adapted.append(self._adapt(market,side,book,stamp,clock))
                capacity = max(item[2] for item in adapted)
                if decision["status"] != "SIGNAL":
                    receipt = {"source":"OBSERVED_BOTH_OUTCOME_ASK_CAPACITY", "sha256":hashlib.sha256(_json([[b,f,str(c)] for b,f,c in adapted]).encode()).hexdigest(),
                               "observed_at":min(item[0]["observed_at"] for item in adapted),
                               "market_id":market["market_id"], "decision_status":decision["status"]}
                    self.account.record_cohort_gate(cohort["cohort_id"],stream,decision["status"],capacity,receipt,now=stamp)
                    continue
                signal = dict(decision["signal"])
                if signal.get("market_id") not in (None,market["market_id"]):
                    raise AccountRejected("signal market conflicts with official contract")
                signal["market_id"] = market["market_id"]
                if signal.get("stream") != stream or _ts(signal["window_start"]) != boundary:
                    raise AccountRejected("signal cohort mismatch")
                # ``stamp`` may be the server-time upper bound. Require the
                # lower bound to be after decision availability before recording
                # a paper fill, rather than relying on a possibly future bound.
                if not _ts(signal["decision_at"]) < stamp-timedelta(seconds=2*float(_d(clock["uncertainty_seconds"]))):
                    raise AccountRejected("fill clock lower bound has not passed the decision")
                summary = self.account.overview(now=stamp)
                coin_cost = sum(p["cost_usd"]+p["fee_usd"] for p in summary["positions"] if p["underlying"]==signal["underlying"])
                amount = min(capacity,_d(cohort["single_usd"]),_d(cohort["remaining_usd"]),_d(summary["cash"]),
                             Decimal(1000)-_d(summary["locked_cost"]),Decimal(500)-_d(coin_cost))
                if amount <= 0:
                    raise AccountRejected("persistent budget or existing position limit exhausted")
                amount = amount.quantize(Decimal(".000001"),rounding=ROUND_DOWN)
                preceding = [{"signal_id":s["signal_id"],"rank":s["rank"],"required_cost_usd":s["approved_usd"]}
                             for s in cohort["streams"] if s["state"]=="FILLED_CAPACITY"]
                authorization = {"policy":"RANKED_CAPACITY_WATERFALL", "plan_id":cohort["plan_id"],"rank":rank,
                    "signal_id":signal["signal_id"],"market_id":market["market_id"],"approved_stake_usd":float(amount),
                    "planned_at":_iso(stamp),"valid_until":_iso(min(_ts(signal["valid_until"]),stamp+timedelta(seconds=2),_ts(cohort["expires_at"]))),
                    "policy_version":cohort["policy_version"],"preceding_allocations":preceding,
                    "cohort_id":cohort["cohort_id"],"cohort_capacity_usd":float(capacity)}
                book,fee,_ = adapted[0]
                fill = self.account.place_paper_buy(signal,book,fee,now=self._paper_acceptance_clock(now,clock,book,fee,signal),
                                                    stake=amount,allocation_authorization=authorization)
                result["fills"].append(fill)
            result["status"] = "PROCESSED"
        except (AccountRejected, KeyError, TypeError, ValueError) as exc:
            # AccountRejected messages are fixed strings, never raw transport text.
            result["blocked"].append(str(exc) if isinstance(exc,AccountRejected) else type(exc).__name__)
            result["status"] = "PARTIAL_BLOCKED" if result["fills"] else "BLOCKED"
        except Exception as exc:
            result["blocked"].append(type(exc).__name__)
            result["status"] = "PARTIAL_BLOCKED" if result["fills"] else "BLOCKED"
        if result["cohort"]:
            result["cohort"] = self.account.cohort_state(result["cohort"]["cohort_id"])
        return result

    @staticmethod
    def _precheck_market_refresh(signal, stream, boundary, stamp, summary, cohort):
        """Check eligibility before optional I/O, without inventing a market ID.

        This is an extra precondition, never a replacement for the account's
        full signal validation after a real market identifier has been obtained.
        """
        required = ("signal_id", "model_id", "model_sha256", "feature_available_at",
                    "decision_at", "valid_until", "window_start", "window_end")
        if any(signal.get(key) in (None, "") for key in required):
            raise AccountRejected("incomplete signal before market refresh")
        asset, duration = ALL_STREAMS[stream]
        if (signal.get("validation_status") != "INPUTS_VALIDATED" or signal.get("stream") != stream
                or signal.get("underlying") != asset or signal.get("outcome") not in ("UP", "DOWN")
                or not re.fullmatch(r"[0-9a-fA-F]{64}", str(signal["model_sha256"]))):
            raise AccountRejected("invalid signal before market refresh")
        available, decision, expiry, start, end = (_ts(signal[key]) for key in
            ("feature_available_at", "decision_at", "valid_until", "window_start", "window_end"))
        if (start != boundary or end != boundary + timedelta(seconds=duration)
                or not _ts(summary["started_at"]) <= decision <= stamp
                or not available <= decision or not start <= decision < end
                or not decision < expiry <= end or not stamp < expiry
                or (stamp-decision).total_seconds() > 20):
            raise AccountRejected("invalid signal timing before market refresh")
        coin_cost = sum(_d(p["cost_usd"])+_d(p["fee_usd"]) for p in summary["positions"]
                        if p["underlying"] == asset)
        budget = min(_d(cohort["single_usd"]), _d(cohort["remaining_usd"]), _d(summary["cash"]),
                     Decimal(200), Decimal(1000)-_d(summary["locked_cost"]), Decimal(500)-coin_cost)
        if budget < Decimal(".000001"):
            raise AccountRejected("persistent budget or existing position limit exhausted")

    def _process_eligible_cohort(self, cohort_id, decisions, markets, now, book_refresh, clock, market_refresh=None):
        """v5: skip verified ineligible streams, then fill each approved budget.

        Raw observed depth remains evidence, not the amount a higher rank must
        consume. Only actual paid cost satisfies its approved target. Unknown
        inputs, partial IOC fills and previous partial states block lower ranks.
        """
        from .stage import stage_profile, disabled_decision, DISABLED
        stream_results = {s:{"status":"UNREACHED"} for s in self.policy.get("order",[]) if s in ALL_STREAMS}
        result = {"fills":[],"blocked":[],"cohort":None,"live_enabled":False,
                  "execution":"OBSERVED_BOOK_IOC_PAPER_ONLY","waterfall_semantics":V5_SEMANTICS,
                  "stream_results":stream_results}
        current, look_claim = None, None
        try:
            multi_look = multi_look_settings(self.policy)
            cascade_enabled = (isinstance(self.policy.get("execution"), dict)
                               and self.policy["execution"].get("partial_fill_remainder_cascade") is True)
            profile = stage_profile(self.policy, cohort_id)
            result["stage"] = profile
            for stream in profile["disabled_streams"]:
                stream_results[stream] = disabled_decision(profile)
            stamp = _stamp(now)
            self._clock(clock,stamp)
            cohort = self.account.ensure_cohort(cohort_id,self.policy,now=stamp)
            result["cohort"] = cohort
            for stream in profile["disabled_streams"]:
                previous = next((s for s in cohort["streams"] if s["stream"] == stream), None)
                if previous is None or previous["state"] != DISABLED or any(
                        previous[k] != 0 for k in ("approved_usd", "spent_usd", "capacity_usd")):
                    raise AccountRejected("disabled stage stream must have a persisted zero budget")
                stream_results[stream]["persisted"] = True
            if cohort["blocked_reason"]:
                raise AccountRejected(cohort["blocked_reason"])
            boundary = _ts(cohort["cohort_id"])
            for rank,stream in enumerate(cohort["policy"]["order"],1):
                if stream in profile["disabled_streams"]:
                    continue
                current, stamp = stream, _stamp(now)
                self._clock(clock,stamp)
                if not boundary <= stamp < _ts(cohort["expires_at"]):
                    raise AccountRejected("decision cohort expired")
                cohort = self.account.cohort_state(cohort["cohort_id"])
                previous = next((s for s in cohort["streams"] if s["stream"]==stream),None)
                if previous:
                    state = previous["state"]
                    stream_results[stream] = {"status":state if state in V5_UNLOCKED else "PARTIAL",
                        "persisted":True,"approved_usd":previous["approved_usd"],"spent_usd":previous["spent_usd"]}
                    if state not in ("NO_SIGNAL","NOT_DUE"):
                        stream_results[stream]["raw_capacity_usd"] = previous["capacity_usd"]
                    if state in V5_UNLOCKED or (cascade_enabled and state == "PARTIAL"
                                                and _d(previous["spent_usd"]) > 0
                                                and (_ts(cohort["expires_at"]) - stamp).total_seconds() >= 4):
                        continue
                    raise AccountRejected("upper stream has an unsatisfied approved budget")
                decision = decisions.get(stream)
                if not isinstance(decision,dict) or decision.get("status") not in ("SIGNAL","NO_SIGNAL","NOT_DUE"):
                    raise AccountRejected("higher-ranked decision unavailable")
                _safe(decision)
                due = int(boundary.timestamp()) % ALL_STREAMS[stream][1] == 0
                if (decision["status"]=="NOT_DUE") == due:
                    raise AccountRejected("decision due-state does not match UTC boundary")
                if decision["status"] != "SIGNAL":
                    if decision.get("signal") is not None or decision.get("error"):
                        raise AccountRejected("conflicting absent-signal decision")
                    receipt = ({"source":"UTC_COHORT_SCHEDULE","cohort_id":cohort["cohort_id"]}
                               if decision["status"]=="NOT_DUE" else
                               {"source":"VALIDATED_MODEL_DECISION","prediction":decision.get("prediction")})
                    self.account.record_cohort_skip(cohort["cohort_id"],stream,decision["status"],receipt,now=stamp)
                    stream_results[stream] = {"status":decision["status"],"approved_usd":0.,"spent_usd":0.,
                                              "book_checked":False,"persisted":True}
                    continue
                if multi_look is not None:
                    old_look = self.account.book_look_state(cohort["cohort_id"], stream)
                    if old_look is not None:
                        if old_look["state"] != "PRICE_LIMIT_NOT_MET" or not old_look["next_retry_at"]:
                            raise AccountRejected("previous book look is not retryable")
                        if stamp > boundary + timedelta(seconds=18.5):
                            look_claim = (stream, old_look["attempt"])
                            raise AccountRejected("price recheck start deadline expired")
                        if stamp < _ts(old_look["next_retry_at"]):
                            stream_results[stream] = {"status":"WAITING_PRICE_RECHECK","persisted":True,
                                "book_checked":False,"attempt":old_look["attempt"],
                                "next_retry_at":old_look["next_retry_at"],"approved_usd":0.,"spent_usd":0.}
                            result.update(status="WAITING_PRICE_RECHECK", next_retry_at=old_look["next_retry_at"])
                            break
                summary = self.account.overview(now=stamp)
                if _d(cohort["remaining_usd"]) <= 0:
                    stream_results[stream] = {"status":"UNREACHED","reason":"COHORT_BUDGET_EXHAUSTED"}
                    break
                signal = dict(decision["signal"])
                limit_price = self._stream_limit_price(decision)
                if market_refresh is not None:
                    self._precheck_market_refresh(signal,stream,boundary,stamp,summary,cohort)
                try:
                    market = self._market(markets,stream,boundary)
                except AccountRejected as exc:
                    if market_refresh is None or str(exc) not in ("market is missing", "verified contract window is missing"):
                        raise
                    market = None
                if market is not None and (market.get("asset") != ALL_STREAMS[stream][0]
                        or market.get("duration_seconds") != ALL_STREAMS[stream][1]):
                    raise AccountRejected("market underlying or horizon mismatch")
                if market is not None and signal.get("market_id") not in (None,market["market_id"]):
                    raise AccountRejected("signal market conflicts with official contract")
                needs_refresh = market is None
                if market_refresh is not None and market is not None:
                    meta_at = self._received_upper(market.get("metadata_received_at_local",market.get("metadata_received_at")),clock)
                    needs_refresh = (stamp-meta_at).total_seconds() > 90
                if market_refresh is not None and needs_refresh:
                    stamp = _stamp(now)
                    self._clock(clock,stamp)
                    if not boundary <= stamp < _ts(cohort["expires_at"]):
                        raise AccountRejected("decision cohort expired before market refresh")
                    self._precheck_market_refresh(signal,stream,boundary,stamp,summary,cohort)
                    existing = market
                    refreshed = market_refresh(stream,boundary,deepcopy(existing))
                    stamp = _stamp(now)
                    self._clock(clock,stamp)
                    if not boundary <= stamp < _ts(cohort["expires_at"]):
                        raise AccountRejected("decision cohort expired after market refresh")
                    market = self._market({stream:refreshed},stream,boundary)
                    if _ts(market.get("window_end")) != boundary + timedelta(seconds=ALL_STREAMS[stream][1]):
                        raise AccountRejected("refreshed market contract window mismatch")
                    if existing is not None and any(market.get(key) != existing.get(key)
                            for key in ("market_id", "condition_id", "protocol", "outcome_ids")):
                        raise AccountRejected("refreshed market contract identity mismatch")
                if market.get("asset") != ALL_STREAMS[stream][0] or market.get("duration_seconds") != ALL_STREAMS[stream][1]:
                    raise AccountRejected("market underlying or horizon mismatch")
                if signal.get("market_id") not in (None,market["market_id"]):
                    raise AccountRejected("signal market conflicts with official contract")
                signal["market_id"] = market["market_id"]
                # Validate before fetching or allowing a price skip. The skip
                # must not launder an expired/mismatched model signal.
                signal = self.account._validate_signal(signal,stamp)
                if signal["stream"] != stream or _ts(signal["window_start"]) != boundary:
                    raise AccountRejected("signal cohort mismatch")
                side = signal["outcome"]
                if multi_look is not None:
                    claim = self.account.begin_book_look(cohort["cohort_id"], stream, signal, now=now)
                    look_claim = (stream, claim["attempt"])
                    stamp = _stamp(now)
                    self._clock(clock, stamp)
                    if (stamp >= _ts(cohort["expires_at"])
                            or (claim["attempt"] > 0 and stamp > boundary+timedelta(seconds=18.5))):
                        raise AccountRejected("book look deadline expired before HTTP dispatch")
                observed = book_refresh(market,side) if book_refresh else market.get("books",{}).get(side)
                stamp = _stamp(now)
                self._clock(clock,stamp)
                if not stamp < _ts(cohort["expires_at"]):
                    raise AccountRejected("decision cohort expired")
                book,fee,capacity = self._adapt(market,side,observed,stamp,clock,allow_price_skip=True,price_limit=limit_price)
                if not _ts(signal["decision_at"]) < stamp-timedelta(seconds=2*float(_d(clock["uncertainty_seconds"]))):
                    raise AccountRejected("fill clock lower bound has not passed the decision")
                if capacity == 0:
                    receipt = {"source":"OBSERVED_PRICE_LIMIT_CHECK","signal":signal,"orderbook":book,"fee_quote":fee}
                    if limit_price != Decimal(".515"):
                        receipt["limit_price"] = float(limit_price)
                    if multi_look is not None:
                        completed = self.account.finish_book_look(cohort["cohort_id"],stream,look_claim[1],
                                                                 "PRICE_LIMIT_NOT_MET",receipt,now=now)
                        look_claim = None
                        if completed["next_retry_at"]:
                            best = min(_d(a["price"]) for a in book["asks"] if _d(a["size"]) > 0)
                            stream_results[stream] = {"status":"WAITING_PRICE_RECHECK","persisted":True,
                                "book_checked":True,"attempt":completed["attempt"],"best_ask":float(best),
                                "limit_price":float(limit_price),"next_retry_at":completed["next_retry_at"],
                                "book_observed_at":book["observed_at"],"approved_usd":0.,"spent_usd":0.}
                            result.update(status="WAITING_PRICE_RECHECK", next_retry_at=completed["next_retry_at"])
                            break
                    self.account.record_cohort_skip(cohort["cohort_id"],stream,"PRICE_LIMIT_NOT_MET",receipt,
                                                    now=now if multi_look is not None else stamp)
                    best = min(_d(a["price"]) for a in book["asks"] if _d(a["size"])>0)
                    stream_results[stream] = {"status":"PRICE_LIMIT_NOT_MET","raw_capacity_usd":0.,
                        "best_ask":float(best),"limit_price":float(limit_price),"book_observed_at":book["observed_at"],
                        "book_checked":True,"approved_usd":0.,"spent_usd":0.,"persisted":True}
                    continue
                if multi_look is not None:
                    receipt = {"source":"OBSERVED_EXECUTABLE_BOOK","signal":signal,"orderbook":book,"fee_quote":fee}
                    self.account.finish_book_look(cohort["cohort_id"],stream,look_claim[1],"EXECUTABLE_BOOK",receipt,now=now)
                # Re-read balances after I/O; both this target and the actual
                # account mutation enforce the unchanged aggregate risk limits.
                summary = self.account.overview(now=stamp)
                coin_cost = sum(_d(p["cost_usd"])+_d(p["fee_usd"]) for p in summary["positions"]
                                if p["underlying"]==signal["underlying"])
                amount = min(capacity,_d(cohort["single_usd"]),_d(cohort["remaining_usd"]),_d(summary["cash"]),
                             Decimal(200),Decimal(1000)-_d(summary["locked_cost"]),Decimal(500)-coin_cost)
                amount = amount.quantize(Decimal(".000001"),rounding=ROUND_DOWN)
                if amount <= 0:
                    raise AccountRejected("persistent budget or existing position limit exhausted")
                preceding = [{"signal_id":s["signal_id"],"rank":s["rank"],
                              "required_cost_usd":s["approved_usd"] if s["state"]=="FILLED_BUDGET" else s["spent_usd"]}
                             for s in cohort["streams"]
                             if s["state"]=="FILLED_BUDGET" or (cascade_enabled and s["state"]=="PARTIAL" and _d(s["spent_usd"]) > 0)]
                authorization = {"policy":"RANKED_CAPACITY_WATERFALL","waterfall_semantics":V5_SEMANTICS,
                    "plan_id":cohort["plan_id"],"rank":rank,"signal_id":signal["signal_id"],"market_id":market["market_id"],
                    "approved_stake_usd":float(amount),"planned_at":_iso(stamp),
                    "valid_until":_iso(min(_ts(signal["valid_until"]),stamp+timedelta(seconds=2),_ts(cohort["expires_at"]))),
                    "policy_version":cohort["policy_version"],"preceding_allocations":preceding,
                    "cohort_id":cohort["cohort_id"],"cohort_capacity_usd":float(capacity)}
                if limit_price != Decimal(".515"):
                    authorization["max_buy_price"] = float(limit_price)
                if multi_look is not None:
                    authorization["book_look_attempt"] = look_claim[1]
                fill = self.account.place_paper_buy(signal,book,fee,now=self._paper_acceptance_clock(now,clock,book,fee,signal),
                                                    stake=amount,limit_price=limit_price,allocation_authorization=authorization)
                look_claim = None
                result["fills"].append(fill)
                state = fill["cohort_stream_state"]
                spent = _d(fill["cost_usd"])+_d(fill["fee_usd"])
                stream_results[stream] = {"status":state,"raw_capacity_usd":float(capacity),"approved_usd":float(amount),
                    "spent_usd":float(spent),"order_id":fill["order_id"],
                    "book_checked":True,"persisted":True}
                cascade_allowed = (cascade_enabled and state == "PARTIAL" and spent > 0
                                   and (_ts(cohort["expires_at"]) - stamp).total_seconds() >= 4)
                if state != "FILLED_BUDGET" and not cascade_allowed:
                    raise AccountRejected("upper stream has an unsatisfied approved budget")
            if result.get("status") != "WAITING_PRICE_RECHECK":
                result["status"] = "PROCESSED"
        except Exception as exc:
            reason = str(exc) if isinstance(exc,AccountRejected) else type(exc).__name__
            if look_claim is not None and result["cohort"]:
                try:
                    self.account.fail_book_look(result["cohort"]["cohort_id"],look_claim[0],look_claim[1],reason,now=_stamp(now))
                except Exception:
                    # A failed audit write leaves the durable claim sealed from
                    # retry by its REQUESTED/EXECUTABLE state, never unlocked.
                    pass
            result["blocked"].append(reason)
            result["status"] = "PARTIAL_BLOCKED" if result["fills"] else "BLOCKED"
            if current is not None and stream_results[current]["status"] != "PARTIAL":
                stream_results[current] = {"status":"ERROR","reason":reason}
        if result["cohort"]:
            result["cohort"] = self.account.cohort_state(result["cohort"]["cohort_id"])
        return result

    def settle_observed(self, observations=None, now=None, fetcher=None):
        """Apply only final official states, retaining unresolved positions as pending.

        observations maps market_id -> (raw Gamma JSON, complete GET receipt).
        Optional fetcher has the same contract; no fetch occurs unless supplied.
        """
        result = {"settled":[],"pending":[],"live_enabled":False}
        with self.account._connection() as con:
            rows = con.execute("SELECT market_id,payload FROM orders WHERE status='OPEN'").fetchall()
        for market_id in sorted({r["market_id"] for r in rows}):
            try:
                order = json.loads(next(r["payload"] for r in rows if r["market_id"]==market_id))
                expected = order.get("venue_market")
                if not expected:
                    raise AccountRejected("original market identity receipt is missing")
                pair = (observations or {}).get(market_id)
                if pair is None and fetcher:
                    pair = fetcher(market_id)
                if pair is None:
                    raise AccountRejected("WAITING_OFFICIAL_FINAL_RESOLUTION")
                parsed = adapt_official_resolution(pair[0],pair[1],expected,now=now)
                result["settled"].append(self.account.settle(market_id,parsed,now=_stamp(now)))
            except (AccountRejected,KeyError,ValueError,TypeError) as exc:
                result["pending"].append({"market_id":market_id,"reason":str(exc) if isinstance(exc,AccountRejected) else type(exc).__name__})
            except Exception as exc:
                result["pending"].append({"market_id":market_id,"reason":type(exc).__name__})
        return result
