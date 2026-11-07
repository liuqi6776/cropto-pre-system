"""Public metadata recovery keeps actual provenance and the original deadline."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.argon4_forward import pipeline


BOUNDARY = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc)


def raw_market():
    start, end = BOUNDARY.isoformat(), (BOUNDARY+timedelta(minutes=5)).isoformat()
    slug = f"eth-updown-5m-{int(BOUNDARY.timestamp())}"
    return {"id": "123", "slug": slug, "version": "v2", "conditionId": "0x"+"a"*64,
            "question": "Ethereum Up or Down - fixture", "eventStartTime": start, "endDate": end,
            "events": [{"slug": slug, "startTime": start, "endDate": end}],
            "active": True, "closed": False, "archived": False, "acceptingOrders": True,
            "enableOrderBook": True, "negRisk": False,
            "cryptoMarketConfig": {"asset": "eth", "duration": "5m", "twapEnabled": True, "twapLookbackSeconds": 60},
            "resolutionSource": "https://data.chain.link/streams/eth-usd-twap-60s-streams",
            "description": 'resolve to "Up" if Chainlink ETH/USD is greater than or equal to; otherwise resolve to "Down"',
            "outcomes": ["Up", "Down"], "positionIds": ["111", "222"],
            "feesEnabled": True, "feeSchedule": {"rate": .07, "exponent": 1, "takerOnly": True, "rebateRate": .2}}


@pytest.fixture
def refresh_fixture(tmp_path, monkeypatch):
    state = SimpleNamespace(now=BOUNDARY+timedelta(seconds=10), calls=[], records=[], payload=[raw_market()], delay=0)
    observations = SimpleNamespace(save=lambda kind, value: state.records.append((kind, deepcopy(value))))
    p = pipeline.ForwardPipeline(None, tmp_path, observations)
    p.clock = {"receipt_sha256": "b"*64}
    monkeypatch.setattr(pipeline, "calibrated_now", lambda clock: state.now)
    def get(kind, value):
        state.calls.append((kind, value))
        started = state.now
        state.now += timedelta(seconds=state.delay)
        raw = json.dumps(state.payload)
        return state.payload, {"requested_at": started.isoformat(), "received_at": state.now.isoformat(),
            "sha256": hashlib.sha256(raw.encode()).hexdigest(), "raw_text": raw,
            "http_status": 200, "body_complete": True}, None
    monkeypatch.setattr(p.public_transport, "get", get)
    yield p, state
    p.close()


def test_exact_metadata_refresh_keeps_official_window_and_original_receipt(refresh_fixture):
    p, state = refresh_fixture
    market = p.refresh_market("ETH_5m", BOUNDARY)
    assert state.calls == [("market", f"eth-updown-5m-{int(BOUNDARY.timestamp())}")]
    assert market["market_id"] == "123" and market["outcome_ids"] == {"UP": "111", "DOWN": "222"}
    kind, saved = state.records[0]
    assert kind == "paper_market_receipt"
    assert market["metadata_received_at"] == saved["receipts"][0]["received_at"]
    assert market["metadata_sha256"] == saved["receipts"][0]["sha256"]
    assert market["books"] == {}  # metadata cannot substitute for executable depth


@pytest.mark.parametrize("bad", [{"eventStartTime": None}, {"positionIds": ["111", "111"]},
                               {"closed": True}, {"resolutionSource": "https://invalid.example"}])
def test_metadata_recovery_does_not_weaken_official_identity_checks(refresh_fixture, bad):
    p, state = refresh_fixture
    state.payload[0].update(bad)
    with pytest.raises(ValueError):
        p.refresh_market("ETH_5m", BOUNDARY)
    assert len(state.calls) == 1 and len(state.records) == 1


def test_late_metadata_is_saved_as_evidence_but_not_accepted(refresh_fixture):
    p, state = refresh_fixture
    state.delay = 10
    with pytest.raises(ValueError, match="exceeded decision window"):
        p.refresh_market("ETH_5m", BOUNDARY)
    assert len(state.calls) == 1 and len(state.records) == 1
    with pytest.raises(ValueError, match="outside decision window"):
        p.refresh_market("ETH_5m", BOUNDARY)
    assert len(state.calls) == 1  # no delayed retry or historical replay
