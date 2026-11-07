"""Stage selection is isolated from model weights, prior cohorts and real venues."""
from copy import deepcopy
from datetime import timedelta
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tools.argon4_forward import pipeline, stage
from tools.argon4_forward.account import Account
from tools.argon4_forward.execution import PaperExecutor
from test_argon4_forward_execution import BOUNDARY, NOW, ORDER, clock, fixture_inputs, no_signal, v5_policy


def staged_policy():
    return {**v5_policy(), "execution_stage": {"id": "three-stream-fixture-v1", "status": "CONFIRMED",
            "enabled_streams": ["BTC_15m", "ETH_5m", "BTC_5m"], "effective_from": BOUNDARY.isoformat()}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("stage tests must remain offline"))


def test_missing_stage_preserves_seven_streams_and_profile_is_detached():
    policy = v5_policy()
    profile = stage.stage_profile(policy)
    assert profile["enabled_streams"] == ORDER and not profile["configured"]
    profile["enabled_streams"].clear()
    assert policy["order"] == ORDER


def test_stage_subset_keeps_full_ranking_and_never_enables_live():
    policy = staged_policy()
    before = deepcopy(policy)
    profile = stage.stage_profile(policy, BOUNDARY)
    assert profile["enabled_streams"] == ["BTC_15m", "ETH_5m", "BTC_5m"]
    assert set(profile["disabled_streams"]) == {"ETH_15m", "SOL_15m", "SOL_5m", "BNB_15m"}
    assert profile["live_enabled"] is False and policy == before
    profile["enabled_streams"].clear()
    assert policy == before


@pytest.mark.parametrize("bad", ["empty", "duplicate", "unknown", "wrong_order", "unconfirmed", "missing_time",
                                  "naive_time", "off_grid", "extra_field", "legacy", "null", "bool_stream"])
def test_invalid_stage_fails_closed(bad):
    policy = staged_policy()
    spec = policy["execution_stage"]
    if bad == "empty": spec["enabled_streams"] = []
    elif bad == "duplicate": spec["enabled_streams"].append("BTC_5m")
    elif bad == "unknown": spec["enabled_streams"] = ["DOGE_5m"]
    elif bad == "wrong_order": spec["enabled_streams"].reverse()
    elif bad == "unconfirmed": spec["status"] = "DRAFT"
    elif bad == "missing_time": spec.pop("effective_from")
    elif bad == "naive_time": spec["effective_from"] = "2026-11-06T00:00:00"
    elif bad == "off_grid": spec["effective_from"] = (BOUNDARY + timedelta(seconds=1)).isoformat()
    elif bad == "extra_field": spec["live_enabled"] = True
    elif bad == "legacy": policy.pop("waterfall_semantics")
    elif bad == "null": policy["execution_stage"] = None
    elif bad == "bool_stream": spec["enabled_streams"] = [True]
    with pytest.raises(stage.StageRejected):
        stage.stage_profile(policy, BOUNDARY)


@pytest.mark.parametrize("boundary", [BOUNDARY - timedelta(minutes=5), BOUNDARY + timedelta(seconds=1)])
def test_no_historical_or_off_grid_replay(boundary):
    with pytest.raises(stage.StageRejected):
        stage.stage_profile(staged_policy(), boundary)


def test_disabled_middle_tiers_need_no_model_or_book_and_do_not_block_lower_budget(tmp_path):
    policy = staged_policy()
    account = Account(tmp_path / "paper.sqlite3", started_at=BOUNDARY - timedelta(seconds=1))
    executor = PaperExecutor(account, policy)
    all_decisions, all_markets = fixture_inputs(capacity=600)
    enabled = policy["execution_stage"]["enabled_streams"]
    decisions = {s: all_decisions[s] for s in enabled}
    markets = {s: all_markets[s] for s in enabled}
    decisions["BTC_15m"] = no_signal("BTC_15m")
    calls = []
    def book(market, side):
        calls.append(market["market_id"])
        return market["books"][side]
    result = executor.process_cohort(BOUNDARY.isoformat(), decisions, markets, now=NOW, clock=clock(), book_refresh=book)
    assert result["status"] == "PROCESSED"
    assert [(x["underlying"], x["cost_usd"]) for x in result["fills"]] == [("ETH", 150), ("BTC", 70)]
    assert calls == [markets["ETH_5m"]["market_id"], markets["BTC_5m"]["market_id"]]
    assert result["cohort"]["used_usd"] == 220 and account.overview()["signal_count"] == 2
    for stream in stage.stage_profile(policy)["disabled_streams"]:
        row = result["stream_results"][stream]
        assert row["status"] == stage.DISABLED and row["approved_usd"] == row["spent_usd"] == 0
        assert not row["book_checked"] and not row["model_evaluated"]
    repeated = executor.process_cohort(BOUNDARY.isoformat(), {}, {}, now=NOW, clock=clock())
    assert repeated["status"] == "PROCESSED" and repeated["fills"] == []
    assert account.overview()["order_count"] == 2


def test_disabled_state_remains_explicit_when_enabled_stream_blocks(tmp_path):
    policy = staged_policy()
    account = Account(tmp_path / "paper.sqlite3", started_at=BOUNDARY - timedelta(seconds=1))
    result = PaperExecutor(account, policy).process_cohort(BOUNDARY.isoformat(), {}, {}, now=NOW, clock=clock())
    assert result["status"] == "BLOCKED" and account.overview()["order_count"] == 0
    assert all(result["stream_results"][s]["status"] == stage.DISABLED
               for s in stage.stage_profile(policy)["disabled_streams"])


def test_pipeline_only_evaluates_enabled_due_streams_without_changing_inputs(tmp_path, monkeypatch):
    from tools.argon4_forward import model
    calls, feature_calls = [], []
    p = pipeline.ForwardPipeline(SimpleNamespace(), tmp_path, SimpleNamespace())
    p.clock = {}
    p.bundles = {s: object() for s in ORDER}
    monkeypatch.setattr(pipeline, "calibrated_now", lambda c, upper=True: NOW)
    frame = pd.DataFrame({"fixture": [1.]}, index=pd.DatetimeIndex([BOUNDARY - timedelta(minutes=5)]))
    def features(root, symbol, **kwargs):
        feature_calls.append((root, symbol, kwargs))
        return {"base": frame, "expert": frame}
    def predict(stream, *args, **kwargs):
        calls.append(stream)
        return {"selected": False, "side": "UP"}
    monkeypatch.setattr(model, "build_feature_frames", features)
    monkeypatch.setattr(model, "predict_stream", predict)
    inputs = {"data_root": "unchanged-four-coin-fixture", "feature_available_at": BOUNDARY.isoformat(),
              "time_basis": "CALIBRATED_UTC_UPPER_BOUND"}
    before = deepcopy(inputs)
    decisions = p.prepare_decisions(staged_policy(), inputs, BOUNDARY)
    assert calls == ["BTC_15m", "ETH_5m", "BTC_5m"]
    assert [x[1] for x in feature_calls] == ["BTCUSDT", "ETHUSDT"]
    assert all(x[0] == inputs["data_root"] for x in feature_calls) and inputs == before
    assert len(p.bundles) == len(decisions) == 7
    assert all(decisions[s] == stage.disabled_decision(stage.stage_profile(staged_policy()))
               for s in stage.stage_profile(staged_policy())["disabled_streams"])


def test_disabled_model_is_not_loaded_and_all_original_policy_pins_remain(tmp_path, monkeypatch):
    from tools.argon4_forward import model
    policy = staged_policy()
    policy.update(paper_trading_active=True, ranking_model_scope="LOCAL_SELF_TRAINED_OOS",
                  deployment_manifest_sha256={s: "a" * 64 for s in ORDER})
    config = tmp_path / "config"
    config.mkdir()
    path = config / "argon4_forward_policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    original = path.read_bytes()
    calls = []
    def load(stream, expected_manifest_sha256):
        assert stream in policy["execution_stage"]["enabled_streams"]
        calls.append(stream)
        return {"manifest_sha256": expected_manifest_sha256}
    monkeypatch.setattr(model, "load_model_bundle", load)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    p = pipeline.ForwardPipeline(SimpleNamespace(), tmp_path, SimpleNamespace())
    assert p.load_ready() == policy
    assert set(calls) == set(policy["execution_stage"]["enabled_streams"])
    assert len(p.bundles) == 3 and len(policy["deployment_manifest_sha256"]) == 7
    assert path.read_bytes() == original


def test_future_stage_waits_but_polls_existing_settlements_first(tmp_path, monkeypatch):
    policy = staged_policy()
    policy["execution_stage"]["effective_from"] = (BOUNDARY + timedelta(minutes=5)).isoformat()
    p = pipeline.ForwardPipeline(SimpleNamespace(heartbeat=lambda *a, **k: None), tmp_path, SimpleNamespace())
    p.warmup_complete = True
    p.clock = {}
    calls = []
    monkeypatch.setattr(p, "poll_settlements", lambda policy: calls.append("settlements"))
    monkeypatch.setattr(p, "load_ready", lambda: policy)
    monkeypatch.setattr(p, "update_clock", lambda: {})
    monkeypatch.setattr(pipeline, "calibrated_now", lambda c, upper=True: NOW)
    p.cycle()
    assert calls == ["settlements"]
    assert p.status()["status"] == "STAGE_WAITING_ACTIVATION" and p.inputs is None
    assert p.last_cohort is None


def test_status_and_disabled_live_preflight_use_same_stage_without_model_removal(tmp_path, monkeypatch):
    from scripts import argon4_forward_server as server
    policy = staged_policy()
    original = server.load_json
    monkeypatch.setattr(server, "load_json", lambda path, default: policy if path.name == "argon4_forward_policy.json" else original(path, default))
    app = server.Application(tmp_path / "runtime")
    app.pipeline = SimpleNamespace(status=lambda: {"active": True, "status": "PAPER_WAITING_WINDOW", "models": {s: {"version": "fixture"} for s in ORDER}})
    result = app.status()
    assert result["stage"] == result["live_preflight"]["execution_stage"]
    assert result["live_enabled"] is False and result["live_preflight"]["live_enabled"] is False
    assert not result["live_preflight"]["signer_connected"]
    assert sum(s["stage_enabled"] for s in result["streams"]) == 3
    assert sum(s["status"] == stage.DISABLED for s in result["streams"]) == 4
    assert all(s["model"] is not None for s in result["streams"])
