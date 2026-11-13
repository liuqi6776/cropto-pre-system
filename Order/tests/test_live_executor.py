"""Comprehensive unit, regression, and security audit tests for the ARGON4 External Live Executor."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import time
from typing import Any
import pytest

from argon4_live.audit_store import AuditStore
from argon4_live.clob_executor import (
    ClobExecutor,
    derive_metamask_polymarket_addresses,
    derive_polymarket_candidate_wallets,
)
from argon4_live.engine import LiveExecutionEngine, resolve_mode_audit_paths
from argon4_live.math_utils import (
    align_limit_price,
    compute_live_order_sizing,
)
from argon4_live.redeemer import (
    AutoRedeemer,
    _RelayerExecResponse,
    encode_redeem_positions_calldata,
    encode_v2_router_redeem_calldata,
)
from argon4_live.safety import (
    OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS,
    OFFICIAL_POLYGON_PUSD_ADDRESS,
    OFFICIAL_POLYGON_USDCE_ADDRESS,
    OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS,
     OFFICIAL_STANDARD_EXCHANGE_V2_ADDRESS,
    SafetyViolationError,
    SingleInstanceFileLock,
    assert_fak_order_only,
    assert_no_withdrawal_permission,
    assert_redeem_only_transaction,
    assert_safe_read_path,
    assert_safe_write_path,
)
from argon4_live.upstream_observer import UpstreamObserver


def test_physical_isolation_blocks_upstream_sqlite_and_writes(tmp_path: Path) -> None:
    upstream_root = tmp_path / "crypto-predict_system"
    paper_rt = upstream_root / "paper_runtime" / "argon4_forward_20261105"
    paper_rt.mkdir(parents=True)
    sqlite_file = paper_rt / "account.sqlite3"
    sqlite_file.write_text("dummy", encoding="utf-8")
    state_file = paper_rt / "pipeline_state.json"
    state_file.write_text("{}", encoding="utf-8")

    assert assert_safe_read_path(state_file, upstream_root) == state_file.resolve()

    with pytest.raises(SafetyViolationError, match="CRITICAL ISOLATION VIOLATION"):
        assert_safe_read_path(sqlite_file, upstream_root)

    with pytest.raises(SafetyViolationError, match="CRITICAL ISOLATION VIOLATION"):
        assert_safe_read_path(paper_rt / "account.sqlite3-journal", upstream_root)

    local_root = tmp_path / "Order"
    local_root.mkdir()
    with pytest.raises(SafetyViolationError, match="Writing to upstream"):
        assert_safe_write_path(paper_rt / "hack.json", local_root, upstream_root)

    valid_write = assert_safe_write_path(local_root / "runtime" / "live_orders.jsonl", local_root, upstream_root)
    assert valid_write == (local_root / "runtime" / "live_orders.jsonl").resolve()


def test_fak_order_type_and_non_withdrawal_guards() -> None:
    assert assert_fak_order_only("FAK", post_only=False) == "FAK"

    for bad_type in ("GTC", "GTD", "FOK", "LIMIT"):
        with pytest.raises(SafetyViolationError, match="Only OrderType.FAK is permitted"):
            assert_fak_order_only(bad_type, post_only=False)

    with pytest.raises(SafetyViolationError, match="post_only=True"):
        assert_fak_order_only("FAK", post_only=True)

    with pytest.raises(SafetyViolationError, match="withdrawal_permission"):
        assert_no_withdrawal_permission({"withdrawal_permission": True})

    cond_id = "0x" + ("ab" * 32)
    calldata = encode_redeem_positions_calldata(cond_id)
    assert calldata.startswith("0x01b7037c")
    assert_redeem_only_transaction(OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS, calldata, value=0)

    # Reject ERC-20 transfer selector (0xa9059cbb)
    with pytest.raises(SafetyViolationError, match="Forbidden transfer/withdrawal selector"):
        assert_redeem_only_transaction(OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS, "0xa9059cbb" + ("00" * 64), value=0)

    # Reject non-whitelisted destination address
    with pytest.raises(SafetyViolationError, match="does not match official CTF/Adapter contracts"):
        assert_redeem_only_transaction("0x1111111111111111111111111111111111111111", calldata, value=0)


def test_two_stage_scaling_and_fee_inclusive_share_sizing() -> None:
    assert align_limit_price("0.01") == Decimal("0.51")
    assert align_limit_price("0.001") == Decimal("0.515")

    s1 = compute_live_order_sizing(
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        tick_size="0.01",
    )
    assert s1["requested_stake_usd"] == 20.0
    assert s1["limit_price"] == 0.51
    assert s1["requested_quantity"] == "37.82"
    assert s1["est_total_spent_usdc"] <= 20.0

    s1_milli = compute_live_order_sizing(
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        tick_size="0.001",
    )
    assert s1_milli["limit_price"] == 0.515
    assert s1_milli["requested_quantity"] == "37.47"
    assert s1_milli["est_total_spent_usdc"] <= 20.0

    s2 = compute_live_order_sizing(
        approved_stake_usd=200.0,
        live_scale_factor=1.00,
        fee_rate=0.075,
        tick_size="0.01",
    )
    assert s2["requested_stake_usd"] == 200.0
    assert s2["requested_quantity"] == "378.25"
    assert s2["est_total_spent_usdc"] <= 200.0


def test_zero_latency_local_signing_makes_zero_http_gets(monkeypatch: pytest.MonkeyPatch) -> None:
    import py_clob_client_v2.http_helpers.helpers as clob_helpers

    def _forbid_http_get(*args: object, **kwargs: object) -> None:
        raise AssertionError("Synchronous HTTP GET called during T+4s local order signing!")

    monkeypatch.setattr(clob_helpers, "get", _forbid_http_get)

    executor = ClobExecutor(
        mode="shadow",
        private_key="0x" + ("11" * 32),
        api_key="test_key",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="test_passphrase",
        signature_type=2,
        funder_address="0x" + ("22" * 20),
        auto_derive_l2_creds=False,
    )
    token_id = "12148846513639042960024489252679543841427766245697318000455787422208948329877"
    t0 = time.perf_counter()
    res = executor.sign_and_execute_fak(
        token_id=token_id,
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert res["live_execution"]["clob_status"] == "SHADOW_MATCHED"
    assert res["live_execution"]["order_type"] == "FAK"
    assert res["live_execution"]["clob_order_id"].startswith("0x")
    assert elapsed_ms < 50.0


def test_two_key_env_metamask_auto_derivation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from py_clob_client_v2.clob_types import ApiCreds
    from py_clob_client_v2.client import ClobClient

    pk = "0x" + ("55" * 32)
    eoa_addr, safe_funder = derive_metamask_polymarket_addresses(pk)
    cands = derive_polymarket_candidate_wallets(pk)
    assert eoa_addr.startswith("0x") and len(eoa_addr) == 42
    assert safe_funder.startswith("0x") and len(safe_funder) == 42
    assert cands["beacon_deposit_wallet"].startswith("0x")

    executor = ClobExecutor(mode="shadow", private_key=pk, auto_derive_l2_creds=False)
    assert executor.signature_type == 2
    assert executor.signer_address.lower() == eoa_addr.lower()
    assert executor.funder_address.lower() == safe_funder.lower()

    # Verify auto-derivation calls `create_or_derive_api_key()` on `py_clob_client_v2.client.ClobClient`
    derive_called = 0

    def _mock_create_or_derive_api_key(self_client: Any, nonce: int | None = None) -> ApiCreds:
        nonlocal derive_called
        derive_called += 1
        return ApiCreds(
            api_key="derived_v2_key",
            api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
            api_passphrase="derived_v2_passphrase",
        )

    monkeypatch.setattr(ClobClient, "create_or_derive_api_key", _mock_create_or_derive_api_key)
    cache_file = tmp_path / ".derived_clob_creds.json"
    exec_live_derived = ClobExecutor(
        mode="live",
        private_key=pk,
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=True,
        creds_cache_path=cache_file,
    )
    assert derive_called == 1
    assert exec_live_derived.has_l2_creds is True
    assert exec_live_derived.client is not None
    assert exec_live_derived.client.creds.api_key == "derived_v2_key"

    redeemer = AutoRedeemer(
        mode="shadow",
        private_key=pk,
        relayer_api_key="my_relayer_key",
    )
    assert redeemer.signature_type == 2
    assert redeemer.signer_address.lower() == eoa_addr.lower()
    assert redeemer.relayer_api_key_address.lower() == eoa_addr.lower()
    assert redeemer.safe_funder_address.lower() == safe_funder.lower()


def test_multi_look_incremental_execution_idempotency_and_deadline_abort(tmp_path: Path) -> None:
    upstream_root = tmp_path / "crypto-predict_system"
    runtime_dir = upstream_root / "paper_runtime" / "argon4_forward_20261105"
    config_dir = upstream_root / "config"
    runtime_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    policy_path = config_dir / "argon4_forward_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "max_order_usd": 200.0,
                "max_open_risk_usd": 1000.0,
                "max_underlying_open_risk_usd": 500.0,
                "max_tick_budget_usd": 1200.0,
                "max_buy_price": 0.515,
                "min_buy_price": 0.37,
            }
        ),
        encoding="utf-8",
    )
    state_path = runtime_dir / "pipeline_state.json"

    local_root = tmp_path / "Order"
    audit_dir = local_root / "runtime"
    audit_store = AuditStore(
        sqlite_path=audit_dir / "live_audit.sqlite3",
        orders_jsonl_path=audit_dir / "live_orders.jsonl",
        quota_jsonl_path=audit_dir / "live_quota_snapshots.jsonl",
        settlements_jsonl_path=audit_dir / "live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    observer = UpstreamObserver(
        pipeline_state_path=state_path,
        policy_path=policy_path,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(
        mode="shadow",
        private_key="0x" + ("33" * 32),
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=2,
        funder_address="0x" + ("44" * 20),
        auto_derive_l2_creds=False,
    )
    redeemer = AutoRedeemer(mode="shadow", withdrawal_permission=False)
    engine = LiveExecutionEngine(
        config={
            "capital_and_scaling": {"live_scale_factor": 0.10},
            "redemption": {"withdrawal_permission": False},
        },
        observer=observer,
        audit_store=audit_store,
        clob_executor=clob,
        redeemer=redeemer,
    )

    cohort_id = "2026-11-08T13:05:00+00:00"
    sig_btc = "a" * 64
    sig_eth = "b" * 64
    cond_btc = "0x" + ("11" * 32)
    cond_eth = "0x" + ("22" * 32)

    fill_btc = {
        "order_id": f"paper:{sig_btc}",
        "signal_id": sig_btc,
        "market_id": "512345",
        "underlying": "BTC",
        "outcome": "UP",
        "status": "OPEN",
        "quantity": "378.12345678",
        "cost_usd": 192.842963,
        "fee_usd": 7.157037,
        "unfilled_budget_usd": 0.0,
        "fill_at": "2026-11-08T13:05:05.110000+00:00",
        "book_observed_at": "2026-11-08T13:05:04.950000+00:00",
        "fee_quote": {"fee_model": "polymarket_crypto", "rate": 0.075, "exponent": 1, "taker_only": True},
        "allocation_authorization": {
            "policy": "RANKED_CAPACITY_WATERFALL",
            "plan_id": "cohort:plan1",
            "cohort_id": cohort_id,
            "rank": 1,
            "signal_id": sig_btc,
            "market_id": "512345",
            "approved_stake_usd": 200.0,
            "cohort_capacity_usd": 450.12,
        },
        "venue_market": {
            "market_id": "512345",
            "condition_id": cond_btc,
            "protocol": "v1",
            "asset": "BTC",
            "window_start": "2026-11-08T13:05:00Z",
            "window_end": "2026-11-08T13:20:00Z",
            "outcome_ids": {"UP": "10001", "DOWN": "10002"},
        },
    }

    fill_eth = {
        "order_id": f"paper:{sig_eth}",
        "signal_id": sig_eth,
        "market_id": "512346",
        "underlying": "ETH",
        "outcome": "DOWN",
        "status": "OPEN",
        "quantity": "300.00000000",
        "cost_usd": 150.0,
        "fee_usd": 5.625,
        "unfilled_budget_usd": 0.0,
        "fill_at": "2026-11-08T13:05:10.200000+00:00",
        "book_observed_at": "2026-11-08T13:05:10.050000+00:00",
        "fee_quote": {"fee_model": "polymarket_crypto", "rate": 0.075, "exponent": 1, "taker_only": True},
        "allocation_authorization": {
            "policy": "RANKED_CAPACITY_WATERFALL",
            "plan_id": "cohort:plan1",
            "cohort_id": cohort_id,
            "rank": 2,
            "signal_id": sig_eth,
            "market_id": "512346",
            "approved_stake_usd": 155.625,
            "cohort_capacity_usd": 310.0,
        },
        "venue_market": {
            "market_id": "512346",
            "condition_id": cond_eth,
            "protocol": "v1",
            "asset": "ETH",
            "window_start": "2026-11-08T13:05:00Z",
            "window_end": "2026-11-08T13:10:00Z",
            "outcome_ids": {"UP": "20001", "DOWN": "20002"},
        },
    }

    decisions = {
        "BTC_15m": {
            "status": "SIGNAL",
            "outcome": "UP",
            "signal": {
                "signal_id": sig_btc,
                "stream": "BTC_15m",
                "underlying": "BTC",
                "outcome": "UP",
                "market_id": "512345",
                "decision_at": "2026-11-08T13:05:04.620000+00:00",
                "valid_until": "2026-11-08T13:05:20+00:00",
                "window_start": cohort_id,
                "window_end": "2026-11-08T13:20:00+00:00",
            },
        },
        "ETH_5m": {
            "status": "SIGNAL",
            "outcome": "DOWN",
            "signal": {
                "signal_id": sig_eth,
                "stream": "ETH_5m",
                "underlying": "ETH",
                "outcome": "DOWN",
                "market_id": "512346",
                "decision_at": "2026-11-08T13:05:04.700000+00:00",
                "valid_until": "2026-11-08T13:05:20+00:00",
                "window_start": cohort_id,
                "window_end": "2026-11-08T13:10:00+00:00",
            },
        },
    }

    state_pass1 = {
        "status": "PAPER_WAITING_PRICE_RECHECK",
        "active": True,
        "last_cohort": cohort_id,
        "decisions": decisions,
        "execution": {"status": "WAITING_PRICE_RECHECK", "fills": [fill_btc]},
    }
    t_detect1 = datetime(2026, 11, 8, 13, 5, 5, 145000, tzinfo=timezone.utc)
    out1 = engine.process_pipeline_state(state_pass1, detected_at_local_utc=t_detect1)
    assert len(out1) == 1
    assert out1[0]["signal_id"] == sig_btc
    assert out1[0]["live_execution"]["clob_status"] == "SHADOW_MATCHED"
    assert out1[0]["timing"]["detection_lag_ms"] == 35.0

    state_pass2 = {
        "status": "PAPER_CYCLE_COMPLETE",
        "active": True,
        "last_cohort": cohort_id,
        "decisions": decisions,
        "execution": {"status": "PROCESSED", "fills": [fill_btc, fill_eth]},
    }
    t_detect2 = datetime(2026, 11, 8, 13, 5, 10, 250000, tzinfo=timezone.utc)
    out2 = engine.process_pipeline_state(state_pass2, detected_at_local_utc=t_detect2)
    assert len(out2) == 1
    assert out2[0]["signal_id"] == sig_eth
    assert out2[0]["live_execution"]["clob_status"] == "SHADOW_MATCHED"

    sig_sol = "c" * 64
    fill_sol = {
        **fill_eth,
        "order_id": f"paper:{sig_sol}",
        "signal_id": sig_sol,
        "market_id": "512347",
        "underlying": "SOL",
    }
    decisions["SOL_15m"] = {
        "status": "SIGNAL",
        "outcome": "DOWN",
        "signal": {
            "signal_id": sig_sol,
            "stream": "SOL_15m",
            "underlying": "SOL",
            "outcome": "DOWN",
            "market_id": "512347",
            "decision_at": "2026-11-08T13:05:04.800000+00:00",
            "valid_until": "2026-11-08T13:05:20+00:00",
            "window_start": cohort_id,
            "window_end": "2026-11-08T13:20:00+00:00",
        },
    }
    state_late = {
        "status": "PAPER_CYCLE_COMPLETE",
        "active": True,
        "last_cohort": cohort_id,
        "decisions": decisions,
        "execution": {"status": "PROCESSED", "fills": [fill_btc, fill_eth, fill_sol]},
    }
    t_late = datetime(2026, 11, 8, 13, 5, 20, 500000, tzinfo=timezone.utc)
    out_late = engine.process_pipeline_state(state_late, detected_at_local_utc=t_late)
    assert len(out_late) == 1
    assert out_late[0]["signal_id"] == sig_sol
    assert out_late[0]["live_execution"]["clob_status"] == "ABORTED_PAST_VALID_UNTIL"
    assert out_late[0]["live_execution"]["filled_quantity"] == "0.00"

    observer.fetch_ledger = lambda allow_in_critical_window=False: [  # type: ignore[method-assign]
        {
            "seq": 2,
            "utc": "2026-11-08T13:20:15+00:00",
            "event_type": "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER",
            "payload": {
                "market_id": "512345",
                "payout_usd": 378.123456,
                "realized_pnl_usd": 178.123456,
                "resolution": {"market_id": "512345", "outcome": "UP"},
            },
        },
        {
            "seq": 1,
            "utc": "2026-11-08T13:10:15+00:00",
            "event_type": "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER",
            "payload": {
                "market_id": "512346",
                "payout_usd": 0.0,
                "realized_pnl_usd": -155.625,
                "resolution": {"market_id": "512346", "outcome": "UP"},
            },
        },
    ]
    settled = engine.poll_settlements_and_redeem(force=True)
    assert len(settled) == 2
    by_mkt = {s["market_id"]: s for s in settled}
    assert by_mkt["512345"]["redeem_status"] == "REDEEMED"
    assert by_mkt["512345"]["redeemed_usdc_amount"] == 37.82
    assert by_mkt["512346"]["redeem_status"] == "LOST_NO_REDEEM"
    assert by_mkt["512346"]["redeemed_usdc_amount"] == 0.0
    observer.close()


def test_p1_1_v2_order_protocol_and_pusd_router_and_adapter_redemption() -> None:
    """[P1-1] Verify V2 order signing (SignedOrderV2, DepositWallet sig_type=3 & Safe sig_type=2)
    and pUSD redemption via Protocol V2 Router (0xd217a3cc) and CtfCollateralAdapter (0x01b7037c).
    """
    from py_clob_client_v2.order_utils.model.order_data_v2 import SignedOrderV2

    pk = "0x" + ("12" * 32)
    cands = derive_polymarket_candidate_wallets(pk)

    exec_deposit = ClobExecutor(
        mode="shadow",
        private_key=pk,
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    assert exec_deposit.signature_type == 3
    assert exec_deposit.wallet_type == "DEPOSIT_WALLET"

    captured_orders: list[Any] = []
    assert exec_deposit.client is not None
    orig_create = exec_deposit.client.create_order

    def _spy_create(order_args: Any, options: Any = None) -> Any:
        signed = orig_create(order_args, options=options)
        captured_orders.append((order_args, options, signed))
        return signed

    exec_deposit.client.create_order = _spy_create  # type: ignore[method-assign]
    exec_deposit.sign_and_execute_fak(
        token_id="0x02" + ("ab" * 31),
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        market_protocol="v2",
    )
    assert len(captured_orders) == 1
    o_args, o_opts, signed_ord = captured_orders[0]
    assert isinstance(signed_ord, SignedOrderV2)
    assert o_opts.version == 3
    assert int(signed_ord.signatureType) == 3
    # ERC-1271 wrapped DepositWallet signature is much longer than 65-byte raw ECDSA
    assert len(signed_ord.signature) > 200

    # V2 Router redeem calldata (0xd217a3cc)
    v2_calldata = encode_v2_router_redeem_calldata(
        condition_id="0x" + ("cd" * 31) + "00",
        outcome_index=0,
        amount_base_units=37820000,
    )
    assert v2_calldata.startswith("0xd217a3cc")
    assert_redeem_only_transaction(OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS, v2_calldata, value=0)

    # V1/CTF CtfCollateralAdapter redeem calldata with pUSD (0x01b7037c)
    ctf_calldata = encode_redeem_positions_calldata(
        condition_id="0x" + ("ef" * 32),
        collateral_address=OFFICIAL_POLYGON_PUSD_ADDRESS,
        target_contract_address=OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS,
    )
    assert ctf_calldata.startswith("0x01b7037c")

    # Legacy USDC.e (0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174) is strictly rejected when pUSD is required
    with pytest.raises(SafetyViolationError, match="collateralToken"):
        encode_redeem_positions_calldata(
            condition_id="0x" + ("ef" * 32),
            collateral_address="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
            target_contract_address=OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS,
        )


def test_p1_2_order_timeout_marked_unknown_locks_risk_and_reconciles(tmp_path: Path) -> None:
    """[P1-2] Verify that an order submission timeout/exception is recorded as ORDER_STATE_UNKNOWN
    (never REJECTED), locks risk quota against subsequent buys, and reconciles via CLOB API.
    """
    pk = "0x" + ("23" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    clob.balance_verified = True
    clob._cached_free_usdc = 500.0
    clob._cached_allowance_usdc = 500.0

    assert clob.client is not None
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.48", "size": "500"}]}  # type: ignore[method-assign]
    clob.client.post_order = lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("HTTP read timed out"))  # type: ignore[method-assign]

    res = clob.sign_and_execute_fak(
        token_id="10001",
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
    )
    live_exec = res["live_execution"]
    assert live_exec["clob_status"] == "ORDER_STATE_UNKNOWN"
    assert live_exec["clob_order_id"] is not None and live_exec["clob_order_id"].startswith("0x")

    # Record in AuditStore and verify full requested stake ($20.00) remains locked!
    audit = AuditStore(
        sqlite_path=tmp_path / "Order/runtime/live_audit.sqlite3",
        orders_jsonl_path=tmp_path / "Order/runtime/live_orders.jsonl",
        quota_jsonl_path=tmp_path / "Order/runtime/live_quota_snapshots.jsonl",
        settlements_jsonl_path=tmp_path / "Order/runtime/live_settlements.jsonl",
        local_root=tmp_path / "Order",
        upstream_root=tmp_path / "crypto-predict_system",
    )
    now_iso = "2026-11-08T13:05:06+00:00"
    audit.record_order_execution(
        {
            "cohort_id": "2026-11-08T13:05:00+00:00",
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_unknown_1",
            "paper_order_id": "paper:sig_unknown_1",
            "market_id": "512345",
            "condition_id": "0x" + ("11" * 32),
            "token_id": "10001",
            "outcome": "UP",
            "timing": {
                "decision_at": now_iso,
                "paper_fill_at": now_iso,
                "live_detected_at": now_iso,
                "live_signed_at": now_iso,
                "live_http_sent_at": now_iso,
                "live_http_ack_at": now_iso,
                "detection_lag_ms": 10.0,
                "clob_rtt_ms": 1500.0,
            },
            "paper_benchmark": {"approved_stake_usd": 200.0, "paper_filled_quantity": 378.0, "avg_fill_price": 0.51, "cost_usd": 192.78, "fee_usd": 7.22, "total_spent_usd": 200.0},
            "live_execution": live_exec,
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start="2026-11-08T13:05:00+00:00",
        window_end="2026-11-08T13:20:00+00:00",
    )
    quotas = audit.compute_live_quotas(
        now_utc=datetime(2026, 11, 8, 13, 5, 10, tzinfo=timezone.utc),
        cohort_id="2026-11-08T13:05:00+00:00",
        live_scale_factor=0.10,
        paper_caps={"max_order_usd": 200.0, "max_open_risk_usd": 1000.0, "max_underlying_open_risk_usd": 500.0, "max_tick_budget_usd": 1200.0},
        proxy_wallet_address=cands["beacon_deposit_wallet"],
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
    )
    assert quotas["total_locked_open_risk_usdc"] == 20.0
    assert quotas["remaining_total_open_risk_quota_usd"] == 80.0

    # Now reconcile when CLOB confirms the order actually matched 37.82 shares at 0.51
    pending = audit.get_orders_needing_reconciliation()
    assert len(pending) == 1
    clob.client.get_order = lambda oid: {"id": oid, "status": "MATCHED", "size_matched": "37.82", "price": "0.51"}  # type: ignore[method-assign]
    clob.client.get_trades = lambda *a, **kw: []  # type: ignore[method-assign]
    reconciled_exec = clob.reconcile_order(pending[0])
    assert reconciled_exec is not None
    assert reconciled_exec["clob_status"] == "MATCHED"
    assert reconciled_exec["filled_quantity"] == "37.82"


def test_p1_3_empty_response_and_unconfirmed_tx_hash_never_treated_as_filled(monkeypatch: pytest.MonkeyPatch) -> None:
    """[P1-3] Verify that `{}` or missing `takingAmount` is never treated as a full MATCHED fill,
    and a transaction hash without a `status == 0x1` receipt is never marked `onchain_confirmed = True`.
    """
    pk = "0x" + ("34" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    sizing = compute_live_order_sizing(200.0, 0.10, 0.075, "0.01")

    # 1. Empty object `{}` -> ORDER_STATE_UNKNOWN, 0 fill (never $19.9971!)
    parsed_empty = clob._parse_clob_post_response({}, sizing, 0.075, fallback_order_id="0xord1")
    assert parsed_empty["clob_status"] == "ORDER_STATE_UNKNOWN"
    assert parsed_empty["filled_quantity"] == "0.00"
    assert parsed_empty["total_spent_usdc"] == 0.0

    # 2. MATCHED without takingAmount -> ORDER_STATE_UNKNOWN, 0 fill
    parsed_no_taking = clob._parse_clob_post_response(
        {"success": True, "orderID": "0xord2", "status": "MATCHED"},
        sizing,
        0.075,
    )
    assert parsed_no_taking["clob_status"] == "ORDER_STATE_UNKNOWN"
    assert parsed_no_taking["filled_quantity"] == "0.00"
    assert parsed_no_taking["total_spent_usdc"] == 0.0

    # 3. MATCHED with explicit takingAmount + tx_hash, but RPC receipt unavailable -> onchain_confirmed is False
    monkeypatch.setattr(clob, "fetch_tx_receipt_confirmation", lambda txh: (False, None))
    parsed_unconf = clob._parse_clob_post_response(
        {
            "success": True,
            "orderID": "0xord3",
            "status": "MATCHED",
            "takingAmount": "37.82",
            "makingAmount": "19.2882",
            "transactionsHashes": ["0x" + ("aa" * 32)],
        },
        sizing,
        0.075,
    )
    assert parsed_unconf["clob_status"] == "MATCHED"
    assert parsed_unconf["filled_quantity"] == "37.82"
    assert parsed_unconf["onchain_confirmed"] is False


def test_p1_4_redeem_poll_timeout_or_failure_never_credits_funds(tmp_path: Path) -> None:
    """[P1-4] Verify that when Relayer returns only a `transaction_id` and polling fails/times out,
    it is recorded as `REDEEM_PENDING` with `$0.00` credited, keeping the market unsettled for retry.
    """
    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)

    class _FakePendingRelayer:
        def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
            return _RelayerExecResponse(transaction_id="relayer_req_999", transaction_hash=None)

        def poll_until_state(self, **kwargs: Any) -> dict[str, Any] | None:
            return None  # Simulate polling timeout

    redeemer.relayer_client = _FakePendingRelayer()
    live_orders = [{"position_status": "OPEN_ACTIVE", "outcome": "UP", "filled_quantity": "12.00"}]
    res_pending = redeemer.redeem_settled_market(
        market_id="512345",
        condition_id="0x" + ("11" * 32),
        official_outcome="UP",
        live_orders=live_orders,
    )
    assert res_pending["redeem_status"] == "REDEEM_PENDING"
    assert res_pending["redeemed_usdc_amount"] == 0.0
    assert res_pending["live_expected_payout_usdc"] == 12.0

    # Verify AuditStore does NOT mark market settled when redeem_status is REDEEM_PENDING
    audit = AuditStore(
        sqlite_path=tmp_path / "Order/runtime/live_audit.sqlite3",
        orders_jsonl_path=tmp_path / "Order/runtime/live_orders.jsonl",
        quota_jsonl_path=tmp_path / "Order/runtime/live_quota_snapshots.jsonl",
        settlements_jsonl_path=tmp_path / "Order/runtime/live_settlements.jsonl",
        local_root=tmp_path / "Order",
        upstream_root=tmp_path / "crypto-predict_system",
    )
    audit.record_settlement_and_redeem(
        {
            "timestamp_utc": "2026-11-08T13:20:15+00:00",
            "market_id": "512345",
            "condition_id": "0x" + ("11" * 32),
            "official_outcome": "UP",
            "paper_payout_usd": 120.0,
            "paper_realized_pnl_usd": 50.0,
            **res_pending,
        }
    )
    assert audit.is_market_settled("512345") is False


def test_p1_5_deadline_rechecked_immediately_before_http_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """[P1-5] Verify that if detection happens at T+19s (before valid_until=T+20s),
    but local signing/processing finishes at T+21s, the order is aborted BEFORE calling `post_order`.
    """
    pk = "0x" + ("45" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    assert clob.client is not None
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.48", "size": "500"}]}  # type: ignore[method-assign]
    post_called = False

    def _forbid_post(*args: Any, **kwargs: Any) -> Any:
        nonlocal post_called
        post_called = True
        raise AssertionError("post_order must NOT be called when T >= valid_until after signing!")

    clob.client.post_order = _forbid_post  # type: ignore[method-assign]

    # Simulate calibrated clock advancing from T+19s (pre-sign) to T+21s (post-sign, pre-HTTP-send)
    times = iter(
        [
            datetime(2026, 11, 8, 13, 5, 19, 0, tzinfo=timezone.utc),
            datetime(2026, 11, 8, 13, 5, 21, 0, tzinfo=timezone.utc),
            datetime(2026, 11, 8, 13, 5, 21, 0, tzinfo=timezone.utc),
        ]
    )
    import argon4_live.clob_executor as ce_mod

    monkeypatch.setattr(ce_mod, "project_to_calibrated_utc", lambda dt, proj=None: next(times))

    res = clob.sign_and_execute_fak(
        token_id="10001",
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        valid_until="2026-11-08T13:05:20+00:00",
    )
    assert post_called is False
    assert res["live_execution"]["clob_status"] == "ABORTED_PAST_VALID_UNTIL"
    assert res["live_execution"]["filled_quantity"] == "0.00"


def test_p1_6_concurrent_instances_cannot_breach_open_risk_cap_and_single_instance_lock(tmp_path: Path) -> None:
    """[P1-6] Reproduce: max_open_risk = $100, existing open risk = $85.
    Two concurrent reservations for $15 and $7 must NOT reach $107; the first reserves $15 ($85+$15=$100)
    and the second is capped at $0 ($100 total cap respected), plus SingleInstanceFileLock blocks duplicate process locks.
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    audit1 = AuditStore(
        sqlite_path=local_root / "runtime/live/live_audit.sqlite3",
        orders_jsonl_path=local_root / "runtime/live/live_orders.jsonl",
        quota_jsonl_path=local_root / "runtime/live/live_quota_snapshots.jsonl",
        settlements_jsonl_path=local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    audit2 = AuditStore(
        sqlite_path=local_root / "runtime/live/live_audit.sqlite3",
        orders_jsonl_path=local_root / "runtime/live/live_orders.jsonl",
        quota_jsonl_path=local_root / "runtime/live/live_quota_snapshots.jsonl",
        settlements_jsonl_path=local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )

    now_utc = datetime(2026, 11, 8, 13, 5, 5, tzinfo=timezone.utc)
    now_iso = "2026-11-08T13:05:05+00:00"
    paper_caps = {
        "max_order_usd": 200.0,
        "max_open_risk_usd": 1000.0,  # 0.10x scale -> $100.00 live open risk cap
        "max_underlying_open_risk_usd": 5000.0,
        "max_tick_budget_usd": 5000.0,
    }

    # Seed $85.00 of existing open risk
    audit1.record_order_execution(
        {
            "cohort_id": "2026-11-08T13:00:00+00:00",
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_seed_85",
            "paper_order_id": "paper:sig_seed_85",
            "market_id": "500001",
            "condition_id": "0x" + ("01" * 32),
            "token_id": "10001",
            "outcome": "UP",
            "timing": {
                "decision_at": now_iso,
                "paper_fill_at": now_iso,
                "live_detected_at": now_iso,
                "live_signed_at": now_iso,
                "live_http_sent_at": now_iso,
                "live_http_ack_at": now_iso,
                "detection_lag_ms": 5.0,
                "clob_rtt_ms": 20.0,
            },
            "paper_benchmark": {"approved_stake_usd": 850.0, "paper_filled_quantity": 1600.0, "avg_fill_price": 0.51, "cost_usd": 816.0, "fee_usd": 34.0, "total_spent_usd": 850.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.51,
                "requested_stake_usd": 85.0,
                "requested_quantity": "160.00",
                "clob_order_id": "0xseed85",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "160.00",
                "avg_fill_price": 0.51,
                "cost_usdc": 81.6,
                "fee_usdc": 3.4,
                "total_spent_usdc": 85.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 100,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start="2026-11-08T13:00:00+00:00",
        window_end="2026-11-08T13:15:00+00:00",
    )

    # Instance 1 reserves $15 BEFORE submitting to CLOB
    res1 = audit1.claim_and_reserve_signal(
        signal_id="sig_inst_1",
        cohort_id="2026-11-08T13:05:00+00:00",
        stream="ETH_15m",
        rank=1,
        paper_order_id="paper:sig_inst_1",
        market_id="500002",
        condition_id="0x" + ("02" * 32),
        token_id="20001",
        outcome="UP",
        underlying="ETH",
        window_start="2026-11-08T13:05:00+00:00",
        window_end="2026-11-08T13:20:00+00:00",
        claimed_at_utc=now_utc,
        approved_stake_usd=Decimal("150.0"),
        live_scale_factor=Decimal("0.10"),
        paper_caps=paper_caps,
        proxy_wallet_address="0x" + ("44" * 20),
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
        balance_verified=True,
    )
    assert res1["claimed"] is True
    assert res1["reserved_stake_usd"] == Decimal("15.000000")

    # Instance 2 attempts to reserve $7 BEFORE Instance 1 even finishes its HTTP call:
    # Because Instance 1's $15 is atomically reserved in `live_orders` (`RESERVED`),
    # Instance 2 sees $85 + $15 = $100 locked and gets $0.00!
    res2 = audit2.claim_and_reserve_signal(
        signal_id="sig_inst_2",
        cohort_id="2026-11-08T13:05:00+00:00",
        stream="SOL_15m",
        rank=2,
        paper_order_id="paper:sig_inst_2",
        market_id="500003",
        condition_id="0x" + ("03" * 32),
        token_id="30001",
        outcome="UP",
        underlying="SOL",
        window_start="2026-11-08T13:05:00+00:00",
        window_end="2026-11-08T13:20:00+00:00",
        claimed_at_utc=now_utc,
        approved_stake_usd=Decimal("70.0"),
        live_scale_factor=Decimal("0.10"),
        paper_caps=paper_caps,
        proxy_wallet_address="0x" + ("44" * 20),
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
        balance_verified=True,
    )
    assert res2["claimed"] is True
    assert res2["reserved_stake_usd"] == Decimal("0.000000")

    # Verify SingleInstanceFileLock blocks a second instance on the same lock file
    lock_file = local_root / "runtime/live/live_executor.lock"
    lock1 = SingleInstanceFileLock(lock_file, local_root=local_root)
    lock2 = SingleInstanceFileLock(lock_file, local_root=local_root)
    lock1.acquire()
    try:
        with pytest.raises(SafetyViolationError, match="Another ARGON4 Live Executor instance"):
            lock2.acquire()
    finally:
        lock1.release()


def test_p1_7_small_balance_base_unit_conversion_and_balance_allowance_cap() -> None:
    """[P1-7] Reproduce: CLOB returns `"balance": "500000"` ($0.50) and `"allowances": {...}`.
    Verify `"500000"` is converted to `0.5` USD (never `500000.0`!), missing spender allowance defaults
    to `0.0` (never `999999.0`!), and order reservation caps stake by available balance & allowance.
    """
    pk = "0x" + ("56" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    assert clob.client is not None
    clob.client.get_balance_allowance = lambda params=None: {  # type: ignore[method-assign]
        "balance": "500000",  # $0.50 in 6-decimal base units
        "allowances": {
            OFFICIAL_STANDARD_EXCHANGE_V2_ADDRESS: "1200000",  # $1.20
        },
    }
    free_bal, allow_bal = clob.query_wallet_balance_and_allowance(refresh_network=True, market_protocol="v1")
    assert free_bal == 0.5
    assert allow_bal == 1.2
    assert clob.balance_verified is True

    # If allowances map is empty, allowance must be 0.0 (never 999999.0!)
    clob.client.get_balance_allowance = lambda params=None: {  # type: ignore[method-assign]
        "balance": "500000",
        "allowances": {},
    }
    free_bal2, allow_bal2 = clob.query_wallet_balance_and_allowance(refresh_network=True, market_protocol="v1")
    assert free_bal2 == 0.5
    assert allow_bal2 == 0.0


def test_design_gaps_loss_cooldown_mode_isolation_downtime_catchup_min_price_wallet_verify(tmp_path: Path) -> None:
    """Verify all 5 design gaps:
    1) Independent live daily realized loss stop & consecutive-loss cooldown.
    2) Separate shadow vs live audit paths.
    3) Downtime catch-up when settlement ages out of `/api/ledger`.
    4) `min_buy_price = 0.37` enforced before real order submission.
    5) Fail-closed when wallet type cannot be verified in live mode.
    """
    # Gap #2: Mode-isolated audit paths
    audit_cfg = {
        "runtime_dir": "runtime",
        "sqlite_db": "runtime/live_audit.sqlite3",
        "orders_jsonl": "runtime/live_orders.jsonl",
        "quota_snapshots_jsonl": "runtime/live_quota_snapshots.jsonl",
        "settlements_jsonl": "runtime/live_settlements.jsonl",
        "modes": {
            "shadow": {
                "runtime_dir": "runtime/shadow",
                "sqlite_db": "runtime/shadow/shadow_audit.sqlite3",
                "orders_jsonl": "runtime/shadow/shadow_orders.jsonl",
                "quota_snapshots_jsonl": "runtime/shadow/shadow_quota_snapshots.jsonl",
                "settlements_jsonl": "runtime/shadow/shadow_settlements.jsonl",
            },
            "live": {
                "runtime_dir": "runtime/live",
                "sqlite_db": "runtime/live/live_audit.sqlite3",
                "orders_jsonl": "runtime/live/live_orders.jsonl",
                "quota_snapshots_jsonl": "runtime/live/live_quota_snapshots.jsonl",
                "settlements_jsonl": "runtime/live/live_settlements.jsonl",
            },
        },
    }
    assert "shadow" in resolve_mode_audit_paths(audit_cfg, "shadow")["sqlite_db"]
    assert "live" in resolve_mode_audit_paths(audit_cfg, "live")["sqlite_db"]
    assert resolve_mode_audit_paths(audit_cfg, "shadow")["sqlite_db"] != resolve_mode_audit_paths(audit_cfg, "live")["sqlite_db"]

    # Gap #4: min_buy_price = 0.37 enforced when observed ask or paper avg_fill_price < 0.37
    pk = "0x" + ("67" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="shadow",
        private_key=pk,
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    res_low_ask = clob.sign_and_execute_fak(
        token_id="10001",
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        observed_best_ask=0.34,
        min_buy_price=0.37,
    )
    assert res_low_ask["live_execution"]["clob_status"] == "ABORTED_BELOW_MIN_BUY_PRICE"
    assert res_low_ask["live_execution"]["filled_quantity"] == "0.00"

    # Gap #5: Live mode fails closed if wallet type cannot be verified
    with pytest.raises(ValueError, match="Polymarket wallet type could not be verified"):
        ClobExecutor(
            mode="live",
            private_key=pk,
            api_key="k",
            api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
            api_passphrase="p",
            auto_derive_l2_creds=False,
            verify_wallet_on_startup=True,
        )

    # Gap #1: Live daily realized loss stop blocks new orders
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    audit = AuditStore(
        sqlite_path=local_root / "runtime/live/live_audit.sqlite3",
        orders_jsonl_path=local_root / "runtime/live/live_orders.jsonl",
        quota_jsonl_path=local_root / "runtime/live/live_quota_snapshots.jsonl",
        settlements_jsonl_path=local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    now_utc = datetime(2026, 11, 8, 14, 0, 5, tzinfo=timezone.utc)
    now_iso = "2026-11-08T13:30:05+00:00"
    audit.record_order_execution(
        {
            "cohort_id": "2026-11-08T13:30:00+00:00",
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_loss_1",
            "paper_order_id": "paper:sig_loss_1",
            "market_id": "600001",
            "condition_id": "0x" + ("a1" * 32),
            "token_id": "10001",
            "outcome": "UP",
            "timing": {
                "decision_at": now_iso,
                "paper_fill_at": now_iso,
                "live_detected_at": now_iso,
                "live_signed_at": now_iso,
                "live_http_sent_at": now_iso,
                "live_http_ack_at": now_iso,
                "detection_lag_ms": 5.0,
                "clob_rtt_ms": 20.0,
            },
            "paper_benchmark": {"approved_stake_usd": 350.0, "paper_filled_quantity": 660.0, "avg_fill_price": 0.51, "cost_usd": 336.6, "fee_usd": 13.4, "total_spent_usd": 350.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.51,
                "requested_stake_usd": 35.0,
                "requested_quantity": "66.00",
                "clob_order_id": "0xloss1",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "66.00",
                "avg_fill_price": 0.51,
                "cost_usdc": 33.66,
                "fee_usdc": 1.34,
                "total_spent_usdc": 35.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 101,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start="2026-11-08T13:30:00+00:00",
        window_end="2026-11-08T13:45:00+00:00",
    )
    audit.record_settlement_and_redeem(
        {
            "timestamp_utc": "2026-11-08T13:45:10+00:00",
            "market_id": "600001",
            "condition_id": "0x" + ("a1" * 32),
            "official_outcome": "DOWN",
            "paper_payout_usd": 0.0,
            "paper_realized_pnl_usd": -350.0,
            "live_expected_payout_usdc": 0.0,
            "redeem_tx_hash": None,
            "redeem_status": "LOST_NO_REDEEM",
            "redeemed_usdc_amount": 0.0,
            "redeem_latency_seconds": 0.0,
        }
    )
    # Daily realized loss is now -$35.00, exceeding max_daily_realized_loss_usd = $30.00 -> next signal is blocked!
    loss_check = audit.claim_and_reserve_signal(
        signal_id="sig_after_daily_loss",
        cohort_id="2026-11-08T14:00:00+00:00",
        stream="BTC_15m",
        rank=1,
        paper_order_id="paper:sig_after_daily_loss",
        market_id="600002",
        condition_id="0x" + ("a2" * 32),
        token_id="10002",
        outcome="UP",
        underlying="BTC",
        window_start="2026-11-08T14:00:00+00:00",
        window_end="2026-11-08T14:15:00+00:00",
        claimed_at_utc=now_utc,
        approved_stake_usd=Decimal("100.0"),
        live_scale_factor=Decimal("0.10"),
        paper_caps={"max_order_usd": 200.0, "max_open_risk_usd": 1000.0, "max_underlying_open_risk_usd": 500.0, "max_tick_budget_usd": 1200.0},
        proxy_wallet_address=cands["beacon_deposit_wallet"],
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
        balance_verified=True,
        max_daily_realized_loss_usd=Decimal("30.0"),
    )
    assert loss_check["aborted"] is True
    assert loss_check["abort_status"] == "ABORTED_DAILY_LOSS_STOP"


from argon4_live.safety import ROUTER_REDEEM_SELECTOR


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _make_sample_pipeline_state(
    cohort_id: str,
    signal_id: str = "sig_sample_1",
    approved_stake: float = 200.0,
) -> dict[str, Any]:
    cond_id = "0x" + ("11" * 32)
    return {
        "last_cohort": cohort_id,
        "status": "PAPER_CYCLE_COMPLETE",
        "decisions": {
            "BTC_15m": {
                "status": "SIGNAL",
                "outcome": "UP",
                "signal": {
                    "signal_id": signal_id,
                    "stream": "BTC_15m",
                    "underlying": "BTC",
                    "outcome": "UP",
                    "market_id": "512345",
                    "decision_at": cohort_id,
                    "valid_until": "2026-11-08T13:05:20+00:00",
                    "window_start": cohort_id,
                    "window_end": "2026-11-08T13:20:00+00:00",
                },
            }
        },
        "execution": {
            "cohort": {"budget_usd": 1200.0, "remaining_usd": 1000.0},
            "fills": [
                {
                    "order_id": f"paper:{signal_id}",
                    "signal_id": signal_id,
                    "market_id": "512345",
                    "underlying": "BTC",
                    "outcome": "UP",
                    "status": "OPEN",
                    "quantity": "378.12",
                    "cost_usd": 192.84,
                    "fee_usd": 7.16,
                    "unfilled_budget_usd": 0.0,
                    "fill_at": "2026-11-08T13:05:04+00:00",
                    "fee_quote": {"rate": 0.075},
                    "allocation_authorization": {
                        "cohort_id": cohort_id,
                        "rank": 1,
                        "signal_id": signal_id,
                        "market_id": "512345",
                        "approved_stake_usd": approved_stake,
                    },
                    "venue_market": {
                        "market_id": "512345",
                        "condition_id": cond_id,
                        "protocol": "v1",
                        "asset": "BTC",
                        "window_start": cohort_id,
                        "window_end": "2026-11-08T13:20:00+00:00",
                        "outcome_ids": {"UP": "10001", "DOWN": "10002"},
                    },
                }
            ],
        },
    }


def test_round2_p1_a_engine_reconcile_no_keyerror_and_pre_send_crash_recovery(tmp_path: Path) -> None:
    """[Round-2 P1-A] Verify that:
    1) Pre-send callback persists `clob_order_id` (`local_order_hash`) and draft metadata to SQLite
       BEFORE `post_order` is sent over the network.
    2) If the process crashes during `post_order` (before `record_order_execution` runs), restarting
       and calling `engine.reconcile_pending_orders(force=True)` recovers the order without `KeyError('decision_at')`.
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_ locked_policy.json"
    _write_json(policy_path, {"max_order_usd": 200.0, "max_open_risk_usd": 1000.0, "max_underlying_open_risk_usd": 500.0, "max_tick_budget_usd": 1200.0})

    cohort_id = "2026-11-08T13:05:00+00:00"
    _write_json(state_path, _make_sample_pipeline_state(cohort_id, signal_id="sig_crash_mid_send", approved_stake=200.0))

    pk = "0x" + ("71" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    observer = UpstreamObserver(
        pipeline_state_path=state_path,
        policy_path=policy_path,
        http_base_url="http://127.0.0.1:8897",
        upstream_root=upstream_root,
    )
    audit = AuditStore(
        sqlite_path=local_root / "runtime/live/live_audit.sqlite3",
        orders_jsonl_path=local_root / "runtime/live/live_orders.jsonl",
        quota_jsonl_path=local_root / "runtime/live/live_quota_snapshots.jsonl",
        settlements_jsonl_path=local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    clob.balance_verified = True
    clob._cached_free_usdc = 500.0
    clob._cached_allowance_usdc = 500.0

    # Simulate process crash (BaseException) inside post_order AFTER exchange received the order,
    # BEFORE record_order_execution is ever called!
    class _SimulatedPowerLoss(BaseException):
        pass

    assert clob.client is not None
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.48", "size": "500"}]}  # type: ignore[method-assign]
    clob.client.post_order = lambda *a, **kw: (_ for _ in ()).throw(_SimulatedPowerLoss("power cut mid-HTTP"))  # type: ignore[method-assign]

    redeemer = AutoRedeemer(mode="shadow", withdrawal_permission=False)
    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    _, state_data, _ = observer.poll_pipeline_state(force=True)
    assert state_data is not None
    with pytest.raises(_SimulatedPowerLoss):
        engine.process_pipeline_state(
            state_data,
            detected_at_local_utc=datetime(2026, 11, 8, 13, 5, 5, tzinfo=timezone.utc),
        )

    # Simulate restart: open a fresh AuditStore on the same SQLite DB
    audit_restarted = AuditStore(
        sqlite_path=local_root / "runtime/live/live_audit.sqlite3",
        orders_jsonl_path=local_root / "runtime/live/live_orders.jsonl",
        quota_jsonl_path=local_root / "runtime/live/live_quota_snapshots.jsonl",
        settlements_jsonl_path=local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    pending = audit_restarted.get_orders_needing_reconciliation()
    assert len(pending) == 1
    # Verify clob_order_id (local_order_hash) and timing fields were persisted BEFORE post_order!
    persisted_oid = pending[0]["clob_order_id"]
    assert persisted_oid is not None and persisted_oid.startswith("0x")
    assert pending[0]["decision_at"] == "2026-11-08T13:05:00+00:00"

    # Now CLOB reports the order matched 37.82 shares at 0.51
    clob.client.get_order = lambda oid: {"id": oid, "status": "MATCHED", "size_matched": "37.82", "price": "0.51"}  # type: ignore[method-assign]
    clob.client.get_trades = lambda *a, **kw: []  # type: ignore[method-assign]

    engine_restarted = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit_restarted,
        clob_executor=clob,
        redeemer=redeemer,
    )
    reconciled = engine_restarted.reconcile_pending_orders(force=True)
    assert len(reconciled) == 1
    assert reconciled[0]["live_execution"]["clob_status"] == "MATCHED"
    assert reconciled[0]["live_execution"]["filled_quantity"] == "37.82"
    open_orders = audit_restarted.get_open_live_orders("512345")
    assert len(open_orders) == 1
    assert open_orders[0]["position_status"] == "OPEN"


def test_round2_p1_b_unknown_order_never_prematurely_settled_as_lost_and_redeems_after_late_fill(tmp_path: Path) -> None:
    """[Round-2 P1-B] Verify that if a market resolves while an order in that market is still
    `ORDER_STATE_UNKNOWN` / `DELAYED` / `RESERVED`, the market is NOT marked settled as `LOST_NO_REDEEM`.
    When the order later reconciles as a winning fill, it is redeemed and transitions to `WON_REDEEMED`.
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {"max_order_usd": 200.0, "max_open_risk_usd": 1000.0})
    _write_json(state_path, _make_sample_pipeline_state("2026-11-08T13:05:00+00:00", signal_id="sig_late_win", approved_stake=200.0))

    pk = "0x" + ("72" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    observer = UpstreamObserver(
        pipeline_state_path=state_path,
        policy_path=policy_path,
        http_base_url="http://127.0.0.1:8897",
        upstream_root=upstream_root,
    )
    audit = AuditStore(
        sqlite_path=local_root / "runtime/live/live_audit.sqlite3",
        orders_jsonl_path=local_root / "runtime/live/live_orders.jsonl",
        quota_jsonl_path=local_root / "runtime/live/live_quota_snapshots.jsonl",
        settlements_jsonl_path=local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    clob.balance_verified = True
    clob._cached_free_usdc = 500.0
    clob._cached_allowance_usdc = 500.0
    assert clob.client is not None
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.48", "size": "500"}]}  # type: ignore[method-assign]
    # Initial order submission times out -> ORDER_STATE_UNKNOWN
    clob.client.post_order = lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("timeout"))  # type: ignore[method-assign]
    # Reconciliation not yet available on pass 1
    clob.client.get_order = lambda oid: None  # type: ignore[method-assign]
    clob.client.get_trades = lambda *a, **kw: []  # type: ignore[method-assign]

    redeemer = AutoRedeemer(mode="shadow", withdrawal_permission=False)
    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    _, state_data, _ = observer.poll_pipeline_state(force=True)
    assert state_data is not None
    recs = engine.process_pipeline_state(
        state_data,
        detected_at_local_utc=datetime(2026, 11, 8, 13, 5, 5, tzinfo=timezone.utc),
    )
    assert recs[0]["live_execution"]["clob_status"] == "ORDER_STATE_UNKNOWN"

    # Market 512345 resolves to UP while order is still ORDER_STATE_UNKNOWN
    observer.fetch_ledger = lambda allow_in_critical_window=False: [  # type: ignore[method-assign]
        {
            "seq": 10,
            "event_type": "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER",
            "utc": "2026-11-08T13:20:05+00:00",
            "payload": {
                "market_id": "512345",
                "resolution": {"outcome": "UP"},
                "payout_usd": 378.2,
                "realized_pnl_usd": 178.2,
            },
        }
    ]
    observer.fetch_status = lambda allow_in_critical_window=False: None  # type: ignore[method-assign]

    pass1 = engine.poll_settlements_and_redeem(force=True)
    assert len(pass1) == 1
    assert pass1[0]["redeem_status"] == "AWAITING_ORDER_RECONCILIATION"
    assert audit.is_market_settled("512345") is False
    assert len(audit.get_unresolved_live_markets()) == 1

    # Now CLOB confirms the unknown order actually filled 37.82 shares UP!
    clob.client.get_order = lambda oid: {"id": oid, "status": "MATCHED", "size_matched": "37.82", "price": "0.51"}  # type: ignore[method-assign]
    # Even if /api/ledger is now empty, pending settlement + reconciled order triggers full redemption!
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]

    pass2 = engine.poll_settlements_and_redeem(force=True)
    assert len(pass2) == 1
    assert pass2[0]["redeem_status"] == "REDEEMED"
    assert pass2[0]["redeemed_usdc_amount"] == 37.82
    assert audit.is_market_settled("512345") is True
    orders_after = audit.get_live_orders_for_market("512345")
    assert orders_after[0]["position_status"] == "WON_REDEEMED"


def test_round2_p1_c_disputed_closed_market_with_099_price_never_resolved_as_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-2 P1-C] Verify that a closed market in `disputed` / `proposed` status with outcomePrice `0.99`
    is NEVER treated as officially resolved, and only an officially resolved market (`umaResolutionStatus='resolved'`)
    with exact `[1, 0]` payout is accepted.
    """
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    observer = UpstreamObserver(
        pipeline_state_path=state_path,
        policy_path=policy_path,
        http_base_url="http://127.0.0.1:8897",
        upstream_root=upstream_root,
    )
    observer.fetch_status = lambda allow_in_critical_window=False: None  # type: ignore[method-assign]

    class _FakeResp:
        def __init__(self, status_code: int, data: dict[str, Any]) -> None:
            self.status_code = status_code
            self._data = data

        def json(self) -> dict[str, Any]:
            return self._data

    # Case 1: Closed market, still in UMA dispute, UP price is 0.99 -> MUST return None!
    def _mock_get_disputed(url: str, timeout: float = 2.5) -> _FakeResp:
        if "/markets/512345" in url:
            return _FakeResp(
                200,
                {
                    "id": "512345",
                    "closed": True,
                    "umaResolutionStatus": "disputed",
                    "outcomes": '["Up", "Down"]',
                    "outcomePrices": '["0.99", "0.01"]',
                },
            )
        return _FakeResp(404, {})

    import argon4_live.upstream_observer as uo_mod

    monkeypatch.setattr(uo_mod.requests, "get", _mock_get_disputed)
    res_disputed = observer.resolve_unsettled_market_fallback(
        market_id="512345",
        condition_id="0x" + ("11" * 32),
        allow_in_critical_window=True,
    )
    assert res_disputed is None

    # Case 2: Closed market with 0.99 price and NO official resolution status -> MUST return None!
    def _mock_get_unsettled_price(url: str, timeout: float = 2.5) -> _FakeResp:
        if "/markets/512345" in url:
            return _FakeResp(
                200,
                {
                    "id": "512345",
                    "closed": True,
                    "outcomes": '["Up", "Down"]',
                    "outcomePrices": '["0.995", "0.005"]',
                },
            )
        return _FakeResp(404, {})

    monkeypatch.setattr(uo_mod.requests, "get", _mock_get_unsettled_price)
    assert (
        observer.resolve_unsettled_market_fallback(
            market_id="512345",
            condition_id="0x" + ("11" * 32),
            allow_in_critical_window=True,
        )
        is None
    )

    # Case 3: Officially resolved (`umaResolutionStatus == 'resolved'`) with exact settled payout `["0", "1"]` -> DOWN wins!
    def _mock_get_official_resolved(url: str, timeout: float = 2.5) -> _FakeResp:
        if "/markets/512345" in url:
            return _FakeResp(
                200,
                {
                    "id": "512345",
                    "closed": True,
                    "umaResolutionStatus": "resolved",
                    "resolutionStatus": "resolved",
                    "outcomes": '["Up", "Down"]',
                    "outcomePrices": '["0", "1"]',
                    "resolvedAt": "2026-11-08T13:22:00Z",
                },
            )
        return _FakeResp(404, {})

    monkeypatch.setattr(uo_mod.requests, "get", _mock_get_official_resolved)
    res_final = observer.resolve_unsettled_market_fallback(
        market_id="512345",
        condition_id="0x" + ("11" * 32),
        allow_in_critical_window=True,
    )
    assert res_final is not None
    assert res_final["official_outcome"] == "DOWN"


def test_round2_p1_d_deposit_wallet_relayer_payload_and_v2_negrisk_engine_redemption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-2 P1-D] Verify that:
    1) Deposit Wallet (`signature_type=3`, `type='WALLET'`) Relayer submission includes
       `depositWalletParams`, wrapped signature, and `to = deposit_wallet_factory`.
    2) `engine._execute_and_record_settlement` passes `market_protocol` and `neg_risk` so a V2 market
       with a 32-byte `condition_id` (`0x...00`) is redeemed via `OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS`
       (`Router.redeem`), NOT the V1 CTF adapter.
    """
    from polymarket._internal.environment import PRODUCTION_CONFIG
    import argon4_live.redeemer as red_mod

    pk = "0x" + ("74" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    redeemer = AutoRedeemer(
        mode="live",
        private_key=pk,
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        relayer_api_key="rk_test_123",
        relayer_api_secret=cands["signer_eoa"],
        withdrawal_permission=False,
    )
    assert redeemer.relayer_client is not None

    submitted_payloads: list[dict[str, Any]] = []

    class _MockHttpResp:
        def __init__(self, status_code: int, body: dict[str, Any]) -> None:
            self.status_code = status_code
            self._body = body

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> dict[str, Any]:
            return self._body

    def _mock_relayer_get(url: str, params: Any = None, headers: Any = None, timeout: float = 5.0) -> _MockHttpResp:
        if "/nonce" in url:
            return _MockHttpResp(200, {"nonce": "42"})
        if "/transaction" in url:
            return _MockHttpResp(200, {"state": "STATE_CONFIRMED", "transactionHash": "0x" + ("cd" * 32)})
        return _MockHttpResp(404, {})

    def _mock_relayer_post(url: str, json: Any = None, headers: Any = None, timeout: float = 8.0) -> _MockHttpResp:
        if "/submit" in url:
            submitted_payloads.append(dict(json))
            return _MockHttpResp(
                200,
                {
                    "transactionID": "relayer_tx_v2_1",
                    "transactionHash": "0x" + ("cd" * 32),
                    "state": "STATE_CONFIRMED",
                },
            )
        if isinstance(json, dict) and json.get("method") == "eth_getTransactionReceipt":
            return _MockHttpResp(200, {"result": {"blockNumber": "0x100", "status": "0x1"}})
        return _MockHttpResp(404, {})

    monkeypatch.setattr(red_mod.requests, "get", _mock_relayer_get)
    monkeypatch.setattr(red_mod.requests, "post", _mock_relayer_post)

    # Set up Engine + AuditStore with a V2 market whose condition_id is 64 hex chars ending in 00
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    observer = UpstreamObserver(
        pipeline_state_path=state_path,
        policy_path=policy_path,
        http_base_url="http://127.0.0.1:8897",
        upstream_root=upstream_root,
    )
    observer.fetch_status = lambda allow_in_critical_window=False: None  # type: ignore[method-assign]
    audit = AuditStore(
        sqlite_path=local_root / "runtime/live/live_audit.sqlite3",
        orders_jsonl_path=local_root / "runtime/live/live_orders.jsonl",
        quota_jsonl_path=local_root / "runtime/live/live_quota_snapshots.jsonl",
        settlements_jsonl_path=local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(
        mode="shadow",
        private_key=pk,
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    v2_cond_32byte = "0x" + ("ab" * 31) + "00"
    now_iso = "2026-11-08T13:05:05+00:00"
    audit.record_order_execution(
        {
            "cohort_id": "2026-11-08T13:05:00+00:00",
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_v2_redeem_1",
            "paper_order_id": "paper:sig_v2_redeem_1",
            "market_id": "777001",
            "condition_id": v2_cond_32byte,
            "token_id": "0x02" + ("ab" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {
                "decision_at": now_iso,
                "paper_fill_at": now_iso,
                "live_detected_at": now_iso,
                "live_signed_at": now_iso,
                "live_http_sent_at": now_iso,
                "live_http_ack_at": now_iso,
                "detection_lag_ms": 5.0,
                "clob_rtt_ms": 15.0,
            },
            "paper_benchmark": {"approved_stake_usd": 200.0, "paper_filled_quantity": 250.0, "avg_fill_price": 0.50, "cost_usd": 125.0, "fee_usd": 5.0, "total_spent_usd": 130.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.51,
                "requested_stake_usd": 20.0,
                "requested_quantity": "25.00",
                "clob_order_id": "0xv2ord1",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "25.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 12.50,
                "fee_usdc": 0.50,
                "total_spent_usdc": 13.00,
                "unfilled_stake_usdc": 7.00,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 100,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start="2026-11-08T13:05:00+00:00",
        window_end="2026-11-08T13:20:00+00:00",
    )

    settlement = engine._execute_and_record_settlement(
        market_id="777001",
        condition_id=v2_cond_32byte,
        official_outcome="UP",
        paper_payout_usd=250.0,
        paper_realized_pnl_usd=120.0,
        paper_settled_at="2026-11-08T13:20:05+00:00",
    )
    assert settlement["redeem_status"] == "REDEEMED"
    assert settlement["redeemed_usdc_amount"] == 25.0
    assert len(submitted_payloads) == 1

    sent = submitted_payloads[0]
    assert sent["type"] == "WALLET"
    assert sent["to"].lower() == str(PRODUCTION_CONFIG.wallet_derivation.deposit_wallet_factory).lower()
    assert "depositWalletParams" in sent
    dw_params = sent["depositWalletParams"]
    assert dw_params["depositWallet"].lower() == cands["beacon_deposit_wallet"].lower()
    assert len(dw_params["calls"]) == 1
    # Verify V2 market was routed to OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS (Router.redeem 0xd217a3cc), NOT V1 CTF adapter!
    call0 = dw_params["calls"][0]
    assert call0["target"].lower() == OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS.lower()
    assert call0["data"].startswith(ROUTER_REDEEM_SELECTOR)


def test_round2_p1_e_state_new_with_tx_hash_and_missing_receipt_never_credits_and_polls_existing_tx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-2 P1-E] Verify that:
    1) `STATE_NEW + tx_hash + missing on-chain receipt` returns `REDEEM_PENDING` with `$0.00` credited
       (never `REDEEMED` or `$12.00`!).
    2) Subsequent settlement passes poll/verify the existing pending redemption transaction FIRST instead
       of submitting duplicate transactions via `execute()`.
    """
    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)
    assert redeemer.require_onchain_receipt is True

    execute_call_count = 0
    poll_call_count = 0
    receipt_available = False

    class _FakeRelayer:
        def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
            nonlocal execute_call_count
            execute_call_count += 1
            return _RelayerExecResponse(
                transaction_id="relayer_tx_555",
                transaction_hash="0x" + ("ee" * 32),
                state="STATE_NEW",
            )

        def poll_until_state(self, **kwargs: Any) -> dict[str, Any] | None:
            nonlocal poll_call_count
            poll_call_count += 1
            if not receipt_available:
                return {"state": "STATE_NEW", "transactionHash": "0x" + ("ee" * 32)}
            return {"state": "STATE_CONFIRMED", "transactionHash": "0x" + ("ee" * 32)}

    redeemer.relayer_client = _FakeRelayer()
    monkeypatch.setattr(
        redeemer,
        "_verify_onchain_receipt_status",
        lambda txh: (True, 123456) if receipt_available else (None, None),
    )

    live_orders = [{"position_status": "OPEN", "outcome": "UP", "filled_quantity": "12.00"}]

    # Pass 1: STATE_NEW + tx_hash + receipt=None -> MUST be REDEEM_PENDING and $0.00 credited!
    res1 = redeemer.redeem_settled_market(
        market_id="888001",
        condition_id="0x" + ("22" * 32),
        official_outcome="UP",
        live_orders=live_orders,
    )
    assert res1["redeem_status"] == "REDEEM_PENDING"
    assert res1["redeemed_usdc_amount"] == 0.0
    assert execute_call_count == 1
    assert poll_call_count == 1

    # Pass 2: Pass existing_settlement while receipt becomes available -> polls existing tx WITHOUT calling execute() again!
    receipt_available = True
    res2 = redeemer.redeem_settled_market(
        market_id="888001",
        condition_id="0x" + ("22" * 32),
        official_outcome="UP",
        live_orders=live_orders,
        existing_settlement=res1,
    )
    assert res2["redeem_status"] == "REDEEMED"
    assert res2["redeemed_usdc_amount"] == 12.0
    assert execute_call_count == 1  # execute() was NOT called a second time!
    assert poll_call_count == 2


def test_round2_p1_f_and_remaining_fixes_stop_script_live_orderbook_min_price_scale_priority_and_spender_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-2 P1-F & Issues #7, #8, #9] Verify:
    1) [P1-F] `stop_live_executor.ps1` and `start_live_executor.ps1` check `runtime\\live\\live_service.json`
       and `runtime\\shadow\\live_service.json`.
    2) [Issue #7] `min_buy_price = 0.37` queries live orderbook (`get_order_book`) and aborts when real-time
       best ask (`0.31`) is below `0.37` even if paper fill price was `0.45`.
    3) [Issue #8] Explicit `live_scale_factor` in config or `scale_override` takes priority over `ARGON4_LIVE_SCALE_FACTOR`.
    4) [Issue #9] `query_wallet_balance_and_allowance` caches allowances per exchange contract and never uses
       Standard Exchange allowance when V2 Exchange V3 allowance is `$0`.
    """
    # 1. [P1-F] Check PowerShell stop/start scripts include runtime\live\live_service.json
    stop_ps1 = Path(r"C:\Users\liuqi\Order\scripts\stop_live_executor.ps1").read_text(encoding="utf-8")
    start_ps1 = Path(r"C:\Users\liuqi\Order\scripts\start_live_executor.ps1").read_text(encoding="utf-8")
    assert "runtime\\live\\live_service.json" in stop_ps1
    assert "runtime\\shadow\\live_service.json" in stop_ps1
    assert "live\\live_service.json" in start_ps1

    # 2. [Issue #7] Real-time CLOB orderbook best_ask (0.31) < min_buy_price (0.37) aborts BEFORE post_order
    pk = "0x" + ("75" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    assert clob.client is not None
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.31", "size": "500"}, {"price": "0.35", "size": "200"}]}  # type: ignore[method-assign]
    post_called = False

    def _never_post(*a: Any, **kw: Any) -> Any:
        nonlocal post_called
        post_called = True
        return {}

    clob.client.post_order = _never_post  # type: ignore[method-assign]
    res_book = clob.sign_and_execute_fak(
        token_id="10001",
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        paper_benchmark={"avg_fill_price": 0.45},  # Paper fill was 0.45, but live orderbook crashed to 0.31!
        observed_best_ask=None,
        min_buy_price=0.37,
    )
    assert post_called is False
    assert res_book["live_execution"]["clob_status"] == "ABORTED_BELOW_MIN_BUY_PRICE"
    assert "LIVE_ORDERBOOK_BEST_ASK_BELOW_MIN_BUY_PRICE" in str(res_book["live_execution"]["error_code"])

    # 3. [Issue #8] Explicit config / scale_override takes priority over ARGON4_LIVE_SCALE_FACTOR env var
    monkeypatch.setenv("ARGON4_LIVE_SCALE_FACTOR", "1.0")
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})
    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    redeemer = AutoRedeemer(mode="shadow", withdrawal_permission=False)
    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.01}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )
    assert engine.live_scale_factor == 0.01

    # 4. [Issue #9] Per-spender allowance cache: Standard Exchange has $100 allowance, V2 Exchange V3 is absent ($0)
    from argon4_live.safety import OFFICIAL_EXCHANGE_V3_ADDRESS, OFFICIAL_STANDARD_EXCHANGE_V2_ADDRESS

    clob.client.get_balance_allowance = lambda params=None: {  # type: ignore[method-assign]
        "balance": "100000000",  # $100.00
        "allowances": {
            OFFICIAL_STANDARD_EXCHANGE_V2_ADDRESS: "100000000",  # $100.00 on V1 Standard Exchange only
        },
    }
    bal_v1, all_v1 = clob.query_wallet_balance_and_allowance(refresh_network=True, market_protocol="v1", neg_risk=False)
    assert bal_v1 == 100.0
    assert all_v1 == 100.0

    # Now query cached allowance (refresh_network=False) for a V2 market -> MUST be 0.0 (never $100.0)!
    bal_v2_cached, all_v2_cached = clob.query_wallet_balance_and_allowance(
        refresh_network=False, market_protocol="v2", neg_risk=False
    )
    assert bal_v2_cached == 100.0
    assert all_v2_cached == 0.0

    # And refreshing network for V2 market when OFFICIAL_EXCHANGE_V3_ADDRESS is absent from allowances -> MUST be 0.0!
    bal_v2_net, all_v2_net = clob.query_wallet_balance_and_allowance(
        refresh_network=True, market_protocol="v2", neg_risk=False
    )
    assert bal_v2_net == 100.0
    assert all_v2_net == 0.0


def test_round3_p1_1_redemption_amount_bound_to_submitted_batch_and_new_fills_redeemed_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-3 P1-1] Reproduce:
    1) Batch 1: Order 1 (12 shares UP, `OPEN`) is submitted for redemption (`tx_1`), returning `REDEEM_PENDING`.
    2) While `tx_1` is pending, Order 2 (8 shares UP, `ORDER_STATE_UNKNOWN`) reconciles to `OPEN`.
    3) Next poll checks `tx_1` (now `STATE_CONFIRMED` + receipt `0x1`):
       - Must credit ONLY `$12.00` (bound to `tx_1`), NEVER `$20.00`!
       - Must mark ONLY Order 1 as `WON_REDEEMED`; Order 2 remains `WON_UNREDEEMED` and market stays unsettled.
    4) Subsequent poll skips already-redeemed Order 1 (`WON_REDEEMED`) and submits `tx_2` strictly for
       Order 2 (8 shares = `$8.00`), crediting `$8.00` upon confirmation (total `$12.00 + $8.00 = $20.00`).
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    pk = "0x" + ("81" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.fetch_status = lambda allow_in_critical_window=False: None  # type: ignore[method-assign]
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]

    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    clob.balance_verified = True
    clob._cached_free_usdc = 100.0
    clob._cached_allowance_usdc = 100.0
    assert clob.client is not None

    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)
    submitted_calldatas: list[str] = []
    tx1_confirmed = False

    class _BatchFakeRelayer:
        def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
            call0 = transactions[0]
            data_hex = str(getattr(call0, "data", ""))
            submitted_calldatas.append(data_hex)
            idx = len(submitted_calldatas)
            return _RelayerExecResponse(
                transaction_id=f"relayer_batch_tx_{idx}",
                transaction_hash=f"0x{idx:064x}",
                state="STATE_NEW" if idx == 1 else "STATE_CONFIRMED",
            )

        def poll_until_state(self, transaction_id: str, **kwargs: Any) -> dict[str, Any] | None:
            if transaction_id == "relayer_batch_tx_1":
                if not tx1_confirmed:
                    return {"state": "STATE_NEW", "transactionHash": f"0x{1:064x}"}
                return {"state": "STATE_CONFIRMED", "transactionHash": f"0x{1:064x}"}
            return {"state": "STATE_CONFIRMED", "transactionHash": f"0x{2:064x}"}

    redeemer.relayer_client = _BatchFakeRelayer()
    monkeypatch.setattr(redeemer, "_verify_onchain_receipt_status", lambda txh: (True, 200))

    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    now_iso = "2026-11-08T13:05:05+00:00"
    v2_cond = "0x" + ("ab" * 31) + "00"
    # Seed Order 1: 12 shares UP (OPEN)
    audit.record_order_execution(
        {
            "cohort_id": "2026-11-08T13:05:00+00:00",
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_batch_12",
            "paper_order_id": "paper:sig_batch_12",
            "market_id": "900001",
            "condition_id": v2_cond,
            "token_id": "0x02" + ("ab" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 60.0, "paper_filled_quantity": 120.0, "avg_fill_price": 0.50, "cost_usd": 60.0, "fee_usd": 0.0, "total_spent_usd": 60.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.51,
                "requested_stake_usd": 6.0,
                "requested_quantity": "12.00",
                "clob_order_id": "0xord_batch_12",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "12.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 6.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 6.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 100,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start="2026-11-08T13:05:00+00:00",
        window_end="2026-11-08T13:20:00+00:00",
    )
    # Seed Order 2: 8 shares UP (initially ORDER_STATE_UNKNOWN)
    audit.record_order_execution(
        {
            "cohort_id": "2026-11-08T13:05:00+00:00",
            "stream": "BTC_15m",
            "rank": 2,
            "signal_id": "sig_batch_8",
            "paper_order_id": "paper:sig_batch_8",
            "market_id": "900001",
            "condition_id": v2_cond,
            "token_id": "0x02" + ("ab" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 40.0, "paper_filled_quantity": 80.0, "avg_fill_price": 0.50, "cost_usd": 40.0, "fee_usd": 0.0, "total_spent_usd": 40.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.51,
                "requested_stake_usd": 4.0,
                "requested_quantity": "8.00",
                "clob_order_id": "0xord_batch_8",
                "clob_status": "ORDER_STATE_UNKNOWN",
                "error_code": "TIMEOUT",
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": 4.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": False,
                "onchain_block_number": None,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start="2026-11-08T13:05:00+00:00",
        window_end="2026-11-08T13:20:00+00:00",
    )
    clob.client.get_order = lambda oid: None  # type: ignore[method-assign]
    clob.client.get_trades = lambda *a, **kw: []  # type: ignore[method-assign]

    # Pass 1: Market settles to UP -> submits tx_1 for Order 1 (12 shares), stays REDEEM_PENDING ($0 credited)
    s1 = engine._execute_and_record_settlement(
        market_id="900001",
        condition_id=v2_cond,
        official_outcome="UP",
        paper_payout_usd=200.0,
        paper_realized_pnl_usd=100.0,
        paper_settled_at="2026-11-08T13:20:05+00:00",
    )
    assert s1["redeem_status"] == "REDEEM_PENDING"
    assert s1["redeemed_usdc_amount"] == 0.0
    assert clob._cached_free_usdc == 100.0
    assert len(submitted_calldatas) == 1

    # Now Order 2 (8 shares) reconciles as MATCHED, and tx_1 (12 shares) confirms on-chain!
    clob.client.get_order = lambda oid: {"id": oid, "status": "MATCHED", "size_matched": "8.00", "price": "0.50"}  # type: ignore[method-assign]
    tx1_confirmed = True

    pass2 = engine.poll_settlements_and_redeem(force=True)
    assert len(pass2) == 1
    # MUST credit ONLY $12.00 for tx_1 (NEVER $20.00!), and keep market in REDEEM_PENDING for Order 2!
    # Also deducts $4.00 for newly reconciled Order 2 ([P2-R6-2]): $100 - $4 + $12 = $108.00!
    assert pass2[0]["redeemed_usdc_amount"] == 12.0
    assert clob._cached_free_usdc == 108.0
    assert len(submitted_calldatas) == 1  # Only polled tx_1 on this pass!
    assert audit.is_market_settled("900001") is False

    orders_mid = {o["signal_id"]: o["position_status"] for o in audit.get_live_orders_for_market("900001")}
    assert orders_mid["sig_batch_12"] == "WON_REDEEMED"
    assert orders_mid["sig_batch_8"] == "WON_UNREDEEMED"

    # Pass 3: Next settlement poll skips `sig_batch_12` (WON_REDEEMED) and submits `tx_2` strictly for `sig_batch_8` (8 shares = $8.00)!
    pass3 = engine.poll_settlements_and_redeem(force=True)
    assert len(pass3) == 1
    assert len(submitted_calldatas) == 2
    # Verify V2 Router calldata for tx_2 encodes 8,000,000 base units (0x7a1200), NOT 20,000,000!
    expected_tx2_calldata = encode_v2_router_redeem_calldata(
        condition_id=v2_cond,
        outcome_index=0,
        amount_base_units=8_000_000,
    )
    assert submitted_calldatas[1] == expected_tx2_calldata
    assert pass3[0]["redeem_status"] == "REDEEMED"
    assert pass3[0]["redeemed_usdc_amount"] == 20.0  # Cumulative across Batch 1 ($12) + Batch 2 ($8)
    assert clob._cached_free_usdc == 116.0  # $100 - $4 (Order 2 fill) + $12 + $8 = $116 (never double-counted!)
    assert audit.is_market_settled("900001") is True
    orders_final = {o["signal_id"]: o["position_status"] for o in audit.get_live_orders_for_market("900001")}
    assert orders_final["sig_batch_12"] == "WON_REDEEMED"
    assert orders_final["sig_batch_8"] == "WON_REDEEMED"


def test_round3_p1_2_unknown_orders_never_mask_confirmed_losses_or_bypass_loss_stops(tmp_path: Path) -> None:
    """[Round-3 P1-2] Reproduce:
    4 markets each have a confirmed `$20.00` losing order (`DOWN` when `UP` wins) AND a `$1.00` `ORDER_STATE_UNKNOWN` order.
    Even though each market has an unknown order (`AWAITING_ORDER_RECONCILIATION`), the `$80.00` of confirmed losses
    and 4 consecutive losses MUST be accounted for immediately so a 5th order of `$20.00` is blocked by
    both `ABORTED_DAILY_LOSS_STOP` and `ABORTED_CONSECUTIVE_LOSS_COOLDOWN`.
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    now_iso = "2026-11-08T13:05:05+00:00"
    settle_iso = "2026-11-08T13:20:05+00:00"

    for i in range(1, 5):
        mid = f"91000{i}"
        cond = f"0x{i:064x}"
        # 1. Confirmed losing order ($20.00 on DOWN)
        audit.record_order_execution(
            {
                "cohort_id": "2026-11-08T13:05:00+00:00",
                "stream": "BTC_15m",
                "rank": 1,
                "signal_id": f"sig_lost_20_{i}",
                "paper_order_id": f"paper:sig_lost_20_{i}",
                "market_id": mid,
                "condition_id": cond,
                "token_id": "10002",
                "outcome": "DOWN",
                "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
                "paper_benchmark": {"approved_stake_usd": 200.0, "paper_filled_quantity": 400.0, "avg_fill_price": 0.50, "cost_usd": 200.0, "fee_usd": 0.0, "total_spent_usd": 200.0},
                "live_execution": {
                    "live_scale_factor": 0.10,
                    "order_type": "FAK",
                    "limit_price": 0.51,
                    "requested_stake_usd": 20.0,
                    "requested_quantity": "40.00",
                    "clob_order_id": f"0xlost_{i}",
                    "clob_status": "MATCHED",
                    "error_code": None,
                    "filled_quantity": "40.00",
                    "avg_fill_price": 0.50,
                    "cost_usdc": 20.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 20.0,
                    "unfilled_stake_usdc": 0.0,
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": True,
                    "onchain_block_number": 100 + i,
                },
                "diff_vs_paper": {},
            },
            underlying="BTC",
            window_start="2026-11-08T13:05:00+00:00",
            window_end="2026-11-08T13:20:00+00:00",
        )
        # 2. Residual $1.00 ORDER_STATE_UNKNOWN order in the same market
        audit.record_order_execution(
            {
                "cohort_id": "2026-11-08T13:05:00+00:00",
                "stream": "BTC_15m",
                "rank": 2,
                "signal_id": f"sig_unk_1_{i}",
                "paper_order_id": f"paper:sig_unk_1_{i}",
                "market_id": mid,
                "condition_id": cond,
                "token_id": "10002",
                "outcome": "DOWN",
                "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
                "paper_benchmark": {"approved_stake_usd": 10.0, "paper_filled_quantity": 20.0, "avg_fill_price": 0.50, "cost_usd": 10.0, "fee_usd": 0.0, "total_spent_usd": 10.0},
                "live_execution": {
                    "live_scale_factor": 0.10,
                    "order_type": "FAK",
                    "limit_price": 0.51,
                    "requested_stake_usd": 1.0,
                    "requested_quantity": "2.00",
                    "clob_order_id": f"0xunk_{i}",
                    "clob_status": "ORDER_STATE_UNKNOWN",
                    "error_code": "TIMEOUT",
                    "filled_quantity": "0.00",
                    "avg_fill_price": 0.0,
                    "cost_usdc": 0.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 0.0,
                    "unfilled_stake_usdc": 1.0,
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                },
                "diff_vs_paper": {},
            },
            underlying="BTC",
            window_start="2026-11-08T13:05:00+00:00",
            window_end="2026-11-08T13:20:00+00:00",
        )
        # Market resolves to UP -> $20 order is LOST, $1 order is AWAITING_ORDER_RECONCILIATION
        audit.record_settlement_and_redeem(
            {
                "timestamp_utc": settle_iso,
                "market_id": mid,
                "condition_id": cond,
                "official_outcome": "UP",
                "paper_payout_usd": 0.0,
                "paper_realized_pnl_usd": -200.0,
                "live_expected_payout_usdc": 0.0,
                "redeem_tx_hash": None,
                "redeem_status": "AWAITING_ORDER_RECONCILIATION",
                "redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": 0.0,
            }
        )
        settle_rec = audit.get_settlement_record(mid)
        assert settle_rec is not None
        assert settle_rec["redeem_status"] == "AWAITING_ORDER_RECONCILIATION"
        assert settle_rec["live_realized_pnl_usdc"] == -20.0

    # Attempt to buy $20 in a 5th market:
    # 1) Blocked by daily loss stop (-$80.00 <= -$30.00)
    res_daily = audit.claim_and_reserve_signal(
        signal_id="sig_5th_market_daily",
        cohort_id="2026-11-08T13:25:00+00:00",
        stream="BTC_15m",
        rank=1,
        paper_order_id="paper:sig_5th_market_daily",
        market_id="910005",
        condition_id="0x" + ("05" * 32),
        token_id="10001",
        outcome="UP",
        underlying="BTC",
        window_start="2026-11-08T13:25:00+00:00",
        window_end="2026-11-08T13:40:00+00:00",
        claimed_at_utc=datetime(2026, 11, 8, 13, 25, 5, tzinfo=timezone.utc),
        approved_stake_usd=Decimal("200.0"),
        live_scale_factor=Decimal("0.10"),
        paper_caps={"max_order_usd": 200.0, "max_open_risk_usd": 1000.0, "max_underlying_open_risk_usd": 500.0, "max_tick_budget_usd": 1200.0},
        proxy_wallet_address="0x" + ("44" * 20),
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
        balance_verified=True,
        max_daily_realized_loss_usd=Decimal("30.0"),
        max_consecutive_losses=4,
        cooldown_minutes_after_consecutive_losses=30.0,
    )
    assert res_daily["aborted"] is True
    assert res_daily["abort_status"] == "ABORTED_DAILY_LOSS_STOP"
    assert res_daily["loss_stats"]["daily_realized_pnl_usdc"] == -80.0
    assert res_daily["loss_stats"]["consecutive_losses"] == 4

    # 2) Even if daily loss limit were $200.0, blocked by 4 consecutive losses cooldown!
    res_streak = audit.claim_and_reserve_signal(
        signal_id="sig_5th_market_streak",
        cohort_id="2026-11-08T13:25:00+00:00",
        stream="BTC_15m",
        rank=1,
        paper_order_id="paper:sig_5th_market_streak",
        market_id="910006",
        condition_id="0x" + ("06" * 32),
        token_id="10001",
        outcome="UP",
        underlying="BTC",
        window_start="2026-11-08T13:25:00+00:00",
        window_end="2026-11-08T13:40:00+00:00",
        claimed_at_utc=datetime(2026, 11, 8, 13, 25, 5, tzinfo=timezone.utc),
        approved_stake_usd=Decimal("200.0"),
        live_scale_factor=Decimal("0.10"),
        paper_caps={"max_order_usd": 200.0, "max_open_risk_usd": 1000.0, "max_underlying_open_risk_usd": 500.0, "max_tick_budget_usd": 1200.0},
        proxy_wallet_address="0x" + ("44" * 20),
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
        balance_verified=True,
        max_daily_realized_loss_usd=Decimal("200.0"),
        max_consecutive_losses=4,
        cooldown_minutes_after_consecutive_losses=30.0,
    )
    assert res_streak["aborted"] is True
    assert res_streak["abort_status"] == "ABORTED_CONSECUTIVE_LOSS_COOLDOWN"


def test_round3_p1_3_deadline_rechecked_after_pre_send_sqlite_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """[Round-3 P1-3] Reproduce:
    1 second remains before `valid_until` (`T+19s` vs `T+20s`) when local signing finishes,
    but `pre_send_callback` (SQLite persistence) takes 2 seconds (`T+21s`).
    Verify the final deadline check and `live_http_sent_at` timestamp occur AFTER `pre_send_callback`,
    aborting the order BEFORE `post_order` is ever called!
    """
    pk = "0x" + ("83" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    assert clob.client is not None
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.48", "size": "500"}]}  # type: ignore[method-assign]

    post_called = False

    def _forbid_post(*a: Any, **kw: Any) -> Any:
        nonlocal post_called
        post_called = True
        raise AssertionError("post_order must NOT be called when pre_send_callback exceeds valid_until!")

    clob.client.post_order = _forbid_post  # type: ignore[method-assign]

    current_time = datetime(2026, 11, 8, 13, 5, 19, 0, tzinfo=timezone.utc)
    import argon4_live.clob_executor as ce_mod

    monkeypatch.setattr(ce_mod, "project_to_calibrated_utc", lambda dt, proj=None: current_time)

    callback_ran = False

    def _slow_pre_send_callback(*a: Any, **kw: Any) -> None:
        nonlocal current_time, callback_ran
        callback_ran = True
        # Simulate SQLite write taking 2 seconds (clock advances from T+19s to T+21s)
        current_time = datetime(2026, 11, 8, 13, 5, 21, 0, tzinfo=timezone.utc)

    res = clob.sign_and_execute_fak(
        token_id="10001",
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        valid_until="2026-11-08T13:05:20+00:00",
        pre_send_callback=_slow_pre_send_callback,
    )
    assert callback_ran is True
    assert post_called is False
    assert res["live_execution"]["clob_status"] == "ABORTED_PAST_VALID_UNTIL"
    assert res["live_http_sent_at"] == "2026-11-08T13:05:21+00:00"


def test_round3_p1_4_live_order_aborts_when_orderbook_fetch_fails_or_is_stale() -> None:
    """[Round-3 P1-4] Reproduce:
    1) Real-time orderbook queries (`get_order_book` & `get_price`) all time out, paper benchmark has a
       30-minute-old `0.40` price, and `post_order` would return a `0.20` fill -> MUST abort with
       `ABORTED_ORDERBOOK_UNAVAILABLE` BEFORE `post_order` is ever called!
    2) Real-time orderbook returns a stale book timestamp (or `book_observed_at` is 30 minutes old) ->
       MUST abort with `ABORTED_STALE_ORDERBOOK` BEFORE `post_order` is ever called!
    """
    pk = "0x" + ("84" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    assert clob.client is not None

    post_called = False

    def _simulated_020_fill(*a: Any, **kw: Any) -> dict[str, Any]:
        nonlocal post_called
        post_called = True
        return {
            "success": True,
            "orderID": "0xbad_fill",
            "status": "MATCHED",
            "takingAmount": "37.82",
            "makingAmount": "7.564",  # 0.20 fill price!
        }

    clob.client.post_order = _simulated_020_fill  # type: ignore[method-assign]
    clob.client.get_order_book = lambda tid: (_ for _ in ()).throw(TimeoutError("orderbook timeout"))  # type: ignore[method-assign]
    clob.client.get_price = lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("price timeout"))  # type: ignore[method-assign]

    now_utc = datetime(2026, 11, 8, 13, 35, 5, tzinfo=timezone.utc)
    # 1. Real-time orderbook timeout + paper price 0.40 -> MUST abort (never send or accept 0.20 fill!)
    res_timeout = clob.sign_and_execute_fak(
        token_id="10001",
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        paper_benchmark={"avg_fill_price": 0.40},
        reference_now_utc=now_utc,
    )
    assert post_called is False
    assert res_timeout["live_execution"]["clob_status"] == "ABORTED_ORDERBOOK_UNAVAILABLE"
    assert res_timeout["live_execution"]["filled_quantity"] == "0.00"

    # 2. Stale `book_observed_at` (30 minutes old) -> MUST abort with ABORTED_STALE_ORDERBOOK!
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.48", "size": "500"}]}  # type: ignore[method-assign]
    res_stale_obs = clob.sign_and_execute_fak(
        token_id="10001",
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        observed_best_ask=0.40,
        book_observed_at="2026-11-08T13:05:05+00:00",  # 30 minutes before 13:35:05!
        reference_now_utc=now_utc,
    )
    assert post_called is False
    assert res_stale_obs["live_execution"]["clob_status"] == "ABORTED_STALE_ORDERBOOK"

    # 3. Stale `timestamp` inside live orderbook response -> MUST abort with ABORTED_STALE_ORDERBOOK!
    clob.client.get_order_book = lambda tid: {  # type: ignore[method-assign]
        "timestamp": "2026-11-08T13:05:05+00:00",
        "asks": [{"price": "0.48", "size": "500"}],
    }
    res_stale_book = clob.sign_and_execute_fak(
        token_id="10001",
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.075,
        reference_now_utc=now_utc,
    )
    assert post_called is False
    assert res_stale_book["live_execution"]["clob_status"] == "ABORTED_STALE_ORDERBOOK"


def test_round3_p2_5_fallback_relayer_v1_params_and_snake_case_transaction_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[Round-3 P2-5] Verify that when `/nonce` and `/transaction` return 404 and fall back to
    `/v1/account/transactions/params` and `/v1/account/transactions/{id}`:
      1) `/v1/account/transactions/params` sends `params={'address': ..., 'type': ...}` (NOT `signerAddress`/`txType`).
      2) `/v1/account/transactions/{id}` response with snake_case `transaction_hash` is recognized and confirmed.
    """
    import argon4_live.redeemer as red_mod

    pk = "0x" + ("85" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    redeemer = AutoRedeemer(
        mode="live",
        private_key=pk,
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        relayer_api_key="rk_test_v1_fallback",
        relayer_api_secret=cands["signer_eoa"],
        withdrawal_permission=False,
    )
    assert redeemer.relayer_client is not None

    captured_params_query: dict[str, Any] = {}

    class _MockResp:
        def __init__(self, status_code: int, body: dict[str, Any]) -> None:
            self.status_code = status_code
            self._body = body

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> dict[str, Any]:
            return self._body

    def _mock_get(url: str, params: Any = None, headers: Any = None, timeout: float = 5.0) -> _MockResp:
        if url.endswith("/nonce"):
            return _MockResp(404, {})
        if "/v1/account/transactions/params" in url:
            captured_params_query.update(dict(params or {}))
            return _MockResp(200, {"nonce": "77"})
        if url.endswith("/transaction"):
            return _MockResp(404, {})
        if "/v1/account/transactions/relayer_v1_tx_99" in url:
            # Return snake_case `transaction_hash` from new endpoint
            return _MockResp(
                200,
                {
                    "id": "relayer_v1_tx_99",
                    "state": "STATE_CONFIRMED",
                    "transaction_hash": "0x" + ("fa" * 32),
                },
            )
        return _MockResp(404, {})

    def _mock_post(url: str, json: Any = None, headers: Any = None, timeout: float = 8.0) -> _MockResp:
        if url.endswith("/submit"):
            return _MockResp(200, {"id": "relayer_v1_tx_99", "state": "STATE_NEW"})
        if isinstance(json, dict) and json.get("method") == "eth_getTransactionReceipt":
            return _MockResp(200, {"result": {"blockNumber": "0x200", "status": "0x1"}})
        return _MockResp(404, {})

    monkeypatch.setattr(red_mod.requests, "get", _mock_get)
    monkeypatch.setattr(red_mod.requests, "post", _mock_post)

    res = redeemer.redeem_settled_market(
        market_id="950001",
        condition_id="0x" + ("33" * 32),
        official_outcome="UP",
        live_orders=[{"signal_id": "sig_v1_fb", "position_status": "OPEN", "outcome": "UP", "filled_quantity": "15.00"}],
    )
    assert captured_params_query == {"address": cands["signer_eoa"], "type": "WALLET"}
    assert res["redeem_status"] == "REDEEMED"
    assert res["redeemed_usdc_amount"] == 15.0
    assert res["redeem_tx_hash"] == "0x" + ("fa" * 32)


def test_round4_p1_1_unsent_or_local_sign_failed_order_never_claims_historical_trades(
    tmp_path: Path,
) -> None:
    """[Round-4 P1-1] Reproduce:
    1) Local signing (`create_order`) raises an exception BEFORE HTTP send (`sent_count == 0`),
       while `get_trades` contains a historical 25-share / $10.00 fill on the same `token_id`.
    2) Verify the signal is recorded as `ABORTED_LOCAL_SIGN_FAILED` (`NO_FILL`, `0.00` shares, `$0.00` spent)
       and `reconcile_pending_orders` NEVER claims the historical trade!
    3) Also verify that a row in `RESERVED` (pre-send crash) is reconciled to `ABORTED_PRE_SEND_NOT_SUBMITTED`
       (`0.00` shares) and an `ORDER_STATE_UNKNOWN` row lacking `clob_order_id` NEVER matches trades by `token_id`.
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})

    now_dt = datetime.now(timezone.utc)
    w_start = (now_dt - timedelta(seconds=5)).isoformat()
    w_end = (now_dt + timedelta(minutes=15)).isoformat()
    valid_until = (now_dt + timedelta(seconds=15)).isoformat()

    _write_json(
        state_path,
        {
            "status": "PAPER_CYCLE_COMPLETE",
            "active": True,
            "last_cohort": w_start,
            "decisions": {
                "BTC_15m": {
                    "status": "SIGNAL",
                    "outcome": "UP",
                    "signal": {
                        "signal_id": "sig_local_sign_fail",
                        "stream": "BTC_15m",
                        "underlying": "BTC",
                        "outcome": "UP",
                        "market_id": "960001",
                        "decision_at": now_dt.isoformat(),
                        "valid_until": valid_until,
                        "window_start": w_start,
                        "window_end": w_end,
                    },
                }
            },
            "execution": {
                "status": "PROCESSED",
                "fills": [
                    {
                        "order_id": "paper:sig_local_sign_fail",
                        "signal_id": "sig_local_sign_fail",
                        "market_id": "960001",
                        "underlying": "BTC",
                        "outcome": "UP",
                        "status": "OPEN",
                        "quantity": "250.00000000",
                        "cost_usd": 100.0,
                        "fee_usd": 0.0,
                        "unfilled_budget_usd": 0.0,
                        "fill_at": now_dt.isoformat(),
                        "book_observed_at": now_dt.isoformat(),
                        "fee_quote": {"fee_model": "polymarket_crypto", "rate": 0.0, "exponent": 1, "taker_only": True},
                        "allocation_authorization": {
                            "policy": "RANKED_CAPACITY_WATERFALL",
                            "plan_id": "cohort:plan_sign_fail",
                            "cohort_id": w_start,
                            "rank": 1,
                            "signal_id": "sig_local_sign_fail",
                            "market_id": "960001",
                            "approved_stake_usd": 100.0,
                            "cohort_capacity_usd": 500.0,
                        },
                        "venue_market": {
                            "market_id": "960001",
                            "condition_id": "0x" + ("61" * 32),
                            "protocol": "v1",
                            "asset": "BTC",
                            "window_start": w_start,
                            "window_end": w_end,
                            "outcome_ids": {"UP": "770001", "DOWN": "770002"},
                        },
                    }
                ],
            },
        },
    )

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.fetch_status = lambda allow_in_critical_window=False: None  # type: ignore[method-assign]
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]
    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    pk = "0x" + ("91" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    clob._cached_free_usdc = 500.0
    clob._cached_allowance_usdc = 500.0
    clob.balance_verified = True
    assert clob.client is not None
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.40", "size": "500"}]}  # type: ignore[method-assign]

    sent_count = 0

    def _fail_create_order(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("Simulated local EIP-712 signing failure")

    def _post_order(*a: Any, **kw: Any) -> Any:
        nonlocal sent_count
        sent_count += 1
        return {"status": "MATCHED"}

    # Historical trade on the same token_id (`770001`) for 25 shares @ $0.40 ($10.00)
    clob.client.create_order = _fail_create_order  # type: ignore[method-assign]
    clob.client.post_order = _post_order  # type: ignore[method-assign]
    clob.client.get_order = lambda oid: None  # type: ignore[method-assign]
    clob.client.get_trades = lambda *a, **kw: [  # type: ignore[method-assign]
        {"asset_id": "770001", "size": "25.00", "price": "0.40", "status": "CONFIRMED"}
    ]

    redeemer = AutoRedeemer(mode="shadow", withdrawal_permission=False)
    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )
    clob.query_wallet_balance_and_allowance = lambda refresh_network=False, **kw: (500.0, 500.0)  # type: ignore[method-assign]

    changed, state_payload, _ = observer.poll_pipeline_state(force=True)
    assert changed is True and state_payload is not None
    records = engine.process_pipeline_state(state_payload)
    assert sent_count == 0
    assert len(records) == 1
    assert records[0]["live_execution"]["clob_status"] == "ABORTED_LOCAL_SIGN_FAILED"
    assert records[0]["live_execution"]["filled_quantity"] == "0.00"
    assert records[0]["live_execution"]["total_spent_usdc"] == 0.0

    # Reconcile pending orders: must NEVER claim the historical 25-share / $10 trade!
    reconciled = engine.reconcile_pending_orders(force=True)
    assert reconciled == []
    saved_orders = audit.get_live_orders_for_market("960001")
    assert len(saved_orders) == 1
    assert saved_orders[0]["position_status"] == "NO_FILL"
    assert saved_orders[0]["filled_quantity"] == "0.00"
    assert saved_orders[0]["total_spent_usdc"] == 0.0

    # Also test:
    # (a) A `RESERVED` row (crashed before pre-send) transitions to `ABORTED_PRE_SEND_NOT_SUBMITTED` (`0.00` shares).
    # (b) An `ORDER_STATE_UNKNOWN` row without `clob_order_id` does NOT match historical trades by `token_id`.
    claim_res = audit.claim_and_reserve_signal(
        signal_id="sig_reserved_never_sent",
        cohort_id=w_start,
        stream="BTC_15m",
        rank=1,
        paper_order_id="paper:sig_reserved_never_sent",
        market_id="960002",
        condition_id="0x" + ("62" * 32),
        token_id="770001",
        outcome="UP",
        underlying="BTC",
        window_start=w_start,
        window_end=w_end,
        claimed_at_utc=now_dt,
        approved_stake_usd=Decimal("100.0"),
        live_scale_factor=Decimal("0.10"),
        paper_caps={"max_order_usd": 200.0, "max_open_risk_usd": 1000.0, "max_underlying_open_risk_usd": 500.0, "max_tick_budget_usd": 1200.0},
        proxy_wallet_address=cands["beacon_deposit_wallet"],
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
        balance_verified=True,
    )
    assert claim_res["claimed"] is True

    reconciled_reserved = engine.reconcile_pending_orders(force=True)
    assert len(reconciled_reserved) == 1
    assert reconciled_reserved[0]["live_execution"]["clob_status"] == "ABORTED_PRE_SEND_NOT_SUBMITTED"
    assert reconciled_reserved[0]["live_execution"]["filled_quantity"] == "0.00"
    assert reconciled_reserved[0]["live_execution"]["total_spent_usdc"] == 0.0
    m2_orders = audit.get_live_orders_for_market("960002")
    assert m2_orders[0]["position_status"] == "NO_FILL"
    assert m2_orders[0]["filled_quantity"] == "0.00"

    assert (
        clob.reconcile_order(
            {
                "signal_id": "sig_unknown_no_id",
                "position_status": "ORDER_STATE_UNKNOWN",
                "clob_status": "ORDER_STATE_UNKNOWN",
                "clob_order_id": "",
                "token_id": "770001",
                "requested_stake_usd": 10.0,
                "requested_quantity": "25.00",
                "limit_price": 0.40,
            }
        )
        is None
    )


def test_round4_p1_2_repolling_pending_settlements_never_shifts_pnl_date_or_bypasses_loss_stops(
    tmp_path: Path,
) -> None:
    """[Round-4 P1-2] Reproduce:
    1) Yesterday (`2026-11-08T10:00:00+00:00`): Market `m_yest_win` settles with `+$40.00` realized profit on
       Batch 1 (`sig_yest_win_100`: 100 shares redeemed for `$100` vs `$60` cost = `+$40`), while a second
       order (`sig_yest_unk`: `$1`) is still `ORDER_STATE_UNKNOWN`, keeping `m_yest_win` in pending settlements.
    2) Today (`2026-11-09T10:00:00+00:00`): Markets `m_today_loss_1` (`-$40`) and `m_today_loss_2` (`-$40`)
       settle with `-$80.00` total loss today, triggering the `$60.00` daily loss stop (`daily_loss_limit = $60`).
    3) Re-polling `m_yest_win` today (`2026-11-09T10:05:00+00:00`):
       - Must preserve `settled_at_utc == "2026-11-08T10:00:00+00:00"` (and per-delta `pnl_history`).
       - Must NOT shift yesterday's `+$40.00` profit into today's PnL!
       - Today's daily realized PnL remains `-$80.00`, `daily_loss_triggered` remains `True`, and a new `$20`
         order is strictly blocked (`ABORTED_DAILY_LOSS_STOP`).
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )

    yest_iso = "2026-11-08T10:00:00+00:00"
    today_iso = "2026-11-09T10:00:00+00:00"
    today_repoll_iso = "2026-11-09T10:05:00+00:00"
    today_order_dt = datetime(2026, 11, 9, 10, 10, 0, tzinfo=timezone.utc)

    # 1) Yesterday's market `m_yest_win`: 100 winning shares ($60 cost -> $100 payout = +$40 profit) + 1 unknown order
    audit.record_order_execution(
        {
            "timestamp_utc": yest_iso,
            "cohort_id": yest_iso,
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_yest_win_100",
            "paper_order_id": "paper:sig_yest_win_100",
            "market_id": "m_yest_win",
            "condition_id": "0x" + ("71" * 32),
            "token_id": "10001",
            "outcome": "UP",
            "timing": {"decision_at": yest_iso, "paper_fill_at": yest_iso, "live_detected_at": yest_iso, "live_signed_at": yest_iso, "live_http_sent_at": yest_iso, "live_http_ack_at": yest_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 600.0, "paper_filled_quantity": 1000.0, "avg_fill_price": 0.60, "cost_usd": 600.0, "fee_usd": 0.0, "total_spent_usd": 600.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.60,
                "requested_stake_usd": 60.0,
                "requested_quantity": "100.00",
                "clob_order_id": "0xyest_win",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "100.00",
                "avg_fill_price": 0.60,
                "cost_usdc": 60.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 60.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": ["0x" + ("aa" * 32)],
                "onchain_confirmed": True,
                "onchain_block_number": 100,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=yest_iso,
        window_end="2026-11-08T10:15:00+00:00",
    )
    audit.record_order_execution(
        {
            "timestamp_utc": yest_iso,
            "cohort_id": yest_iso,
            "stream": "BTC_15m",
            "rank": 2,
            "signal_id": "sig_yest_unk",
            "paper_order_id": "paper:sig_yest_unk",
            "market_id": "m_yest_win",
            "condition_id": "0x" + ("71" * 32),
            "token_id": "10001",
            "outcome": "UP",
            "timing": {"decision_at": yest_iso, "paper_fill_at": yest_iso, "live_detected_at": yest_iso, "live_signed_at": yest_iso, "live_http_sent_at": yest_iso, "live_http_ack_at": yest_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 10.0, "paper_filled_quantity": 20.0, "avg_fill_price": 0.50, "cost_usd": 10.0, "fee_usd": 0.0, "total_spent_usd": 10.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 1.0,
                "requested_quantity": "2.00",
                "clob_order_id": "0xyest_unk",
                "clob_status": "ORDER_STATE_UNKNOWN",
                "error_code": "TIMEOUT",
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": 1.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": False,
                "onchain_block_number": None,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=yest_iso,
        window_end="2026-11-08T10:15:00+00:00",
    )
    audit.record_settlement_and_redeem(
        {
            "timestamp_utc": yest_iso,
            "market_id": "m_yest_win",
            "condition_id": "0x" + ("71" * 32),
            "official_outcome": "UP",
            "paper_payout_usd": 1000.0,
            "paper_realized_pnl_usd": 400.0,
            "live_expected_payout_usdc": 100.0,
            "redeem_tx_hash": "0x" + ("b1" * 32),
            "redeem_status": "REDEEMED",
            "redeemed_usdc_amount": 100.0,
            "newly_redeemed_usdc_amount": 100.0,
            "redeemed_order_keys": ["sig_yest_win_100"],
            "newly_redeemed_order_keys": ["sig_yest_win_100"],
            "redeem_latency_seconds": 1.0,
            "live_order_count": 2,
        }
    )

    # 2) Today's two losing markets (`-$40` each = `-$80` total today)
    for idx in (1, 2):
        mid = f"m_today_loss_{idx}"
        sig = f"sig_today_loss_{idx}"
        audit.record_order_execution(
            {
                "timestamp_utc": today_iso,
                "cohort_id": today_iso,
                "stream": "BTC_15m",
                "rank": 1,
                "signal_id": sig,
                "paper_order_id": f"paper:{sig}",
                "market_id": mid,
                "condition_id": "0x" + (f"7{idx + 1}" * 32),
                "token_id": "10001",
                "outcome": "UP",
                "timing": {"decision_at": today_iso, "paper_fill_at": today_iso, "live_detected_at": today_iso, "live_signed_at": today_iso, "live_http_sent_at": today_iso, "live_http_ack_at": today_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
                "paper_benchmark": {"approved_stake_usd": 400.0, "paper_filled_quantity": 800.0, "avg_fill_price": 0.50, "cost_usd": 400.0, "fee_usd": 0.0, "total_spent_usd": 400.0},
                "live_execution": {
                    "live_scale_factor": 0.10,
                    "order_type": "FAK",
                    "limit_price": 0.50,
                    "requested_stake_usd": 40.0,
                    "requested_quantity": "80.00",
                    "clob_order_id": f"0xtoday_loss_{idx}",
                    "clob_status": "MATCHED",
                    "error_code": None,
                    "filled_quantity": "80.00",
                    "avg_fill_price": 0.50,
                    "cost_usdc": 40.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 40.0,
                    "unfilled_stake_usdc": 0.0,
                    "polygon_tx_hashes": ["0x" + ("cc" * 32)],
                    "onchain_confirmed": True,
                    "onchain_block_number": 200 + idx,
                },
                "diff_vs_paper": {},
            },
            underlying="BTC",
            window_start=today_iso,
            window_end="2026-11-09T10:15:00+00:00",
        )
        audit.record_settlement_and_redeem(
            {
                "timestamp_utc": today_iso,
                "market_id": mid,
                "condition_id": "0x" + (f"7{idx + 1}" * 32),
                "official_outcome": "DOWN",
                "paper_payout_usd": 0.0,
                "paper_realized_pnl_usd": -400.0,
                "live_expected_payout_usdc": 0.0,
                "redeem_tx_hash": None,
                "redeem_status": "LOST_NO_REDEEM",
                "redeemed_usdc_amount": 0.0,
                "newly_redeemed_usdc_amount": 0.0,
                "redeemed_order_keys": [],
                "newly_redeemed_order_keys": [],
                "redeem_latency_seconds": 0.0,
                "live_order_count": 1,
            }
        )

    # 3) Re-poll yesterday's pending market (`m_yest_win`) today at `today_repoll_iso`:
    audit.record_settlement_and_redeem(
        {
            "timestamp_utc": today_repoll_iso,
            "market_id": "m_yest_win",
            "condition_id": "0x" + ("71" * 32),
            "official_outcome": "UP",
            "paper_payout_usd": 1000.0,
            "paper_realized_pnl_usd": 400.0,
            "live_expected_payout_usdc": 100.0,
            "redeem_tx_hash": None,
            "redeem_status": "AWAITING_ORDER_RECONCILIATION",
            "redeemed_usdc_amount": 100.0,
            "newly_redeemed_usdc_amount": 0.0,
            "redeemed_order_keys": ["sig_yest_win_100"],
            "newly_redeemed_order_keys": [],
            "redeem_latency_seconds": 0.1,
            "live_order_count": 2,
        }
    )

    saved_yest = audit.get_settlement_record("m_yest_win")
    assert saved_yest is not None
    # Initial confirmation timestamp MUST be preserved!
    assert saved_yest["settled_at_utc"] == yest_iso

    # Attempting to claim a new $20 order today MUST be blocked by the $60 daily loss stop (-$80 today)!
    claim_today = audit.claim_and_reserve_signal(
        signal_id="sig_today_blocked_buy",
        cohort_id=today_iso,
        stream="BTC_15m",
        rank=1,
        paper_order_id="paper:sig_today_blocked_buy",
        market_id="m_today_new",
        condition_id="0x" + ("79" * 32),
        token_id="10001",
        outcome="UP",
        underlying="BTC",
        window_start=today_iso,
        window_end="2026-11-09T10:25:00+00:00",
        claimed_at_utc=today_order_dt,
        approved_stake_usd=Decimal("200.0"),
        live_scale_factor=Decimal("0.10"),
        paper_caps={"max_order_usd": 200.0, "max_open_risk_usd": 1000.0, "max_underlying_open_risk_usd": 500.0, "max_tick_budget_usd": 1200.0},
        proxy_wallet_address="0x" + ("44" * 20),
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
        balance_verified=True,
        max_daily_realized_loss_usd=Decimal("60.0"),
        max_consecutive_losses=10,
        cooldown_minutes_after_consecutive_losses=30.0,
    )
    assert claim_today["aborted"] is True
    assert claim_today["abort_status"] == "ABORTED_DAILY_LOSS_STOP"
    assert claim_today["loss_stats"]["daily_realized_pnl_usdc"] == -80.0


def test_round4_p1_3_crash_during_redemption_confirm_wait_persists_tx_id_and_prevents_duplicate_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-4 P1-3] Reproduce:
    1) `relayer_client.execute()` succeeds and returns `transaction_id="relayer_tx_crash_1"`.
    2) While waiting in `poll_until_state`, a simulated power failure / crash (`RuntimeError`) occurs
       before `redeem_settled_market` returns.
    3) Verify `live_settlements` in SQLite ALREADY persisted `redeem_tx_id="relayer_tx_crash_1"` and
       `pending_batch_order_keys=["sig_crash_1"]` BEFORE waiting for confirmation!
    4) On restart, polling settlements queries `relayer_tx_crash_1` and NEVER calls `execute()` a second time!
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]
    observer.fetch_status = lambda allow_in_critical_window=False: {}  # type: ignore[method-assign]
    observer.is_in_critical_window = lambda now_utc=None: False  # type: ignore[method-assign]
    observer.fetch_market_resolution = lambda market_id, condition_id=None: {  # type: ignore[method-assign]
        "resolved": True,
        "official_outcome": "UP",
        "condition_id": condition_id or ("0x" + ("81" * 32)),
        "market_protocol": "v2",
        "neg_risk": False,
    }

    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(mode="shadow", private_key=None)
    clob._cached_free_usdc = 100.0
    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)

    execute_calls = 0
    should_crash_on_poll = True

    class _CrashDuringWaitRelayer:
        def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
            nonlocal execute_calls
            execute_calls += 1
            return _RelayerExecResponse(
                transaction_id=f"relayer_tx_crash_{execute_calls}",
                transaction_hash=None,
                state="STATE_NEW",
            )

        def poll_until_state(self, **kwargs: Any) -> dict[str, Any] | None:
            if should_crash_on_poll:
                raise BaseException("SIMULATED_POWER_LOSS_DURING_POLL")
            return {
                "id": "relayer_tx_crash_1",
                "state": "STATE_CONFIRMED",
                "transactionHash": "0x" + ("c1" * 32),
            }

    redeemer.relayer_client = _CrashDuringWaitRelayer()

    import argon4_live.redeemer as red_mod

    class _MockRpcResp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"result": {"blockNumber": "0x300", "status": "0x1"}}

    monkeypatch.setattr(red_mod.requests, "post", lambda *a, **kw: _MockRpcResp())

    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    now_iso = "2026-11-09T11:00:00+00:00"
    v2_cond = "0x00" + ("81" * 31)
    audit.record_order_execution(
        {
            "timestamp_utc": now_iso,
            "cohort_id": now_iso,
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_crash_1",
            "paper_order_id": "paper:sig_crash_1",
            "market_id": "970001",
            "condition_id": v2_cond,
            "token_id": "0x02" + ("81" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 75.0, "paper_filled_quantity": 150.0, "avg_fill_price": 0.50, "cost_usd": 75.0, "fee_usd": 0.0, "total_spent_usd": 75.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 7.5,
                "requested_quantity": "15.00",
                "clob_order_id": "0xord_crash_1",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "15.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 7.5,
                "fee_usdc": 0.0,
                "total_spent_usdc": 7.5,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": ["0x" + ("dd" * 32)],
                "onchain_confirmed": True,
                "onchain_block_number": 300,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=now_iso,
        window_end="2026-11-09T11:15:00+00:00",
    )

    # First attempt crashes with BaseException (power loss) inside `poll_until_state` BEFORE function returns:
    with pytest.raises(BaseException, match="SIMULATED_POWER_LOSS_DURING_POLL"):
        engine._execute_and_record_settlement(
            market_id="970001",
            condition_id=v2_cond,
            official_outcome="UP",
            paper_payout_usd=150.0,
            paper_realized_pnl_usd=75.0,
            paper_settled_at=now_iso,
        )

    assert execute_calls == 1
    # Verify SQLite ALREADY persisted `relayer_tx_crash_1` and the pending batch before polling!
    persisted = audit.get_settlement_record("970001")
    assert persisted is not None
    assert persisted["redeem_status"] == "REDEEM_PENDING"
    assert persisted["redeem_tx_id"] == "relayer_tx_crash_1"
    assert persisted["pending_batch_order_keys"] == ["sig_crash_1"]
    assert persisted["pending_batch_payout_usdc"] == 15.0

    # Restart / next poll: must poll `relayer_tx_crash_1` and NEVER call `execute()` again!
    should_crash_on_poll = False
    recovered = engine.poll_settlements_and_redeem(force=True)
    assert execute_calls == 1
    assert len(recovered) == 1
    assert recovered[0]["redeem_status"] == "REDEEMED"
    assert recovered[0]["redeem_tx_id"] == "relayer_tx_crash_1"
    assert recovered[0]["redeemed_usdc_amount"] == 15.0


def test_round4_p2_4_balance_refresh_and_redemption_credit_never_double_count_funds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-4 P2-4] Reproduce:
    1) Redemption of `$12.00` is submitted when `_cached_free_usdc == 88.0`, returning `REDEEM_PENDING`.
    2) While `REDEEM_PENDING`, the wallet balance is refreshed to `$100.00` (via `query_wallet_balance_and_allowance`
       or after restart).
    3) Subsequently, the pending `$12.00` redemption confirms (`REDEEMED`).
    4) Verify `_cached_free_usdc` remains `$100.00` and is NEVER inflated to `$112.00`!
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]
    observer.fetch_status = lambda allow_in_critical_window=False: {}  # type: ignore[method-assign]
    observer.is_in_critical_window = lambda now_utc=None: False  # type: ignore[method-assign]
    observer.fetch_market_resolution = lambda market_id, condition_id=None: {  # type: ignore[method-assign]
        "resolved": True,
        "official_outcome": "UP",
        "condition_id": condition_id or ("0x" + ("82" * 32)),
        "market_protocol": "v2",
        "neg_risk": False,
    }

    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(mode="shadow", private_key=None)
    clob._cached_free_usdc = 88.0
    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)

    tx_confirmed = False

    class _PendingThenConfirmedRelayer:
        def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
            return _RelayerExecResponse(transaction_id="relayer_tx_bal_12", transaction_hash=None, state="STATE_NEW")

        def poll_until_state(self, **kwargs: Any) -> dict[str, Any] | None:
            if not tx_confirmed:
                return None
            return {
                "id": "relayer_tx_bal_12",
                "state": "STATE_CONFIRMED",
                "transactionHash": "0x" + ("c2" * 32),
            }

    redeemer.relayer_client = _PendingThenConfirmedRelayer()

    import argon4_live.redeemer as red_mod

    class _MockRpcResp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"result": {"blockNumber": "0x301", "status": "0x1"}}

    monkeypatch.setattr(red_mod.requests, "post", lambda *a, **kw: _MockRpcResp())

    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    now_iso = "2026-11-09T11:30:00+00:00"
    v2_cond = "0x00" + ("82" * 31)
    audit.record_order_execution(
        {
            "timestamp_utc": now_iso,
            "cohort_id": now_iso,
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_bal_12",
            "paper_order_id": "paper:sig_bal_12",
            "market_id": "970002",
            "condition_id": v2_cond,
            "token_id": "0x02" + ("82" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 60.0, "paper_filled_quantity": 120.0, "avg_fill_price": 0.50, "cost_usd": 60.0, "fee_usd": 0.0, "total_spent_usd": 60.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 6.0,
                "requested_quantity": "12.00",
                "clob_order_id": "0xord_bal_12",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "12.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 6.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 6.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": ["0x" + ("ee" * 32)],
                "onchain_confirmed": True,
                "onchain_block_number": 301,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=now_iso,
        window_end="2026-11-09T11:45:00+00:00",
    )

    # Step 1: Submit $12.00 redemption -> REDEEM_PENDING, balance is $88.00
    s1 = engine._execute_and_record_settlement(
        market_id="970002",
        condition_id=v2_cond,
        official_outcome="UP",
        paper_payout_usd=120.0,
        paper_realized_pnl_usd=60.0,
        paper_settled_at=now_iso,
    )
    assert s1["redeem_status"] == "REDEEM_PENDING"
    assert clob._cached_free_usdc == 88.0

    # Step 2: Balance is refreshed to $100.00 (which already includes the $12.00 redemption)
    clob._cached_free_usdc = 100.0
    clob.query_wallet_balance_and_allowance(refresh_network=True)
    assert clob._cached_free_usdc == 100.0

    # Step 3: Pending redemption confirms -> must NOT add $12.00 again on top of $100.00!
    tx_confirmed = True
    s2 = engine.poll_settlements_and_redeem(force=True)
    assert len(s2) == 1
    assert s2[0]["redeem_status"] == "REDEEMED"
    assert s2[0]["redeemed_usdc_amount"] == 12.0
    assert clob._cached_free_usdc == 100.0


def test_round4_p2_5_zero_fee_rate_preserved_during_order_reconciliation(
    tmp_path: Path,
) -> None:
    """[Round-4 P2-5] Reproduce:
    An order in a fee-free market (`fee_rate = 0.0`) enters `ORDER_STATE_UNKNOWN` and is later reconciled
    as `MATCHED` (`12.00` shares @ `$0.50` = `$6.00` cost, no explicit fee in trade object).
    Verify `fee_rate = 0.0` is preserved (NOT replaced by `0.075`), so `fee_usdc == 0.0` and
    `total_spent_usdc == 6.0` (never `$6.45`!).
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.is_in_critical_window = lambda now_utc=None: False  # type: ignore[method-assign]

    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    pk = "0x" + ("95" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    assert clob.client is not None
    clob.client.get_order = lambda oid: {  # type: ignore[method-assign]
        "id": oid,
        "status": "MATCHED",
        "size_matched": "12.00",
        "price": "0.50",
    }
    clob.client.get_trades = lambda *a, **kw: []  # type: ignore[method-assign]

    redeemer = AutoRedeemer(mode="shadow", withdrawal_permission=False)
    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    now_iso = "2026-11-09T12:00:00+00:00"
    audit.record_order_execution(
        {
            "timestamp_utc": now_iso,
            "cohort_id": now_iso,
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_zero_fee_rec",
            "paper_order_id": "paper:sig_zero_fee_rec",
            "market_id": "970005",
            "condition_id": "0x" + ("85" * 32),
            "token_id": "10005",
            "outcome": "UP",
            "fee_rate": 0.0,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 60.0, "paper_filled_quantity": 120.0, "avg_fill_price": 0.50, "cost_usd": 60.0, "fee_usd": 0.0, "total_spent_usd": 60.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 6.0,
                "requested_quantity": "12.00",
                "clob_order_id": "0xord_zero_fee",
                "clob_status": "ORDER_STATE_UNKNOWN",
                "error_code": "TIMEOUT",
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": 6.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": False,
                "onchain_block_number": None,
                "fee_rate": 0.0,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=now_iso,
        window_end="2026-11-09T12:15:00+00:00",
    )

    reconciled = engine.reconcile_pending_orders(force=True)
    assert len(reconciled) == 1
    assert reconciled[0]["live_execution"]["fee_usdc"] == 0.0
    assert reconciled[0]["live_execution"]["cost_usdc"] == 6.0
    assert reconciled[0]["live_execution"]["total_spent_usdc"] == 6.0

    saved = audit.get_live_orders_for_market("970005")
    assert len(saved) == 1
    assert saved[0]["fee_rate"] == 0.0
    assert saved[0]["fee_usdc"] == 0.0
    assert saved[0]["total_spent_usdc"] == 6.0


def test_round4_p2_6_legacy_pending_settlement_without_batch_keys_never_guesses_order_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[Round-4 P2-6] Reproduce:
    A legacy `REDEEM_PENDING` settlement record has `redeem_tx_id="legacy_tx_12"`,
    `live_expected_payout_usdc=12.0`, and `pending_batch_order_keys=[]` (missing batch keys).
    Meanwhile, `live_orders` has two unredeemed winning orders: a newly reconciled 8-share order (`sig_new_8`)
    followed by a 12-share order (`sig_old_12`).
    Verify `redeem_settled_market` refuses to guess attribution by order sequence (`newly_redeemed_order_keys == []`,
    `redeem_status == "REDEEM_PENDING"`, `error == "LEGACY_PENDING_BATCH_ATTRIBUTION_UNVERIFIED"`).
    """
    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)

    class _LegacyConfirmedRelayer:
        def get_transaction(self, transaction_id: str) -> dict[str, Any]:
            return {
                "id": transaction_id,
                "state": "STATE_CONFIRMED",
                "transactionHash": "0x" + ("c6" * 32),
            }

    redeemer.relayer_client = _LegacyConfirmedRelayer()

    import argon4_live.redeemer as red_mod

    class _MockRpcResp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"result": {"blockNumber": "0x306", "status": "0x1"}}

    monkeypatch.setattr(red_mod.requests, "post", lambda *a, **kw: _MockRpcResp())

    res = redeemer.redeem_settled_market(
        market_id="970006",
        condition_id="0x" + ("86" * 32),
        official_outcome="UP",
        live_orders=[
            {"signal_id": "sig_new_8", "position_status": "WON_UNREDEEMED", "outcome": "UP", "filled_quantity": "8.00"},
            {"signal_id": "sig_old_12", "position_status": "WON_UNREDEEMED", "outcome": "UP", "filled_quantity": "12.00"},
        ],
        existing_settlement={
            "market_id": "970006",
            "redeem_status": "REDEEM_PENDING",
            "redeem_tx_id": "legacy_tx_12",
            "live_expected_payout_usdc": 12.0,
            "pending_batch_order_keys": [],
            "redeemed_order_keys": [],
            "redeemed_usdc_amount": 0.0,
        },
    )
    assert res["redeem_status"] == "REDEEM_PENDING"
    assert res["error"] == "LEGACY_PENDING_BATCH_ATTRIBUTION_UNVERIFIED"
    assert res["newly_redeemed_order_keys"] == []
    assert res["redeemed_order_keys"] == []
    assert res["newly_redeemed_usdc_amount"] == 0.0


def test_round5_p1_1_daily_loss_counted_when_market_cumulative_pnl_nets_to_zero(
    tmp_path: Path,
) -> None:
    """[Round-5 P1-1] Reproduce:
    4 markets each won `+$20.00` yesterday (`2026-11-08`) and each lost `-$20.00` today (`2026-11-09`),
    while still having an unknown order (`ORDER_STATE_UNKNOWN`), so each market has
    `redeem_status == "AWAITING_ORDER_RECONCILIATION"` and cumulative `live_realized_pnl_usdc == 0.0`.
    Verify today's realized PnL is computed as `-$80.00` (NEVER `0.0`!), triggering the `$60.00`
    daily loss stop and blocking new orders with `ABORTED_DAILY_LOSS_STOP`.
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )

    yest_iso = "2026-11-08T10:00:00+00:00"
    today_iso = "2026-11-09T10:00:00+00:00"
    today_check_dt = datetime(2026, 11, 9, 10, 5, 0, tzinfo=timezone.utc)

    for idx in range(1, 5):
        mid = f"m_net_zero_{idx}"
        cond_id = "0x" + (f"9{idx}" * 32)
        win_sig = f"sig_win_{idx}"
        loss_sig = f"sig_loss_{idx}"
        unk_sig = f"sig_unk_{idx}"

        # Order 1: Winning order yesterday ($20 cost -> $40 payout = +$20 profit)
        audit.record_order_execution(
            {
                "timestamp_utc": yest_iso,
                "cohort_id": yest_iso,
                "stream": "BTC_15m",
                "rank": 1,
                "signal_id": win_sig,
                "paper_order_id": f"paper:{win_sig}",
                "market_id": mid,
                "condition_id": cond_id,
                "token_id": "10001",
                "outcome": "UP",
                "timing": {"decision_at": yest_iso, "paper_fill_at": yest_iso, "live_detected_at": yest_iso, "live_signed_at": yest_iso, "live_http_sent_at": yest_iso, "live_http_ack_at": yest_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
                "paper_benchmark": {"approved_stake_usd": 200.0, "paper_filled_quantity": 400.0, "avg_fill_price": 0.50, "cost_usd": 200.0, "fee_usd": 0.0, "total_spent_usd": 200.0},
                "live_execution": {
                    "live_scale_factor": 0.10,
                    "order_type": "FAK",
                    "limit_price": 0.50,
                    "requested_stake_usd": 20.0,
                    "requested_quantity": "40.00",
                    "clob_order_id": f"0xwin_{idx}",
                    "clob_status": "MATCHED",
                    "error_code": None,
                    "filled_quantity": "40.00",
                    "avg_fill_price": 0.50,
                    "cost_usdc": 20.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 20.0,
                    "unfilled_stake_usdc": 0.0,
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": True,
                    "onchain_block_number": 100 + idx,
                },
                "diff_vs_paper": {},
            },
            underlying="BTC",
            window_start=yest_iso,
            window_end="2026-11-08T10:15:00+00:00",
        )
        # Order 2: Late losing order ($20 cost on DOWN, initially ORDER_STATE_UNKNOWN yesterday)
        audit.record_order_execution(
            {
                "timestamp_utc": yest_iso,
                "cohort_id": yest_iso,
                "stream": "BTC_15m",
                "rank": 2,
                "signal_id": loss_sig,
                "paper_order_id": f"paper:{loss_sig}",
                "market_id": mid,
                "condition_id": cond_id,
                "token_id": "10002",
                "outcome": "DOWN",
                "timing": {"decision_at": yest_iso, "paper_fill_at": yest_iso, "live_detected_at": yest_iso, "live_signed_at": yest_iso, "live_http_sent_at": yest_iso, "live_http_ack_at": yest_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
                "paper_benchmark": {"approved_stake_usd": 200.0, "paper_filled_quantity": 400.0, "avg_fill_price": 0.50, "cost_usd": 200.0, "fee_usd": 0.0, "total_spent_usd": 200.0},
                "live_execution": {
                    "live_scale_factor": 0.10,
                    "order_type": "FAK",
                    "limit_price": 0.50,
                    "requested_stake_usd": 20.0,
                    "requested_quantity": "40.00",
                    "clob_order_id": f"0xloss_{idx}",
                    "clob_status": "ORDER_STATE_UNKNOWN",
                    "error_code": "TIMEOUT",
                    "filled_quantity": "0.00",
                    "avg_fill_price": 0.0,
                    "cost_usdc": 0.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 0.0,
                    "unfilled_stake_usdc": 20.0,
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                },
                "diff_vs_paper": {},
            },
            underlying="BTC",
            window_start=yest_iso,
            window_end="2026-11-08T10:15:00+00:00",
        )
        # Order 3: Still-unknown order ($1 stake) so market stays AWAITING_ORDER_RECONCILIATION
        audit.record_order_execution(
            {
                "timestamp_utc": yest_iso,
                "cohort_id": yest_iso,
                "stream": "BTC_15m",
                "rank": 3,
                "signal_id": unk_sig,
                "paper_order_id": f"paper:{unk_sig}",
                "market_id": mid,
                "condition_id": cond_id,
                "token_id": "10001",
                "outcome": "UP",
                "timing": {"decision_at": yest_iso, "paper_fill_at": yest_iso, "live_detected_at": yest_iso, "live_signed_at": yest_iso, "live_http_sent_at": yest_iso, "live_http_ack_at": yest_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
                "paper_benchmark": {"approved_stake_usd": 10.0, "paper_filled_quantity": 20.0, "avg_fill_price": 0.50, "cost_usd": 10.0, "fee_usd": 0.0, "total_spent_usd": 10.0},
                "live_execution": {
                    "live_scale_factor": 0.10,
                    "order_type": "FAK",
                    "limit_price": 0.50,
                    "requested_stake_usd": 1.0,
                    "requested_quantity": "2.00",
                    "clob_order_id": f"0xunk_{idx}",
                    "clob_status": "ORDER_STATE_UNKNOWN",
                    "error_code": "TIMEOUT",
                    "filled_quantity": "0.00",
                    "avg_fill_price": 0.0,
                    "cost_usdc": 0.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 0.0,
                    "unfilled_stake_usdc": 1.0,
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                },
                "diff_vs_paper": {},
            },
            underlying="BTC",
            window_start=yest_iso,
            window_end="2026-11-08T10:15:00+00:00",
        )

        # Yesterday: Order 1 redeemed for $40 (+$20 realized PnL)
        audit.record_settlement_and_redeem(
            {
                "timestamp_utc": yest_iso,
                "market_id": mid,
                "condition_id": cond_id,
                "official_outcome": "UP",
                "paper_payout_usd": 400.0,
                "paper_realized_pnl_usd": 200.0,
                "live_expected_payout_usdc": 40.0,
                "redeem_tx_hash": "0x" + (f"a{idx}" * 32),
                "redeem_status": "REDEEMED",
                "redeemed_usdc_amount": 40.0,
                "newly_redeemed_usdc_amount": 40.0,
                "redeemed_order_keys": [win_sig],
                "newly_redeemed_order_keys": [win_sig],
                "redeem_latency_seconds": 0.5,
                "live_order_count": 3,
            }
        )

        # Today: Order 2 reconciles as MATCHED ($20 spent on losing outcome DOWN) -> -$20 realized PnL today!
        audit.record_order_execution(
            {
                "timestamp_utc": today_iso,
                "cohort_id": yest_iso,
                "stream": "BTC_15m",
                "rank": 2,
                "signal_id": loss_sig,
                "paper_order_id": f"paper:{loss_sig}",
                "market_id": mid,
                "condition_id": cond_id,
                "token_id": "10002",
                "outcome": "DOWN",
                "timing": {"decision_at": yest_iso, "paper_fill_at": yest_iso, "live_detected_at": yest_iso, "live_signed_at": yest_iso, "live_http_sent_at": yest_iso, "live_http_ack_at": today_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
                "paper_benchmark": {"approved_stake_usd": 200.0, "paper_filled_quantity": 400.0, "avg_fill_price": 0.50, "cost_usd": 200.0, "fee_usd": 0.0, "total_spent_usd": 200.0},
                "live_execution": {
                    "live_scale_factor": 0.10,
                    "order_type": "FAK",
                    "limit_price": 0.50,
                    "requested_stake_usd": 20.0,
                    "requested_quantity": "40.00",
                    "clob_order_id": f"0xloss_{idx}",
                    "clob_status": "MATCHED",
                    "error_code": None,
                    "filled_quantity": "40.00",
                    "avg_fill_price": 0.50,
                    "cost_usdc": 20.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 20.0,
                    "unfilled_stake_usdc": 0.0,
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": True,
                    "onchain_block_number": 200 + idx,
                },
                "diff_vs_paper": {},
            },
            underlying="BTC",
            window_start=yest_iso,
            window_end="2026-11-08T10:15:00+00:00",
        )
        audit.record_settlement_and_redeem(
            {
                "timestamp_utc": today_iso,
                "market_id": mid,
                "condition_id": cond_id,
                "official_outcome": "UP",
                "paper_payout_usd": 400.0,
                "paper_realized_pnl_usd": 0.0,
                "live_expected_payout_usdc": 40.0,
                "redeem_tx_hash": None,
                "redeem_status": "AWAITING_ORDER_RECONCILIATION",
                "redeemed_usdc_amount": 40.0,
                "newly_redeemed_usdc_amount": 0.0,
                "redeemed_order_keys": [win_sig],
                "newly_redeemed_order_keys": [],
                "redeem_latency_seconds": 0.1,
                "live_order_count": 3,
            }
        )
        rec = audit.get_settlement_record(mid)
        assert rec is not None
        assert rec["redeem_status"] == "AWAITING_ORDER_RECONCILIATION"
        assert rec["live_realized_pnl_usdc"] == 0.0  # Cumulative PnL nets to 0.0 (+20 yesterday, -20 today)

    # Today's realized loss MUST be -80.0 USDC (4 * -$20), blocking a new order when daily loss stop is $60!
    claim = audit.claim_and_reserve_signal(
        signal_id="sig_r5_p1_1_blocked",
        cohort_id=today_iso,
        stream="BTC_15m",
        rank=1,
        paper_order_id="paper:sig_r5_p1_1_blocked",
        market_id="m_r5_new",
        condition_id="0x" + ("99" * 32),
        token_id="10001",
        outcome="UP",
        underlying="BTC",
        window_start=today_iso,
        window_end="2026-11-09T10:15:00+00:00",
        claimed_at_utc=today_check_dt,
        approved_stake_usd=Decimal("200.0"),
        live_scale_factor=Decimal("0.10"),
        paper_caps={"max_order_usd": 200.0, "max_open_risk_usd": 1000.0, "max_underlying_open_risk_usd": 500.0, "max_tick_budget_usd": 1200.0},
        proxy_wallet_address="0x" + ("44" * 20),
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
        balance_verified=True,
        max_daily_realized_loss_usd=Decimal("60.0"),
        max_consecutive_losses=10,
        cooldown_minutes_after_consecutive_losses=30.0,
    )
    assert claim["aborted"] is True
    assert claim["abort_status"] == "ABORTED_DAILY_LOSS_STOP"
    assert claim["loss_stats"]["daily_realized_pnl_usdc"] == -80.0


def test_round5_p1_2_late_order_newly_confirmed_loss_reorders_and_retriggers_cooldown(
    tmp_path: Path,
) -> None:
    """[Round-5 P1-2] Reproduce:
    1) 4 consecutive losses are confirmed at `09:50`, `09:55`, `10:00`, `10:02` (where market `m_loss_1` at `09:50`
       also had a 2nd `ORDER_STATE_UNKNOWN` order).
    2) At `11:00` (after the initial `10:02 + 30m = 10:32` cooldown expired), the late order in `m_loss_1`
       is reconciled as `LOST` (`-$10.00` new loss delta confirmed at `11:00`).
    3) At `11:01`, attempting to reserve a new order MUST use `11:00` (NOT `10:02` or `09:50`) as the latest
       loss confirmation time, re-triggering the 30-minute cooldown (`ABORTED_CONSECUTIVE_LOSS_COOLDOWN`).
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )

    loss_times = [
        "2026-11-09T09:50:00+00:00",
        "2026-11-09T09:55:00+00:00",
        "2026-11-09T10:00:00+00:00",
        "2026-11-09T10:02:00+00:00",
    ]
    for idx, ts_iso in enumerate(loss_times, start=1):
        mid = f"m_streak_{idx}"
        cond_id = "0x" + (f"a{idx}" * 32)
        sig = f"sig_streak_{idx}"
        audit.record_order_execution(
            {
                "timestamp_utc": ts_iso,
                "cohort_id": ts_iso,
                "stream": "BTC_15m",
                "rank": 1,
                "signal_id": sig,
                "paper_order_id": f"paper:{sig}",
                "market_id": mid,
                "condition_id": cond_id,
                "token_id": "10001",
                "outcome": "UP",
                "timing": {"decision_at": ts_iso, "paper_fill_at": ts_iso, "live_detected_at": ts_iso, "live_signed_at": ts_iso, "live_http_sent_at": ts_iso, "live_http_ack_at": ts_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
                "paper_benchmark": {"approved_stake_usd": 100.0, "paper_filled_quantity": 200.0, "avg_fill_price": 0.50, "cost_usd": 100.0, "fee_usd": 0.0, "total_spent_usd": 100.0},
                "live_execution": {
                    "live_scale_factor": 0.10,
                    "order_type": "FAK",
                    "limit_price": 0.50,
                    "requested_stake_usd": 10.0,
                    "requested_quantity": "20.00",
                    "clob_order_id": f"0xstreak_{idx}",
                    "clob_status": "MATCHED",
                    "error_code": None,
                    "filled_quantity": "20.00",
                    "avg_fill_price": 0.50,
                    "cost_usdc": 10.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 10.0,
                    "unfilled_stake_usdc": 0.0,
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": True,
                    "onchain_block_number": 300 + idx,
                },
                "diff_vs_paper": {},
            },
            underlying="BTC",
            window_start=ts_iso,
            window_end=ts_iso,
        )
        if idx == 1:
            # Market 1 also has a late unknown order
            audit.record_order_execution(
                {
                    "timestamp_utc": ts_iso,
                    "cohort_id": ts_iso,
                    "stream": "BTC_15m",
                    "rank": 2,
                    "signal_id": "sig_streak_1_late",
                    "paper_order_id": "paper:sig_streak_1_late",
                    "market_id": mid,
                    "condition_id": cond_id,
                    "token_id": "10001",
                    "outcome": "UP",
                    "timing": {"decision_at": ts_iso, "paper_fill_at": ts_iso, "live_detected_at": ts_iso, "live_signed_at": ts_iso, "live_http_sent_at": ts_iso, "live_http_ack_at": ts_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
                    "paper_benchmark": {"approved_stake_usd": 100.0, "paper_filled_quantity": 200.0, "avg_fill_price": 0.50, "cost_usd": 100.0, "fee_usd": 0.0, "total_spent_usd": 100.0},
                    "live_execution": {
                        "live_scale_factor": 0.10,
                        "order_type": "FAK",
                        "limit_price": 0.50,
                        "requested_stake_usd": 10.0,
                        "requested_quantity": "20.00",
                        "clob_order_id": "0xstreak_1_late",
                        "clob_status": "ORDER_STATE_UNKNOWN",
                        "error_code": "TIMEOUT",
                        "filled_quantity": "0.00",
                        "avg_fill_price": 0.0,
                        "cost_usdc": 0.0,
                        "fee_usdc": 0.0,
                        "total_spent_usdc": 0.0,
                        "unfilled_stake_usdc": 10.0,
                        "polygon_tx_hashes": [],
                        "onchain_confirmed": False,
                        "onchain_block_number": None,
                    },
                    "diff_vs_paper": {},
                },
                underlying="BTC",
                window_start=ts_iso,
                window_end=ts_iso,
            )

        audit.record_settlement_and_redeem(
            {
                "timestamp_utc": ts_iso,
                "market_id": mid,
                "condition_id": cond_id,
                "official_outcome": "DOWN",
                "paper_payout_usd": 0.0,
                "paper_realized_pnl_usd": -100.0,
                "live_expected_payout_usdc": 0.0,
                "redeem_tx_hash": None,
                "redeem_status": "LOST_NO_REDEEM",
                "redeemed_usdc_amount": 0.0,
                "newly_redeemed_usdc_amount": 0.0,
                "redeemed_order_keys": [],
                "newly_redeemed_order_keys": [],
                "redeem_latency_seconds": 0.0,
                "live_order_count": 2 if idx == 1 else 1,
            }
        )

    # At 11:00:00+00:00, `sig_streak_1_late` in `m_streak_1` reconciles as MATCHED (losing $10) -> new loss confirmed at 11:00!
    late_confirm_iso = "2026-11-09T11:00:00+00:00"
    audit.record_order_execution(
        {
            "timestamp_utc": late_confirm_iso,
            "cohort_id": loss_times[0],
            "stream": "BTC_15m",
            "rank": 2,
            "signal_id": "sig_streak_1_late",
            "paper_order_id": "paper:sig_streak_1_late",
            "market_id": "m_streak_1",
            "condition_id": "0x" + ("a1" * 32),
            "token_id": "10001",
            "outcome": "UP",
            "timing": {"decision_at": loss_times[0], "paper_fill_at": loss_times[0], "live_detected_at": loss_times[0], "live_signed_at": loss_times[0], "live_http_sent_at": loss_times[0], "live_http_ack_at": late_confirm_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 100.0, "paper_filled_quantity": 200.0, "avg_fill_price": 0.50, "cost_usd": 100.0, "fee_usd": 0.0, "total_spent_usd": 100.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 10.0,
                "requested_quantity": "20.00",
                "clob_order_id": "0xstreak_1_late",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "20.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 10.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 10.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 350,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=loss_times[0],
        window_end=loss_times[0],
    )
    audit.record_settlement_and_redeem(
        {
            "timestamp_utc": late_confirm_iso,
            "market_id": "m_streak_1",
            "condition_id": "0x" + ("a1" * 32),
            "official_outcome": "DOWN",
            "paper_payout_usd": 0.0,
            "paper_realized_pnl_usd": -200.0,
            "live_expected_payout_usdc": 0.0,
            "redeem_tx_hash": None,
            "redeem_status": "LOST_NO_REDEEM",
            "redeemed_usdc_amount": 0.0,
            "newly_redeemed_usdc_amount": 0.0,
            "redeemed_order_keys": [],
            "newly_redeemed_order_keys": [],
            "redeem_latency_seconds": 0.0,
            "live_order_count": 2,
        }
    )

    # At 11:01:00+00:00, cooldown MUST be active (anchored at 11:00:00, NOT 10:02:00)!
    check_1101_dt = datetime(2026, 11, 9, 11, 1, 0, tzinfo=timezone.utc)
    claim_1101 = audit.claim_and_reserve_signal(
        signal_id="sig_r5_p1_2_blocked_cooldown",
        cohort_id=late_confirm_iso,
        stream="BTC_15m",
        rank=1,
        paper_order_id="paper:sig_r5_p1_2_blocked_cooldown",
        market_id="m_streak_new",
        condition_id="0x" + ("a9" * 32),
        token_id="10001",
        outcome="UP",
        underlying="BTC",
        window_start=late_confirm_iso,
        window_end="2026-11-09T11:15:00+00:00",
        claimed_at_utc=check_1101_dt,
        approved_stake_usd=Decimal("100.0"),
        live_scale_factor=Decimal("0.10"),
        paper_caps={"max_order_usd": 200.0, "max_open_risk_usd": 1000.0, "max_underlying_open_risk_usd": 500.0, "max_tick_budget_usd": 1200.0},
        proxy_wallet_address="0x" + ("44" * 20),
        onchain_free_usdc_balance=500.0,
        clob_allowance_usdc=500.0,
        balance_verified=True,
        max_daily_realized_loss_usd=Decimal("500.0"),
        max_consecutive_losses=4,
        cooldown_minutes_after_consecutive_losses=30.0,
    )
    assert claim_1101["aborted"] is True
    assert claim_1101["abort_status"] == "ABORTED_CONSECUTIVE_LOSS_COOLDOWN"
    assert claim_1101["loss_stats"]["consecutive_losses"] >= 4


def test_round5_p1_3_ambiguous_redemption_submission_never_resubmits_before_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-5 P1-3] Reproduce both failure modes:
    1) Case A (Server accepted request, HTTP response timed out):
       `relayer_client.execute()` increments `submit_count_a` to 1 and raises `TimeoutError`.
       SQLite saves `submit_state == "REDEEM_SUBMIT_UNKNOWN"`, `redeem_request_id`, and `pending_batch_order_keys`
       with `redeem_tx_id == None`. On the next pass, `poll_settlements_and_redeem` enters reconciliation
       and NEVER calls `execute()` a second time (`submit_count_a == 1`).
    2) Case B (`execute()` returned ID, but saving the returned ID failed):
       `relayer_client.execute()` increments `submit_count_b` to 1 and returns `transaction_id="relayer_tx_b_1"`,
       and the post-submit `on_progress_callback` raises `RuntimeError("DB write failed")`.
       SQLite retains the pre-submit `REDEEM_SUBMIT_UNKNOWN` + `pending_batch_order_keys` record with `redeem_tx_id == None`.
       On the next pass, `poll_settlements_and_redeem` NEVER calls `execute()` a second time (`submit_count_b == 1`).
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]
    observer.fetch_status = lambda allow_in_critical_window=False: {}  # type: ignore[method-assign]
    observer.is_in_critical_window = lambda now_utc=None: False  # type: ignore[method-assign]
    observer.fetch_market_resolution = lambda market_id, condition_id=None: {  # type: ignore[method-assign]
        "resolved": True,
        "official_outcome": "UP",
        "condition_id": condition_id or ("0x00" + ("b1" * 31)),
        "market_protocol": "v2",
        "neg_risk": False,
    }

    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(mode="shadow", private_key=None)
    clob._cached_free_usdc = 100.0
    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)

    submit_count_a = 0

    class _TimeoutAfterAcceptRelayer:
        def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
            nonlocal submit_count_a
            submit_count_a += 1
            raise TimeoutError("Simulated HTTP response timeout after Relayer accepted request")

    redeemer.relayer_client = _TimeoutAfterAcceptRelayer()

    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    now_iso = "2026-11-09T12:30:00+00:00"
    v2_cond_a = "0x00" + ("b1" * 31)
    audit.record_order_execution(
        {
            "timestamp_utc": now_iso,
            "cohort_id": now_iso,
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_r5_timeout_a",
            "paper_order_id": "paper:sig_r5_timeout_a",
            "market_id": "980001",
            "condition_id": v2_cond_a,
            "token_id": "0x02" + ("b1" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 60.0, "paper_filled_quantity": 120.0, "avg_fill_price": 0.50, "cost_usd": 60.0, "fee_usd": 0.0, "total_spent_usd": 60.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 6.0,
                "requested_quantity": "12.00",
                "clob_order_id": "0xord_r5_a",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "12.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 6.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 6.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 401,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=now_iso,
        window_end="2026-11-09T12:45:00+00:00",
    )

    # Case A Pass 1: `execute()` raises TimeoutError -> saves REDEEM_SUBMIT_UNKNOWN + redeem_request_id + pending batch
    res_a1 = engine._execute_and_record_settlement(
        market_id="980001",
        condition_id=v2_cond_a,
        official_outcome="UP",
        paper_payout_usd=120.0,
        paper_realized_pnl_usd=60.0,
        paper_settled_at=now_iso,
    )
    assert submit_count_a == 1
    assert res_a1["redeem_status"] == "REDEEM_PENDING"
    saved_a1 = audit.get_settlement_record("980001")
    assert saved_a1 is not None
    assert saved_a1["submit_state"] == "REDEEM_SUBMIT_UNKNOWN"
    assert saved_a1.get("redeem_request_id") is not None
    assert saved_a1["pending_batch_order_keys"] == ["sig_r5_timeout_a"]
    assert saved_a1.get("redeem_tx_id") is None

    # Case A Pass 2: Next poll MUST NOT call `execute()` a second time (`submit_count_a` stays 1)!
    res_a2 = engine.poll_settlements_and_redeem(force=True)
    assert submit_count_a == 1
    assert len(res_a2) == 1
    assert res_a2[0]["redeem_status"] == "REDEEM_PENDING"
    assert res_a2[0]["submit_state"] == "REDEEM_SUBMIT_UNKNOWN"

    # Case B: `execute()` returns ID, but saving the returned ID fails (raises RuntimeError on 2nd callback)
    submit_count_b = 0

    class _ReturnIdRelayer:
        def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
            nonlocal submit_count_b
            submit_count_b += 1
            return _RelayerExecResponse(
                transaction_id=f"relayer_tx_b_{submit_count_b}",
                transaction_hash=None,
                state="STATE_NEW",
            )

    redeemer.relayer_client = _ReturnIdRelayer()

    v2_cond_b = "0x00" + ("b2" * 31)
    audit.record_order_execution(
        {
            "timestamp_utc": now_iso,
            "cohort_id": now_iso,
            "stream": "ETH_15m",
            "rank": 1,
            "signal_id": "sig_r5_save_fail_b",
            "paper_order_id": "paper:sig_r5_save_fail_b",
            "market_id": "980002",
            "condition_id": v2_cond_b,
            "token_id": "0x02" + ("b2" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 60.0, "paper_filled_quantity": 120.0, "avg_fill_price": 0.50, "cost_usd": 60.0, "fee_usd": 0.0, "total_spent_usd": 60.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 6.0,
                "requested_quantity": "12.00",
                "clob_order_id": "0xord_r5_b",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "12.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 6.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 6.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 402,
            },
            "diff_vs_paper": {},
        },
        underlying="ETH",
        window_start=now_iso,
        window_end="2026-11-09T12:45:00+00:00",
    )

    orig_record_settle = audit.record_settlement_and_redeem
    fail_on_tx_id_save = True

    def _flaky_record_settle(rec: Mapping[str, Any], *, append_jsonl: bool = True) -> None:
        if fail_on_tx_id_save and str(rec.get("market_id")) == "980002" and rec.get("redeem_tx_id"):
            raise RuntimeError("Simulated DB failure when saving returned redeem_tx_id")
        orig_record_settle(rec, append_jsonl=append_jsonl)

    monkeypatch.setattr(audit, "record_settlement_and_redeem", _flaky_record_settle)

    with pytest.raises(RuntimeError, match="Simulated DB failure when saving returned redeem_tx_id"):
        engine._execute_and_record_settlement(
            market_id="980002",
            condition_id=v2_cond_b,
            official_outcome="UP",
            paper_payout_usd=120.0,
            paper_realized_pnl_usd=60.0,
            paper_settled_at=now_iso,
        )

    assert submit_count_b == 1
    saved_b1 = audit.get_settlement_record("980002")
    assert saved_b1 is not None
    assert saved_b1["pending_batch_order_keys"] == ["sig_r5_save_fail_b"]
    assert saved_b1["submit_state"] == "REDEEM_SUBMIT_UNKNOWN"
    assert saved_b1.get("redeem_tx_id") is None

    # Next pass: restore DB writes and poll -> MUST NOT call `execute()` a second time (`submit_count_b` stays 1)!
    fail_on_tx_id_save = False
    engine.poll_settlements_and_redeem(force=True)
    assert submit_count_b == 1


def test_round5_p2_4_pre_confirmation_balance_refresh_never_skips_new_redemption_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-5 P2-4] Reproduce:
    1) Redemption of `$12.00` is submitted while `_cached_free_usdc == 88.0`, returning `REDEEM_PENDING`.
    2) Before the redemption confirms, `query_wallet_balance_and_allowance(refresh_network=True)` runs
       and the wallet balance is STILL `$88.0` (because the redemption has not landed on-chain yet).
    3) Subsequently, the `$12.00` redemption confirms (`REDEEMED`).
    4) Verify `_cached_free_usdc` is updated to `$100.00` (`$88 + $12 = $100`), NEVER stuck at `$88.00`!
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]
    observer.fetch_status = lambda allow_in_critical_window=False: {}  # type: ignore[method-assign]
    observer.is_in_critical_window = lambda now_utc=None: False  # type: ignore[method-assign]
    observer.fetch_market_resolution = lambda market_id, condition_id=None: {  # type: ignore[method-assign]
        "resolved": True,
        "official_outcome": "UP",
        "condition_id": condition_id or ("0x00" + ("b4" * 31)),
        "market_protocol": "v2",
        "neg_risk": False,
    }

    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(mode="shadow", private_key=None)
    clob._cached_free_usdc = 88.0
    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)

    tx_confirmed = False

    class _PendingThenConfirmedRelayer:
        def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
            return _RelayerExecResponse(transaction_id="relayer_tx_r5_bal_12", transaction_hash=None, state="STATE_NEW")

        def poll_until_state(self, **kwargs: Any) -> dict[str, Any] | None:
            if not tx_confirmed:
                return None
            return {
                "id": "relayer_tx_r5_bal_12",
                "state": "STATE_CONFIRMED",
                "transactionHash": "0x" + ("d4" * 32),
            }

    redeemer.relayer_client = _PendingThenConfirmedRelayer()

    import argon4_live.redeemer as red_mod

    class _MockRpcResp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"result": {"blockNumber": "0x404", "status": "0x1"}}

    monkeypatch.setattr(red_mod.requests, "post", lambda *a, **kw: _MockRpcResp())

    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    now_iso = "2026-11-09T13:00:00+00:00"
    v2_cond = "0x00" + ("b4" * 31)
    audit.record_order_execution(
        {
            "timestamp_utc": now_iso,
            "cohort_id": now_iso,
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_r5_bal_12",
            "paper_order_id": "paper:sig_r5_bal_12",
            "market_id": "980004",
            "condition_id": v2_cond,
            "token_id": "0x02" + ("b4" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 60.0, "paper_filled_quantity": 120.0, "avg_fill_price": 0.50, "cost_usd": 60.0, "fee_usd": 0.0, "total_spent_usd": 60.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 6.0,
                "requested_quantity": "12.00",
                "clob_order_id": "0xord_r5_bal_12",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "12.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 6.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 6.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 404,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=now_iso,
        window_end="2026-11-09T13:15:00+00:00",
    )

    # Step 1: Submit $12.00 redemption -> REDEEM_PENDING, balance is $88.00
    s1 = engine._execute_and_record_settlement(
        market_id="980004",
        condition_id=v2_cond,
        official_outcome="UP",
        paper_payout_usd=120.0,
        paper_realized_pnl_usd=60.0,
        paper_settled_at=now_iso,
    )
    assert s1["redeem_status"] == "REDEEM_PENDING"
    assert clob._cached_free_usdc == 88.0

    # Step 2: Balance is refreshed BEFORE confirmation, still returning $88.00!
    clob.query_wallet_balance_and_allowance(refresh_network=True)
    assert clob._cached_free_usdc == 88.0

    # Step 3: Redemption of $12.00 confirms -> `_cached_free_usdc` MUST be credited to $100.00!
    tx_confirmed = True
    s2 = engine.poll_settlements_and_redeem(force=True)
    assert len(s2) == 1
    assert s2[0]["redeem_status"] == "REDEEMED"
    assert s2[0]["redeemed_usdc_amount"] == 12.0
    assert clob._cached_free_usdc == 100.0


def test_round6_p1_1_authoritative_balance_on_redemption_confirm_never_double_credits_after_intervening_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-6 P1-1] Reproduce:
    1) Balance at redemption submission is `$88.00` (`free_usdc_at_submit == 88.0`), returning `REDEEM_PENDING` for `$12.00`.
    2) While redemption is pending, an order spends `$20.00` (`88 -> 68`).
    3) Redemption of `$12.00` lands on-chain, so the balance API (`get_balance_allowance`) rightly returns `$80.00` (`88 - 20 + 12 = 80`).
    4) Verify `apply_confirmed_redemption_credit` uses the real `$80.00` balance directly and NEVER adds `$12.00` a second time to get `$92.00`!
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]
    observer.fetch_status = lambda allow_in_critical_window=False: {}  # type: ignore[method-assign]
    observer.is_in_critical_window = lambda now_utc=None: False  # type: ignore[method-assign]
    observer.fetch_market_resolution = lambda market_id, condition_id=None: {  # type: ignore[method-assign]
        "resolved": True,
        "official_outcome": "UP",
        "condition_id": condition_id or ("0x00" + ("c1" * 31)),
        "market_protocol": "v2",
        "neg_risk": False,
    }

    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    pk = "0x" + ("96" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    assert clob.client is not None
    clob._cached_free_usdc = 88.0
    clob._cached_allowance_usdc = 500.0
    clob.balance_verified = True

    wallet_real_balance_usdc = 88.0
    clob.client.get_balance_allowance = lambda params=None: {  # type: ignore[method-assign]
        "balance": str(int(round(wallet_real_balance_usdc * 1_000_000))),
        "allowance": "500000000",
    }

    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)
    tx_confirmed = False

    class _PendingThenConfirmedRelayer:
        def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
            return _RelayerExecResponse(transaction_id="relayer_tx_r6_12", transaction_hash=None, state="STATE_NEW")

        def poll_until_state(self, **kwargs: Any) -> dict[str, Any] | None:
            if not tx_confirmed:
                return None
            return {
                "id": "relayer_tx_r6_12",
                "state": "STATE_CONFIRMED",
                "transactionHash": "0x" + ("e1" * 32),
            }

    redeemer.relayer_client = _PendingThenConfirmedRelayer()

    import argon4_live.redeemer as red_mod

    class _MockRpcResp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"result": {"blockNumber": "0x501", "status": "0x1"}}

    monkeypatch.setattr(red_mod.requests, "post", lambda *a, **kw: _MockRpcResp())

    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    now_iso = "2026-11-09T14:00:00+00:00"
    v2_cond = "0x00" + ("c1" * 31)
    audit.record_order_execution(
        {
            "timestamp_utc": now_iso,
            "cohort_id": now_iso,
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_r6_redeem_12",
            "paper_order_id": "paper:sig_r6_redeem_12",
            "market_id": "990001",
            "condition_id": v2_cond,
            "token_id": "0x02" + ("c1" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 60.0, "paper_filled_quantity": 120.0, "avg_fill_price": 0.50, "cost_usd": 60.0, "fee_usd": 0.0, "total_spent_usd": 60.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 6.0,
                "requested_quantity": "12.00",
                "clob_order_id": "0xord_r6_12",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "12.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 6.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 6.0,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 501,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=now_iso,
        window_end="2026-11-09T14:15:00+00:00",
    )

    # Step 1: Submit $12.00 redemption when balance is $88.00 -> REDEEM_PENDING
    s1 = engine._execute_and_record_settlement(
        market_id="990001",
        condition_id=v2_cond,
        official_outcome="UP",
        paper_payout_usd=120.0,
        paper_realized_pnl_usd=60.0,
        paper_settled_at=now_iso,
    )
    assert s1["redeem_status"] == "REDEEM_PENDING"
    assert clob._cached_free_usdc == 88.0

    # Step 2: During redemption wait, another order spends $20.00 (88 -> 68), and then $12.00 redemption arrives (68 + 12 = 80)
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.50", "size": "500"}]}  # type: ignore[method-assign]
    clob.client.post_order = lambda *a, **kw: {  # type: ignore[method-assign]
        "success": True,
        "orderID": "0xord_r6_spend_20",
        "status": "MATCHED",
        "takingAmount": "40.00",
        "makingAmount": "20.00",
    }
    spend_res = clob.sign_and_execute_fak(
        token_id="10001",
        approved_stake_usd=200.0,
        live_scale_factor=0.10,
        fee_rate=0.0,
    )
    assert spend_res["live_execution"]["total_spent_usdc"] == 20.0
    assert clob._cached_free_usdc == 68.0

    # Wallet balance API now reflects 88 - 20 + 12 = 80.0 USDC
    wallet_real_balance_usdc = 80.0
    tx_confirmed = True

    # Step 3: Redemption confirms -> must use the real $80.00 balance directly, NEVER $80 + $12 = $92.00!
    s2 = engine.poll_settlements_and_redeem(force=True)
    assert len(s2) == 1
    assert s2[0]["redeem_status"] == "REDEEMED"
    assert s2[0]["redeemed_usdc_amount"] == 12.0
    assert clob._cached_free_usdc == 80.0
    assert clob.balance_verified is True


def test_round6_p2_2_reconciled_timed_out_order_reduces_cached_free_balance_and_avoids_duplicate_deduction(
    tmp_path: Path,
) -> None:
    """[Round-6 P2-2] Reproduce:
    1) Initial wallet balance is `$5.00` (`_cached_free_usdc == 5.0`).
    2) Order 1 (`sig_r6_timeout_1`) times out on submission (`ORDER_STATE_UNKNOWN`), reserving `$5.00`.
    3) `reconcile_pending_orders` confirms Order 1 filled and spent `$4.90` (`0.10` remaining).
    4) Verify `_cached_free_usdc` is reduced to `$0.10` BEFORE releasing the reservation, so the next signal
       requesting `$5.00` sees only `$0.10` available and aborts locally (`ABORTED_ZERO_SIZE`) without ever
       calling `post_order` on the simulated exchange!
    5) Also verify that if the balance was already refreshed to include the fill spend, `apply_reconciled_order_spend`
       does NOT deduct `$4.90` a second time.
    """
    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})

    now_dt = datetime.now(timezone.utc)
    w_start = (now_dt - timedelta(seconds=3)).isoformat()
    w_end = (now_dt + timedelta(minutes=15)).isoformat()
    valid_until = (now_dt + timedelta(seconds=15)).isoformat()

    def _make_state(sig_id: str, mkt_id: str) -> dict[str, Any]:
        return {
            "status": "PAPER_CYCLE_COMPLETE",
            "active": True,
            "last_cohort": w_start,
            "decisions": {
                "BTC_15m": {
                    "status": "SIGNAL",
                    "outcome": "UP",
                    "signal": {
                        "signal_id": sig_id,
                        "stream": "BTC_15m",
                        "underlying": "BTC",
                        "outcome": "UP",
                        "market_id": mkt_id,
                        "decision_at": now_dt.isoformat(),
                        "valid_until": valid_until,
                        "window_start": w_start,
                        "window_end": w_end,
                    },
                }
            },
            "execution": {
                "cohort": {"cohort_id": w_start, "budget_usd": 1200.0},
                "fills": [
                    {
                        "order_id": f"paper:{sig_id}",
                        "signal_id": sig_id,
                        "market_id": mkt_id,
                        "underlying": "BTC",
                        "outcome": "UP",
                        "status": "OPEN",
                        "quantity": "100.00000000",
                        "cost_usd": 49.0,
                        "fee_usd": 0.0,
                        "unfilled_budget_usd": 0.0,
                        "fill_at": now_dt.isoformat(),
                        "book_observed_at": now_dt.isoformat(),
                        "fee_quote": {"fee_model": "polymarket_crypto", "rate": 0.0, "exponent": 1, "taker_only": True},
                        "allocation_authorization": {
                            "policy": "RANKED_CAPACITY_WATERFALL",
                            "plan_id": "cohort:plan_r6",
                            "cohort_id": w_start,
                            "rank": 1,
                            "signal_id": sig_id,
                            "market_id": mkt_id,
                            "approved_stake_usd": 50.0,
                            "cohort_capacity_usd": 500.0,
                        },
                        "venue_market": {
                            "market_id": mkt_id,
                            "condition_id": "0x" + ("c2" * 32),
                            "protocol": "v1",
                            "asset": "BTC",
                            "window_start": w_start,
                            "window_end": w_end,
                            "outcome_ids": {"UP": "10001", "DOWN": "10002"},
                        },
                    }
                ],
            },
        }

    _write_json(state_path, _make_state("sig_r6_timeout_1", "990002"))

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.fetch_status = lambda allow_in_critical_window=False: {}  # type: ignore[method-assign]
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]
    observer.is_in_critical_window = lambda now_utc=None: False  # type: ignore[method-assign]

    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    pk = "0x" + ("97" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
    )
    assert clob.client is not None
    clob._cached_free_usdc = 5.0
    clob._cached_allowance_usdc = 500.0
    clob.balance_verified = True
    clob.client.get_order_book = lambda tid: {"asks": [{"price": "0.49", "size": "500"}]}  # type: ignore[method-assign]

    exchange_remaining_usdc = 5.0
    exchange_rejected_insufficient_balance = False
    post_order_calls = 0

    def _simulated_post_order(signed_order: Any, *a: Any, **kw: Any) -> dict[str, Any]:
        nonlocal exchange_remaining_usdc, exchange_rejected_insufficient_balance, post_order_calls
        post_order_calls += 1
        if post_order_calls == 1:
            # Exchange executes Order 1 for $4.90 (10 shares @ $0.49), leaving $0.10, but HTTP response times out!
            exchange_remaining_usdc = round(exchange_remaining_usdc - 4.90, 6)
            raise TimeoutError("Simulated HTTP timeout after exchange matched $4.90")
        # For Order 2, check if the executor sent a $5.00 order (which exceeds $0.10 remaining on the exchange!)
        ord_obj = getattr(signed_order, "order", None)
        if isinstance(ord_obj, dict):
            maker_amt_atoms = int(ord_obj.get("makerAmount", 0) or 0)
        elif hasattr(ord_obj, "values") and isinstance(getattr(ord_obj, "values", None), dict):
            maker_amt_atoms = int(ord_obj.values.get("makerAmount", 0) or 0)
        else:
            maker_amt_atoms = int(getattr(ord_obj, "makerAmount", 0) or 0)
        order_usdc = round(maker_amt_atoms / 1e6, 6)
        if order_usdc > exchange_remaining_usdc + 1e-6:
            exchange_rejected_insufficient_balance = True
            raise AssertionError(
                f"Exchange rejected order of ${order_usdc} when only ${exchange_remaining_usdc} remained!"
            )
        exchange_remaining_usdc = round(max(0.0, exchange_remaining_usdc - order_usdc), 6)
        return {
            "success": True,
            "orderID": "0xord_r6_second_small",
            "status": "MATCHED",
            "takingAmount": "0.19",
            "makingAmount": str(round(0.19 * 0.49, 4)),
        }

    clob.client.post_order = _simulated_post_order  # type: ignore[method-assign]

    redeemer = AutoRedeemer(mode="shadow", withdrawal_permission=False)
    engine = LiveExecutionEngine(
        config={
            "capital_and_scaling": {"live_scale_factor": 0.10, "fee_rate": 0.0},
            "withdrawal_permission": False,
        },
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    # Step 1: Signal 1 times out -> recorded as ORDER_STATE_UNKNOWN
    recs_1 = engine.process_pipeline_state(_make_state("sig_r6_timeout_1", "990002"), detected_at_local_utc=now_dt)
    assert len(recs_1) == 1
    assert recs_1[0]["live_execution"]["clob_status"] == "ORDER_STATE_UNKNOWN"
    assert post_order_calls == 1

    # Step 2: Reconcile Order 1 -> confirmed 10.00 shares @ $0.49 = $4.90 spent!
    clob.client.get_order = lambda oid: {  # type: ignore[method-assign]
        "id": oid,
        "status": "MATCHED",
        "size_matched": "10.00",
        "price": "0.49",
    }
    clob.client.get_trades = lambda *a, **kw: []  # type: ignore[method-assign]

    reconciled = engine.reconcile_pending_orders(force=True)
    assert len(reconciled) == 1
    assert reconciled[0]["live_execution"]["total_spent_usdc"] == 4.90
    # Cached free balance MUST now be synchronized to $0.10 (NEVER left at $5.00!)
    assert clob._cached_free_usdc == 0.10
    assert clob.balance_verified is True

    # Calling `apply_reconciled_order_spend` a second time for the same order is idempotent (never double-deducts)
    clob.apply_reconciled_order_spend(reconciled[0], reconciled[0]["live_execution"])
    assert clob._cached_free_usdc == 0.10

    # Step 3: Signal 2 arrives requesting $5.00 -> sees $0.10 free balance (NEVER $5.00!),
    # caps `requested_stake_usd` to $0.10 (`0.19` shares @ $0.51 = $0.0969 <= $0.10),
    # and is NEVER rejected by the simulated exchange for exceeding the remaining $0.10 balance!
    recs_2 = engine.process_pipeline_state(_make_state("sig_r6_after_reconcile_2", "990003"), detected_at_local_utc=now_dt)
    assert len(recs_2) == 1
    assert recs_2[0]["live_execution"]["requested_stake_usd"] == 0.10
    assert recs_2[0]["live_execution"]["requested_quantity"] == "0.19"
    assert recs_2[0]["live_execution"]["clob_status"] == "MATCHED"
    assert exchange_rejected_insufficient_balance is False

    # Step 4: Verify that when the balance API already reflects the reconciled order's spend
    # (e.g. balance was $15.00 before order, order spent $4.90, and `get_balance_allowance` returns $10.10),
    # `reconcile_pending_orders` uses the authoritative $10.10 balance and NEVER deducts $4.90 a second time!
    clob._cached_free_usdc = 15.0
    exchange_remaining_usdc = 15.0

    def _timeout_order_3(*a: Any, **kw: Any) -> dict[str, Any]:
        nonlocal exchange_remaining_usdc
        exchange_remaining_usdc = 10.10
        raise TimeoutError("Order 3 timed out after spending $4.90")

    clob.client.post_order = _timeout_order_3  # type: ignore[method-assign]
    recs_3 = engine.process_pipeline_state(_make_state("sig_r6_timeout_3", "990005"), detected_at_local_utc=now_dt)
    assert len(recs_3) == 1
    assert recs_3[0]["live_execution"]["clob_status"] == "ORDER_STATE_UNKNOWN"

    # Authoritative balance API now returns $10.10 (10_100_000 micro-USDC)
    clob.client.get_balance_allowance = lambda params=None: {  # type: ignore[method-assign]
        "balance": "10100000",
        "allowance": "500000000",
    }
    reconciled_3 = engine.reconcile_pending_orders(force=True)
    assert len(reconciled_3) == 1
    assert reconciled_3[0]["live_execution"]["total_spent_usdc"] == 4.90
    # Must be $10.10 (NEVER $10.10 - $4.90 = $5.20!)
    assert clob._cached_free_usdc == 10.10
    assert clob.balance_verified is True


def test_round6_p2_3_real_relayer_v2_client_recovers_unknown_redemption_with_attribution_and_completes_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[Round-6 P2-3] Reproduce end-to-end with the production `_PolymarketRelayerV2Client`:
    1) `AutoRedeemer` is initialized in `live` mode with real credentials so `redeemer.relayer_client` is
       an instance of `_PolymarketRelayerV2Client`.
    2) Pass 1: `POST /submit` succeeds on the server (`submit_http_calls == 1`), but the HTTP response times out
       before returning `transaction_id`, leaving `live_settlements` in `REDEEM_SUBMIT_UNKNOWN` (`redeem_tx_id == None`).
    3) Pass 2 (Restart / Recovery): `_PolymarketRelayerV2Client` queries `GET /transactions`, filters out an
       unrelated transaction (wrong wallet / wrong metadata / wrong calldata), matches our transaction via
       `redeem_request_id` + wallet ownership + calldata attribution, recovers `redeem_tx_id="relayer_real_tx_77"`
       and `redeem_tx_hash`, verifies the on-chain receipt, and completes settlement (`redeem_status == "REDEEMED"`,
       `redeemed_usdc_amount == 15.0`) without calling `POST /submit` a second time (`submit_http_calls == 1`)!
    """
    import argon4_live.redeemer as red_mod

    local_root = tmp_path / "Order"
    upstream_root = tmp_path / "crypto-predict_system"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "configs/argon4_locked_policy.json"
    _write_json(policy_path, {})
    _write_json(state_path, {})

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    observer.fetch_ledger = lambda allow_in_critical_window=False: []  # type: ignore[method-assign]
    observer.fetch_status = lambda allow_in_critical_window=False: {}  # type: ignore[method-assign]
    observer.is_in_critical_window = lambda now_utc=None: False  # type: ignore[method-assign]
    v2_cond = "0x00" + ("c3" * 31)
    observer.fetch_market_resolution = lambda market_id, condition_id=None: {  # type: ignore[method-assign]
        "resolved": True,
        "official_outcome": "UP",
        "condition_id": condition_id or v2_cond,
        "market_protocol": "v2",
        "neg_risk": False,
    }

    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(mode="shadow", private_key=None)
    clob._cached_free_usdc = 100.0

    pk = "0x" + ("98" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    redeemer = AutoRedeemer(
        mode="live",
        private_key=pk,
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        relayer_api_key="rk_live_test_round6",
        relayer_api_secret=cands["signer_eoa"],
        withdrawal_permission=False,
    )
    assert isinstance(redeemer.relayer_client, red_mod._PolymarketRelayerV2Client)

    submit_http_calls = 0
    captured_submitted_payload: dict[str, Any] = {}

    class _MockHttpResp:
        def __init__(self, status_code: int, body: Any) -> None:
            self.status_code = status_code
            self._body = body

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> Any:
            return self._body

    def _mock_requests_get(url: str, params: Any = None, headers: Any = None, timeout: float = 5.0) -> _MockHttpResp:
        if url.endswith("/nonce"):
            return _MockHttpResp(200, {"nonce": "101"})
        if url.endswith("/transactions") or url.endswith("/v1/account/transactions"):
            # Return a list containing:
            # 1) A spoofed/unrelated tx from another wallet or with wrong calldata
            # 2) Our actual submitted transaction with matching metadata, wallet, and calldata
            return _MockHttpResp(
                200,
                {
                    "transactions": [
                        {
                            "id": "relayer_wrong_wallet_tx",
                            "from": "0x" + ("11" * 20),
                            "metadata": captured_submitted_payload.get("metadata", ""),
                            "state": "STATE_CONFIRMED",
                            "transactionHash": "0x" + ("bad0" * 16),
                        },
                        {
                            "id": "relayer_wrong_calldata_tx",
                            "from": cands["signer_eoa"],
                            "metadata": captured_submitted_payload.get("metadata", ""),
                            "depositWalletParams": {
                                "depositWallet": cands["beacon_deposit_wallet"],
                                "calls": [{"target": redeemer.v2_router_address, "data": "0xdeadbeef"}],
                            },
                            "state": "STATE_CONFIRMED",
                            "transactionHash": "0x" + ("bad1" * 16),
                        },
                        {
                            "id": "relayer_real_tx_77",
                            "from": cands["signer_eoa"],
                            "metadata": captured_submitted_payload.get("metadata", ""),
                            "depositWalletParams": captured_submitted_payload.get("depositWalletParams", {}),
                            "state": "STATE_CONFIRMED",
                            "transactionHash": "0x" + ("f7" * 32),
                        },
                    ]
                },
            )
        if url.endswith("/transaction") or "/v1/account/transactions/relayer_real_tx_77" in url:
            return _MockHttpResp(
                200,
                {
                    "id": "relayer_real_tx_77",
                    "state": "STATE_CONFIRMED",
                    "transactionHash": "0x" + ("f7" * 32),
                },
            )
        return _MockHttpResp(404, {})

    def _mock_requests_post(url: str, json: Any = None, headers: Any = None, timeout: float = 8.0) -> _MockHttpResp:
        nonlocal submit_http_calls
        if url.endswith("/submit") or url.endswith("/v1/account/transactions"):
            submit_http_calls += 1
            if isinstance(json, dict):
                captured_submitted_payload.update(dict(json))
            raise TimeoutError("Simulated network drop after Relayer accepted POST /submit")
        if isinstance(json, dict) and json.get("method") == "eth_getTransactionReceipt":
            return _MockHttpResp(200, {"result": {"blockNumber": "0x601", "status": "0x1"}})
        return _MockHttpResp(404, {})

    monkeypatch.setattr(red_mod.requests, "get", _mock_requests_get)
    monkeypatch.setattr(red_mod.requests, "post", _mock_requests_post)

    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 0.10}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    now_iso = "2026-11-09T15:00:00+00:00"
    audit.record_order_execution(
        {
            "timestamp_utc": now_iso,
            "cohort_id": now_iso,
            "stream": "BTC_15m",
            "rank": 1,
            "signal_id": "sig_r6_unknown_redeem",
            "paper_order_id": "paper:sig_r6_unknown_redeem",
            "market_id": "990004",
            "condition_id": v2_cond,
            "token_id": "0x02" + ("c3" * 31),
            "outcome": "UP",
            "market_protocol": "v2",
            "neg_risk": False,
            "timing": {"decision_at": now_iso, "paper_fill_at": now_iso, "live_detected_at": now_iso, "live_signed_at": now_iso, "live_http_sent_at": now_iso, "live_http_ack_at": now_iso, "detection_lag_ms": 1.0, "clob_rtt_ms": 10.0},
            "paper_benchmark": {"approved_stake_usd": 75.0, "paper_filled_quantity": 150.0, "avg_fill_price": 0.50, "cost_usd": 75.0, "fee_usd": 0.0, "total_spent_usd": 75.0},
            "live_execution": {
                "live_scale_factor": 0.10,
                "order_type": "FAK",
                "limit_price": 0.50,
                "requested_stake_usd": 7.5,
                "requested_quantity": "15.00",
                "clob_order_id": "0xord_r6_unk_red",
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": "15.00",
                "avg_fill_price": 0.50,
                "cost_usdc": 7.5,
                "fee_usdc": 0.0,
                "total_spent_usdc": 7.5,
                "unfilled_stake_usdc": 0.0,
                "polygon_tx_hashes": [],
                "onchain_confirmed": True,
                "onchain_block_number": 600,
            },
            "diff_vs_paper": {},
        },
        underlying="BTC",
        window_start=now_iso,
        window_end="2026-11-09T15:15:00+00:00",
    )

    # Pass 1: POST /submit times out -> settlement saved in REDEEM_SUBMIT_UNKNOWN
    r1 = engine._execute_and_record_settlement(
        market_id="990004",
        condition_id=v2_cond,
        official_outcome="UP",
        paper_payout_usd=150.0,
        paper_realized_pnl_usd=75.0,
        paper_settled_at=now_iso,
    )
    assert submit_http_calls == 1
    assert r1["redeem_status"] == "REDEEM_PENDING"
    saved_unknown = audit.get_settlement_record("990004")
    assert saved_unknown is not None
    assert saved_unknown["submit_state"] == "REDEEM_SUBMIT_UNKNOWN"
    assert saved_unknown.get("redeem_tx_id") is None

    # Pass 2 (Simulated Restart with a fresh AutoRedeemer + _PolymarketRelayerV2Client instance):
    # Recovers `relayer_real_tx_77` via `_PolymarketRelayerV2Client.find_transaction_by_metadata` / `get_transactions`
    # and completes settlement to `REDEEMED` without re-submitting!
    redeemer_restarted = AutoRedeemer(
        mode="live",
        private_key=pk,
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        relayer_api_key="rk_live_test_round6",
        relayer_api_secret=cands["signer_eoa"],
        withdrawal_permission=False,
    )
    engine.redeemer = redeemer_restarted

    r2 = engine.poll_settlements_and_redeem(force=True)
    assert submit_http_calls == 1  # Strictly 0 duplicate submissions!
    assert len(r2) == 1
    assert r2[0]["redeem_status"] == "REDEEMED"
    assert r2[0]["redeem_tx_id"] == "relayer_real_tx_77"
    assert r2[0]["redeem_tx_hash"] == "0x" + ("f7" * 32)
    assert r2[0]["redeemed_usdc_amount"] == 15.0
    assert clob._cached_free_usdc == 115.0
    assert audit.is_market_settled("990004") is True


def test_policy_hot_reload_and_dynamic_daily_loss_cap(tmp_path: Path) -> None:
    """Verify that UpstreamObserver.read_policy() hot-reloads on mtime/size changes
    and LiveExecutionEngine._paper_risk_caps() / _live_loss_limits() pick up new policy caps immediately.
    """
    upstream_root = tmp_path / "crypto-predict_system"
    local_root = tmp_path / "Order"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "config/argon4_forward_policy.json"
    _write_json(state_path, {})
    _write_json(
        policy_path,
        {
            "initial_capital_usd": 1000,
            "max_order_usd": 40,
            "max_tick_budget_usd": 80,
            "max_open_risk_usd": 200,
            "max_underlying_open_risk_usd": 120,
            "daily_loss_fraction": 0.08,
        },
    )

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(mode="shadow", private_key="0x" + ("99" * 32), auto_derive_l2_creds=False)
    redeemer = AutoRedeemer(mode="shadow", withdrawal_permission=False)
    engine = LiveExecutionEngine(
        config={"capital_and_scaling": {"live_scale_factor": 1.0}, "withdrawal_permission": False},
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )

    caps_old = engine._paper_risk_caps()
    assert caps_old["max_order_usd"] == 40.0
    loss_cap_old, _, _ = engine._live_loss_limits(Decimal("1.0"))
    assert loss_cap_old == Decimal("80.000000")

    # Modify policy on disk (new size & mtime) without calling read_policy(force=True)
    _write_json(
        policy_path,
        {
            "initial_capital_usd": 1269,
            "max_order_usd": 240,
            "stream_max_order_usd": {"BTC_15m": 240, "ETH_5m": 180, "SOL_5m": 45},
            "max_tick_budget_usd": 380,
            "max_open_risk_usd": 650,
            "max_underlying_open_risk_usd": 450,
            "daily_loss_fraction": 0.15,
        },
    )
    caps_new = engine._paper_risk_caps()
    assert caps_new["max_order_usd"] == 240.0
    assert caps_new["stream_max_order_usd"]["ETH_5m"] == 180.0
    assert caps_new["max_open_risk_usd"] == 650.0
    loss_cap_new, _, _ = engine._live_loss_limits(Decimal("1.0"))
    assert loss_cap_new == Decimal("190.350000")
    observer.close()


def test_fak_killed_no_match_multi_look_retry_and_cohort_nonblocking(tmp_path: Path) -> None:
    """Verify:
    1. Multi-signal cohort dispatches Rank 2 Attempt 0 BEFORE sleeping for Rank 1's FAK retry,
       and defers synchronous Polygon RPC receipt polling during multi-signal dispatch.
    2. Rank 1 (`FAK_KILLED_NO_MATCH` on Attempt 0) retries inside `valid_until`:
       - Look 1 (`ask=0.54 > 0.51`): skips `post_order` and waits for Look 2.
       - Look 2 (`ask=0.49 <= 0.51`): signs & submits a fresh V2 FAK order and fills `MATCHED`!
    3. Knife-catch guard on retry (`ask=0.34 < 0.37`) aborts with `ABORTED_BELOW_MIN_BUY_PRICE` (`0` extra `post_order`).
    """
    pk = "0x" + ("78" * 32)
    cands = derive_polymarket_candidate_wallets(pk)
    upstream_root = tmp_path / "crypto-predict_system"
    local_root = tmp_path / "Order"
    state_path = upstream_root / "paper_runtime/argon4_forward_20261105/pipeline_state.json"
    policy_path = upstream_root / "config/argon4_forward_policy.json"
    _write_json(state_path, {})
    _write_json(
        policy_path,
        {
            "initial_capital_usd": 1269,
            "max_order_usd": 240,
            "stream_max_order_usd": {"BTC_15m": 240, "ETH_5m": 180},
            "max_tick_budget_usd": 380,
            "max_open_risk_usd": 650,
            "max_underlying_open_risk_usd": 450,
            "min_buy_price": 0.37,
            "max_buy_price": 0.515,
            "daily_loss_fraction": 0.15,
        },
    )

    observer = UpstreamObserver(state_path, policy_path, upstream_root=upstream_root)
    audit = AuditStore(
        local_root / "runtime/live/live_audit.sqlite3",
        local_root / "runtime/live/live_orders.jsonl",
        local_root / "runtime/live/live_quota_snapshots.jsonl",
        local_root / "runtime/live/live_settlements.jsonl",
        local_root=local_root,
        upstream_root=upstream_root,
    )
    clob = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
        fresh_observed_book_max_age_seconds=1.5,
    )
    clob._cached_free_usdc = 1000.0
    clob._cached_allowance_usdc = 1000.0
    clob.balance_verified = True
    for tid_k in ("11111", "22222", "33333"):
        clob._tick_sizes[tid_k] = "0.01"
        clob._fee_rates_bps[tid_k] = 1000
        clob._neg_risk[tid_k] = False

    events: list[str] = []
    rpc_calls = 0

    def _forbid_rpc_in_batch(tx_hash: str) -> tuple[bool, int | None]:
        nonlocal rpc_calls
        rpc_calls += 1
        events.append("rpc_receipt")
        return True, 123456

    clob.fetch_tx_receipt_confirmation = _forbid_rpc_in_batch  # type: ignore[method-assign]

    # Book sequence for Rank 1 (`11111`): Attempt 0 ask=0.50, Retry 1 ask=0.54 (>0.51), Retry 2 ask=0.49 (<=0.51)
    book_seq_11111 = iter(
        [
            {"asks": [{"price": "0.50", "size": "100"}]},
            {"asks": [{"price": "0.54", "size": "100"}]},
            {"asks": [{"price": "0.49", "size": "100"}]},
        ]
    )

    def _mock_get_book(tid: str) -> dict[str, Any]:
        events.append(f"book:{tid}")
        if tid == "11111":
            return next(book_seq_11111)
        if tid == "22222":
            return {"asks": [{"price": "0.48", "size": "200"}]}
        return {"asks": [{"price": "0.34", "size": "200"}]}

    assert clob.client is not None
    clob.client.get_order_book = _mock_get_book  # type: ignore[method-assign]

    rank1_posts = 0

    def _mock_post_order(signed_order: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal rank1_posts
        order_dict = signed_order.dict() if hasattr(signed_order, "dict") else getattr(signed_order, "__dict__", signed_order)
        raw_pos = str(order_dict.get("positionId") or order_dict.get("tokenId") or order_dict)
        pos_dec = str(int(raw_pos, 16)) if raw_pos.startswith(("0x", "0X")) else raw_pos
        if "22222" in pos_dec:
            events.append("post:rank2")
            return {
                "success": True,
                "status": "matched",
                "orderID": "0xrank2_ok",
                "takingAmount": "100.00",
                "makingAmount": "48.00",
                "transactionsHashes": ["0x" + ("aa" * 32)],
            }
        if "33333" in pos_dec:
            events.append("post:knife")
            err_400 = RuntimeError("PolyApiException[status_code=400, error_message={'error': 'no orders found to match with FAK order'}]")
            err_400.status_code = 400  # type: ignore[attr-defined]
            raise err_400
        rank1_posts += 1
        events.append(f"post:rank1#{rank1_posts}")
        if rank1_posts == 1:
            return {
                "success": False,
                "status": "canceled",
                "errorMsg": "no orders found to match with FAK order. FAK orders are partially filled or killed if no match is found.",
            }
        return {
            "success": True,
            "status": "matched",
            "orderID": "0xrank1_retry2_ok",
            "takingAmount": "200.00",
            "makingAmount": "98.00",
            "transactionsHashes": ["0x" + ("bb" * 32)],
        }

    clob.client.post_order = _mock_post_order  # type: ignore[method-assign]

    redeemer = AutoRedeemer(mode="live", withdrawal_permission=False)
    engine = LiveExecutionEngine(
        config={
            "capital_and_scaling": {"live_scale_factor": 1.0},
            "order_execution": {
                "fak_no_match_max_retries": 2,
                "fak_no_match_retry_interval_seconds": 1.2,
                "fak_no_match_min_remaining_seconds": 1.8,
            },
            "withdrawal_permission": False,
        },
        observer=observer,
        audit_store=audit,
        clob_executor=clob,
        redeemer=redeemer,
    )
    engine._wait_retry_interval = lambda s: (events.append(f"sleep:{s}") or False)  # type: ignore[method-assign]

    cohort_iso = "2026-11-10T17:35:00+00:00"
    detected_dt = datetime(2026, 11, 10, 17, 35, 10, tzinfo=timezone.utc)
    state_batch = {
        "last_cohort": cohort_iso,
        "decisions": {
            "BTC_15m": {
                "signal": {
                    "signal_id": "sig_r1",
                    "market_id": "70001",
                    "outcome": "UP",
                    "underlying": "BTC",
                    "window_start": cohort_iso,
                    "window_end": "2026-11-10T17:50:00+00:00",
                    "decision_at": "2026-11-10T17:35:09+00:00",
                    "valid_until": "2026-11-10T17:35:20+00:00",
                },
                "book": {"best_ask": 0.50, "observed_at": "2026-11-10T17:35:09.800000+00:00"},
            },
            "ETH_5m": {
                "signal": {
                    "signal_id": "sig_r2",
                    "market_id": "70002",
                    "outcome": "UP",
                    "underlying": "ETH",
                    "window_start": cohort_iso,
                    "window_end": "2026-11-10T17:40:00+00:00",
                    "decision_at": "2026-11-10T17:35:09+00:00",
                    "valid_until": "2026-11-10T17:35:20+00:00",
                },
                "book": {"best_ask": 0.48, "observed_at": "2026-11-10T17:35:09.800000+00:00"},
            },
        },
        "execution": {
            "cohort": {"cohort_id": cohort_iso, "budget_usd": 380.0},
            "fills": [
                {
                    "signal_id": "sig_r1",
                    "order_id": "paper:sig_r1",
                    "stream": "BTC_15m",
                    "market_id": "70001",
                    "outcome": "UP",
                    "underlying": "BTC",
                    "status": "OPEN",
                    "fill_at": "2026-11-10T17:35:09.900000+00:00",
                    "book_observed_at": "2026-11-10T17:35:09.800000+00:00",
                    "book_snapshot": {"best_ask": 0.50, "observed_at": "2026-11-10T17:35:09.800000+00:00"},
                    "quantity": "200.00000000",
                    "cost_usd": 100.0,
                    "fee_usd": 3.75,
                    "fee_quote": {"rate": 0.075},
                    "allocation_authorization": {"rank": 1, "cohort_id": cohort_iso, "approved_stake_usd": 110.0},
                    "venue_market": {
                        "condition_id": "0x" + ("11" * 32),
                        "protocol": "v2",
                        "outcome_ids": {"UP": "11111", "DOWN": "11112"},
                    },
                },
                {
                    "signal_id": "sig_r2",
                    "order_id": "paper:sig_r2",
                    "stream": "ETH_5m",
                    "market_id": "70002",
                    "outcome": "UP",
                    "underlying": "ETH",
                    "status": "OPEN",
                    "fill_at": "2026-11-10T17:35:09.900000+00:00",
                    "book_observed_at": "2026-11-10T17:35:09.800000+00:00",
                    "book_snapshot": {"best_ask": 0.48, "observed_at": "2026-11-10T17:35:09.800000+00:00"},
                    "quantity": "100.00000000",
                    "cost_usd": 48.0,
                    "fee_usd": 1.87,
                    "fee_quote": {"rate": 0.075},
                    "allocation_authorization": {"rank": 2, "cohort_id": cohort_iso, "approved_stake_usd": 55.0},
                    "venue_market": {
                        "condition_id": "0x" + ("22" * 32),
                        "protocol": "v2",
                        "outcome_ids": {"UP": "22222", "DOWN": "22223"},
                    },
                },
            ],
        },
    }

    recs = engine.process_pipeline_state(state_batch, detected_at_local_utc=detected_dt)
    assert len(recs) == 2
    assert recs[0]["signal_id"] == "sig_r1"
    assert recs[1]["signal_id"] == "sig_r2"
    assert recs[0]["live_execution"]["clob_status"] == "MATCHED"
    assert recs[0]["live_execution"]["fak_retry_count"] == 2
    assert recs[1]["live_execution"]["clob_status"] == "MATCHED"
    # Rank 2 dispatched BEFORE any retry sleep, and RPC receipt check was deferred during multi-signal Phase 1!
    assert events.index("post:rank2") < events.index("sleep:1.2")
    assert "rpc_receipt" not in events[: events.index("sleep:1.2")]
    assert rank1_posts == 2  # Look 1 (0.54 > 0.51) skipped post_order; Look 2 (0.49 <= 0.51) posted & matched!
    assert rpc_calls == 1  # Only called after Rank 2 already completed and Rank 1 was the sole remaining retry

    # Knife-catch check on retry: Attempt 0 gets FAK_KILLED_NO_MATCH, Retry 1 sees ask=0.34 < 0.37 -> ABORTED_BELOW_MIN_BUY_PRICE
    knife_books = iter(
        [
            {"asks": [{"price": "0.49", "size": "100"}]},
            {"asks": [{"price": "0.34", "size": "100"}]},
        ]
    )
    clob.client.get_order_book = lambda tid: next(knife_books)  # type: ignore[method-assign]
    state_knife = {
        "last_cohort": cohort_iso,
        "decisions": {
            "ETH_5m": {
                "signal": {
                    "signal_id": "sig_knife",
                    "market_id": "70003",
                    "outcome": "DOWN",
                    "underlying": "ETH",
                    "window_start": cohort_iso,
                    "window_end": "2026-11-10T17:40:00+00:00",
                    "decision_at": "2026-11-10T17:35:09+00:00",
                    "valid_until": "2026-11-10T17:35:20+00:00",
                },
                "book": {"best_ask": 0.49, "observed_at": "2026-11-10T17:35:09.800000+00:00"},
            }
        },
        "execution": {
            "cohort": {"cohort_id": cohort_iso, "budget_usd": 380.0},
            "fills": [
                {
                    "signal_id": "sig_knife",
                    "order_id": "paper:sig_knife",
                    "stream": "ETH_5m",
                    "market_id": "70003",
                    "outcome": "DOWN",
                    "underlying": "ETH",
                    "status": "OPEN",
                    "fill_at": "2026-11-10T17:35:09.900000+00:00",
                    "book_observed_at": "2026-11-10T17:35:09.800000+00:00",
                    "book_snapshot": {"best_ask": 0.49, "observed_at": "2026-11-10T17:35:09.800000+00:00"},
                    "quantity": "80.00000000",
                    "cost_usd": 39.2,
                    "fee_usd": 1.5,
                    "fee_quote": {"rate": 0.075},
                    "allocation_authorization": {"rank": 1, "cohort_id": cohort_iso, "approved_stake_usd": 45.0},
                    "venue_market": {
                        "condition_id": "0x" + ("33" * 32),
                        "protocol": "v2",
                        "outcome_ids": {"UP": "33332", "DOWN": "33333"},
                    },
                }
            ],
        },
    }
    recs_knife = engine.process_pipeline_state(state_knife, detected_at_local_utc=detected_dt)
    assert len(recs_knife) == 1
    assert recs_knife[0]["live_execution"]["clob_status"] == "ABORTED_BELOW_MIN_BUY_PRICE"
    assert "ON_RETRY" in str(recs_knife[0]["live_execution"]["error_code"])
    assert events.count("post:knife") == 1

    # Also verify fresh `observed_best_ask` (`0.2s <= 1.5s`) skips redundant pre-flight `_fetch_live_best_ask` in production live mode
    clob_prod = ClobExecutor(
        mode="live",
        private_key=pk,
        api_key="k",
        api_secret="dGVzdF9zZWNyZXRfYmFzZTY0X3N0cmluZ18xMjM0NTY=",
        api_passphrase="p",
        signature_type=3,
        funder_address=cands["beacon_deposit_wallet"],
        auto_derive_l2_creds=False,
        fresh_observed_book_max_age_seconds=1.5,
    )
    clob_prod._cached_free_usdc = 500.0
    clob_prod._cached_allowance_usdc = 500.0
    clob_prod.balance_verified = True
    clob_prod._tick_sizes["22222"] = "0.01"
    clob_prod._fee_rates_bps["22222"] = 1000
    clob_prod._neg_risk["22222"] = False
    assert clob_prod._has_custom_book_hook() is False
    fetch_book_calls = 0

    def _count_fetch(*args: Any, **kwargs: Any) -> tuple[Decimal | None, str | None]:
        nonlocal fetch_book_calls
        fetch_book_calls += 1
        return Decimal("0.48"), None

    clob_prod._fetch_live_best_ask = _count_fetch  # type: ignore[method-assign]
    assert clob_prod.client is not None
    clob_prod.client.post_order = _mock_post_order  # type: ignore[method-assign]

    # Note: assigning `_fetch_live_best_ask` makes `_has_custom_book_hook()` True; test real fast-path via `client.get_order_book` unbound!
    del clob_prod.__dict__["_fetch_live_best_ask"]
    assert clob_prod._has_custom_book_hook() is False
    fast_res = clob_prod.sign_and_execute_fak(
        token_id="22222",
        approved_stake_usd=55.0,
        live_scale_factor=1.0,
        fee_rate=0.075,
        market_protocol="v2",
        valid_until="2026-11-10T17:35:20+00:00",
        observed_best_ask=0.48,
        book_observed_at="2026-11-10T17:35:09.800000+00:00",
        reference_now_utc=detected_dt,
        reference_mono=time.perf_counter(),
        defer_tx_receipt_check=True,
    )
    assert fast_res["live_execution"]["clob_status"] == "MATCHED"
    observer.close()







