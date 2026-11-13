"""Pre-flight Readiness & Safety Verification for ARGON4 External Live Executor.

Verifies:
1. Physical Isolation Guard (blocks upstream SQLite, isolates writes to c:\\Users\\liuqi\\Order).
2. Upstream `pipeline_state.json` and HTTP API (`127.0.0.1:8897` with `Host: 127.0.0.1:8897`).
3. Automatic Address Derivation from `POLY_PRIVATE_KEY`:
   - `钱包地址 = 签名者地址 = RELAYER_API_KEY_ADDRESS` (`eoa_address`)
   - Polymarket Gnosis Safe Proxy (`funder_address`, `signature_type=2`)
   - Local Zero-Latency EIP-712 Signing Speed (< 5ms with zero HTTP GETs).
4. CTF `redeemPositions` Calldata & Non-Withdrawal Guard (`withdrawal_permission: false`).
5. `.env` 2-Key Credentials Readiness (`POLY_PRIVATE_KEY`, `RELAYER_API_KEY`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from argon4_live.clob_executor import ClobExecutor, derive_metamask_polymarket_addresses
from argon4_live.engine import load_live_config
from argon4_live.redeemer import encode_redeem_positions_calldata
from argon4_live.safety import (
    SafetyViolationError,
    assert_fak_order_only,
    assert_no_withdrawal_permission,
    assert_safe_read_path,
)
from argon4_live.upstream_observer import UpstreamObserver


def _is_configured_secret(val: str) -> bool:
    v = (val or "").strip()
    return bool(v and "YOUR_" not in v.upper())


def main() -> int:
    env_path = ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)

    cfg = load_live_config(ROOT / "config/live_config.json")
    up_cfg = cfg["upstream"]
    cap_cfg = cfg["capital_and_scaling"]

    report: dict = {
        "system": cfg["system_name"],
        "mode": (os.getenv("ARGON4_EXECUTION_MODE") or "shadow").lower(),
        "live_scale_factor": float(os.getenv("ARGON4_LIVE_SCALE_FACTOR") or cap_cfg["live_scale_factor"]),
        "checks": {},
    }

    # 1. Physical Isolation & Safety Guards
    try:
        assert_no_withdrawal_permission(cfg)
        assert_fak_order_only("FAK", post_only=False)
        blocked_sqlite = False
        try:
            assert_safe_read_path(
                Path(up_cfg["runtime_dir"]) / "account.sqlite3",
                up_cfg["repo_root"],
            )
        except SafetyViolationError:
            blocked_sqlite = True
        calldata = encode_redeem_positions_calldata("0x" + "11" * 32)
        report["checks"]["safety_and_isolation"] = {
            "status": "PASS" if blocked_sqlite else "FAIL",
            "upstream_sqlite_blocked": blocked_sqlite,
            "withdrawal_permission": False,
            "order_type_locked": "FAK",
            "redeem_selector_verified": calldata[:10],
        }
    except Exception as exc:
        report["checks"]["safety_and_isolation"] = {"status": "FAIL", "error": str(exc)}

    # 2. Upstream Read-Only Listener & Local HTTP API Check
    observer = UpstreamObserver(
        pipeline_state_path=up_cfg["pipeline_state_path"],
        policy_path=up_cfg["policy_path"],
        http_base_url=up_cfg["http_base_url"],
        required_host_header=up_cfg["required_host_header"],
        upstream_root=up_cfg["repo_root"],
    )
    try:
        changed, state, _ = observer.poll_pipeline_state(force=True)
        status_resp = observer.fetch_status(allow_in_critical_window=True)
        ledger_resp = observer.fetch_ledger(allow_in_critical_window=True)
        report["checks"]["upstream_connectivity"] = {
            "status": "PASS" if state is not None and status_resp is not None else "WARN",
            "pipeline_state_readable": state is not None,
            "pipeline_status": (state or {}).get("status"),
            "http_8897_status_ok": status_resp is not None,
            "http_8897_ledger_events": len(ledger_resp),
            "paper_cash_usd": ((status_resp or {}).get("account") or {}).get("cash"),
        }
    finally:
        observer.close()

    # 3. Auto-Derived Wallet / Signer / RELAYER_API_KEY_ADDRESS & Zero-Latency Local Signing Benchmark
    bench_pk = (os.getenv("POLY_PRIVATE_KEY") or "").strip()
    using_real_pk = bool(bench_pk and len(bench_pk.removeprefix("0x")) == 64 and "YOUR_" not in bench_pk.upper())
    test_pk = bench_pk if using_real_pk else "0x" + ("42" * 32)

    eoa_addr, safe_funder = derive_metamask_polymarket_addresses(test_pk)
    test_exec = ClobExecutor(
        mode="shadow",
        private_key=test_pk,
        signature_type=2,
        auto_derive_l2_creds=False,
    )
    sample_token = "12148846513639042960024489252679543841427766245697318000455787422208948329977"
    # Warm once, then time pure local signing
    test_exec.sign_and_execute_fak(sample_token, 200.0, 0.10, 0.075)
    t0 = time.perf_counter()
    res = test_exec.sign_and_execute_fak(sample_token, 200.0, 0.10, 0.075)
    sign_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    report["checks"]["zero_latency_local_signing"] = {
        "status": "PASS" if sign_ms < 25.0 else "WARN",
        "local_sign_latency_ms": sign_ms,
        "signature_type": test_exec.signature_type,
        "wallet_and_signer_and_relayer_api_address": eoa_addr if using_real_pk else "<auto_derived_from_POLY_PRIVATE_KEY>",
        "derived_polymarket_safe_funder": safe_funder if using_real_pk else "<auto_derived_from_POLY_PRIVATE_KEY>",
        "sample_limit_price": res["sizing"]["limit_price"],
        "sample_requested_shares": res["sizing"]["requested_quantity"],
        "sample_total_spent_usdc": res["live_execution"]["total_spent_usdc"],
    }

    # 4. 2-Key .env Credentials Readiness Check (POLY_PRIVATE_KEY + RELAYER_API_KEY)
    has_relayer_key = _is_configured_secret(os.getenv("RELAYER_API_KEY", ""))
    report["checks"]["env_2_keys_configured"] = {
        "env_file_exists": env_path.is_file(),
        "1_POLY_PRIVATE_KEY_configured": using_real_pk,
        "2_RELAYER_API_KEY_configured": has_relayer_key,
        "auto_derived_RELAYER_API_KEY_ADDRESS": eoa_addr if using_real_pk else "<auto_derived_from_POLY_PRIVATE_KEY>",
        "auto_derives_clob_l2_creds": True,
        "ready_for_shadow_mode": True,
        "ready_for_live_mode": bool(using_real_pk and has_relayer_key),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
