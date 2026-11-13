"""Zero-latency local EIP-712 V2 order signing and Polymarket CLOB V2 FAK execution.

Implements Section 4 (4.1, 4.2) of the ARGON4 External Live Executor Specification
with all 2026 Polymarket V2 protocol and security audit hardening:
  - Uses official `py-clob-client-v2` (`OrderArgsV2`, `SignedOrderV2`, `SignatureTypeV2`)
    and `polymarket-client` (`0.17.0`) supporting both Protocol V2 (`ExchangeV3`,
    `version=3`, `position_id`) and V1/CTF (`StandardExchange` / `NegRiskExchange` V2,
    `version=2`, `token_id`) markets.
  - Supports and verifies Polymarket `Deposit Wallet` (`signature_type=3`, Beacon & UUPS),
    `Gnosis Safe` (`signature_type=2`), `Polymarket Proxy` (`signature_type=1`), and
    `EOA` (`signature_type=0`), failing closed in `live` mode if wallet type cannot be verified.
  - Enforces `min_buy_price = 0.37` and `max_buy_price = 0.515` AND re-checks the 20s
    `valid_until` hard deadline immediately before HTTP transmission.
  - Treats submission timeouts/ambiguous errors as `ORDER_STATE_UNKNOWN` (keeping risk
    quota locked) and reconciles via CLOB `/data/order/<id>` and `/data/trades`.
  - Never defaults empty `{}` responses or missing `takingAmount` to full `MATCHED` fills,
    and requires `receipt.status == "0x1"` for `onchain_confirmed = True`.
  - Converts all integer base-unit collateral balances/allowances by `10**6` (pUSD 6 decimals)
    and fails closed when wallet balance/allowance cannot be verified.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import requests

from eth_account import Account as EthAccount
from polymarket._internal.environment import PRODUCTION_CONFIG
from polymarket._internal.protocol import is_v2_position_id
from polymarket._internal.wallet import (
    derive_beacon_deposit_wallet_address,
    derive_proxy_wallet_address,
    derive_safe_wallet_address,
    derive_uups_deposit_wallet_address,
    signature_type_for,
    try_classify_wallet_type,
)
from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    FeeInfo,
    OrderArgsV2,
    OrderType,
    PartialCreateOrderOptions,
    TradeParams,
)
from py_clob_client_v2.config import get_contract_config as get_clob_v2_contract_config
from py_clob_client_v2.order_builder.constants import BUY
from py_clob_client_v2.order_utils.exchange_order_builder_v2 import ExchangeOrderBuilderV2
from py_clob_client_v2.order_utils.model.order_data_v2 import SignedOrderV2
from py_clob_client_v2.order_utils.model.signature_type_v2 import SignatureTypeV2

from .math_utils import (
    HARD_WINDOW_DEADLINE_SECONDS,
    USD_MICRO,
    check_hard_deadline,
    compute_fee_per_share,
    compute_live_order_sizing,
    format_utc_iso,
    parse_utc_timestamp,
    project_to_calibrated_utc,
    to_decimal,
    validate_clock_projection,
)
from .safety import (
    OFFICIAL_EXCHANGE_V3_ADDRESS,
    OFFICIAL_NEG_RISK_EXCHANGE_V2_ADDRESS,
    OFFICIAL_POLYGON_PUSD_ADDRESS,
    OFFICIAL_STANDARD_EXCHANGE_V2_ADDRESS,
    assert_fak_order_only,
    assert_safe_write_path,
)

logger = logging.getLogger("argon4_live.clob")


def is_valid_private_key(private_key: str | None) -> bool:
    if not private_key:
        return False
    clean = private_key.strip()
    if "YOUR_" in clean.upper():
        return False
    hex_part = clean.removeprefix("0x").removeprefix("0X")
    return len(hex_part) == 64 and all(c in "0123456789abcdefABCDEF" for c in hex_part)


def is_valid_eth_address(addr: str | None) -> bool:
    if not addr:
        return False
    clean = str(addr).strip()
    return (
        len(clean) == 42
        and clean.startswith(("0x", "0X"))
        and all(c in "0123456789abcdefABCDEF" for c in clean[2:])
    )


def derive_polymarket_candidate_wallets(private_key: str, chain_id: int = 137) -> dict[str, Any]:
    """Derive all candidate Polymarket wallet addresses for a given EOA private key."""
    pk = private_key.strip()
    if not pk.startswith(("0x", "0X")):
        pk = "0x" + pk
    signer_eoa = EthAccount.from_key(pk).address
    w_cfg = PRODUCTION_CONFIG.wallet_derivation
    beacon_deposit = str(derive_beacon_deposit_wallet_address(signer_eoa, w_cfg))
    uups_deposit = str(derive_uups_deposit_wallet_address(signer_eoa, w_cfg))
    safe_wallet = str(derive_safe_wallet_address(signer_eoa, w_cfg))
    proxy_wallet = str(derive_proxy_wallet_address(signer_eoa, w_cfg))
    return {
        "signer_eoa": signer_eoa,
        "beacon_deposit_wallet": beacon_deposit,
        "uups_deposit_wallet": uups_deposit,
        "gnosis_safe_wallet": safe_wallet,
        "poly_proxy_wallet": proxy_wallet,
    }


def derive_metamask_polymarket_addresses(private_key: str, chain_id: int = 137) -> tuple[str, str]:
    """Derive `(eoa_signer_address, gnosis_safe_address)` from private key (kept for compatibility)."""
    candidates = derive_polymarket_candidate_wallets(private_key, chain_id)
    return candidates["signer_eoa"], candidates["gnosis_safe_wallet"]


def _parse_base_unit_collateral(raw_val: Any) -> Decimal | None:
    """Convert a CLOB or ERC-20 collateral amount to human USD Decimal (6 decimals).

    Fixes [P1-7]: CLOB `/balance-allowance` returns integer base units (10^6 per $1.00)
    as strings without a decimal point (e.g., `"500000"` = `$0.50`, `"25000000"` = `$25.00`).
    Any integer string or int is ALWAYS divided by `1,000,000`.
    """
    if raw_val is None or isinstance(raw_val, bool):
        return None
    raw_str = str(raw_val).strip()
    if not raw_str:
        return None
    try:
        dec = to_decimal(raw_str, "collateral_amount")
    except ValueError:
        return None
    if "." not in raw_str and "e" not in raw_str.lower():
        scaled = dec / Decimal("1000000")
        if scaled >= Decimal("1000000000000"):
            return Decimal("1000000000000.000000")
        return scaled.quantize(USD_MICRO, rounding=ROUND_DOWN)
    if dec >= Decimal("1000000000000"):
        return Decimal("1000000000000.000000")
    return dec.quantize(USD_MICRO, rounding=ROUND_DOWN)


def normalize_rpc_endpoints(
    primary_url: str | None,
    fallback_urls: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return deduplicated non-empty RPC endpoints ordered with `primary_url` first."""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in [primary_url, *(fallback_urls or ())]:
        u = str(raw or "").strip()
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    return tuple(ordered)


def post_polygon_json_rpc(
    session: requests.Session,
    rpc_urls: Sequence[str],
    payload: Mapping[str, Any],
    *,
    timeout: float = 2.0,
) -> Mapping[str, Any] | None:
    """Execute JSON-RPC POST via persistent keep-alive session (unless `requests.post` is monkeypatched)
    with automatic failover across `rpc_urls` on HTTP 429/5xx, timeout, or JSON-RPC error.
    """
    if not rpc_urls:
        return None
    post_fn = (
        session.post
        if getattr(requests.post, "__module__", "").startswith("requests")
        else requests.post
    )
    req_body = dict(payload)
    for rpc_url in rpc_urls:
        try:
            resp = post_fn(rpc_url, json=req_body, timeout=timeout)
            if resp.status_code != 200:
                continue
            body = resp.json()
            if isinstance(body, Mapping) and not body.get("error"):
                return body
        except Exception:
            continue
    return None


class ClobExecutor:
    """Persistent Polymarket CLOB V2 client with zero-GET local signing and FAK execution."""

    def __init__(
        self,
        mode: str = "shadow",
        host: str = "https://clob.polymarket.com",
        relayer_host: str = "https://relayer-v2.polymarket.com",
        chain_id: int = 137,
        private_key: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        signature_type: int | None = None,
        funder_address: str | None = None,
        polygon_rpc_url: str = "https://polygon-rpc.com",
        polygon_rpc_fallback_urls: Sequence[str] | None = None,
        pusd_collateral_address: str = OFFICIAL_POLYGON_PUSD_ADDRESS,
        default_tick_size: str = "0.01",
        default_clob_fee_rate_bps: int = 1000,
        tick_size_limit_prices: Mapping[str, Any] | None = None,
        min_buy_price: float | Decimal = Decimal("0.37"),
        max_buy_price: float | Decimal = Decimal("0.515"),
        creds_cache_path: str | Path | None = None,
        auto_derive_l2_creds: bool = True,
        verify_wallet_on_startup: bool | None = None,
        fresh_observed_book_max_age_seconds: float = 1.5,
    ) -> None:
        self.mode = str(mode or "shadow").strip().lower()
        if self.mode not in ("shadow", "live"):
            raise ValueError(f"Unsupported execution mode: {self.mode!r} (expected 'shadow' or 'live')")

        self.host = host.rstrip("/")
        self.relayer_host = relayer_host.rstrip("/")
        self.chain_id = int(chain_id)
        self.polygon_rpc_url = polygon_rpc_url
        self.polygon_rpc_urls: tuple[str, ...] = normalize_rpc_endpoints(
            polygon_rpc_url, polygon_rpc_fallback_urls
        )
        self.pusd_collateral_address = pusd_collateral_address
        self.default_tick_size = str(default_tick_size)
        self.default_clob_fee_rate_bps = int(default_clob_fee_rate_bps)
        self.tick_size_limit_prices = dict(tick_size_limit_prices or {"0.01": 0.51, "0.001": 0.515})
        self.min_buy_price = to_decimal(min_buy_price, "min_buy_price")
        self.max_buy_price = to_decimal(max_buy_price, "max_buy_price")
        self.max_book_age_seconds: float = 20.0
        self.fresh_observed_book_max_age_seconds: float = max(0.0, float(fresh_observed_book_max_age_seconds))
        self.creds_cache_path = Path(creds_cache_path).resolve() if creds_cache_path else None

        self._tick_sizes: dict[str, str] = {}
        self._fee_rates_bps: dict[str, int] = {}
        self._neg_risk: dict[str, bool] = {}
        self._preferred_book_token_id: dict[str, str] = {}
        self._rpc_session = requests.Session()

        # In shadow mode without network, start with simulated balance; in live mode start unverified (0.0)
        if self.mode == "shadow":
            self._cached_free_usdc: float = 1000.0
            self._cached_allowance_usdc: float = 1000.0
            self.balance_verified: bool = True
        else:
            self._cached_free_usdc = 0.0
            self._cached_allowance_usdc = 0.0
            self.balance_verified = False
        self._cached_allowances_by_spender: dict[str, float] = {}
        self._has_spender_allowance_map: bool = False
        self._boot_id: str = f"boot_{id(self)}_{time.perf_counter_ns()}"
        self._last_balance_snapshot_seq: int = 0
        self._last_authoritative_balance_seq: int = 0
        self._last_refresh_read_authoritative: bool = False
        self._internal_balance_call_token: int = 0
        self._last_balance_snapshot_utc: datetime | None = None
        self._credited_redemption_keys: set[str] = set()
        self._cumulative_local_spent_usdc: float = 0.0
        self._deducted_order_keys: set[str] = set()
        self._order_balance_baseline: dict[str, tuple[int, int, float, float]] = {}

        self.has_valid_signer = is_valid_private_key(private_key)
        self.signer_address: str = ""
        self.funder_address: str = (funder_address or "").strip()
        self.wallet_type: str = "UNKNOWN"
        self.wallet_verified: bool = False
        self.candidate_wallets: dict[str, str] = {}

        should_probe_wallet = (
            verify_wallet_on_startup
            if verify_wallet_on_startup is not None
            else (self.mode == "live" and auto_derive_l2_creds)
        )

        if self.has_valid_signer:
            assert private_key is not None
            pk_clean = private_key.strip()
            if not pk_clean.startswith(("0x", "0X")):
                pk_clean = "0x" + pk_clean
            self.candidate_wallets = derive_polymarket_candidate_wallets(pk_clean, self.chain_id)
            self.signer_address = self.candidate_wallets["signer_eoa"]
            self._resolve_and_verify_wallet(
                explicit_funder=self.funder_address,
                explicit_sig_type=signature_type,
                probe_network=should_probe_wallet,
            )
        else:
            pk_clean = ""
            self.signature_type = int(signature_type if signature_type is not None else 2)

        self.has_valid_funder = is_valid_eth_address(self.funder_address)

        # Resolve L2 ApiCreds: 1) explicit args -> 2) local cache -> 3) auto-derive via CLOB L1 signature
        creds: ApiCreds | None = None
        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                api_key=api_key.strip(),
                api_secret=api_secret.strip(),
                api_passphrase=api_passphrase.strip(),
            )
        elif self.has_valid_signer:
            creds = self._load_cached_l2_creds(self.signer_address)

        self.client: ClobClient | None = None
        if self.has_valid_signer:
            self.client = ClobClient(
                host=self.host,
                chain_id=self.chain_id,
                key=pk_clean,
                creds=creds,
                signature_type=int(self.signature_type),
                funder=self.funder_address if self.has_valid_funder else None,
            )
            self.client._ClobClient__cached_version = 2
            if creds is None and auto_derive_l2_creds:
                try:
                    derive_fn = getattr(self.client, "create_or_derive_api_key", None) or getattr(
                        self.client, "create_or_derive_api_creds", None
                    )
                    derived_creds = derive_fn() if callable(derive_fn) else None
                    if derived_creds and derived_creds.api_key:
                        creds = derived_creds
                        self.client.set_api_creds(creds)
                        self._save_cached_l2_creds(self.signer_address, creds)
                        logger.info(
                            "Auto-derived Polymarket L2 CLOB API credentials for signer %s (funder %s, wallet_type=%s, sig_type=%d)",
                            self.signer_address,
                            self.funder_address,
                            self.wallet_type,
                            self.signature_type,
                        )
                except Exception as exc:
                    logger.warning("Could not auto-derive L2 API creds at startup: %s", exc)

        self.has_l2_creds = bool(creds and creds.api_key and creds.api_secret and creds.api_passphrase)

        if self.mode == "live":
            if not (self.has_valid_signer and self.has_valid_funder and self.has_l2_creds):
                raise ValueError(
                    "LIVE mode requires a valid POLY_PRIVATE_KEY, verified funder wallet, and L2 CLOB API credentials."
                )
            if not self.wallet_verified:
                raise ValueError(
                    f"LIVE mode blocked: Polymarket wallet type could not be verified for signer {self.signer_address}. "
                    f"Candidates: Deposit(Beacon)={self.candidate_wallets.get('beacon_deposit_wallet')}, "
                    f"Deposit(UUPS)={self.candidate_wallets.get('uups_deposit_wallet')}, "
                    f"Safe={self.candidate_wallets.get('gnosis_safe_wallet')}, "
                    f"Proxy={self.candidate_wallets.get('poly_proxy_wallet')}. "
                    "Ensure your Polymarket wallet is deployed or set POLY_FUNDER_ADDRESS and POLY_SIGNATURE_TYPE in .env."
                )

    def close(self) -> None:
        """Close persistent HTTP sessions held by the CLOB executor."""
        try:
            self._rpc_session.close()
        except Exception:
            pass

    def _post_polygon_rpc(
        self,
        payload: Mapping[str, Any],
        *,
        timeout: float = 2.0,
    ) -> Mapping[str, Any] | None:
        return post_polygon_json_rpc(self._rpc_session, self.polygon_rpc_urls, payload, timeout=timeout)

    def _resolve_and_verify_wallet(
        self,
        explicit_funder: str,
        explicit_sig_type: int | None,
        probe_network: bool,
    ) -> None:
        """Resolve and verify `(funder_address, signature_type, wallet_type)` without blindly assuming Safe."""
        cands = self.candidate_wallets
        if is_valid_eth_address(explicit_funder):
            classified = try_classify_wallet_type(
                signer=self.signer_address,
                wallet=explicit_funder,
                config=PRODUCTION_CONFIG.wallet_derivation,
            )
            if classified is not None:
                self.funder_address = explicit_funder
                self.wallet_type = str(classified)
                self.signature_type = int(
                    explicit_sig_type if explicit_sig_type is not None else signature_type_for(classified)
                )
                self.wallet_verified = True
                return
            # Custom explicit funder address
            self.funder_address = explicit_funder
            self.signature_type = int(explicit_sig_type if explicit_sig_type is not None else 2)
            self.wallet_type = {0: "EOA", 1: "POLY_PROXY", 2: "GNOSIS_SAFE", 3: "DEPOSIT_WALLET"}.get(
                self.signature_type, "CUSTOM"
            )
            self.wallet_verified = explicit_sig_type is not None
            return

        if probe_network:
            # Probe candidate wallets in priority order: Beacon Deposit, UUPS Deposit, Gnosis Safe, Poly Proxy, EOA
            probe_order = [
                ("DEPOSIT_WALLET", 3, cands["beacon_deposit_wallet"]),
                ("DEPOSIT_WALLET", 3, cands["uups_deposit_wallet"]),
                ("GNOSIS_SAFE", 2, cands["gnosis_safe_wallet"]),
                ("POLY_PROXY", 1, cands["poly_proxy_wallet"]),
                ("EOA", 0, cands["signer_eoa"]),
            ]
            deployed_matches: list[tuple[str, int, str, Decimal]] = []
            for w_type, sig_t, addr in probe_order:
                is_dep = True if sig_t == 0 else self._check_contract_deployed(addr)
                pusd_bal = self._query_onchain_erc20_balance(addr, self.pusd_collateral_address) or Decimal("0")
                if (sig_t != 0 and is_dep) or pusd_bal > Decimal("0"):
                    deployed_matches.append((w_type, sig_t, addr, pusd_bal))

            if deployed_matches:
                # Prefer wallet with positive pUSD balance; otherwise first deployed smart wallet
                deployed_matches.sort(key=lambda item: (item[3] > Decimal("0"), item[1] != 0), reverse=True)
                best_type, best_sig, best_addr, _ = deployed_matches[0]
                self.wallet_type = best_type
                self.signature_type = best_sig
                self.funder_address = best_addr
                self.wallet_verified = True
                return

        # Fallback for offline shadow mode or when network probe is disabled
        sig_t = int(explicit_sig_type if explicit_sig_type is not None else 2)
        self.signature_type = sig_t
        if sig_t == 3:
            self.funder_address = cands["beacon_deposit_wallet"]
            self.wallet_type = "DEPOSIT_WALLET"
        elif sig_t == 1:
            self.funder_address = cands["poly_proxy_wallet"]
            self.wallet_type = "POLY_PROXY"
        elif sig_t == 0:
            self.funder_address = cands["signer_eoa"]
            self.wallet_type = "EOA"
        else:
            self.signature_type = 2
            self.funder_address = cands["gnosis_safe_wallet"]
            self.wallet_type = "GNOSIS_SAFE"
        self.wallet_verified = not probe_network

    def _check_contract_deployed(self, address: str) -> bool:
        """Check whether a smart wallet is deployed via Relayer `/deployed` or Polygon RPC `eth_getCode`."""
        if not is_valid_eth_address(address):
            return False
        try:
            resp = requests.get(
                f"{self.relayer_host}/deployed",
                params={"address": address},
                timeout=1.5,
            )
            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, Mapping) and body.get("deployed") is True:
                    return True
        except Exception:
            pass
        try:
            body = self._post_polygon_rpc(
                {"jsonrpc": "2.0", "id": 1, "method": "eth_getCode", "params": [address, "latest"]},
                timeout=1.2,
            )
            if body is not None:
                code = body.get("result")
                if isinstance(code, str) and code.startswith("0x") and len(code) > 2 and int(code, 16) != 0:
                    return True
        except Exception:
            pass
        return False

    def _query_onchain_erc20_balance(self, holder: str, token_address: str) -> Decimal | None:
        if not (is_valid_eth_address(holder) and is_valid_eth_address(token_address)):
            return None
        try:
            padded_addr = holder[2:].lower().rjust(64, "0")
            data_hex = "0x70a08231" + padded_addr
            body = self._post_polygon_rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_call",
                    "params": [{"to": token_address, "data": data_hex}, "latest"],
                },
                timeout=2.0,
            )
            if body is not None:
                hex_val = body.get("result")
                if isinstance(hex_val, str) and hex_val.startswith("0x") and len(hex_val) > 2:
                    atoms = int(hex_val, 16)
                    return (Decimal(atoms) / Decimal("1000000")).quantize(USD_MICRO, rounding=ROUND_DOWN)
        except Exception:
            pass
        return None

    def _load_cached_l2_creds(self, signer_address: str) -> ApiCreds | None:
        if not self.creds_cache_path or not self.creds_cache_path.is_file():
            return None
        try:
            data = json.loads(self.creds_cache_path.read_text(encoding="utf-8"))
            entry = data.get(signer_address.lower())
            if isinstance(entry, Mapping) and entry.get("api_key") and entry.get("api_secret") and entry.get("api_passphrase"):
                return ApiCreds(
                    api_key=str(entry["api_key"]),
                    api_secret=str(entry["api_secret"]),
                    api_passphrase=str(entry["api_passphrase"]),
                )
        except Exception:
            pass
        return None

    def _save_cached_l2_creds(self, signer_address: str, creds: ApiCreds) -> None:
        if not self.creds_cache_path:
            return
        try:
            assert_safe_write_path(self.creds_cache_path)
            self.creds_cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if self.creds_cache_path.is_file():
                try:
                    data = json.loads(self.creds_cache_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data[signer_address.lower()] = {
                "signer_address": signer_address,
                "funder_address": self.funder_address,
                "wallet_type": self.wallet_type,
                "signature_type": self.signature_type,
                "api_key": creds.api_key,
                "api_secret": creds.api_secret,
                "api_passphrase": creds.api_passphrase,
                "derived_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.creds_cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _seed_client_local_caches(
        self,
        token_id: str,
        tick_size: str,
        fee_rate_bps: int,
        neg_risk: bool = False,
    ) -> None:
        """Seed `ClobClient` V2 private dicts so `create_order()` makes ZERO synchronous HTTP GETs."""
        tid = str(token_id)
        self._tick_sizes[tid] = str(tick_size)
        self._fee_rates_bps[tid] = int(fee_rate_bps)
        self._neg_risk[tid] = bool(neg_risk)
        if self.client is not None:
            self.client._ClobClient__tick_sizes[tid] = str(tick_size)
            self.client._ClobClient__neg_risk[tid] = bool(neg_risk)
            self.client._ClobClient__fee_rates[tid] = int(fee_rate_bps)
            self.client._ClobClient__fee_infos[tid] = FeeInfo(
                rate=float(Decimal(int(fee_rate_bps)) / Decimal("10000")),
                exponent=1.0,
            )
            self.client._ClobClient__cached_version = 2

    def get_cached_tick_size(self, token_id: str) -> str:
        return self._tick_sizes.get(str(token_id), self.default_tick_size)

    def prewarm_token_metadata(self, token_ids: list[str], fees_enabled: bool = True) -> dict[str, dict[str, Any]]:
        """Pre-fetch `tick_size`, `fee_rate_bps`, and `neg_risk` outside `T+0s..T+20s`."""
        results: dict[str, dict[str, Any]] = {}
        for raw_tid in token_ids:
            tid = str(raw_tid).strip()
            if not tid:
                continue
            tick = self._tick_sizes.get(tid, self.default_tick_size)
            fee_bps = self._fee_rates_bps.get(tid, self.default_clob_fee_rate_bps if fees_enabled else 0)
            neg_risk = self._neg_risk.get(tid, False)
            if self.client is not None:
                try:
                    tick = str(self.client.get_tick_size(tid))
                    fee_bps = int(self.client.get_fee_rate_bps(tid))
                    if not is_v2_position_id(tid):
                        neg_risk = bool(self.client.get_neg_risk(tid))
                except Exception:
                    pass
            self._seed_client_local_caches(tid, tick, fee_bps, neg_risk=neg_risk)
            results[tid] = {"tick_size": tick, "fee_rate_bps": fee_bps, "neg_risk": neg_risk}
        return results

    def _compute_local_order_hash(
        self,
        signed_order: Any,
        version: int,
        neg_risk: bool,
    ) -> str | None:
        """Compute the EIP-712 V2 order hash locally before HTTP submission."""
        if isinstance(signed_order, Mapping):
            raw_id = signed_order.get("orderID") or signed_order.get("orderId") or signed_order.get("id") or signed_order.get("hash")
            if raw_id:
                return str(raw_id)
        if not isinstance(signed_order, SignedOrderV2) or self.client is None:
            return None
        try:
            contract_cfg = get_clob_v2_contract_config(self.chain_id)
            exchange_address = (
                contract_cfg.exchange_v3
                if version == 3
                else contract_cfg.neg_risk_exchange_v2
                if neg_risk
                else contract_cfg.exchange_v2
            )
            builder = ExchangeOrderBuilderV2(
                exchange_address,
                self.chain_id,
                self.client.signer,
                domain_version=str(version),
            )
            typed_data = builder.build_order_typed_data(signed_order)
            return builder.build_order_hash(typed_data)
        except Exception:
            return None

    @staticmethod
    def _build_aborted_result(
        sizing: Mapping[str, Any],
        order_type_str: str,
        clob_status: str,
        error_code: str,
        live_signed_at: datetime,
        clock_projection: Mapping[str, Any] | None = None,
        live_http_sent_at: datetime | None = None,
        fee_rate: float | Decimal | None = None,
    ) -> dict[str, Any]:
        sent_dt = live_http_sent_at or project_to_calibrated_utc(datetime.now(timezone.utc), clock_projection)
        now_cal = format_utc_iso(sent_dt)
        exec_dict: dict[str, Any] = {
            "live_scale_factor": sizing["live_scale_factor"],
            "order_type": order_type_str,
            "limit_price": sizing["limit_price"],
            "requested_stake_usd": sizing["requested_stake_usd"],
            "requested_quantity": sizing["requested_quantity"],
            "clob_order_id": None,
            "clob_status": clob_status,
            "error_code": error_code,
            "filled_quantity": "0.00",
            "avg_fill_price": 0.0,
            "cost_usdc": 0.0,
            "fee_usdc": 0.0,
            "total_spent_usdc": 0.0,
            "unfilled_stake_usdc": sizing["requested_stake_usd"],
            "polygon_tx_hashes": [],
            "onchain_confirmed": False,
            "onchain_block_number": None,
        }
        if fee_rate is not None:
            exec_dict["fee_rate"] = float(fee_rate)
        return {
            "sizing": sizing,
            "live_signed_at": format_utc_iso(live_signed_at),
            "live_http_sent_at": now_cal,
            "live_http_ack_at": now_cal,
            "clob_rtt_ms": 0.0,
            "live_execution": exec_dict,
        }

    @staticmethod
    def _parse_book_timestamp(raw_ts: Any) -> datetime | None:
        """Parse an orderbook timestamp (ISO-8601 string, epoch seconds, or epoch milliseconds) into UTC datetime."""
        if raw_ts is None:
            return None
        if isinstance(raw_ts, datetime):
            return raw_ts.astimezone(timezone.utc) if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
        if isinstance(raw_ts, (int, float)):
            val = float(raw_ts)
            if val > 1e11:
                val /= 1000.0
            if val > 1e8:
                return datetime.fromtimestamp(val, tz=timezone.utc)
            return None
        s = str(raw_ts).strip()
        if not s:
            return None
        try:
            val = float(s)
            if val > 1e11:
                val /= 1000.0
            if val > 1e8:
                return datetime.fromtimestamp(val, tz=timezone.utc)
        except ValueError:
            pass
        try:
            return parse_utc_timestamp(s)
        except ValueError:
            return None

    def _has_custom_book_hook(self) -> bool:
        if self.client is None:
            return False
        client_dict = getattr(self.client, "__dict__", {}) or {}
        return any(
            name in client_dict or not isinstance(self.client, ClobClient)
            for name in ("get_order_book", "get_price", "get_last_trade_price", "get_midpoint")
        )

    def _ordered_book_token_candidates(self, tid: str, raw_tid: str) -> tuple[str, ...]:
        preferred = self._preferred_book_token_id.get(tid)
        base = (tid, raw_tid) if raw_tid != tid else (tid,)
        if preferred and preferred in base:
            return (preferred, *(c for c in base if c != preferred))
        return base

    def _fetch_live_best_ask(
        self,
        tid: str,
        raw_tid: str,
        now_dt: datetime | None = None,
        max_book_age_seconds: float | None = None,
    ) -> tuple[Decimal | None, str | None]:
        """Query CLOB orderbook/price for a real-time, non-stale best ask (`[P1-R3-4]`).

        Returns:
            `(best_ask_decimal, error_reason)`
        """
        if self.client is None:
            return None, "CLOB_CLIENT_NOT_INITIALIZED"
        has_custom_book_hook = self._has_custom_book_hook()
        if self.mode != "live" and not has_custom_book_hook:
            return None, None

        max_age = float(max_book_age_seconds if max_book_age_seconds is not None else self.max_book_age_seconds)
        stale_error: str | None = None
        empty_error: str | None = None
        last_exc_error: str | None = None
        candidates = self._ordered_book_token_candidates(tid, raw_tid)

        # 1. Try get_order_book(token_id)
        for token_candidate in candidates:
            try:
                book_fn = getattr(self.client, "get_order_book", None)
                if callable(book_fn):
                    book = book_fn(token_candidate)
                    if book is None:
                        continue
                    raw_book_ts = (
                        getattr(book, "timestamp", None)
                        or getattr(book, "observed_at", None)
                        or (
                            book.get("timestamp")
                            or book.get("observed_at")
                            or book.get("updated_at")
                            if isinstance(book, Mapping)
                            else None
                        )
                    )
                    if raw_book_ts is not None:
                        book_dt = self._parse_book_timestamp(raw_book_ts)
                        if book_dt is None:
                            stale_error = f"INVALID_LIVE_ORDERBOOK_TIMESTAMP:{raw_book_ts}"
                            continue
                        if now_dt is not None:
                            age_sec = (now_dt - book_dt).total_seconds()
                            if age_sec > max_age:
                                stale_error = f"STALE_LIVE_ORDERBOOK:age={age_sec:.2f}s>max={max_age:.2f}s"
                                continue

                    asks = getattr(book, "asks", None)
                    if asks is None and isinstance(book, Mapping):
                        asks = book.get("asks")
                        if asks is None and book.get("best_ask") is not None:
                            val = to_decimal(book["best_ask"], "best_ask")
                            if val > Decimal("0"):
                                self._preferred_book_token_id[tid] = token_candidate
                                return val, None
                    if isinstance(asks, list):
                        if not asks:
                            empty_error = "EMPTY_LIVE_ORDERBOOK_ASKS"
                            continue
                        ask_prices: list[Decimal] = []
                        for a in asks:
                            p_raw = getattr(a, "price", None)
                            if p_raw is None and isinstance(a, Mapping):
                                p_raw = a.get("price")
                            if p_raw is not None:
                                try:
                                    p_dec = to_decimal(p_raw, "ask_price")
                                    if p_dec > Decimal("0"):
                                        ask_prices.append(p_dec)
                                except ValueError:
                                    pass
                        if ask_prices:
                            self._preferred_book_token_id[tid] = token_candidate
                            return min(ask_prices), None
                        empty_error = "EMPTY_LIVE_ORDERBOOK_ASKS"
            except Exception as exc:
                last_exc_error = f"GET_ORDER_BOOK_ERROR:{type(exc).__name__}:{str(exc)[:120]}"

        if stale_error or empty_error:
            return None, stale_error or empty_error

        # 2. Fallback to get_price(token_id, side="BUY")
        for token_candidate in candidates:
            try:
                price_fn = getattr(self.client, "get_price", None)
                if callable(price_fn):
                    try:
                        res = price_fn(token_candidate, side=BUY)
                    except TypeError:
                        res = price_fn(token_candidate, BUY)
                    if isinstance(res, Mapping):
                        raw_p_ts = res.get("timestamp") or res.get("observed_at")
                        if raw_p_ts is not None:
                            p_dt = self._parse_book_timestamp(raw_p_ts)
                            if p_dt is None:
                                stale_error = f"INVALID_LIVE_PRICE_TIMESTAMP:{raw_p_ts}"
                                continue
                            if now_dt is not None and (now_dt - p_dt).total_seconds() > max_age:
                                stale_error = f"STALE_LIVE_PRICE:age={(now_dt - p_dt).total_seconds():.2f}s>max={max_age:.2f}s"
                                continue
                        raw_p = res.get("price")
                    else:
                        raw_p = res
                    if raw_p is not None:
                        p_dec = to_decimal(raw_p, "live_price")
                        if p_dec > Decimal("0"):
                            self._preferred_book_token_id[tid] = token_candidate
                            return p_dec, None
            except Exception as exc:
                if not last_exc_error:
                    last_exc_error = f"GET_PRICE_ERROR:{type(exc).__name__}:{str(exc)[:120]}"

        return None, stale_error or empty_error or last_exc_error or "LIVE_ORDERBOOK_UNAVAILABLE"

    def sign_and_execute_fak(
        self,
        token_id: str,
        approved_stake_usd: float | Decimal,
        live_scale_factor: float | Decimal,
        fee_rate: float | Decimal,
        clock_projection: Mapping[str, Any] | None = None,
        quota_cap_usd: float | Decimal | None = None,
        paper_benchmark: Mapping[str, Any] | None = None,
        market_protocol: str = "v1",
        neg_risk: bool = False,
        valid_until: str | datetime | None = None,
        window_start: str | datetime | None = None,
        hard_deadline_seconds: float = HARD_WINDOW_DEADLINE_SECONDS,
        observed_best_ask: float | Decimal | None = None,
        book_observed_at: str | datetime | None = None,
        min_buy_price: float | Decimal | None = None,
        max_buy_price: float | Decimal | None = None,
        reference_now_utc: datetime | None = None,
        reference_mono: float | None = None,
        pre_send_callback: Any | None = None,
        skip_preflight_book_fetch: bool = False,
        defer_tx_receipt_check: bool = False,
    ) -> dict[str, Any]:
        """Sign a V2 order locally with zero HTTP GETs and submit as `OrderType.FAK`.

        Enforces:
          - [P1-1] V2 `OrderArgsV2` + `SignedOrderV2` (`version=3` for V2 `position_id`, `version=2` for CTF `token_id`).
          - [P1-2] / [P1-A] Persists `local_order_hash` via `pre_send_callback` BEFORE `post_order`; ambiguous transport/timeout errors return `ORDER_STATE_UNKNOWN`.
          - [P1-3] Strict response parsing (`{}` or missing `takingAmount` is never treated as a full fill).
          - [P1-5] Fail-closed `valid_until` & 20s window hard-cap check at entry, after signing, and immediately before HTTP `post_order`.
          - [Design Gap #4 / Issue #7] Enforces `min_buy_price` (`0.37`) and `max_buy_price` (`0.515`) against live orderbook, observed snapshot, and paper fill price before sending.
        """
        def _current_cal_utc() -> datetime:
            if reference_now_utc is not None and reference_mono is not None:
                base_dt = reference_now_utc + timedelta(seconds=max(0.0, time.perf_counter() - reference_mono))
            elif reference_now_utc is not None:
                base_dt = reference_now_utc
            else:
                base_dt = datetime.now(timezone.utc)
            return project_to_calibrated_utc(base_dt, clock_projection)

        tid_raw = str(token_id).strip()
        is_v2_mkt = str(market_protocol or "").strip().lower() == "v2" or is_v2_position_id(tid_raw)
        tid = str(int(tid_raw, 16)) if tid_raw.startswith(("0x", "0X")) else tid_raw
        tick_size = self.get_cached_tick_size(tid_raw) if tid_raw in self._tick_sizes else self.get_cached_tick_size(tid)
        clob_fee_bps = self._fee_rates_bps.get(
            tid,
            self.default_clob_fee_rate_bps if to_decimal(fee_rate, "fee_rate") > 0 else 0,
        )
        effective_neg_risk = False if is_v2_mkt else bool(neg_risk or self._neg_risk.get(tid, False))
        order_version = 3 if is_v2_mkt else 2
        self._seed_client_local_caches(tid, tick_size, clob_fee_bps, neg_risk=effective_neg_risk)

        sizing = compute_live_order_sizing(
            approved_stake_usd=approved_stake_usd,
            live_scale_factor=live_scale_factor,
            fee_rate=fee_rate,
            tick_size=tick_size,
            quota_cap_usd=quota_cap_usd,
            tick_size_limit_prices=self.tick_size_limit_prices,
        )

        # Polymarket CLOB FAK (market buy) requires makerAmount (size * limit_price) to have at most 2 decimals ($0.01)
        if (
            self.mode == "live"
            and self.client is not None
            and "post_order" not in getattr(self.client, "__dict__", {})
            and sizing["size_shares_dec"] >= Decimal("1.00")
        ):
            adj_shares = sizing["size_shares_dec"]
            lp_dec = sizing["limit_price_dec"]
            cent = Decimal("0.01")
            while adj_shares > Decimal("0") and (adj_shares * lp_dec) != (adj_shares * lp_dec).quantize(
                cent, rounding=ROUND_DOWN
            ):
                adj_shares = (adj_shares - cent).quantize(cent, rounding=ROUND_DOWN)
            if adj_shares > Decimal("0") and adj_shares != sizing["size_shares_dec"]:
                fps_dec = sizing["fee_per_share_dec"]
                adj_cost = (adj_shares * lp_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
                adj_fee = (adj_shares * fps_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
                adj_spent = (adj_cost + adj_fee).quantize(USD_MICRO, rounding=ROUND_DOWN)
                req_stake_dec = to_decimal(sizing["requested_stake_usd"], "requested_stake_usd")
                sizing["size_shares_dec"] = adj_shares
                sizing["size_shares"] = float(adj_shares)
                sizing["requested_quantity"] = format(adj_shares, ".2f")
                sizing["est_cost_usdc"] = float(adj_cost)
                sizing["est_fee_usdc"] = float(adj_fee)
                sizing["est_total_spent_usdc"] = float(adj_spent)
                sizing["est_unfilled_stake_usdc"] = float(
                    max(Decimal("0"), (req_stake_dec - adj_spent).quantize(USD_MICRO, rounding=ROUND_DOWN))
                )

        # Enforce FAK-only guard before any signing or network dispatch
        order_type_str = assert_fak_order_only(OrderType.FAK, post_only=False)
        try:
            validate_clock_projection(clock_projection)
        except Exception as exc:
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_INVALID_CLOCK",
                error_code=f"INVALID_CLOCK_PROJECTION:{exc}",
                live_signed_at=datetime.now(timezone.utc),
                clock_projection=clock_projection,
                fee_rate=fee_rate,
            )

        now_dt = _current_cal_utc()

        # Fail-closed validation of mandatory deadline and optional window ceiling in ALL modes
        if valid_until is None or (isinstance(valid_until, str) and not valid_until.strip()):
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_INVALID_DEADLINE",
                error_code="MISSING_VALID_UNTIL",
                live_signed_at=now_dt,
                clock_projection=clock_projection,
                fee_rate=fee_rate,
            )
        try:
            valid_until_dt = parse_utc_timestamp(valid_until)
            hard_cap_s = float(hard_deadline_seconds)
            if not (0.0 < hard_cap_s <= HARD_WINDOW_DEADLINE_SECONDS):
                raise ValueError(f"hard_deadline_seconds={hard_cap_s} out of bounds (0, {HARD_WINDOW_DEADLINE_SECONDS}]")
            if window_start is not None and not (isinstance(window_start, str) and not str(window_start).strip()):
                w_start_dt = parse_utc_timestamp(window_start)
                if not (w_start_dt < valid_until_dt <= (w_start_dt + timedelta(seconds=hard_cap_s))):
                    raise ValueError(
                        f"valid_until={format_utc_iso(valid_until_dt)} outside "
                        f"(window_start={format_utc_iso(w_start_dt)}, +{hard_cap_s}s]"
                    )
        except Exception as exc:
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_INVALID_DEADLINE",
                error_code=f"INVALID_DEADLINE:{exc}",
                live_signed_at=now_dt,
                clock_projection=clock_projection,
                fee_rate=fee_rate,
            )

        within_deadline, _ = check_hard_deadline(now_dt, valid_until_dt)
        if not within_deadline:
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_PAST_VALID_UNTIL",
                error_code=f"HARD_DEADLINE_EXCEEDED_PRE_BOOK:now={format_utc_iso(now_dt)}>=valid_until={format_utc_iso(valid_until_dt)}",
                live_signed_at=now_dt,
                clock_projection=clock_projection,
                fee_rate=fee_rate,
            )

        if sizing["size_shares_dec"] <= Decimal("0"):
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="SKIPPED_ZERO_QUOTA",
                error_code="QUOTA_EXHAUSTED_OR_BELOW_MIN_SHARE",
                live_signed_at=now_dt,
                clock_projection=clock_projection,
            )

        # Enforce min_buy_price (0.37) and max_buy_price (0.515) floor/ceiling before signing & sending
        min_p = to_decimal(min_buy_price, "min_buy_price") if min_buy_price is not None else self.min_buy_price
        max_p = to_decimal(max_buy_price, "max_buy_price") if max_buy_price is not None else self.max_buy_price
        limit_p = sizing["limit_price_dec"]
        if limit_p < min_p:
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_BELOW_MIN_BUY_PRICE",
                error_code=f"LIMIT_PRICE_BELOW_MIN:{limit_p}<{min_p}",
                live_signed_at=now_dt,
                clock_projection=clock_projection,
            )
        if limit_p > max_p:
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_ABOVE_MAX_BUY_PRICE",
                error_code=f"LIMIT_PRICE_ABOVE_MAX:{limit_p}>{max_p}",
                live_signed_at=now_dt,
                clock_projection=clock_projection,
            )

        # [P1-R3-4] Validate orderbook freshness (`book_observed_at`) and require a valid, fresh orderbook in live mode!
        eff_book_obs_at = book_observed_at
        obs_age_sec: float | None = None
        if eff_book_obs_at is None and isinstance(paper_benchmark, Mapping):
            eff_book_obs_at = paper_benchmark.get("book_observed_at")
        if eff_book_obs_at is not None:
            obs_dt = self._parse_book_timestamp(eff_book_obs_at)
            if obs_dt is None:
                return self._build_aborted_result(
                    sizing=sizing,
                    order_type_str=order_type_str,
                    clob_status="ABORTED_STALE_ORDERBOOK",
                    error_code=f"INVALID_BOOK_OBSERVED_AT:{eff_book_obs_at}",
                    live_signed_at=now_dt,
                    clock_projection=clock_projection,
                )
            obs_age_sec = (now_dt - obs_dt).total_seconds()
            if obs_age_sec > self.max_book_age_seconds:
                return self._build_aborted_result(
                    sizing=sizing,
                    order_type_str=order_type_str,
                    clob_status="ABORTED_STALE_ORDERBOOK",
                    error_code=f"STALE_BOOK_OBSERVED_AT:age={obs_age_sec:.2f}s>max={self.max_book_age_seconds:.2f}s",
                    live_signed_at=now_dt,
                    clock_projection=clock_projection,
                )

        # Fast-path when upstream Paper just verified the exact Polymarket `/book` `< 1.5s` ago (or caller just fetched it on retry)
        parsed_observed_ask: Decimal | None = None
        if observed_best_ask is not None:
            try:
                cand_ask = to_decimal(observed_best_ask, "observed_best_ask")
                if cand_ask > Decimal("0"):
                    parsed_observed_ask = cand_ask
            except ValueError:
                parsed_observed_ask = None

        can_reuse_fresh_observed_book = (
            parsed_observed_ask is not None
            and (
                skip_preflight_book_fetch
                or (
                    self.mode == "live"
                    and not self._has_custom_book_hook()
                    and obs_age_sec is not None
                    and -1.0 <= obs_age_sec <= self.fresh_observed_book_max_age_seconds
                )
            )
        )

        # Query real-time CLOB orderbook best ask and enforce freshness + availability in live mode ([P1-R3-4])
        if can_reuse_fresh_observed_book:
            live_book_ask, live_book_err = parsed_observed_ask, None
        else:
            live_book_ask, live_book_err = self._fetch_live_best_ask(
                tid,
                tid_raw,
                now_dt=now_dt,
                max_book_age_seconds=self.max_book_age_seconds,
            )
        if self.mode == "live":
            if live_book_ask is None or live_book_ask <= Decimal("0"):
                abort_st = (
                    "ABORTED_STALE_ORDERBOOK"
                    if live_book_err and "STALE" in str(live_book_err).upper()
                    else "ABORTED_ORDERBOOK_UNAVAILABLE"
                )
                return self._build_aborted_result(
                    sizing=sizing,
                    order_type_str=order_type_str,
                    clob_status=abort_st,
                    error_code=str(live_book_err or "LIVE_ORDERBOOK_UNAVAILABLE"),
                    live_signed_at=now_dt,
                    clock_projection=clock_projection,
                )
        elif live_book_err and "STALE" in str(live_book_err).upper():
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_STALE_ORDERBOOK",
                error_code=str(live_book_err),
                live_signed_at=now_dt,
                clock_projection=clock_projection,
            )

        candidate_prices: list[tuple[str, Decimal]] = []
        if live_book_ask is not None and live_book_ask > Decimal("0"):
            candidate_prices.append(("live_orderbook_best_ask", live_book_ask))
        if observed_best_ask is not None:
            try:
                ask_dec = to_decimal(observed_best_ask, "observed_best_ask")
                if ask_dec > Decimal("0"):
                    candidate_prices.append(("observed_best_ask", ask_dec))
            except ValueError:
                pass
        if self.mode != "live" and isinstance(paper_benchmark, Mapping) and paper_benchmark.get("avg_fill_price") is not None:
            try:
                pb_dec = to_decimal(paper_benchmark["avg_fill_price"], "avg_fill_price")
                if pb_dec > Decimal("0"):
                    candidate_prices.append(("paper_avg_fill_price", pb_dec))
            except ValueError:
                pass

        for label, p_val in candidate_prices:
            if p_val < min_p:
                return self._build_aborted_result(
                    sizing=sizing,
                    order_type_str=order_type_str,
                    clob_status="ABORTED_BELOW_MIN_BUY_PRICE",
                    error_code=f"{label.upper()}_BELOW_MIN_BUY_PRICE:{p_val}<{min_p}",
                    live_signed_at=now_dt,
                    clock_projection=clock_projection,
                )

        if is_v2_mkt:
            order_args = OrderArgsV2(
                position_id=tid,
                price=float(sizing["limit_price"]),
                size=float(sizing["size_shares"]),
                side=BUY,
            )
        else:
            order_args = OrderArgsV2(
                token_id=tid,
                price=float(sizing["limit_price"]),
                size=float(sizing["size_shares"]),
                side=BUY,
            )
        partial_options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=effective_neg_risk,
            version=order_version,
        )

        signed_order: Any = None
        local_order_hash: str | None = None
        if self.client is not None:
            try:
                signed_order = self.client.create_order(order_args, options=partial_options)
                local_order_hash = self._compute_local_order_hash(signed_order, order_version, effective_neg_risk)
            except Exception as exc:
                # [P1-R4-1]: Distinguish pre-send local signing failure from post-send ORDER_STATE_UNKNOWN!
                # Never call pre_send_callback or post_order when local signing fails.
                return self._build_aborted_result(
                    sizing=sizing,
                    order_type_str=order_type_str,
                    clob_status="ABORTED_LOCAL_SIGN_FAILED",
                    error_code=f"LOCAL_SIGN_ERROR:{type(exc).__name__}:{str(exc)[:160]}",
                    live_signed_at=_current_cal_utc(),
                    clock_projection=clock_projection,
                    fee_rate=fee_rate,
                )
        else:
            raw_sig = f"{tid}:{sizing['limit_price']}:{sizing['requested_quantity']}:{time.time_ns()}"
            signed_order = {"shadow_digest": "0x" + hashlib.sha256(raw_sig.encode()).hexdigest()}

        live_signed_at = _current_cal_utc()
        sizing["free_usdc_before_order"] = round(float(self._cached_free_usdc), 6)
        sizing["balance_seq_before_order"] = int(self._last_balance_snapshot_seq)
        sizing["auth_balance_seq_before_order"] = int(self._last_authoritative_balance_seq)
        sizing["local_spent_before_order"] = round(float(self._cumulative_local_spent_usdc), 6)
        if local_order_hash:
            self._order_balance_baseline[str(local_order_hash)] = (
                int(self._last_balance_snapshot_seq),
                int(self._last_authoritative_balance_seq),
                round(float(self._cached_free_usdc), 6),
                round(float(self._cumulative_local_spent_usdc), 6),
            )

        # [P1-5] Re-check 20-second hard deadline immediately after signing (in both shadow and live modes)
        within_deadline, _ = check_hard_deadline(live_signed_at, valid_until_dt)
        if not within_deadline:
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_PAST_VALID_UNTIL",
                error_code=f"HARD_DEADLINE_EXCEEDED_PRE_SEND:signed_at={format_utc_iso(live_signed_at)}>=valid_until={format_utc_iso(valid_until_dt)}",
                live_signed_at=live_signed_at,
                clock_projection=clock_projection,
                fee_rate=fee_rate,
            )

        # Shadow mode: local V2 sign completed, simulate FAK match without network order submission
        if self.mode == "shadow":
            t_sent = _current_cal_utc()
            t_ack = _current_cal_utc()
            bench_price = (
                live_book_ask
                if live_book_ask is not None and live_book_ask > Decimal("0")
                else to_decimal(paper_benchmark.get("avg_fill_price"), "avg_fill_price")
                if isinstance(paper_benchmark, Mapping) and paper_benchmark.get("avg_fill_price")
                else sizing["limit_price_dec"]
            )
            if bench_price <= 0 or bench_price > sizing["limit_price_dec"]:
                bench_price = sizing["limit_price_dec"]
            if bench_price < min_p:
                return self._build_aborted_result(
                    sizing=sizing,
                    order_type_str=order_type_str,
                    clob_status="ABORTED_BELOW_MIN_BUY_PRICE",
                    error_code=f"BENCH_PRICE_BELOW_MIN:{bench_price}<{min_p}",
                    live_signed_at=live_signed_at,
                    clock_projection=clock_projection,
                    fee_rate=fee_rate,
                )
            qty_dec = sizing["size_shares_dec"]
            cost_dec = (qty_dec * bench_price).quantize(USD_MICRO, rounding=ROUND_DOWN)
            fee_ps = compute_fee_per_share(fee_rate, bench_price)
            fee_dec = (qty_dec * fee_ps).quantize(USD_MICRO, rounding=ROUND_DOWN)
            spent_dec = (cost_dec + fee_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
            unfilled_dec = max(
                Decimal("0"),
                (to_decimal(sizing["requested_stake_usd"]) - spent_dec).quantize(USD_MICRO, rounding=ROUND_DOWN),
            )
            shadow_oid = local_order_hash or (
                "0xshadow_"
                + hashlib.sha256(
                    f"{tid}:{live_signed_at.isoformat()}:{sizing['requested_quantity']}".encode()
                ).hexdigest()[:56]
            )
            if callable(pre_send_callback):
                pre_send_callback(
                    shadow_oid,
                    sizing,
                    format_utc_iso(live_signed_at),
                    format_utc_iso(t_sent),
                )
            if shadow_oid:
                self._deducted_order_keys.add(f"clob:{shadow_oid}")
            self._cumulative_local_spent_usdc = round(self._cumulative_local_spent_usdc + float(spent_dec), 6)
            self._cached_free_usdc = max(0.0, round(self._cached_free_usdc - float(spent_dec), 6))
            return {
                "sizing": sizing,
                "live_signed_at": format_utc_iso(live_signed_at),
                "live_http_sent_at": format_utc_iso(t_sent),
                "live_http_ack_at": format_utc_iso(t_ack),
                "clob_rtt_ms": 0.0,
                "live_execution": {
                    "live_scale_factor": sizing["live_scale_factor"],
                    "order_type": order_type_str,
                    "limit_price": sizing["limit_price"],
                    "requested_stake_usd": sizing["requested_stake_usd"],
                    "requested_quantity": sizing["requested_quantity"],
                    "clob_order_id": shadow_oid,
                    "clob_status": "SHADOW_MATCHED",
                    "error_code": None,
                    "filled_quantity": sizing["requested_quantity"],
                    "avg_fill_price": float(bench_price),
                    "cost_usdc": float(cost_dec),
                    "fee_usdc": float(fee_dec),
                    "total_spent_usdc": float(spent_dec),
                    "unfilled_stake_usdc": float(unfilled_dec),
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                    "fee_rate": float(fee_rate),
                },
            }

        # Live mode: persist pre-send draft first, then record `live_http_sent_at` and perform
        # the final `valid_until` deadline check immediately before `post_order` ([P1-A], [P1-5], [P1-R3-3])!
        assert self.client is not None
        assert_fak_order_only(OrderType.FAK, post_only=False)
        pre_cb_now = _current_cal_utc()
        within_pre_cb, _ = check_hard_deadline(pre_cb_now, valid_until_dt)
        if not within_pre_cb:
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_PAST_VALID_UNTIL",
                error_code=f"HARD_DEADLINE_EXCEEDED_PRE_CB:now={format_utc_iso(pre_cb_now)}>=valid_until={format_utc_iso(valid_until_dt)}",
                live_signed_at=live_signed_at,
                clock_projection=clock_projection,
                fee_rate=fee_rate,
            )

        if callable(pre_send_callback):
            pre_send_callback(
                local_order_hash,
                sizing,
                format_utc_iso(live_signed_at),
                format_utc_iso(live_signed_at),
            )

        live_http_sent_at = _current_cal_utc()
        within_deadline, _ = check_hard_deadline(live_http_sent_at, valid_until_dt)
        if not within_deadline:
            return self._build_aborted_result(
                sizing=sizing,
                order_type_str=order_type_str,
                clob_status="ABORTED_PAST_VALID_UNTIL",
                error_code=f"HARD_DEADLINE_EXCEEDED_AT_SEND:http_sent_at={format_utc_iso(live_http_sent_at)}>=valid_until={valid_until}",
                live_signed_at=live_signed_at,
                clock_projection=clock_projection,
                live_http_sent_at=live_http_sent_at,
                fee_rate=fee_rate,
            )

        t_sent_mono = time.perf_counter()
        try:
            try:
                resp = self.client.post_order(signed_order, OrderType.FAK, False)
            except TypeError:
                resp = self.client.post_order(signed_order, orderType=OrderType.FAK, post_only=False)
            clob_rtt_ms = round((time.perf_counter() - t_sent_mono) * 1000.0, 2)
            live_http_ack_at = _current_cal_utc()
            parsed_exec = self._parse_clob_post_response(
                resp=resp,
                sizing=sizing,
                fee_rate=fee_rate,
                fallback_order_id=local_order_hash,
                defer_tx_receipt_check=defer_tx_receipt_check,
            )
            parsed_exec.setdefault("fee_rate", float(fee_rate))
        except Exception as exc:
            # [P1-2] Ambiguous transport / timeout exception: NEVER mark as REJECTED!
            # Mark as ORDER_STATE_UNKNOWN so AuditStore keeps the reserved risk quota locked until reconciled.
            # Exception: deterministic CLOB HTTP 400 validation rejection (e.g. invalid amounts/price/signature)
            # where the CLOB server explicitly rejected the order before matching (`0` fill).
            clob_rtt_ms = round((time.perf_counter() - t_sent_mono) * 1000.0, 2)
            live_http_ack_at = _current_cal_utc()
            exc_text_lower = str(exc).lower()
            is_fak_no_match_400 = (
                getattr(exc, "status_code", None) == 400
                and any(
                    msg in exc_text_lower
                    for msg in ("no orders found to match", "killed if no match is found", "order is not marketable")
                )
            )
            is_deterministic_http_400 = is_fak_no_match_400 or (
                getattr(exc, "status_code", None) == 400
                and any(
                    msg in exc_text_lower
                    for msg in ("invalid amounts", "invalid price", "invalid signature", "min size", "order is too small")
                )
            )
            parsed_exec = {
                "live_scale_factor": sizing["live_scale_factor"],
                "order_type": order_type_str,
                "limit_price": sizing["limit_price"],
                "requested_stake_usd": sizing["requested_stake_usd"],
                "requested_quantity": sizing["requested_quantity"],
                "clob_order_id": local_order_hash,
                "clob_status": (
                    "CANCELED"
                    if is_fak_no_match_400
                    else ("REJECTED" if is_deterministic_http_400 else "ORDER_STATE_UNKNOWN")
                ),
                "error_code": (
                    f"FAK_KILLED_NO_MATCH:{str(exc)[:160]}"
                    if is_fak_no_match_400
                    else (
                        f"CLOB_HTTP_400_REJECTED:{str(exc)[:160]}"
                        if is_deterministic_http_400
                        else f"SUBMIT_EXCEPTION_REQUIRES_RECONCILE:{type(exc).__name__}:{str(exc)[:160]}"
                    )
                ),
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": sizing["requested_stake_usd"],
                "polygon_tx_hashes": [],
                "onchain_confirmed": False,
                "onchain_block_number": None,
                "fee_rate": float(fee_rate),
            }

        return {
            "sizing": sizing,
            "live_signed_at": format_utc_iso(live_signed_at),
            "live_http_sent_at": format_utc_iso(live_http_sent_at),
            "live_http_ack_at": format_utc_iso(live_http_ack_at),
            "clob_rtt_ms": clob_rtt_ms,
            "live_execution": parsed_exec,
        }

    def _parse_clob_post_response(
        self,
        resp: Any,
        sizing: Mapping[str, Any],
        fee_rate: float | Decimal,
        fallback_order_id: str | None = None,
        defer_tx_receipt_check: bool = False,
    ) -> dict[str, Any]:
        """Strictly parse Polymarket CLOB V2 `POST /order` response ([P1-3]).

        Never defaults missing `status` to `"MATCHED"` or missing `takingAmount` to `requested_quantity`.
        Never marks `onchain_confirmed = True` unless the Polygon receipt has `status == 1` (`"0x1"`).
        """
        if not isinstance(resp, Mapping) or not resp:
            return {
                "live_scale_factor": sizing["live_scale_factor"],
                "order_type": "FAK",
                "limit_price": sizing["limit_price"],
                "requested_stake_usd": sizing["requested_stake_usd"],
                "requested_quantity": sizing["requested_quantity"],
                "clob_order_id": fallback_order_id,
                "clob_status": "ORDER_STATE_UNKNOWN",
                "error_code": "EMPTY_OR_MALFORMED_CLOB_RESPONSE",
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": sizing["requested_stake_usd"],
                "polygon_tx_hashes": [],
                "onchain_confirmed": False,
                "onchain_block_number": None,
            }

        data = resp
        raw_success = data.get("success")
        error_msg = data.get("errorMsg") or data.get("error") or None
        order_id = str(data.get("orderID") or data.get("orderId") or data.get("id") or "").strip() or fallback_order_id
        raw_status_val = data.get("status")
        status_raw = str(raw_status_val).strip().upper() if raw_status_val is not None else ""

        tx_hashes = data.get("transactionsHashes") or data.get("transactionHashes") or []
        if isinstance(tx_hashes, str):
            tx_hashes = [tx_hashes] if tx_hashes else []
        trade_ids = data.get("tradeIDs") or data.get("tradeIds") or []
        if isinstance(trade_ids, str):
            trade_ids = [trade_ids] if trade_ids else []

        # Explicit server rejection
        if raw_success is False or status_raw in ("REJECTED", "INVALID", "CANCELED", "CANCELLED", "KILLED", "UNMATCHED"):
            final_status = status_raw if status_raw in ("REJECTED", "INVALID", "CANCELED", "CANCELLED", "KILLED", "UNMATCHED") else "REJECTED"
            return {
                "live_scale_factor": sizing["live_scale_factor"],
                "order_type": "FAK",
                "limit_price": sizing["limit_price"],
                "requested_stake_usd": sizing["requested_stake_usd"],
                "requested_quantity": sizing["requested_quantity"],
                "clob_order_id": order_id,
                "clob_status": final_status,
                "error_code": str(error_msg) if error_msg else "CLOB_ORDER_REJECTED",
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": sizing["requested_stake_usd"],
                "polygon_tx_hashes": [],
                "onchain_confirmed": False,
                "onchain_block_number": None,
            }

        # Missing explicit status or order_id -> treat as ORDER_STATE_UNKNOWN for reconciliation
        if raw_success is not True or not status_raw or not order_id:
            return {
                "live_scale_factor": sizing["live_scale_factor"],
                "order_type": "FAK",
                "limit_price": sizing["limit_price"],
                "requested_stake_usd": sizing["requested_stake_usd"],
                "requested_quantity": sizing["requested_quantity"],
                "clob_order_id": order_id,
                "clob_status": "ORDER_STATE_UNKNOWN",
                "error_code": str(error_msg) if error_msg else "AMBIGUOUS_CLOB_ACK_MISSING_STATUS_OR_ID",
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": sizing["requested_stake_usd"],
                "polygon_tx_hashes": list(tx_hashes),
                "onchain_confirmed": False,
                "onchain_block_number": None,
            }

        if status_raw in ("DELAYED", "LIVE"):
            return {
                "live_scale_factor": sizing["live_scale_factor"],
                "order_type": "FAK",
                "limit_price": sizing["limit_price"],
                "requested_stake_usd": sizing["requested_stake_usd"],
                "requested_quantity": sizing["requested_quantity"],
                "clob_order_id": order_id,
                "clob_status": "DELAYED",
                "error_code": f"ASYNC_COMMIT_PENDING:trade_ids={','.join(str(t) for t in trade_ids)}" if trade_ids else "ASYNC_COMMIT_PENDING",
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": sizing["requested_stake_usd"],
                "polygon_tx_hashes": list(tx_hashes),
                "onchain_confirmed": False,
                "onchain_block_number": None,
            }

        if status_raw in ("MATCHED", "FILLED"):
            taking_raw = data.get("takingAmount")
            making_raw = data.get("makingAmount")
            # [P1-3] Never substitute requested_quantity when takingAmount is missing!
            if taking_raw in (None, ""):
                return {
                    "live_scale_factor": sizing["live_scale_factor"],
                    "order_type": "FAK",
                    "limit_price": sizing["limit_price"],
                    "requested_stake_usd": sizing["requested_stake_usd"],
                    "requested_quantity": sizing["requested_quantity"],
                    "clob_order_id": order_id,
                    "clob_status": "ORDER_STATE_UNKNOWN",
                    "error_code": "MATCHED_WITHOUT_TAKING_AMOUNT_REQUIRES_RECONCILE",
                    "filled_quantity": "0.00",
                    "avg_fill_price": 0.0,
                    "cost_usdc": 0.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 0.0,
                    "unfilled_stake_usdc": sizing["requested_stake_usd"],
                    "polygon_tx_hashes": list(tx_hashes),
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                }
            filled_qty_dec = to_decimal(taking_raw, "takingAmount").quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            if filled_qty_dec <= Decimal("0"):
                return {
                    "live_scale_factor": sizing["live_scale_factor"],
                    "order_type": "FAK",
                    "limit_price": sizing["limit_price"],
                    "requested_stake_usd": sizing["requested_stake_usd"],
                    "requested_quantity": sizing["requested_quantity"],
                    "clob_order_id": order_id,
                    "clob_status": "KILLED",
                    "error_code": "ZERO_TAKING_AMOUNT",
                    "filled_quantity": "0.00",
                    "avg_fill_price": 0.0,
                    "cost_usdc": 0.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 0.0,
                    "unfilled_stake_usdc": sizing["requested_stake_usd"],
                    "polygon_tx_hashes": list(tx_hashes),
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                }
            if making_raw not in (None, "", "0"):
                cost_dec = to_decimal(making_raw, "makingAmount").quantize(USD_MICRO, rounding=ROUND_DOWN)
            else:
                cost_dec = (filled_qty_dec * sizing["limit_price_dec"]).quantize(USD_MICRO, rounding=ROUND_DOWN)
            avg_price_dec = (cost_dec / filled_qty_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
            fee_ps = compute_fee_per_share(fee_rate, avg_price_dec if avg_price_dec > 0 else sizing["limit_price_dec"])
            fee_dec = (filled_qty_dec * fee_ps).quantize(USD_MICRO, rounding=ROUND_DOWN)
            spent_dec = (cost_dec + fee_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
            unfilled_dec = max(
                Decimal("0"),
                (to_decimal(sizing["requested_stake_usd"]) - spent_dec).quantize(USD_MICRO, rounding=ROUND_DOWN),
            )
            onchain_confirmed, block_number = (
                self.fetch_tx_receipt_confirmation(tx_hashes[0])
                if (tx_hashes and not defer_tx_receipt_check)
                else (False, None)
            )
            if order_id:
                self._deducted_order_keys.add(f"clob:{order_id}")
            if fallback_order_id:
                self._deducted_order_keys.add(f"clob:{fallback_order_id}")
            self._cumulative_local_spent_usdc = round(self._cumulative_local_spent_usdc + float(spent_dec), 6)
            self._cached_free_usdc = max(0.0, round(self._cached_free_usdc - float(spent_dec), 6))
            return {
                "live_scale_factor": sizing["live_scale_factor"],
                "order_type": "FAK",
                "limit_price": sizing["limit_price"],
                "requested_stake_usd": sizing["requested_stake_usd"],
                "requested_quantity": sizing["requested_quantity"],
                "clob_order_id": order_id,
                "clob_status": status_raw,
                "error_code": str(error_msg) if error_msg else None,
                "filled_quantity": format(filled_qty_dec, ".2f"),
                "avg_fill_price": float(avg_price_dec),
                "cost_usdc": float(cost_dec),
                "fee_usdc": float(fee_dec),
                "total_spent_usdc": float(spent_dec),
                "unfilled_stake_usdc": float(unfilled_dec),
                "polygon_tx_hashes": list(tx_hashes),
                "onchain_confirmed": onchain_confirmed,
                "onchain_block_number": block_number,
            }

        # Unknown status string -> keep locked for reconciliation
        return {
            "live_scale_factor": sizing["live_scale_factor"],
            "order_type": "FAK",
            "limit_price": sizing["limit_price"],
            "requested_stake_usd": sizing["requested_stake_usd"],
            "requested_quantity": sizing["requested_quantity"],
            "clob_order_id": order_id,
            "clob_status": "ORDER_STATE_UNKNOWN",
            "error_code": f"UNRECOGNIZED_CLOB_STATUS:{status_raw}",
            "filled_quantity": "0.00",
            "avg_fill_price": 0.0,
            "cost_usdc": 0.0,
            "fee_usdc": 0.0,
            "total_spent_usdc": 0.0,
            "unfilled_stake_usdc": sizing["requested_stake_usd"],
            "polygon_tx_hashes": list(tx_hashes),
            "onchain_confirmed": False,
            "onchain_block_number": None,
        }

    def reconcile_order(
        self,
        order_row: Mapping[str, Any],
        fee_rate: float | Decimal = Decimal("0.075"),
    ) -> dict[str, Any] | None:
        """Query CLOB `/data/order/<id>` and `/data/trades` to reconcile `ORDER_STATE_UNKNOWN` / `DELAYED` orders.

        Enforces `[P1-R4-1]`:
          - Any order still in `RESERVED` state (never reached `record_order_pre_send` / HTTP dispatch)
            was never sent to CLOB and is immediately transitioned to `ABORTED_PRE_SEND_NOT_SUBMITTED` (`0` fill).
          - Never claims or matches trades from `get_trades` by `token_id` alone when `clob_order_id` is missing.
        """
        order_id = str(order_row.get("clob_order_id") or "").strip()
        token_id = str(order_row.get("token_id") or "").strip()
        req_stake = to_decimal(order_row.get("requested_stake_usd", 0), "requested_stake_usd")
        req_qty = str(order_row.get("requested_quantity") or "0.00")
        limit_price = to_decimal(order_row.get("limit_price", 0.51), "limit_price")
        scale_factor = float(order_row.get("live_scale_factor") or 0.10)
        row_clob_status = str(order_row.get("clob_status") or "").strip().upper()
        row_pos_status = str(order_row.get("position_status") or "").strip().upper()

        # [P1-R4-1]: Distinguish pre-send failures (`RESERVED` without pre-send order hash) from post-send unknown orders.
        # Because `record_order_pre_send` is always persisted to SQLite BEFORE `post_order` is invoked,
        # any row still in `RESERVED` state was guaranteed NEVER sent over HTTP.
        if row_clob_status == "RESERVED" or row_pos_status == "RESERVED":
            return {
                "live_scale_factor": scale_factor,
                "order_type": "FAK",
                "limit_price": float(limit_price),
                "requested_stake_usd": float(req_stake),
                "requested_quantity": req_qty,
                "clob_order_id": None,
                "clob_status": "ABORTED_PRE_SEND_NOT_SUBMITTED",
                "error_code": "ORDER_NEVER_REACHED_PRE_SEND_OR_HTTP_DISPATCH",
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": float(req_stake),
                "polygon_tx_hashes": [],
                "onchain_confirmed": False,
                "onchain_block_number": None,
                "fee_rate": float(fee_rate),
            }

        if self.client is None or not self.has_l2_creds:
            return None

        live_exec_info = order_row.get("live_execution") if isinstance(order_row.get("live_execution"), Mapping) else {}
        prior_err_code = str(live_exec_info.get("error_code") or order_row.get("error_code") or "").lower()
        if "status_code=400" in prior_err_code and any(
            msg in prior_err_code
            for msg in (
                "no orders found to match",
                "killed if no match is found",
                "order is not marketable",
                "invalid amounts",
                "invalid price",
                "invalid signature",
                "min size",
                "order is too small",
            )
        ):
            is_no_match = any(
                msg in prior_err_code
                for msg in ("no orders found to match", "killed if no match is found", "order is not marketable")
            )
            return {
                "live_scale_factor": scale_factor,
                "order_type": "FAK",
                "limit_price": float(limit_price),
                "requested_stake_usd": float(req_stake),
                "requested_quantity": req_qty,
                "clob_order_id": order_id,
                "clob_status": "CANCELED" if is_no_match else "REJECTED",
                "error_code": "RECONCILED_FAK_KILLED_NO_MATCH" if is_no_match else "RECONCILED_HTTP_400_REJECTED",
                "filled_quantity": "0.00",
                "avg_fill_price": 0.0,
                "cost_usdc": 0.0,
                "fee_usdc": 0.0,
                "total_spent_usdc": 0.0,
                "unfilled_stake_usdc": float(req_stake),
                "polygon_tx_hashes": [],
                "onchain_confirmed": False,
                "onchain_block_number": None,
                "fee_rate": float(fee_rate),
            }

        # [P1-R4-1]: Never query or claim historical trades by `token_id` when `order_id` is absent!
        if not order_id or order_id.startswith("0xshadow_"):
            return None

        order_info: Mapping[str, Any] | None = None
        try:
            res = self.client.get_order(order_id)
            if isinstance(res, Mapping):
                order_info = res
        except Exception:
            order_info = None

        matched_trades: list[Mapping[str, Any]] = []
        if token_id and self.has_valid_funder:
            try:
                trades = self.client.get_trades(
                    TradeParams(asset_id=token_id, maker_address=self.funder_address),
                    only_first_page=True,
                )
                if isinstance(trades, list):
                    for tr in trades:
                        if not isinstance(tr, Mapping):
                            continue
                        tr_oid = str(tr.get("taker_order_id") or tr.get("order_id") or "").strip()
                        maker_orders = tr.get("maker_orders") or []
                        in_maker = any(
                            isinstance(m, Mapping) and str(m.get("order_id") or "").strip() == order_id
                            for m in maker_orders
                        )
                        if tr_oid == order_id or in_maker:
                            matched_trades.append(tr)
            except Exception:
                pass

        if matched_trades:
            total_shares = Decimal("0")
            total_cost = Decimal("0")
            tx_hashes: list[str] = []
            any_pending = False
            all_failed = True
            for tr in matched_trades:
                tr_status = str(tr.get("status") or "").upper()
                if tr_status in ("FAILED", "REJECTED", "CANCELED"):
                    continue
                all_failed = False
                if tr_status in ("MATCHED", "RETRYING") and not tr.get("transaction_hash"):
                    any_pending = True
                sz = to_decimal(tr.get("size", 0), "size")
                pr = to_decimal(tr.get("price", limit_price), "price")
                total_shares += sz
                total_cost += sz * pr
                txh = tr.get("transaction_hash")
                if txh and str(txh) not in tx_hashes:
                    tx_hashes.append(str(txh))

            if all_failed:
                return {
                    "live_scale_factor": scale_factor,
                    "order_type": "FAK",
                    "limit_price": float(limit_price),
                    "requested_stake_usd": float(req_stake),
                    "requested_quantity": req_qty,
                    "clob_order_id": order_id,
                    "clob_status": "REJECTED",
                    "error_code": "RECONCILED_TRADE_FAILED",
                    "filled_quantity": "0.00",
                    "avg_fill_price": 0.0,
                    "cost_usdc": 0.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 0.0,
                    "unfilled_stake_usdc": float(req_stake),
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                    "fee_rate": float(fee_rate),
                }
            if any_pending:
                return None

            filled_qty_dec = total_shares.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            cost_dec = total_cost.quantize(USD_MICRO, rounding=ROUND_DOWN)
            avg_price_dec = (cost_dec / filled_qty_dec).quantize(USD_MICRO, rounding=ROUND_DOWN) if filled_qty_dec > 0 else Decimal("0")
            fee_ps = compute_fee_per_share(fee_rate, avg_price_dec if avg_price_dec > 0 else limit_price)
            fee_dec = (filled_qty_dec * fee_ps).quantize(USD_MICRO, rounding=ROUND_DOWN)
            spent_dec = (cost_dec + fee_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
            unfilled_dec = max(Decimal("0"), (req_stake - spent_dec).quantize(USD_MICRO, rounding=ROUND_DOWN))
            onchain_confirmed, block_number = (
                self.fetch_tx_receipt_confirmation(tx_hashes[0]) if tx_hashes else (False, None)
            )
            return {
                "live_scale_factor": scale_factor,
                "order_type": "FAK",
                "limit_price": float(limit_price),
                "requested_stake_usd": float(req_stake),
                "requested_quantity": req_qty,
                "clob_order_id": order_id,
                "clob_status": "MATCHED",
                "error_code": None,
                "filled_quantity": format(filled_qty_dec, ".2f"),
                "avg_fill_price": float(avg_price_dec),
                "cost_usdc": float(cost_dec),
                "fee_usdc": float(fee_dec),
                "total_spent_usdc": float(spent_dec),
                "unfilled_stake_usdc": float(unfilled_dec),
                "polygon_tx_hashes": tx_hashes,
                "onchain_confirmed": onchain_confirmed,
                "onchain_block_number": block_number,
                "fee_rate": float(fee_rate),
            }

        if isinstance(order_info, Mapping):
            ord_status = str(order_info.get("status") or "").upper()
            size_matched = to_decimal(order_info.get("size_matched") or 0, "size_matched").quantize(
                Decimal("0.01"), rounding=ROUND_DOWN
            )
            ord_price = to_decimal(order_info.get("price") or limit_price, "price")
            if size_matched > Decimal("0") and ord_status in ("MATCHED", "FILLED", "CANCELED", "CANCELLED"):
                cost_dec = (size_matched * ord_price).quantize(USD_MICRO, rounding=ROUND_DOWN)
                fee_ps = compute_fee_per_share(fee_rate, ord_price)
                fee_dec = (size_matched * fee_ps).quantize(USD_MICRO, rounding=ROUND_DOWN)
                spent_dec = (cost_dec + fee_dec).quantize(USD_MICRO, rounding=ROUND_DOWN)
                unfilled_dec = max(Decimal("0"), (req_stake - spent_dec).quantize(USD_MICRO, rounding=ROUND_DOWN))
                return {
                    "live_scale_factor": scale_factor,
                    "order_type": "FAK",
                    "limit_price": float(limit_price),
                    "requested_stake_usd": float(req_stake),
                    "requested_quantity": req_qty,
                    "clob_order_id": order_id or str(order_info.get("id") or ""),
                    "clob_status": "MATCHED",
                    "error_code": None,
                    "filled_quantity": format(size_matched, ".2f"),
                    "avg_fill_price": float(ord_price),
                    "cost_usdc": float(cost_dec),
                    "fee_usdc": float(fee_dec),
                    "total_spent_usdc": float(spent_dec),
                    "unfilled_stake_usdc": float(unfilled_dec),
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                    "fee_rate": float(fee_rate),
                }
            if size_matched <= Decimal("0") and ord_status in ("CANCELED", "CANCELLED", "KILLED", "REJECTED", "INVALID", "UNMATCHED"):
                return {
                    "live_scale_factor": scale_factor,
                    "order_type": "FAK",
                    "limit_price": float(limit_price),
                    "requested_stake_usd": float(req_stake),
                    "requested_quantity": req_qty,
                    "clob_order_id": order_id,
                    "clob_status": "CANCELED",
                    "error_code": f"RECONCILED_ZERO_FILL:{ord_status}",
                    "filled_quantity": "0.00",
                    "avg_fill_price": 0.0,
                    "cost_usdc": 0.0,
                    "fee_usdc": 0.0,
                    "total_spent_usdc": 0.0,
                    "unfilled_stake_usdc": float(req_stake),
                    "polygon_tx_hashes": [],
                    "onchain_confirmed": False,
                    "onchain_block_number": None,
                    "fee_rate": float(fee_rate),
                }

        return None

    def fetch_tx_receipt_confirmation(self, tx_hash: str) -> tuple[bool, int | None]:
        """Query Polygon RPC `eth_getTransactionReceipt` and verify `status == 1` (`"0x1"`) ([P1-3])."""
        if not tx_hash or not str(tx_hash).startswith("0x"):
            return False, None
        try:
            body = self._post_polygon_rpc(
                {"jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionReceipt", "params": [str(tx_hash)]},
                timeout=1.2,
            )
            if body is not None:
                res = body.get("result")
                if isinstance(res, Mapping) and res.get("blockNumber"):
                    block_num = int(str(res["blockNumber"]), 16)
                    raw_status = res.get("status")
                    if raw_status in ("0x1", 1, "1", True):
                        return True, block_num
                    return False, block_num
        except Exception:
            pass
        return False, None

    def fetch_tx_block_number(self, tx_hash: str) -> int | None:
        confirmed, block_num = self.fetch_tx_receipt_confirmation(tx_hash)
        return block_num if confirmed else None

    @staticmethod
    def _target_exchange_spender(market_protocol: str = "v1", neg_risk: bool = False) -> str:
        if str(market_protocol or "v1").strip().lower() == "v2":
            return OFFICIAL_EXCHANGE_V3_ADDRESS.lower()
        if neg_risk:
            return OFFICIAL_NEG_RISK_EXCHANGE_V2_ADDRESS.lower()
        return OFFICIAL_STANDARD_EXCHANGE_V2_ADDRESS.lower()

    def query_wallet_balance_and_allowance(
        self,
        refresh_network: bool = True,
        market_protocol: str = "v1",
        neg_risk: bool = False,
    ) -> tuple[float, float]:
        """Return `(onchain_free_pusd_balance, clob_allowance_pusd)` and update `self.balance_verified` ([P1-7], Issue #9).

        Maintains per-exchange-contract allowance cache (`_cached_allowances_by_spender`) so a V2 or NegRisk
        market never reuses the Standard Exchange allowance when `refresh_network=False` or when the target
        spender is absent from `allowances`.
        """
        target_spender = self._target_exchange_spender(market_protocol=market_protocol, neg_risk=neg_risk)
        if not refresh_network:
            if self._has_spender_allowance_map or target_spender in self._cached_allowances_by_spender:
                spender_all = float(self._cached_allowances_by_spender.get(target_spender, 0.0))
                self._cached_allowance_usdc = spender_all
                return self._cached_free_usdc, spender_all
            return self._cached_free_usdc, self._cached_allowance_usdc

        self._internal_balance_call_token = getattr(self, "_internal_balance_call_token", 0) + 1
        self._last_refresh_read_authoritative = False
        verified_bal: Decimal | None = None
        verified_all: Decimal | None = None

        if self.client is not None and (
            self.has_l2_creds or "get_balance_allowance" in getattr(self.client, "__dict__", {})
        ):
            try:
                res = self.client.get_balance_allowance(
                    BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=int(self.signature_type))
                )
                if isinstance(res, Mapping):
                    verified_bal = _parse_base_unit_collateral(res.get("balance"))
                    raw_all = res.get("allowance")
                    raw_allowances_map = res.get("allowances")
                    if raw_all is not None:
                        self._has_spender_allowance_map = False
                        verified_all = _parse_base_unit_collateral(raw_all)
                    elif isinstance(raw_allowances_map, Mapping):
                        self._has_spender_allowance_map = True
                        self._cached_allowances_by_spender.clear()
                        parsed_spenders: dict[str, Decimal] = {}
                        for spender, val in raw_allowances_map.items():
                            parsed = _parse_base_unit_collateral(val)
                            if parsed is not None:
                                spender_key = str(spender).lower()
                                parsed_spenders[spender_key] = parsed
                                self._cached_allowances_by_spender[spender_key] = float(parsed)
                        if target_spender in parsed_spenders:
                            verified_all = parsed_spenders[target_spender]
                        else:
                            verified_all = Decimal("0")
                            self._cached_allowances_by_spender[target_spender] = 0.0
            except Exception as exc:
                logger.warning("Failed to fetch CLOB balance/allowance: %s", exc)

        if verified_bal is None and self.has_valid_funder:
            verified_bal = self._query_onchain_erc20_balance(self.funder_address, self.pusd_collateral_address)

        if verified_bal is not None:
            self._last_refresh_read_authoritative = True
            self._last_balance_snapshot_seq += 1
            self._last_authoritative_balance_seq = getattr(self, "_last_authoritative_balance_seq", 0) + 1
            self._last_balance_snapshot_utc = datetime.now(timezone.utc)

        if verified_bal is not None and verified_all is not None:
            self._cached_free_usdc = float(verified_bal)
            self._cached_allowance_usdc = float(verified_all)
            self.balance_verified = True
        elif self.mode == "shadow":
            if verified_bal is not None:
                self._cached_free_usdc = float(verified_bal)
            if verified_all is not None:
                self._cached_allowance_usdc = float(verified_all)
            self.balance_verified = True
            self._last_balance_snapshot_seq += 1
            self._last_balance_snapshot_utc = datetime.now(timezone.utc)
        else:
            if verified_bal is not None:
                self._cached_free_usdc = float(verified_bal)
            if verified_all is not None:
                self._cached_allowance_usdc = float(verified_all)
            else:
                self._cached_allowance_usdc = 0.0
            self.balance_verified = False

        return self._cached_free_usdc, self._cached_allowance_usdc

    def _try_refresh_authoritative_balance(self) -> float | None:
        """Attempt to read the authoritative wallet balance without clobbering cached allowance/verification
        when no external balance endpoint is configured (`[P1-R6-1]`, `[P2-R6-2]`).
        """
        prev_verified = bool(self.balance_verified)
        prev_allowance = float(self._cached_allowance_usdc)
        prev_allowances_map = dict(self._cached_allowances_by_spender)
        prev_has_map = bool(self._has_spender_allowance_map)
        token_before = int(getattr(self, "_internal_balance_call_token", 0))
        self._last_refresh_read_authoritative = False

        try:
            res = self.query_wallet_balance_and_allowance(refresh_network=True)
        except Exception:
            self.balance_verified = prev_verified
            self._cached_allowance_usdc = prev_allowance
            self._cached_allowances_by_spender = prev_allowances_map
            self._has_spender_allowance_map = prev_has_map
            return None

        token_after = int(getattr(self, "_internal_balance_call_token", 0))
        monkeypatched_query = (token_after == token_before) or ("query_wallet_balance_and_allowance" in self.__dict__)

        if monkeypatched_query:
            if isinstance(res, tuple) and len(res) >= 1 and res[0] is not None:
                self._cached_free_usdc = round(float(res[0]), 6)
                if len(res) >= 2 and res[1] is not None:
                    self._cached_allowance_usdc = float(res[1])
                else:
                    self._cached_allowance_usdc = prev_allowance
                if prev_verified:
                    self.balance_verified = True
                self._last_balance_snapshot_seq += 1
                self._last_authoritative_balance_seq = getattr(self, "_last_authoritative_balance_seq", 0) + 1
                self._last_balance_snapshot_utc = datetime.now(timezone.utc)
                return self._cached_free_usdc
            return None

        if self._last_refresh_read_authoritative:
            if not self.balance_verified and prev_verified and prev_allowance > 0:
                self._cached_allowance_usdc = prev_allowance
                self._cached_allowances_by_spender = prev_allowances_map
                self._has_spender_allowance_map = prev_has_map
                self.balance_verified = True
            return round(float(self._cached_free_usdc), 6)

        # No authoritative balance read succeeded (e.g. offline mode without balance mock) -> preserve prior verification state
        self.balance_verified = prev_verified
        self._cached_allowance_usdc = prev_allowance
        self._cached_allowances_by_spender = prev_allowances_map
        self._has_spender_allowance_map = prev_has_map
        return None

    def apply_reconciled_order_spend(
        self,
        row: Mapping[str, Any],
        reconciled_exec: Mapping[str, Any],
    ) -> float:
        """Synchronize `_cached_free_usdc` when a timed-out/unknown order reconciles as filled (`[P2-R6-2]`).

        Called BEFORE `AuditStore.record_order_execution` releases the order's reserved risk quota:
          1. Idempotent per `signal_id` and `clob_order_id` (`_deducted_order_keys`).
          2. Attempts an authoritative wallet balance refresh (`_try_refresh_authoritative_balance`):
             - If an authoritative balance is returned and is already below the pre-order balance, uses it directly
               without double-deducting `total_spent_usdc`.
          3. Otherwise, if an authoritative balance refresh already occurred after the order was sent (or the
             cached balance was already reduced below `free_usdc_before_order - spent`), avoids duplicate deduction;
             else deducts `total_spent_usdc` from `_cached_free_usdc`.
        """
        status_upper = str(reconciled_exec.get("clob_status") or "").strip().upper()
        spent = round(float(reconciled_exec.get("total_spent_usdc") or 0.0), 6)
        if spent <= 0.0 or status_upper not in ("MATCHED", "FILLED", "SHADOW_MATCHED"):
            return self._cached_free_usdc

        sig_id = str(row.get("signal_id") or "").strip()
        oid = str(reconciled_exec.get("clob_order_id") or row.get("clob_order_id") or "").strip()
        candidate_keys = {k for k in (f"sig:{sig_id}" if sig_id else "", f"clob:{oid}" if oid else "") if k}
        if candidate_keys and bool(candidate_keys & self._deducted_order_keys):
            return self._cached_free_usdc
        self._deducted_order_keys.update(candidate_keys)

        prior_cum_spent = round(float(self._cumulative_local_spent_usdc), 6)
        self._cumulative_local_spent_usdc = round(prior_cum_spent + spent, 6)

        live_exec_old = row.get("live_execution") if isinstance(row.get("live_execution"), Mapping) else {}
        baseline_free: float | None = None
        baseline_auth_seq: int | None = None
        baseline_local_spent: float | None = None

        for cand_free in (
            row.get("free_usdc_before_order"),
            live_exec_old.get("free_usdc_before_order") if isinstance(live_exec_old, Mapping) else None,
            row.get("free_usdc_at_reserve"),
        ):
            if cand_free is not None:
                try:
                    baseline_free = round(float(cand_free), 6)
                    break
                except (TypeError, ValueError):
                    pass

        for cand_seq in (
            row.get("auth_balance_seq_before_order"),
            live_exec_old.get("auth_balance_seq_before_order") if isinstance(live_exec_old, Mapping) else None,
        ):
            if cand_seq is not None:
                try:
                    baseline_auth_seq = int(cand_seq)
                    break
                except (TypeError, ValueError):
                    pass

        for cand_ls in (
            row.get("local_spent_before_order"),
            live_exec_old.get("local_spent_before_order") if isinstance(live_exec_old, Mapping) else None,
        ):
            if cand_ls is not None:
                try:
                    baseline_local_spent = round(float(cand_ls), 6)
                    break
                except (TypeError, ValueError):
                    pass

        if oid and oid in self._order_balance_baseline:
            _, b_auth_seq, b_free, b_local_spent = self._order_balance_baseline[oid]
            if baseline_free is None:
                baseline_free = b_free
            if baseline_auth_seq is None:
                baseline_auth_seq = b_auth_seq
            if baseline_local_spent is None:
                baseline_local_spent = b_local_spent

        pre_reconcile_cached = round(float(self._cached_free_usdc), 6)
        auth_bal = self._try_refresh_authoritative_balance()
        if auth_bal is not None:
            # Authoritative balance endpoint returned a live balance.
            # If the endpoint is a static mock that still returned >= `baseline_free` (unaware of the fill),
            # deduct `spent`; otherwise `auth_bal` already reflects the fill expenditure!
            if baseline_free is not None and round(auth_bal, 6) >= round(baseline_free - 1e-6, 6):
                self._cached_free_usdc = max(0.0, round(auth_bal - spent, 6))
            else:
                self._cached_free_usdc = max(0.0, round(auth_bal, 6))
            return self._cached_free_usdc

        # No authoritative balance endpoint at reconciliation time: check if balance was already updated after order send
        if (
            baseline_auth_seq is not None
            and int(getattr(self, "_last_authoritative_balance_seq", 0)) > baseline_auth_seq
        ):
            if baseline_free is None or pre_reconcile_cached <= round(baseline_free - spent + 1e-6, 6):
                return self._cached_free_usdc

        if baseline_free is not None:
            other_spent = max(0.0, round(prior_cum_spent - float(baseline_local_spent or 0.0), 6))
            expected_after_all_fills = max(0.0, round(baseline_free - other_spent - spent, 6))
            if pre_reconcile_cached <= round(expected_after_all_fills + 1e-6, 6):
                return self._cached_free_usdc

        self._cached_free_usdc = max(0.0, round(self._cached_free_usdc - spent, 6))
        return self._cached_free_usdc

    def apply_confirmed_redemption_credit(
        self,
        newly_credited: float,
        *,
        batch_credit_key: str | None = None,
        was_pending_redemption: bool = False,
        submit_boot_id: str | None = None,
        balance_snapshot_seq_at_submit: int | None = None,
        tx_submitted_at_utc: str | datetime | None = None,
        free_usdc_at_submit: float | None = None,
        local_spent_at_submit: float | None = None,
    ) -> float:
        """Apply confirmed redemption payout to `_cached_free_usdc` without double-counting or skipping (`[P2-R4-4]`, `[P2-R5-4]`, `[P1-R6-1]`).

        Deduplication rules:
          1. Idempotent per `batch_credit_key` (`redeem_tx_id` / `redeem_tx_hash` / `market_id:keys`).
          2. At confirmation time, attempts an authoritative wallet balance read (`_try_refresh_authoritative_balance`):
             - If an authoritative real balance is successfully read (e.g. `$80.00` after `$88.00` submit, `-$20.00`
               order spend, and `+$12.00` redemption payout), that real balance is used DIRECTLY as the source of
               truth without guessing based on whether the balance rose or fell (`[P1-R6-1]`).
          3. In offline/shadow mode where no authoritative balance endpoint is available:
             - Accounts for any local order spend since redemption submission (`spent_since_submit`) so
               `expected_without_redeem = free_usdc_at_submit - spent_since_submit` is compared before adding `newly_credited`.
        """
        amt = round(float(newly_credited or 0.0), 6)
        if amt <= 0.0:
            return self._cached_free_usdc

        if batch_credit_key:
            if batch_credit_key in self._credited_redemption_keys:
                return self._cached_free_usdc
            self._credited_redemption_keys.add(batch_credit_key)

        if was_pending_redemption:
            pre_confirm_cached = round(float(self._cached_free_usdc), 6)
            auth_bal = self._try_refresh_authoritative_balance()
            if auth_bal is not None:
                # [P1-R6-1]: Authoritative real balance successfully read after confirmation -> use directly!
                self._cached_free_usdc = round(float(auth_bal), 6)
                return self._cached_free_usdc

            current_cached = round(float(self._cached_free_usdc), 6)
            if free_usdc_at_submit is not None:
                spent_since_submit = max(
                    0.0,
                    round(float(self._cumulative_local_spent_usdc) - float(local_spent_at_submit or 0.0), 6),
                )
                expected_without_redeem = max(0.0, round(float(free_usdc_at_submit) - spent_since_submit, 6))
                if current_cached >= round(expected_without_redeem + amt - 1e-6, 6):
                    return self._cached_free_usdc
            elif current_cached >= round(pre_confirm_cached + amt - 1e-6, 6):
                return self._cached_free_usdc
        else:
            auth_bal = self._try_refresh_authoritative_balance()
            if auth_bal is not None:
                self._cached_free_usdc = round(float(auth_bal), 6)
                return self._cached_free_usdc

        self._cached_free_usdc = round(self._cached_free_usdc + amt, 6)
        return self._cached_free_usdc
