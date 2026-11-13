"""Post-settlement Auto-Redeem back to Polymarket Smart Wallet Cash (pUSD).

Implements Section 4.3 & Section 5.3 of the ARGON4 External Live Executor Specification
with 2026 Polymarket Protocol V2 & Security Audit Hardening:
  - [P1-1] Protocol-aware redemption with pUSD (`0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB`):
    * `protocol == "v2"` -> `Router.redeem(bytes31 conditionId, uint256 outcomeIndex, uint256 amount)`
      on `0x12121212006e4CD160D18e3f00711DA5c3372600` (`0xd217a3cc`).
    * `protocol == "v1"` / `"ctf"` -> `redeemPositions(pUSD, bytes32(0), conditionId, [1, 2])`
      on `CtfCollateralAdapter` (`0xAdA100Db00Ca00073811820692005400218FcE1f`) or
      `NegRiskCtfCollateralAdapter` (`0xadA2005600Dec949baf300f4C6120000bDB6eAab`) (`0x01b7037c`).
  - [P1-4] Strict redemption confirmation:
    * NEVER marks `REDEEMED` or credits `redeemed_usdc_amount` merely because a `transaction_id`
      was returned! Requires confirmed `0x...` `transactionHash` from terminal Relayer state
      (`STATE_CONFIRMED` / `CONFIRMED` / `STATE_MINED` / `MINED`) and non-reverted receipt.
    * Failed or timed-out redemptions return `REDEEM_PENDING` / `REDEEM_FAILED` with
      `redeemed_usdc_amount = 0.0` so positions stay open for automatic retry.
  - Supports `Deposit Wallet` (`signature_type=3`), `Gnosis Safe` (`signature_type=2`), and
    `Polymarket Proxy` (`signature_type=1`) via `polymarket-client` (`0.17.0`).
  - Strictly enforces `withdrawal_permission = False` and `assert_redeem_only_transaction()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import hashlib
import re
import time
from typing import Any, Mapping
import requests

from eth_abi import encode as abi_encode
from eth_account import Account as EthAccount
from polymarket._internal.actions.relayer.calls import (
    TransactionCall,
    ctf_redeem_positions_call,
    encode_proxy_call,
    encode_safe_multisend_call,
    router_redeem_call,
)
from polymarket._internal.actions.relayer.signing.deposit_wallet import sign_deposit_wallet_batch
from polymarket._internal.actions.relayer.signing.proxy import (
    build_proxy_transaction_hash,
    sign_proxy_message,
)
from polymarket._internal.actions.relayer.signing.safe import sign_safe_transaction
from polymarket._internal.environment import PRODUCTION_CONFIG
from polymarket._internal.protocol import is_v2_position_id
from polymarket._internal.wallet import wrap_deposit_wallet_signature
from polymarket.types import EvmAddress, HexString

from .clob_executor import (
    derive_polymarket_candidate_wallets,
    is_valid_eth_address,
    is_valid_private_key,
)
from .math_utils import USD_MICRO, to_decimal
from .safety import (
    OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS,
    OFFICIAL_NEG_RISK_COLLATERAL_ADAPTER_ADDRESS,
    OFFICIAL_POLYGON_CTF_ADDRESS,
    OFFICIAL_POLYGON_PUSD_ADDRESS,
    OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS,
    REDEEM_POSITIONS_SELECTOR,
    ROUTER_REDEEM_SELECTOR,
    assert_no_withdrawal_permission,
    assert_redeem_only_transaction,
)


def encode_redeem_positions_calldata(
    condition_id: str,
    collateral_address: str = OFFICIAL_POLYGON_PUSD_ADDRESS,
    index_sets: list[int] | None = None,
    target_contract_address: str = OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS,
) -> str:
    """ABI-encode `redeemPositions(address,bytes32,bytes32,uint256[])` for Polymarket CTF / CollateralAdapter."""
    cond_clean = str(condition_id or "").strip()
    if not re.fullmatch(r"0x[0-9a-fA-F]{64}", cond_clean):
        raise ValueError(f"Invalid 32-byte condition_id: {condition_id!r}")

    sets = list(index_sets or [1, 2])
    if not sets or any(not isinstance(i, int) or i <= 0 for i in sets):
        raise ValueError(f"Invalid binary index_sets: {sets!r}")

    encoded_args = abi_encode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        [
            collateral_address,
            b"\x00" * 32,
            bytes.fromhex(cond_clean[2:]),
            sets,
        ],
    )
    calldata_hex = REDEEM_POSITIONS_SELECTOR + encoded_args.hex()
    assert_redeem_only_transaction(
        to_address=target_contract_address,
        calldata_hex=calldata_hex,
        value=0,
        expected_ctf_address=target_contract_address,
        expected_collateral_address=OFFICIAL_POLYGON_PUSD_ADDRESS,
    )
    return calldata_hex


def encode_v2_router_redeem_calldata(
    condition_id: str,
    outcome_index: int,
    amount_base_units: int,
    router_address: str = OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS,
) -> str:
    """ABI-encode `Router.redeem(bytes31 conditionId, uint256 outcomeIndex, uint256 amount)` for Protocol V2."""
    cond_clean = str(condition_id or "").strip().removeprefix("0x").removeprefix("0X")
    if len(cond_clean) == 64:
        cond_hex31 = "0x" + cond_clean[:62]
    elif len(cond_clean) == 62:
        cond_hex31 = "0x" + cond_clean
    else:
        raise ValueError(f"Invalid condition_id for V2 Router redeem: {condition_id!r}")

    call = router_redeem_call(
        router=EvmAddress(router_address),
        condition_id=cond_hex31,
        outcome_index=int(outcome_index),
        amount=int(amount_base_units),
    )
    calldata_hex = str(call.data)
    assert_redeem_only_transaction(
        to_address=router_address,
        calldata_hex=calldata_hex,
        value=0,
        expected_ctf_address=router_address,
    )
    return calldata_hex


@dataclass
class _RelayerExecResponse:
    transaction_id: str | None
    transaction_hash: str | None
    state: str | None = None


class _PolymarketRelayerV2Client:
    """Direct Relayer V2 HTTP client supporting DepositWallet (`WALLET`), Safe (`SAFE`), and Proxy (`PROXY`)."""

    def __init__(
        self,
        relayer_url: str,
        chain_id: int,
        private_key: str,
        signer_address: str,
        funder_address: str,
        signature_type: int,
        relayer_api_key: str,
        relayer_api_key_address: str,
        relayer_api_secret: str = "",
    ) -> None:
        self.relayer_url = relayer_url.rstrip("/")
        self.chain_id = int(chain_id)
        self._account = EthAccount.from_key(private_key)
        self.signer_address = signer_address
        self.funder_address = funder_address
        self.signature_type = int(signature_type)
        self.relayer_api_key = relayer_api_key.strip()
        self.relayer_api_key_address = relayer_api_key_address.strip() or signer_address
        self.relayer_api_secret = relayer_api_secret.strip()

    def _headers(self) -> dict[str, str]:
        headers = {
            "RELAYER_API_KEY": self.relayer_api_key,
            "RELAYER_API_KEY_ADDRESS": self.relayer_api_key_address,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.relayer_api_secret and self.relayer_api_secret != self.relayer_api_key_address:
            headers["RELAYER_API_SECRET"] = self.relayer_api_secret
        return headers

    def execute(self, transactions: list[Any], metadata: str = "") -> _RelayerExecResponse:
        calls: list[TransactionCall] = []
        for t in transactions:
            to_addr = getattr(t, "to", None) or (t.get("to") if isinstance(t, Mapping) else None)
            data_hex = getattr(t, "data", None) or (t.get("data") if isinstance(t, Mapping) else None)
            val_raw = getattr(t, "value", 0) if not isinstance(t, Mapping) else t.get("value", 0)
            calls.append(
                TransactionCall(
                    to=EvmAddress(str(to_addr)),
                    data=HexString(str(data_hex)),
                    value=int(val_raw or 0),
                )
            )

        tx_type = {3: "WALLET", 2: "SAFE", 1: "PROXY"}.get(self.signature_type, "SAFE")
        nonce_str = "0"
        nonce_resp = requests.get(
            f"{self.relayer_url}/nonce",
            params={"address": self.signer_address, "type": tx_type},
            headers=self._headers(),
            timeout=5.0,
        )
        if nonce_resp.status_code == 200:
            nonce_str = str(nonce_resp.json().get("nonce", "0"))
        else:
            alt_resp = requests.get(
                f"{self.relayer_url}/v1/account/transactions/params",
                params={"address": self.signer_address, "type": tx_type},
                headers=self._headers(),
                timeout=5.0,
            )
            alt_resp.raise_for_status()
            nonce_str = str(alt_resp.json().get("nonce", "0"))

        if tx_type == "WALLET":
            deadline_str = str(int(time.time()) + 3600)
            sig = sign_deposit_wallet_batch(
                self._account,
                wallet=EvmAddress(self.funder_address),
                calls=calls,
                nonce=nonce_str,
                deadline=deadline_str,
                chain_id=self.chain_id,
            )
            wrapped_sig = wrap_deposit_wallet_signature(
                signer=EvmAddress(self.signer_address),
                signer_type="OWNER",
                signature=sig,
            )
            payload: dict[str, Any] = {
                "type": "WALLET",
                "from": self.signer_address,
                "to": str(PRODUCTION_CONFIG.wallet_derivation.deposit_wallet_factory),
                "nonce": nonce_str,
                "signature": str(wrapped_sig),
                "metadata": metadata,
                "depositWalletParams": {
                    "depositWallet": str(self.funder_address),
                    "deadline": deadline_str,
                    "calls": [
                        {
                            "target": str(c.to),
                            "value": str(c.value),
                            "data": str(c.data),
                        }
                        for c in calls
                    ],
                },
            }
        elif tx_type == "SAFE":
            if len(calls) == 1:
                target_to = calls[0].to
                safe_data = calls[0].data
                operation = 0
            else:
                target_to = EvmAddress(PRODUCTION_CONFIG.safe_multisend)
                safe_data = encode_safe_multisend_call(calls)
                operation = 1
            sig = sign_safe_transaction(
                self._account,
                safe_address=EvmAddress(self.funder_address),
                to=target_to,
                data=safe_data,
                value=0,
                operation=operation,
                nonce=nonce_str,
                chain_id=self.chain_id,
            )
            payload = {
                "from": self.signer_address,
                "to": str(target_to),
                "proxyWallet": self.funder_address,
                "data": str(safe_data),
                "nonce": nonce_str,
                "signature": str(sig),
                "signatureParams": {
                    "gasPrice": "0",
                    "operation": str(int(operation)),
                    "safeTxnGas": "0",
                    "baseGas": "0",
                    "gasToken": "0x0000000000000000000000000000000000000000",
                    "refundReceiver": "0x0000000000000000000000000000000000000000",
                },
                "type": "SAFE",
                "metadata": metadata,
            }
        else:
            rp_resp = requests.get(
                f"{self.relayer_url}/relay-payload",
                params={"address": self.signer_address, "type": "PROXY"},
                headers=self._headers(),
                timeout=5.0,
            )
            rp_resp.raise_for_status()
            rp_data = rp_resp.json()
            relayer_addr = EvmAddress(str(rp_data["address"]))
            rp_nonce = str(rp_data.get("nonce", nonce_str))
            proxy_factory = EvmAddress(PRODUCTION_CONFIG.wallet_derivation.proxy_factory)
            relay_hub = EvmAddress(PRODUCTION_CONFIG.relay_hub)
            proxy_data = encode_proxy_call(calls)
            gas_limit_str = "1000000"
            struct_hash = build_proxy_transaction_hash(
                from_address=EvmAddress(self.signer_address),
                to=proxy_factory,
                data=proxy_data,
                relayer_fee="0",
                gas_price="0",
                gas_limit=gas_limit_str,
                nonce=rp_nonce,
                relay_hub=relay_hub,
                relay=relayer_addr,
            )
            sig = sign_proxy_message(self._account, struct_hash)
            payload = {
                "from": self.signer_address,
                "to": str(proxy_factory),
                "proxyWallet": self.funder_address,
                "data": str(proxy_data),
                "nonce": rp_nonce,
                "signature": str(sig),
                "signatureParams": {
                    "gasPrice": "0",
                    "gasLimit": gas_limit_str,
                    "relayerFee": "0",
                    "relayHub": str(relay_hub),
                    "relay": str(relayer_addr),
                },
                "type": "PROXY",
                "metadata": metadata,
            }

        sub_resp = requests.post(
            f"{self.relayer_url}/submit",
            json=payload,
            headers=self._headers(),
            timeout=8.0,
        )
        if sub_resp.status_code == 404:
            sub_resp = requests.post(
                f"{self.relayer_url}/v1/account/transactions",
                json=payload,
                headers=self._headers(),
                timeout=8.0,
            )
        sub_resp.raise_for_status()
        body = sub_resp.json()
        return _RelayerExecResponse(
            transaction_id=(
                body.get("transactionID")
                or body.get("transactionId")
                or body.get("transaction_id")
                or body.get("id")
            ),
            transaction_hash=(
                body.get("transactionHash")
                or body.get("transaction_hash")
                or body.get("hash")
                or body.get("txHash")
            ),
            state=body.get("state"),
        )

    def get_transaction(self, transaction_id: str) -> dict[str, Any] | None:
        """Fetch a single Relayer transaction by ID across V2 and V1 endpoints (`[P2-R6-3]`)."""
        tx_id = str(transaction_id or "").strip()
        if not tx_id:
            return None
        try:
            resp = requests.get(
                f"{self.relayer_url}/transaction",
                params={"id": tx_id},
                headers=self._headers(),
                timeout=4.0,
            )
            if resp.status_code != 200:
                resp = requests.get(
                    f"{self.relayer_url}/v1/account/transactions/{tx_id}",
                    headers=self._headers(),
                    timeout=4.0,
                )
            if resp.status_code == 200:
                body = resp.json()
                row: Any = body
                if isinstance(body, list) and body:
                    row = body[0]
                elif isinstance(body, Mapping):
                    for key in ("transaction", "data", "result"):
                        if isinstance(body.get(key), Mapping):
                            row = body[key]
                            break
                if isinstance(row, Mapping):
                    out_row = dict(row)
                    tx_h = (
                        out_row.get("transactionHash")
                        or out_row.get("transaction_hash")
                        or out_row.get("hash")
                        or out_row.get("txHash")
                    )
                    if tx_h:
                        out_row.setdefault("transactionHash", tx_h)
                    return out_row
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_tx_list(body: Any) -> list[dict[str, Any]]:
        if isinstance(body, list):
            return [dict(item) for item in body if isinstance(item, Mapping)]
        if isinstance(body, Mapping):
            for key in ("transactions", "data", "results", "items", "records"):
                val = body.get(key)
                if isinstance(val, list):
                    return [dict(item) for item in val if isinstance(item, Mapping)]
            if any(
                body.get(k) is not None
                for k in ("id", "transactionID", "transactionId", "transaction_id", "transactionHash", "transaction_hash")
            ):
                return [dict(body)]
        return []

    def get_transactions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch recent Relayer transactions for this signer/wallet across V2 (`/transactions`) and V1 (`/v1/account/transactions`) (`[P2-R6-3]`)."""
        tx_type = {3: "WALLET", 2: "SAFE", 1: "PROXY"}.get(self.signature_type, "SAFE")
        endpoints = [
            (
                f"{self.relayer_url}/transactions",
                {"address": self.signer_address, "type": tx_type, "limit": int(limit)},
            ),
            (
                f"{self.relayer_url}/v1/account/transactions",
                {"address": self.signer_address, "type": tx_type, "limit": int(limit)},
            ),
        ]
        for url, params in endpoints:
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=self._headers(),
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    extracted = self._extract_tx_list(resp.json())
                    if extracted:
                        return extracted
            except Exception:
                continue
        return []

    def list_transactions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Alias for `get_transactions` (`[P2-R6-3]`)."""
        return self.get_transactions(limit=limit)

    def _matches_transaction_attribution(
        self,
        item: Mapping[str, Any],
        *,
        metadata: str = "",
        target_to: str = "",
        calldata_hex: str = "",
    ) -> bool:
        """Verify transaction ownership (`signer_address` / `funder_address`) and batch attribution (`metadata`, `to`, `data`) (`[P2-R6-3]`)."""
        rec_id = (
            item.get("transactionID")
            or item.get("transactionId")
            or item.get("transaction_id")
            or item.get("id")
        )
        rec_hash = (
            item.get("transactionHash")
            or item.get("transaction_hash")
            or item.get("hash")
            or item.get("txHash")
        )
        if not rec_id and not rec_hash:
            return False

        # 1. Verify wallet / signer ownership if present on the Relayer record
        allowed_owners = {
            a.strip().lower()
            for a in (self.signer_address, self.funder_address, self.relayer_api_key_address)
            if a and str(a).strip()
        }
        if allowed_owners:
            item_from = str(
                item.get("from") or item.get("signer") or item.get("signerAddress") or item.get("user") or ""
            ).strip().lower()
            item_proxy = str(
                item.get("proxyWallet") or item.get("proxy_wallet") or item.get("safe") or item.get("wallet") or ""
            ).strip().lower()
            dw_params = item.get("depositWalletParams") if isinstance(item.get("depositWalletParams"), Mapping) else {}
            item_dw = str(
                (dw_params.get("depositWallet") if isinstance(dw_params, Mapping) else None)
                or item.get("depositWallet")
                or ""
            ).strip().lower()
            if item_from and item_from not in allowed_owners:
                return False
            if item_proxy and item_proxy not in allowed_owners:
                return False
            if item_dw and item_dw not in allowed_owners:
                return False

        # 2. Verify target contract & calldata attribution if present on the Relayer record
        exp_to = str(target_to or "").strip().lower()
        exp_data = str(calldata_hex or "").strip().lower()
        exp_data_clean = exp_data.removeprefix("0x")

        nested_calls: list[tuple[str, str]] = []
        dw_params = item.get("depositWalletParams")
        if isinstance(dw_params, Mapping) and isinstance(dw_params.get("calls"), list):
            for c in dw_params["calls"]:
                if isinstance(c, Mapping):
                    nested_calls.append(
                        (
                            str(c.get("target") or c.get("to") or "").strip().lower(),
                            str(c.get("data") or c.get("calldata") or "").strip().lower(),
                        )
                    )
        for list_key in ("calls", "transactions"):
            raw_calls = item.get(list_key)
            if isinstance(raw_calls, list):
                for c in raw_calls:
                    if isinstance(c, Mapping):
                        nested_calls.append(
                            (
                                str(c.get("target") or c.get("to") or "").strip().lower(),
                                str(c.get("data") or c.get("calldata") or "").strip().lower(),
                            )
                        )

        item_to = str(item.get("to") or item.get("target") or "").strip().lower()
        item_data = str(item.get("data") or item.get("calldata") or "").strip().lower()

        call_matched = False
        if nested_calls:
            for c_to, c_data in nested_calls:
                to_ok = (not exp_to) or (not c_to) or (c_to == exp_to)
                data_ok = (not exp_data_clean) or (exp_data_clean in c_data.removeprefix("0x"))
                if to_ok and data_ok and (c_to or c_data):
                    call_matched = True
                    break
            if (exp_to or exp_data_clean) and not call_matched:
                return False
        elif item_data:
            if exp_data_clean and exp_data_clean not in item_data.removeprefix("0x"):
                return False
            wrapper_addrs = {
                str(PRODUCTION_CONFIG.wallet_derivation.deposit_wallet_factory).lower(),
                str(PRODUCTION_CONFIG.safe_multisend).lower(),
                str(PRODUCTION_CONFIG.wallet_derivation.proxy_factory).lower(),
            }
            if exp_to and item_to and item_to not in wrapper_addrs and item_to != exp_to:
                return False
            if (exp_data_clean and exp_data_clean in item_data.removeprefix("0x")) or (exp_to and item_to == exp_to):
                call_matched = True
        elif item_to:
            wrapper_addrs = {
                str(PRODUCTION_CONFIG.wallet_derivation.deposit_wallet_factory).lower(),
                str(PRODUCTION_CONFIG.safe_multisend).lower(),
                str(PRODUCTION_CONFIG.wallet_derivation.proxy_factory).lower(),
            }
            if exp_to and item_to not in wrapper_addrs and item_to != exp_to:
                return False

        # 3. Verify `metadata` (`redeem_request_id`) or exact calldata match
        req_meta = str(metadata or "").strip()
        item_meta = str(
            item.get("metadata")
            or item.get("request_id")
            or item.get("requestId")
            or item.get("clientRequestId")
            or ""
        ).strip()
        if req_meta:
            if item_meta:
                return item_meta == req_meta
            return bool(exp_data_clean and call_matched)
        return bool(exp_to and exp_data_clean and call_matched)

    def find_transaction_by_metadata(
        self,
        metadata: str,
        target_to: str = "",
        calldata_hex: str = "",
    ) -> dict[str, Any] | None:
        """Search recent Relayer transactions and return the transaction matching `metadata` and attribution (`[P2-R6-3]`)."""
        txs = self.get_transactions()
        for item in txs:
            if not isinstance(item, Mapping):
                continue
            if self._matches_transaction_attribution(
                item,
                metadata=metadata,
                target_to=target_to,
                calldata_hex=calldata_hex,
            ):
                return dict(item)
        return None

    def poll_until_state(
        self,
        transaction_id: str,
        states: list[str],
        fail_state: str = "STATE_FAILED",
        max_polls: int = 8,
        poll_frequency: int = 1000,
    ) -> dict[str, Any] | None:
        target_states = {s.upper() for s in states}
        fail_states = {fail_state.upper(), "FAILED", "STATE_FAILED", "STATE_INVALID", "INVALID"}
        sleep_sec = max(0.05, float(poll_frequency) / 1000.0)
        for _ in range(max(1, int(max_polls))):
            row = self.get_transaction(transaction_id)
            if isinstance(row, Mapping):
                st = str(row.get("state") or "").upper()
                if st in fail_states:
                    return dict(row)
                tx_h = (
                    row.get("transactionHash")
                    or row.get("transaction_hash")
                    or row.get("hash")
                    or row.get("txHash")
                )
                if st in target_states and tx_h:
                    out_row = dict(row)
                    out_row.setdefault("transactionHash", tx_h)
                    return out_row
            time.sleep(sleep_sec)
        return None


CONFIRMED_RELAYER_STATES = {"STATE_CONFIRMED", "CONFIRMED", "STATE_MINED", "MINED"}
FAILED_RELAYER_STATES = {"STATE_FAILED", "FAILED", "STATE_INVALID", "INVALID"}


class AutoRedeemer:
    """Automatic post-settlement V2/CTF redeemer to Polymarket Wallet Cash (zero withdrawal capability)."""

    def __init__(
        self,
        mode: str = "shadow",
        relayer_host: str = "https://relayer-v2.polymarket.com",
        chain_id: int = 137,
        private_key: str | None = None,
        signature_type: int = 2,
        funder_address: str | None = None,
        relayer_api_key: str | None = None,
        relayer_api_secret: str | None = None,
        relayer_api_key_address: str | None = None,
        builder_api_key: str | None = None,
        builder_secret: str | None = None,
        builder_passphrase: str | None = None,
        polygon_rpc_url: str = "https://polygon-rpc.com",
        ctf_address: str = OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS,
        neg_risk_adapter_address: str = OFFICIAL_NEG_RISK_COLLATERAL_ADAPTER_ADDRESS,
        v2_router_address: str = OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS,
        pusd_address: str = OFFICIAL_POLYGON_PUSD_ADDRESS,
        usdce_address: str | None = None,
        withdrawal_permission: bool = False,
        require_onchain_receipt: bool = True,
    ) -> None:
        assert_no_withdrawal_permission({"withdrawal_permission": withdrawal_permission})
        self.withdrawal_permission = False
        self.mode = str(mode or "shadow").strip().lower()
        self.relayer_host = relayer_host.rstrip("/")
        self.chain_id = int(chain_id)
        self.signature_type = int(signature_type if signature_type is not None else 2)
        self.polygon_rpc_url = polygon_rpc_url
        self.ctf_address = ctf_address
        self.neg_risk_adapter_address = neg_risk_adapter_address
        self.v2_router_address = v2_router_address
        self.pusd_address = pusd_address or OFFICIAL_POLYGON_PUSD_ADDRESS
        self.usdce_address = self.pusd_address
        self.require_onchain_receipt = bool(require_onchain_receipt)

        has_valid_pk = is_valid_private_key(private_key)
        signer_eoa = ""
        default_funder = ""
        pk_clean = ""
        if has_valid_pk and private_key:
            pk_clean = private_key.strip()
            if not pk_clean.startswith(("0x", "0X")):
                pk_clean = "0x" + pk_clean
            cands = derive_polymarket_candidate_wallets(pk_clean, self.chain_id)
            signer_eoa = cands["signer_eoa"]
            if self.signature_type == 3:
                default_funder = cands["beacon_deposit_wallet"]
            elif self.signature_type == 1:
                default_funder = cands["poly_proxy_wallet"]
            elif self.signature_type == 0:
                default_funder = signer_eoa
            else:
                default_funder = cands["gnosis_safe_wallet"]

        self.signer_address = signer_eoa
        self.safe_funder_address = (
            funder_address.strip() if is_valid_eth_address(funder_address) else default_funder
        )
        self.funder_address = self.safe_funder_address
        self.relayer_api_key_address = signer_eoa

        key_clean = (relayer_api_key or builder_api_key or "").strip()
        secret_clean = (relayer_api_secret or relayer_api_key_address or builder_secret or "").strip()
        has_relayer_key = bool(key_clean and "YOUR_" not in key_clean.upper())
        has_relayer_secret = bool(secret_clean and "YOUR_" not in secret_clean.upper())

        if has_relayer_secret and is_valid_eth_address(secret_clean):
            self.relayer_api_key_address = secret_clean

        self.relayer_client: Any = None
        if has_valid_pk and has_relayer_key and self.funder_address:
            self.relayer_client = _PolymarketRelayerV2Client(
                relayer_url=self.relayer_host,
                chain_id=self.chain_id,
                private_key=pk_clean,
                signer_address=self.signer_address,
                funder_address=self.funder_address,
                signature_type=self.signature_type,
                relayer_api_key=key_clean,
                relayer_api_key_address=self.relayer_api_key_address,
                relayer_api_secret=secret_clean if has_relayer_secret else "",
            )

    def _verify_onchain_receipt_status(self, tx_hash: str) -> tuple[bool | None, int | None]:
        """Check Polygon RPC `eth_getTransactionReceipt`.

        Returns:
            `(True, block_number)` if confirmed (`status == 1`)
            `(False, block_number)` if reverted (`status == 0`)
            `(None, None)` if receipt not yet available or RPC unreachable.
        """
        if not tx_hash or not str(tx_hash).startswith("0x"):
            return None, None
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getTransactionReceipt",
                "params": [str(tx_hash)],
            }
            resp = requests.post(self.polygon_rpc_url, json=payload, timeout=2.0)
            if resp.status_code == 200:
                res = resp.json().get("result")
                if isinstance(res, Mapping) and res.get("blockNumber"):
                    blk = int(str(res["blockNumber"]), 16)
                    st = res.get("status")
                    if st in ("0x1", 1, "1", True):
                        return True, blk
                    if st in ("0x0", 0, "0", False):
                        return False, blk
        except Exception:
            pass
        return None, None

    def _finalize_or_poll_redemption(
        self,
        *,
        tx_id: str | None,
        tx_hash: str | None,
        polled_state: str,
        expected_payout: Decimal,
        t0: float,
    ) -> dict[str, Any]:
        """Poll Relayer (when needed) and strictly verify both Relayer state and on-chain receipt (`[P1-4]`, `[P1-E]`)."""
        state_upper = str(polled_state or "").strip().upper()
        if (
            tx_id
            and hasattr(self.relayer_client, "poll_until_state")
            and (not tx_hash or state_upper not in CONFIRMED_RELAYER_STATES)
            and state_upper not in FAILED_RELAYER_STATES
        ):
            polled = self.relayer_client.poll_until_state(
                transaction_id=tx_id,
                states=list(CONFIRMED_RELAYER_STATES),
                fail_state="STATE_FAILED",
                max_polls=5,
                poll_frequency=1000,
            )
            if isinstance(polled, Mapping):
                state_upper = str(polled.get("state") or state_upper).strip().upper()
                tx_hash = (
                    polled.get("transactionHash")
                    or polled.get("transaction_hash")
                    or polled.get("hash")
                    or polled.get("txHash")
                    or tx_hash
                )

        latency = round(time.perf_counter() - t0, 4)

        if state_upper in FAILED_RELAYER_STATES:
            return {
                "live_expected_payout_usdc": float(expected_payout),
                "redeem_tx_id": tx_id,
                "redeem_tx_hash": f"relayer_id:{tx_id}" if tx_id else None,
                "redeem_status": "REDEEM_FAILED",
                "redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": latency,
                "error": f"RELAYER_TX_FAILED:{state_upper}",
            }

        has_valid_tx_hash = bool(
            tx_hash
            and isinstance(tx_hash, str)
            and tx_hash.startswith("0x")
            and len(tx_hash) >= 10
        )

        # [P1-E] If Relayer reported a non-confirmed state (e.g. STATE_NEW / STATE_EXECUTED), NEVER mark REDEEMED!
        if state_upper and state_upper not in CONFIRMED_RELAYER_STATES:
            return {
                "live_expected_payout_usdc": float(expected_payout),
                "redeem_tx_id": tx_id,
                "redeem_tx_hash": str(tx_hash) if has_valid_tx_hash else (f"relayer_id:{tx_id}" if tx_id else None),
                "redeem_status": "REDEEM_PENDING",
                "redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": latency,
                "error": f"AWAITING_RELAYER_CONFIRMATION:{state_upper}",
            }

        if not has_valid_tx_hash:
            return {
                "live_expected_payout_usdc": float(expected_payout),
                "redeem_tx_id": tx_id,
                "redeem_tx_hash": f"relayer_id:{tx_id}" if tx_id else None,
                "redeem_status": "REDEEM_PENDING",
                "redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": latency,
                "error": "AWAITING_CONFIRMED_TX_HASH",
            }

        receipt_ok, _ = self._verify_onchain_receipt_status(str(tx_hash))
        if receipt_ok is False:
            return {
                "live_expected_payout_usdc": float(expected_payout),
                "redeem_tx_id": tx_id,
                "redeem_tx_hash": str(tx_hash),
                "redeem_status": "REDEEM_FAILED",
                "redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": latency,
                "error": "ONCHAIN_RECEIPT_REVERTED",
            }
        if self.require_onchain_receipt and receipt_ok is not True:
            return {
                "live_expected_payout_usdc": float(expected_payout),
                "redeem_tx_id": tx_id,
                "redeem_tx_hash": str(tx_hash),
                "redeem_status": "REDEEM_PENDING",
                "redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": latency,
                "error": "AWAITING_ONCHAIN_RECEIPT",
            }

        return {
            "live_expected_payout_usdc": float(expected_payout),
            "redeem_tx_id": tx_id,
            "redeem_tx_hash": str(tx_hash),
            "redeem_status": "REDEEMED",
            "redeemed_usdc_amount": float(expected_payout),
            "redeem_latency_seconds": latency,
        }

    @staticmethod
    def _order_batch_key(order: Mapping[str, Any], index: int) -> str:
        if order.get("signal_id"):
            return str(order["signal_id"])
        if order.get("clob_order_id"):
            return f"clob:{order['clob_order_id']}"
        if order.get("paper_order_id"):
            return f"paper:{order['paper_order_id']}"
        return f"idx:{index}:{str(order.get('outcome', '')).upper()}:{order.get('filled_quantity', '0')}"

    def _verify_reconciled_transaction_attribution(
        self,
        item: Mapping[str, Any],
        *,
        request_id: str,
        target_to: str,
        calldata_hex: str,
    ) -> bool:
        """Verify ownership (`signer_address` / `funder_address`) and batch attribution (`request_id`, `target_to`, `calldata_hex`)
        for any candidate transaction returned by `self.relayer_client` (`[P2-R6-3]`).
        """
        rec_id = (
            item.get("transactionID")
            or item.get("transactionId")
            or item.get("transaction_id")
            or item.get("id")
        )
        rec_hash = (
            item.get("transactionHash")
            or item.get("transaction_hash")
            or item.get("hash")
            or item.get("txHash")
        )
        if not rec_id and not rec_hash:
            return False

        allowed_owners = {
            a.strip().lower()
            for a in (
                self.signer_address,
                self.funder_address,
                getattr(self.relayer_client, "signer_address", ""),
                getattr(self.relayer_client, "funder_address", ""),
            )
            if a and str(a).strip()
        }
        if allowed_owners:
            item_from = str(
                item.get("from") or item.get("signer") or item.get("signerAddress") or item.get("user") or ""
            ).strip().lower()
            item_proxy = str(
                item.get("proxyWallet") or item.get("proxy_wallet") or item.get("safe") or item.get("wallet") or ""
            ).strip().lower()
            dw_params = item.get("depositWalletParams") if isinstance(item.get("depositWalletParams"), Mapping) else {}
            item_dw = str(
                (dw_params.get("depositWallet") if isinstance(dw_params, Mapping) else None)
                or item.get("depositWallet")
                or ""
            ).strip().lower()
            if item_from and item_from not in allowed_owners:
                return False
            if item_proxy and item_proxy not in allowed_owners:
                return False
            if item_dw and item_dw not in allowed_owners:
                return False

        exp_to = str(target_to or "").strip().lower()
        exp_data = str(calldata_hex or "").strip().lower()
        exp_data_clean = exp_data.removeprefix("0x")

        nested_calls: list[tuple[str, str]] = []
        dw_params = item.get("depositWalletParams")
        if isinstance(dw_params, Mapping) and isinstance(dw_params.get("calls"), list):
            for c in dw_params["calls"]:
                if isinstance(c, Mapping):
                    nested_calls.append(
                        (
                            str(c.get("target") or c.get("to") or "").strip().lower(),
                            str(c.get("data") or c.get("calldata") or "").strip().lower(),
                        )
                    )
        for list_key in ("calls", "transactions"):
            raw_calls = item.get(list_key)
            if isinstance(raw_calls, list):
                for c in raw_calls:
                    if isinstance(c, Mapping):
                        nested_calls.append(
                            (
                                str(c.get("target") or c.get("to") or "").strip().lower(),
                                str(c.get("data") or c.get("calldata") or "").strip().lower(),
                            )
                        )

        item_to = str(item.get("to") or item.get("target") or "").strip().lower()
        item_data = str(item.get("data") or item.get("calldata") or "").strip().lower()

        call_matched = False
        if nested_calls:
            for c_to, c_data in nested_calls:
                to_ok = (not exp_to) or (not c_to) or (c_to == exp_to)
                data_ok = (not exp_data_clean) or (exp_data_clean in c_data.removeprefix("0x"))
                if to_ok and data_ok and (c_to or c_data):
                    call_matched = True
                    break
            if (exp_to or exp_data_clean) and not call_matched:
                return False
        elif item_data:
            if exp_data_clean and exp_data_clean not in item_data.removeprefix("0x"):
                return False
            wrapper_addrs = {
                str(PRODUCTION_CONFIG.wallet_derivation.deposit_wallet_factory).lower(),
                str(PRODUCTION_CONFIG.safe_multisend).lower(),
                str(PRODUCTION_CONFIG.wallet_derivation.proxy_factory).lower(),
            }
            if exp_to and item_to and item_to not in wrapper_addrs and item_to != exp_to:
                return False
            if (exp_data_clean and exp_data_clean in item_data.removeprefix("0x")) or (exp_to and item_to == exp_to):
                call_matched = True
        elif item_to:
            wrapper_addrs = {
                str(PRODUCTION_CONFIG.wallet_derivation.deposit_wallet_factory).lower(),
                str(PRODUCTION_CONFIG.safe_multisend).lower(),
                str(PRODUCTION_CONFIG.wallet_derivation.proxy_factory).lower(),
            }
            if exp_to and item_to not in wrapper_addrs and item_to != exp_to:
                return False

        req_meta = str(request_id or "").strip()
        item_meta = str(
            item.get("metadata")
            or item.get("request_id")
            or item.get("requestId")
            or item.get("clientRequestId")
            or ""
        ).strip()
        if req_meta:
            if item_meta:
                return item_meta == req_meta
            return bool(exp_data_clean and call_matched)
        return bool(exp_to and exp_data_clean and call_matched)

    def _reconcile_unknown_redemption_submission(
        self,
        *,
        existing_settlement: Mapping[str, Any],
        request_id: str,
        fallback_target_to: str = "",
        fallback_calldata_hex: str = "",
    ) -> dict[str, Any] | None:
        """Attempt to reconcile an ambiguous redemption submission (`REDEEM_SUBMIT_UNKNOWN`) via Relayer (`[P1-R5-3]`, `[P2-R6-3]`)."""
        if self.relayer_client is None:
            return None
        target_to = str(existing_settlement.get("redeem_target_to") or fallback_target_to or "").strip().lower()
        calldata_hex = str(existing_settlement.get("redeem_calldata_hex") or fallback_calldata_hex or "").strip().lower()
        try:
            if hasattr(self.relayer_client, "find_transaction_by_metadata") and request_id:
                try:
                    found = self.relayer_client.find_transaction_by_metadata(
                        request_id,
                        target_to=target_to,
                        calldata_hex=calldata_hex,
                    )
                except TypeError:
                    found = self.relayer_client.find_transaction_by_metadata(request_id)
                if isinstance(found, Mapping) and self._verify_reconciled_transaction_attribution(
                    found,
                    request_id=request_id,
                    target_to=target_to,
                    calldata_hex=calldata_hex,
                ):
                    return dict(found)
            for method_name in ("get_transactions", "list_transactions"):
                if hasattr(self.relayer_client, method_name):
                    txs = getattr(self.relayer_client, method_name)()
                    if isinstance(txs, list):
                        for item in txs:
                            if not isinstance(item, Mapping):
                                continue
                            if self._verify_reconciled_transaction_attribution(
                                item,
                                request_id=request_id,
                                target_to=target_to,
                                calldata_hex=calldata_hex,
                            ):
                                return dict(item)
        except Exception:
            pass
        return None

    def redeem_settled_market(
        self,
        market_id: str,
        condition_id: str,
        official_outcome: str,
        live_orders: list[Mapping[str, Any]],
        market_protocol: str = "v1",
        neg_risk: bool = False,
        outcome_index: int | None = None,
        existing_settlement: Mapping[str, Any] | None = None,
        on_progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Evaluate live positions for a settled market and redeem winning tokens back to Wallet pUSD.

        Enforces `[P1-R3-1]`, `[P1-R4-3]`, and `[P2-R4-6]`:
          - Binds every submitted redemption transaction (`redeem_tx_id` / `redeem_tx_hash`) to the exact
            batch of orders (`pending_batch_order_keys`), shares (`pending_batch_shares`), and expected payout
            (`pending_batch_payout_usdc`) at submission time.
          - Persists the pending batch before submission and immediately persists `redeem_tx_id`/`redeem_tx_hash`
            via `on_progress_callback` BEFORE polling for confirmation (`[P1-R4-3]`).
          - Never guesses batch attribution by order sequence when a legacy settlement record lacks
            `pending_batch_order_keys` (`[P2-R4-6]`).
        """
        winner = str(official_outcome).upper().strip()
        if not live_orders:
            return {
                "live_expected_payout_usdc": 0.0,
                "redeem_tx_hash": None,
                "redeem_status": "NO_LIVE_POSITION",
                "redeemed_usdc_amount": 0.0,
                "newly_redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": 0.0,
                "redeemed_order_keys": [],
                "newly_redeemed_order_keys": [],
                "pending_batch_order_keys": [],
                "pending_batch_payout_usdc": 0.0,
                "has_remaining_unredeemed": False,
            }

        already_redeemed_keys: set[str] = set()
        if isinstance(existing_settlement, Mapping):
            raw_red_keys = existing_settlement.get("redeemed_order_keys")
            if isinstance(raw_red_keys, list):
                already_redeemed_keys.update(str(k) for k in raw_red_keys if k is not None)

        unredeemed_orders: list[tuple[str, Mapping[str, Any], Decimal]] = []
        unredeemed_winning_shares = Decimal("0")
        already_redeemed_shares = Decimal("0")
        has_in_flight = False

        for idx, order in enumerate(live_orders):
            pos_status = str(order.get("position_status") or "").upper()
            if pos_status in ("RESERVED", "ORDER_STATE_UNKNOWN", "DELAYED", "SUBMITTING"):
                has_in_flight = True
                continue
            if pos_status == "NO_FILL":
                continue
            if str(order.get("outcome", "")).upper() == winner:
                qty = to_decimal(order.get("filled_quantity", "0"), "filled_quantity")
                if qty <= Decimal("0"):
                    continue
                okey = self._order_batch_key(order, idx)
                if pos_status == "WON_REDEEMED" or okey in already_redeemed_keys:
                    already_redeemed_keys.add(okey)
                    already_redeemed_shares += qty
                    continue
                unredeemed_orders.append((okey, order, qty))
                unredeemed_winning_shares += qty

        t0 = time.perf_counter()

        # [P1-E], [P1-R3-1], & [P1-R5-3]: If an existing pending redemption transaction or an
        # un-reconciled submission (`REDEEM_SUBMIT_UNKNOWN` / `pending_batch_order_keys` without ID)
        # exists, recover/reconcile it first and NEVER auto-submit a duplicate transaction!
        if (
            isinstance(existing_settlement, Mapping)
            and str(existing_settlement.get("redeem_status") or "").upper() in ("REDEEM_PENDING", "REDEEM_SUBMIT_UNKNOWN")
            and self.mode != "shadow"
            and self.relayer_client is not None
        ):
            prev_tx_id = existing_settlement.get("redeem_tx_id")
            prev_tx_hash = existing_settlement.get("redeem_tx_hash")
            if isinstance(prev_tx_hash, str) and prev_tx_hash.startswith("relayer_id:"):
                if not prev_tx_id:
                    prev_tx_id = prev_tx_hash.split("relayer_id:", 1)[1]
                prev_tx_hash = None

            raw_pending_keys = existing_settlement.get("pending_batch_order_keys")
            bound_keys: list[str] = (
                [str(k) for k in raw_pending_keys if k is not None]
                if isinstance(raw_pending_keys, list) and raw_pending_keys
                else []
            )
            prev_submit_state = str(existing_settlement.get("submit_state") or "").upper()
            prev_request_id = str(existing_settlement.get("redeem_request_id") or "").strip()
            prev_error_lower = str(existing_settlement.get("error") or "").lower()
            is_retryable_http_400 = (
                prev_submit_state == "SUBMIT_HTTP_400_REJECTED"
                or "400 client error" in prev_error_lower
                or "status_code=400" in prev_error_lower
            )
            has_unknown_submission = (
                not prev_tx_id
                and not (isinstance(prev_tx_hash, str) and prev_tx_hash.startswith("0x"))
                and not is_retryable_http_400
                and (
                    bool(bound_keys)
                    or prev_submit_state in ("REDEEM_SUBMIT_UNKNOWN", "SUBMITTING", "SUBMITTED")
                    or bool(prev_request_id)
                )
            )

            reconciled_state = ""
            if has_unknown_submission:
                cond_fb = str(condition_id or "").strip()
                ord_protos = {str(o.get("market_protocol") or "").strip().lower() for o in live_orders}
                has_v2_tok = any(is_v2_position_id(str(o.get("token_id") or "").strip()) for o in live_orders)
                is_v2_fb = (
                    str(market_protocol or "").strip().lower() == "v2"
                    or "v2" in ord_protos
                    or has_v2_tok
                    or len(cond_fb.removeprefix("0x").removeprefix("0X")) == 62
                )
                eff_neg_fb = bool(neg_risk or any(bool(o.get("neg_risk")) for o in live_orders))
                fb_shares = (
                    sum((q for (k, _, q) in unredeemed_orders if k in set(bound_keys)), Decimal("0"))
                    if bound_keys
                    else unredeemed_winning_shares
                )
                fb_target_to = ""
                fb_calldata_hex = ""
                try:
                    if is_v2_fb and fb_shares > Decimal("0"):
                        res_idx_fb = (
                            int(outcome_index)
                            if outcome_index is not None
                            else (0 if winner in ("UP", "YES", "0") else 1)
                        )
                        atoms_fb = int((fb_shares * Decimal("1000000")).to_integral_value(rounding=ROUND_DOWN))
                        fb_target_to = self.v2_router_address
                        fb_calldata_hex = encode_v2_router_redeem_calldata(
                            condition_id=cond_fb,
                            outcome_index=res_idx_fb,
                            amount_base_units=atoms_fb,
                            router_address=fb_target_to,
                        )
                    elif cond_fb:
                        fb_target_to = self.neg_risk_adapter_address if eff_neg_fb else self.ctf_address
                        fb_calldata_hex = encode_redeem_positions_calldata(
                            condition_id=cond_fb,
                            collateral_address=self.pusd_address,
                            index_sets=[1, 2],
                            target_contract_address=fb_target_to,
                        )
                except Exception:
                    pass

                reconciled_tx = self._reconcile_unknown_redemption_submission(
                    existing_settlement=existing_settlement,
                    request_id=prev_request_id,
                    fallback_target_to=fb_target_to,
                    fallback_calldata_hex=fb_calldata_hex,
                )
                if isinstance(reconciled_tx, Mapping):
                    rec_id = (
                        reconciled_tx.get("transactionID")
                        or reconciled_tx.get("transactionId")
                        or reconciled_tx.get("transaction_id")
                        or reconciled_tx.get("id")
                    )
                    rec_hash = (
                        reconciled_tx.get("transactionHash")
                        or reconciled_tx.get("transaction_hash")
                        or reconciled_tx.get("hash")
                        or reconciled_tx.get("txHash")
                    )
                    reconciled_state = str(reconciled_tx.get("state") or "").strip().upper()
                    if rec_id:
                        prev_tx_id = str(rec_id)
                    if rec_hash and str(rec_hash).startswith("0x"):
                        prev_tx_hash = str(rec_hash)

            if prev_tx_id or (isinstance(prev_tx_hash, str) and prev_tx_hash.startswith("0x")):
                raw_pending_payout = existing_settlement.get("pending_batch_payout_usdc")
                if raw_pending_payout is None or to_decimal(raw_pending_payout or 0) <= Decimal("0"):
                    raw_pending_payout = existing_settlement.get("live_expected_payout_usdc")

                if raw_pending_payout is not None and to_decimal(raw_pending_payout or 0) > Decimal("0"):
                    bound_payout = to_decimal(raw_pending_payout, "pending_batch_payout_usdc").quantize(
                        USD_MICRO, rounding=ROUND_DOWN
                    )
                elif bound_keys:
                    bound_shares = sum(
                        (q for (k, _, q) in unredeemed_orders if k in set(bound_keys)),
                        Decimal("0"),
                    )
                    bound_payout = (bound_shares * Decimal("1.0")).quantize(USD_MICRO, rounding=ROUND_DOWN)
                else:
                    bound_payout = (unredeemed_winning_shares * Decimal("1.0")).quantize(
                        USD_MICRO, rounding=ROUND_DOWN
                    )

                # [P2-R4-6]: Never guess batch attribution by order sequence when a legacy record
                # lacks `pending_batch_order_keys`! Only attribute if there is a single unambiguous
                # order whose exact share payout matches `bound_payout` and no in-flight orders exist;
                # otherwise retain `REDEEM_PENDING` for verification without guessing.
                if not bound_keys:
                    if (
                        len(unredeemed_orders) == 1
                        and not has_in_flight
                        and (unredeemed_orders[0][2] * Decimal("1.0")).quantize(USD_MICRO, rounding=ROUND_DOWN)
                        == bound_payout
                    ):
                        bound_keys = [unredeemed_orders[0][0]]
                    else:
                        return {
                            "live_expected_payout_usdc": float(bound_payout),
                            "redeem_tx_id": str(prev_tx_id) if prev_tx_id else None,
                            "redeem_tx_hash": (
                                str(prev_tx_hash)
                                if prev_tx_hash
                                else (f"relayer_id:{prev_tx_id}" if prev_tx_id else None)
                            ),
                            "redeem_status": "REDEEM_PENDING",
                            "redeemed_usdc_amount": 0.0,
                            "newly_redeemed_usdc_amount": 0.0,
                            "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                            "error": "LEGACY_PENDING_BATCH_ATTRIBUTION_UNVERIFIED",
                            "redeemed_order_keys": sorted(already_redeemed_keys),
                            "newly_redeemed_order_keys": [],
                            "pending_batch_order_keys": [],
                            "pending_batch_shares": str(bound_payout),
                            "pending_batch_payout_usdc": float(bound_payout),
                            "has_remaining_unredeemed": True,
                        }

                if has_unknown_submission and callable(on_progress_callback):
                    try:
                        on_progress_callback(
                            {
                                "live_expected_payout_usdc": float(bound_payout),
                                "redeem_tx_id": str(prev_tx_id) if prev_tx_id else None,
                                "redeem_tx_hash": (
                                    str(prev_tx_hash)
                                    if prev_tx_hash
                                    else (f"relayer_id:{prev_tx_id}" if prev_tx_id else None)
                                ),
                                "redeem_status": "REDEEM_PENDING",
                                "submit_state": "SUBMITTED",
                                "redeem_request_id": prev_request_id or None,
                                "redeem_target_to": existing_settlement.get("redeem_target_to"),
                                "redeem_calldata_hex": existing_settlement.get("redeem_calldata_hex"),
                                "redeemed_usdc_amount": 0.0,
                                "newly_redeemed_usdc_amount": 0.0,
                                "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                                "redeemed_order_keys": sorted(already_redeemed_keys),
                                "newly_redeemed_order_keys": [],
                                "pending_batch_order_keys": bound_keys,
                                "pending_batch_shares": str(bound_payout),
                                "pending_batch_payout_usdc": float(bound_payout),
                                "has_remaining_unredeemed": False,
                            }
                        )
                    except Exception:
                        pass

                try:
                    resumed = self._finalize_or_poll_redemption(
                        tx_id=str(prev_tx_id) if prev_tx_id else None,
                        tx_hash=str(prev_tx_hash) if prev_tx_hash else None,
                        polled_state=reconciled_state or ("STATE_CONFIRMED" if (prev_tx_hash and not prev_tx_id) else "STATE_NEW"),
                        expected_payout=bound_payout,
                        t0=t0,
                    )
                except Exception as exc:
                    resumed = {
                        "live_expected_payout_usdc": float(bound_payout),
                        "redeem_tx_id": str(prev_tx_id) if prev_tx_id else None,
                        "redeem_tx_hash": str(prev_tx_hash) if prev_tx_hash else (f"relayer_id:{prev_tx_id}" if prev_tx_id else None),
                        "redeem_status": "REDEEM_PENDING",
                        "redeemed_usdc_amount": 0.0,
                        "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                        "error": f"{type(exc).__name__}:{str(exc)[:160]}",
                    }

                resumed["submit_state"] = "SUBMITTED"
                if prev_request_id:
                    resumed["redeem_request_id"] = prev_request_id

                if resumed["redeem_status"] == "REDEEM_PENDING":
                    resumed["newly_redeemed_usdc_amount"] = 0.0
                    resumed["redeemed_order_keys"] = sorted(already_redeemed_keys)
                    resumed["newly_redeemed_order_keys"] = []
                    resumed["pending_batch_order_keys"] = bound_keys
                    resumed["pending_batch_shares"] = str(bound_payout)
                    resumed["pending_batch_payout_usdc"] = float(bound_payout)
                    resumed["has_remaining_unredeemed"] = any(
                        k not in set(bound_keys) for (k, _, _) in unredeemed_orders
                    )
                    return resumed

                if resumed["redeem_status"] == "REDEEMED":
                    updated_redeemed_keys = set(already_redeemed_keys) | set(bound_keys)
                    remaining_orders = [
                        (k, o, q) for (k, o, q) in unredeemed_orders if k not in updated_redeemed_keys
                    ]
                    rem_payout = sum((q for (_, _, q) in remaining_orders), Decimal("0")).quantize(
                        USD_MICRO, rounding=ROUND_DOWN
                    )
                    resumed["newly_redeemed_usdc_amount"] = float(bound_payout)
                    resumed["redeemed_order_keys"] = sorted(updated_redeemed_keys)
                    resumed["newly_redeemed_order_keys"] = list(bound_keys)
                    resumed["pending_batch_order_keys"] = []
                    resumed["pending_batch_shares"] = "0.00"
                    resumed["pending_batch_payout_usdc"] = 0.0
                    resumed["submit_state"] = "CONFIRMED"
                    resumed["has_remaining_unredeemed"] = bool(remaining_orders)
                    resumed["remaining_unredeemed_order_keys"] = [k for (k, _, _) in remaining_orders]
                    resumed["remaining_unredeemed_payout_usdc"] = float(rem_payout)
                    return resumed
                # If REDEEM_FAILED, fall through to submit a fresh transaction for all currently unredeemed winning orders
            elif has_unknown_submission:
                # [P1-R5-3]: Submission outcome is unknown (`REDEEM_SUBMIT_UNKNOWN`) and could not yet be
                # reconciled to a transaction ID. Strictly forbid re-submitting until reconciled!
                raw_pending_payout = existing_settlement.get("pending_batch_payout_usdc")
                if raw_pending_payout is None or to_decimal(raw_pending_payout or 0) <= Decimal("0"):
                    raw_pending_payout = existing_settlement.get("live_expected_payout_usdc")
                if raw_pending_payout is not None and to_decimal(raw_pending_payout or 0) > Decimal("0"):
                    bound_payout = to_decimal(raw_pending_payout, "pending_batch_payout_usdc").quantize(
                        USD_MICRO, rounding=ROUND_DOWN
                    )
                elif bound_keys:
                    bound_shares = sum(
                        (q for (k, _, q) in unredeemed_orders if k in set(bound_keys)),
                        Decimal("0"),
                    )
                    bound_payout = (bound_shares * Decimal("1.0")).quantize(USD_MICRO, rounding=ROUND_DOWN)
                else:
                    bound_payout = (unredeemed_winning_shares * Decimal("1.0")).quantize(
                        USD_MICRO, rounding=ROUND_DOWN
                    )
                return {
                    "live_expected_payout_usdc": float(bound_payout),
                    "redeem_tx_id": None,
                    "redeem_tx_hash": None,
                    "redeem_status": "REDEEM_PENDING",
                    "submit_state": "REDEEM_SUBMIT_UNKNOWN",
                    "redeem_request_id": prev_request_id or None,
                    "redeem_target_to": existing_settlement.get("redeem_target_to"),
                    "redeem_calldata_hex": existing_settlement.get("redeem_calldata_hex"),
                    "redeemed_usdc_amount": 0.0,
                    "newly_redeemed_usdc_amount": 0.0,
                    "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                    "error": "REDEEM_SUBMIT_UNKNOWN_AWAITING_RECONCILIATION",
                    "redeemed_order_keys": sorted(already_redeemed_keys),
                    "newly_redeemed_order_keys": [],
                    "pending_batch_order_keys": bound_keys,
                    "pending_batch_shares": str(bound_payout),
                    "pending_batch_payout_usdc": float(bound_payout),
                    "has_remaining_unredeemed": any(
                        k not in set(bound_keys) for (k, _, _) in unredeemed_orders
                    ),
                }

        expected_payout = (unredeemed_winning_shares * Decimal("1.0")).quantize(USD_MICRO, rounding=ROUND_DOWN)
        batch_keys = [okey for (okey, _, _) in unredeemed_orders]
        if expected_payout <= Decimal("0"):
            if has_in_flight:
                return {
                    "live_expected_payout_usdc": 0.0,
                    "redeem_tx_hash": None,
                    "redeem_status": "AWAITING_ORDER_RECONCILIATION",
                    "redeemed_usdc_amount": 0.0,
                    "newly_redeemed_usdc_amount": 0.0,
                    "redeem_latency_seconds": 0.0,
                    "redeemed_order_keys": sorted(already_redeemed_keys),
                    "newly_redeemed_order_keys": [],
                    "pending_batch_order_keys": [],
                    "pending_batch_payout_usdc": 0.0,
                    "has_remaining_unredeemed": False,
                }
            if already_redeemed_shares > Decimal("0"):
                already_payout = (already_redeemed_shares * Decimal("1.0")).quantize(USD_MICRO, rounding=ROUND_DOWN)
                return {
                    "live_expected_payout_usdc": float(already_payout),
                    "redeem_tx_hash": existing_settlement.get("redeem_tx_hash") if isinstance(existing_settlement, Mapping) else None,
                    "redeem_status": "REDEEMED",
                    "redeemed_usdc_amount": 0.0,
                    "newly_redeemed_usdc_amount": 0.0,
                    "redeem_latency_seconds": 0.0,
                    "redeemed_order_keys": sorted(already_redeemed_keys),
                    "newly_redeemed_order_keys": [],
                    "pending_batch_order_keys": [],
                    "pending_batch_payout_usdc": 0.0,
                    "has_remaining_unredeemed": False,
                }
            return {
                "live_expected_payout_usdc": 0.0,
                "redeem_tx_hash": None,
                "redeem_status": "LOST_NO_REDEEM",
                "redeemed_usdc_amount": 0.0,
                "newly_redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": 0.0,
                "redeemed_order_keys": [],
                "newly_redeemed_order_keys": [],
                "pending_batch_order_keys": [],
                "pending_batch_payout_usdc": 0.0,
                "has_remaining_unredeemed": False,
            }

        cond_clean = str(condition_id or "").strip()
        order_protocols = {str(o.get("market_protocol") or "").strip().lower() for o in live_orders}
        has_v2_token = any(is_v2_position_id(str(o.get("token_id") or "").strip()) for o in live_orders)
        is_v2 = (
            str(market_protocol or "").strip().lower() == "v2"
            or "v2" in order_protocols
            or has_v2_token
            or len(cond_clean.removeprefix("0x").removeprefix("0X")) == 62
        )
        effective_neg_risk = bool(neg_risk or any(bool(o.get("neg_risk")) for o in live_orders))

        if is_v2:
            resolved_idx = (
                int(outcome_index)
                if outcome_index is not None
                else (0 if winner in ("UP", "YES", "0") else 1)
            )
            amount_atoms = int((unredeemed_winning_shares * Decimal("1000000")).to_integral_value(rounding=ROUND_DOWN))
            target_to = self.v2_router_address
            calldata_hex = encode_v2_router_redeem_calldata(
                condition_id=cond_clean,
                outcome_index=resolved_idx,
                amount_base_units=amount_atoms,
                router_address=target_to,
            )
        else:
            target_to = self.neg_risk_adapter_address if effective_neg_risk else self.ctf_address
            calldata_hex = encode_redeem_positions_calldata(
                condition_id=cond_clean,
                collateral_address=self.pusd_address,
                index_sets=[1, 2],
                target_contract_address=target_to,
            )

        assert_redeem_only_transaction(
            to_address=target_to,
            calldata_hex=calldata_hex,
            value=0,
            expected_ctf_address=target_to,
            expected_collateral_address=self.pusd_address,
        )

        if self.mode == "shadow":
            latency = round(time.perf_counter() - t0, 4)
            shadow_tx = "0xshadow_redeem_" + hashlib.sha256(
                f"{market_id}:{condition_id}:{winner}:{expected_payout}".encode()
            ).hexdigest()[:50]
            updated_redeemed_keys = set(already_redeemed_keys) | set(batch_keys)
            return {
                "live_expected_payout_usdc": float(expected_payout),
                "redeem_tx_hash": shadow_tx,
                "redeem_status": "REDEEMED",
                "redeemed_usdc_amount": float(expected_payout),
                "newly_redeemed_usdc_amount": float(expected_payout),
                "redeem_latency_seconds": latency,
                "redeemed_order_keys": sorted(updated_redeemed_keys),
                "newly_redeemed_order_keys": batch_keys,
                "pending_batch_order_keys": [],
                "pending_batch_payout_usdc": 0.0,
                "has_remaining_unredeemed": False,
            }

        if self.relayer_client is None:
            return {
                "live_expected_payout_usdc": float(expected_payout),
                "redeem_tx_hash": None,
                "redeem_status": "REDEEM_PENDING",
                "redeemed_usdc_amount": 0.0,
                "newly_redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                "error": "RELAYER_CREDENTIALS_NOT_CONFIGURED",
                "redeemed_order_keys": sorted(already_redeemed_keys),
                "newly_redeemed_order_keys": [],
                "pending_batch_order_keys": [],
                "pending_batch_payout_usdc": 0.0,
                "has_remaining_unredeemed": True,
            }

        batch_sig = hashlib.sha256(
            f"{market_id}:{condition_id}:{winner}:{','.join(sorted(batch_keys))}:{expected_payout}".encode()
        ).hexdigest()[:16]
        redeem_request_id = f"argon4_auto_redeem_{market_id}_{batch_sig}"

        # [P1-R4-3] & [P1-R5-3]: Step 1 — Persist the pending batch, `submit_state = "REDEEM_SUBMIT_UNKNOWN"`,
        # and queryable `redeem_request_id` BEFORE calling `relayer_client.execute(...)`.
        if callable(on_progress_callback):
            on_progress_callback(
                {
                    "live_expected_payout_usdc": float(expected_payout),
                    "redeem_tx_id": None,
                    "redeem_tx_hash": None,
                    "redeem_status": "REDEEM_PENDING",
                    "submit_state": "REDEEM_SUBMIT_UNKNOWN",
                    "redeem_request_id": redeem_request_id,
                    "redeem_target_to": str(target_to),
                    "redeem_calldata_hex": str(calldata_hex),
                    "redeemed_usdc_amount": 0.0,
                    "newly_redeemed_usdc_amount": 0.0,
                    "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                    "redeemed_order_keys": sorted(already_redeemed_keys),
                    "newly_redeemed_order_keys": [],
                    "pending_batch_order_keys": batch_keys,
                    "pending_batch_shares": str(unredeemed_winning_shares),
                    "pending_batch_payout_usdc": float(expected_payout),
                    "has_remaining_unredeemed": False,
                }
            )

        try:
            txn = TransactionCall(to=EvmAddress(target_to), data=HexString(calldata_hex), value=0)
            relayer_resp = self.relayer_client.execute(
                transactions=[txn],
                metadata=redeem_request_id,
            )
        except Exception as exc:
            exc_lower = str(exc).lower()
            resp_code = getattr(getattr(exc, "response", None), "status_code", None)
            is_http_400_rejected = (
                resp_code in (400, 422)
                or "400 client error" in exc_lower
                or "422 client error" in exc_lower
            )
            if is_http_400_rejected:
                return {
                    "live_expected_payout_usdc": float(expected_payout),
                    "redeem_tx_id": None,
                    "redeem_tx_hash": None,
                    "redeem_status": "REDEEM_PENDING",
                    "submit_state": "SUBMIT_HTTP_400_REJECTED",
                    "redeem_request_id": None,
                    "redeem_target_to": str(target_to),
                    "redeem_calldata_hex": str(calldata_hex),
                    "redeemed_usdc_amount": 0.0,
                    "newly_redeemed_usdc_amount": 0.0,
                    "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                    "error": f"RELAYER_HTTP_400_REJECTED:{type(exc).__name__}:{str(exc)[:140]}",
                    "redeemed_order_keys": sorted(already_redeemed_keys),
                    "newly_redeemed_order_keys": [],
                    "pending_batch_order_keys": [],
                    "pending_batch_shares": "0.00",
                    "pending_batch_payout_usdc": 0.0,
                    "has_remaining_unredeemed": True,
                }
            # [P1-R5-3]: Submission outcome is unknown (e.g. server accepted request before timeout).
            # Retain `submit_state = "REDEEM_SUBMIT_UNKNOWN"`, `redeem_request_id`, and `pending_batch_order_keys`
            # so subsequent passes enter reconciliation instead of re-submitting!
            return {
                "live_expected_payout_usdc": float(expected_payout),
                "redeem_tx_id": None,
                "redeem_tx_hash": None,
                "redeem_status": "REDEEM_PENDING",
                "submit_state": "REDEEM_SUBMIT_UNKNOWN",
                "redeem_request_id": redeem_request_id,
                "redeem_target_to": str(target_to),
                "redeem_calldata_hex": str(calldata_hex),
                "redeemed_usdc_amount": 0.0,
                "newly_redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                "error": f"REDEEM_SUBMIT_UNKNOWN:{type(exc).__name__}:{str(exc)[:140]}",
                "redeemed_order_keys": sorted(already_redeemed_keys),
                "newly_redeemed_order_keys": [],
                "pending_batch_order_keys": batch_keys,
                "pending_batch_shares": str(unredeemed_winning_shares),
                "pending_batch_payout_usdc": float(expected_payout),
                "has_remaining_unredeemed": False,
            }

        tx_hash = getattr(relayer_resp, "transaction_hash", None)
        tx_id = getattr(relayer_resp, "transaction_id", None)
        polled_state = str(getattr(relayer_resp, "state", "") or "").upper()
        submitted_tx_id = str(tx_id) if tx_id else None
        submitted_tx_hash = (
            str(tx_hash)
            if tx_hash
            else (f"relayer_id:{submitted_tx_id}" if submitted_tx_id else None)
        )
        tx_submitted_at_iso = datetime.now(timezone.utc).isoformat()

        # [P1-R4-3] & [P1-R5-3]: Step 2 — Immediately persist `redeem_tx_id` / `redeem_tx_hash` and bound batch
        # to SQLite BEFORE polling for confirmation (`_finalize_or_poll_redemption`), so a crash or
        # power loss during polling never loses the transaction ID or causes duplicate submission!
        if callable(on_progress_callback):
            on_progress_callback(
                {
                    "live_expected_payout_usdc": float(expected_payout),
                    "redeem_tx_id": submitted_tx_id,
                    "redeem_tx_hash": submitted_tx_hash,
                    "redeem_status": "REDEEM_PENDING",
                    "submit_state": "SUBMITTED" if (submitted_tx_id or submitted_tx_hash) else "REDEEM_SUBMIT_UNKNOWN",
                    "redeem_request_id": redeem_request_id,
                    "redeem_target_to": str(target_to),
                    "redeem_calldata_hex": str(calldata_hex),
                    "redeemed_usdc_amount": 0.0,
                    "newly_redeemed_usdc_amount": 0.0,
                    "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                    "redeemed_order_keys": sorted(already_redeemed_keys),
                    "newly_redeemed_order_keys": [],
                    "pending_batch_order_keys": batch_keys,
                    "pending_batch_shares": str(unredeemed_winning_shares),
                    "pending_batch_payout_usdc": float(expected_payout),
                    "has_remaining_unredeemed": False,
                    "tx_submitted_at_utc": tx_submitted_at_iso,
                }
            )

        try:
            res = self._finalize_or_poll_redemption(
                tx_id=submitted_tx_id,
                tx_hash=str(tx_hash) if tx_hash else None,
                polled_state=polled_state,
                expected_payout=expected_payout,
                t0=t0,
            )
        except Exception as exc:
            res = {
                "live_expected_payout_usdc": float(expected_payout),
                "redeem_tx_id": submitted_tx_id,
                "redeem_tx_hash": submitted_tx_hash,
                "redeem_status": "REDEEM_PENDING",
                "redeemed_usdc_amount": 0.0,
                "redeem_latency_seconds": round(time.perf_counter() - t0, 4),
                "error": f"{type(exc).__name__}:{str(exc)[:160]}",
            }

        res["tx_submitted_at_utc"] = tx_submitted_at_iso
        res["redeem_request_id"] = redeem_request_id
        res["redeem_target_to"] = str(target_to)
        res["redeem_calldata_hex"] = str(calldata_hex)
        if res["redeem_status"] == "REDEEMED":
            updated_redeemed_keys = set(already_redeemed_keys) | set(batch_keys)
            res["newly_redeemed_usdc_amount"] = float(expected_payout)
            res["redeemed_order_keys"] = sorted(updated_redeemed_keys)
            res["newly_redeemed_order_keys"] = batch_keys
            res["pending_batch_order_keys"] = []
            res["pending_batch_shares"] = "0.00"
            res["pending_batch_payout_usdc"] = 0.0
            res["submit_state"] = "CONFIRMED"
            res["has_remaining_unredeemed"] = False
        elif res["redeem_status"] == "REDEEM_PENDING":
            res["newly_redeemed_usdc_amount"] = 0.0
            res["redeemed_order_keys"] = sorted(already_redeemed_keys)
            res["newly_redeemed_order_keys"] = []
            res["pending_batch_order_keys"] = batch_keys
            res["pending_batch_shares"] = str(unredeemed_winning_shares)
            res["pending_batch_payout_usdc"] = float(expected_payout)
            res["submit_state"] = "SUBMITTED" if (submitted_tx_id or submitted_tx_hash) else "REDEEM_SUBMIT_UNKNOWN"
            res["has_remaining_unredeemed"] = False
        else:
            res["newly_redeemed_usdc_amount"] = 0.0
            res["redeemed_order_keys"] = sorted(already_redeemed_keys)
            res["newly_redeemed_order_keys"] = []
            res["pending_batch_order_keys"] = []
            res["pending_batch_shares"] = "0.00"
            res["pending_batch_payout_usdc"] = 0.0
            res["submit_state"] = "FAILED"
            res["has_remaining_unredeemed"] = True
        return res
