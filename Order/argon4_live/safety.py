"""Physical isolation, order-type, single-instance lock, and non-withdrawal safety guards.

Enforces the core invariants from the ARGON4 External Live Executor Spec (2026 Polymarket V2):
1. NEVER open or lock upstream's account.sqlite3 (or any file write in upstream).
2. NEVER submit resting orders (GTC/GTD/post_only); strictly lock to OrderType.FAK.
3. NEVER allow external withdrawals or token transfers; only:
   - CTF / Collateral Adapter `redeemPositions(pUSD, bytes32(0), conditionId, [1,2])` (`0x01b7037c`)
     on `CtfCollateralAdapter` (`0xAdA100Db00Ca00073811820692005400218FcE1f`),
     `NegRiskCtfCollateralAdapter` (`0xadA2005600Dec949baf300f4C6120000bDB6eAab`), or
     `ConditionalTokens` (`0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`), OR
   - Protocol V2 `Router.redeem(bytes31,uint256,uint256)` (`0xd217a3cc`) on
     `Router` (`0x12121212006e4CD160D18e3f00711DA5c3372600`)
   are permitted.
4. Enforce an OS-level exclusive single-instance process lock per runtime directory.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from eth_abi import decode as abi_decode

# Official 2026 Polymarket V2 & CTF Polygon Mainnet (Chain 137) Contract Addresses
OFFICIAL_POLYGON_PUSD_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS = "0x12121212006e4CD160D18e3f00711DA5c3372600"
OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS = "0xAdA100Db00Ca00073811820692005400218FcE1f"
OFFICIAL_NEG_RISK_COLLATERAL_ADAPTER_ADDRESS = "0xadA2005600Dec949baf300f4C6120000bDB6eAab"
OFFICIAL_POLYGON_CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
OFFICIAL_V2_POSITION_MANAGER_ADDRESS = "0x006F54F7f9A22e0000CC2AB60031000000ae9fEF"
OFFICIAL_EXCHANGE_V3_ADDRESS = "0xe3333700cA9d93003F00f0F71f8515005F6c00Aa"
OFFICIAL_STANDARD_EXCHANGE_V2_ADDRESS = "0xE111180000d2663C0091e4f400237545B87B996B"
OFFICIAL_NEG_RISK_EXCHANGE_V2_ADDRESS = "0xe2222d279d744050d28e00520010520000310F59"

# Backward-compatible alias pointing to pUSD (post-April 28, 2026 collateral cutover)
OFFICIAL_POLYGON_USDCE_ADDRESS = OFFICIAL_POLYGON_PUSD_ADDRESS

REDEEM_POSITIONS_SELECTOR = "0x01b7037c"  # redeemPositions(address,bytes32,bytes32,uint256[])
ROUTER_REDEEM_SELECTOR = "0xd217a3cc"  # redeem(bytes31,uint256,uint256)

ALLOWED_CTF_REDEEM_TARGETS = frozenset(
    {
        OFFICIAL_CTF_COLLATERAL_ADAPTER_ADDRESS.lower(),
        OFFICIAL_NEG_RISK_COLLATERAL_ADAPTER_ADDRESS.lower(),
        OFFICIAL_POLYGON_CTF_ADDRESS.lower(),
    }
)

# Explicitly banned ERC-20 / ERC-1155 / Proxy transfer & withdrawal selectors
FORBIDDEN_SELECTORS = {
    "0xa9059cbb": "ERC20.transfer",
    "0x23b872dd": "ERC20.transferFrom",
    "0x095ea7b3": "ERC20.approve",
    "0xf242432a": "ERC1155.safeTransferFrom",
    "0x2eb2c2d6": "ERC1155.safeBatchTransferFrom",
    "0xa22cb465": "ERC1155.setApprovalForAll",
    "0x2e1a7d4d": "withdraw(uint256)",
    "0x51cff8d9": "withdraw(address)",
    "0x3ccfd60b": "withdraw()",
}


class SafetyViolationError(RuntimeError):
    """Raised when any physical isolation, order-type, single-instance, or withdrawal rule is violated."""


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def assert_safe_read_path(
    target_path: str | Path,
    upstream_root: str | Path = r"C:\Users\liuqi\crypto-predict_system",
) -> Path:
    """Ensure read operations never touch upstream SQLite databases (`account.sqlite3`, etc.)."""
    resolved = _normalize_path(target_path)
    up_root = _normalize_path(upstream_root)
    name_lower = resolved.name.lower()

    # Hard block on any upstream SQLite database or journal/wal/shm file
    if resolved == up_root or up_root in resolved.parents:
        if (
            "sqlite" in name_lower
            or name_lower.endswith((".db", ".sqlite", ".sqlite3", "-journal", "-wal", "-shm"))
            or "account.sqlite3" in str(resolved).lower()
        ):
            raise SafetyViolationError(
                f"CRITICAL ISOLATION VIOLATION: Direct read of upstream SQLite file '{resolved}' "
                "is strictly prohibited to prevent database lock contention."
            )
    return resolved


def assert_safe_write_path(
    target_path: str | Path,
    local_root: str | Path = r"C:\Users\liuqi\Order",
    upstream_root: str | Path = r"C:\Users\liuqi\crypto-predict_system",
) -> Path:
    """Ensure all file/DB writes occur strictly inside `c:\\Users\\liuqi\\Order` and never in upstream."""
    resolved = _normalize_path(target_path)
    loc_root = _normalize_path(local_root)
    up_root = _normalize_path(upstream_root)

    if resolved == up_root or up_root in resolved.parents:
        raise SafetyViolationError(
            f"CRITICAL ISOLATION VIOLATION: Writing to upstream repository '{resolved}' is forbidden."
        )
    if resolved != loc_root and loc_root not in resolved.parents:
        raise SafetyViolationError(
            f"SAFETY VIOLATION: Write target '{resolved}' is outside allowed workspace '{loc_root}'."
        )
    return resolved


def assert_fak_order_only(order_type: Any, post_only: bool = False) -> str:
    """Strictly lock order execution to Fill-And-Kill (FAK). Prohibit GTC / GTD / FOK / post_only."""
    if post_only is True:
        raise SafetyViolationError("SAFETY VIOLATION: post_only=True (resting maker order) is strictly prohibited.")
    normalized = str(getattr(order_type, "value", order_type)).upper().strip()
    if normalized.endswith(".FAK"):
        normalized = "FAK"
    if normalized != "FAK":
        raise SafetyViolationError(
            f"SAFETY VIOLATION: OrderType '{order_type}' is forbidden. Only OrderType.FAK is permitted."
        )
    return "FAK"


def assert_no_withdrawal_permission(config_dict: dict[str, Any]) -> None:
    """Ensure `withdrawal_permission` is explicitly False in configuration."""
    redemption_cfg = config_dict.get("redemption", {}) if isinstance(config_dict, dict) else {}
    if redemption_cfg.get("withdrawal_permission") is not False and config_dict.get("withdrawal_permission") is not False:
        raise SafetyViolationError(
            "SAFETY VIOLATION: withdrawal_permission must be explicitly set to false."
        )


def assert_redeem_only_transaction(
    to_address: str,
    calldata_hex: str,
    value: int | str = 0,
    expected_ctf_address: str | None = None,
    expected_collateral_address: str = OFFICIAL_POLYGON_PUSD_ADDRESS,
) -> None:
    """Verify a relayer/contract transaction is strictly an official Polymarket redemption call.

    Allowed targets & selectors:
      1. `redeemPositions(address,bytes32,bytes32,uint256[])` (`0x01b7037c`) on
         `CtfCollateralAdapter` (`0xAdA100Db00Ca00073811820692005400218FcE1f`),
         `NegRiskCtfCollateralAdapter` (`0xadA2005600Dec949baf300f4C6120000bDB6eAab`),
         or `ConditionalTokens` (`0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`), with
         `collateralToken == pUSD` (`0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB`).
      2. `redeem(bytes31,uint256,uint256)` (`0xd217a3cc`) on Protocol V2 `Router`
         (`0x12121212006e4CD160D18e3f00711DA5c3372600`).

    Never allows native token transfers (`value != 0`), ERC-20/ERC-1155 transfers, or calls to arbitrary addresses.
    """
    if str(value).strip() not in ("0", "0x0", ""):
        raise SafetyViolationError(f"SAFETY VIOLATION: Non-zero transaction value ({value}) is forbidden.")

    if not isinstance(to_address, str) or not to_address.strip().startswith("0x") or len(to_address.strip()) != 42:
        raise SafetyViolationError(f"SAFETY VIOLATION: Invalid target contract address '{to_address}'.")
    target_lower = to_address.strip().lower()

    if not isinstance(calldata_hex, str) or not calldata_hex.startswith("0x") or len(calldata_hex) < 10:
        raise SafetyViolationError("SAFETY VIOLATION: Invalid transaction calldata.")

    selector = calldata_hex[:10].lower()
    if selector in FORBIDDEN_SELECTORS:
        raise SafetyViolationError(
            f"CRITICAL SAFETY VIOLATION: Forbidden transfer/withdrawal selector {selector} "
            f"({FORBIDDEN_SELECTORS[selector]}) blocked!"
        )

    raw_args = bytes.fromhex(calldata_hex[10:])

    # Route 1: Protocol V2 Router `redeem(bytes31,uint256,uint256)` (`0xd217a3cc`)
    if selector == ROUTER_REDEEM_SELECTOR:
        expected_router = (
            expected_ctf_address.strip().lower()
            if expected_ctf_address and expected_ctf_address.strip().lower() == OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS.lower()
            else OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS.lower()
        )
        if target_lower != expected_router:
            raise SafetyViolationError(
                f"SAFETY VIOLATION: Router.redeem target '{to_address}' does not match official Protocol V2 Router "
                f"'{OFFICIAL_PROTOCOL_V2_ROUTER_ADDRESS}'."
            )
        try:
            cond_bytes, outcome_idx, amount = abi_decode(["bytes31", "uint256", "uint256"], raw_args)
        except Exception as exc:
            raise SafetyViolationError(f"SAFETY VIOLATION: Malformed Router.redeem calldata: {exc}") from exc
        if len(cond_bytes) != 31 or outcome_idx not in (0, 1) or amount <= 0:
            raise SafetyViolationError(
                f"SAFETY VIOLATION: Invalid Router.redeem parameters (outcome_idx={outcome_idx}, amount={amount})."
            )
        return

    # Route 2: CTF / Collateral Adapter `redeemPositions(address,bytes32,bytes32,uint256[])` (`0x01b7037c`)
    if selector == REDEEM_POSITIONS_SELECTOR:
        if expected_ctf_address is not None:
            exp_lower = expected_ctf_address.strip().lower()
            if exp_lower not in ALLOWED_CTF_REDEEM_TARGETS:
                raise SafetyViolationError(
                    f"SAFETY VIOLATION: Configured target '{expected_ctf_address}' is not an official CTF/Adapter contract."
                )
            if target_lower != exp_lower:
                raise SafetyViolationError(
                    f"SAFETY VIOLATION: Transaction target '{to_address}' does not match official CTF contract '{expected_ctf_address}'."
                )
        elif target_lower not in ALLOWED_CTF_REDEEM_TARGETS:
            raise SafetyViolationError(
                f"SAFETY VIOLATION: Transaction target '{to_address}' does not match official CTF/Adapter contracts."
            )

        try:
            collateral, parent_coll, cond_bytes, index_sets = abi_decode(
                ["address", "bytes32", "bytes32", "uint256[]"], raw_args
            )
        except Exception as exc:
            raise SafetyViolationError(f"SAFETY VIOLATION: Malformed CTF.redeemPositions calldata: {exc}") from exc

        if str(collateral).lower() != OFFICIAL_POLYGON_PUSD_ADDRESS.lower() or (
            expected_collateral_address and str(collateral).lower() != expected_collateral_address.strip().lower()
        ):
            raise SafetyViolationError(
                f"SAFETY VIOLATION: redeemPositions collateralToken '{collateral}' does not match official pUSD "
                f"'{OFFICIAL_POLYGON_PUSD_ADDRESS}'."
            )
        if parent_coll != b"\x00" * 32 or len(cond_bytes) != 32 or not index_sets:
            raise SafetyViolationError("SAFETY VIOLATION: Invalid redeemPositions parentCollectionId or indexSets.")
        return

    raise SafetyViolationError(
        f"SAFETY VIOLATION: Only CTF/Adapter.redeemPositions ({REDEEM_POSITIONS_SELECTOR}) or "
        f"Router.redeem ({ROUTER_REDEEM_SELECTOR}) is allowed; got selector {selector}."
    )


class SingleInstanceFileLock:
    """OS-level exclusive non-blocking process lock to prevent multiple executor instances (`[P1-6]`)."""

    def __init__(self, lock_path: str | Path, local_root: str | Path = r"C:\Users\liuqi\Order") -> None:
        self.lock_path = assert_safe_write_path(lock_path, local_root=local_root)
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                # Ensure at least 1 byte exists for msvcrt.locking
                os.write(fd, b"L")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            payload = f"pid={os.getpid()}\n".encode("ascii")
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
            self._fd = fd
        except OSError as exc:
            os.close(fd)
            raise SafetyViolationError(
                f"SINGLE_INSTANCE_LOCK_HELD: Another ARGON4 Live Executor instance is already holding "
                f"'{self.lock_path}'."
            ) from exc

    def release(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
