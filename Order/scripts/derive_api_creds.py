"""Inspect and pre-derive Polymarket MetaMask Safe (`funder_address`, `signature_type=2`) and
L2 CLOB API credentials (`api_key`, `api_secret`, `api_passphrase`) directly from `POLY_PRIVATE_KEY` in `.env`.

Note: `ClobExecutor` already performs this derivation automatically on startup and caches
the derived L2 credentials in `runtime/.derived_clob_creds.json`, so your `.env` only needs 3 items!

Usage:
    c:\\Users\\liuqi\\Order\\.venv\\Scripts\\python.exe scripts/derive_api_creds.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from py_clob_client_v2.client import ClobClient

from argon4_live.clob_executor import derive_metamask_polymarket_addresses


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect auto-derived Polymarket addresses and L2 API credentials")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env", help="Path to .env file")
    args = parser.parse_args()

    env_path = args.env_file.resolve()
    if env_path.is_file():
        load_dotenv(env_path, override=True)

    pk = (os.getenv("POLY_PRIVATE_KEY") or "").strip()
    if not pk or len(pk.removeprefix("0x")) != 64 or "YOUR_" in pk.upper():
        print("[ERROR] Please set your 64-hex MetaMask private key in c:\\Users\\liuqi\\Order\\.env (`POLY_PRIVATE_KEY`).")
        print("        请先在 c:\\Users\\liuqi\\Order\\.env 中填入您的 64 位 MetaMask 私钥 (`POLY_PRIVATE_KEY`)。")
        return 1

    eoa_addr, safe_funder = derive_metamask_polymarket_addresses(pk)
    print(f"MetaMask Signer EOA Address / MetaMask 签名钱包地址:          {eoa_addr}")
    print(f"Polymarket Gnosis Safe Proxy (`funder_address`, sig_type=2): {safe_funder}")

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=pk,
        signature_type=2,
        funder=safe_funder,
    )

    print("\nAuto-deriving L2 CLOB API Credentials from Polymarket CLOB...")
    creds = client.create_or_derive_api_key()
    cache_path = ROOT / "runtime" / ".derived_clob_creds.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "signer_address": eoa_addr,
                "funder_address": safe_funder,
                "signature_type": 2,
                "api_key": creds.api_key,
                "api_secret": creds.api_secret,
                "api_passphrase": creds.api_passphrase,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Successfully derived and cached L2 credentials to {cache_path}:")
    print(f"  api_key:        {creds.api_key}")
    print(f"  api_secret:     {creds.api_secret[:8]}... (cached)")
    print(f"  api_passphrase: {creds.api_passphrase[:8]}... (cached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
