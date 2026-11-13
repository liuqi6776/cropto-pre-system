"""Tests for ARGON-4 100GB three-tier storage remediation, keyframe extraction, and cap enforcement."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from test_argon4_forward_revisions import NOW, _add_receipt, _error, no_network, runtime
from tools.argon4_forward import revisions
from tools.argon4_forward.storage_janitor import StorageJanitor, _physical_dir_size_bytes


def test_tiered_storage_strip_sqlite_keyframes_compression_and_cold_cap(runtime, tmp_path, no_network):
    """Verify:
    1. runtime/storage_override.json configures 2-day retention, strip_revision_sqlite, compression, and cold cap;
    2. permanent_core backs up account.sqlite3 & non-sensitive live_order_ledgers (blocking .env/secret/key files)
       and extracts [-60s, +300s] trade keyframes before hot SQLite pruning;
    3. strip_revision_sqlite strips redundant SQLite snapshots from aged revisions while keeping manifest/diffs;
    4. _enforce_cold_cap prunes oldest Tier-2 rolling files under max_cold_total_gb without touching permanent_core/.
    """
    # Create active revision in runtime
    _add_receipt(runtime)
    _add_receipt(runtime, "2026-11-06T01:40:05Z")
    res1 = revisions.reconcile_confirmed_price_revisions(runtime, [_error()], NOW)
    assert res1["status"] == "COMMITTED"

    # Create an aged unpinned revision (2026-11-01) containing redundant SQLite & proposed diff
    revs_dir = runtime / "features" / "revisions"
    old_id = "20261101T000000000000Z_bloatrev0001"
    old_dir = revs_dir / old_id
    (old_dir / "raw_data").mkdir(parents=True, exist_ok=True)
    (old_dir / "proposed_raw_data" / "intraday").mkdir(parents=True, exist_ok=True)
    (old_dir / "manifest.json").write_text(
        json.dumps({"status": "COMMITTED", "revision_id": old_id, "view_effective_at": "2026-11-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (old_dir / "prepared.json").write_text(
        json.dumps({"validation_at": "2026-11-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (old_dir / "observations.sqlite3").write_bytes(b"R" * (512 * 1024))
    (old_dir / "observations_delta.sqlite3").write_bytes(b"D" * (256 * 1024))
    (old_dir / "receipts.sqlite3").write_bytes(b"C" * (128 * 1024))
    (old_dir / "raw_data" / "BTCUSDT_5m.parquet").write_bytes(b"U" * (64 * 1024))
    diff_payload = b"PROPOSED_PARQUET_DIFF_BYTES"
    (old_dir / "proposed_raw_data" / "intraday" / "BNBUSDT_5m.parquet").write_bytes(diff_payload)

    # Create account.sqlite3 with a trade at 2026-11-02T10:00:12+00:00
    acc_path = runtime / "account.sqlite3"
    with closing(sqlite3.connect(acc_path)) as con, con:
        con.execute("CREATE TABLE orders(order_id TEXT PRIMARY KEY, fill_at TEXT, payload TEXT)")
        con.execute(
            "INSERT INTO orders(order_id, fill_at, payload) VALUES(?, ?, ?)",
            ("ord-1", "2026-11-02T10:00:12+00:00", '{"pnl": 12.5}'),
        )

    # Create observations.sqlite3 & book_telemetry.sqlite3 with trade-window and idle aged rows
    obs_path = runtime / "observations.sqlite3"
    with closing(sqlite3.connect(obs_path)) as con, con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY, kind TEXT, saved_at TEXT, payload TEXT)"
        )
        con.execute(
            "INSERT INTO snapshots(kind, saved_at, payload) VALUES(?, ?, ?)",
            ("prediction_decisions", "2026-11-02T10:00:05+00:00", '{"BTC_15m": {"z_v4": 2.1}}'),
        )
        con.execute(
            "INSERT INTO snapshots(kind, saved_at, payload) VALUES(?, ?, ?)",
            ("polymarket", "2026-11-02T05:00:00+00:00", "idle_poll_" + ("z" * 4096)),
        )

    tel_path = runtime / "book_telemetry.sqlite3"
    with closing(sqlite3.connect(tel_path)) as con, con:
        con.execute("CREATE TABLE samples(id INTEGER PRIMARY KEY, target_at TEXT, stream TEXT, ask REAL)")
        con.execute(
            "INSERT INTO samples(target_at, stream, ask) VALUES(?, ?, ?)",
            ("2026-11-02T10:01:00+00:00", "BTC_15m", 0.49),  # Inside [-60s, +300s] of ord-1!
        )
        con.execute(
            "INSERT INTO samples(target_at, stream, ask) VALUES(?, ?, ?)",
            ("2026-11-02T03:00:00+00:00", "BTC_15m", 0.52),  # Far outside trade window
        )

    # Create external Order runtime with both ledger and sensitive files
    ext_order = tmp_path / "Order" / "runtime"
    ext_order.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(ext_order / "live_orders.sqlite3")) as con, con:
        con.execute("CREATE TABLE fills(id INT, tx TEXT)")
        con.execute("INSERT INTO fills VALUES(1, '0xabc')")
    (ext_order / ".env").write_text("PRIVATE_KEY=NEVER_TOUCH", encoding="utf-8")
    (ext_order / "wallet_secret.json").write_text("{}", encoding="utf-8")
    (ext_order / "execution_audit.jsonl").write_text('{"fill":1}\n', encoding="utf-8")

    import os
    cold_root = tmp_path / "cloud_archive"
    # Pre-seed an old bulky file in cold_root/sqlite to test max_cold_total_gb FIFO cap
    (cold_root / "sqlite").mkdir(parents=True, exist_ok=True)
    old_bulky = cold_root / "sqlite" / "observations_archive_20260101_000000.sqlite3.gz"
    old_bulky.write_bytes(b"B" * (2 * 1024 * 1024))  # 2 MB
    os.utime(old_bulky, (1767225600, 1767225600))

    (runtime / "storage_override.json").write_text(
        json.dumps(
            {
                "scan_interval_days": 2,
                "retain_hot_days": 2,
                "cold_archive_root": str(cold_root),
                "max_hot_total_gb": 0.0001,  # Trigger HOT_TOTAL_SIZE_EXCEEDED check
                "strip_revision_sqlite": True,
                "compress_cold_files": True,
                "extract_trade_keyframes": True,
                "always_prune_aged_observations": True,
                "max_cold_total_gb": 0.001,  # ~1 MB cap so 2 MB old_bulky gets FIFO-pruned!
                "external_order_runtime": str(ext_order),
            }
        ),
        encoding="utf-8",
    )

    janitor = StorageJanitor(runtime)
    due, reason = janitor.should_run(now="2026-11-06T02:00:00+00:00")
    assert due is True
    assert reason == "HOT_TOTAL_SIZE_EXCEEDED"

    maint = janitor.run_maintenance(now="2026-11-06T02:00:00+00:00")
    assert maint["status"] == "COMPLETED"

    # 1. Stripped revision verified: SQLite copies & unchanged raw_data omitted; manifest & proposed diff kept
    migrated_rev = cold_root / "revisions" / "2026-11" / old_id
    assert migrated_rev.is_dir()
    assert (migrated_rev / "manifest.json").is_file()
    assert (migrated_rev / "proposed_raw_data" / "intraday" / "BNBUSDT_5m.parquet").read_bytes() == diff_payload
    assert not (migrated_rev / "observations.sqlite3").exists()
    assert not (migrated_rev / "observations_delta.sqlite3").exists()
    assert not (migrated_rev / "receipts.sqlite3").exists()
    assert not (migrated_rev / "raw_data").exists()
    assert maint["revisions"]["bytes_stripped"] >= (512 + 256 + 128 + 64) * 1024

    # 2. Permanent core intact & sensitive files excluded
    perm_dir = cold_root / "permanent_core"
    assert (perm_dir / "ledgers" / "account_latest.sqlite3").is_file()
    assert (perm_dir / "live_order_ledgers" / "live_orders.sqlite3").is_file()
    assert (perm_dir / "live_order_ledgers" / "execution_audit.jsonl").is_file()
    assert not list(perm_dir.rglob("*env*"))
    assert not list(perm_dir.rglob("*secret*"))

    # Keyframes extracted (1 prediction_decisions + 1 trade-window book sample = 2 records)
    assert maint["permanent_core"]["keyframes_extracted"] == 2
    kf_path = Path(maint["permanent_core"]["keyframe_archive"])
    assert kf_path.is_file()

    # 3. Cold cap enforced: 2MB old_bulky was FIFO-pruned, while permanent_core survived 100% intact!
    assert not old_bulky.exists()
    assert maint["cold_cap"]["pruned_files"] >= 1
    assert (perm_dir / "ledgers" / "account_latest.sqlite3").is_file()
    assert kf_path.is_file()

    # 4. Active revision remains 100% COMMITTED
    assert revisions.ensure_view_committed(runtime)["status"] == "COMMITTED"
    assert _physical_dir_size_bytes(perm_dir) > 0
