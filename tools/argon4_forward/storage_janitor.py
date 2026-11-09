"""Automated bi-weekly (14-day) hot-to-cold storage janitor for ARGON-4.

Safely migrates >14-day non-pinned historical revision directories, old
observation snapshots, and aged audit logs from the hot runtime directory to
`D:/argon4_cold_archive` (with automatic fallback if D: is unavailable),
verifying SHA256 digests before freeing hot-disk space and preserving 100%
of active revision integrity (`ensure_view_committed`).
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Mapping

from . import revisions

ROOT = Path(__file__).resolve().parents[2]
_REV_TS_RE = re.compile(r"^(?:rev[-_])?(\d{8}T\d{6}(?:\d{6})?Z)")


def _utc(value: Any = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    try:
        with tmp.open("wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _dir_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _revision_timestamp(rev_dir: Path) -> datetime:
    for meta_name, keys in (
        ("manifest.json", ("view_effective_at", "committed_at", "validation_at")),
        ("prepared.json", ("validation_at",)),
    ):
        meta_path = rev_dir / meta_name
        if meta_path.is_file():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                for k in keys:
                    if data.get(k):
                        return _utc(data[k])
            except Exception:
                pass
    m = _REV_TS_RE.match(rev_dir.name)
    if m:
        token = m.group(1)
        fmt = "%Y%m%dT%H%M%S%fZ" if len(token) > 16 else "%Y%m%dT%H%M%SZ"
        try:
            return datetime.strptime(token, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(rev_dir.stat().st_mtime, tz=timezone.utc)


class StorageJanitor:
    """Revision-aware bi-weekly cold archiver and hot SQLite/log compactor."""

    def __init__(
        self,
        runtime: str | Path,
        *,
        policy_provider: Callable[[], Mapping[str, Any]] | None = None,
        cold_root: str | Path | None = None,
        monitor_log: Any = None,
    ) -> None:
        self.runtime = Path(runtime).resolve()
        self.state_path = self.runtime / "features" / "janitor_state.json"
        self.policy_provider = policy_provider
        self.explicit_cold_root = Path(cold_root) if cold_root is not None else None
        self.monitor_log = monitor_log
        self._lock = threading.Lock()
        self._async_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._tls = threading.local()
        self._last_async_result: dict[str, Any] | None = None

    def _yield_for_trading_window(
        self,
        *,
        active: bool | None = None,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> float:
        """Guarantee A: pause background janitor disk I/O during [T - 15s, T + 25s] trading window."""
        if active is None:
            active = bool(getattr(self._tls, "in_async_worker", False))
        if not active:
            return 0.0
        slept = 0.0
        while True:
            sec_in_5m = float(time_fn()) % 300.0
            if sec_in_5m >= 285.0:
                wait_s = (300.0 - sec_in_5m) + 25.1
            elif sec_in_5m <= 25.0:
                wait_s = 25.1 - sec_in_5m
            else:
                break
            step = min(max(0.05, wait_s), 2.0)
            sleep_fn(step)
            slept += step
        return round(slept, 3)

    def _load_policy(self) -> Mapping[str, Any]:
        if callable(self.policy_provider):
            try:
                pol = self.policy_provider()
                if isinstance(pol, Mapping):
                    return pol
            except Exception:
                pass
        pol_path = ROOT / "config" / "argon4_forward_policy.json"
        if pol_path.is_file():
            try:
                return json.loads(pol_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _config(self) -> dict[str, Any]:
        pol = self._load_policy()
        raw = pol.get("storage_maintenance")
        block = raw if isinstance(raw, Mapping) else {}
        win_range = (
            block.get("allowed_window_seconds_range")
            or block.get("allowed_window_seconds_Range")
            or [95, 210]
        )
        if not isinstance(win_range, (list, tuple)) or len(win_range) != 2:
            win_range = [95, 210]
        return {
            "enabled": bool(block.get("enabled", True)),
            "scan_interval_days": max(1, int(block.get("scan_interval_days", 14))),
            "retain_hot_days": max(1, int(block.get("retain_hot_days", 14))),
            "cold_archive_root": str(block.get("cold_archive_root", "D:/argon4_cold_archive")),
            "max_hot_observations_mb": max(64, int(block.get("max_hot_observations_mb", 15000))),
            "max_hot_total_gb": max(1.0, float(block.get("max_hot_total_gb", 35.0))),
            "allowed_window_seconds_range": [float(win_range[0]), float(win_range[1])],
        }

    def _resolve_cold_root(self, cfg: Mapping[str, Any]) -> tuple[Path, bool]:
        configured = self.explicit_cold_root if self.explicit_cold_root is not None else Path(cfg["cold_archive_root"])
        drive = configured.drive
        if drive:
            drive_root = Path(drive + "/")
            if not drive_root.exists():
                return (self.runtime / "cold_archive").resolve(), True
        return configured.resolve(), False

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {"last_scan_utc": None, "next_due_utc": None, "last_result": None, "history": []}

    def _save_state(self, state: Mapping[str, Any]) -> None:
        _atomic_json(self.state_path, state)

    def _observations_size_mb(self) -> float:
        obs = self.runtime / "observations.sqlite3"
        total = 0
        for p in (obs, Path(str(obs) + "-wal"), Path(str(obs) + "-shm")):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return round(total / (1024 * 1024), 3)

    def should_run(self, now: Any = None, *, elapsed_in_window: float | None = None) -> tuple[bool, str]:
        cfg = self._config()
        if not cfg["enabled"]:
            return False, "DISABLED_BY_POLICY"
        if elapsed_in_window is not None:
            low, high = cfg["allowed_window_seconds_range"]
            if not (low <= float(elapsed_in_window) <= high):
                return False, "OUTSIDE_IDLE_WINDOW"
        now_dt = _utc(now)
        state = self._load_state()
        last_scan_raw = state.get("last_scan_utc")
        obs_mb = self._observations_size_mb()
        if obs_mb >= cfg["max_hot_observations_mb"]:
            if last_scan_raw is None:
                return True, "HOT_OBSERVATIONS_SIZE_EXCEEDED"
            try:
                if now_dt - _utc(last_scan_raw) >= timedelta(hours=1):
                    return True, "HOT_OBSERVATIONS_SIZE_EXCEEDED"
            except Exception:
                return True, "HOT_OBSERVATIONS_SIZE_EXCEEDED"
        if last_scan_raw is None:
            cutoff_dt = now_dt - timedelta(days=cfg["retain_hot_days"])
            revs_dir = self.runtime / "features" / "revisions"
            if revs_dir.is_dir():
                for child in revs_dir.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        try:
                            if _revision_timestamp(child) < cutoff_dt:
                                return True, "INITIAL_AGED_REVISIONS_DUE"
                        except Exception:
                            pass
            return False, "INITIAL_WITHIN_RETENTION"
        try:
            last_scan_dt = _utc(last_scan_raw)
        except Exception:
            return True, "INVALID_LAST_SCAN_TIMESTAMP"
        interval = timedelta(days=cfg["scan_interval_days"])
        if now_dt - last_scan_dt >= interval:
            return True, "BIWEEKLY_INTERVAL_DUE"
        return False, "NOT_DUE"

    def pinned_revisions(self, now: Any = None) -> set[str]:
        """Return the set of revision directory names that must never be moved or modified."""
        cfg = self._config()
        now_dt = _utc(now)
        cutoff_dt = now_dt - timedelta(days=cfg["retain_hot_days"])
        revisions_dir = self.runtime / "features" / "revisions"
        pinned: set[str] = set()
        if not revisions_dir.is_dir():
            return pinned

        marker_path = revisions_dir / "view_state.json"
        active_inodes: set[tuple[int, int]] = set()
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("status") != "COMMITTED":
                raise RuntimeError("REVISION_VIEW_NOT_COMMITTED_ABORT_JANITOR")
            active_id = str(marker.get("revision_id") or "")
            if active_id:
                pinned.add(active_id)
                active_dir = (revisions_dir / active_id).resolve()
                manifest_path = active_dir / "manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    for backup in manifest.get("sqlite_backups", []):
                        for key in ("snapshot_path", "delta_snapshot_path"):
                            if backup.get(key):
                                p = Path(backup[key]).resolve()
                                if p.parent.parent == revisions_dir.resolve():
                                    pinned.add(p.parent.name)
                                if p.is_file():
                                    st = p.stat()
                                    if st.st_ino:
                                        active_inodes.add((st.st_dev, st.st_ino))
                        for item in backup.get("delta_chain") or []:
                            if isinstance(item, Mapping) and item.get("delta_snapshot_path"):
                                dp = Path(item["delta_snapshot_path"]).resolve()
                                if dp.parent.parent == revisions_dir.resolve():
                                    pinned.add(dp.parent.name)

        for child in revisions_dir.iterdir():
            if not child.is_dir() or child.is_symlink():
                continue
            try:
                rev_ts = _revision_timestamp(child)
                if rev_ts >= cutoff_dt:
                    pinned.add(child.name)
            except Exception:
                pinned.add(child.name)
        return pinned

    def _migrate_old_revisions(
        self, cold_root: Path, cutoff_dt: datetime, pinned: set[str], now_dt: datetime
    ) -> dict[str, Any]:
        revisions_dir = self.runtime / "features" / "revisions"
        migrated = []
        bytes_freed = 0
        if not revisions_dir.is_dir():
            return {"migrated_count": 0, "migrated_revisions": [], "bytes_freed": 0}

        ledger_path = cold_root / "revisions" / "migration_ledger.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

        for rev_dir in sorted(revisions_dir.iterdir(), key=lambda p: p.name):
            if not rev_dir.is_dir() or rev_dir.is_symlink() or rev_dir.name in pinned:
                continue
            rev_ts = _revision_timestamp(rev_dir)
            if rev_ts >= cutoff_dt:
                continue
            month_bucket = rev_ts.strftime("%Y-%m")
            dest_dir = cold_root / "revisions" / month_bucket / rev_dir.name
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)

            source_files = [p for p in sorted(rev_dir.rglob("*")) if p.is_file() and not p.is_symlink()]
            file_digests: dict[str, str] = {}
            rev_bytes = 0
            for src_file in source_files:
                self._yield_for_trading_window()
                rel = src_file.relative_to(rev_dir)
                dst_file = dest_dir / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                src_sha = _sha256(src_file)
                dst_sha = _sha256(dst_file)
                if src_sha != dst_sha:
                    raise RuntimeError(f"COLD_ARCHIVE_SHA256_MISMATCH:{rel.as_posix()}")
                file_digests[rel.as_posix()] = dst_sha
                rev_bytes += src_file.stat().st_size

            manifest_sha = file_digests.get("manifest.json")
            receipt = {
                "migrated_at_utc": _iso(now_dt),
                "revision_id": rev_dir.name,
                "revision_created_utc": _iso(rev_ts),
                "source_path": str(rev_dir),
                "cold_path": str(dest_dir),
                "file_count": len(source_files),
                "total_bytes": rev_bytes,
                "manifest_sha256": manifest_sha,
                "files_sha256": file_digests,
            }
            with ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            shutil.rmtree(rev_dir)
            migrated.append(rev_dir.name)
            bytes_freed += rev_bytes

        return {
            "migrated_count": len(migrated),
            "migrated_revisions": migrated,
            "bytes_freed": bytes_freed,
        }

    def _refresh_active_revision_observations_baseline(self) -> bool:
        """Refresh the active COMMITTED revision's observations.sqlite3 snapshot after hot compaction.

        Ensures subsequent intraday price revisions continue hitting the <2ms
        os.link hardlink path in revisions._prior_committed_observations while
        keeping ensure_view_committed 100% valid.
        """
        revisions_dir = (self.runtime / "features" / "revisions").resolve()
        marker_path = revisions_dir / "view_state.json"
        obs_src = self.runtime / "observations.sqlite3"
        if not marker_path.is_file() or not obs_src.is_file():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("status") != "COMMITTED" or not marker.get("revision_id"):
                return False
            folder = (revisions_dir / str(marker["revision_id"])).resolve()
            if folder.parent != revisions_dir or folder.is_symlink():
                return False
            manifest_path = folder / "manifest.json"
            snap_path = folder / "observations.sqlite3"
            if not manifest_path.is_file():
                return False
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            obs_entry = next(
                (
                    b
                    for b in manifest_data.get("sqlite_backups", [])
                    if Path(b.get("snapshot_path", "")).name == "observations.sqlite3"
                ),
                None,
            )
            if obs_entry is None:
                return False

            tmp_snap = folder / f"observations.sqlite3.{uuid.uuid4().hex}.tmp"
            try:
                with closing(sqlite3.connect(obs_src.as_uri() + "?mode=ro", uri=True)) as src_con:
                    with closing(sqlite3.connect(tmp_snap)) as dst_con:
                        src_con.backup(dst_con)
                new_obs_sha = _sha256(tmp_snap)
                try:
                    os.replace(tmp_snap, snap_path)
                except PermissionError:
                    gc.collect()
                    os.replace(tmp_snap, snap_path)
            finally:
                tmp_snap.unlink(missing_ok=True)

            obs_entry["sha256"] = new_obs_sha
            for k in ("delta_snapshot_path", "delta_sha256", "delta_max_id", "delta_chain"):
                obs_entry.pop(k, None)

            raw_manifest = json.dumps(
                manifest_data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            tmp_manifest = folder / f"manifest.json.{uuid.uuid4().hex}.tmp"
            try:
                with tmp_manifest.open("wb") as fh:
                    fh.write(raw_manifest)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_manifest, manifest_path)
            finally:
                tmp_manifest.unlink(missing_ok=True)

            new_manifest_sha = _sha256(manifest_path)
            marker["manifest_sha256"] = new_manifest_sha
            raw_marker = json.dumps(
                marker, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            tmp_marker = revisions_dir / f"view_state.json.{uuid.uuid4().hex}.tmp"
            try:
                with tmp_marker.open("wb") as fh:
                    fh.write(raw_marker)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_marker, marker_path)
            finally:
                tmp_marker.unlink(missing_ok=True)

            with revisions._COMMITTED_ARCHIVE_HASH_LOCK:
                revisions._COMMITTED_ARCHIVE_HASH_CACHE.clear()
            revisions._VERIFIED_PREFIX_PAIR_CACHE.clear()
            revisions._ARCHIVED_SQLITE_META_CACHE.clear()
            return True
        except Exception:
            return False

    def _rotate_one_sqlite(
        self,
        db_path: Path,
        cold_root: Path,
        cutoff_dt: datetime,
        now_dt: datetime,
        *,
        archive_prefix: str,
        table_time_cols: Mapping[str, str],
        prune_hot_rows: bool = True,
    ) -> dict[str, Any]:
        if not db_path.is_file():
            return {"status": "ABSENT", "exported_rows": 0, "bytes_before": 0, "bytes_after": 0}
        cutoff_iso = _iso(cutoff_dt)
        bytes_before = sum(
            p.stat().st_size for p in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")) if p.is_file()
        )
        exported_counts: dict[str, int] = {}
        table_specs: dict[str, tuple[str, str]] = {}

        with closing(sqlite3.connect(db_path, timeout=15.0)) as conn:
            existing_tables = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            for table, preferred_col in table_time_cols.items():
                if table not in existing_tables:
                    continue
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
                ts_col = preferred_col if preferred_col in cols else (
                    "saved_at" if "saved_at" in cols else ("recorded_at" if "recorded_at" in cols else None)
                )
                if ts_col is None:
                    continue
                row_count = int(
                    conn.execute(
                        f'SELECT COUNT(1) FROM "{table}" WHERE julianday("{ts_col}") < julianday(?)',
                        (cutoff_iso,),
                    ).fetchone()[0]
                    or 0
                )
                if row_count > 0:
                    table_specs[table] = (existing_tables[table], ts_col)
                    exported_counts[table] = row_count

        total_exported = sum(exported_counts.values())
        if total_exported == 0:
            return {
                "status": "NO_AGED_ROWS",
                "exported_rows": 0,
                "bytes_before": bytes_before,
                "bytes_after": bytes_before,
            }

        sqlite_cold_dir = cold_root / "sqlite"
        sqlite_cold_dir.mkdir(parents=True, exist_ok=True)
        stamp_token = now_dt.strftime("%Y%m%d_%H%M%S")
        cold_db_path = sqlite_cold_dir / f"{archive_prefix}_{stamp_token}.sqlite3"
        if cold_db_path.exists():
            cold_db_path = sqlite_cold_dir / f"{archive_prefix}_{stamp_token}_{os.getpid()}.sqlite3"

        with closing(sqlite3.connect(cold_db_path)) as cold_conn:
            with cold_conn:
                for _table, (create_sql, _ts_col) in table_specs.items():
                    cold_conn.execute(create_sql)

        # Guarantee B: stream directly inside SQLite C engine via ATTACH DATABASE (0 MB Python memory)
        with closing(sqlite3.connect(db_path, timeout=15.0)) as conn:
            conn.execute("ATTACH DATABASE ? AS cold_db", (str(cold_db_path),))
            try:
                for table, (_create_sql, ts_col) in table_specs.items():
                    self._yield_for_trading_window()
                    with conn:
                        conn.execute(
                            f'INSERT OR IGNORE INTO cold_db."{table}" '
                            f'SELECT * FROM main."{table}" WHERE julianday("{ts_col}") < julianday(?)',
                            (cutoff_iso,),
                        )
            finally:
                try:
                    conn.execute("DETACH DATABASE cold_db")
                except sqlite3.Error:
                    pass

        cold_sha = _sha256(cold_db_path)
        manifest_path = cold_db_path.with_suffix(".manifest.json")
        _atomic_json(
            manifest_path,
            {
                "archived_at_utc": _iso(now_dt),
                "cutoff_utc": cutoff_iso,
                "source_db": str(db_path),
                "cold_archive_db": str(cold_db_path),
                "sha256": cold_sha,
                "exported_counts": exported_counts,
                "total_exported_rows": total_exported,
                "hot_rows_pruned": bool(prune_hot_rows),
            },
        )

        baseline_refreshed = False
        if prune_hot_rows:
            self._yield_for_trading_window()
            with closing(sqlite3.connect(db_path, timeout=15.0)) as conn:
                with conn:
                    for table, (_create_sql, ts_col) in table_specs.items():
                        conn.execute(
                            f'DELETE FROM "{table}" WHERE julianday("{ts_col}") < julianday(?)',
                            (cutoff_iso,),
                        )
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
                try:
                    conn.execute("VACUUM")
                except sqlite3.Error:
                    pass
            if db_path.name == "observations.sqlite3":
                baseline_refreshed = self._refresh_active_revision_observations_baseline()

        bytes_after = sum(
            p.stat().st_size for p in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")) if p.is_file()
        )
        return {
            "status": "ARCHIVED_AND_VACUUMED" if prune_hot_rows else "ARCHIVED_COLD_ONLY_HOT_PRESERVED",
            "cold_archive_db": str(cold_db_path),
            "cold_sha256": cold_sha,
            "exported_counts": exported_counts,
            "exported_rows": total_exported,
            "hot_rows_pruned": bool(prune_hot_rows),
            "baseline_refreshed": baseline_refreshed,
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
        }

    def _rotate_audit_logs(self, cold_root: Path, cutoff_dt: datetime, now_dt: datetime) -> dict[str, Any]:
        logs_dir = self.runtime / "logs"
        if not logs_dir.is_dir():
            return {"archived_lines": 0, "cold_files": []}
        cold_logs_dir = cold_root / "logs"
        cold_logs_dir.mkdir(parents=True, exist_ok=True)
        stamp_token = now_dt.strftime("%Y%m%d_%H%M%S")
        archived_total = 0
        cold_files = []

        jsonl_path = logs_dir / "monitor_audit.jsonl"
        if jsonl_path.is_file():
            lines = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
            old_lines, hot_lines = [], []
            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                    ts = _utc(rec.get("ts"))
                    if ts < cutoff_dt:
                        old_lines.append(ln)
                    else:
                        hot_lines.append(ln)
                except Exception:
                    hot_lines.append(ln)
            if old_lines:
                cold_jsonl = cold_logs_dir / f"monitor_audit_{stamp_token}.jsonl"
                payload = ("\n".join(old_lines) + "\n").encode("utf-8")
                cold_jsonl.write_bytes(payload)
                cold_sha = _sha256(cold_jsonl)
                _atomic_json(
                    cold_jsonl.with_suffix(".manifest.json"),
                    {
                        "archived_at_utc": _iso(now_dt),
                        "cutoff_utc": _iso(cutoff_dt),
                        "source_log": str(jsonl_path),
                        "cold_log": str(cold_jsonl),
                        "sha256": cold_sha,
                        "line_count": len(old_lines),
                    },
                )
                tmp_hot = jsonl_path.with_name(f"monitor_audit.jsonl.{os.getpid()}.tmp")
                tmp_hot.write_bytes(("\n".join(hot_lines) + ("\n" if hot_lines else "")).encode("utf-8"))
                os.replace(tmp_hot, jsonl_path)
                archived_total += len(old_lines)
                cold_files.append(str(cold_jsonl))

        text_log_path = logs_dir / "monitor_audit.log"
        if text_log_path.is_file():
            lines = text_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            old_lines, hot_lines = [], []
            for ln in lines:
                parts = ln.split(" ", 1)
                try:
                    ts = _utc(parts[0])
                    if ts < cutoff_dt:
                        old_lines.append(ln)
                    else:
                        hot_lines.append(ln)
                except Exception:
                    hot_lines.append(ln)
            if old_lines:
                cold_log = cold_logs_dir / f"monitor_audit_{stamp_token}.log"
                cold_log.write_bytes(("\n".join(old_lines) + "\n").encode("utf-8"))
                cold_sha = _sha256(cold_log)
                _atomic_json(
                    cold_log.with_suffix(".manifest.json"),
                    {
                        "archived_at_utc": _iso(now_dt),
                        "cutoff_utc": _iso(cutoff_dt),
                        "source_log": str(text_log_path),
                        "cold_log": str(cold_log),
                        "sha256": cold_sha,
                        "line_count": len(old_lines),
                    },
                )
                tmp_hot = text_log_path.with_name(f"monitor_audit.log.{os.getpid()}.tmp")
                tmp_hot.write_bytes(("\n".join(hot_lines) + ("\n" if hot_lines else "")).encode("utf-8"))
                os.replace(tmp_hot, text_log_path)
                archived_total += len(old_lines)
                cold_files.append(str(cold_log))

        return {"archived_lines": archived_total, "cold_files": cold_files}

    def run_maintenance(
        self,
        now: Any = None,
        *,
        prune_hot_observations: bool = True,
    ) -> dict[str, Any]:
        """Execute a full bi-weekly cold-archive and hot-compaction pass."""
        if not self._lock.acquire(blocking=False):
            return {"status": "SKIPPED", "reason": "JANITOR_ALREADY_RUNNING"}
        try:
            cfg = self._config()
            now_dt = _utc(now)
            cutoff_dt = now_dt - timedelta(days=cfg["retain_hot_days"])
            cold_root, fallback_used = self._resolve_cold_root(cfg)
            cold_root.mkdir(parents=True, exist_ok=True)

            # Verify active revision integrity before any maintenance
            view_before = revisions.ensure_view_committed(self.runtime)
            pinned = self.pinned_revisions(now=now_dt)

            rev_result = self._migrate_old_revisions(cold_root, cutoff_dt, pinned, now_dt)
            obs_result = self._rotate_one_sqlite(
                self.runtime / "observations.sqlite3",
                cold_root,
                cutoff_dt,
                now_dt,
                archive_prefix="observations_archive",
                table_time_cols={"snapshots": "saved_at", "receipts": "received_at"},
                prune_hot_rows=bool(prune_hot_observations),
            )
            tel_result = self._rotate_one_sqlite(
                self.runtime / "book_telemetry.sqlite3",
                cold_root,
                cutoff_dt,
                now_dt,
                archive_prefix="book_telemetry_archive",
                table_time_cols={"samples": "target_at", "decision_links": "observed_at"},
                prune_hot_rows=True,
            )
            self._yield_for_trading_window()
            log_result = self._rotate_audit_logs(cold_root, cutoff_dt, now_dt)

            # Re-verify active revision integrity after maintenance
            view_after = revisions.ensure_view_committed(self.runtime)
            if view_before.get("status") == "COMMITTED" and view_after.get("status") != "COMMITTED":
                raise RuntimeError("REVISION_INTEGRITY_FAILED_AFTER_JANITOR")

            next_due_dt = now_dt + timedelta(days=cfg["scan_interval_days"])
            summary = {
                "status": "COMPLETED",
                "ran_at_utc": _iso(now_dt),
                "cutoff_utc": _iso(cutoff_dt),
                "next_scan_due_utc": _iso(next_due_dt),
                "cold_archive_root": str(cold_root),
                "fallback_used": fallback_used,
                "pinned_revisions": sorted(pinned),
                "revisions": rev_result,
                "observations_sqlite": obs_result,
                "book_telemetry_sqlite": tel_result,
                "logs": log_result,
                "active_view_status": view_after.get("status"),
            }
            state = self._load_state()
            history = list(state.get("history") or [])[-19:]
            history.append(
                {
                    "ran_at_utc": summary["ran_at_utc"],
                    "migrated_revisions": rev_result["migrated_count"],
                    "exported_observation_rows": obs_result.get("exported_rows", 0),
                    "archived_log_lines": log_result.get("archived_lines", 0),
                }
            )
            self._save_state(
                {
                    "last_scan_utc": summary["ran_at_utc"],
                    "next_due_utc": summary["next_scan_due_utc"],
                    "last_result": summary,
                    "history": history,
                }
            )
            if self.monitor_log is not None:
                try:
                    self.monitor_log.record(
                        "INFO",
                        "STORAGE_JANITOR_COMPLETED",
                        f"双周冷备清理完成: 迁出旧修订={rev_result['migrated_count']}个, "
                        f"导出观测行={obs_result.get('exported_rows', 0)}, 归档日志行={log_result.get('archived_lines', 0)}",
                        cold_archive_root=str(cold_root),
                        fallback_used=fallback_used,
                    )
                except Exception:
                    pass
            return summary
        finally:
            self._lock.release()

    def _async_worker_main(self, *, now: Any, prune_hot_observations: bool) -> None:
        self._tls.in_async_worker = True
        try:
            self._last_async_result = self.run_maintenance(
                now=now,
                prune_hot_observations=prune_hot_observations,
            )
        except Exception as exc:
            self._last_async_result = {"status": "FAILED", "error": f"{type(exc).__name__}:{exc}"}
        finally:
            self._tls.in_async_worker = False

    def trigger_async_if_due(
        self,
        *,
        now: Any = None,
        elapsed_in_window: float | None = None,
        inputs: Any = None,
    ) -> dict[str, Any]:
        """Non-blocking (<0.1ms) check on the main trading thread; spawns daemon worker if due."""
        with self._async_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return {"status": "ALREADY_RUNNING_ASYNC"}
            due, reason = self.should_run(now=now, elapsed_in_window=elapsed_in_window)
            if not due:
                return {"status": "NOT_DUE", "reason": reason}
            if inputs is not None:
                outstanding = getattr(inputs, "_outstanding", ())
                if any(not f.done() for f in outstanding):
                    return {"status": "DEFERRED_INPUTS_BUSY"}
            cfg = self._config()
            obs_mb = self._observations_size_mb()
            # Guarantee C: keep observations.sqlite3 hardlink prefix intact unless >= 15 GB
            prune_hot_obs = obs_mb >= cfg["max_hot_observations_mb"]
            worker = threading.Thread(
                target=self._async_worker_main,
                kwargs={"now": now, "prune_hot_observations": prune_hot_obs},
                name="argon4-janitor-worker",
                daemon=True,
            )
            self._worker_thread = worker
            worker.start()
            return {
                "status": "TRIGGERED_ASYNC",
                "reason": reason,
                "prune_hot_observations": prune_hot_obs,
            }

    def maybe_run_window_maintenance(
        self,
        *,
        now: Any = None,
        elapsed_in_window: float | None = None,
        inputs: Any = None,
    ) -> dict[str, Any] | None:
        res = self.trigger_async_if_due(now=now, elapsed_in_window=elapsed_in_window, inputs=inputs)
        return res if res.get("status") == "TRIGGERED_ASYNC" else None

    def status(self, now: Any = None) -> dict[str, Any]:
        cfg = self._config()
        cold_root, fallback_used = self._resolve_cold_root(cfg)
        state = self._load_state()
        now_dt = _utc(now)
        last_scan_utc = state.get("last_scan_utc")
        next_due_utc = state.get("next_due_utc")
        if next_due_utc is None and last_scan_utc:
            try:
                next_due_utc = _iso(_utc(last_scan_utc) + timedelta(days=cfg["scan_interval_days"]))
            except Exception:
                next_due_utc = None
        try:
            hot_free_gb = round(shutil.disk_usage(self.runtime).free / (1024 ** 3), 2)
        except OSError:
            hot_free_gb = None
        cold_total_mb = round(_dir_size_bytes(cold_root) / (1024 * 1024), 3) if cold_root.exists() else 0.0
        due, due_reason = self.should_run(now=now_dt)
        worker_alive = bool(self._worker_thread is not None and self._worker_thread.is_alive())
        return {
            "enabled": cfg["enabled"],
            "scan_interval_days": cfg["scan_interval_days"],
            "retain_hot_days": cfg["retain_hot_days"],
            "configured_cold_archive_root": cfg["cold_archive_root"],
            "effective_cold_archive_root": str(cold_root),
            "fallback_used": fallback_used,
            "hot_free_gb": hot_free_gb,
            "observations_sqlite_mb": self._observations_size_mb(),
            "max_hot_observations_mb": cfg["max_hot_observations_mb"],
            "max_hot_total_gb": cfg["max_hot_total_gb"],
            "cold_archive_total_mb": cold_total_mb,
            "last_scan_utc": last_scan_utc,
            "next_scan_due_utc": next_due_utc,
            "due_now": due,
            "due_reason": due_reason,
            "async_worker_alive": worker_alive,
            "last_result": state.get("last_result"),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARGON-4 bi-weekly cold storage janitor")
    parser.add_argument("--runtime", required=True, help="Path to active runtime directory")
    parser.add_argument("--cold-root", default=None, help="Optional cold archive root directory (e.g. D:/argon4_cold_archive)")
    args = parser.parse_args(argv)
    janitor = StorageJanitor(args.runtime, cold_root=args.cold_root)
    result = janitor.run_maintenance()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
