"""Sync ARGON-4 tiered cold archives to Google Drive via rclone with 100GB quota enforcement.

Workflow:
1. Optionally runs `StorageJanitor.run_maintenance()` to snapshot `permanent_core/`
   (ledgers + `[-60s, +300s]` trade keyframes) and compress aged SQLite/log slices.
2. Checks if the configured `rclone` remote (default `gdrive:`) is authorized;
   if not yet configured, exits cleanly while local `/data/archive` cap still protects disk.
3. Copies `permanent_core/` and rolling (`sqlite/`, `revisions/`, `logs/`) to Google Drive
   over Google's free internal network path.
4. Enforces `--max-gdrive-gb` (default `88.0`, leaving 12GB buffer under 100GB) on Google
   Drive by FIFO-deleting only the oldest Tier-2 rolling raw files with
   `--drive-use-trash=false` (never touching `permanent_core/`).
5. Optionally removes already-synced Tier-2 rolling files older than `--keep-local-days`
   from the VM's local `/data/archive` directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.argon4_forward.storage_janitor import StorageJanitor


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _rclone_remotes(rclone_bin: str) -> set[str]:
    res = subprocess.run(
        [rclone_bin, "listremotes"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if res.returncode != 0:
        return set()
    return {ln.strip().rstrip(":") for ln in res.stdout.splitlines() if ln.strip()}


def _rclone_copy(rclone_bin: str, src: Path, dst_remote: str) -> dict[str, Any]:
    if not src.exists():
        return {"status": "ABSENT", "src": str(src)}
    cmd = [
        rclone_bin,
        "copy",
        str(src),
        dst_remote,
        "--transfers=4",
        "--checkers=8",
        "--checksum",
        "--fast-list",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"rclone copy failed ({src} -> {dst_remote}): {res.stderr.strip()[:400]}")
    return {"status": "SYNCED", "src": str(src), "dst": dst_remote}


def _rclone_total_bytes(rclone_bin: str, remote_root: str) -> int:
    res = subprocess.run(
        [rclone_bin, "size", "--json", remote_root],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if res.returncode != 0:
        return 0
    try:
        data = json.loads(res.stdout)
        return int(data.get("bytes", 0))
    except Exception:
        return 0


def _enforce_gdrive_cap(
    rclone_bin: str, remote_root: str, max_gdrive_gb: float
) -> dict[str, Any]:
    """FIFO-delete oldest Tier-2 rolling files on Google Drive with --drive-use-trash=false."""
    if max_gdrive_gb <= 0:
        return {"enforced": False}
    limit_bytes = int(max_gdrive_gb * (1024 ** 3))
    total_bytes = _rclone_total_bytes(rclone_bin, remote_root)
    if total_bytes <= limit_bytes:
        return {
            "enforced": True,
            "pruned_remote_files": 0,
            "remote_bytes_after": total_bytes,
            "limit_bytes": limit_bytes,
        }

    candidates: list[tuple[str, str, int]] = []
    for sub in ("sqlite", "logs"):
        sub_remote = f"{remote_root.rstrip('/')}/{sub}"
        res = subprocess.run(
            [rclone_bin, "lsjson", "-R", "--files-only", sub_remote],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if res.returncode != 0:
            continue
        try:
            items = json.loads(res.stdout)
        except Exception:
            continue
        for item in items:
            rel_path = str(item.get("Path") or "")
            if not rel_path or rel_path.endswith(("migration_ledger.jsonl", "cap_enforcement_ledger.jsonl")):
                continue
            mod_time = str(item.get("ModTime") or "")
            size_b = int(item.get("Size") or 0)
            candidates.append((mod_time, f"{sub_remote}/{rel_path}", size_b))

    candidates.sort(key=lambda x: (x[0], x[1]))
    deleted = 0
    freed = 0
    for _mtime, full_remote_file, sz in candidates:
        if total_bytes <= limit_bytes:
            break
        del_res = subprocess.run(
            [rclone_bin, "deletefile", "--drive-use-trash=false", full_remote_file],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if del_res.returncode == 0:
            total_bytes -= sz
            freed += sz
            deleted += 1

    return {
        "enforced": True,
        "pruned_remote_files": deleted,
        "remote_bytes_freed": freed,
        "remote_bytes_after": max(0, total_bytes),
        "limit_bytes": limit_bytes,
    }


def _prune_synced_local_rolling(archive_dir: Path, keep_local_days: int) -> dict[str, Any]:
    if keep_local_days <= 0:
        return {"pruned_local_files": 0, "bytes_freed": 0}
    cutoff_ts = (_utc_now() - timedelta(days=keep_local_days)).timestamp()
    pruned = 0
    freed = 0
    for sub in ("sqlite", "logs"):
        sub_dir = archive_dir / sub
        if not sub_dir.is_dir():
            continue
        for p in sorted(sub_dir.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            try:
                st = p.stat()
                if st.st_mtime < cutoff_ts:
                    p.unlink(missing_ok=True)
                    pruned += 1
                    freed += st.st_size
            except OSError:
                continue
    return {"pruned_local_files": pruned, "bytes_freed": freed}


def sync_archive(
    *,
    runtime: Path,
    archive_dir: Path,
    remote: str = "gdrive:argon4_archive",
    run_janitor: bool = True,
    max_gdrive_gb: float = 88.0,
    keep_local_days: int = 7,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"ran_at_utc": _utc_now().isoformat()}
    if run_janitor:
        janitor = StorageJanitor(runtime, cold_root=archive_dir)
        summary["janitor"] = janitor.run_maintenance()

    rclone_bin = shutil.which("rclone")
    if not rclone_bin:
        summary["gdrive_sync"] = {"status": "SKIPPED_RCLONE_MISSING"}
        return summary

    remote_name = remote.split(":", 1)[0].strip()
    if remote_name not in _rclone_remotes(rclone_bin):
        summary["gdrive_sync"] = {
            "status": "SKIPPED_REMOTE_NOT_CONFIGURED",
            "expected_remote": remote_name,
            "hint": f"Run `rclone config` to authorize `{remote_name}:` (Google Drive).",
        }
        return summary

    remote_base = remote.rstrip("/")
    copied = []
    # 1. Always sync permanent_core first (PnL ledgers + trade keyframes)
    copied.append(_rclone_copy(rclone_bin, archive_dir / "permanent_core", f"{remote_base}/permanent_core"))
    # 2. Sync Tier-2 rolling directories
    for sub in ("sqlite", "revisions", "logs"):
        copied.append(_rclone_copy(rclone_bin, archive_dir / sub, f"{remote_base}/{sub}"))

    cap_info = _enforce_gdrive_cap(rclone_bin, remote_base, max_gdrive_gb)
    local_cleanup = _prune_synced_local_rolling(archive_dir, keep_local_days)
    summary["gdrive_sync"] = {
        "status": "COMPLETED",
        "remote": remote_base,
        "copied": copied,
        "gdrive_cap": cap_info,
        "local_post_sync_cleanup": local_cleanup,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "paper_runtime/argon4_forward_20261105",
    )
    parser.add_argument("--archive-dir", type=Path, default=Path("/data/archive"))
    parser.add_argument("--remote", default="gdrive:argon4_archive")
    parser.add_argument("--no-janitor", action="store_true")
    parser.add_argument("--max-gdrive-gb", type=float, default=88.0)
    parser.add_argument("--keep-local-days", type=int, default=7)
    args = parser.parse_args(argv)

    result = sync_archive(
        runtime=args.runtime.resolve(),
        archive_dir=args.archive_dir.resolve(),
        remote=args.remote,
        run_janitor=not args.no_janitor,
        max_gdrive_gb=args.max_gdrive_gb,
        keep_local_days=args.keep_local_days,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
