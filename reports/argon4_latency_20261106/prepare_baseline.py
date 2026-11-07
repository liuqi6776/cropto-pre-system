"""Read-only latency baseline; writes only this report directory."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPAIR = ROOT / "reports/argon4_mark_revision_repair_20261106"
spec = importlib.util.spec_from_file_location("prior_repair_audit", REPAIR / "audit.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
read, metadata, now = repair.read, repair.metadata, repair.prior.now


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if (OUT / "baseline_summary.json").exists():
        raise RuntimeError("baseline already exists; do not overwrite original evidence")
    started = now()
    policy = ROOT / "config/argon4_forward_policy.json"
    policy_bytes = policy.read_bytes()
    (OUT / "policy_before.json").write_bytes(policy_bytes)
    policy_meta = metadata(policy)
    if repair.prior.digest(policy_bytes) != policy_meta["sha256"]:
        raise RuntimeError("policy changed during baseline read")
    models = {path: metadata(ROOT / path) for path in read(REPAIR / "production_models_before.json")}
    write("production_models_before.json", models)
    account = repair.prior.account_snapshot()
    write("account_before.json", account)
    old_sources = read(REPAIR / "final_after_focused_snapshot.json")["source"]
    paths = set(old_sources)
    for folder, pattern in [("tools/argon4_forward", "*.py"), ("scripts", "*argon4_forward*"),
                            ("tests", "test_argon4*.py"), ("docs", "ARGON4*.md")]:
        paths.update(path.relative_to(ROOT).as_posix() for path in (ROOT / folder).glob(pattern) if path.is_file())
    sources = {path: metadata(ROOT / path) for path in sorted(paths)}
    write("source_files_before.json", sources)
    write("source_list_prepared.json", {"prepared_at_utc": now(), "source": sorted(paths),
        "scope": "Preparation baseline only; refresh list and capture code freeze separately after root authorization."})
    dynamic = repair.dynamic_snapshot()
    write("runtime_before.json", dynamic)
    authority = {path: metadata(ROOT / path) for path in repair.AUTHORITY}
    write("authority_before.json", authority)
    (OUT / "mark_registry_before.json").write_text(json.dumps(dynamic["mark_registry"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    value = {"started_at_utc": started, "completed_at_utc": now(), "policy": policy_meta,
        "policy_backup_sha256": metadata(OUT / "policy_before.json")["sha256"], "source_count": len(sources),
        "models_count": len(models), "account_counts": account["row_counts"], "account_captured_at_utc": account["captured_at_utc"],
        "old_event_prefix_length": len(account["tables"]["events"]), "funds_conserved": account["funds_conserved"],
        "invalid_event_hashes": account["invalid_event_hashes"], "source_changes_vs_previous_repair_final": [path for path in old_sources if sources[path]["sha256"] != old_sources[path]["sha256"]],
        "models_match_previous_repair_baseline": all(models[path]["sha256"] == row["sha256"] for path,row in read(REPAIR / "production_models_before.json").items()),
        "runtime": dynamic["pipeline"], "service": dynamic["service"], "verification_pause_exists": dynamic["verification_pause_exists"],
        "scope": "Read-only baseline; no chain, model inference, external data retrieval, source/policy/runtime/account mutation or service control.",
        "dynamic_data_policy": "Online data and research-only mark registry references may continue updating; audit separately from frozen source/model/policy."}
    write("baseline_summary.json", value)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
