"""GEN4 validation evidence only; never constructs or mutates a live Account."""
from __future__ import annotations

import hashlib
import ast
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
PRIOR = ROOT / "reports/argon4_waterfall_v5_20261106/workspace_checks"
RUNTIME = ROOT / "paper_runtime/argon4_forward_20261105"
PHASE2_REPORT = ROOT / "reports/argon4_gen4_sparse_candidate_20261106"
PHASE2_MODELS = ROOT / "research/argon4_gen4_sparse_candidate_20261106"
PHASE2_FROZEN = [(PHASE2_REPORT / name).relative_to(ROOT).as_posix() for name in
                 ("preregistration.json", "baseline_snapshot.json", "author_claim_audit.json", "dependency_install_receipt.json",
                  "dependency_support_receipt.json", "dependency_torch_receipt.json", "specialist_tests_final.log")]
AUTHORITY = ["archive/catalog/dataset_registry.json", "archive/catalog/mark_candle_registry.json",
             "source_data/MANIFEST.json", "source_data/staging/core4_p0_20260925/partition_bindings_v1.json"]
UNCHANGING_TABLES = ["signals", "consumed_book_depth", "allocation_consumption", "decision_cohorts", "cohort_streams"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def metadata(path):
    stat = path.stat()
    return {"sha256": digest(path.read_bytes()), "mtime_ns": stat.st_mtime_ns,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), "bytes": stat.st_size}


def prepare():
    old = read(PRIOR / "pytest_end_snapshot.json")
    paths = set(old["source"])
    for folder, pattern in [("tools/argon4_forward", "gen4*.py"), ("scripts", "*argon4_gen4*.py"),
                            ("tests", "test_argon4_gen4*.py"), ("docs", "ARGON4_GEN4*.md")]:
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).glob(pattern))
    paths.update(["tools/argon4_forward/stage.py", "tests/test_argon4_forward_stage.py"])
    result = {"prepared_at_utc": now(), "source": sorted(paths), "old_source_count": len(old["source"]),
              "new_source": sorted(paths - set(old["source"])),
              "phase2_frozen": PHASE2_FROZEN,
              "note": "Refresh this list at explicit FREEZE, before chain_start_snapshot. No verification chain started by prepare."}
    write("source_list.json", result)
    print(json.dumps({"source_count": len(paths), "new_source": result["new_source"]}))


def connect(path):
    con = sqlite3.connect("file:" + path.resolve().as_posix() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("BEGIN")
    return con


def account_snapshot():
    con = connect(RUNTIME / "account.sqlite3")
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
              if r[0] not in ("runtime", "sqlite_sequence")]
    data = {}
    for table in tables:
        columns = list(con.execute('PRAGMA table_info("' + table + '")'))
        order = [c[1] for c in sorted(columns, key=lambda c: c[5]) if c[5]]
        sql = 'SELECT * FROM "' + table + '"' + (" ORDER BY " + ",".join('"' + c + '"' for c in order) if order else "")
        data[table] = [dict(r) for r in con.execute(sql)]
    con.close()
    obs = connect(RUNTIME / "observations.sqlite3")
    receipts = {}
    for row in data["resolutions"]:
        value = json.loads(row["payload"])
        sha = value.get("source_sha256")
        if sha:
            receipts[sha] = [dict(r) for r in obs.execute("SELECT id,received_at,url,sha256,http_status,raw_text FROM receipts WHERE sha256=?", (sha,))]
    obs.close()
    previous, invalid = "0" * 64, []
    for row in data["events"]:
        actual = digest((previous + row["utc"] + row["event_type"] + row["payload"]).encode())
        if row["previous_hash"] != previous or row["event_hash"] != actual:
            invalid.append(row["seq"])
        previous = row["event_hash"]
    a = data["account"][0]
    locked = sum(r["cost_atoms"] + r["fee_atoms"] for r in data["orders"] if r["status"] == "OPEN")
    return {"captured_at_utc": now(), "tables": data, "official_receipts": receipts,
            "row_counts": {key: len(rows) for key, rows in data.items()}, "invalid_event_hashes": invalid,
            "locked_atoms": locked, "funds_conserved": a["cash_atoms"] + locked == a["initial_atoms"] + a["realized_atoms"]}


def protected_research_identity():
    baseline = read(REPORT / "baseline_snapshot.json")
    policy_path = ROOT / "config/argon4_forward_policy.json"
    changed = [Path(path).as_posix() for path, sha in baseline["baseline_artifacts"].items() if metadata(ROOT / path)["sha256"] != sha]
    code = baseline["baseline_code"]
    policy = read(policy_path)
    return {"running_policy_byte_identical": metadata(policy_path)["sha256"] == baseline["running_policy_sha256"],
            "running_policy_semantically_identical": policy == baseline["running_policy"],
            "baseline_model_artifact_changes": changed,
            "baseline_training_code_unchanged": metadata(ROOT / code["path"])["sha256"] == code["sha256"],
            "live_enabled": policy.get("live_enabled"), "candidate_status": read(REPORT / "research_assessment.json")["status"]}


def snapshot(name):
    assert (RUNTIME / "verification_pause.json").exists(), "Explicit pause required before frozen-chain snapshots"
    listed = read(OUT / "source_list.json")["source"]
    baseline = read(REPORT / "baseline_snapshot.json")
    originals = sorted(Path(path).as_posix() for path in baseline["baseline_artifacts"])
    candidates = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "research/argon4_gen4_candidate_20261106").rglob("*")
                        if p.is_file() and (p.suffix == ".joblib" or p.name.endswith(".manifest.json") or p.name.endswith(".manifest.json.sha256")))
    assert len(originals) == len(candidates) == 42
    recent_root = ROOT / "research_data/argon4_gen4_candidate_20261106/recent_inputs"
    recent_manifest = read(recent_root / "snapshot_manifest.json")
    recent = [(recent_root / "raw_data" / row["relative_path"]).relative_to(ROOT).as_posix() for row in recent_manifest["files"]]
    recent.append((recent_root / "snapshot_manifest.json").relative_to(ROOT).as_posix())
    assert len(recent) == 19
    evidence_names = ["baseline_snapshot.json", "preregistration.json", "author_claim_audit.json", "run.json",
                      "research_assessment.json", "historical_evaluation.json", "recent_evaluation.json", "comparison_table.json",
                      "portfolio_sensitivity_registration.json", "portfolio_scenario.json",
                      "historical_seven_stream_comparison.csv", "recent_seven_stream_comparison.csv"]
    evidence = [(REPORT / name).relative_to(ROOT).as_posix() for name in evidence_names if (REPORT / name).is_file()]
    for item in read(REPORT / "run.json")["completed"]:
        evidence.extend(Path(row["path"]).resolve().relative_to(ROOT).as_posix() for row in item["outputs"].values())
    evidence.extend(Path(row["path"]).resolve().relative_to(ROOT).as_posix()
                    for row in read(REPORT / "research_assessment.json")["recent_prediction_receipts"])
    evidence = sorted(set(evidence))
    groups = {"source": listed, "original_models": originals, "candidate_models": candidates,
              "recent_inputs": recent, "authority": AUTHORITY, "study_evidence": evidence,
              "phase2_frozen": PHASE2_FROZEN}
    result = {"recorded_at_utc": now(), **{group: {path: metadata(ROOT / path) for path in paths} for group, paths in groups.items()},
              "protected_research_identity": protected_research_identity(), "account": account_snapshot(),
              "verification_pause": metadata(RUNTIME / "verification_pause.json"),
              "phase2_output_lifecycle": "Training/evaluation outputs may be created during this frozen-source test run; separate final hash/provenance inventory is required. No claim they existed at chain start."}
    result["recent_copy_hash_mismatches"] = [row["relative_path"] for row in recent_manifest["files"]
                                           if metadata(recent_root / "raw_data" / row["relative_path"])["sha256"] != row["sha256"]]
    write(name + ".json", result)
    print(json.dumps({"snapshot": name, "counts": {key: len(result[key]) for key in groups},
                      "protected_identity": result["protected_research_identity"], "account_counts": result["account"]["row_counts"],
                      "funds_conserved": result["account"]["funds_conserved"], "invalid_event_hashes": result["account"]["invalid_event_hashes"]}))


def atoms(value):
    number = Decimal(str(value)) * 1_000_000
    assert number == number.to_integral_value(), "Non-atomic USD value"
    return int(number)


def timepoint(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def array(value):
    return json.loads(value) if isinstance(value, str) else value


def validate_resolution(resolution, market, event, receipts):
    errors = []
    identifier = str(market["market_id"])
    url = "https://gamma-api.polymarket.com/markets/" + identifier
    if (resolution.get("kind") != "OBSERVED_OFFICIAL_RESOLUTION" or resolution.get("market_id") != identifier
            or resolution.get("source") != url or resolution.get("final_status") != "resolved"
            or not timepoint(market["window_end"]) <= timepoint(resolution["resolved_at"]) <= timepoint(resolution["received_at"]) <= timepoint(event["utc"])):
        errors.append("resolution_identity_or_time")
    matches = receipts.get(resolution.get("source_sha256"), [])
    valid = False
    for receipt in matches:
        body = receipt.get("raw_text")
        if (not isinstance(body, str) or digest(body.encode()) != resolution.get("source_sha256")
                or receipt.get("url") != url or receipt.get("http_status") != 200):
            continue
        raw = json.loads(body)
        protocol = raw.get("version")
        if (str(raw.get("id")) != identifier or raw.get("conditionId") != market["condition_id"]
                or protocol != market["protocol"] or raw.get("resolutionSource") != market["resolution_source"]
                or timepoint(raw["eventStartTime"]) != timepoint(market["window_start"])
                or timepoint(raw["endDate"]) != timepoint(market["window_end"])):
            continue
        if not ((protocol == "v2" and raw.get("resolutionStatus") == "resolved")
                or (protocol == "v1" and raw.get("umaResolutionStatus") in ("resolved", "settled"))):
            continue
        labels = array(raw.get("outcomes"))
        tokens = array(raw.get("positionIds" if protocol == "v2" else "clobTokenIds"))
        prices = [Decimal(str(x)) for x in array(raw.get("outcomePrices"))]
        if (len(labels) != 2 or set(labels) != {"Up", "Down"} or sorted(prices) != [Decimal(0), Decimal(1)]
                or {label.upper(): str(token) for label, token in zip(labels, tokens)} != market["outcome_ids"]):
            continue
        winner = labels[prices.index(Decimal(1))].upper()
        if resolution.get("outcome") != winner or resolution.get("resolution_id") != digest((identifier + market["condition_id"] + winner).encode()):
            continue
        valid = True
    if not valid:
        errors.append("no_matching_official_final_raw_receipt")
    return errors


def account_delta(before, after):
    a, b = before["tables"], after["tables"]
    issues = []
    for table in UNCHANGING_TABLES:
        if a[table] != b[table]:
            issues.append(table + "_changed_during_pause")
    if b["events"][:len(a["events"])] != a["events"]:
        issues.append("old_event_prefix_changed")
    if b["equity_observations"][:len(a["equity_observations"])] != a["equity_observations"]:
        issues.append("old_equity_prefix_changed")
    old_orders = {r["order_id"]: r for r in a["orders"]}
    new_orders = {r["order_id"]: r for r in b["orders"]}
    if set(old_orders) != set(new_orders):
        issues.append("order_identity_set_changed")
    for key in old_orders.keys() & new_orders.keys():
        if {k: v for k, v in old_orders[key].items() if k != "status"} != {k: v for k, v in new_orders[key].items() if k != "status"}:
            issues.append("order_immutable_fields_changed:" + key)
    account = dict(a["account"][0])
    expected_statuses = {key: row["status"] for key, row in old_orders.items()}
    expected_resolutions = {row["market_id"]: row for row in a["resolutions"]}
    expected_equity = []
    added = b["events"][len(a["events"]):]
    settlements = []
    for event in added:
        if event["event_type"] != "OFFICIAL_RESOLUTION_APPLIED_TO_PAPER":
            issues.append("unexpected_new_event:" + event["event_type"])
            continue
        payload = json.loads(event["payload"])
        market_id = payload["market_id"]
        resolution = payload["resolution"]
        eligible = [row for key, row in old_orders.items() if row["market_id"] == market_id and expected_statuses[key] == "OPEN"]
        if not eligible or market_id in expected_resolutions:
            issues.append("settlement_without_original_open_order:" + market_id)
            continue
        expected_market = json.loads(eligible[0]["payload"])["venue_market"]
        issues.extend("official_evidence:" + issue for issue in validate_resolution(resolution, expected_market, event, after["official_receipts"]))
        paid = realized = 0
        expected_items = []
        for order in eligible:
            invested = order["cost_atoms"] + order["fee_atoms"]
            if resolution["outcome"] == "VOID":
                payout, status = invested, "VOID"
            elif resolution["outcome"] == order["outcome"]:
                payout = int((Decimal(order["quantity"]) * 1_000_000).to_integral_value(rounding=ROUND_DOWN))
                status = "WON"
            else:
                payout, status = 0, "LOST"
            paid += payout
            realized += payout - invested
            expected_statuses[order["order_id"]] = status
            expected_items.append({"order_id": order["order_id"], "status": status, "payout_atoms": payout, "realized_atoms": payout - invested})
        actual_items = [{"order_id": row["order_id"], "status": row["status"], "payout_atoms": atoms(row["payout_usd"]),
                         "realized_atoms": atoms(row["realized_pnl_usd"])} for row in payload["orders"]]
        if sorted(actual_items, key=lambda x: x["order_id"]) != sorted(expected_items, key=lambda x: x["order_id"]):
            issues.append("settlement_order_cashflows_mismatch:" + market_id)
        if payload.get("status") != "SETTLED" or atoms(payload["payout_usd"]) != paid or atoms(payload["realized_pnl_usd"]) != realized:
            issues.append("settlement_total_cashflows_mismatch:" + market_id)
        account["cash_atoms"] += paid
        account["realized_atoms"] += realized
        balance = account["initial_atoms"] + account["realized_atoms"]
        account["peak_realized_atoms"] = max(account["peak_realized_atoms"], balance)
        peak = account["peak_realized_atoms"]
        account["max_realized_dd"] = min(account["max_realized_dd"], (balance - peak) / peak if peak else 0.0)
        locked = sum(row["cost_atoms"] + row["fee_atoms"] for key, row in old_orders.items() if expected_statuses[key] == "OPEN")
        expected_equity.append({"seq": a["equity_observations"][-1]["seq"] + len(expected_equity) + 1,
                                "utc": event["utc"], "cash_atoms": account["cash_atoms"], "locked_atoms": locked,
                                "realized_atoms": account["realized_atoms"], "equity_atoms": None if locked else account["cash_atoms"],
                                "valuation_status": "UNVALUED_OPEN_POSITIONS" if locked else "CASH_ONLY"})
        expected_resolutions[market_id] = {"resolution": resolution, "applied_utc": event["utc"]}
        settlements.append({"event_seq": event["seq"], "market_id": market_id, "payout_atoms": paid, "realized_atoms": realized,
                            "source_sha256": resolution.get("source_sha256")})
    if account != b["account"][0]:
        issues.append("account_delta_not_explained_by_official_settlements")
    for key in old_orders.keys() & new_orders.keys():
        if new_orders[key]["status"] != expected_statuses[key]:
            issues.append("order_status_not_explained_by_official_settlement:" + key)
    new_resolution_rows = {r["market_id"]: r for r in b["resolutions"]}
    if set(new_resolution_rows) != set(expected_resolutions):
        issues.append("resolution_identity_set_mismatch")
    for key, row in expected_resolutions.items():
        current = new_resolution_rows.get(key)
        if "payload" in row:
            if current != row:
                issues.append("old_resolution_changed:" + key)
        elif current is None or json.loads(current["payload"]) != row["resolution"]:
            issues.append("resolution_payload_mismatch:" + key)
    actual_equity = b["equity_observations"][len(a["equity_observations"]):]
    if actual_equity != expected_equity:
        issues.append("new_equity_rows_not_explained_by_official_settlements")
    if before["invalid_event_hashes"] or after["invalid_event_hashes"]:
        issues.append("event_hash_chain_invalid")
    if not before["funds_conserved"] or not after["funds_conserved"]:
        issues.append("funds_not_conserved")
    return {"valid": not issues, "issues": issues, "old_event_count": len(a["events"]), "new_event_count": len(added),
            "permitted_official_settlements": settlements, "old_order_count": len(old_orders), "final_order_count": len(new_orders),
            "old_signal_count": len(a["signals"]), "final_signal_count": len(b["signals"]),
            "old_cohort_count": len(a["decision_cohorts"]), "final_cohort_count": len(b["decision_cohorts"]),
            "old_account": a["account"][0], "final_account": b["account"][0],
            "rule": "Original event prefix and order immutable fields preserved; only official receipt-backed settlement events, statuses and reconciled cashflows may advance during input pause."}


def parse_pytest(path):
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    summaries = [line.strip("= \r") for line in content.splitlines() if re.search(r"\d+ passed", line) and re.search(r" in \d+\.\d+s", line)]
    summary = summaries[-1] if summaries else None
    return {"log": path.relative_to(ROOT).as_posix(), "summary": summary,
            "counts": {key: int(value) for value, key in re.findall(r"(\d+) (failed|passed|skipped|warnings|errors)", summary or "")},
            "failed": sorted(set(re.findall(r"^FAILED (\S+)", content, re.M))), "errors": sorted(set(re.findall(r"^ERROR (\S+)", content, re.M)))}


def phase2_final_inventory():
    """Hash/provenance audit only: the base environment never unpickles tensors."""
    run_path = PHASE2_REPORT / "run.json"
    if not run_path.is_file() or read(run_path).get("status") != "COMPLETED":
        run_status = read(run_path).get("status") if run_path.is_file() else "NOT_STARTED"
        result = {"status": "FAILED_TRAINING" if run_status == "FAILED" else "PENDING_TRAINING_COMPLETION", "checked_at_utc": now(), "valid": False,
                  "run_status": run_status,
                  "note": "Phase-two artifacts are permitted new outputs, not frozen pre-existing files."}
        write("phase2_final_inventory.json", result)
        return result
    run = read(run_path)
    prereg_path = PHASE2_REPORT / "preregistration.json"
    prereg = read(prereg_path)
    problems, inventory, output_inventory = [], {}, {}
    canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    object_sha = lambda value: digest(canonical(value).encode())
    code_path = ROOT / "tools/argon4_forward/gen4_sparse_candidate.py"
    network_path = ROOT / "tools/argon4_forward/gen4_sparse_network.py"
    tree = ast.parse(network_path.read_text(encoding="utf-8"))
    config_node = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets))
    network_config = ast.literal_eval(config_node)
    expected_streams = set(read(REPORT / "baseline_snapshot.json")["running_policy"]["order"])
    expected = {(stream, phase) for stream in expected_streams for phase in ("oos_2026", "deployment_q4")}
    completed = run.get("completed", [])
    if len(completed) != 14 or {(item["stream"], item["phase"]) for item in completed} != expected:
        problems.append("incomplete_or_duplicate_fourteen_packages")
    if (run.get("candidate_code_sha256") != metadata(code_path)["sha256"]
            or run.get("network_code_sha256") != metadata(network_path)["sha256"]
            or run.get("config_sha256") != object_sha(run.get("config"))
            or run.get("config_sha256") != prereg.get("config_sha256")
            or run.get("config", {}).get("network_config") != network_config):
        problems.append("run_code_or_config_pin_mismatch")
    provenance = run.get("provenance", {})
    for label, expected_sha in {
            "preregistration_sha256": metadata(prereg_path)["sha256"],
            "tail_run_sha256": metadata(REPORT / "run.json")["sha256"],
            "paper_policy_sha256": metadata(ROOT / "config/argon4_forward_policy.json")["sha256"],
            "recent_snapshot_manifest_sha256": metadata(ROOT / "research_data/argon4_gen4_candidate_20261106/recent_inputs/snapshot_manifest.json")["sha256"],
            "training_script_sha256": metadata(ROOT / "scripts/train_argon4_gen4_sparse_candidate.py")["sha256"],
    }.items():
        if provenance.get(label) != expected_sha:
            problems.append("run_provenance_mismatch:" + label)
    tensor_count = 0
    for item in completed:
        stream, phase = item["stream"], item["phase"]
        folder = PHASE2_MODELS / phase
        manifest_path = folder / (stream + ".manifest.json")
        manifest = read(manifest_path)
        pin = metadata(manifest_path)["sha256"]
        sidecar = manifest_path.with_suffix(".json.sha256")
        bundle_path = folder / (stream + ".bundle.json")
        package = read(bundle_path)
        if (pin != item["manifest_sha256"] or sidecar.read_text(encoding="ascii").strip() != pin
                or manifest.get("model_filename") != bundle_path.name or manifest.get("model_sha256") != metadata(bundle_path)["sha256"]
                or item.get("model_sha256") != manifest.get("model_sha256")):
            problems.append("package_hash_mismatch:" + stream + ":" + phase)
        for path in (manifest_path, sidecar, bundle_path):
            inventory[path.relative_to(ROOT).as_posix()] = metadata(path)
        for key, expected_value in {"stream": stream, "phase": phase, "candidate_code_sha256": run["candidate_code_sha256"],
                "network_code_sha256": run["network_code_sha256"], "config_sha256": run["config_sha256"],
                "network_config_sha256": object_sha(network_config), "config": run["config"], "run_provenance": provenance,
                "live_enabled": False, "production_published": False}.items():
            if manifest.get(key) != expected_value:
                problems.append("manifest_identity_mismatch:" + stream + ":" + phase + ":" + key)
        for label, model_root in (("tail_reference", ROOT / "research/argon4_gen4_candidate_20261106"),
                                  ("baseline_reference", ROOT / "research/argon4_local_v4_20261106")):
            reference = manifest[label]
            reference_manifest = model_root / phase / (stream + ".manifest.json")
            referenced = read(reference_manifest)
            if (reference.get("manifest_sha256") != metadata(reference_manifest)["sha256"]
                    or reference.get("model_sha256") != referenced["model_sha256"]
                    or metadata(reference_manifest.with_name(referenced["model_filename"]))["sha256"] != referenced["model_sha256"]
                    or package[label] != reference):
                problems.append("transitive_reference_mismatch:" + stream + ":" + phase + ":" + label)
        skipped = stream in ("SOL_15m", "BNB_15m")
        wanted_seeds = set() if skipped else {"42", "456", "789"}
        networks = manifest.get("network_files", {})
        if set(networks) != wanted_seeds or package.get("network_files") != networks or manifest.get("skipped_by_fixed_route") != skipped:
            problems.append("fixed_route_or_seed_set_mismatch:" + stream + ":" + phase)
        for seed, tensor in networks.items():
            name = stream + ".seed" + seed + ".pt"
            path = folder / name
            if tensor.get("filename") != name or metadata(path)["sha256"] != tensor.get("sha256"):
                problems.append("tensor_hash_mismatch:" + stream + ":" + phase + ":" + seed)
            inventory[path.relative_to(ROOT).as_posix()] = metadata(path)
            tensor_count += 1
        if (item.get("controls_parity", {}).get("status") != "PASS" or item.get("parity", {}).get("saved_batch") != "EXACT"
                or item.get("parity", {}).get("one_row_batch") != "PASS"):
            problems.append("prediction_parity_missing:" + stream + ":" + phase)
        required_splits = {"validation", "test"} if phase == "oos_2026" else {"validation", "recent"}
        if not required_splits.issubset(item.get("outputs", {})) or not required_splits.issubset(item.get("controls_parity", {}).get("splits", {})):
            problems.append("required_prediction_window_missing:" + stream + ":" + phase)
        for split, evidence in item.get("outputs", {}).items():
            path = Path(evidence["path"]).resolve()
            if not path.is_relative_to(PHASE2_REPORT) or metadata(path)["sha256"] != evidence["sha256"]:
                problems.append("prediction_output_hash_mismatch:" + stream + ":" + phase + ":" + split)
            else:
                output_inventory[path.relative_to(ROOT).as_posix()] = metadata(path)
    if len(inventory) != 72 or tensor_count != 30:
        problems.append("expected_72_artifacts_including_30_tensors")
    actual_artifacts = {p.relative_to(ROOT).as_posix() for p in PHASE2_MODELS.rglob("*") if p.is_file()}
    if actual_artifacts != set(inventory):
        problems.append("unexpected_or_missing_phase2_model_artifacts")
    for name in ("run.json", "research_assessment.json", "historical_evaluation.json", "recent_evaluation.json",
                 "comparison_table.json", "portfolio_scenario.json"):
        path = PHASE2_REPORT / name
        if path.is_file():
            output_inventory[path.relative_to(ROOT).as_posix()] = metadata(path)
    assessment_path = PHASE2_REPORT / "research_assessment.json"
    evaluation_done = assessment_path.is_file()
    if evaluation_done:
        assessment = read(assessment_path)
        for field, filename in (("historical_evaluation_sha256", "historical_evaluation.json"),
                                ("recent_evaluation_sha256", "recent_evaluation.json"),
                                ("capital_sensitivity_sha256", "portfolio_scenario.json")):
            if assessment.get(field) != metadata(PHASE2_REPORT / filename)["sha256"]:
                problems.append("evaluation_hash_mismatch:" + field)
        for field, path in {"preregistration_sha256": prereg_path,
                            "evaluation_script_sha256": ROOT / "scripts/evaluate_argon4_gen4_sparse_candidate.py",
                            "evaluation_module_sha256": ROOT / "tools/argon4_forward/gen4_evaluation.py"}.items():
            if assessment.get(field) != metadata(path)["sha256"]:
                problems.append("evaluation_source_pin_mismatch:" + field)
        execution_path = PHASE2_REPORT / "evaluation_execution_receipt.json"
        execution = read(execution_path)
        fixed_evaluation_pins = {
            "research_assessment.json": "3580b070a2dd56f2eb299b87a1ffcb21f6c1f27a7b437b81b6f0f1c812178835",
            "evaluation_execution_receipt.json": "8014505d932e03898bcf96f6475076a241ecfa1c8af492cf7cfeb30d89b12fe3",
        }
        for name, expected_sha in fixed_evaluation_pins.items():
            path = PHASE2_REPORT / name
            if metadata(path)["sha256"] != expected_sha:
                problems.append("root_reported_evaluation_pin_mismatch:" + name)
            output_inventory[path.relative_to(ROOT).as_posix()] = metadata(path)
        if (execution.get("status") != "EVALUATION_COMPLETED" or execution.get("exit_code") != 0
                or execution.get("production_published") is not False or execution.get("live_enabled") is not False):
            problems.append("evaluation_execution_receipt_invalid")
        for name, evidence in execution.get("artifacts", {}).items():
            path = Path(evidence["path"]).resolve()
            if not path.is_relative_to(PHASE2_REPORT) or path.name != name or metadata(path)["sha256"] != evidence["sha256"]:
                problems.append("evaluation_execution_artifact_hash_mismatch:" + name)
            else:
                output_inventory[path.relative_to(ROOT).as_posix()] = metadata(path)
        if execution.get("scopes", {}).get("stage1_primary", {}).get("status") != assessment.get("status"):
            problems.append("primary_scope_conclusion_mismatch")
    result = {"status": "COMPLETE" if not problems and evaluation_done else "PENDING_EVALUATION" if not problems else "FAILED",
              "checked_at_utc": now(), "valid": not problems and evaluation_done, "issues": problems,
              "research_assessment_status": assessment.get("status") if evaluation_done else "PENDING",
              "artifact_count": len(inventory), "tensor_file_count": tensor_count, "artifacts": inventory,
              "output_inventory": output_inventory, "run_sha256": metadata(run_path)["sha256"],
              "tensor_payload_loading": "Not loaded in base-environment audit. Restricted weights-only reload and prediction parity are independently exercised in the pinned Torch specialist environment and recorded by training.",
              "temporal_scope": "Final inventory of newly generated research artifacts; never represented as present at chain start."}
    write("phase2_final_inventory.json", result)
    return result


def finalize():
    first, start, end = (read(OUT / (name + ".json")) for name in ("chain_start_snapshot", "pytest_start_snapshot", "pytest_end_snapshot"))
    baseline = parse_pytest(PRIOR / "05_pytest_full.log")
    current = parse_pytest(OUT / "05_pytest_full.log")
    specialist = parse_pytest(PHASE2_REPORT / "specialist_tests_final.log")
    comparison = {"new_failures": sorted(set(current["failed"]) - set(baseline["failed"])),
                  "no_longer_failing": sorted(set(baseline["failed"]) - set(current["failed"])),
                  "failure_set_identical": current["failed"] == baseline["failed"],
                  "passed_count_change": current["counts"].get("passed", 0) - baseline["counts"].get("passed", 0)}
    integrity = {}
    for group in ("source", "original_models", "candidate_models", "recent_inputs", "authority", "study_evidence", "phase2_frozen"):
        base = start if group == "authority" else first
        changes = {kind + "_changes": [path for path, value in base[group].items() if end[group].get(path, {}).get(kind) != value[kind]]
                   for kind in ("sha256", "mtime_ns")}
        integrity[group] = {"count": len(end[group]), **changes, "stable": not any(changes.values())}
    ledger = account_delta(first["account"], end["account"])
    phase2 = phase2_final_inventory()
    write("account_settlement_audit.json", ledger)
    stage_specs = [("01_fetch_verify", "fetch_local_source_copies --verify"), ("02_import", "import_perp_data"),
                   ("03_inventory", "build_inventory"), ("04_reports", "build_reports"), ("05_pytest_full", "pytest tests -q")]
    stages = [{"order": index, "stage": label, "exit_code": int((OUT / (stem + ".exit.txt")).read_text(encoding="utf-8-sig").strip()),
               "log": stem + ".log", "sha256": metadata(OUT / (stem + ".log"))["sha256"]}
              for index, (stem, label) in enumerate(stage_specs, 1)]
    protected = end["protected_research_identity"]
    may_resume = (all(row["stable"] for row in integrity.values()) and ledger["valid"]
                  and protected["running_policy_byte_identical"] and protected["running_policy_semantically_identical"]
                  and not protected["baseline_model_artifact_changes"] and protected["baseline_training_code_unchanged"]
                  and protected["live_enabled"] is False and protected["candidate_status"] == "DO_NOT_PROMOTE"
                  and not end["recent_copy_hash_mismatches"] and not comparison["new_failures"] and not current["errors"]
                  and specialist["counts"].get("passed") == 43 and not specialist["failed"] and not specialist["errors"]
                  and current["summary"] is not None and first["verification_pause"] == end["verification_pause"]
                  and [row["exit_code"] for row in stages[:4]] == [1, 3, 0, 0])
    summary = {"completed_at_utc": now(), "scope": "GEN4_ISOLATED_RESEARCH_FINAL_WORKSPACE_VALIDATION",
               "status": "COMPLETED_WITH_INHERITED_FAILURES" if may_resume else "REQUIRES_REVIEW",
               "preflight_stages": stages[:4], "pytest_stage": stages[4],
               "pytest": {"baseline": baseline, "current": current, "comparison": comparison}, "integrity": integrity,
               "torch_specialist_pytest": specialist,
               "test_environment_scope": "Full workspace pytest uses the original project environment without research PYTHONPATH; unavailable Torch modules are explicit skips. Separately, all 43 sparse-network/candidate/evaluation specialist tests actually pass in the pinned compatible Torch environment. Skips are never counted as passes.",
               "phase2_final_artifact_audit": {key: value for key, value in phase2.items() if key not in ("artifacts", "output_inventory")},
               "protected_research_identity": protected, "account_settlement_audit": ledger,
               "required_generation_authority_changes": [path for path in AUTHORITY if first["authority"][path] != start["authority"][path]],
               "verification_pause_unchanged": first["verification_pause"] == end["verification_pause"],
               "runtime_pause_may_be_removed": may_resume, "candidate_promoted": False,
               "overall_research_evidence_complete": may_resume and phase2["valid"],
               "source_verification": "54 inherited source identities remain unverified; conversion refuses them. No canonical readiness promotion.",
               "limitations": "Software verification does not establish trading effectiveness. The phase-one tail candidate remains DO_NOT_PROMOTE; the phase-two verdict is recorded independently, without promotion by this audit. Original model and policy remain in production paper.",
               "changes_by_check_agent": "Verification artifacts and required generated manifest/inventory/reports only; no business source, model, policy, UI or live-account mutation."}
    write("failure_comparison.json", summary["pytest"])
    write("summary.json", summary)
    print(json.dumps({"summary": str(OUT / "summary.json"), "pytest": current["summary"], "comparison": comparison,
                      "integrity": integrity, "account_valid": ledger["valid"], "new_settlements": len(ledger["permitted_official_settlements"]),
                      "may_resume": may_resume}))


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "phase2_final":
        result = phase2_final_inventory()
        print(json.dumps({key: value for key, value in result.items() if key not in ("artifacts", "output_inventory")}))
    elif sys.argv[1] == "finalize":
        finalize()
    else:
        snapshot(sys.argv[1])
