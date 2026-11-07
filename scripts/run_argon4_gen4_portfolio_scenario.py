"""Run registered post-hoc capital scenarios; never alter model selection."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import pandas as pd
from threadpoolctl import threadpool_limits
from tools.argon4_forward import gen4_portfolio_scenario as scenario

REPORT=ROOT/"reports/argon4_gen4_candidate_20261106"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    registration_path=REPORT/"portfolio_sensitivity_registration.json"
    registration=json.loads(registration_path.read_text(encoding="utf-8"))
    required={"classification":scenario.CLASSIFICATION,
              "observed_primary_result_before_registration":"DO_NOT_PROMOTE",
              "fixed_variants":list(scenario.VARIANTS),"fee_rates":list(scenario.FEE_RATES),
              "settlement_delay_minutes":list(scenario.DELAYS),"buy_price_assumption":float(scenario.PRICE),
              "no_account_database_writes":True,"cannot_override_preregistered_model_screen":True,
              "no_claim_of_actual_account_returns":True}
    if any(registration.get(key)!=value for key,value in required.items()):
        raise ValueError("fixed post-hoc sensitivity registration required")
    snapshot_path=REPORT/"baseline_snapshot.json"
    snapshot=json.loads(snapshot_path.read_text(encoding="utf-8"))
    policy=snapshot["running_policy"]
    policy_path=ROOT/"config/argon4_forward_policy.json"
    if sha(policy_path)!=snapshot["running_policy_sha256"]:
        raise ValueError("production policy differs from frozen study policy")
    assessment_path=REPORT/"research_assessment.json"
    assessment=json.loads(assessment_path.read_text(encoding="utf-8"))
    if assessment["status"]!="DO_NOT_PROMOTE":
        raise ValueError("post-hoc scenario cannot replace the original study decision")
    protected_before={name:sha(ROOT/name) for name in snapshot["baseline_artifacts"]}
    if protected_before!=snapshot["baseline_artifacts"]:
        raise ValueError("baseline artifact changed")
    assessment_hash=sha(assessment_path)
    run=json.loads((REPORT/"run.json").read_text(encoding="utf-8"))
    if run["status"]!="COMPLETED":
        raise ValueError("candidate scoring run incomplete")
    sources={"historical":[],"recent":[]}
    receipts=[]
    for item in run["completed"]:
        if item["phase"]!="oos_2026":
            continue
        source=item["outputs"]["test"]
        if sha(source["path"])!=source["sha256"]:
            raise ValueError("historical prediction changed")
        sources["historical"].append(pd.read_parquet(source["path"]))
        receipts.append({"window":"historical",**source})
    for source in assessment["recent_prediction_receipts"]:
        if sha(source["path"])!=source["sha256"]:
            raise ValueError("recent prediction changed")
        sources["recent"].append(pd.read_parquet(source["path"]))
        receipts.append({"window":"recent",**source})
    if any(len(parts)!=7 for parts in sources.values()):
        raise ValueError("all seven streams required in each fixed window")
    frames={window:pd.concat(parts,ignore_index=True) for window,parts in sources.items()}
    started=datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=2):
        result=scenario.run_portfolio_scenarios(frames,policy,
            progress=lambda step:print("SCENARIO "+json.dumps(step,separators=(",",":")),flush=True))
    if (sha(policy_path)!=snapshot["running_policy_sha256"]
            or any(sha(ROOT/name)!=value for name,value in protected_before.items())
            or sha(assessment_path)!=assessment_hash):
        raise ValueError("protected model, policy or assessment changed during scenarios")
    result.update(started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  sensitivity_registration_sha256=sha(registration_path),baseline_snapshot_sha256=sha(snapshot_path),
                  unchanged_preregistered_assessment_sha256=assessment_hash,
                  scenario_module_sha256=sha(scenario.__file__),scenario_script_sha256=sha(__file__),
                  frozen_policy_sha256=snapshot["running_policy_sha256"],
                  prediction_sources=receipts,threads=2,protected_artifacts_unchanged=True)
    output=REPORT/"portfolio_scenario.json"
    temporary=output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"scenario_count":result["scenario_count"],"path":str(output),"sha256":sha(output),
                      "classification":result["classification"],"preregistered_model_screen":"DO_NOT_PROMOTE",
                      "cannot_override_preregistered_model_screen":True}),flush=True)


if __name__=="__main__":
    main()

