"""Phase-two independent sparse candidate, with an ex-ante fixed stream route.

Production files and the frozen phase-one tail packages are never written.
Only five streams train neural experts; the two tail-only routes reuse phase-one
arrays and thresholds exactly. No route is selected from test/recent outcomes.
"""
from __future__ import annotations

from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from . import gen4_candidate as tail
from . import gen4_sparse_network as network

frozen=tail.frozen
ROOT=Path(__file__).resolve().parents[2]
MODEL_ROOT=ROOT/"research/argon4_gen4_sparse_candidate_20261106"
REPORT_ROOT=ROOT/"reports/argon4_gen4_sparse_candidate_20261106"
TAIL_RUN=ROOT/"reports/argon4_gen4_candidate_20261106/run.json"
TAIL_RUN_SHA256="c0353115ddd32b125d901a9320075acd1ff4854eb382dd5e4a700b722baaa273"
TAIL_CODE_SHA256="9d4e6114634b58f422ecc149fe438c8491cf8e6bc61541df8499663c63fcb72f"
CLASSIFICATION="LOCAL_311_PLUS_65_SPARSE_FIXED_ROUTE_PHASE2_RESEARCH_ONLY"
CONTROL_VARIANTS=("CURRENT_FROZEN","CURRENT_RESET","TAIL_CURRENT_RESET")
VARIANTS=(*CONTROL_VARIANTS,"SPARSE_ADAPTIVE_RESET")
SPARSE_STREAMS=("BTC_15m","ETH_5m","ETH_15m","SOL_5m","BTC_5m")
TAIL_ONLY_STREAMS=("SOL_15m","BNB_15m")
SEEDS=(42,456,789)


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


CONFIG={
    "study_id":"argon4_gen4_sparse_candidate_20261106","schema_version":1,
    "primary_variant":"SPARSE_ADAPTIVE_RESET","primary_comparator":"CURRENT_RESET",
    "also_must_beat":"CURRENT_FROZEN","diagnostic_comparator":"TAIL_CURRENT_RESET",
    "variants":list(VARIANTS),"sparse_streams":list(SPARSE_STREAMS),"tail_only_streams":list(TAIL_ONLY_STREAMS),
    "fixed_route_not_selected_on_local_test":True,"sparse_input":"ORDERED_INDEPENDENT_65_EXPERT_ONLY",
    "seeds":list(SEEDS),"seed_prediction":"SIGMOID_UP_PROBABILITY",
    "sparse_ensemble":"EACH_SEED_VALIDATION_Z_THEN_MEAN_THEN_VALIDATION_Z",
    "sparse_route_weights":{"CURRENT_VALIDATION_Z":.79,"TAIL_VALIDATION_Z":.15,"SPARSE_VALIDATION_Z":.06},
    "tail_route_weights":{"CURRENT_VALIDATION_Z":.85,"TAIL_VALIDATION_Z":.15},
    "tail_only_route":"EXACT_PHASE1_TAIL_CURRENT_RESET_NO_UNUSED_NETWORK_TRAINING",
    "composite_normalization_and_threshold":"VALIDATION_ONLY_Z_THEN_ORIGINAL_STREAM_ABS_QUANTILE",
    "training_labels":"DIRECTION_OF_FROZEN_RAW_RETURN_NONZERO_TRAIN_AND_VALIDATION",
    "network_config":json.loads(canonical(network.CONFIG)),"network_config_sha256":object_sha(network.CONFIG),
    "tail_run_sha256":TAIL_RUN_SHA256,"tail_code_sha256":TAIL_CODE_SHA256,
    "baseline_code_sha256":tail.BASE_CODE_SHA256,"data_manifest_sha256":tail.DATA_MANIFEST_SHA256,
    "control_score_atol":1e-12,"control_score_rtol":1e-12,
    "network_one_row_atol":1e-12,"network_one_row_rtol":1e-12,
    "inference_device":"cpu","maximum_cpu_threads":2,
}


class SparseCandidateError(ValueError):
    pass


def config_sha256():
    return object_sha(CONFIG)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def verify_frozen_dependencies():
    tail.verify_frozen_code()
    if frozen.sha256(tail.__file__)!=TAIL_CODE_SHA256 or frozen.sha256(TAIL_RUN)!=TAIL_RUN_SHA256:
        raise SparseCandidateError("frozen tail code or training receipt changed")
    if object_sha(network.CONFIG)!=CONFIG["network_config_sha256"]:
        raise SparseCandidateError("network configuration differs from candidate configuration")


def validate_preregistration(path):
    path=Path(path).resolve()
    if path!=(REPORT_ROOT/"preregistration.json").resolve() or path.is_symlink():
        raise SparseCandidateError("fixed sparse preregistration path required")
    value=json.loads(path.read_text(encoding="utf-8"))
    required={"status":"FROZEN","study_id":CONFIG["study_id"],"primary_variant":"SPARSE_ADAPTIVE_RESET",
              "primary_comparator":"CURRENT_RESET","also_must_beat":"CURRENT_FROZEN","config_sha256":config_sha256()}
    if any(value.get(key)!=expected for key,expected in required.items()):
        raise SparseCandidateError("frozen sparse preregistration/config mismatch")
    return {"path":str(path),"sha256":frozen.sha256(path),"document":value}


def _tail_reference(bundle):
    m=bundle["manifest"]
    return {"stream":m["stream"],"phase":m["phase"],"manifest_sha256":bundle["manifest_sha256"],
            "model_sha256":m["model_sha256"],"model_filename":m["model_filename"],
            "candidate_code_sha256":m["candidate_code_sha256"],"config_sha256":m["config_sha256"]}


def load_tail(stream,phase,expected_reference=None):
    verify_frozen_dependencies()
    if stream not in frozen.STREAMS or phase not in frozen.FOLDS:
        raise SparseCandidateError("unknown stream or phase")
    run=json.loads(TAIL_RUN.read_text(encoding="utf-8"))
    matches=[r for r in run["completed"] if r["stream"]==stream and r["phase"]==phase]
    if run["status"]!="COMPLETED" or len(matches)!=1:
        raise SparseCandidateError("exact completed frozen tail package required")
    bundle=tail.load_candidate_bundle(stream,phase,matches[0]["manifest_sha256"])
    reference=_tail_reference(bundle)
    if reference["model_sha256"]!=matches[0]["model_sha256"] or (expected_reference and reference!=expected_reference):
        raise SparseCandidateError("tail artifact reference mismatch")
    return bundle


def _probabilities(state,x,columns):
    result=np.asarray(network.predict_sparse_expert(state,np.ascontiguousarray(x,dtype=np.float32),
                       feature_names=columns,device="cpu"),dtype=np.float64)
    if result.shape!=(len(x),) or not np.isfinite(result).all() or np.any((result<0)|(result>1)):
        raise SparseCandidateError("network must return finite UP probabilities")
    return result


def _tail_z(bundle,base,expert):
    values=np.hstack([base.to_numpy(np.float32),expert.to_numpy(np.float32)])
    return tail._tail_score(bundle["package"]["tail"],values)


def fit_sparse_candidate(stream,phase,frames,tail_bundle=None,*,provenance=None,device=None):
    verify_frozen_dependencies()
    inherited=tail_bundle or load_tail(stream,phase)
    if inherited["manifest"]["stream"]!=stream or inherited["manifest"]["phase"]!=phase:
        raise SparseCandidateError("tail identity mismatch")
    base,expert=frames["base"],frames["expert"]
    if (list(base.columns)!=inherited["package"]["base_columns"]
            or list(expert.columns)!=inherited["package"]["expert_columns"]
            or len(expert.columns)!=65 or not base.index.equals(expert.index)
            or not base.index.equals(frames["raw_target"].index)):
        raise SparseCandidateError("frozen 311+65 feature schema mismatch")
    _,k,quantile=frozen.STREAMS[stream]
    labels=frozen._label(frames["raw_target"],k)
    tr,va,te=frozen._split(base.index,labels,frozen.FOLDS[phase],k)
    current=tail.predict_candidate_scores(inherited,base.iloc[va],expert.iloc[va])
    skipped=stream in TAIL_ONLY_STREAMS
    states,seed_metadata,seed_normalizers={},{},{}
    ensemble_normalizer,composite_normalizer=None,None
    if skipped:
        primary_threshold=inherited["package"]["thresholds"]["TAIL_CURRENT_RESET"]
    else:
        train_x=np.ascontiguousarray(expert.iloc[tr].to_numpy(np.float32)).copy()
        val_x=np.ascontiguousarray(expert.iloc[va].to_numpy(np.float32)).copy()
        seed_z=[]
        for seed in SEEDS:
            state,metadata=network.fit_sparse_expert(train_x,(labels[tr]>0).astype(np.int64),
                val_x,(labels[va]>0).astype(np.int64),seed,list(expert.columns),device=device,config=network.CONFIG)
            probability=_probabilities(state,val_x,list(expert.columns))
            normalizer=frozen._normalizer(probability)
            states[str(seed)]=state
            seed_metadata[str(seed)]=metadata
            seed_normalizers[str(seed)]=normalizer
            seed_z.append(frozen._z(probability,normalizer))
        average=np.mean(np.stack(seed_z),axis=0)
        ensemble_normalizer=frozen._normalizer(average)
        sparse_z=frozen._z(average,ensemble_normalizer)
        raw=.79*current["CURRENT_RESET"]+.15*_tail_z(inherited,base.iloc[va],expert.iloc[va])+.06*sparse_z
        composite_normalizer=frozen._normalizer(raw)
        composite=frozen._z(raw,composite_normalizer)
        primary_threshold=float(np.quantile(np.abs(composite),quantile))
    thresholds={key:inherited["package"]["thresholds"][key] for key in CONTROL_VARIANTS}
    thresholds["SPARSE_ADAPTIVE_RESET"]=primary_threshold
    package={"base_columns":list(base.columns),"expert_columns":list(expert.columns),"thresholds":thresholds,
             "tail_reference":_tail_reference(inherited),"baseline_reference":inherited["manifest"]["baseline_reference"],
             "skipped_by_fixed_route":skipped,"seed_normalizers":seed_normalizers,
             "ensemble_normalizer":ensemble_normalizer,"composite_normalizer":composite_normalizer,
             "config_sha256":config_sha256()}
    manifest={"stream":stream,"phase":phase,"classification":CLASSIFICATION,
              "purpose":"INDEPENDENT_LOCAL_TRAINABLE_SPARSE_NETWORK_NOT_AUTHOR_ARTIFACT",
              "model_version":f"local311_65_sparse_fixedroute_{phase}_v1","fold":dict(frozen.FOLDS[phase]),
              "config":CONFIG,"config_sha256":config_sha256(),"candidate_code_sha256":frozen.sha256(__file__),
              "network_code_sha256":frozen.sha256(network.__file__),"network_config_sha256":object_sha(network.CONFIG),
              "tail_code_sha256":TAIL_CODE_SHA256,"baseline_code_sha256":tail.BASE_CODE_SHA256,
              "torch_version":str(torch.__version__),"numpy_version":np.__version__,
              "data_manifest_sha256":tail.DATA_MANIFEST_SHA256,"feature_provenance":frames.get("provenance",{}),
              "run_provenance":provenance or {},"seed_training_metadata":seed_metadata,
              "rows":{name:tail._row_receipt(base.index,labels,ix) for name,ix in (("train",tr),("validation",va),("test",te))},
              "train_last_label_end":str(base.index[tr[-1]]+pd.Timedelta(minutes=5*(k+1))),
              "validation_last_label_end":str(base.index[va[-1]]+pd.Timedelta(minutes=5*(k+1))),
              "live_enabled":False,"production_published":False,**package}
    return {"package":package,"manifest":manifest,"tail":inherited,"states":states}


def predict_sparse_scores(bundle,base_frame,expert_frame):
    package=bundle["package"]
    if list(base_frame.columns)!=package["base_columns"] or list(expert_frame.columns)!=package["expert_columns"]:
        raise SparseCandidateError("ordered sparse feature schema mismatch")
    inherited=tail.predict_candidate_scores(bundle["tail"],base_frame,expert_frame)
    result={name:inherited[name] for name in CONTROL_VARIANTS}
    if package["skipped_by_fixed_route"]:
        result["SPARSE_ADAPTIVE_RESET"]=inherited["TAIL_CURRENT_RESET"]
        return result
    seed_z=[]
    x=expert_frame.to_numpy(np.float32)
    for seed in SEEDS:
        probability=_probabilities(bundle["states"][str(seed)],x,package["expert_columns"])
        seed_z.append(frozen._z(probability,package["seed_normalizers"][str(seed)]))
    sparse=frozen._z(np.mean(np.stack(seed_z),axis=0),package["ensemble_normalizer"])
    raw=.79*inherited["CURRENT_RESET"]+.15*_tail_z(bundle["tail"],base_frame,expert_frame)+.06*sparse
    result["SPARSE_ADAPTIVE_RESET"]=frozen._z(raw,package["composite_normalizer"])
    return result


def _folder(stream,phase):
    if stream not in frozen.STREAMS or phase not in frozen.FOLDS:
        raise SparseCandidateError("unknown candidate identity")
    root=MODEL_ROOT.resolve()
    if any(root==p.resolve() or root.is_relative_to(p.resolve()) for p in (frozen.MODEL_ROOT,tail.MODEL_ROOT)):
        raise SparseCandidateError("sparse output cannot overlap frozen artifacts")
    folder=MODEL_ROOT/phase
    if folder.is_symlink() or not folder.resolve().is_relative_to(root):
        raise SparseCandidateError("untrusted sparse model folder")
    return folder


def save_sparse_bundle(bundle):
    manifest=dict(bundle["manifest"])
    stream,phase=manifest["stream"],manifest["phase"]
    folder=_folder(stream,phase)
    folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/f"{stream}.manifest.json"
    bundle_path=folder/f"{stream}.bundle.json"
    seed_paths={str(seed):folder/f"{stream}.seed{seed}.pt" for seed in SEEDS} if not bundle["package"]["skipped_by_fixed_route"] else {}
    if any(path.exists() for path in (manifest_path,bundle_path,*seed_paths.values())):
        raise SparseCandidateError("sparse artifacts exist; successful models cannot be overwritten")
    network_files={}
    for seed,path in seed_paths.items():
        network.save_sparse_expert(bundle["states"][seed],path)
        network_files[seed]={"filename":path.name,"sha256":frozen.sha256(path)}
    package={**bundle["package"],"network_files":network_files}
    _write_json(bundle_path,package)
    manifest.update(network_files=network_files,model_filename=bundle_path.name,
                    model_sha256=frozen.sha256(bundle_path),created_at=_now(),
                    serialization="JSON bundle plus locally generated torch tensor state; weights_only=True")
    _write_json(manifest_path,manifest)
    pin=frozen.sha256(manifest_path)
    manifest_path.with_suffix(".json.sha256").write_text(pin+"\n",encoding="ascii")
    return pin


def load_sparse_bundle(stream,phase,expected_manifest_sha256):
    verify_frozen_dependencies()
    folder=_folder(stream,phase)
    manifest_path=folder/f"{stream}.manifest.json"
    if manifest_path.is_symlink() or not expected_manifest_sha256 or frozen.sha256(manifest_path)!=expected_manifest_sha256:
        raise SparseCandidateError("sparse manifest hash mismatch")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    expected={"stream":stream,"phase":phase,"classification":CLASSIFICATION,
              "candidate_code_sha256":frozen.sha256(__file__),"network_code_sha256":frozen.sha256(network.__file__),
              "network_config_sha256":object_sha(network.CONFIG),"config_sha256":config_sha256(),
              "torch_version":str(torch.__version__),"numpy_version":np.__version__}
    if any(manifest.get(key)!=value for key,value in expected.items()) or canonical(manifest.get("config"))!=canonical(CONFIG):
        raise SparseCandidateError("sparse identity/code/config/runtime mismatch")
    if manifest.get("model_filename")!=f"{stream}.bundle.json":
        raise SparseCandidateError("sparse bundle filename not allowed")
    path=folder/manifest["model_filename"]
    if path.is_symlink() or frozen.sha256(path)!=manifest["model_sha256"]:
        raise SparseCandidateError("sparse bundle hash mismatch")
    package=json.loads(path.read_text(encoding="utf-8"))
    for key in ("base_columns","expert_columns","thresholds","tail_reference","baseline_reference","skipped_by_fixed_route",
                "seed_normalizers","ensemble_normalizer","composite_normalizer","config_sha256","network_files"):
        if package.get(key)!=manifest.get(key):
            raise SparseCandidateError("sparse package and manifest differ")
    if package["skipped_by_fixed_route"]!=(stream in TAIL_ONLY_STREAMS):
        raise SparseCandidateError("fixed sparse routing changed")
    expected_seeds=set() if package["skipped_by_fixed_route"] else {str(seed) for seed in SEEDS}
    if set(package["network_files"])!=expected_seeds:
        raise SparseCandidateError("sparse seed package incomplete")
    inherited=load_tail(stream,phase,package["tail_reference"])
    if inherited["manifest"]["baseline_reference"]!=package["baseline_reference"]:
        raise SparseCandidateError("transitive baseline pin mismatch")
    states={}
    for seed,item in package["network_files"].items():
        if item["filename"]!=f"{stream}.seed{seed}.pt":
            raise SparseCandidateError("sparse seed filename not allowed")
        path=folder/item["filename"]
        if path.is_symlink() or frozen.sha256(path)!=item["sha256"]:
            raise SparseCandidateError("sparse seed tensor hash mismatch")
        state=network.load_sparse_expert(path)
        if (state["seed"]!=int(seed) or list(state["input_feature_names"])!=package["expert_columns"]
                or canonical(state["config"])!=canonical(network.CONFIG)):
            raise SparseCandidateError("sparse state identity or input schema mismatch")
        states[seed]=state
    return {"package":package,"manifest":manifest,"manifest_sha256":expected_manifest_sha256,"tail":inherited,"states":states}


def evaluate_rows(bundle,frames,split="test",*,start=None,end=None,row_indices=None):
    original=tail.evaluate_rows(bundle["tail"],frames,split,start=start,end=end,row_indices=row_indices)
    if original.empty:
        return original
    controls=original.loc[original.variant.isin(CONTROL_VARIANTS)].copy()
    primary=controls.loc[controls.variant=="CURRENT_RESET"].copy()
    positions=frames["base"].index.get_indexer(pd.DatetimeIndex(primary.bar_open_utc))
    if np.any(positions<0):
        raise SparseCandidateError("evaluated rows missing from feature frame")
    score=predict_sparse_scores(bundle,frames["base"].iloc[positions],frames["expert"].iloc[positions])["SPARSE_ADAPTIVE_RESET"]
    threshold=bundle["package"]["thresholds"]["SPARSE_ADAPTIVE_RESET"]
    selected=np.abs(score)>=threshold
    returns=primary.forward_return.to_numpy(float)
    settled=selected & np.isfinite(returns)
    primary["variant"]="SPARSE_ADAPTIVE_RESET"
    primary["score"],primary["threshold"]=score,threshold
    primary["selected"],primary["settled"]=selected,settled
    primary["side"]=np.where(score>0,"UP","DOWN")
    primary["win"]=settled & ((score>0)==(returns>=0))
    primary["model_sha256"]=bundle["manifest"].get("model_sha256")
    primary["manifest_sha256"]=bundle.get("manifest_sha256")
    primary["classification"]=CLASSIFICATION
    return pd.concat([controls,primary],ignore_index=True)


def controls_parity(predictions,source_path,expected_sha256):
    path=Path(source_path)
    if frozen.sha256(path)!=expected_sha256:
        raise SparseCandidateError("frozen control prediction file changed")
    original=pd.read_parquet(path)
    maximum=0.
    bitwise=True
    for variant in CONTROL_VARIANTS:
        old=original.loc[original.variant==variant].reset_index(drop=True)
        new=predictions.loc[predictions.variant==variant].reset_index(drop=True)
        if len(old)!=len(new) or not len(old):
            raise SparseCandidateError("control prediction row count changed")
        for name in ("stream","phase","split","variant","bar_open_utc","decision_close_utc","threshold","selected","side","label_tie","settled","win"):
            if not old[name].equals(new[name]):
                raise SparseCandidateError("control prediction differs: "+name)
        if not np.array_equal(old.forward_return.to_numpy(),new.forward_return.to_numpy(),equal_nan=True):
            raise SparseCandidateError("control labels changed")
        if not np.array_equal(old.model_sha256.to_numpy(),new.model_sha256.to_numpy()):
            raise SparseCandidateError("control artifact identity changed")
        if not np.allclose(old.score,new.score,atol=CONFIG["control_score_atol"],rtol=CONFIG["control_score_rtol"]):
            raise SparseCandidateError("control scores changed")
        maximum=max(maximum,float(np.max(np.abs(old.score.to_numpy()-new.score.to_numpy()))))
        bitwise=bitwise and np.array_equal(old.score.to_numpy(),new.score.to_numpy())
    return {"status":"PASS","path":str(path),"sha256":expected_sha256,"all_decisions_labels_thresholds_exact":True,
            "scores_bitwise_equal":bool(bitwise),"max_abs_score_difference":maximum}


def verify_parity(before,after,base,expert):
    a=predict_sparse_scores(before,base,expert)
    b=predict_sparse_scores(after,base,expert)
    for variant in VARIANTS:
        if not np.array_equal(a[variant],b[variant]):
            raise SparseCandidateError("sparse saved/reloaded batch scores differ")
    maximum=0.
    for index in sorted({0,len(base)//2,len(base)-1}):
        single=predict_sparse_scores(after,base.iloc[[index]],expert.iloc[[index]])
        for variant in VARIANTS:
            atol=CONFIG["network_one_row_atol"] if variant=="SPARSE_ADAPTIVE_RESET" else CONFIG["control_score_atol"]
            rtol=CONFIG["network_one_row_rtol"] if variant=="SPARSE_ADAPTIVE_RESET" else CONFIG["control_score_rtol"]
            if not np.allclose(single[variant],b[variant][[index]],atol=atol,rtol=rtol):
                raise SparseCandidateError("sparse one-row/batch score mismatch")
            threshold=after["package"]["thresholds"][variant]
            if (not np.array_equal(abs(single[variant])>=threshold,abs(b[variant][[index]])>=threshold)
                    or not np.array_equal(single[variant]>0,b[variant][[index]]>0)):
                raise SparseCandidateError("sparse one-row/batch selected side mismatch")
            maximum=max(maximum,float(abs(single[variant][0]-b[variant][index])))
    return {"saved_batch":"EXACT","one_row_batch":"PASS","max_abs_one_row_difference":maximum}


def _recent_inputs(document):
    spec=document["recent_research"]
    manifest_path=(ROOT/spec["snapshot_manifest"]).resolve()
    if not manifest_path.is_relative_to(ROOT) or frozen.sha256(manifest_path)!=spec["sha256"]:
        raise SparseCandidateError("recent snapshot manifest mismatch")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    root=manifest_path.parent/"raw_data"
    for item in manifest["files"]:
        path=(root/item["relative_path"]).resolve()
        if not path.is_relative_to(root.resolve()) or frozen.sha256(path)!=item["copy_sha256"]:
            raise SparseCandidateError("recent snapshot input changed")
    return spec,root


def train_all_sparse(*,preregistration_path,threads=2,phases=("oos_2026","deployment_q4"),streams=None,device=None):
    if threads!=2 or not phases or any(p not in frozen.FOLDS for p in phases) or len(set(phases))!=len(phases):
        raise SparseCandidateError("sparse training threads/phases invalid")
    selected=list(streams or frozen.STREAMS)
    if not selected or len(set(selected))!=len(selected) or any(s not in frozen.STREAMS for s in selected):
        raise SparseCandidateError("sparse training streams invalid")
    if (REPORT_ROOT/"run.json").exists():
        raise SparseCandidateError("study run receipt already exists; refusing to overwrite research evidence")
    prereg=validate_preregistration(preregistration_path)
    verify_frozen_dependencies()
    tail.verify_historical_data()
    spec,recent_root=_recent_inputs(prereg["document"])
    inherited_run=json.loads(TAIL_RUN.read_text(encoding="utf-8"))
    old_outputs={(x["stream"],x["phase"]):x["outputs"] for x in inherited_run["completed"]}
    old_assessment=json.loads((tail.REPORT_ROOT/"research_assessment.json").read_text(encoding="utf-8"))
    recent_controls={x["stream"]:x for x in old_assessment["recent_prediction_receipts"]}
    policy_path=ROOT/"config/argon4_forward_policy.json"
    policy_hash=frozen.sha256(policy_path)
    provenance={"preregistration_sha256":prereg["sha256"],"preregistration_path":prereg["path"],
                "tail_run_sha256":TAIL_RUN_SHA256,"paper_policy_sha256":policy_hash,
                "recent_snapshot_manifest_sha256":spec["sha256"],
                "training_script_sha256":frozen.sha256(ROOT/"scripts/train_argon4_gen4_sparse_candidate.py")}
    run={"status":"TRAINING","started_at":_now(),"config":CONFIG,"config_sha256":config_sha256(),
         "candidate_code_sha256":frozen.sha256(__file__),"network_code_sha256":frozen.sha256(network.__file__),
         "threads":threads,"device_requested":device,"streams":selected,"phases":list(phases),
         "provenance":provenance,"completed":[],"production_published":False,"live_enabled":False}
    _write_json(REPORT_ROOT/"run.json",run)
    torch.set_num_threads(threads)
    try:
        with threadpool_limits(limits=threads):
            for symbol in frozen.SYMBOLS:
                own=[s for s in selected if frozen.STREAMS[s][0]==symbol]
                if not own:
                    continue
                print("FEATURES "+symbol,flush=True)
                frames=frozen.build_feature_frames(frozen.DATA_ROOT,symbol)
                recent_frames=None
                for stream in own:
                    for phase in phases:
                        print(f"FIT_SPARSE {stream} {phase} route={'TAIL_ONLY' if stream in TAIL_ONLY_STREAMS else 'TRIPLE'}",flush=True)
                        inherited=load_tail(stream,phase)
                        candidate=fit_sparse_candidate(stream,phase,frames,inherited,provenance=provenance,device=device)
                        pin=save_sparse_bundle(candidate)
                        restored=load_sparse_bundle(stream,phase,pin)
                        labels=frozen._label(frames["raw_target"],frozen.STREAMS[stream][1])
                        _,va,_=frozen._split(frames["base"].index,labels,frozen.FOLDS[phase],frozen.STREAMS[stream][1])
                        ix=va[np.linspace(0,len(va)-1,min(64,len(va)),dtype=int)]
                        parity=verify_parity(candidate,restored,frames["base"].iloc[ix],frames["expert"].iloc[ix])
                        outputs,checks={},{}
                        for split in ("validation","test","recent"):
                            if split=="recent":
                                if phase!="deployment_q4":
                                    continue
                                if recent_frames is None:
                                    recent_frames=frozen.build_feature_frames(recent_root,symbol,as_of=None)
                                    if frozen._timestamp(spec["decision_start_utc"])-recent_frames["index"][0]<pd.Timedelta(days=15):
                                        raise SparseCandidateError("recent input lacks fixed fifteen-day warmup")
                                predictions=evaluate_rows(restored,recent_frames,split,start=spec["decision_start_utc"],end=spec["decision_end_utc"])
                                control=recent_controls[stream]
                            else:
                                predictions=evaluate_rows(restored,frames,split)
                                if predictions.empty:
                                    continue
                                control=old_outputs[(stream,phase)][split]
                            if predictions.empty:
                                raise SparseCandidateError("empty required sparse prediction window")
                            checks[split]=controls_parity(predictions,control["path"],control["sha256"])
                            path=REPORT_ROOT/phase/f"{stream}_{split}_predictions.parquet"
                            path.parent.mkdir(parents=True,exist_ok=True)
                            if path.exists():
                                raise SparseCandidateError("successful sparse prediction output already exists")
                            predictions.to_parquet(path,index=False)
                            outputs[split]={"path":str(path),"sha256":frozen.sha256(path),"rows":len(predictions)}
                            del predictions
                        item={"stream":stream,"phase":phase,"manifest_sha256":pin,"model_sha256":restored["manifest"]["model_sha256"],
                              "skipped_by_fixed_route":restored["package"]["skipped_by_fixed_route"],
                              "seed_training_metadata":restored["manifest"]["seed_training_metadata"],
                              "parity":parity,"controls_parity":{"status":"PASS","splits":checks},"outputs":outputs}
                        run["completed"].append(item)
                        _write_json(REPORT_ROOT/"run.json",run)
                        print("COMPLETE "+canonical({"stream":stream,"phase":phase,"manifest_sha256":pin,"outputs":outputs}),flush=True)
                        del inherited,candidate,restored
                del frames,recent_frames
                gc.collect()
        if frozen.sha256(policy_path)!=policy_hash or frozen.sha256(prereg["path"])!=prereg["sha256"]:
            raise SparseCandidateError("policy or preregistration changed during sparse run")
        verify_frozen_dependencies()
        run.update(status="COMPLETED",completed_at=_now())
    except Exception as exc:
        run.update(status="FAILED",error=type(exc).__name__,failed_at=_now())
        _write_json(REPORT_ROOT/"run.json",run)
        raise
    _write_json(REPORT_ROOT/"run.json",run)
    return run

