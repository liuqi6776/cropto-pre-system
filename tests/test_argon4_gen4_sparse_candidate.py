"""Phase-two runner contracts; real network internals have separate tests."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip("torch",reason="Run sparse GPU/CPU tests in the explicitly verified Torch environment")
from tools.argon4_forward import gen4_sparse_candidate as g


@pytest.fixture(scope="module")
def case():
    patch=pytest.MonkeyPatch()
    fold={"train_start":"2024-01-05T00:00:00Z","train_end":"2024-01-12T21:55:00Z",
          "validation_start":"2024-01-13T00:00:00Z","validation_end":"2024-01-16T21:55:00Z",
          "test_start":"2024-01-17T00:00:00Z","test_end":"2024-01-25T18:00:00Z",
          "valid_from":"2024-01-17T00:00:00Z","valid_until":"2024-02-01T00:00:00Z"}
    patch.setitem(g.frozen.FOLDS,"oos_2026",fold)
    rng=np.random.default_rng(3137)
    index=pd.date_range("2024-01-05",periods=6000,freq="5min",tz="UTC")
    base=pd.DataFrame(rng.normal(size=(len(index),311)).astype(np.float32),index=index,columns=[f"b{i}" for i in range(311)])
    expert=pd.DataFrame(rng.normal(size=(len(index),65)).astype(np.float32),index=index,columns=[f"e{i}" for i in range(65)])
    raw=pd.DataFrame({"open":100.,"close":100*(1+rng.normal(0,.01,len(index)))},index=index)
    frames={"base":base,"expert":expert,"raw_target":raw,"index":index,"provenance":{"fixture":"SYNTHETIC_OFFLINE_TEST"}}
    labels=g.frozen._label(raw,1)
    tr,va,te=g.frozen._split(index,labels,fold,1)
    val_current=base.iloc[va,0].to_numpy(float)+.15*expert.iloc[va,0].to_numpy(float)
    val_tail=expert.iloc[va,1].to_numpy(float)+.1*base.iloc[va,1].to_numpy(float)
    current_norm=g.frozen._normalizer(val_current); tail_norm=g.frozen._normalizer(val_tail)
    raw_mix=.85*g.frozen._z(val_current,current_norm)+.15*g.frozen._z(val_tail,tail_norm)
    mix_norm=g.frozen._normalizer(raw_mix)
    thresholds={v:1.2 for v in g.tail.VARIANTS}
    thresholds["CURRENT_RESET"]=float(np.quantile(abs(g.frozen._z(val_current,current_norm)),.92))
    thresholds["TAIL_CURRENT_RESET"]=float(np.quantile(abs(g.frozen._z(raw_mix,mix_norm)),.92))
    inherited={"package":{"base_columns":list(base),"expert_columns":list(expert),"thresholds":thresholds,
                           "tail":{"normalization":tail_norm},"normalizers":{"current":current_norm,"tail_current":mix_norm}},
               "manifest_sha256":"a"*64,"manifest":{"stream":"BTC_5m","phase":"oos_2026","fold":fold,
                    "model_sha256":"b"*64,"model_filename":"BTC_5m.joblib","candidate_code_sha256":g.TAIL_CODE_SHA256,
                    "config_sha256":"c"*64,"baseline_reference":{"stream":"BTC_5m","phase":"oos_2026","model_sha256":"d"*64}}}

    def tail_scores(bundle,b,e):
        package=bundle["package"]
        current=b.iloc[:,0].to_numpy(float)+.15*e.iloc[:,0].to_numpy(float)
        cz=g.frozen._z(current,package["normalizers"]["current"])
        tz=g.frozen._z(e.iloc[:,1].to_numpy(float)+.1*b.iloc[:,1].to_numpy(float),package["tail"]["normalization"])
        blended=g.frozen._z(.85*cz+.15*tz,package["normalizers"]["tail_current"])
        return {"CURRENT_FROZEN":current,"CURRENT_RESET":cz,"TAIL_CURRENT_RESET":blended,
                "DOC_BASE_RESET":cz,"TAIL_BASE_RESET":blended,"TAIL_CURRENT_FROZEN":blended}
    patch.setattr(g.tail,"predict_candidate_scores",tail_scores)
    patch.setattr(g.tail,"_tail_score",lambda tail,x:g.frozen._z(x[:,312].astype(float)+.1*x[:,1].astype(float),tail["normalization"]))
    calls=[]

    def fit_network(x,y,v,vy,seed,names,**kwargs):
        calls.append({"seed":seed,"y":y.copy(),"vy":vy.copy(),"x":x.copy(),"v":v.copy()})
        state={"format_version":1,"model_version":g.network.MODEL_VERSION,
               "classification":"SYNTHETIC_OFFLINE_FIXTURE","input_feature_names":list(names),
               "selected_indices":[0,1],"selected_feature_names":list(names[:2]),
               "scaler_mean":torch.zeros(2,dtype=torch.float64),"scaler_scale":torch.ones(2,dtype=torch.float64),
               "config":deepcopy(g.network.CONFIG),"seed":seed,"best_epoch":1,
               "model_state_dict":{"fixture_weight":torch.tensor([float(y.mean()),seed/1000],dtype=torch.float64)}}
        return state,{"best_epoch":1,"epochs_trained":1,"test_data_used":False,"fixture":True}
    def predict_network(state,x,**kwargs):
        w=state["model_state_dict"]["fixture_weight"].numpy()
        logit=np.asarray(x[:,0],float)*(.2+w[0])+np.asarray(x[:,1],float)*w[1]
        return 1/(1+np.exp(-logit))
    patch.setattr(g.network,"fit_sparse_expert",fit_network)
    patch.setattr(g.network,"predict_sparse_expert",predict_network)
    candidate=g.fit_sparse_candidate("BTC_5m","oos_2026",frames,inherited)
    yield {"frames":frames,"tail":inherited,"candidate":candidate,"tr":tr,"va":va,"te":te,"labels":labels,"calls":calls,"fold":fold}
    patch.undo()


def test_three_seeds_receive_only_frozen_train_validation_rows(case):
    assert [call["seed"] for call in case["calls"][:3]]==list(g.SEEDS)
    for call in case["calls"][:3]:
        np.testing.assert_array_equal(call["y"],case["labels"][case["tr"]]>0)
        np.testing.assert_array_equal(call["vy"],case["labels"][case["va"]]>0)
        np.testing.assert_array_equal(call["x"],case["frames"]["expert"].iloc[case["tr"]])
        np.testing.assert_array_equal(call["v"],case["frames"]["expert"].iloc[case["va"]])
    assert g.CONFIG["seed_prediction"]=="SIGMOID_UP_PROBABILITY"
    assert g.CONFIG["network_one_row_atol"]==1e-12


def test_seed_and_composite_validation_normalization_is_fixed(case):
    bundle=case["candidate"]; p=bundle["package"]; f=case["frames"]; va=case["va"]
    z=[]
    for seed in g.SEEDS:
        probability=g._probabilities(bundle["states"][str(seed)],f["expert"].iloc[va].to_numpy(),list(f["expert"]))
        expected=g.frozen._normalizer(probability)
        assert p["seed_normalizers"][str(seed)]==expected
        z.append(g.frozen._z(probability,expected))
    np.testing.assert_allclose(p["ensemble_normalizer"]["std"],np.mean(np.stack(z),axis=0).std()+1e-12,atol=1e-12)
    scores=g.predict_sparse_scores(bundle,f["base"].iloc[va],f["expert"].iloc[va])
    assert abs(scores["SPARSE_ADAPTIVE_RESET"].mean())<1e-12
    np.testing.assert_allclose(scores["SPARSE_ADAPTIVE_RESET"].std(),1.,atol=1e-10)
    assert p["thresholds"]["SPARSE_ADAPTIVE_RESET"]==pytest.approx(np.quantile(abs(scores["SPARSE_ADAPTIVE_RESET"]),.92))


@pytest.mark.parametrize("stream",g.TAIL_ONLY_STREAMS)
def test_fixed_tail_routes_train_no_network_and_are_bitwise_identical(case,monkeypatch,stream):
    inherited=deepcopy(case["tail"])
    inherited["manifest"]["stream"]=stream
    inherited["manifest"]["baseline_reference"]["stream"]=stream
    monkeypatch.setattr(g.network,"fit_sparse_expert",lambda *a,**k:pytest.fail("unused network trained"))
    monkeypatch.setattr(g.network,"predict_sparse_expert",lambda *a,**k:pytest.fail("unused network evaluated"))
    bundle=g.fit_sparse_candidate(stream,"oos_2026",case["frames"],inherited)
    assert bundle["package"]["skipped_by_fixed_route"] and not bundle["states"]
    frames=case["frames"]
    result=g.predict_sparse_scores(bundle,frames["base"].iloc[:20],frames["expert"].iloc[:20])
    assert np.array_equal(result["SPARSE_ADAPTIVE_RESET"],result["TAIL_CURRENT_RESET"])
    assert bundle["package"]["thresholds"]["SPARSE_ADAPTIVE_RESET"]==bundle["package"]["thresholds"]["TAIL_CURRENT_RESET"]


def test_changing_test_labels_cannot_change_fitted_sparse_recipe(case):
    frames=dict(case["frames"],raw_target=case["frames"]["raw_target"].copy())
    frames["raw_target"].loc[frames["raw_target"].index>=pd.Timestamp("2024-01-17",tz="UTC"),"close"]=900.
    changed=g.fit_sparse_candidate("BTC_5m","oos_2026",frames,case["tail"])
    assert changed["package"]==case["candidate"]["package"]
    assert changed["manifest"]["rows"]["train"]==case["candidate"]["manifest"]["rows"]["train"]
    assert changed["manifest"]["rows"]["validation"]==case["candidate"]["manifest"]["rows"]["validation"]
    assert changed["manifest"]["rows"]["test"]!=case["candidate"]["manifest"]["rows"]["test"]


def test_evaluation_preserves_three_controls_and_future_tie_selection(case):
    output=g.evaluate_rows(case["candidate"],case["frames"],"test")
    original=g.tail.evaluate_rows(case["tail"],case["frames"],"test")
    for variant in g.CONTROL_VARIANTS:
        old=original.loc[original.variant==variant].reset_index(drop=True)
        new=output.loc[output.variant==variant].reset_index(drop=True)
        pd.testing.assert_frame_equal(new,old)
    altered=dict(case["frames"],raw_target=case["frames"]["raw_target"].copy())
    pos=case["te"][10]
    altered["raw_target"].iloc[pos+1,altered["raw_target"].columns.get_loc("close")]=np.nan
    changed=g.evaluate_rows(case["candidate"],altered,"test")
    assert output.selected.equals(changed.selected) and output.score.equals(changed.score)
    assert changed.loc[changed.bar_open_utc==case["frames"]["index"][pos],"settled"].eq(False).all()


@pytest.fixture
def saved(case,tmp_path,monkeypatch):
    monkeypatch.setattr(g,"MODEL_ROOT",tmp_path/"sparse")
    monkeypatch.setattr(g,"load_tail",lambda stream,phase,expected_reference=None:case["tail"])
    pin=g.save_sparse_bundle(case["candidate"])
    loaded=g.load_sparse_bundle("BTC_5m","oos_2026",pin)
    return loaded,pin


def test_json_and_weights_only_reload_and_one_row_parity(case,saved):
    loaded,pin=saved
    assert len(loaded["states"])==3
    ix=case["va"][::30]
    result=g.verify_parity(case["candidate"],loaded,case["frames"]["base"].iloc[ix],case["frames"]["expert"].iloc[ix])
    assert result["saved_batch"]=="EXACT" and result["one_row_batch"]=="PASS"
    assert loaded["manifest"]["model_filename"].endswith(".bundle.json")
    assert "weights_only=True" in loaded["manifest"]["serialization"]


def test_seed_corruption_is_rejected_before_tensor_loader(saved,monkeypatch):
    _,pin=saved
    path=g.MODEL_ROOT/"oos_2026/BTC_5m.seed42.pt"
    path.write_bytes(path.read_bytes()+b"changed")
    monkeypatch.setattr(g.network,"load_sparse_expert",lambda *_:pytest.fail("unverified tensor loaded"))
    with pytest.raises(g.SparseCandidateError,match="tensor hash"):
        g.load_sparse_bundle("BTC_5m","oos_2026",pin)


def test_network_source_pin_is_rejected_before_loader(saved,monkeypatch):
    path=g.MODEL_ROOT/"oos_2026/BTC_5m.manifest.json"
    value=json.loads(path.read_text()); value["network_code_sha256"]="0"*64
    path.write_text(json.dumps(value))
    monkeypatch.setattr(g.network,"load_sparse_expert",lambda *_:pytest.fail("wrong-code tensor loaded"))
    with pytest.raises(g.SparseCandidateError,match="code"):
        g.load_sparse_bundle("BTC_5m","oos_2026",g.frozen.sha256(path))


def test_candidate_cannot_overwrite_existing_or_frozen_models(case,saved,monkeypatch):
    with pytest.raises(g.SparseCandidateError,match="exist"):
        g.save_sparse_bundle(case["candidate"])
    for root in (g.tail.MODEL_ROOT,g.frozen.MODEL_ROOT):
        monkeypatch.setattr(g,"MODEL_ROOT",root)
        with pytest.raises(g.SparseCandidateError,match="overlap"):
            g.save_sparse_bundle(case["candidate"])


def test_three_control_cache_requires_exact_decisions_and_labels(case,tmp_path):
    expected=g.tail.evaluate_rows(case["tail"],case["frames"],"test")
    path=tmp_path/"controls.parquet"; expected.to_parquet(path,index=False)
    actual=g.evaluate_rows(case["candidate"],case["frames"],"test")
    receipt=g.controls_parity(actual,path,g.frozen.sha256(path))
    assert receipt["scores_bitwise_equal"] and receipt["all_decisions_labels_thresholds_exact"]
    broken=actual.copy(); broken.loc[broken.index[0],"selected"]=not broken.loc[broken.index[0],"selected"]
    with pytest.raises(g.SparseCandidateError,match="selected"):
        g.controls_parity(broken,path,g.frozen.sha256(path))


def test_preregistration_blocks_training_before_data_io(tmp_path,monkeypatch):
    monkeypatch.setattr(g,"REPORT_ROOT",tmp_path)
    path=tmp_path/"preregistration.json"
    data={"status":"FROZEN","study_id":g.CONFIG["study_id"],"primary_variant":"SPARSE_ADAPTIVE_RESET",
          "primary_comparator":"CURRENT_RESET","also_must_beat":"CURRENT_FROZEN","config_sha256":g.config_sha256()}
    path.write_text(json.dumps(data))
    assert g.validate_preregistration(path)["sha256"]==g.frozen.sha256(path)
    data["config_sha256"]="wrong"; path.write_text(json.dumps(data))
    monkeypatch.setattr(g.tail,"verify_historical_data",lambda:pytest.fail("training started without matching preregistration"))
    with pytest.raises(g.SparseCandidateError,match="preregistration"):
        g.train_all_sparse(preregistration_path=path)
    with pytest.raises(g.SparseCandidateError,match="threads"):
        g.train_all_sparse(preregistration_path=path,threads=3)
    with pytest.raises(g.SparseCandidateError,match="threads"):
        g.train_all_sparse(preregistration_path=path,threads=1)


def test_existing_successful_run_receipt_cannot_be_replaced(tmp_path,monkeypatch):
    monkeypatch.setattr(g,"REPORT_ROOT",tmp_path)
    path=tmp_path/"run.json"
    original=b'{"status":"COMPLETED","immutable_receipt":true}'
    path.write_bytes(original)
    monkeypatch.setattr(g,"validate_preregistration",lambda *_:pytest.fail("existing run must be preserved before new work"))
    with pytest.raises(g.SparseCandidateError,match="run receipt already exists"):
        g.train_all_sparse(preregistration_path=tmp_path/"preregistration.json")
    assert path.read_bytes()==original


def test_recent_evaluation_requires_deployment_and_keeps_publication_label(case):
    with pytest.raises(g.tail.CandidateError,match="deployment"):
        g.evaluate_rows(case["candidate"],case["frames"],"recent",start="2024-01-18T00:00:00Z",end="2024-01-18T00:10:00Z")
    deployed=deepcopy(case["candidate"])
    deployed["manifest"]["phase"]="deployment_q4"
    deployed["tail"]["manifest"]["phase"]="deployment_q4"
    rows=g.evaluate_rows(deployed,case["frames"],"recent",start="2024-01-18T00:00:00Z",end="2024-01-18T00:10:00Z")
    assert rows.groupby("variant").size().eq(3).all()
    assert rows.decision_close_utc.min()==pd.Timestamp("2024-01-18T00:00:00Z")
    assert rows.availability_scope.eq("RETROSPECTIVE_PUBLICATION_ASSUMPTION_NOT_HISTORIC_TRADEABILITY").all()


def test_wrong_column_order_or_invalid_probability_fails(case,monkeypatch):
    f=case["frames"]
    with pytest.raises(g.SparseCandidateError,match="schema"):
        g.predict_sparse_scores(case["candidate"],f["base"].iloc[:2,::-1],f["expert"].iloc[:2])
    monkeypatch.setattr(g.network,"predict_sparse_expert",lambda *a,**k:np.array([np.nan,np.nan]))
    with pytest.raises(g.SparseCandidateError,match="probabilities"):
        g.predict_sparse_scores(case["candidate"],f["base"].iloc[:2],f["expert"].iloc[:2])

