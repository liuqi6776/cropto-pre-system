"""Isolated tail training, leakage, serialization and frozen-cache contracts."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from tools.argon4_forward import gen4_candidate as g


@pytest.fixture(scope="module")
def fitted_case():
    # Entirely synthetic offline test inputs; production files remain read-only.
    patch=pytest.MonkeyPatch()
    fold={"train_start":"2024-01-05T00:00:00Z","train_end":"2024-01-12T21:55:00Z",
          "validation_start":"2024-01-13T00:00:00Z","validation_end":"2024-01-16T21:55:00Z",
          "test_start":"2024-01-17T00:00:00Z","test_end":"2024-01-25T18:00:00Z",
          "valid_from":"2024-01-17T00:00:00Z","valid_until":"2024-02-01T00:00:00Z"}
    patch.setitem(g.frozen.FOLDS,"oos_2026",fold)
    rng=np.random.default_rng(573)
    index=pd.date_range("2024-01-05",periods=6000,freq="5min",tz="UTC")
    base=pd.DataFrame(rng.normal(size=(len(index),311)).astype(np.float32),index=index,columns=[f"b{i}" for i in range(311)])
    expert=pd.DataFrame(rng.normal(size=(len(index),65)).astype(np.float32),index=index,columns=[f"e{i}" for i in range(65)])
    raw=pd.DataFrame({"open":100.,"close":100*(1+rng.normal(0,.03,len(index)))},index=index)
    frames={"base":base,"expert":expert,"raw_target":raw,"index":index,"provenance":{"fixture":"SYNTHETIC_OFFLINE_TEST_ONLY"}}
    labels=g.frozen._label(raw,1)
    tr,va,te=g.frozen._split(index,labels,fold,1)
    with threadpool_limits(limits=1):
        base_family,z0=g.frozen._fit_family(base.iloc[tr],np.clip(labels[tr],-.008,.008),base.iloc[va])
        expert_family,zd=g.frozen._fit_family(expert.iloc[tr],np.clip(labels[tr],-.008,.008),expert.iloc[va],expert=True)
        package={"base":base_family,"expert":expert_family,"base_columns":list(base),"expert_columns":list(expert),
                 "weight":.15,"threshold_frozen":float(np.quantile(abs(z0),.92)),"threshold_reset":float(np.quantile(abs(z0+.15*zd),.92))}
        baseline={"package":package,"manifest_sha256":"1"*64,
                  "manifest":{"stream":"BTC_5m","phase":"oos_2026","model_sha256":"2"*64,
                              "model_filename":"BTC_5m.joblib","model_version":"SYNTHETIC_FIXTURE",
                              "training_code_sha256":g.BASE_CODE_SHA256}}
        candidate=g.fit_candidate("BTC_5m","oos_2026",frames,baseline)
        yield {"frames":frames,"baseline":baseline,"candidate":candidate,"tr":tr,"va":va,"te":te,"labels":labels}
    patch.undo()


def test_tail_targets_use_raw_train_quantiles_and_balance_each_class():
    y=np.linspace(-.1,.2,1000)
    targets=g.tail_training_targets(y)
    np.testing.assert_allclose([targets["q32"],targets["q68"]],np.quantile(y,[.32,.68]))
    assert targets["q68"]>.008
    for name in ("up","down"):
        item=targets[name]
        assert item["positive_count"]==320 and item["negative_count"]==680
        assert item["positive_weight"]==2.125
        assert np.all(item["sample_weight"][item["labels"]==0]==1)
        np.testing.assert_allclose(item["sample_weight"][item["labels"]==1].sum(),680)


@pytest.mark.parametrize("returns",[[0.,0.,0.],[1.,np.nan,2.],[],[[1.,2.]]])
def test_degenerate_tail_targets_fail_closed(returns):
    with pytest.raises(g.CandidateError):
        g.tail_training_targets(returns)


def test_train_only_scaler_and_prescribed_tail_hgb(fitted_case):
    c=fitted_case; tail=c["candidate"]["package"]["tail"]
    matrix=np.hstack([c["frames"]["base"].iloc[c["tr"]],c["frames"]["expert"].iloc[c["tr"]]])
    np.testing.assert_allclose(tail["scaler"].mean_,matrix.astype(np.float64).mean(axis=0),rtol=1e-12,atol=1e-12)
    assert tail["scaler"].n_samples_seen_==len(c["tr"])
    for side in ("up","down"):
        params=tail[side].get_params()
        assert all(params[key]==value for key,value in g.CONFIG["hgb"].items())
        assert params["min_samples_leaf"]==max(50,len(c["tr"])//100)
    expected=g.tail_training_targets(c["labels"][c["tr"]])
    assert c["candidate"]["manifest"]["tail_metadata"]["q68"]==expected["q68"]


def test_validation_normalizers_and_thresholds_reproduce_scores(fitted_case):
    c=fitted_case
    scores=g.predict_candidate_scores(c["candidate"],c["frames"]["base"].iloc[c["va"]],c["frames"]["expert"].iloc[c["va"]])
    for name in ("CURRENT_RESET","TAIL_CURRENT_RESET","DOC_BASE_RESET","TAIL_BASE_RESET"):
        assert abs(scores[name].mean())<1e-12
        np.testing.assert_allclose(scores[name].std(),1.,atol=1e-10)
        np.testing.assert_allclose(c["candidate"]["package"]["thresholds"][name],np.quantile(abs(scores[name]),.92),atol=1e-12)
    assert c["candidate"]["package"]["thresholds"]["CURRENT_FROZEN"]==c["baseline"]["package"]["threshold_frozen"]
    assert c["candidate"]["package"]["thresholds"]["TAIL_CURRENT_FROZEN"]==c["baseline"]["package"]["threshold_frozen"]


def test_test_labels_never_change_training_or_thresholds(fitted_case):
    c=fitted_case
    altered=dict(c["frames"],raw_target=c["frames"]["raw_target"].copy())
    altered["raw_target"].loc[altered["raw_target"].index>=pd.Timestamp("2024-01-17",tz="UTC"),"close"]=900.
    other=g.fit_candidate("BTC_5m","oos_2026",altered,c["baseline"])
    assert other["manifest"]["tail_metadata"]==c["candidate"]["manifest"]["tail_metadata"]
    assert other["package"]["normalizers"]==c["candidate"]["package"]["normalizers"]
    assert other["package"]["thresholds"]==c["candidate"]["package"]["thresholds"]
    assert other["manifest"]["rows"]["train"]==c["candidate"]["manifest"]["rows"]["train"]
    assert other["manifest"]["rows"]["validation"]==c["candidate"]["manifest"]["rows"]["validation"]
    assert other["manifest"]["rows"]["test"]["raw_returns_sha256"]!=c["candidate"]["manifest"]["rows"]["test"]["raw_returns_sha256"]


def test_test_selection_retains_ties_and_unsettled_labels(fitted_case):
    c=fitted_case
    altered=dict(c["frames"],raw_target=c["frames"]["raw_target"].copy())
    positions=c["te"]
    altered["raw_target"].iloc[positions[10]+1,altered["raw_target"].columns.get_loc("close")]=100.
    altered["raw_target"].iloc[positions[20]+1,altered["raw_target"].columns.get_loc("close")]=np.nan
    before=g.evaluate_rows(c["candidate"],c["frames"],"test")
    after=g.evaluate_rows(c["candidate"],altered,"test")
    assert before.selected.equals(after.selected) and before.score.equals(after.score)
    for name in g.VARIANTS:
        subset=after.loc[after.variant==name].reset_index(drop=True)
        assert subset.loc[10,"label_tie"]
        assert pd.isna(subset.loc[20,"forward_return"]) and not subset.loc[20,"settled"]
        np.testing.assert_array_equal(subset.selected,abs(subset.score)>=subset.threshold)


@pytest.fixture
def saved_candidate(fitted_case,tmp_path,monkeypatch):
    monkeypatch.setattr(g,"MODEL_ROOT",tmp_path/"candidate")
    monkeypatch.setattr(g,"load_baseline",lambda stream,phase,expected_reference=None:fitted_case["baseline"])
    pin=g.save_candidate_bundle(fitted_case["candidate"])
    loaded=g.load_candidate_bundle("BTC_5m","oos_2026",pin)
    return loaded,pin


def test_saved_candidate_and_one_row_parity(fitted_case,saved_candidate):
    loaded,_=saved_candidate
    c=fitted_case; positions=c["va"][::20]
    result=g.verify_score_parity(c["candidate"],loaded,c["frames"]["base"].iloc[positions],c["frames"]["expert"].iloc[positions])
    assert result["saved_batch"]=="EXACT" and result["one_row_batch"]=="PASS"
    assert set(loaded["package"])==set(c["candidate"]["package"])
    assert "base" not in loaded["package"] and "expert" not in loaded["package"]


def test_artifact_corruption_rejected_before_unpickle(saved_candidate,monkeypatch):
    _,pin=saved_candidate
    path=g.MODEL_ROOT/"oos_2026/BTC_5m.joblib"
    path.write_bytes(path.read_bytes()+b"corrupted")
    monkeypatch.setattr(g.joblib,"load",lambda *_:pytest.fail("corrupt artifact unpickled"))
    with pytest.raises(g.CandidateError,match="model hash"):
        g.load_candidate_bundle("BTC_5m","oos_2026",pin)


def test_candidate_code_pin_is_checked_before_unpickle(saved_candidate,monkeypatch):
    path=g.MODEL_ROOT/"oos_2026/BTC_5m.manifest.json"
    manifest=json.loads(path.read_text()); manifest["candidate_code_sha256"]="0"*64
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(g.joblib,"load",lambda *_:pytest.fail("wrong-code artifact unpickled"))
    with pytest.raises(g.CandidateError,match="code"):
        g.load_candidate_bundle("BTC_5m","oos_2026",g.frozen.sha256(path))


def test_baseline_root_and_existing_artifacts_cannot_be_overwritten(fitted_case,saved_candidate,monkeypatch):
    with pytest.raises(g.CandidateError,match="already exist"):
        g.save_candidate_bundle(fitted_case["candidate"])
    monkeypatch.setattr(g,"MODEL_ROOT",g.frozen.MODEL_ROOT)
    with pytest.raises(g.CandidateError,match="overlap"):
        g.save_candidate_bundle(fitted_case["candidate"])


def test_reordered_features_and_wrong_baseline_are_rejected(fitted_case):
    c=fitted_case
    with pytest.raises(g.CandidateError,match="schema"):
        g.predict_candidate_scores(c["candidate"],c["frames"]["base"].iloc[:2,::-1],c["frames"]["expert"].iloc[:2])
    baseline=deepcopy(c["baseline"]); baseline["manifest"]["stream"]="ETH_5m"
    with pytest.raises(g.CandidateError,match="baseline identity"):
        g.fit_candidate("BTC_5m","oos_2026",c["frames"],baseline)


def test_preregistration_required_before_training_io(tmp_path,monkeypatch):
    monkeypatch.setattr(g,"REPORT_ROOT",tmp_path)
    path=tmp_path/"preregistration.json"
    doc={"status":"FROZEN","study_id":g.CONFIG["study_id"],"primary_variant":"TAIL_CURRENT_RESET",
         "primary_comparator":"CURRENT_RESET","also_must_beat":"CURRENT_FROZEN","config_sha256":g.config_sha256()}
    path.write_text(json.dumps(doc))
    assert g.validate_preregistration(path)["sha256"]==g.frozen.sha256(path)
    doc["config_sha256"]="wrong"; path.write_text(json.dumps(doc))
    monkeypatch.setattr(g,"verify_historical_data",lambda:pytest.fail("training must not start"))
    with pytest.raises(g.CandidateError,match="preregistration"):
        g.train_all_candidate(preregistration_path=path)
    with pytest.raises(g.CandidateError,match="threads"):
        g.train_all_candidate(preregistration_path=path,threads=3)


def test_baseline_cache_must_match_labels_decisions_and_scores(fitted_case,tmp_path):
    c=fitted_case
    prediction=g.evaluate_rows(c["candidate"],c["frames"],"test")
    baseline=prediction.loc[prediction.variant=="CURRENT_FROZEN"].copy()
    baseline["variant"]="LOCAL_V4_FROZEN"; baseline["model_sha256"]=baseline["baseline_model_sha256"]
    path=tmp_path/"baseline.parquet"; baseline.to_parquet(path,index=False)
    assert g.assert_baseline_cache("BTC_5m",prediction,path)["all_decisions_and_labels_exact"]
    broken=prediction.copy(); position=broken.index[broken.variant=="CURRENT_FROZEN"][0]
    broken.loc[position,"selected"]=not broken.loc[position,"selected"]
    with pytest.raises(g.CandidateError,match="selected"):
        g.assert_baseline_cache("BTC_5m",broken,path)
    broken=prediction.copy(); broken.loc[position,"score"]+=.01
    with pytest.raises(g.CandidateError,match="score mismatch"):
        g.assert_baseline_cache("BTC_5m",broken,path)


def test_recent_bounds_use_decision_time_and_require_deployment(fitted_case):
    c=fitted_case
    with pytest.raises(g.CandidateError,match="deployment"):
        g.evaluate_rows(c["candidate"],c["frames"],"recent",start="2024-01-18T00:00:00Z",end="2024-01-18T00:10:00Z")
    deployed=dict(c["candidate"],manifest=dict(c["candidate"]["manifest"],phase="deployment_q4"))
    evaluated=g.evaluate_rows(deployed,c["frames"],"recent",start="2024-01-18T00:00:00Z",end="2024-01-18T00:10:00Z")
    assert evaluated.groupby("variant").size().eq(3).all()
    assert evaluated.decision_close_utc.min()==pd.Timestamp("2024-01-18T00:00:00Z")
    assert evaluated.bar_open_utc.min()==pd.Timestamp("2024-01-17T23:55:00Z")
    assert evaluated.availability_scope.eq("RETROSPECTIVE_PUBLICATION_ASSUMPTION_NOT_HISTORIC_TRADEABILITY").all()
    with pytest.raises(g.CandidateError,match="row indices"):
        g.evaluate_rows(deployed,c["frames"],"recent",row_indices=[2000,2000])
    with pytest.raises(g.CandidateError,match="validity"):
        g.evaluate_rows(deployed,c["frames"],"recent",row_indices=[0])


def test_recent_15m_explicit_indices_cannot_bypass_due_clock(fitted_case):
    c=fitted_case
    deployed=dict(c["candidate"],manifest=dict(c["candidate"]["manifest"],phase="deployment_q4",stream="BTC_15m"))
    position=c["frames"]["base"].index.get_loc(pd.Timestamp("2024-01-18T00:00:00Z"))
    with pytest.raises(g.CandidateError,match="due clock"):
        g.evaluate_rows(deployed,c["frames"],"recent",row_indices=[position])

