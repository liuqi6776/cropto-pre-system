"""Saved-model identity, split isolation and feature causality checks."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from tools.argon4_forward import model as m


def test_validation_weight_selection_has_no_test_argument():
    base = np.array([-.8, .2, -.3, 1.3, -.5, .8, 2., -2.])
    expert = np.array([1., -.1, -1., .5, 2., -.5, 1., 1.])
    y = np.array([-1., 1., -1., 1., 1., -1., 1., -1.])
    weight, threshold, rows = m.select_weight(base, expert, y, .6)
    best = max(rows, key=lambda row: row['accuracy'])
    assert weight == best['weight'] and threshold == best['threshold_reset']
    assert {row['weight'] for row in rows} == set(m.WEIGHTS)


def test_test_and_online_selection_do_not_filter_future_ties():
    idx = pd.date_range('2024-01-01', '2024-02-01', freq='5min', tz='UTC')
    y = np.sin(np.arange(len(idx)) / 8) * .001
    fold = {'train_start':'2024-01-05T00:00:00Z','train_end':'2024-01-20T21:55:00Z',
            'validation_start':'2024-01-21T00:00:00Z','validation_end':'2024-01-25T21:55:00Z',
            'test_start':'2024-01-26T00:00:00Z','test_end':'2024-01-30T22:00:00Z'}
    y[idx >= pd.Timestamp('2024-01-26',tz='UTC')] = 0
    tr, va, te = m._split(idx, y, fold, 1)
    assert len(te) > 1000 and np.all(y[te] == 0)
    assert idx[tr[-1]] + pd.Timedelta(minutes=10) < pd.Timestamp(fold['validation_start'])
    assert idx[va[-1]] + pd.Timedelta(minutes=10) < pd.Timestamp(fold['test_start'])
    y[te] = np.nan
    assert np.array_equal(te, m._split(idx, y, fold, 1)[2])


@pytest.fixture
def saved_small_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'MODEL_ROOT', tmp_path)
    rng = np.random.default_rng(42)
    x, xd = rng.normal(size=(400,3)).astype('float32'), rng.normal(size=(400,2)).astype('float32')
    y = (x[:,0] * .001 + rng.normal(0,.001,400))
    base, z0 = m._fit_family(x[:300],y[:300],x[300:])
    expert, zd = m._fit_family(xd[:300],y[:300],xd[300:],expert=True)
    package = {'base':base,'expert':expert,'base_columns':['a','b','c'],
               'expert_columns':['d','e'],'weight':.15,'threshold_frozen':.5,'threshold_reset':.6}
    metadata = {'stream':'BTC_5m','phase':'deployment_q4','classification':m.CLASSIFICATION,
                'model_version':'fixture','fold':m.FOLDS['deployment_q4'],
                'base_columns':package['base_columns'],'expert_columns':package['expert_columns'],
                'sklearn_version':m.sklearn.__version__}
    digest = m._save_stream(tmp_path/'deployment_q4','BTC_5m',package,metadata)
    loaded = m.load_model_bundle('BTC_5m', expected_manifest_sha256=digest)
    index = pd.date_range('2026-11-06T00:00:00Z',periods=100,freq='5min')
    return loaded, pd.DataFrame(x[300:],index=index,columns=package['base_columns']), pd.DataFrame(xd[300:],index=index,columns=package['expert_columns']), z0, zd


def test_serialized_bundle_reproduces_scores(saved_small_bundle):
    loaded, base, expert, z0, zd = saved_small_bundle
    actual = m.predict_scores(loaded,base,expert)
    np.testing.assert_allclose(actual['BASE'],z0,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(actual['LOCAL_V4_FROZEN'],z0 + .15*zd,rtol=1e-12,atol=1e-12)
    single=m.predict_scores(loaded,base.iloc[:1],expert.iloc[:1])
    for key in single:
        np.testing.assert_allclose(single[key],actual[key][:1],rtol=1e-12,atol=1e-12)


def test_ridge_normalization_preserves_batch_shape_parity():
    rng=np.random.default_rng(184)
    x=rng.normal(size=(600,311)).astype('float32')
    y=rng.normal(size=600)*.0001
    family,expected=m._fit_family(x[:400],y[:400],x[400:])
    assert family['ridge'].coef_.dtype==np.float64
    for count in (1,64,200):
        actual=m._score_family(family,x[400:400+count])
        np.testing.assert_allclose(actual,expected[:count],rtol=1e-12,atol=1e-12)


def test_model_load_rejects_corruption_before_unpickle(saved_small_bundle, monkeypatch):
    path = m.MODEL_ROOT/'deployment_q4/BTC_5m.joblib'
    path.write_bytes(path.read_bytes()+b'altered')
    monkeypatch.setattr(m.joblib,'load',lambda *_:pytest.fail('untrusted pickle was loaded'))
    with pytest.raises(m.ModelInputError,match='hash'):
        m.load_model_bundle('BTC_5m')


def test_manifest_identity_and_filename_are_locked(saved_small_bundle, monkeypatch):
    path = m.MODEL_ROOT/'deployment_q4/BTC_5m.manifest.json'
    payload=json.loads(path.read_text()); payload['model_filename']='../../elsewhere.joblib'
    m._write_json(path,payload)
    digest=m.sha256(path)
    monkeypatch.setattr(m.joblib,'load',lambda *_:pytest.fail('untrusted pickle was loaded'))
    with pytest.raises(m.ModelInputError,match='filename'):
        m.load_model_bundle('BTC_5m',expected_manifest_sha256=digest)
    with pytest.raises(m.ModelInputError,match='unknown'):
        m.load_model_bundle('../BTC_5m')


def test_online_only_accepts_ordered_closed_fresh_features(saved_small_bundle):
    loaded,base,expert,_,_=saved_small_bundle
    base,expert=base.iloc[:1],expert.iloc[:1]
    prediction=m.predict_stream('BTC_5m',base,expert,'2026-11-06T00:05:04Z',bundle=loaded)
    assert prediction['model_sha256']==loaded['manifest']['model_sha256']
    assert prediction['side'] in ('UP','DOWN')
    assert prediction['classification']==m.CLASSIFICATION
    with pytest.raises(m.ModelInputError,match='unclosed'):
        m.predict_stream('BTC_5m',base,expert,'2026-11-06T00:04:59Z',bundle=loaded)
    with pytest.raises(m.ModelInputError,match='too late'):
        m.predict_stream('BTC_5m',base,expert,'2026-11-06T00:08:00Z',bundle=loaded)
    with pytest.raises(m.ModelInputError,match='schema'):
        m.predict_stream('BTC_5m',base[['c','b','a']],expert,'2026-11-06T00:05:01Z',bundle=loaded)


@pytest.fixture
def raw_fixture(tmp_path, monkeypatch):
    # All generated prices are solely an offline unit-test fixture.
    monkeypatch.setattr(m,'ROOT',tmp_path)
    root=tmp_path/'raw_data'; (root/'intraday').mkdir(parents=True); (root/'auxiliary').mkdir()
    idx=pd.date_range('2026-01-01',periods=2000,freq='5min',tz='UTC')
    for n,symbol in enumerate(m.SYMBOLS):
        price=(100 + np.arange(len(idx))*.002 + np.sin(np.arange(len(idx))/20))*(n+1)
        df=pd.DataFrame({'open_time':idx,'open':price,'high':price+1,'low':price-1,
                         'close':price+.05,'volume':10.,'quote_volume':1000.,'count':20.,'taker_buy_quote_volume':520.})
        df.to_parquet(root/'intraday'/f'{symbol}_5m_futures.parquet',index=False)
        dates=pd.date_range('2025-12-01','2026-01-08',freq='8h',tz='UTC')
        pd.DataFrame({'date':dates,'funding_rate':.0001}).to_parquet(root/'auxiliary'/f'{symbol}_8h_funding.parquet',index=False)
        hours=pd.date_range('2025-12-01','2026-01-08',freq='h',tz='UTC')
        hourly=pd.DataFrame({'date':hours,'open':100.,'high':101.,'low':99.,'close':100.,'volume':0.})
        for kind in ('mark','futures'):
            hourly.to_parquet(root/'auxiliary'/f'{symbol}_1h_{kind}.parquet',index=False)
    bidx=pd.date_range('2025-12-01','2026-01-08',freq='4h',tz='UTC')
    pd.DataFrame({s:.001 for s in m.SYMBOLS[:3]},index=bidx).to_parquet(root/'auxiliary/binance_basis_4h.parquet')
    days=pd.date_range('2025-12-01','2026-01-08',freq='D',tz='UTC')
    dv=pd.concat([pd.DataFrame({'bucket_utc':days,'symbol':coin,'open':50.,'high':51.,'low':49.,'close':50.}) for coin in ('BTC','ETH')],ignore_index=True)
    dv.to_parquet(root/'auxiliary/deribit_dvol_daily_2021_2026.parquet',index=False)
    return root,idx


def test_feature_prefix_invariance_and_no_future_label_dependency(raw_fixture):
    root,idx=raw_fixture
    earlier=idx[1800]+pd.Timedelta(minutes=5)
    prefix=m.build_feature_frames(root,'BTCUSDT',as_of=earlier)
    full=m.build_feature_frames(root,'BTCUSDT',as_of=idx[-1]+pd.Timedelta(minutes=5))
    assert prefix['base'].shape==(1801,311) and prefix['expert'].shape==(1801,65)
    np.testing.assert_allclose(prefix['base'],full['base'].iloc[:1801],atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(prefix['expert'],full['expert'].iloc[:1801],atol=1e-6,rtol=1e-6)
    assert all(v['availability_policy']=='HISTORICAL_PUBLICATION_ASSUMPTION' for v in prefix['provenance'].values())


def test_online_first_received_at_excludes_not_yet_received_bar(raw_fixture):
    root,idx=raw_fixture
    cutoff=idx[-1]+pd.Timedelta(minutes=5)
    for symbol in m.SYMBOLS:
        p=root/'intraday'/f'{symbol}_5m_futures.parquet'; frame=pd.read_parquet(p)
        frame['first_received_at']=pd.to_datetime(frame.open_time,utc=True)+pd.Timedelta(minutes=5)
        frame.loc[len(frame)-1,'first_received_at']=cutoff+pd.Timedelta(seconds=3)
        frame.to_parquet(p,index=False)
    parsed=m.build_feature_frames(root,'BTCUSDT',as_of=cutoff,tail_rows=1)
    assert parsed['index'][-1]==idx[-2]
    assert parsed['base'].shape==(1,311)


def test_raw_candle_gap_is_not_imputed(raw_fixture):
    root,_=raw_fixture
    p=root/'intraday/BTCUSDT_5m_futures.parquet'; df=pd.read_parquet(p)
    df.drop(index=1800).to_parquet(p,index=False)
    with pytest.raises(m.ModelInputError,match='contiguous'):
        m.build_feature_frames(root,'BTCUSDT')


def test_auxiliary_values_only_appear_after_declared_close(raw_fixture):
    root,_=raw_fixture
    p=root/'auxiliary/deribit_dvol_daily_2021_2026.parquet'
    dv=pd.read_parquet(p)
    mask=(dv.symbol=='BTC') & (dv.bucket_utc==pd.Timestamp('2026-01-06',tz='UTC'))
    dv.loc[mask,['open','high','low','close']]=[60.,61.,59.,60.]
    dv.to_parquet(p,index=False)
    before=m.build_feature_frames(root,'BTCUSDT',as_of='2026-01-07T00:00:00Z',tail_rows=1)
    after=m.build_feature_frames(root,'BTCUSDT',as_of='2026-01-07T00:05:00Z',tail_rows=1)
    assert before['base']['dvol_BTC_lvl'].iloc[-1]==50.
    assert after['base']['dvol_BTC_lvl'].iloc[-1]==60.
    p=root/'auxiliary/BTCUSDT_1h_mark.parquet'; mark=pd.read_parquet(p)
    mark.loc[mark.date==pd.Timestamp('2026-01-07T00:00:00Z'),['open','high','low','close']]=[160.,161.,159.,160.]
    mark.to_parquet(p,index=False)
    before=m.build_feature_frames(root,'BTCUSDT',as_of='2026-01-07T00:55:00Z',tail_rows=1)
    after=m.build_feature_frames(root,'BTCUSDT',as_of='2026-01-07T01:00:00Z',tail_rows=1)
    assert before['base']['fb_tgt_mark_basis_1h'].iloc[-1]==0.
    assert after['base']['fb_tgt_mark_basis_1h'].iloc[-1]==np.float32(.6)
    p=root/'auxiliary/binance_basis_4h.parquet'; basis=pd.read_parquet(p)
    basis.loc[pd.Timestamp('2026-01-07T00:00:00Z'),'BTCUSDT']=.1
    basis.to_parquet(p)
    before=m.build_feature_frames(root,'BTCUSDT',as_of='2026-01-07T03:55:00Z',tail_rows=1)
    after=m.build_feature_frames(root,'BTCUSDT',as_of='2026-01-07T04:00:00Z',tail_rows=1)
    assert before['base']['fb_b4_BTCUSDT_lvl'].iloc[-1]==np.float32(.001)
    assert after['base']['fb_b4_BTCUSDT_lvl'].iloc[-1]==np.float32(.1)
