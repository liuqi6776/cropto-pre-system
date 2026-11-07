from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from tools.argon4_forward.account import Account, AccountRejected, STREAMS, V5_SEMANTICS
from tools.argon4_forward.execution import PaperExecutor, adapt_official_resolution, fetch_public_resolution


BOUNDARY = datetime(2026,11,6,0,0,tzinfo=timezone.utc)
NOW = BOUNDARY+timedelta(seconds=1)
ORDER = ["BTC_15m","ETH_5m","ETH_15m","SOL_15m","SOL_5m","BNB_15m","BTC_5m"]


def policy():
    return dict(status="CONFIRMED",version="local-paper-test-v1",ranking_source="frozen user-confirmed ranking",order=ORDER[:])


def v5_policy():
    return {**policy(),"version":"local-paper-test-v5","waterfall_semantics":V5_SEMANTICS}


def no_signal(stream, boundary=BOUNDARY):
    return {"status":"NO_SIGNAL","prediction":{"stream":stream,"selected":False,"z_v4":.2,
        "threshold_frozen":1.,"side":"UP","model_version":"frozen-test-model","model_sha256":"b"*64,
        "manifest_sha256":"c"*64,"feature_bar_open":(boundary-timedelta(minutes=5)).isoformat(),
        "decision_time":(boundary+timedelta(milliseconds=500)).isoformat()}}


def clock(now=NOW):
    return dict(status="CALIBRATED",source="POLYMARKET_CLOB_SERVER_TIME",uncertainty_seconds=.1,observed_at=now.isoformat())


def fixture_inputs(capacity=100, boundary=BOUNDARY):
    now = boundary+timedelta(seconds=1)
    decisions, markets = {}, {}
    for i,stream in enumerate(ORDER,1):
        asset,duration = STREAMS[stream]
        start = datetime.fromtimestamp(int(boundary.timestamp())//duration*duration,timezone.utc)
        end = start+timedelta(seconds=duration)
        market_id = str(100+i)
        condition = '0x'+str(i)*64
        ids = {'UP':str(i*10+1),'DOWN':str(i*10+2)}
        market = dict(status="VERIFIED_MARKET",market_id=market_id,condition_id=condition,protocol="v2",asset=asset,
                      duration_seconds=duration,window_start=start.isoformat(),window_end=end.isoformat(),
                      active=True,closed=False,accepting_orders=True,outcome_ids=ids,
                      metadata_received_at=now.isoformat(),metadata_sha256="a"*64,
                      resolution=dict(source=f"https://data.chain.link/streams/{asset.lower()}-usd-twap-60s-streams"),
                      fee=dict(status="KNOWN_DISABLED",fees_enabled=False),books={})
        for side,token in ids.items():
            market['books'][side] = dict(status="OBSERVED",asset_id=token,condition_id=condition,
                source_at=now.isoformat(),received_at=now.isoformat(),asks=[dict(price=.5,size=capacity*2)],bids=[dict(price=.49,size=1000)])
        markets[stream] = market
        due = boundary.timestamp()%duration==0
        decisions[stream] = dict(status="NOT_DUE" if not due else "SIGNAL")
        if due:
            decisions[stream]['signal'] = dict(signal_id=f"model:{stream}:{int(boundary.timestamp())}",stream=stream,underlying=asset,
                outcome="UP",model_id="frozen-test-model",model_sha256="b"*64,validation_status="INPUTS_VALIDATED",
                feature_available_at=boundary.isoformat(),decision_at=(boundary+timedelta(milliseconds=500)).isoformat(),
                valid_until=(boundary+timedelta(seconds=15)).isoformat(),window_start=start.isoformat(),window_end=end.isoformat())
    return decisions,markets


@pytest.fixture
def executor(tmp_path):
    return PaperExecutor(Account(tmp_path/'paper.sqlite3',started_at=BOUNDARY-timedelta(seconds=1)),policy())


@pytest.fixture
def executor_v5(tmp_path):
    return PaperExecutor(Account(tmp_path/'paper-v5.sqlite3',started_at=BOUNDARY-timedelta(seconds=1)),v5_policy())


def run(executor, decisions=None, markets=None, boundary=BOUNDARY, now=None, **kwargs):
    d,m = fixture_inputs(boundary=boundary)
    stamp = now or boundary+timedelta(seconds=1)
    return executor.process_cohort(boundary.isoformat(),d if decisions is None else decisions,m if markets is None else markets,
                                  now=stamp,clock=clock(stamp),**kwargs)


def test_polling_and_restart_do_not_refresh_tick_or_stream_budget(executor):
    first = run(executor)
    assert [o['cost_usd'] for o in first['fills']] == [100,100,20]
    assert first['cohort']['used_usd']==220
    assert first['cohort']['remaining_usd']==0
    second = run(executor,now=BOUNDARY+timedelta(seconds=2))
    assert second['fills']==[]
    assert second['cohort']['used_usd']==220
    again = PaperExecutor(Account(executor.account.path),policy())
    third = run(again,now=BOUNDARY+timedelta(seconds=3))
    assert third['fills']==[]
    assert again.account.overview()['locked_cost']==220


def test_highest_capacity_unfilled_locks_every_lower_stream(executor):
    d,m = fixture_inputs(capacity=600)
    result = run(executor,d,m)
    assert [(o['underlying'],o['cost_usd']) for o in result['fills']]==[('BTC',150)]
    assert result['cohort']['remaining_usd']==70
    # A new signal identity and a smaller future book cannot reset the upper lock.
    d['BTC_15m']['signal']['signal_id']='replacement'
    m['BTC_15m']['books']['UP']['asks']=[dict(price=.5,size=20)]
    repeated = run(executor,d,m,now=BOUNDARY+timedelta(seconds=2))
    assert repeated['fills']==[]
    assert executor.account.overview()['order_count']==1


def test_v5_deep_book_fills_approved_150_then_70_and_preserves_raw_depth(executor_v5):
    d,m=fixture_inputs(capacity=600)
    calls=[]
    def refresh(market,side):
        calls.append((market['asset'],side))
        return market['books'][side]
    result=run(executor_v5,d,m,book_refresh=refresh)
    assert [o['cost_usd'] for o in result['fills']]==[150,70]
    assert calls==[('BTC','UP'),('ETH','UP')]
    states=result['stream_results']
    assert states[ORDER[0]]['status']==states[ORDER[1]]['status']=='FILLED_BUDGET'
    assert states[ORDER[0]]['raw_capacity_usd']==600
    assert states[ORDER[0]]['approved_usd']==150
    assert states[ORDER[2]]=={'status':'UNREACHED','reason':'COHORT_BUDGET_EXHAUSTED'}
    assert states[ORDER[-1]]['status']=='UNREACHED'
    assert result['cohort']['used_usd']==220
    assert result['cohort']['streams'][0]['capacity_usd']==600
    assert result['cohort']['streams'][0]['approved_usd']==150
    assert executor_v5.account.overview()['cash']==9780


def test_v5_no_signal_and_not_due_skip_without_any_market_read(executor_v5):
    boundary=BOUNDARY+timedelta(minutes=5)
    d,m=fixture_inputs(capacity=600,boundary=boundary)
    d['ETH_5m']=no_signal('ETH_5m',boundary)
    for stream in ORDER:
        if d[stream]['status']!='SIGNAL':
            m.pop(stream)
    calls=[]
    def refresh(market,side):
        calls.append((market['asset'],side))
        return market['books'][side]
    result=run(executor_v5,d,m,boundary=boundary,book_refresh=refresh)
    assert [o['cost_usd'] for o in result['fills']]==[150,70]
    assert calls==[('SOL','UP'),('BTC','UP')]
    assert result['stream_results']['BTC_15m']['status']=='NOT_DUE'
    assert result['stream_results']['ETH_5m']['status']=='NO_SIGNAL'
    assert result['stream_results']['ETH_5m']['book_checked'] is False
    assert result['cohort']['used_usd']==220
    skips=[e for e in executor_v5.account.ledger() if e['event_type']=='COHORT_STREAM_SKIPPED']
    assert len(skips)==5


def test_v5_known_price_limit_failure_skips_own_side_and_reaches_lower(executor_v5):
    d,m=fixture_inputs(capacity=600)
    m[ORDER[0]]['books']['UP']['asks']=[{'price':.6,'size':1000}]
    m[ORDER[0]]['books']['DOWN']=None  # An unselected outcome is never required.
    result=run(executor_v5,d,m)
    assert [o['underlying'] for o in result['fills']]==['ETH','ETH']
    assert [o['cost_usd'] for o in result['fills']]==[150,70]
    first=result['stream_results'][ORDER[0]]
    assert first['status']=='PRICE_LIMIT_NOT_MET' and first['best_ask']==.6
    assert first['raw_capacity_usd']==0 and first['spent_usd']==0
    assert result['cohort']['streams'][0]['state']=='PRICE_LIMIT_NOT_MET'
    receipt=result['cohort']['streams'][0]['receipt']
    assert receipt['signal']['model_sha256']=='b'*64
    assert receipt['orderbook']['observed_at']==NOW.isoformat()
    assert receipt['fee_quote']['market_id']==m[ORDER[0]]['market_id']


@pytest.mark.parametrize('invalid',['empty','zero_size','stale','wrong_token','wrong_condition','missing_fee_hash',
                                  'bad_fee_rate','unsupported_fee','expired_signal','wrong_model_hash','future_signal'])
def test_v5_invalid_price_evidence_never_becomes_a_skip(executor_v5,invalid):
    d,m=fixture_inputs(capacity=600)
    first=m[ORDER[0]]
    first['books']['UP']['asks']=[{'price':.6,'size':1000}]
    if invalid=='empty':first['books']['UP']['asks']=[]
    elif invalid=='zero_size':first['books']['UP']['asks']=[{'price':.6,'size':0}]
    elif invalid=='stale':first['books']['UP']['source_at']=(BOUNDARY-timedelta(seconds=5)).isoformat()
    elif invalid=='wrong_token':first['books']['UP']['asset_id']='wrong'
    elif invalid=='wrong_condition':first['books']['UP']['condition_id']='wrong'
    elif invalid=='missing_fee_hash':first['metadata_sha256']=''
    elif invalid=='bad_fee_rate':first['fee']={'status':'KNOWN_SCHEDULE','fees_enabled':True,'exponent':1,'taker_only':True,'rate':-1}
    elif invalid=='unsupported_fee':first['fee']={'status':'UNKNOWN'}
    elif invalid=='expired_signal':d[ORDER[0]]['signal']['valid_until']=NOW.isoformat()
    elif invalid=='wrong_model_hash':d[ORDER[0]]['signal']['model_sha256']='bad'
    else:d[ORDER[0]]['signal']['decision_at']=(NOW+timedelta(seconds=1)).isoformat()
    result=run(executor_v5,d,m)
    assert result['fills']==[]
    assert result['stream_results'][ORDER[0]]['status']=='ERROR'
    assert result['stream_results'][ORDER[1]]['status']=='UNREACHED'
    assert result['cohort']['streams']==[]
    assert executor_v5.account.overview()['cash']==10000


@pytest.mark.parametrize('invalid',['unknown','missing','missing_prediction','selected_true','above_threshold','wrong_pin',
                                  'wrong_stream','future_decision','wrong_bar'])
def test_v5_unknown_or_unvalidated_no_signal_blocks_lower(executor_v5,invalid):
    d,m=fixture_inputs()
    d[ORDER[0]]=no_signal(ORDER[0])
    p=d[ORDER[0]]['prediction']
    if invalid=='unknown':d[ORDER[0]]={'status':'BLOCKED'}
    elif invalid=='missing':d.pop(ORDER[0])
    elif invalid=='missing_prediction':d[ORDER[0]]={'status':'NO_SIGNAL'}
    elif invalid=='selected_true':p['selected']=True
    elif invalid=='above_threshold':p['z_v4']=1
    elif invalid=='wrong_pin':executor_v5.policy['deployment_manifest_sha256']={ORDER[0]:'d'*64}
    elif invalid=='wrong_stream':p['stream']='BTC_5m'
    elif invalid=='future_decision':p['decision_time']=(NOW+timedelta(seconds=1)).isoformat()
    else:p['feature_bar_open']=BOUNDARY.isoformat()
    result=run(executor_v5,d,m,book_refresh=lambda *a:pytest.fail('invalid skip must not fetch books'))
    assert result['fills']==[]
    assert result['stream_results'][ORDER[0]]['status']=='ERROR'
    assert result['stream_results'][ORDER[1]]['status']=='UNREACHED'
    assert result['cohort']['streams']==[]


def preconsume_depth(executor,decisions,markets,amount,identifier='prior'):
    s=deepcopy(decisions[ORDER[0]]['signal']);s['signal_id']=identifier
    market=markets[ORDER[0]];s['market_id']=market['market_id']
    book,fee,_=executor._adapt(market,'UP',market['books']['UP'],NOW,clock())
    auth={'policy':'RANKED_CAPACITY_WATERFALL','plan_id':identifier,'rank':1,'signal_id':identifier,
          'market_id':market['market_id'],'approved_stake_usd':amount,'planned_at':NOW.isoformat(),
          'valid_until':s['valid_until']}
    return executor.account.place_paper_buy(s,book,fee,now=NOW,stake=amount,allocation_authorization=auth)


@pytest.mark.parametrize('consumed,filled',[(50,50),(100,0)])
def test_v5_actual_partial_or_unfilled_depth_blocks_lower_and_survives_restart(executor_v5,consumed,filled):
    d,m=fixture_inputs(capacity=100)
    preconsume_depth(executor_v5,d,m,consumed)
    result=run(executor_v5,d,m)
    assert len(result['fills'])==1 and result['fills'][0]['cost_usd']==filled
    assert result['stream_results'][ORDER[0]]['status']=='PARTIAL'
    assert result['stream_results'][ORDER[1]]['status']=='UNREACHED'
    assert result['cohort']['used_usd']==filled
    restarted=PaperExecutor(Account(executor_v5.account.path),v5_policy())
    again=run(restarted,d,m,now=BOUNDARY+timedelta(seconds=2),book_refresh=lambda *a:pytest.fail('partial cannot retry'))
    assert again['fills']==[] and again['stream_results'][ORDER[0]]['status']=='PARTIAL'
    assert restarted.account.overview()['order_count']==2


@pytest.mark.parametrize('capacity',[.000001,.000002])
def test_v5_rounding_tolerance_cannot_unlock_a_zero_quantity_fill(executor_v5,capacity):
    d,m=fixture_inputs(capacity=capacity)
    preconsume_depth(executor_v5,d,m,capacity)
    result=run(executor_v5,d,m)
    assert len(result['fills'])==1 and result['fills'][0]['quantity']=='0'
    assert result['stream_results'][ORDER[0]]['status']=='PARTIAL'
    assert result['stream_results'][ORDER[1]]['status']=='UNREACHED'
    assert result['cohort']['used_usd']==0


def test_v5_sub_micro_dollar_observed_capacity_blocks_instead_of_price_skip(executor_v5):
    d,m=fixture_inputs(capacity=.0000001)
    result=run(executor_v5,d,m)
    assert result['fills']==[]
    assert result['stream_results'][ORDER[0]]['status']=='ERROR'
    assert 'below the accounting unit' in result['blocked'][0]
    assert result['stream_results'][ORDER[1]]['status']=='UNREACHED'
    assert executor_v5.account.overview()['order_count']==0


def test_v5_zero_single_budget_after_quantization_cannot_place_order(tmp_path):
    tiny=PaperExecutor(Account(tmp_path/'tiny.sqlite3',initial_capital=.00005,
                              started_at=BOUNDARY-timedelta(seconds=1)),v5_policy())
    result=run(tiny)
    assert result['cohort']['budget_usd']==.000001
    assert result['cohort']['single_usd']==0
    assert result['fills']==[] and result['stream_results'][ORDER[0]]['status']=='ERROR'
    assert tiny.account.overview()['order_count']==0
    assert tiny.account.overview()['cash']==.00005


def test_v5_polling_restart_and_split_signals_cannot_refill_budget(executor_v5):
    d,m=fixture_inputs(capacity=600)
    first=run(executor_v5,d,m)
    d[ORDER[0]]['signal']['signal_id']='new-signal-cannot-split'
    restarted=PaperExecutor(Account(executor_v5.account.path),v5_policy())
    second=run(restarted,d,m,now=BOUNDARY+timedelta(seconds=2),book_refresh=lambda *a:pytest.fail('budget must not reset'))
    assert second['fills']==[] and second['cohort']['used_usd']==first['cohort']['used_usd']==220
    assert restarted.account.overview()['order_count']==2
    old=run(restarted,d,m,now=BOUNDARY+timedelta(seconds=20))
    assert old['fills']==[] and old['status']=='BLOCKED'


def test_v5_existing_underlying_risk_clips_target_without_treating_raw_depth_as_target(executor_v5):
    d,m=fixture_inputs(capacity=600)
    preconsume_depth(executor_v5,d,m,200,'prior-1')
    preconsume_depth(executor_v5,d,m,200,'prior-2')
    result=run(executor_v5,d,m)
    assert [o['cost_usd'] for o in result['fills']]==[100,120]
    assert result['stream_results'][ORDER[0]]['status']=='FILLED_BUDGET'
    assert result['cohort']['used_usd']==220
    positions=executor_v5.account.overview()['positions']
    assert sum(p['cost_usd']+p['fee_usd'] for p in positions if p['underlying']=='BTC')==500


def test_v5_actual_fee_is_inside_approved_budget(executor_v5):
    d,m=fixture_inputs(capacity=600)
    for market in m.values():
        market['fee']={'status':'KNOWN_SCHEDULE','fees_enabled':True,'rate':.07,'exponent':1,'taker_only':True}
    result=run(executor_v5,d,m)
    assert len(result['fills'])==2
    assert all(o['fee_usd']>0 for o in result['fills'])
    assert all(o['cohort_stream_state']=='FILLED_BUDGET' for o in result['fills'])
    assert 219.999996<=result['cohort']['used_usd']<=220
    assert executor_v5.account.overview()['cash']+executor_v5.account.overview()['locked_cost']==10000


def test_v5_does_not_reinterpret_a_persisted_v4_cohort(executor):
    d,m=fixture_inputs(capacity=600)
    old=run(executor,d,m)
    assert old['cohort']['streams'][0]['state']=='PARTIAL_CAPACITY'
    upgraded=PaperExecutor(executor.account,v5_policy())
    rejected=run(upgraded,d,m,now=BOUNDARY+timedelta(seconds=2))
    assert rejected['fills']==[] and rejected['status']=='BLOCKED'
    assert 'policy cannot change' in rejected['blocked'][0]
    assert executor.account.cohort_state(BOUNDARY)['streams'][0]['state']=='PARTIAL_CAPACITY'
    boundary=BOUNDARY+timedelta(minutes=5)
    d,m=fixture_inputs(capacity=600,boundary=boundary)
    later=run(upgraded,d,m,boundary=boundary)
    assert [o['cost_usd'] for o in later['fills']]==[150,70]
    assert later['cohort']['policy']['waterfall_semantics']==V5_SEMANTICS
    assert executor.account.cohort_state(BOUNDARY)['streams'][0]['state']=='PARTIAL_CAPACITY'


def test_two_hundred_ceiling_is_per_stream_cohort_even_with_larger_equity(tmp_path):
    executor=PaperExecutor(Account(tmp_path/'large.sqlite3',initial_capital=40000,started_at=BOUNDARY-timedelta(seconds=1)),policy())
    d,m=fixture_inputs(capacity=600)
    first=run(executor,d,m)
    assert [o['cost_usd'] for o in first['fills']]==[200]
    d['BTC_15m']['signal']['signal_id']='a-new-id-cannot-split-the-order'
    second=run(executor,d,m,now=BOUNDARY+timedelta(seconds=2))
    assert second['fills']==[]
    assert second['cohort']['used_usd']==200


def test_filled_upper_credit_survives_a_later_missing_lower_decision(executor):
    d,m = fixture_inputs()
    deferred=deepcopy(d)
    deferred[ORDER[1]]={'status':'BLOCKED'}
    first=run(executor,deferred,m)
    assert len(first['fills'])==1
    second=run(executor,d,m,now=BOUNDARY+timedelta(seconds=2))
    assert [o['cost_usd'] for o in second['fills']]==[100,20]
    assert second['cohort']['used_usd']==220


def test_missing_or_zero_or_stale_capacity_does_not_unlock(executor):
    d,m=fixture_inputs()
    m['BTC_15m']['books']['UP']['asks']=[]
    assert run(executor,d,m)['fills']==[]
    d,m=fixture_inputs()
    m['BTC_15m']['books']['UP']['source_at']=(BOUNDARY-timedelta(seconds=10)).isoformat()
    assert run(executor,d,m)['fills']==[]


def test_not_due_does_not_bypass_capital_capacity_gate(executor):
    boundary=BOUNDARY+timedelta(minutes=5)
    d,m=fixture_inputs(capacity=600,boundary=boundary)
    result=run(executor,d,m,boundary=boundary)
    assert d['BTC_15m']['status']=='NOT_DUE'
    assert result['fills']==[]
    assert result['cohort']['streams'][0]['state']=='CAPACITY_LOCKED'


def test_not_due_can_release_lower_only_after_observed_capacity_gate(executor):
    boundary=BOUNDARY+timedelta(minutes=5)
    d,m=fixture_inputs(capacity=100,boundary=boundary)
    result=run(executor,d,m,boundary=boundary)
    assert result['fills'][0]['underlying']=='ETH'
    assert result['cohort']['streams'][0]['state']=='NOT_DUE'


def test_clock_proof_is_mandatory_and_does_not_fabricate_trades(executor):
    d,m=fixture_inputs()
    result=executor.process_cohort(BOUNDARY,d,m,now=NOW)
    assert result['fills']==[] and result['blocked']==['CLOCK_NOT_CALIBRATED']
    bad=clock();bad['uncertainty_seconds']=5
    result=executor.process_cohort(BOUNDARY,d,m,now=NOW,clock=bad)
    assert result['fills']==[]


def test_book_refresh_cannot_relabel_an_old_source_time(executor):
    d,m=fixture_inputs()
    def refresh(market,side):
        b=deepcopy(market['books'][side])
        b['source_at']=(BOUNDARY-timedelta(seconds=60)).isoformat()
        b['received_at']=NOW.isoformat()
        return b
    result=run(executor,d,m,book_refresh=refresh)
    assert result['fills']==[]
    assert any('older than two seconds' in b for b in result['blocked'])


def test_explicit_clock_projection_preserves_local_receipts_and_source_timestamp(executor):
    d,m=fixture_inputs(capacity=600)
    for market in m.values():
        market['metadata_received_at']=(NOW-timedelta(seconds=4.6)).isoformat()
        for b in market['books'].values():
            b['received_at']=(NOW-timedelta(seconds=4.6)).isoformat()
    calibrated=clock()
    calibrated.update(server_minus_local_seconds=4.5,receipt_sha256='c'*64,receipt={'public_clock_samples':6})
    result=executor.process_cohort(BOUNDARY,d,m,now=NOW,clock=calibrated)
    assert len(result['fills'])==1
    order=result['fills'][0]
    assert order['book_observed_at']==NOW.isoformat()
    assert order['book_received_at_local']==(NOW-timedelta(seconds=4.6)).isoformat()
    assert order['clock_projection']['receipt_sha256']=='c'*64
    assert m['BTC_15m']['books']['UP']['received_at']==(NOW-timedelta(seconds=4.6)).isoformat()


def test_clock_projection_without_receipt_cannot_make_book_fresh(executor):
    d,m=fixture_inputs()
    calibrated=clock();calibrated['server_minus_local_seconds']=4.5
    result=executor.process_cohort(BOUNDARY,d,m,now=NOW,clock=calibrated)
    assert result['fills']==[]
    assert result['blocked']==['clock projection receipt hash required']


def test_new_cohort_cannot_change_policy_or_reuse_a_consumed_signal(executor):
    run(executor)
    changed=policy();changed['version']='changed'
    with pytest.raises(AccountRejected,match='cannot change'):
        executor.account.ensure_cohort(BOUNDARY,changed,now=NOW)
    with pytest.raises(AccountRejected,match='current five-minute'):
        executor.account.ensure_cohort(BOUNDARY,policy(),now=BOUNDARY+timedelta(seconds=21))


def official_pair(market, resolved=True, winner='UP', received=None):
    received=received or _end(market)+timedelta(seconds=2)
    raw=dict(id=market['market_id'],conditionId=market['condition_id'],version='v2',
             eventStartTime=market['window_start'],endDate=market['window_end'],
             resolutionSource=market['resolution']['source'],resolutionStatus='resolved' if resolved else 'active',
             closed=True,outcomes=['Up','Down'],positionIds=[market['outcome_ids']['UP'],market['outcome_ids']['DOWN']],
             outcomePrices=['1','0'] if winner=='UP' else ['0','1'])
    body=json.dumps(raw)
    receipt=dict(url=f"https://gamma-api.polymarket.com/markets/{market['market_id']}",method='GET',http_status=200,
                 body_complete=True,raw_text=body,sha256=hashlib.sha256(body.encode()).hexdigest(),received_at=received.isoformat())
    return raw,receipt


def _end(market):
    return datetime.fromisoformat(market['window_end'])


def test_only_official_final_states_settle_and_empty_observation_stays_pending(executor):
    d,m=fixture_inputs(capacity=600)
    run(executor,d,m)
    before=executor.account.overview()['cash']
    assert executor.settle_observed()['settled']==[]
    assert executor.account.overview()['cash']==before
    market=m['BTC_15m'];stamp=_end(market)+timedelta(seconds=3)
    proposed=official_pair(market,resolved=False)
    assert executor.settle_observed({market['market_id']:proposed},now=stamp)['settled']==[]
    final=official_pair(market)
    result=executor.settle_observed({market['market_id']:final},now=stamp)
    assert len(result['settled'])==1
    assert executor.account.overview()['cash']==10150
    assert executor.settle_observed({market['market_id']:final},now=stamp)['settled']==[]


def test_final_resolution_must_match_contract_and_receipt_integrity(executor):
    d,m=fixture_inputs(capacity=600)
    run(executor,d,m)
    market=m['BTC_15m'];stamp=_end(market)+timedelta(seconds=3)
    raw,receipt=official_pair(market)
    raw['conditionId']='0x'+'f'*64
    result=executor.settle_observed({market['market_id']:(raw,receipt)},now=stamp)
    assert result['settled']==[]
    assert executor.account.overview()['wins']==0


def test_binary_tie_price_is_not_invented_as_refund(executor):
    d,m=fixture_inputs(capacity=600)
    run(executor,d,m)
    market=m['BTC_15m'];stamp=_end(market)+timedelta(seconds=3)
    raw,receipt=official_pair(market)
    raw['outcomePrices']=['.5','.5']
    receipt['raw_text']=json.dumps(raw);receipt['sha256']=hashlib.sha256(receipt['raw_text'].encode()).hexdigest()
    assert executor.settle_observed({market['market_id']:(raw,receipt)},now=stamp)['settled']==[]


def test_resolution_get_rejects_non_numeric_identifier_before_network():
    with pytest.raises(AccountRejected):
        fetch_public_resolution('../orders')
