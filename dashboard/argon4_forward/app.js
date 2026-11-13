"use strict";
const $=id=>document.getElementById(id);
const money=v=>v===null||v===undefined||Number.isNaN(Number(v))?"待估值":new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(Number(v));
const moneyStrict=(v,fb="未知")=>(v===null||v===undefined||v===""||Number.isNaN(Number(v)))?String(fb):new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(Number(v));
const signedMoney=v=>v===null||v===undefined||Number.isNaN(Number(v))?"待结算":(Number(v)>0?"+":"")+money(Number(v));
const when=v=>v?new Date(v).toLocaleString("zh-CN",{hour12:false}):"尚无观测";
function elem(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n;}
function cell(row,text,cls){row.appendChild(elem("td",text,cls));}

function initTabs(){
 const btnLive=$("tab-btn-live"),btnMon=$("tab-btn-monitor");
 const panLive=$("tab-panel-live"),panMon=$("tab-panel-monitor");
 if(!btnLive||!btnMon||!panLive||!panMon)return;
 const activate=tab=>{
  const isLive=tab==="live";
  btnLive.classList.toggle("active",isLive);
  btnLive.setAttribute("aria-selected",String(isLive));
  btnMon.classList.toggle("active",!isLive);
  btnMon.setAttribute("aria-selected",String(!isLive));
  panLive.hidden=!isLive;
  panMon.hidden=isLive;
 };
 btnLive.addEventListener("click",()=>activate("live"));
 btnMon.addEventListener("click",()=>activate("monitor"));
}

function renderLiveError(reason){
 if(!$("tab-panel-live"))return;
 const msg=reason||"实盘接口请求失败";
 if($("tab-badge-live")){
  $("tab-badge-live").textContent="接口异常 · "+msg;
  $("tab-badge-live").className="tab-pill offline";
 }
 if($("live-banner")){
  $("live-banner").className="banner offline";
  $("live-banner").textContent="🟠 实盘接口异常（"+msg+"）：未能获取实盘账户与配额状态，请检查 /api/live_dashboard 或 argon4-order 服务";
 }
 if($("live-proc-status"))$("live-proc-status").textContent="接口异常 ("+msg+")";
}
function renderLive(d){
 if(!d||!$("tab-panel-live"))return;
 if(d.__live_error){renderLiveError(d.reason);return;}
 const sum=d.summary||{};
 const q=(d.latest_quota||{}).live_wallet_and_quotas||{};
 const pm=(d.latest_quota||{}).paper_account_mirror||{};
 const svc=d.service||{};
 const caps=d.scaled_live_caps||{};
 const pcaps=d.paper_policy_caps||{};
 const running=Boolean(d.process_running);
 const snapStatus=d.quota_snapshot_status||((d.latest_quota||{}).timestamp_utc?"FRESH":"MISSING");
 const snapAge=d.quota_snapshot_age_seconds;
 const capSrc=d.capital_sources||{};
 const pnl=Number(sum.realized_pnl_usdc||0);
 const wins=Number(sum.wins||0),losses=Number(sum.losses||0);
 const filled=Number(sum.filled_orders||0),totalSig=Number(sum.total_signals||0);

 if($("tab-badge-live")){
  $("tab-badge-live").textContent=(snapStatus==="STALE"?"快照过期 · ":snapStatus==="MISSING"?"无快照 · ":"")+(running?("运行中 · "+signedMoney(pnl)):"已停止 · "+signedMoney(pnl));
  $("tab-badge-live").className="tab-pill "+((running&&snapStatus==="FRESH")?"ok":"offline");
 }
 if($("live-banner")){
  $("live-banner").className=(running&&snapStatus==="FRESH")?"banner ok":"banner offline";
  const snapNote=snapStatus==="MISSING"?" · ⚠️ 暂无配额快照（未伪造余额）":snapStatus==="STALE"?" · ⚠️ 配额快照已过期("+Number(snapAge||0).toFixed(0)+"s)":"";
  $("live-banner").textContent=(running
   ?("外部实盘执行器运行中 (PID "+svc.pid+") · 模式 "+(d.active_mode||"live").toUpperCase()+" · 缩放 "+(Number(d.live_scale_factor||1)*100).toFixed(0)+"% · 实盘已结算 "+wins+" 胜 "+losses+" 负 · 净收益 "+signedMoney(pnl))
   :("外部实盘执行器未运行 · 历史已结算 "+wins+" 胜 "+losses+" 负 · 累计净收益 "+signedMoney(pnl)))+snapNote;
 }
 if($("live-free-usdc"))$("live-free-usdc").textContent=snapStatus==="MISSING"?"未知":(moneyStrict(q.onchain_free_usdc_balance)+(snapStatus==="STALE"?" (已过期)":""));
 if($("live-allowance-note")){
  const srcObj=capSrc.live_wallet_free_pusd||{};
  const srcTag=srcObj.source?("来源: "+srcObj.source):"";
  if(snapStatus==="MISSING")$("live-allowance-note").textContent="未知（暂无实时余额快照）";
  else if(snapStatus==="STALE")$("live-allowance-note").textContent="已过期("+Number(snapAge||0).toFixed(0)+"s) · "+(srcTag||"请核实钱包快照");
  else $("live-allowance-note").textContent=(Number(q.clob_allowance_usdc)>1e6?"CLOB 授权充足 · 提现硬锁已开启":q.clob_allowance_usdc!==null&&q.clob_allowance_usdc!==undefined?"CLOB 授权 "+moneyStrict(q.clob_allowance_usdc):"授权未知")+(srcTag?" · "+srcTag:"");
 }
 if($("live-open-risk"))$("live-open-risk").textContent=snapStatus==="MISSING"?"未知":(moneyStrict(q.total_locked_open_risk_usdc)+(snapStatus==="STALE"?" (已过期)":""));
 if($("live-remaining-quota"))$("live-remaining-quota").textContent=snapStatus==="MISSING"?"剩余总风险额度 未知 · 窗口剩余 未知":("剩余总风险额度 "+moneyStrict(q.remaining_total_open_risk_quota_usd)+" · 窗口剩余 "+moneyStrict(q.remaining_cohort_budget_quota_usd)+(snapStatus==="STALE"?" (快照已过期)":""));
 if($("live-pnl")){
  $("live-pnl").textContent=signedMoney(pnl);
  $("live-pnl").className=pnl>0?"pnl-pos":(pnl<0?"pnl-neg":"");
 }
 if($("live-winrate")){
  $("live-winrate").textContent="已结算 "+wins+" 胜 / "+losses+" 负"+(sum.win_rate_pct!==null&&sum.win_rate_pct!==undefined?"（胜率 "+Number(sum.win_rate_pct).toFixed(2)+"%）":"")+" · 已赎回 "+money(sum.redeemed_usdc_total||0);
 }
 if($("live-fills-count"))$("live-fills-count").textContent=filled+" 笔成交 / "+totalSig+" 信号";
 if($("live-latency-note"))$("live-latency-note").textContent="累计实付 "+money(sum.total_spent_usdc||0)+(sum.avg_clob_rtt_ms?" · CLOB 往返 "+Number(sum.avg_clob_rtt_ms).toFixed(0)+"ms":"")+(sum.avg_slippage_bps!==null&&sum.avg_slippage_bps!==undefined?" · 均价滑点 "+Number(sum.avg_slippage_bps).toFixed(1)+" bps":"");

 if($("live-quota-time"))$("live-quota-time").textContent=snapStatus==="MISSING"?"无快照":(when((d.latest_quota||{}).timestamp_utc)+(snapStatus==="STALE"?" (已过期 "+Number(snapAge||0).toFixed(0)+"s)":""));
 if($("live-lock-active"))$("live-lock-active").textContent=moneyStrict(q.active_window_locked_usdc);
 if($("live-lock-expired"))$("live-lock-expired").textContent=moneyStrict(q.expired_awaiting_resolution_usdc);
 if($("live-lock-unredeemed"))$("live-lock-unredeemed").textContent=moneyStrict(q.resolved_unredeemed_winning_usdc);
 if($("live-lock-inflight"))$("live-lock-inflight").textContent=moneyStrict(q.in_flight_reserved_usdc);
 if($("live-rem-cohort"))$("live-rem-cohort").textContent=moneyStrict(q.remaining_cohort_budget_quota_usd);
 if($("live-scaled-caps"))$("live-scaled-caps").textContent="单笔 "+moneyStrict(caps.max_order_usd)+" / 窗口 "+moneyStrict(caps.max_tick_budget_usd)+" / 活动 "+moneyStrict(caps.max_active_window_risk_usd)+" / 总敞口 "+moneyStrict(caps.max_open_risk_usd)+"（模拟基准 "+moneyStrict(pcaps.max_order_usd)+" / "+moneyStrict(pcaps.max_tick_budget_usd)+"）";

 const uqBody=$("live-underlying-quotas");
 if(uqBody){
  uqBody.replaceChildren();
  const remMap=q.remaining_underlying_quota_usd||{};
  for(const [coin,rem] of Object.entries(remMap)){
   const r=elem("tr");
   cell(r,coin);
   cell(r,moneyStrict(caps.max_underlying_open_risk_usd));
   cell(r,moneyStrict(rem));
   uqBody.append(r);
  }
 }

 if($("live-pid-badge"))$("live-pid-badge").textContent=svc.pid?"PID "+svc.pid:"未启动";
 if($("live-proc-status"))$("live-proc-status").textContent=running?"正常运行中 (LIVE)":"已停止";
 if($("live-started-at"))$("live-started-at").textContent=when(svc.started_at);
 if($("live-mode-info"))$("live-mode-info").textContent=(svc.mode||"live").toUpperCase()+" / "+(svc.order_type||"FAK");
 if($("live-scale-info"))$("live-scale-info").textContent=(Number(d.live_scale_factor||1)*100).toFixed(0)+"% (1:1 跟随 "+moneyStrict(pcaps.initial_capital_usd)+" 统一配置)";
 if($("live-wallet-addr"))$("live-wallet-addr").textContent=svc.funder_address||q.proxy_wallet_address||"未知";
 if($("live-withdraw-lock"))$("live-withdraw-lock").textContent=svc.withdrawal_permission===false?"已硬锁禁止提现 (withdrawal_permission=false)":"未知";
 if($("live-paper-mirror"))$("live-paper-mirror").textContent=(pm.paper_cash_usd!==null&&pm.paper_cash_usd!==undefined)?("镜像现金 "+moneyStrict(pm.paper_cash_usd)+" · 镜像净盈亏 "+signedMoney(pm.paper_realized_pnl_usd)+((capSrc.paper_account_cash||{}).source?" ("+capSrc.paper_account_cash.source+")":"")):"未知（无镜像快照）";

 const stByMkt={};
 for(const st of (d.settlements||[])){if(st.market_id)stByMkt[String(st.market_id)]=st;}

 if($("live-orders-summary"))$("live-orders-summary").textContent="共 "+totalSig+" 笔信号 · "+filled+" 笔实盘成交 · 实盘净收益 "+signedMoney(pnl);
 const ordBody=$("live-orders-body");
 if(ordBody){
  ordBody.replaceChildren();
  const orders=d.orders||[];
  if(!orders.length){
   const r=elem("tr");for(let i=0;i<7;i++)cell(r,i===0?"暂无记录":"—");ordBody.append(r);
  }
  for(const o of orders){
   const r=elem("tr");
   const tm=o.timing||{},pb=o.paper_benchmark||{},le=o.live_execution||{},df=o.diff_vs_paper||{};
   const st=stByMkt[String(o.market_id)];
   cell(r,when(tm.live_http_ack_at||tm.live_detected_at||tm.paper_fill_at||o.cohort_id));
   cell(r,(o.stream||"—")+" · "+(o.outcome==="UP"?"上涨 (UP)":"下跌 (DOWN)")+" (#"+(o.rank||"")+")");
   cell(r,"均价 "+Number(pb.avg_fill_price||0).toFixed(4)+" / "+Number(pb.filled_quantity||0).toFixed(2)+" 股 / "+money( Number(pb.cost_usd||0)+Number(pb.fee_usd||0) ));
   cell(r,"限价 "+Number(le.limit_price||0).toFixed(3)+" / "+Number(le.requested_quantity||0).toFixed(2)+" 股 / "+money(le.requested_stake_usd));
   const matched=Number(le.filled_quantity||0)>0;
   cell(r,(le.clob_status||"—")+(matched?(" · 均价 "+Number(le.avg_fill_price||0).toFixed(4)+" / "+Number(le.filled_quantity||0).toFixed(2)+" 股 / 实付 "+money(le.total_spent_usdc)):" · 0 股 ($0.00)"),matched?"pnl-pos":"");
   cell(r,(df.avg_price_slippage_bps!==undefined&&df.avg_price_slippage_bps!==null?Number(df.avg_price_slippage_bps).toFixed(1)+" bps":"—")+" / "+(tm.detection_lag_ms?Number(tm.detection_lag_ms).toFixed(0)+"ms":"—")+" / "+(tm.clob_rtt_ms?Number(tm.clob_rtt_ms).toFixed(0)+"ms":"—"));
   let posText="未成交 (NO_FILL)",posCls="";
   if(matched){
    if(st){
     const spnl=Number(st.live_realized_pnl_usdc||0);
     if(spnl>0){posText="【已胜赎回】"+signedMoney(spnl);posCls="pnl-pos";}
     else if(spnl<0){posText="【已负结算】"+signedMoney(spnl);posCls="pnl-neg";}
     else{posText=st.redeem_status||"已结算";}
    }else{
     posText="持仓待官方结算";
    }
   }else if(le.clob_status==="ORDER_STATE_UNKNOWN"){
    posText="状态待对账";
   }
   cell(r,posText,posCls);
   ordBody.append(r);
  }
 }

 if($("live-settlements-summary"))$("live-settlements-summary").textContent="已结算 "+wins+" 胜 "+losses+" 负 · 累计赎回 "+money(sum.redeemed_usdc_total||0)+" · 实盘净盈亏 "+signedMoney(pnl);
 const stBody=$("live-settlements-body");
 if(stBody){
  stBody.replaceChildren();
  const allSt=d.settlements||[];
  const activeSt=allSt.filter(s=>Number(s.live_cost_usdc||0)>0||Number(s.live_expected_payout_usdc||0)>0||Number(s.redeemed_usdc_amount||0)>0||Math.abs(Number(s.live_realized_pnl_usdc||0))>1e-6||s.redeem_status==="AWAITING_ORDER_RECONCILIATION");
  const rowsToShow=activeSt.length?activeSt:allSt;
  if(!rowsToShow.length){
   const r=elem("tr");for(let i=0;i<8;i++)cell(r,i===0?"暂无结算":"—");stBody.append(r);
  }
  for(const st of rowsToShow){
   const r=elem("tr");
   cell(r,when(st.first_settled_at_utc||st.timestamp_utc));
   cell(r,(st.stream?st.stream+" · ":"")+"#"+st.market_id+(st.order_outcome?" (买 "+st.order_outcome+")":""));
   cell(r,st.official_outcome==="UP"?"上涨 (UP)":"下跌 (DOWN)");
   cell(r,money(st.paper_payout_usd||0));
   cell(r,money(st.live_expected_payout_usdc||0)+(st.live_cost_usdc?"（成本 "+money(st.live_cost_usdc)+"）":""));
   cell(r,money(st.redeemed_usdc_amount||0));
   const spnl=Number(st.live_realized_pnl_usdc||0);
   const pCls=spnl>1e-6?"pnl-pos":(spnl<-1e-6?"pnl-neg":"");
   cell(r,signedMoney(spnl),pCls);
   const txShort=st.redeem_tx_hash?(st.redeem_tx_hash.slice(0,10)+"…"+st.redeem_tx_hash.slice(-6)):"—";
   cell(r,(st.redeem_status||"—")+(st.redeem_tx_hash?" · "+txShort:""));
   stBody.append(r);
  }
 }

 if($("live-logs-box")){
  const logs=d.recent_logs||[];
  $("live-logs-box").textContent=logs.length?logs.join("\n"):"暂无日志输出";
 }
}

function renderEngineering(s){
 renderRisk(s);
 const e=s.engineering||{},t=(s.pipeline||{}).timing||{},b=s.book_telemetry||{},q=b.last_sample||{};
 const sec=v=>typeof v==="number"&&Number.isFinite(v)?v.toFixed(2)+" 秒":"—";
 $("engineeringversion").textContent=(s.allocation||{}).version||e.profile||"—";
 $("latencycohort").textContent=t.cohort?when(t.cohort):"等待新窗口";
 const statuses={PROCESSED:"已完成检查",COMPLETE:"已完成检查",WAITING_PRICE_RECHECK:"等待本窗口价格重检",WINDOW_EXPIRED:"窗口到期，停止重检",INPUTS_NOT_VALIDATED:"输入未通过，跳过",FAILED:"检查失败，跳过",STOPPED_BEFORE_EXECUTION:"执行前停止"};
 $("latencytotal").textContent=sec(t.completed_after_boundary_seconds)+" · "+(statuses[t.status]||t.status||"等待检查")+(typeof t.deadline_margin_seconds==="number"?" · 距 20 秒截止 "+sec(t.deadline_margin_seconds):"");
 $("latencybreakdown").textContent=sec(t.input_refresh_seconds)+" / "+sec(t.decisions_seconds);
 $("latencyfirstlook").textContent=sec(t.first_execution_completed_after_boundary_seconds)+(t.execution_passes?" · 执行检查轮数 "+t.execution_passes:"");
 const names={CALIBRATED:"实时采样中",WAITING_CLOCK:"等待时钟检查",WAITING_VALID_CLOCK:"时钟暂不可用",DISABLED:"未启用",STOPPED:"已停止"};
 $("telemetrystate").textContent=(names[b.status]||b.status||"等待启动")+(b.completed_this_process!==undefined?" · 本次启动已记录 "+b.completed_this_process+" 个切片 · 待采 "+b.pending_samples:"");
 const sampleNames={OBSERVED:"严格时间窗内观测",AMBIGUOUS_SAMPLE:"时间区间与目标交叠",MISSED_SAMPLE:"漏采 / 迟到",FAILED_OBSERVATION:"采集失败",INVALID_OBSERVATION:"校验未通过"};
 $("telemetrylatest").textContent=q.stream?q.stream+" "+q.outcome+" · T+"+q.target_offset_seconds+"s · "+(sampleNames[q.status]||q.status)+(q.best_ask!==null&&q.best_ask!==undefined?" · 卖一 "+q.best_ask:"")+" · "+when(q.received_at_local||q.target_at):"等待未来窗口；不补历史切片";
 const actual=q.actual_observation||{};
 $("telemetryactual").textContent=actual.received_at_server_center?"实际接收 "+when(actual.received_at_server_center)+" · 卖一 "+actual.best_ask+" · 限价内含费深度 "+money((actual.capacity||{}).usd)+"；不是目标瞬间报价":"等待带真实时间的有效盘口";
 $("telemetryusable").textContent=b.actual_quote_samples_this_process!==undefined?"有效报价 "+b.actual_quote_samples_this_process+" / 费用可核验深度 "+b.actual_depth_samples_this_process+"；严格切片 "+((b.quality_counts_this_process||{}).OBSERVED||0):"—";
 const p=(s.live_preflight||{}).preparation||{};
 $("livepreparation").textContent=p.configuration_safe&&p.risk_limits_match?"共用采集配置与资金上限检查已同步 · 真实执行关闭":"准备检查未完成 · 真实执行关闭";
}
function renderRisk(s){
 const r=s.open_risk||{};
 $("riskactive").textContent=money(r.active_window_cost_usd);
 $("riskpending").textContent=money(r.awaiting_resolution_cost_usd);
 $("riskunknown").textContent=money(r.unknown_cost_usd);
 $("risktotal").textContent=money(r.total_locked_cost_usd);
 const body=$("riskcoins");body.replaceChildren();
 for(const [coin,costs] of Object.entries(r.by_underlying||{})){const row=elem("tr");cell(row,coin);cell(row,money(costs.active_window_cost_usd));cell(row,money(costs.awaiting_resolution_cost_usd));cell(row,money(costs.unknown_cost_usd));cell(row,money(costs.total_locked_cost_usd));body.append(row);}
 $("riskstate").textContent=r.reconciled?"与未结算成本对账一致 · 未释放任何额度":"分类暂不可核验；原资金与风险限制继续生效";
}
function executionText(result){
 if(!result)return "等待新窗口记录";
 const status=result.status;
 if(status==="DISABLED_BY_STAGE")return "本阶段停用 · 不占预算 · 不阻拦";
 if(status==="NO_SIGNAL")return "无信号，已跳过";
 if(status==="NOT_DUE")return "未到决策周期，已跳过";
 if(status==="PRICE_LIMIT_NOT_MET"){
  if(result.best_ask!==undefined&&result.min_buy_price!==undefined&&Number(result.best_ask)<Number(result.min_buy_price))
   return "卖一价 "+result.best_ask+" 低于逆向保护线 "+result.min_buy_price+"，已跳过";
  if(result.min_executable_depth_usd!==undefined&&result.observed_executable_depth_usd!==undefined&&Number(result.observed_executable_depth_usd)<Number(result.min_executable_depth_usd))
   return "限价内可执行深度 "+money(result.observed_executable_depth_usd)+" 低于门槛 "+money(result.min_executable_depth_usd)+"，已跳过";
  const lim=result.limit_price!==undefined?result.limit_price:0.515;
  return "卖一价 "+(result.best_ask!==undefined?result.best_ask+" ":"")+"高于限价 "+lim+"，已跳过";
 }
 if(status==="WAITING_PRICE_RECHECK"){
  if(result.min_executable_depth_usd!==undefined&&result.observed_executable_depth_usd!==undefined&&Number(result.observed_executable_depth_usd)<Number(result.min_executable_depth_usd))
   return "第 "+((result.attempt||0)+1)+" 次检查盘口深度 "+money(result.observed_executable_depth_usd)+" < "+money(result.min_executable_depth_usd)+" · "+when(result.next_retry_at)+" 重检";
  return "第 "+((result.attempt||0)+1)+" 次检查价格未达限价 · "+when(result.next_retry_at)+" 重检 · 下层等待";
 }
 if(status==="BLEED_WINDOW_SINGLE_ORDER_CAP")return "2h阴跌/逼空期同窗口限 1 单护栏触发，已跳过第 2 单";
 if(status==="FILLED_BUDGET")return "本层额度已完成 · "+money(result.spent_usd)+" / "+money(result.approved_usd);
 if(status==="PARTIAL")return "本层未完成 · 已成交 "+money(result.spent_usd)+" / "+money(result.approved_usd)+"，下层等待";
 if(status==="UNREACHED")return "本轮未检查 · 上层或预算已阻断";
 if(status==="ERROR"){
  const reason=result.reason||"";
  if(/budget|position limit|risk/i.test(reason))return "资金或风险额度不足，停止分配";
  if(/higher|upper|partial|unfilled/i.test(reason))return "上层获批额度尚未完成，停止分配";
  if(/stale|future|seconds|clock|expired|window/i.test(reason))return "报价或决策时间检查未通过";
  if(/fee/i.test(reason))return "费用信息未核验，暂停分配";
  return "数据、身份或预测检查未通过";
 }
 return "本轮未完成执行检查";
}
function drawChart(data){
 const svg=$("chart");svg.replaceChildren();const ns="http://www.w3.org/2000/svg";
 const items=Array.isArray(data)?data:(data.points||data.history||[]);
 const rawPts=items.map(p=>{
  const t=Date.parse(p.utc||p.at||p.timestamp||p.recorded_at||p.time||p.observed_at);
  const val=p.equity??p.realized_balance??(typeof p.cash==="number"&&typeof p.locked_cost==="number"?p.cash+p.locked_cost:undefined)??p.equity_usd;
  return {t,v:val===null||val===undefined?NaN:Number(val),raw:p};
 }).filter(p=>Number.isFinite(p.t)&&Number.isFinite(p.v));
 const pts=[];
 for(const p of rawPts){
  if(!pts.length||Math.abs(pts[pts.length-1].v-p.v)>1e-4||p === rawPts[rawPts.length-1])pts.push(p);
 }
 if(!pts.length){$("chartnote").textContent="尚无可绘制的账户权益记录；初始本金不是盈利。";return;}
 const lo=Math.min(...pts.map(p=>p.v)),hi=Math.max(...pts.map(p=>p.v)),spread=Math.max(hi-lo,20);
 const t0=pts[0].t,t1=pts[pts.length-1].t;
 const x=p=>t1===t0?390:65+665*(p.t-t0)/(t1-t0);
 const y=p=>165-120*(p.v-lo+spread*.1)/(spread*1.2);
 const levels=[hi, (hi+lo)/2, lo];
 for(let k=0;k<3;k++){
  const yk=35+k*65;
  const line=document.createElementNS(ns,"line");
  Object.entries({x1:65,x2:735,y1:yk,y2:yk,stroke:"#edf1ef"}).forEach(([a,v])=>line.setAttribute(a,v));
  svg.append(line);
  const lbl=document.createElementNS(ns,"text");
  Object.entries({x:60,y:yk+4,"text-anchor":"end",fill:"#7c8b8c","font-size":"10"}).forEach(([a,v])=>lbl.setAttribute(a,v));
  lbl.textContent="$"+levels[k].toFixed(0);
  svg.append(lbl);
 }
 if(pts.length>1){
  let d="";
  pts.forEach((p,i)=>{d+=(i===0?"M":"L")+x(p).toFixed(1)+","+y(p).toFixed(1)+" ";});
  const areaD=d+"L"+x(pts[pts.length-1]).toFixed(1)+",175 L"+x(pts[0]).toFixed(1)+",175 Z";
  const area=document.createElementNS(ns,"path");
  Object.entries({d:areaD,fill:"rgba(57,134,111,0.10)",stroke:"none"}).forEach(([a,v])=>area.setAttribute(a,v));
  svg.append(area);
  const path=document.createElementNS(ns,"path");
  Object.entries({d,fill:"none",stroke:"#39866f","stroke-width":2.4,"stroke-linejoin":"round","stroke-linecap":"round"}).forEach(([a,v])=>path.setAttribute(a,v));
  svg.append(path);
 }
 pts.forEach((p,idx)=>{
  const dot=document.createElementNS(ns,"circle");
  const isLast=idx===pts.length-1;
  Object.entries({cx:x(p),cy:y(p),r:isLast?4.5:3,fill:isLast?"#186a44":"#39866f",stroke:"#fff","stroke-width":1.2}).forEach(([a,v])=>dot.setAttribute(a,v));
  svg.append(dot);
 });
 const lastPt=pts[pts.length-1];
 $("chartnote").textContent=pts.length===1?"当前只有初始本金记录；有真实模拟账户变化后才产生收益曲线。":"已绘制 "+pts.length+" 个已实现净值节点（起点 "+money(pts[0].v)+" → 当前 "+money(lastPt.v)+"）。";
}
function renderBacktest(s){const rows=(s.backtest||{}).rows||[];const body=$("backtestrows");body.replaceChildren();for(const name of s.allocation.order||[]){const own=rows.filter(x=>x.stream===name);if(!own.length)continue;const r=elem("tr");cell(r,name);for(const v of ["BASE","APPEND376_SAME_FOLD","LOCAL_V4_FROZEN"]){const x=own.find(x=>x.variant===v);cell(r,x?Number(x.raw_win_rate_pct).toFixed(2)+"%":"等待结果");}const v4=own.find(x=>x.variant==="LOCAL_V4_FROZEN");cell(r,v4?money(v4.hypothetical_200_at_0_515_no_fee_pnl):"—");body.append(r);}}
function renderTrades(s){
 const tb=$("tradeorders");if(!tb)return;
 tb.replaceChildren();
 const a=s.account||{};
 const orders=Array.isArray(a.recent_orders)?a.recent_orders:[];
 if($("tradesummary")){
  $("tradesummary").textContent="共 "+(a.filled_order_count||orders.length)+" 笔成交 · 已结算 "+(a.wins||0)+" 胜 / "+(a.losses||0)+" 负 · 累计净收益 "+signedMoney(a.realized_pnl)+"（"+(Number(a.realized_return_pct||0)>0?"+":"")+Number(a.realized_return_pct||0).toFixed(2)+"%）";
 }
 if(!orders.length){
  const r=elem("tr");cell(r,"—");cell(r,"暂无成交");cell(r,"等待信号触发与盘口深度核验");
  for(let i=0;i<7;i++)cell(r,"—");
  tb.append(r);return;
 }
 for(const o of orders){
  const r=elem("tr");
  cell(r,"#"+o.seq_num);
  cell(r,when(o.fill_at));
  cell(r,o.stream||o.underlying||"—");
  cell(r,o.outcome==="UP"?"上涨 (UP)":"下跌 (DOWN)");
  cell(r,Number(o.quantity||0).toFixed(2));
  cell(r,money(o.total_cost_usd)+"（费 "+money(o.fee_usd)+"）");
  const stCell=elem("td");
  const stMap={WON:["已胜 (WON)","tag ok"],LOST:["已负 (LOST)","tag err"],OPEN:["在途待结算","tag"],VOID:["无效退回","tag"]};
  const [stText,stCls]=stMap[o.status]||[o.status||"未知","tag"];
  stCell.append(elem("span",stText,stCls));
  r.append(stCell);
  cell(r,o.payout_usd===null||o.payout_usd===undefined?"等待官方结算":money(o.payout_usd));
  const pnlCls=o.realized_pnl_usd===null||o.realized_pnl_usd===undefined?"":(Number(o.realized_pnl_usd)>0?"pnl-pos":(Number(o.realized_pnl_usd)<0?"pnl-neg":""));
  cell(r,o.realized_pnl_usd===null||o.realized_pnl_usd===undefined?"待结算":signedMoney(o.realized_pnl_usd),pnlCls);
  const cumCls=Number(o.cumulative_pnl_usd)>0?"pnl-pos":(Number(o.cumulative_pnl_usd)<0?"pnl-neg":"");
  cell(r,signedMoney(o.cumulative_pnl_usd)+"（"+money(o.cumulative_balance_usd)+"）",cumCls);
  tb.append(r);
 }
}
function render(s){
 renderBacktest(s);renderTrades(s);
 const pipe=s.pipeline||{};const inputBlocked=(pipe.inputs||{}).status==="BLOCKED";
 const decisions=Object.values(pipe.decisions||{}).filter(x=>x.status!=="DISABLED_BY_STAGE");
 const execution=pipe.execution||{};
 $("lastdecision").textContent=inputBlocked?"最近一轮输入检查未通过，没有生成新预测或模拟成交。":decisions.length?"最近窗口 "+when(pipe.last_attempt_cohort||pipe.last_cohort)+" · "+decisions.length+" 条流状态 · "+decisions.filter(x=>x.status==="SIGNAL").length+" 个信号 · "+(execution.fills||[]).filter(x=>Number(x.quantity)>0).length+" 笔模拟成交"+((execution.blocked||[]).length?" · 容量、报价或风控门槛未通过":""):"等待下一次实时预测";
 const a=s.account;const t=s.training||{};
 if($("tab-badge-monitor")){
  $("tab-badge-monitor").textContent=(s.paper_trading_active?"运行中 · ":"等待 · ")+signedMoney(a.realized_pnl);
  $("tab-badge-monitor").className="tab-pill "+(s.paper_trading_active?"ok":"");
 }
 $("trainingstatus").textContent=({COMPLETED:"本地训练与回测完成",TRAINING:"正在本地训练",FAILED:"训练检查未通过",PREPARING:"准备本地训练"})[t.status]||t.status||"准备中";
 $("trainingdetail").textContent="训练阶段："+(t.phase||"本地训练准备")+(t.current_stream?" · 当前 "+t.current_stream:"")+" · 完成 "+(Array.isArray(t.completed_streams)?t.completed_streams.filter(x=>x.phase==="deployment_q4").length:0)+" 条流";
 $("paperstate").textContent=inputBlocked?"暂停新单 · 输入检查未通过":s.paper_trading_active?"本地模拟执行已接通":s.pipeline&&s.pipeline.status!=="NOT_ENABLED"?"已启动 · 当前等待 / 暂停":"未启动 · 本地训练 / 数据接线中";
 $("connection").className=inputBlocked?"banner":(s.paper_trading_active?"banner ok":"banner");
 $("connection").textContent=inputBlocked?"本地模拟暂停新单："+(pipe.reason||"输入检查未通过"):s.paper_trading_active?"本地模拟运行中 · 仅虚拟资金，真实交易关闭 · 累计已实现收益 "+signedMoney(a.realized_pnl)+"（"+(Number(a.realized_return_pct||0)>0?"+":"")+Number(a.realized_return_pct||0).toFixed(2)+"%）":"本地模型模拟服务："+((s.pipeline||{}).reason||"等待数据和执行检查");
 $("equity").textContent=money(a.equity??a.realized_balance);
 $("cash").textContent=money(a.cash);
 $("pnl").textContent=signedMoney(a.realized_pnl);
 $("pnl").className=Number(a.realized_pnl)>0?"pnl-pos":(Number(a.realized_pnl)<0?"pnl-neg":"");
 const retPct=Number(a.realized_return_pct||0);
 const ddPct=Number(a.max_realized_drawdown_pct||0);
 $("return").textContent="累计收益率 "+(retPct>0?"+":"")+retPct.toFixed(2)+"%"+(ddPct?" · 最大回撤 "+ddPct.toFixed(2)+"%":"");
 $("locked").textContent="未结算仓位成本 "+money(a.locked_cost);
 $("wins").textContent=(a.wins||0)+" 胜 / "+(a.losses||0)+" 负";
 $("winrate").textContent=a.win_rate_pct===null||a.win_rate_pct===undefined?"尚无已结算交易":"结算胜率 "+Number(a.win_rate_pct).toFixed(2)+"%（共 "+((a.wins||0)+(a.losses||0))+" 笔已结算）";
 $("valuation").textContent=a.equity===null?"已实现净值（含在途成本 "+money(a.locked_cost)+"）· 初始 "+money(a.initial_capital):"初始本金 "+money(a.initial_capital);
 if($("header-subtitle")){const rl=a.risk_limits||{};$("header-subtitle").textContent="Polymarket · 共享本金 "+money(a.initial_capital)+(rl.max_order_usd?"（单笔上限 "+money(rl.max_order_usd)+" / 窗口 "+money(rl.max_tick_budget_usd)+" / 总敞口 "+money(rl.max_open_risk_usd)+"）":"")+" · 模拟盘与外部实盘双轨监控";}
 $("since").textContent="起算 "+when(a.started_at);
 $("lastcycle").textContent=when(s.last_cycle);
 $("observing").textContent=s.markets.stale?"观测已过期":(s.markets.market_data_status==="COMPLETE"?"市场与盘口已观测":s.markets.market_data_status==="PARTIAL"?"部分市场 / 盘口不可用":s.last_cycle?"循环运行 · 市场数据不可用":"等待首次采集");
 $("blockers").replaceChildren(...s.blockers.map(b=>elem("div",b.message,"block")));
 $("ranking").textContent=s.allocation.status==="CONFIRMED"?s.allocation.ranking_label:"排名口径待确认，执行关闭";
 const stage=s.stage||{};const disabled=stage.disabled_streams||[];
 if(stage.configured&&stage.status==="CONFIRMED")$("ranking").textContent+=" · 本阶段："+stage.enabled_streams.join(" → ");
 const order=s.allocation.order||s.streams.map(x=>x.stream);
 const tb=$("streams");tb.replaceChildren();
 order.forEach((name,i)=>{
  const row=elem("tr");cell(row,disabled.includes(name)?"—":"#"+(i+1));cell(row,name);
  const state=elem("td");const st=(s.streams||[]).find(x=>x.stream===name);
  state.append(elem("span",disabled.includes(name)?"本阶段停用 · 模型保留":st&&st.model?"本地模型已加载":t.status==="COMPLETED"?"本地训练完成":"本地模型准备中","tag"));
  row.append(state);
  const m=(s.markets.streams||{})[name];const current=m&&m.current;
  cell(row,current?(current.question||current.title||current.slug||"已发现市场"):(m?m.market_status||m.status:"等待市场发现"),"market");
  const up=current&&current.books&&current.books.UP;const down=current&&current.books&&current.books.DOWN;const fee=current&&current.fee;
  cell(row,up&&down&&up.status==="OBSERVED"&&down.status==="OBSERVED"?"UP卖 "+up.best_ask+" / DOWN卖 "+down.best_ask+(fee&&fee.rate!==undefined?" · 费率参数 "+fee.rate:" · 费用未知"):"观察盘口缺失或过期");
  cell(row,executionText(disabled.includes(name)?{status:"DISABLED_BY_STAGE"}:(execution.stream_results||{})[name]));
  tb.append(row);
 });
 const qs=s.reference.quotes||[];$("quotes").closest("section").hidden=!qs.length;
 $("quotes").replaceChildren(...qs.map(q=>{const n=elem("div",undefined,"quote");n.append(elem("b",q.symbol));const age=q.event_at?(Date.now()-Date.parse(q.event_at))/1000:Infinity;const fresh=q.status==="OBSERVED"&&!s.reference.stale&&age>=-2&&age<120;n.append(elem("strong",fresh?money((q.bid+q.ask)/2):"行情不可用 / 过期"));n.append(elem("small",fresh?"买 "+q.bid+" / 卖 "+q.ask:when(q.received_at)));return n;}));
}
function eventView(e){
 const k=e.kind||e.event_type||e.type;const p=e.payload||e.details||{};
 const side=(p.outcome||p.order_outcome)==="UP"?"上涨 (UP)":"下跌 (DOWN)";
 if(k==="ACCOUNT_CREATED")return["账户建立","虚拟本金 "+money(p.initial_capital)];
 if(k==="DECISION_COHORT_CREATED")return["决策窗口","本窗口预算 "+money(p.budget_usd)+" · 每流上限 "+money(p.single_stream_limit_usd??p.single_usd)];
 if(k==="COHORT_STREAM_SKIPPED")return["本层跳过",(p.stream||"")+" · "+executionText({status:p.state||p.decision_status||p.status,best_ask:((p.receipt||{}).best_ask),min_buy_price:((p.receipt||{}).min_buy_price)})];
 if(k==="COHORT_CAPACITY_GATE")return["容量检查",(p.stream||"")+" · "+(p.state==="CAPACITY_LOCKED"?"上层容量未被预算覆盖，低层保持锁定":p.state==="NO_SIGNAL"?"本轮无信号，容量条件已核对":"本轮未到触发周期，容量条件已核对")];
 if(k==="SIGNAL_ADMITTED")return["信号登记",(p.stream||p.underlying||"")+" · "+side+" · 输入检查通过"];
 if(k==="PAPER_IOC")return[p.status==="NO_FILL"?"未成交":"模拟买入",(p.stream||p.underlying||"")+" · "+side+" · 成本 "+money(p.cost_usd)+" · 手续费 "+money(p.fee_usd)+" · 份额 "+Number(p.quantity||0).toFixed(2)];
 if(k==="OFFICIAL_RESOLUTION_APPLIED_TO_PAPER"){
  const tag=(p.order_status==="WON"||Number(p.realized_pnl_usd)>0)?"【胜】":(Number(p.realized_pnl_usd)<0?"【负】":"【平】");
  const coin=p.stream||p.underlying?((p.stream||p.underlying)+" · "+side+" · "):"";
  return["官方结果结算",tag+" "+coin+"返还 "+money(p.payout_usd)+" · 已实现净盈亏 "+signedMoney(p.realized_pnl_usd)];
 }
 return["账户记录","已保存完整审计记录。"];
}
let diag5mState={filter:"ALL",expanded:false,timeline:[],summary:{},live5m:null};

function fmtCohortLabel(iso){
 if(!iso)return "—";
 const d=new Date(iso);
 if(Number.isNaN(d.getTime()))return String(iso);
 const loc=d.toLocaleString("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false});
 const utcHm=String(d.getUTCHours()).padStart(2,"0")+":"+String(d.getUTCMinutes()).padStart(2,"0");
 return loc+" ("+utcHm+" UTC)";
}

function fmtDurSec(s){
 if(typeof s!=="number"||!Number.isFinite(s))return "—";
 const m=Math.floor(s/60),r=Math.floor(s%60);
 return m>0?(m+"分"+String(r).padStart(2,"0")+"秒"):(r+"秒");
}

function tick5mClock(){
 if(!$("diag5m-win-main"))return;
 const nowMs=Date.now();
 const floorMs=Math.floor(nowMs/300000)*300000;
 const elapsedSec=Math.max(0,(nowMs-floorMs)/1000);
 const remSec=Math.max(0,300-elapsedSec);
 const d=new Date(floorMs);
 const locHm=String(d.getHours()).padStart(2,"0")+":"+String(d.getMinutes()).padStart(2,"0");
 const utcHm=String(d.getUTCHours()).padStart(2,"0")+":"+String(d.getUTCMinutes()).padStart(2,"0");
 if(elapsedSec<=20){
  $("diag5m-win-main").textContent="⚡ "+locHm+" 窗定检期 (T+"+elapsedSec.toFixed(0)+"s / 20s)";
  $("diag5m-win-sub").textContent="UTC "+utcHm+" · K线闭合采集与七流瀑布流定检黄金窗";
 }else{
  $("diag5m-win-main").textContent=locHm+" 窗 · 距下窗 "+fmtDurSec(remSec);
  $("diag5m-win-sub").textContent="UTC "+utcHm+" 已过 "+fmtDurSec(elapsedSec)+" · 下一定检整点倒计时中";
 }
}

function render5mTimelineRows(){
 const tb=$("diag5m-tbody");
 if(!tb)return;
 tb.replaceChildren();
 const all=diag5mState.timeline||[];
 const flt=diag5mState.filter||"ALL";
 const filtered=flt==="ALL"?all:all.filter(x=>x.category===flt);
 const limit=diag5mState.expanded?144:12;
 const rows=filtered.slice(0,limit);
 if($("diag5m-expand-btn")){
  $("diag5m-expand-btn").textContent=diag5mState.expanded?"收起至最近 12 窗 (1h)":"展开最近 "+Math.min(all.length||144,144)+" 窗 (12h)";
 }
 if(!rows.length){
  const r=elem("tr");
  cell(r,"—");
  cell(r,"当前筛选下无记录");
  cell(r,"—");
  cell(r,"—");
  cell(r,"切换上方筛选按钮可查看全部 5 分钟窗口定检记录","diag5m-detail");
  tb.append(r);
  return;
 }
 for(const item of rows){
  const r=elem("tr");
  cell(r,fmtCohortLabel(item.cohort));
  const stTd=elem("td");
  stTd.append(elem("span",item.badge_text||item.category||"—","cat-pill cat-"+(item.category||"WARMUP")));
  r.append(stTd);
  const dur=item.duration_seconds;
  cell(r,typeof dur==="number"&&Number.isFinite(dur)?("T+"+dur.toFixed(2)+"s"):(item.category==="IN_PROGRESS"?"执行中…":"—"));
  const pOrds=item.paper_orders||[];
  const lOrds=item.live_orders||[];
  let actText="无下单";
  let actCls="";
  if(pOrds.length||lOrds.length){
   const pSum=pOrds.reduce((acc,o)=>acc+Number(o.total_cost_usd||0),0);
   const lFilled=lOrds.filter(o=>Number(((o.live_execution||{}).filled_quantity)||0)>0);
   const lSum=lFilled.reduce((acc,o)=>acc+Number(((o.live_execution||{}).total_spent_usdc)||0),0);
   const streams=[...new Set([...pOrds.map(o=>(o.stream||"")+" "+(o.outcome||"")),...lOrds.map(o=>(o.stream||"")+" "+(o.outcome||""))])].filter(Boolean).join(",");
   actText=streams+" · 模拟 "+(pOrds.length?money(pSum):"0")+(lOrds.length?(" / 实盘 "+(lFilled.length?money(lSum):(lOrds[0].live_execution||{}).clob_status||"未成")):"");
   actCls="pnl-pos";
  }
  cell(r,actText,actCls);
  cell(r,item.summary_text||"—","diag5m-detail");
  tb.append(r);
 }
}

function render5mDiagnostics(statusRes,liveRes){
 const liveErr=!liveRes||Boolean(liveRes.__live_error);
 const validLive=liveErr?null:liveRes;
 const src=(validLive&&validLive.cohort_timeline&&validLive.cohort_timeline.length)?validLive:statusRes;
 if(!src||!$("diag5m-section"))return;
 const live5m=src.status_5m_live||statusRes.status_5m_live||{};
 const sum=src.cohort_summary||statusRes.cohort_summary||{};
 diag5mState.timeline=src.cohort_timeline||statusRes.cohort_timeline||[];
 diag5mState.summary=sum;
 diag5mState.live5m=live5m;

 const ov=live5m.overall_state||"HEALTHY";
 const pillClsMap={HEALTHY:"diag-pill diag-healthy",IN_WINDOW:"diag-pill diag-inwin",DATA_MISMATCH:"diag-pill diag-warn",RATE_LIMIT:"diag-pill diag-warn",STUCK:"diag-pill diag-err",WARMUP:"diag-pill diag-muted"};
 if($("diag5m-badge")){
  if(liveErr){
   $("diag5m-badge").className="diag-pill diag-warn";
   $("diag5m-badge").textContent="🟠 实盘接口异常";
  }else{
   $("diag5m-badge").className=pillClsMap[ov]||"diag-pill diag-healthy";
   $("diag5m-badge").textContent=live5m.overall_badge||"运行中";
  }
 }
 if($("diag5m-headline")){
  const baseHeadline=live5m.headline||"—";
  $("diag5m-headline").textContent=liveErr
   ?("[告警] 实盘接口 /api/live_dashboard 异常（"+(liveRes&&liveRes.reason?liveRes.reason:"响应为空")+"） · "+baseHeadline)
   :baseHeadline;
 }

 const hb=live5m.heartbeat||{},clk=live5m.clock||{},da=live5m.data_alignment||{},lf=live5m.latest_finished_window;
 const hbAge=hb.pipeline_age_seconds;
 const stalled=Boolean(hb.is_stalled)||ov==="STUCK";
 if($("diag5m-card-hb"))$("diag5m-card-hb").className="diag5m-card"+(stalled?" alert-err":"");
 if($("diag5m-hb-main")){
  $("diag5m-hb-main").textContent=stalled
   ?"🔴 异常 / 阻塞告警"
   :("🟢 正常无卡死"+(typeof hbAge==="number"?"（心跳 "+hbAge.toFixed(1)+"s）":""));
 }
 if($("diag5m-hb-sub")){
  const ph=(hb.active_phases||[])[0];
  const clkTxt=clk.status==="CALIBRATED"?("时钟已校准 ±"+Number(clk.uncertainty_seconds||0).toFixed(2)+"s"):("时钟状态 "+clk.status);
  $("diag5m-hb-sub").textContent=ph?("当前阶段 "+ph.worker+":"+ph.phase+" ("+Number(ph.elapsed_seconds||0).toFixed(1)+"s) · "+clkTxt):(clkTxt+" · 工作线程空闲守候");
 }

 tick5mClock();

 const dataBad=da.state==="MISMATCH"||da.state==="RATE_LIMIT";
 if($("diag5m-card-data"))$("diag5m-card-data").className="diag5m-card"+(dataBad?" alert-warn":"");
 if($("diag5m-data-main"))$("diag5m-data-main").textContent=da.label||"—";
 if($("diag5m-data-sub"))$("diag5m-data-sub").textContent=da.detail||"—";

 if($("diag5m-last-main")){
  if(lf){
   const tHm=new Date(lf.cohort).toLocaleTimeString("zh-CN",{hour:"2-digit",minute:"2-digit",hour12:false});
   $("diag5m-last-main").textContent=tHm+" 窗 · "+(lf.badge_text||lf.category);
   $("diag5m-last-sub").textContent=(typeof lf.duration_seconds==="number"?"开窗 T+"+lf.duration_seconds.toFixed(2)+"s 完成 · ":"")+(lf.summary_text||"").slice(0,86);
  }else{
   $("diag5m-last-main").textContent="等待首个窗口";
   $("diag5m-last-sub").textContent="尚无已完成定检窗口";
  }
 }

 const setCnt=(id,lbl,n)=>{const b=$(id);if(b)b.textContent=lbl+" ("+Number(n||0)+")";};
 setCnt("flt-all","全部窗口",sum.total_cohorts);
 setCnt("flt-nosig","🔵 正常无信号",sum.no_signal_cohorts);
 setCnt("flt-traded","🟢 已成交",sum.traded_cohorts);
 setCnt("flt-risk","🛡️ 风控拦截",sum.risk_guard_cohorts);
 setCnt("flt-data","🟠 数据对不上",sum.data_mismatch_cohorts);
 setCnt("flt-stuck","🔴 卡死/限流/时钟异常",sum.stuck_error_cohorts);

 render5mTimelineRows();
}

function init5mControls(){
 const wrap=$("diag5m-filters");
 if(wrap){
  wrap.addEventListener("click",ev=>{
   const btn=ev.target.closest("button[data-filter]");
   if(!btn)return;
   diag5mState.filter=btn.getAttribute("data-filter")||"ALL";
   for(const b of wrap.querySelectorAll(".flt-btn"))b.classList.toggle("active",b===btn);
   render5mTimelineRows();
  });
 }
 const expBtn=$("diag5m-expand-btn");
 if(expBtn){
  expBtn.addEventListener("click",()=>{
   diag5mState.expanded=!diag5mState.expanded;
   render5mTimelineRows();
  });
 }
 tick5mClock();
 setInterval(tick5mClock,1000);
}

async function refresh(){
 try{
  const liveFetch=fetch("/api/live_dashboard",{cache:"no-store"}).then(async r=>{
   if(!r.ok)return {__live_error:true,reason:"HTTP "+r.status};
   try{return await r.json();}catch(_e){return {__live_error:true,reason:"JSON解析失败"};}
  }).catch(e=>({__live_error:true,reason:(e&&e.message)?String(e.message):"网络请求失败"}));
  const [statusRes,ledgerRes,equityRes,liveRes]=await Promise.all([
   fetch("/api/status",{cache:"no-store"}).then(r=>{if(!r.ok)throw Error("HTTP");return r.json();}),
   fetch("/api/ledger",{cache:"no-store"}).then(r=>{if(!r.ok)throw Error("HTTP");return r.json();}),
   fetch("/api/equity",{cache:"no-store"}).then(r=>{if(!r.ok)throw Error("HTTP");return r.json();}),
   liveFetch
  ]);
  render5mDiagnostics(statusRes,liveRes);
  if(liveRes&&liveRes.__live_error)renderLiveError(liveRes.reason);
  else if(liveRes)renderLive(liveRes);
  renderEngineering(statusRes);render(statusRes);drawChart(equityRes);
  const tradeEvents=((statusRes.account||{}).trade_events)||[];
  const rawRows=Array.isArray(ledgerRes)?ledgerRes:(ledgerRes.events||ledgerRes.ledger||[]);
  const rows=tradeEvents.length?tradeEvents:rawRows;
  const body=$("ledger");body.replaceChildren();
  if(!rows.length){const r=elem("tr");cell(r,"—");cell(r,"暂无交易");cell(r,"模型未接通期间，账户不产生模拟成交或利润。");body.append(r);}
  for(const event of rows.slice(0,30)){const r=elem("tr");cell(r,when(event.utc||event.at||event.timestamp||event.recorded_at||event.created_at));const view=eventView(event);cell(r,view[0]);cell(r,view[1]);body.append(r);}
 }catch(e){
  $("connection").className="banner offline";
  $("connection").textContent="本地服务未连接，以下为上次读取状态，不代表当前实时数据。";
  $("observing").textContent="断线 / 未知";
  if($("diag5m-badge")){
   $("diag5m-badge").className="diag-pill diag-err";
   $("diag5m-badge").textContent="🔴 本地监控服务已断线 / 未响应";
  }
  if($("diag5m-headline")){
   $("diag5m-headline").textContent="告警：无法连接 http://127.0.0.1:8897/api/status，本地 Python 模拟/看板进程可能已停止或卡死！";
  }
  if($("diag5m-card-hb"))$("diag5m-card-hb").className="diag5m-card alert-err";
  if($("diag5m-hb-main"))$("diag5m-hb-main").textContent="🔴 服务断线 / 无响应";
 }
}
initTabs();
init5mControls();
refresh();setInterval(refresh,5000);
