"use strict";
const $=id=>document.getElementById(id);
const money=v=>v===null||v===undefined?"待估值":new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(v);
const signedMoney=v=>v===null||v===undefined?"待结算":(Number(v)>0?"+":"")+money(Number(v));
const when=v=>v?new Date(v).toLocaleString("zh-CN",{hour12:false}):"尚无观测";
function elem(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n;}
function cell(row,text,cls){row.appendChild(elem("td",text,cls));}
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
  return "买价高于 0.515，已跳过";
 }
 if(status==="WAITING_PRICE_RECHECK")return "第 "+((result.attempt||0)+1)+" 次检查价格未达限价 · "+when(result.next_retry_at)+" 重检 · 下层等待";
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
 if($("header-subtitle")){const rl=a.risk_limits||{};$("header-subtitle").textContent="Polymarket · 共享本金 "+money(a.initial_capital)+(rl.max_order_usd?"（打法 B：单笔上限 "+money(rl.max_order_usd)+" / 窗口 "+money(rl.max_tick_budget_usd)+"）":"")+" · 从账户创建时刻开始";}
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
async function refresh(){
 try{
  const results=await Promise.all(["status","ledger","equity"].map(p=>fetch("/api/"+p,{cache:"no-store"}).then(r=>{if(!r.ok)throw Error("HTTP");return r.json();})));
  renderEngineering(results[0]);render(results[0]);drawChart(results[2]);
  const tradeEvents=((results[0].account||{}).trade_events)||[];
  const rawRows=Array.isArray(results[1])?results[1]:(results[1].events||results[1].ledger||[]);
  const rows=tradeEvents.length?tradeEvents:rawRows;
  const body=$("ledger");body.replaceChildren();
  if(!rows.length){const r=elem("tr");cell(r,"—");cell(r,"暂无交易");cell(r,"模型未接通期间，账户不产生模拟成交或利润。");body.append(r);}
  for(const event of rows.slice(0,30)){const r=elem("tr");cell(r,when(event.utc||event.at||event.timestamp||event.recorded_at||event.created_at));const view=eventView(event);cell(r,view[0]);cell(r,view[1]);body.append(r);}
 }catch(e){
  $("connection").className="banner offline";
  $("connection").textContent="本地服务未连接，以下为上次读取状态，不代表当前实时数据。";
  $("observing").textContent="断线 / 未知";
 }
}
refresh();setInterval(refresh,10000);
