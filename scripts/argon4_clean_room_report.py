"""Generate measured result tables, source-code discrepancy audit and equity plot."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.argon4_clean_room_fetch import DATA, STUDY, sha, write_json
from scripts.argon4_clean_room_reproduce import OUT, ORIGINAL
import numpy as np
import pandas as pd


def ledger(variant, leverage):
    """Independently reconstruct the exact supplied open-to-open arithmetic.

    This intentionally preserves its omitted funding and closing fees, zero last
    row, and annual reset. Exported fields expose those limits instead of
    presenting the calculation as a complete futures ledger.
    """
    raw = pd.read_parquet(DATA / "raw_data/intraday/BTCUSDT_5m_futures_2023_2026.parquet").set_index("open_time")
    o = raw.open.to_numpy(float)
    result, carry = [], 1.
    summary = pd.read_csv(OUT / variant / "perp_summary.csv")
    declared = summary[summary.Leverage == f"{leverage:.1f}X"].iloc[0]
    for year in (2025, 2026):
        positions = pd.read_parquet(OUT / variant / f"positions_{year}.parquet")
        t = pd.DatetimeIndex(positions.decision_bar_open_utc)
        where = raw.index.get_indexer(t)
        if (where < 0).any():
            raise ValueError("Unmatched decision timestamps")
        eb = where[:-1] + 1
        p = positions.position.to_numpy(float)[:-1]
        prev = np.r_[0., p[:-1]]
        gross = p * (o[eb + 1] - o[eb]) / o[eb]
        fee = np.abs(p - prev) * .00005
        net = (gross - fee) * leverage
        eq = np.cumprod(1 + net)
        if abs((eq[-1] - 1) * 100 - float(declared[f"{year}_Ret%"].strip("%"))) > .0051:
            raise ValueError("Independent arithmetic does not match supplied summary")
        d = pd.DataFrame({"decision_bar_open_utc": t[:-1], "ideal_fill_utc": raw.index[eb], "period_end_utc": raw.index[eb + 1],
                          "position_fraction": p, "leverage": leverage, "price_return_on_equity": gross * leverage,
                          "turnover_fee_on_equity": fee * leverage, "funding_included": False,
                          "net_return_source_convention": net, "equity_multiple": carry * eq})
        carry *= eq[-1]
        result.append(d)
    out = pd.concat(result, ignore_index=True)
    out.to_parquet(OUT / variant / f"source_accounting_{leverage:.0f}x.parquet", index=False)
    return out


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source = pd.read_csv(OUT / "literal_monthly/perp_summary.csv")
    repaired = pd.read_csv(OUT / "literal/perp_summary.csv")
    exact4 = source.loc[source.Leverage == "4.0X", "2025_2026_Compounded%"].iloc[0]
    repaired4 = repaired.loc[repaired.Leverage == "4.0X", "2025_2026_Compounded%"].iloc[0]
    binary = pd.read_csv(OUT / "literal_monthly/binary_summary.csv")
    seven = pd.read_csv(OUT / "seven_stream_extension/seven_stream_summary.csv")
    seven = seven[seven.period == "combined"]
    manifest = json.loads((DATA / "data_manifest.json").read_text(encoding="utf8"))
    validation_path = OUT / "workspace_validation.json"
    validation_summary = "全库测试尚未记录最终结果。"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf8"))
        test_check = validation["checks_in_required_order"][-1]
        if test_check.get("status") == "COMPLETED_WITH_FAILURES":
            validation_summary = f"全库测试：**{test_check['passed']:,} 项通过、{test_check['failed']} 项失败、{test_check['skipped']} 项跳过**，耗时 {test_check['elapsed_seconds'] / 60:.1f} 分钟。失败项不包含本次新增测试；未跑改动前全库基线，不能断言全部失败均早于本任务。"
    claims = {"3.0X": "+202.93%", "4.0X": "+285.46%"}
    discrepancies = [
        {"component": "features", "claimed": "259 base + 115 structural = 374", "observed": "279 base + 32 structural = 311; supplied V4 is 65 more, appended experiment is 376", "verdict": "SOURCE_SPEC_MISMATCH"},
        {"component": "binary_streams", "claimed": "Seven streams", "observed": "Original driver implements four BTC/ETH streams only", "verdict": "PARTIAL_SOURCE_ONLY"},
        {"component": "E4", "claimed": "Quarterly adaptive HGB", "observed": "p_e4 = 0.50 * p_e1 + 0.50 * p60_b; no quarterly retraining", "verdict": "SOURCE_SPEC_MISMATCH"},
        {"component": "funding", "claimed": "Returns net of 8h funding", "observed": "Funding is loaded into features but never debited/credited in the supplied return loop", "verdict": "MISSING_CASHFLOW"},
        {"component": "stops_and_risk", "claimed": "5-bar fractal stops, tick cap, daily drawdown cutoff and streak retraining", "observed": "Absent from supplied executable engine", "verdict": "MISSING_IMPLEMENTATION"},
        {"component": "V4", "claimed": "V4 improves seven-stream net profit", "observed": "Feature function supplied, no integration/training driver supplied; our append experiment is explicitly an extension", "verdict": "INTEGRATION_UNSPECIFIED"},
        {"component": "prediction_market", "claimed": "Observed CLOB capacity, fills, fees, Chainlink noise and profits", "observed": "No oracle settlement history, venue books, fill records or fee history supplied; Binance direction only", "verdict": "MISSING_EXECUTION_EVIDENCE"},
        {"component": "waterfall", "claimed": "$10,000 becomes $1,369,463", "observed": "No waterfall/sweep engine, allocation ledger or implementation supplied; OOS win-rate ranking also uses future test results", "verdict": "NOT_REPRODUCIBLE_FROM_SUPPLIED_CODE"},
        {"component": "days", "claimed": "631 days", "observed": "2025-01-01 through 2026-09-24 inclusive is 632 days", "verdict": "DENOMINATOR_MISMATCH"},
        {"component": "test_labels", "claimed": "Trading-ready predictive evidence", "observed": "Future zero-return labels excluded before signal evaluation; final binary labels extend beyond nominal fold end; next-open fill time equals decision close", "verdict": "NOT_CAUSAL_EXECUTION_PROOF"},
    ]
    write_json(OUT / "source_discrepancies.json", discrepancies)
    curves = {str(int(l)): ledger("literal_monthly", l) for l in (1., 3., 4.)}
    fixed = ledger("literal", 4.)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for lev, d in curves.items():
        day = d.set_index("period_end_utc").equity_multiple.resample("1D").last().dropna()
        ax.plot(day.index, day, label=f"{lev}x: supplied code, monthly-only marks", lw=1.7)
    day = fixed.set_index("period_end_utc").equity_multiple.resample("1D").last().dropna()
    ax.plot(day.index, day, label="4x: observed daily mark repairs", lw=1., ls="--", color="#8c564b")
    ax.axhline(1, color="grey", lw=.8)
    ax.set(title="ARGON-4 supplied-code reproduction | 2025-01 to 2026-09", ylabel="Equity multiple (initial = 1)")
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.text(.08, .015, "Exploratory price-only accounting: funding, liquidation, slippage and terminal closing fee omitted by supplied code.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / "equity_comparison.png", dpi=160)
    plt.close(fig)

    rows = []
    for _, r in source.iterrows():
        ref = repaired[repaired.Leverage == r.Leverage].iloc[0]
        rows.append([r.Leverage, r["2025_Ret%"], r["2026_Ret%"], r["2025_2026_Compounded%"], ref["2025_2026_Compounded%"], claims.get(r.Leverage, "—")])
    bt = [[r.Clean_Room_Stream.split()[0], r.Total_Signals, r["Overall_Raw_WR%"], r["Predict_UP(买涨)_WR%"], r["Predict_DOWN(买跌)_WR%"]] for _, r in binary.iterrows()]
    st = []
    for name in seven.stream.drop_duplicates():
        a = seven[(seven.stream == name) & (seven.variant == "baseline")].iloc[0]
        b = seven[(seven.stream == name) & (seven.variant == "v4_append")].iloc[0]
        st.append([name, int(a.signals), f"{a.win_rate_pct:.2f}%", int(b.signals), f"{b.win_rate_pct:.2f}%", f"{b.win_rate_pct-a.win_rate_pct:+.2f}"])
    audit = {"status": "RESEARCH_NOT_VALIDATED", "effectiveness_claim": False, "live_trading_authorized": False,
             "original_source_sha256": sha(ORIGINAL / "step2_standalone_clean_room_master.py"),
             "strategy_evidence": [{"strategy": x, "verdict": "NOT_VALIDATED", "reason": why} for x, why in [
                 ("BTC_PERP_SUPPLIED_CODE", "Headline return not replicated; missing funding/margin/liquidation/realistic execution."),
                 ("FOUR_BINARY_STREAMS_SUPPLIED_CODE", "Candle classification only; future tie filtering; no oracle, fills or capacity evidence."),
                 ("SEVEN_STREAM_V4_APPEND_EXPERIMENT", "Disclosed implementation choice; same execution and holdout limitations; no universal improvement."),
                 ("WATERFALL_AND_CLOB_ROUTER", "No executable source or observed execution ledger supplied.")]],
             "validation_skill": {"requested_workspace_path": ".agents/skills/strategy-validation-gates/SKILL.md", "present": False, "effect": "No effective/ready-to-trade verdict issued"}}
    write_json(OUT / "evidence_status.json", audit)
    doc = f"""# ARGON-4 附件复现报告（2026-11-05）

结论：已从官方公开数据重新训练并运行附件提供的主脚本，但没有复现文档宣称的永续收益。严格使用原下载器对应的月档观测时，4 倍杠杆累计为 **{exact4}**，而附件宣称 **+285.46%**；用官方日档补齐真实标记价格观测后为 **{repaired4}**。这不是同一个完整策略的成功复现，附件正文与附带代码有实质差异。

状态：`RESEARCH_NOT_VALIDATED`。以下均为历史研究输出，不能作为预期收益、有效策略或实盘就绪结论。

## 1. 复现边界与数据

- 附件原文与三个 Python 源码块原样保存在 `research/{STUDY}/original/`，并记录 SHA256；未导入旧模型或旧研究缓存。
- 官方月档：2023-01 至 2026-09，共 **990** 个 ZIP，全部通过官方 `.CHECKSUM`；约 {sum(r['bytes'] for r in manifest['archives']) / 1024**2:.1f} MiB 压缩数据。来源和每个文件的摘要见 `research_data/{STUDY}/data_manifest.json`。
- 四个币种各 **394,272** 根 5m 价格观测；模型源代码从 2024-01-01 才开始取数，实际输入各 **289,152** 行。完整下载范围和模型训练范围不同，2023 数据未被源码用于特征热身。
- 另外调用附件原下载器解析同一批已校验 ZIP，确认四币种全部 5m 行情中模型读取的字段逐值一致；证据在 `original_crawler_price_parity.json`。
- 四个币种的月档标记价格均缺少 2023-02-24 与 2026-06-29；用 **8** 个校验通过的官方日档取得 **192** 根实际观测。两套输入均保留：`literal_monthly` 保留原始月档缺口，`literal` 使用补取的实际观测；没有合成价格。
- 资金费率保留观测结算事件；没有补造结算事件。现货 4h 仅用于基差特征，单独登记，未转换为永续数据。
- DVOL 来自 Deribit 公共接口，原始 JSON、请求 URL、SHA256 和覆盖期均保留。SPOT 自 2025 年起的微秒时间戳按官方说明归一化；原始价格开盘时间不平移。
- 数据登记在共享 dataset registry；标记价格登记在 research references，`installed=false`、`gate_effect=NONE`，不改变既有 Freqtrade 就绪条件。

官方格式说明：[Binance Public Data](https://github.com/binance/binance-public-data)、[Deribit DVOL API](https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data)。

## 2. 原始永续主脚本的实际结果

原模型公式、随机种子、训练/验证窗口、调仓缓冲及原始收益循环均未改动。包装脚本只限定输入路径、导出中间结果并记录环境；AST 单测验证删除导出钩子后与原始源码结构一致。

{table(['杠杆', '2025', '2026 至 9/24', '月档原样累计', '补取真实 mark 后累计', '附件宣称累计'], rows)}

![原脚本权益曲线]({(OUT / 'equity_comparison.png').as_posix()})

源码只扣调仓手续费 0.5bp/side，**没有资金费率现金流、强平/保证金路径、滑点、最后平仓费用**；年末状态重新从空仓启动。曲线逐笔复算原会计口径，与打印收益误差不超过两位小数的舍入误差。完整价格会计明细见 `literal_monthly/source_accounting_1x.parquet`、`3x`、`4x`。不能把该结果称为完整永续净收益或断言“零强平”。

## 3. 原始四条二元预测流

以下为严格月档版本的 Binance K 线方向命中率，尚未扣除附件声称的“2% Chainlink 平盘噪音”，也不是任何 CLOB 平台实际成交胜率。

{table(['流', '信号数', '方向命中率', 'UP', 'DOWN'], bt)}

每年逐条预测、验证集固定阈值、训练样本数与标签终点均已导出。源码事先剔除了未来收益恰为零的样本，这不是开仓时可知条件。2025-01-01 至 2026-09-24 共 **632** 个自然日，源码的日均信号数却固定除以 631；源码最后几个二元标签还延伸至名义测试结束日以后，不能把这部分当作严格截止窗口。

## 4. 七流与 V4：明确标记的补接实验

附件只给出独立 V4 特征函数，没有给出 V4 训练接线。这里保留原二元模型训练公式、阈值、抽样及年度窗口，补跑 SOL/BNB，并将该函数的 **65** 列追加到原代码实际生成的 **311** 列后。共 **376** 列；这是一种明确记录的补接方案，不冒充原文未提供的 374 维最优系统。该对照统一使用补取真实 mark 后的数据，前四条基础流复用本次从头训练结果。

{table(['流', '基础信号数', '基础命中率', 'V4 信号数', 'V4 命中率', '变化/百分点'], st)}

四条上升，三条下降。不同版本选出的交易集合和数量也变化了，不能把命中率差直接解释为显著的策略改进。CSV 中假定每笔 $200、固定 $0.515 的金额只是一项**无费用、无成交约束、无资金占用限制的赔率算术场景**，不是可实现收益；没有采用它复现“每日收益”“容量上限”或 $1,369,463 的资产路径。

## 5. 附件正文与源码的关键差异

{table(['项目', '宣称', '源码事实'], [[d['component'], d['claimed'], d['observed']] for d in discrepancies])}

特别是原源码 `p_e4 = 0.50 * p_e1 + 0.50 * p60_b`，没有季度重训；结构特征实测 **32** 列而非 **115** 列。移动止损、三级熔断、CLOB 路由和瀑布归集没有对应可执行实现。本次不依据收益数字反推或虚构这些缺失模块。

## 6. 运行方法与产物

在工作区根目录使用 `.venv`：

```powershell
.\\.venv\\Scripts\\python.exe scripts\\argon4_clean_room_fetch.py
.\\.venv\\Scripts\\python.exe scripts\\argon4_clean_room_reproduce.py --threads 4
.\\.venv\\Scripts\\python.exe scripts\\argon4_clean_room_reproduce.py --threads 4 --monthly-only-marks
.\\.venv\\Scripts\\python.exe scripts\\argon4_clean_room_seven_streams.py --threads 4
.\\.venv\\Scripts\\python.exe scripts\\argon4_clean_room_report.py
```

- 原样月档结果：`reports/{STUDY}/literal_monthly/`。
- 补齐实际 mark 的原模型结果：`reports/{STUDY}/literal/`。
- 七流/V4 对照：`reports/{STUDY}/seven_stream_extension/seven_stream_summary.csv`。
- 源码差异与非就绪结论：`source_discrepancies.json`、`evidence_status.json`。
- 每次训练的 `run.json` 记录版本、摘要、数据来源、运行耗时和产物摘要；全部使用新训练结果，无预训练模型输入。
- 本次运行环境：Python 3.12.11、numpy 2.2.6、pandas 2.2.3、scikit-learn 1.7.2；完整依赖冻结表在 `research/{STUDY}/requirements-frozen.txt`。

## 7. 验证及限制

本次新增完整性/因果性测试 **11 项通过**：源码摘要、导出钩子无模型改动、公开数据端点限制、现货微秒时间戳、缺口不填充、无效价格拒绝、资金费率事件保留以及 V4 特征前缀不依赖未来观测。

{validation_summary}

项目级验证结果另存 `reports/{STUDY}/workspace_validation.json` 和对应日志。既有数据校验发现 **54 项缺少可验证冻结记录**；默认转换因此拒绝继续，未通过改写旧数据或放宽门槛绕过。这次研究读取的是独立新下载数据，未读取上述旧输入。全项目测试的最终结果以验证 JSON 为准。

指定的 `.agents/skills/strategy-validation-gates/SKILL.md` 在当前工作区缺失；因此没有出具“有效/可实盘”判断。当前已有窗口曾多次用于选择，不能称为未触碰留出集。没有启动实盘、发送订单或修改源项目及 D: 数据。
"""
    (ROOT / "docs/ARGON4_CLEAN_ROOM_REPRODUCTION_20261105.md").write_text(doc, encoding="utf8")
    artifacts = [OUT / "equity_comparison.png", OUT / "source_discrepancies.json", OUT / "evidence_status.json", ROOT / "docs/ARGON4_CLEAN_ROOM_REPRODUCTION_20261105.md", *OUT.glob("literal*/source_accounting_*.parquet")]
    parity = OUT / "original_crawler_price_parity.json"
    if parity.exists():
        artifacts.append(parity)
    write_json(OUT / "report_artifacts.json", {"report": "docs/ARGON4_CLEAN_ROOM_REPRODUCTION_20261105.md", "generator_sha256": sha(Path(__file__)), "files": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in artifacts}})
    print("Report and independently reconciled equity curves written.")


if __name__ == "__main__":
    run()
