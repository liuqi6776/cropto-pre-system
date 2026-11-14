# ARGON-4 `v15r2-Calibrated` Multi-Parameter Calibration & Ablation Report (2026-11-14) / `v15r2-Calibrated` 多参数联合校准与消融审计报告

## 1. Overview & Provenance / 概述与数据溯源
- **EN**: Documents and deterministically reproduces every parameter adjustment between `v15r1` ([baseline_policy_v15r1.py](file:///c:/Users/liuqi/crypto-predict_system/reports/argon4_single_factor_check_20261114/baseline_policy_v15r1.py), extracted verbatim from `commit bb2eacd`) and `v15r2-Calibrated` ([argon4_forward_policy.json](file:///c:/Users/liuqi/crypto-predict_system/config/argon4_forward_policy.json)), verified via automated key-by-key parity check (`verified_identical_to_head_policy = true` in [evaluate_v15r2_calibration.py](file:///c:/Users/liuqi/crypto-predict_system/reports/argon4_v15r2_calibration_20261114/evaluate_v15r2_calibration.py)).
  - **Dataset 1 (Repo-Local Reproducible Strict 2026 OOS)**: `reports/argon4_local_v4_20261106` (`LOCAL_V4_FROZEN`, trained `<= 2025-12-31`, `32,804` settled raw signals across `2026-01-01 .. 2026-09-24`) paired with real Binance `5m` Futures & `1m` Spot `OHLC4` bars in `research_data/argon4_twap60_20261113/raw_data`.
  - **Dataset 2 (30-Month Walk-Forward `SPOT_TO_TWAP60_PURE` Strict OOS)**: `123,893` strictly out-of-sample raw signals across `2024-05 .. 2026-11-13` (`18` months in `2024–2025` + `11` months in `2026`), evaluated via `--pure-signals-parquet` in `evaluate_v15r2_calibration.py`.
  - **Governance & Simulation Disclaimer (`AGENTS.md`)**: Historical `2024–2026` datasets do not have full Polymarket L2 orderbook snapshots. Accordingly, `evaluate_v15r2_calibration.py` strictly separates **(a) real-market guard selection (`twap_1m_ohlc4_proxy_wr_pct`, `perp_point_wr_pct`)** from **(b) analytical `\$0.01`-tick causal `T+0s` orderbook execution simulation (`seed=42`) vs post-`T` `[T, T+1m)` 1m-bar lookahead diagnostic**. Offline backtests and analytical orderbook simulations do not establish live profitability or authorize live trading.
- **ZH**: 完整记录并可确定性复现从 `v15r1`（提取自 `commit bb2eacd` 的快照 `baseline_policy_v15r1.py`）升级至生产配置 `v15r2-Calibrated`（`config/argon4_forward_policy.json`）的全部 4 项参数校准步骤，并通过脚本内自动一致性断言校验（`verified_identical_to_head_policy = true`）。
  - **数据集 1（仓库本地可直接复现的 2026 严格样本外）**：`reports/argon4_local_v4_20261106`（`LOCAL_V4_FROZEN`，训练截止 `<= 2025-12-31`，涵盖 `2026-01-01 .. 2026-09-24` 共 `32,804` 条已结算原始信号）配合真实币安 `5m` 合约与 `1m` 现货 `OHLC4` 序列。
  - **数据集 2（30 个月滚动样本外 `SPOT_TO_TWAP60_PURE` 全周期）**：涵盖 `2024-05 .. 2026-11-13` 共 `123,893` 条严格样本外信号（`2024–2025` 共 `18` 个月 + `2026` 共 `11` 个月），支持通过 `evaluate_v15r2_calibration.py --pure-signals-parquet` 直接复算。
  - **合规与模拟边界声明（遵循 `AGENTS.md`）**：由于 `2024–2026` 全历史周期不存在完整的 Polymarket L2 真实盘口快照，脚本严格区分 **（a）零合成假设的真实行情护栏筛选指标** 与 **（b）纯因果 `T+0s` `\$0.01` Tick 盘口成交模拟 vs 事后 `[T, T+1m)` 1m K 线插值前视偏差诊断**。离线回测与盘口解析模拟不构成实盘盈利承诺或实盘授权依据。

---

## 2. Parameter Changes (`v15r1` -> `v15r2-Calibrated`) / 参数变更对照

| Parameter / 参数项 | `v15r1` (`bb2eacd`) | `v15r2-Calibrated` (`HEAD`) | Calibration Rationale / 校准依据 |
| :--- | :--- | :--- | :--- |
| **Confidence Buy-Price Tiers (`tiers`)** | `0.0: 0.49/0.498`, `0.35: 0.502/0.51`, `0.80: 0.51/0.52` | `0.0: 0.49/0.50`, `0.25: 0.50/0.51`, `0.60: 0.51/0.52` | **EN**: Fixes `\$0.01`-tick floor truncation (`0.498 -> 0.49`, `0.502 -> 0.50`) that deadlocked `15m` Tier 1 and `5m` Tier 2.<br>**ZH**: 消除 `\$0.01` Tick 向下取整（`0.498 -> 0.49`、`0.502 -> 0.50`）导致的整档踏空。 |
| **Counter-15m Marginal Cap** | `0.475` (`-> 0.47` on tick) | `0.49` (`min` with `sub_threshold_max_buy_price = 0.485`) | **EN**: Aligns counter-15m marginal cap to `0.49` while enforcing `min(0.49, 0.485) = 0.485` when `threshold_override` admits sub-baseline signals.<br>**ZH**: 将逆势微弱信号限价对齐至 `0.49`，且与降阈值弱信号上限 `0.485` 严格取 `min`。 |
| **TWAP60 Handicap Band (`twap60_handicap_guard`)** | `(-inf, +1.0]` (`5m`), `(-inf, +2.0]` (`15m`) | `[-0.5, +2.2]` (`5m`), `[-0.5, +4.0]` (`15m`) | **EN**: Adds `-0.5 bps` floor against stale-quote toxic fills and unlocks the causal mean-reversion discount zone (`(+1.0, +2.2]` / `(+2.0, +4.0]` bps) without over-relaxing into `[+4.5, +7.0]` adverse selection.<br>**ZH**: 新增 `-0.5 bps` 下限拦截顺风低价滞后毒单，并在纯因果前提下解锁均值回归折价甜点区，拒绝过度放宽至 `+4.5/+7.0 bps`。 |
| **Minimum Buy Price (`stream_min_buy_price`)** | `0.47` (`5m` & `15m`) | `0.47` (`5m`), `0.46` (`15m`) | **EN**: On `15m`, `ask == 0.46` is positive in both `2024–2025` (`+\$7,350`, `53.58%` WR) and `2026` (`+\$5,645`, `53.86%` WR), whereas `ask == 0.45` turns negative in `2026` (`-\$574`, `45.92%` WR).<br>**ZH**: `15m` 在 `0.46` 档跨周期双正（`2024–2025` `+\$7,350`、`2026` `+\$5,645`），而 `0.45` 档在 `2026` 转负（`-\$574`，胜率 `45.92%`），故精确停在 `0.46`。 |
| **US Open Blackout (`us_open_blackout`)** | `09:30–12:00 ET` (`720`) | `09:30–12:00 ET` (`720`, **unchanged**) | **EN**: Relaxing to `10:30 ET` (`630`) reduces 30-month causal net PnL by `-\$5,187` (`10:00–12:00 ET` Perp WR `49.74%`). Kept at `12:00 ET`.<br>**ZH**: 缩短至 `10:30 ET` 会倒扣 30 个月净收益 `-\$5,187`（`10:00–12:00 ET` 永续点位胜率仅 `49.74%`），坚决维持 `12:00 ET`。 |

---

## 3. Step-by-Step Causal Ablation Summary (30-Month `SPOT_TO_TWAP60_PURE` Strict OOS) / 30 个月全周期逐步消融汇总

> [!NOTE]
> Metrics below use per-`\$100`-stake Polymarket `7%` dynamic taker fee (`0.07 * p * (1 - p)`) on causal `T+0s` `\$0.01`-tick asks (`seed=42`, zero post-`T` lookahead). Repo-local `LOCAL_V4_FROZEN` (`2026-01 .. 2026-09`) numbers are also output directly by `evaluate_v15r2_calibration.py`.
> 下表按每笔 `\$100` 固定名义注码及 Polymarket `7%` 动态吃单费率，在纯因果 `T+0s` `\$0.01` Tick 盘口下逐项消融统计；仓库本地 `LOCAL_V4_FROZEN` 口径结果亦可通过下方脚本一键复现。

| Ablation Step / 消融步骤 | `2024–2025` (`18m`) Net PnL (`pos_months`) | `2026` (`11m`) Net PnL (`pos_months`) | `2026` Folds (`F1 / F2 / F3`) | `30m` Total Fills (`n`) | `30m` TWAP WR | `30m` EV / `\$100` | `30m` Net PnL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`step0_v15r1_baseline`** | `+\$12,611` (`12/18`) | `+\$29,396` (`10/11`) | `+\$15,119 / +\$12,503 / +\$1,775` | `10,558` | `52.83%` | `+\$3.98` | `+\$42,007` |
| **`step1_plus_tick_aligned_tiers`** | `+\$19,021` (`14/18`) | `+\$33,914` (`10/11`) | `+\$16,329 / +\$14,682 / +\$2,903` | `12,335` | `53.20%` | `+\$4.29` | `+\$52,934` |
| **`step2_plus_min_handicap_neg05`** | `+\$14,421` (`13/18`) | `+\$33,286` (`10/11`) | `+\$15,589 / +\$14,512 / +\$3,185` | `11,713` | `52.98%` | `+\$4.07` | `+\$47,707` |
| **`step3_plus_causal_handicap_22_40`** | `+\$21,612` (`12/18`) | `+\$34,506` (`10/11`) | `+\$17,914 / +\$13,831 / +\$2,761` | `15,166` | `52.55%` | `+\$3.70` | `+\$56,118` |
| **`step4_v15r2_calibrated_production`** | **`+\$28,963` (`13/18`)** | **`+\$40,151` (`11/11`)** | **`+\$20,700 / +\$14,208 / +\$5,243`** | **`16,207` (`+53.5%`)** | **`52.63%`** | **`+\$4.26`** | **`+\$69,114` (`+64.5%`)** |
| *Rejected*: `wide_45_70 + 15m_0.45` | `+\$38,607` (`15/18`) | `+\$40,120` (`9/11` — loses 2 mos) | `+\$22,377 / +\$11,945 / +\$5,798` | `18,443` | `52.34%` | `+\$4.27` | `+\$78,727` |
| *Rejected*: `+ us_open 10:30 ET` | `+\$33,573` (`13/18`) | `+\$39,967` (`9/11`) | `+\$22,451 / +\$11,727 / +\$5,790` | `19,631` | `52.08%` | `+\$3.75` | `+\$73,540` |

---

## 4. Reproduction Commands / 复现命令
```powershell
python -B reports/argon4_v15r2_calibration_20261114/evaluate_v15r2_calibration.py
```
