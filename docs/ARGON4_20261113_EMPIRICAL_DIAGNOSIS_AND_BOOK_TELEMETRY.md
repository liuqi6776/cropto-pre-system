# ARGON-4 Empirical Diagnosis & Public Orderbook Telemetry Report (2026-11-11 to 2026-11-13 UTC)
# ARGON-4 模拟盘实证归因与公开盘口遥测数据集报告（2026-11-11 至 2026-11-13 UTC）

**Date / 报告日期**: 2026-11-14 UTC  
**Scope & Sanitization / 范围与脱敏声明**: Public market data (`Polymarket` `market_id`, official `Gamma` UMA resolutions, recorded public CLOB orderbook `best_ask` & depth, public Binance USD-M `5m` futures OHLC) and isolated forward paper simulation (`paper_runtime` `$1,269.00` initial capital). **Contains zero credentials, zero wallet addresses, zero live balances, and zero private ledgers.**  
本文档仅包含公开市场行情与盘口数据（`Polymarket` `market_id`、`Gamma` 官方 UMA 结算结果、实录 CLOB 盘口深度与币安合约公开 K 线）及隔离模拟盘统计（初始资金 `$1,269.00`），**不包含任何私钥、钱包地址、实盘余额或私有账本**，专供外部策略分析使用。

---

## 1. Executive Summary / 核心结论摘要

### English
1. **Cumulative Paper Portfolio Trajectory (37 Settled Trades, `2026-11-11T15:10Z` – `2026-11-13T23:30Z`)**:
   - Starting from `\$1,269.000000`, the forward paper engine executed **37 trades**: **17 Wins / 20 Losses (`45.95%` win rate)**, ending at **`\$779.324625` (`-\$489.675375`, `-38.59%`)**.
   - All three UTC calendar days (`2026-11-11`, `2026-11-12`, `2026-11-13`) hit the exact **`15%` day-start equity stop (`DAILY_LOSS_LIMIT`)**, forming a downward ratchet:
     - `2026-11-11`: day-start `\$1,269.000000` -> `-15%` stop `-\$190.350000` -> realized `-\$190.350000` -> day-end `\$1,078.650000` (`12` trades: `6W / 6L`)
     - `2026-11-12`: day-start `\$1,078.650000` -> `-15%` stop `-\$161.797500` -> realized `-\$161.797500` -> day-end `\$916.852500` (`21` trades: `10W / 11L`)
     - `2026-11-13`: day-start `\$916.852500` -> `-15%` stop `-\$137.527875` -> realized `-\$137.527875` -> day-end `\$779.324625` (`4` trades: `1W / 3L`)
2. **Primary Drawdown Drivers Across the 37 Settled Paper Trades**:
   - **Multi-Look Retries (`book_look_attempt >= 1`) = Adverse Selection Trap**:
     - First-look fills (`att = 0`): **`28` trades, `14W / 14L` (`50.00%`), `-\$216.84`, avg cost price `0.4731`**.
     - Retry fills (`att >= 1`): **`9` trades, `3W / 6L` (`33.33%`), `-\$272.83`** — just 24.3% of trades caused **`55.72%`** of the entire cumulative drawdown! When an outcome initially trades at `0.53–0.56` at `T+6s..9s` and later drops to `0.45–0.50` at `T+12s..18s`, that drop is caused by spot moving against the predicted direction while market makers dump inventory.
   - **Severe Stream Divergence**: `ETH_5m` (`2W / 8L`, `-\$530.64`) and `BTC_5m` (`1W / 4L`, `-\$299.16`) lost a combined **`-\$829.80`**, whereas `ETH_15m` (`3W / 0L`, `+\$202.76`), `SOL_5m` (`7W / 4L`, `+\$128.00`), and `SOL_15m` (`1W / 0L`, `+\$44.79`) generated **`+\$375.55`** combined (`BTC_15m`: `3W / 4L`, `-\$35.42`).
3. **Post-Circuit-Breaker Discovery (`2026-11-13 17:00Z – 23:30Z`, 22 Blocked Signals)**:
   - Model raw directional accuracy on official Polymarket Chainlink/Gamma settlements was **`17W / 5L = 77.27%` (`17/22`)**.
   - Yet replaying all `22` signals against the **actual recorded public CLOB orderbook telemetry (`t+2s, 6s, 10s, 15s, 20s`)** under the active policy limits (`[0.415, 0.515]` for `15m`, `[0.425, 0.515]` for `5m`) yields **`16` skipped signals (`14W / 2L`)** and only **`6` executable fills (`3W / 3L = 50.00%`)**:
     - At current paper equity (`\$779.32`): **`-\$16.13 USD`** net PnL.
     - At full `1x` reference equity (`\$1,269.00`): **`-\$53.53 USD`** net PnL.
   - **Winner's Curse / Three-Layer Adverse Selection Funnel**: **`13` of `17` true winning signals** were already repriced by Polymarket market makers to `ask = 0.58–0.78` by `t+2s..10s` (above our `0.515` max buy price), whereas losing reversals consistently provided deep liquidity (`\$3,580+` depth at `<= 0.50`) and executable winners often had thin depth (`\$47.32` cap-out). Combined with `~3.5%` taker fee drag at `p ~ 0.48–0.50`, a `77.27%` signal generator filtered down to a `50.00%` negative-EV fill set.

### 中文
1. **模拟盘累计表现（`2026-11-11T15:10Z` – `2026-11-13T23:30Z`，共 `37` 笔已结算订单）**：
   - 初始资金 `\$1,269.000000`，共成交 **`37` 笔**：**`17` 胜 `20` 负（胜率 `45.95%`）**，期末净值 **`\$779.324625`（累计亏损 `-\$489.675375`，`-38.59%`）**。
   - 连续三个 UTC 自然日均准确触发 **`15%` 日初净值单日止损线（`DAILY_LOSS_LIMIT`）**，形成逐日下台阶棘轮：
     - `2026-11-11`：日初净值 `\$1,269.000000` -> `-15%` 止损线 `-\$190.350000` -> 实亏 `-\$190.350000` -> 日末 `\$1,078.650000`（`12` 笔：`6` 胜 `6` 负）
     - `2026-11-12`：日初净值 `\$1,078.650000` -> `-15%` 止损线 `-\$161.797500` -> 实亏 `-\$161.797500` -> 日末 `\$916.852500`（`21` 笔：`10` 胜 `11` 负）
     - `2026-11-13`：日初净值 `\$916.852500` -> `-15%` 止损线 `-\$137.527875` -> 实亏 `-\$137.527875` -> 日末 `\$779.324625`（`4` 笔：`1` 胜 `3` 负）
2. **全量 `37` 笔交易的核心亏损归因**：
   - **多次重看盘口（`book_look_attempt >= 1`）是典型“逆向接飞刀毒药”**：
     - 首拍即成交（`att = 0`）：`28` 笔，`14` 胜 `14` 负（**`50.00%`**），合计 `-\$216.84`，平均裸买价 `0.4731`；
     - 第 `2~4` 拍延迟成交（`att >= 1`）：`9` 笔仅 **`3` 胜 `6` 负（`33.33%`）**，合计亏损 **`-\$272.83`**——仅占总笔数 `24.3%` 的重拍单贡献了总回撤的 **`55.72%`**。原因在于：前 10 秒还在 `0.53–0.56` 的单子，只有现货朝反向急跌时，做市商才会降价砸到 `0.45–0.50` 喂给我们成交。
   - **品种表现极端分化**：`ETH_5m`（`2W / 8L`, `-\$530.64`）与 `BTC_5m`（`1W / 4L`, `-\$299.16`）合计亏损 **`-\$829.80`**；而 `ETH_15m`（`3W / 0L`, `+\$202.76`）、`SOL_5m`（`7W / 4L`, `+\$128.00`）与 `SOL_15m`（`1W / 0L`, `+\$44.79`）合计盈利 **`+\$375.55`**（`BTC_15m`：`3W / 4L`, `-\$35.42`）。
3. **熔断后 `22` 条被拦截信号的实录盘口回放结论（`2026-11-13 17:00Z – 23:30Z`）**：
   - 模型在 Polymarket 官方结算上的**原始方向胜率高达 `17W / 5L = 77.27%`（`17/22`）**。
   - 但按现行策略限价与瀑布分配规则（`15m`: `[0.415, 0.515]`，`5m`: `[0.425, 0.515]`）对 **`book_telemetry.sqlite3` 当时逐秒实录盘口**做严格回放：**`16` 笔因价格超限未能买进（其中错过 `14` 笔赢单！）**，仅 **`6` 笔实际能成交（`3W / 3L = 50.00%`）**：
     - 按 `\$779.32` 当前模拟盘净值：**净亏 `-\$16.13 USD`**；
     - 按 `\$1,269.00` `1x` 满额净值：**净亏 `-\$53.53 USD`**。
   - **三重逆向选择漏斗（Winner's Curse）**：`17` 笔真正做对方向的大趋势赢单中，有 **`13` 笔在 `t+2s..10s` 盘口已被做市商拉至 `0.58–0.78`** 被 `0.515` 上限拦在门外；反之输单往往在 `0.47–0.50` 摆出 `\$3,580+` 巨量卖单让我们满仓接盘，能买到的赢单却常只有 `\$47–\$139` 薄深度，叠加 `~3.5%` 吃单手续费后，将 `77.27%` 的高准度信号池筛成了 `50.00%` 胜率、负期望的成交池。

---

## 2. Cumulative 37-Order Attribution (`account.sqlite3`) / 全量 37 笔模拟交易归因

### 2.1 Breakdown by `book_look_attempt` (`multi_look_ioc`) / 按看盘重试次数统计

| `book_look_attempt` | Trades / 笔数 | Wins / 胜 | Losses / 负 | Win Rate / 胜率 | Avg Cost Price / 平均裸价 | Net PnL (USD) / 净盈亏 | Share of Drawdown / 亏损占比 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | 28 | 14 | 14 | **50.00%** | `0.473082` | `-\$216.843297` | 44.28% |
| `1` | 6 | 2 | 4 | **33.33%** | `0.486628` | `-\$238.350157` | 48.68% |
| `2` | 2 | 1 | 1 | **50.00%** | `0.487406` | `-\$8.045022` | 1.64% |
| `3` | 1 | 0 | 1 | **0.00%** | `0.493954` | `-\$26.436891` | 5.40% |
| **Subtotal Retries (`att >= 1`)** | **9** | **3** | **6** | **33.33%** | `0.487616` | **`-\$272.832070`** | **55.72%** |
| **Total All Trades** | **37** | **17** | **20** | **45.95%** | `0.476617` | **`-\$489.675367`** | **100.00%** |

### 2.2 Breakdown by Stream / 按交易流统计

| Stream / 品种流 | Trades / 笔数 | Wins / 胜 | Losses / 负 | Win Rate / 胜率 | Avg Cost Price / 平均裸价 | Net PnL (USD) / 净盈亏 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ETH_15m` | 3 | 3 | 0 | **100.00%** | `0.492293` | `+\$202.755890` |
| `SOL_5m` | 11 | 7 | 4 | **63.64%** | `0.477163` | `+\$127.996353` |
| `SOL_15m` | 1 | 1 | 0 | **100.00%** | `0.510000` | `+\$44.787986` |
| `BTC_15m` | 7 | 3 | 4 | 42.86% | `0.472173` | `-\$35.419565` |
| `BTC_5m` | 5 | 1 | 4 | 20.00% | `0.465871` | `-\$299.158241` |
| `ETH_5m` | 10 | 2 | 8 | 20.00% | `0.476459` | `-\$530.637790` |

### 2.3 Daily Equity Ratchet (`15%` Day-Start Equity Stop) / 按 UTC 自然日净值止损棘轮

| UTC Date | Trades | W / L | Day-Start Equity (`\$`) | `-15%` Stop (`\$`) | Realized PnL (`\$`) | Day-End Equity (`\$`) | Blocked Cohorts After Stop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2026-11-11` | 12 | 6W / 6L | `1,269.000000` | `-190.350000` | `-190.349997` | `1,078.650003` | Remainder of UTC day |
| `2026-11-12` | 21 | 10W / 11L | `1,078.650003` | `-161.797500` | `-161.797495` | `916.852508` | Remainder of UTC day |
| `2026-11-13` | 4 | 1W / 3L | `916.852508` | `-137.527876` | `-137.527875` | `779.324633` | `71` cohorts (`133` `DAILY_LOSS_LIMIT` events, `17:00Z–23:55Z`) |

---

## 3. Paper Orders on `2026-11-13` UTC (`#34` – `#37`) / `2026-11-13` UTC 模拟盘 4 笔成交微观解剖

| # | Cohort (UTC) | Stream & Side | Market ID | `z_v4 / thr` (`z_base`, `z_exp`) | Decision Time | Fill Time (`att`) | Qty (Shares) | Avg Cost Px | Total Spent (incl. Fee) | Official Winner | Net PnL (USD) | Microstructure Context |
| :-: | :--- | :--- | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **34** | `00:50Z` | `SOL_5m UP` | `6977813` | `+2.131 / 1.726` (`+1.57`, `+1.87`) | `00:50:11.356Z` | `00:50:12.412Z` (`att=0`) | `86.7500` | `0.4939` | `\$44.362596` (`fee \$1.517598`) | **UP** | **`+\$42.387400`** | Clean first-look win; equity briefly peaked at `\$959.24`. |
| **35** | `16:00Z` | `BTC_15m UP` | `6990152` | `+1.433 / 1.425` (`+1.26`, `+2.11`) | `16:00:12.509Z` | `16:00:13.541Z` (`att=0`) | `175.5837` | `0.4200` | `\$76.739192` (`fee \$2.994053`) | **DOWN** | **`-\$76.739192`** | Razor-thin `+0.008` above `thr=1.425`. Prior Binance `5m` BTCUSDT futures surged `68,169.50 -> 68,873.20` (`15:45–16:00Z`), then dumped immediately `68,873.10 -> 68,564.90` at `16:00Z`; book offered `0.42` (`810+` sh). |
| **36** | `16:10Z` | `BTC_5m UP` | `6990307` | `+1.979 / 1.717` (`+1.73`, `+1.69`) | `16:10:08.578Z` | `16:10:12.386Z` (`att=1`) | `155.7283` | `0.4753` | `\$76.739192` (`fee \$2.716621`) | **DOWN** | **`-\$76.739192`** | Overlapped live `16:00–16:15Z BTC_15m UP` exposure. `att=0` at `09.34Z` rejected (`ask=0.53`), `att=1` at `12.13Z` filled at `0.45–0.47` during dump (`68,819.20 -> 68,555.70`). |
| **37** | `16:45Z` | `SOL_5m DOWN` | `6990782` | `-1.786 / 1.726` (`-1.87`, `+0.28`) | `16:45:08.308Z` | `16:45:17.799Z` (`att=3`) | `51.6901` | `0.4940` | `\$26.436891` (`fee \$0.904359`) | **UP** | **`-\$26.436891`** | Expert contradicted base (`+0.28`). `att=0/1/2` rejected (`ask=0.56/0.53/0.53`), filled at `T+17.8s` (`att=3`) during reversal (`79.26 -> 79.38`); stake clamped to remaining daily stop room `\$26.436891`, triggering `DAILY_LOSS_LIMIT`. |

---

## 4. Complete 22 Blocked Signals Dataset & Recorded Orderbook Telemetry (`2026-11-13 17:00Z – 23:30Z`)
## `2026-11-13 17:00Z – 23:30Z` 熔断期间 22 条模型信号与逐秒实录盘口全表

> Policy parameters in force: `max_buy_price = 0.515`, `min_buy_price (15m) = 0.415`, `min_buy_price (5m) = 0.425`, `single_stream_equity_fraction = 0.08`, `tick_equity_fraction = 0.15`, `stream_max_order_usd = {BTC_15m: 240, BTC_5m: 220, ETH_15m: 180, ETH_5m: 180, SOL_15m: 50, SOL_5m: 45}`. All `usd (shares)` capacity columns below reflect **actual recorded public CLOB depth `<= 0.515` inclusive of Polymarket taker fees** (`shares * rate * price * (1 - price)`).

| # | Cohort (UTC) | Stream | Side | Market ID | `z_v4 / thr` | `Dec` | Official Winner | `t+2s` `ask / cap` | `t+6s` `ask / cap` | `t+10s` `ask / cap` | `t+15s` `ask / cap` | `t+20s` `ask / cap` | Replay Status (after `Dec`) |
| :-: | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `17:45Z` | `SOL_15m` | `DOWN` | `6991582` | `-2.085 / 1.743` | `T+5.5s` | **DOWN** (**WIN**) | `0.46 / $133.94 (273.0)` | `0.55 / $0` | `0.58 / $0` | `0.62 / $0` | `0.69 / $0` | **SKIP (`>0.515` at `>=t+6s`)** |
| 02 | `18:25Z` | `SOL_5m` | `DOWN` | `6992267` | `-1.836 / 1.726` | `T+5.8s` | **UP** (**LOSS**) | `0.56 / $0` | **`0.49 / $145.27 (279.3)`** | `0.49 / $192.48` | `0.44 / $328.23` | `0.40 / $472.98` | **FILL @ `t+6s` (`0.49`) -> LOSS** |
| 03 | `18:30Z` | `SOL_15m` | `DOWN` | `6992286` | `-2.173 / 1.743` | `T+9.9s` | **DOWN** (**WIN**) | `0.61 / $0` | `0.61 / $0` | `0.58 / $0` | `0.63 / $0` | `0.65 / $0` | **SKIP (`>0.515`)** |
| 04 | `18:35Z` | `BTC_5m` | `DOWN` | `6992378` | `-2.005 / 1.717` | `T+9.4s` | **DOWN** (**WIN**) | `0.70 / $0` | `0.72 / $0` | `0.74 / $0` | `0.66 / $0` | `0.61 / $0` | **SKIP (`>0.515`)** |
| 05 | `18:35Z` | `SOL_5m` | `DOWN` | `6992375` | `-2.030 / 1.726` | `T+9.4s` | **DOWN** (**WIN**) | `0.63 / $0` | `0.69 / $0` | `0.66 / $0` | `0.64 / $0` | `0.59 / $0` | **SKIP (`>0.515`)** |
| 06 | `18:40Z` | `BTC_5m` | `DOWN` | `6992440` | `-2.186 / 1.717` | `T+9.6s` | **DOWN** (**WIN**) | `0.67 / $0` | `0.67 / $0` | `0.70 / $0` | `0.66 / $0` | `0.68 / $0` | **SKIP (`>0.515`)** |
| 07 | `18:40Z` | `SOL_5m` | `DOWN` | `6992443` | `-2.174 / 1.726` | `T+9.6s` | **DOWN** (**WIN**) | `0.70 / $0` | `0.70 / $0` | `0.70 / $0` | `0.69 / $0` | `0.71 / $0` | **SKIP (`>0.515`)** |
| 08 | `19:50Z` | `SOL_5m` | `UP` | `6995317` | `+1.739 / 1.726` | `T+9.8s` | **UP** (**WIN**) | `0.57 / $0` | `0.53 / $0` | `0.52 / $0` | **`0.44 / $198.42 (400.0)`** | `0.40 / $545.76` | **FILL @ `t+15s` (`0.44`) -> WIN** |
| 09 | `21:25Z` | `ETH_5m` | `UP` | `6998730` | `+2.535 / 1.731` | `T+10.0s` | **UP** (**WIN**) | `0.65 / $0` | `0.65 / $0` | `0.69 / $0` | `0.73 / $0` | `0.80 / $0` | **SKIP (`>0.515`)** |
| 10 | `21:25Z` | `BTC_5m` | `UP` | `6998729` | `+2.182 / 1.717` | `T+10.1s` | **UP** (**WIN**) | `0.67 / $0` | `0.67 / $0` | `0.68 / $0` | `0.70 / $0` | `0.79 / $0` | **SKIP (`>0.515`)** |
| 11 | `21:30Z` | `SOL_15m` | `UP` | `6998766` | `+2.039 / 1.743` | `T+5.8s` | **DOWN** (**LOSS**) | `0.69 / $0` | `0.65 / $0` | `0.68 / $0` | `0.62 / $0` | `0.62 / $0` | **SKIP (`>0.515`)** |
| 12 | `21:30Z` | `SOL_5m` | `UP` | `6998776` | `+1.928 / 1.726` | `T+5.8s` | **UP** (**WIN**) | `0.78 / $0` | `0.74 / $0` | `0.75 / $0` | `0.70 / $0` | `0.67 / $0` | **SKIP (`>0.515`)** |
| 13 | `21:45Z` | `BTC_15m` | `UP` | `6999197` | `+2.542 / 1.425` | `T+10.0s` | **UP** (**WIN**) | `0.61 / $0` | `0.64 / $0` | `0.69 / $0` | `0.66 / $0` | `0.68 / $0` | **SKIP (`>0.515`)** |
| 14 | `21:45Z` | `ETH_15m` | `UP` | `6999196` | `+1.818 / 1.596` | `T+10.0s` | **UP** (**WIN**) | `0.61 / $0` | `0.65 / $0` | `0.65 / $0` | `0.62 / $0` | `0.66 / $0` | **SKIP (`>0.515`)** |
| 15 | `21:55Z` | `BTC_5m` | `UP` | `6999343` | `+2.047 / 1.717` | `T+10.7s` | **UP** (**WIN**) | `0.59 / $0` | `0.62 / $0` | `0.62 / $0` | `0.58 / $0` | `0.63 / $0` | **SKIP (`>0.515`)** |
| 16 | `22:00Z` | `BTC_15m` | `UP` | `6999373` | `+1.769 / 1.425` | `T+13.2s` | **DOWN** (**LOSS**) | `0.46 / $2926.00` | `0.48 / $2968.32` | `0.48 / $2249.55` | **`0.47 / $3580.56 (7000.3)`** | `0.53 / $0` | **FILL @ `t+15s` (`0.47`) -> LOSS** |
| 17 | `22:00Z` | `ETH_15m` | `UP` | `6999368` | `+2.369 / 1.596` | `T+13.2s` | **DOWN** (**LOSS**) | `0.53 / $0` | `0.55 / $0` | `0.55 / $0` | **`0.50 / $115.96 (220.5)`** | `0.55 / $0` | **FILL @ `t+15s` (`0.50`) -> LOSS** |
| 18 | `22:25Z` | `ETH_5m` | `UP` | `6999505` | `+1.926 / 1.731` | `T+5.7s` | **UP** (**WIN**) | `0.54 / $0` | **`0.49 / $139.09 (267.1)`** | `0.47 / $330.41` | `0.44 / $369.15` | `0.50 / $127.29` | **FILL @ `t+6s` (`0.49`) -> WIN** |
| 19 | `22:35Z` | `ETH_5m` | `UP` | `6999582` | `+2.063 / 1.731` | `T+9.8s` | **UP** (**WIN**) | `0.48 / $178.26` | `0.60 / $0` | **`0.50 / $47.32 (90.0)`** | `0.52 / $0` | `0.53 / $0` | **FILL @ `t+10s` (`0.50`, thin) -> WIN** |
| 20 | `22:40Z` | `ETH_5m` | `UP` | `6999640` | `+2.223 / 1.731` | `T+9.7s` | **UP** (**WIN**) | `0.62 / $0` | `0.50 / $75.68` | `0.41 / $624.48` | `0.38 / $572.53` | `0.33 / $858.83` | **SKIP (`0.41 < 0.425` min buy px)** |
| 21 | `23:10Z` | `SOL_5m` | `UP` | `7000339` | `+1.779 / 1.726` | `T+9.7s` | **DOWN** (**LOSS**) | `0.62 / $0` | `0.59 / $0` | `0.64 / $0` | `0.65 / $0` | `0.62 / $0` | **SKIP (`>0.515`)** |
| 22 | `23:15Z` | `SOL_15m` | `UP` | `7000417` | `+2.293 / 1.743` | `T+9.7s` | **UP** (**WIN**) | `0.57 / $0` | `0.59 / $0` | `0.61 / $0` | `0.54 / $0` | `0.57 / $0` | **SKIP (`>0.515`)** |

* **Raw Model Settlement Summary / 原始模型方向统计**: **`17W / 5L = 77.27%`** (`14` winning signals skipped due to price limits: `13` above `0.515` + `#20` below `0.425`; `2` losing signals skipped above `0.515`: `#11`, `#21`). Note that `#01` (`17:45Z SOL_15m DOWN`) was still `0.46` at `t+2s` (`$133.94` depth) before feature/inference completed at `T+5.5s` (`t+6s` was already `0.55`).

---

## 5. Counterfactual Executable PnL Simulation (6 Fills) / 反事实可执行成交逐单模拟（6 笔成交）

### English & 中文对照

| Cohort (UTC) | Stream & Side | Fill Offset & `best_ask` | Official Outcome | `@ $779.32` Paper Equity (Spent / Shares / PnL) | `@ $1,269.00` `1x` Reference (Spent / Shares / PnL) | Execution & Liquidity Note / 盘口深度说明 |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| `18:25Z` | `SOL_5m DOWN` | `t+6s` @ `0.49` | `UP` (**LOSS**) | `\$45.00 / 86.50 sh / -\$45.00` | `\$45.00 / 86.50 sh / -\$45.00` | Stream cap `\$45.00`; filled easily inside `$145.27` depth. |
| `19:50Z` | `SOL_5m UP` | `t+15s` @ `0.44` | `UP` (**WIN**) | `\$45.00 / 90.72 sh / +\$45.72` | `\$45.00 / 90.72 sh / +\$45.72` | Stream cap `\$45.00`; book pulled down to `0.44` at `t+15s`. |
| `22:00Z` | `BTC_15m UP` | `t+15s` @ `0.47` | `DOWN` (**LOSS**) | `\$62.35 / 121.89 sh / -\$62.35` | `\$101.52 / 198.48 sh / -\$101.52` | Huge `\$3,580.56` ask wall let full `8%` target fill (`LOSS`). |
| `22:00Z` | `ETH_15m UP` | `t+15s` @ `0.50` | `DOWN` (**LOSS**) | `\$54.55 / 103.74 sh / -\$54.55` | `\$88.83 / 168.92 sh / -\$88.83` | Capped by cohort remaining budget (`15% - 8% = 7%`); `LOSS`. |
| `22:25Z` | `ETH_5m UP` | `t+6s` @ `0.49` | `UP` (**WIN**) | `\$62.35 / 119.72 sh / +\$57.38` | `\$101.52 / 194.95 sh / +\$93.43` | Full `8%` target filled (`WIN`). |
| `22:35Z` | `ETH_5m UP` | `t+10s` @ `0.50` | `UP` (**WIN**) | `\$47.32 / 90.00 sh / +\$42.68` | `\$47.32 / 90.00 sh / +\$42.68` | Thin book (`90.00` shares = `\$47.32` max) clipped full win size! |
| **Total (6 Fills, 16 Skips)** | **6 Trades (`3W / 3L`)** | **50.00% Win Rate** | — | **`-\$16.13 USD`** | **`-\$53.53 USD`** | **Liquidity asymmetry + taker fees negate 77.27% model signal accuracy.** |

---

## 6. Research Questions for External Quant Analysis / 供外部分析的核心研究方向

1. **T+0..T+6s Polymarket CLOB Repricing Speed vs. `0.515` Ceiling**: Since `13/17` winning signals already trade at `0.58–0.78` by `t+2s..10s`, what EV profile emerges under dynamic probability-calibrated limit pricing (`p_limit = p_model_twap60 - edge_floor - fee`) rather than a static `0.515` ceiling?
2. **Eliminating Negative-Selection Retries (`book_look_attempt >= 1`)**: Across the 37 historical fills, banning `att >= 1` retries (`9` trades, `33.3%` WR, `-\$272.83`) removes `55.72%` of total drawdown.
3. **Deduplicating Concurrent Same-Underlying `15m + 5m` Signals**: Preventing a `5m` signal (`#36 BTC_5m`) from doubling into an already open losing `15m` exposure (`#35 BTC_15m`) or correlated pair (`#16+#17 BTC_15m+ETH_15m`) inside the same window.
4. **`5m` Stream Pruning & Edge Calibration**: Auditing why `ETH_5m` (`2W/8L`) and `BTC_5m` (`1W/4L`) underperformed while `15m` + `SOL_5m` outperformed.
