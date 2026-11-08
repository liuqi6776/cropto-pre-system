# ARGON-4 v14r2/v14r3 Empirical Win-Rate Diagnosis & Pre-Live Readiness Audit
# ARGON-4 v14r2/v14r3 模拟盘胜率归因实证诊断与实盘前链路核查报告

**Date / 日期**: 2026-11-08  
**Baseline / 基线**: `cropto-pre-system` (`v14r2` / `v14r3`)  
**Sample / 实证样本**: 45 forward paper orders (44 settled, 1 open; `2026-11-06 06:20 UTC` to `2026-11-08 08:50 UTC`)

---

## 1. Pre-Live Path & Safety Architecture Audit / 实盘链路与安全架构核查

### English
1. **Fail-Closed Paper-Only Architecture (`live_preflight.py` & `stage.py`)**:
   - The current repository is a **fail-closed public-market-data paper engine** (`"order_adapter_implemented": false`, `"blockers": ["WORKSPACE_LIVE_TRADING_PROHIBITED", "NO_LIVE_ORDER_ADAPTER", "NO_SIGNER", "LIVE_ELIGIBILITY_NOT_CERTIFIED"]`).
   - Setting `"live_enabled": true` in `config/argon4_forward_policy.json` does **not** transmit real orders; instead, `tools/argon4_forward/stage.py`, `tools/argon4_forward/live_preflight.py`, and `scripts/start_argon4_forward.ps1` immediately halt and refuse to start (`LIVE_MODE_NOT_ENABLED`).
   - All 5 active stream models (`BTC_15m`, `ETH_5m`, `SOL_15m`, `SOL_5m`, `BTC_5m`) pass `scripts/argon4_live_preflight.py` with `"preparation_checks_passed": true` and `HASH_VERIFIED`.
2. **Pre-Trial `$1,000` Risk Cap Binding (`_policy_risk_limits`)**:
   - Dynamic risk limits in `tools/argon4_forward/account.py` (`_policy_risk_limits`) bind `max_order_usd`, `max_open_risk_usd`, `max_active_window_risk_usd`, `max_underlying_open_risk_usd`, `max_underlying_active_risk_usd`, and `min_buy_price` across `Account`, `PaperExecutor`, `plan_waterfall`, and `live_preflight`.

### 中文
1. **Fail-Closed 纯模拟盘安全熔断架构 (`live_preflight.py` & `stage.py`)**：
   - 当前工程为 **純公开行情模拟盘与信号验证引擎**（`live_preflight.py` 中 `"order_adapter_implemented": false`，内置 4 道硬锁：`WORKSPACE_LIVE_TRADING_PROHIBITED`、`NO_LIVE_ORDER_ADAPTER`、`NO_SIGNER`、`LIVE_ELIGIBILITY_NOT_CERTIFIED`）。
   - 若直接将 `config/argon4_forward_policy.json` 中的 `"live_enabled"` 设为 `true`，`stage.py`、`live_preflight.py` 与 `start_argon4_forward.ps1` 会立即触发安全熔断停止运行，绝不会外发任何真实订单。
   - 执行 `scripts/argon4_live_preflight.py` 核查显示：`"preparation_checks_passed": true`，5 条启用流的 `.joblib` 与 `.manifest.json` 哈希校验 `100%` 通过 (`HASH_VERIFIED`)。
2. **`$1,000` 试盘动态风控限额绑定 (`_policy_risk_limits`)**：
   - `tools/argon4_forward/account.py` 中的 `_policy_risk_limits` 已在 `Account`、`PaperExecutor`、`allocation.py` 与 `live_preflight.py` 全链路打通，支持一键切换 `$1,000` 试盘风控参数（含未到期活跃风险 `max_active_window_risk_usd` 与最低买价保护 `min_buy_price`）。

---

## 2. Empirical Diagnosis: Why Win Rate Dropped on 2026-11-08 / 为什么今天（11-08）胜率下滑而此前胜率高？

### English
Using `scripts/analyze_argon4_settlement_alignment.py` across all **44 settled paper orders** (`2026-11-06` to `2026-11-08`), we compared each trade's **Polymarket Official Resolution (Chainlink 60s TWAP)** against the **true Binance 5m Futures K-line outcome (`close >= open`)**:

| Period / 统计区间 | Trades / 笔数 | Binance 5m K-Line Win Rate / 币安K线真实胜率 | Polymarket Official Win Rate / 官方结算胜率 | Net PnL (USD) / 净盈亏 |
| :--- | :---: | :---: | :---: | :---: |
| **First 24 Trades (`11-06 14:20 ~ 11-07 19:25 BJT`)** | 24 | **58.33%** (`14/24`) | **58.33%** (`14/24`) | **+$762.21** |
| **Prior Days Total (`11-06 & 11-07`)** | 28 | **53.57%** (`15/28`) | **50.00%** (`14/28`) | **+$501.64** |
| **Today (`2026-11-08`)** | 16 | **56.25%** (`9/16`) | **37.50%** (`6/16`) | **-$531.96** |
| **All Settled Trades (`Total`)** | 44 | **54.55%** (`24/44`) | **45.45%** (`20/44`) | **-$30.32** |

**Core Discovery**: The model's true directional accuracy on Binance 5m futures K-lines **did not decay on 2026-11-08 (`56.25%`, `9/16`)**—in fact, it was slightly higher than the prior two days (`53.57%`)!  
The entire drop from `56.25%` (Binance K-line win rate) to `37.50%` (Polymarket official win rate) came from **two microstructure killers**:

### 中文
通过 `scripts/analyze_argon4_settlement_alignment.py` 对账本中全部 **44 笔已结算订单**逐笔回溯“**币安 5m 合约 K 线真实涨跌 (`close >= open`)**”与“**Polymarket 官方结算结果 (Chainlink 60s TWAP)**”，核心发现如下：
- **模型对币安原生 5m K 线的真实预测准确率今天不仅没有衰退，反而高达 `56.25%` (`9/16`)**（高于前两天的 `53.57%`；其中前 24 笔至 11-07 晚间胜率高达 `58.33%`，净赚 `+$762.21`）。
- 今天 Polymarket 官方结算胜率跌至 `37.50%` (`6/16`)，完全由以下**两大微观结构杀手**导致：

---

## 3. Two Microstructure Killers & Quantitative Evidence / 两大微观结构杀手与量化实证

### Killer #1: Ultra-Low Volatility (`rv12_bps < 3.0 bps`) + Chainlink 60s TWAP vs. Binance 5m Point-Close Flip
### 杀手一：周末极低波动率死水期 (`rv12_bps < 3.0 bps`) 下的 Chainlink 60s TWAP 结算错位

- **Mechanism / 机理**: The LightGBM models are trained on Binance 5m futures point close (`close >= open`), whereas Polymarket settles on Chainlink 60s TWAP. During weekend ultra-low volatility (`ex_ante_rv12_bps < 3.0 bps`), 5m candle bodies shrink to `< 1.5 bps` (`$0.15` on ETH, `$9` on BTC). At that scale, futures-spot basis noise and 60s TWAP averaging flip correct Binance predictions into official losses.
- **Today's 3 Flipped Trades (`2026-11-08 14:05 ~ 16:05 BJT`) / 今天下午被错杀的 3 笔盈利单**:
  1. **Trade #36 (`14:05 BJT`, `BTC_5m`, `rv12 = 3.13 bps`)**: Predicted `UP`, Binance 5m closed **`+1.36 bps` (`UP` ✅)**, Chainlink TWAP settled **`DOWN` ❌** (`-$149.70`).
  2. **Trade #39 (`14:40 BJT`, `ETH_5m`, `rv12 = 1.91 bps`)**: Predicted `UP`, Binance 5m closed **`+0.63 bps` (`UP` ✅)**, Chainlink TWAP settled **`DOWN` ❌** (`-$148.92`).
  3. **Trade #43 (`16:05 BJT`, `BTC_5m`, `rv12 = 2.57 bps`)**: Predicted `UP`, Binance 5m closed **`+0.98 bps` (`UP` ✅)**, Chainlink TWAP settled **`DOWN` ❌** (`-$179.94`).
- **Breakdown by Return Magnitude (`abs_ret_bps`) & Causal `ex_ante_rv12_bps` / 按振幅与事前波动率分层统计**:

| Bucket / 分层维度 | Count / 笔数 | TWAP-Binance Agreement / 标签一致率 | Binance K-Line Win Rate / 币安K线胜率 | Official TWAP Win Rate / 官方胜率 | Net PnL (USD) / 净盈亏 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`abs_ret_bps < 1.0 bps`** | 5 | `40.0%` (`2/5`) | **`80.0%`** (`4/5`) | **`20.0%`** (`1/5`) | `-$346.63` |
| **`abs_ret_bps 1.0 ~ 3.0 bps`** | 9 | `88.9%` (`8/9`) | `44.4%` (`4/9`) | `33.3%` (`3/9`) | `-$172.09` |
| **`abs_ret_bps >= 3.0 bps`** | 30 | **`96.7%`** (`29/30`) | `53.3%` (`16/30`) | `53.3%` (`16/30`) | `+$488.40` |
| **`ex_ante_rv12_bps < 3.0 bps` (`UltraLow`)** | 16 | `87.5%` (`14/16`) | `50.0%` (`8/16`) | **`37.5%`** (`6/16`) | `-$140.36` |
| **`ex_ante_rv12_bps >= 3.0 bps` (`Normal/Low`)** | 28 | `89.3%` (`25/28`) | **`57.1%`** (`16/28`) | **`50.0%`** (`14/28`) | `+$110.04` |

---

### Killer #2: Deep-Discount Adverse Selection (`ask_price <= 0.36`) During `T+0s -> T+15s`
### 杀手二：`T+0s -> T+15s` 延迟窗口内的“深折价逆向选择飞刀盘” (`ask_price <= 0.36`)

- **Mechanism / 机理**: Model features only observe closed candles up to `T+0s`, while order execution happens at `T+6s .. T+18s`. Previously, only an upper price cap (`max_buy_price = 0.515`) was enforced, with no lower floor (`min_buy_price`). When the underlying price moves violently **against** the `T+0s` prediction during the first `5~12 seconds` of the window, Polymarket HFT market makers slash the predicted outcome's ask price down to `0.17 ~ 0.36` (implying `<35%` win probability). Buying these deep-discount asks is pure adverse selection (catching a falling knife).
- **Empirical Proof Across All 44 Settled Trades / 44 笔实盘模拟单按买入 Ask 价格分层铁证**:

| Ask Price Zone / 买入价格区间 | Count / 笔数 | Avg Ask / 均价 | Binance K-Line Win Rate / 币安K线胜率 | Official TWAP Win Rate / 官方胜率 | Net PnL (USD) / 净盈亏 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`NORMAL_ASK (0.37 <= ask <= 0.515)`** | **30** | `0.435` | **`63.33%`** (`19/30`) | **`56.67%`** (`17/30`) | **+$898.84** |
| └─ *NearPar (`0.44 <= ask <= 0.515`)* | *16* | *0.471* | *`62.50%` (`10/16`)* | *`62.50%` (`10/16`)* | *+$464.31* |
| └─ *MidDiscount (`0.37 <= ask < 0.44`)* | *14* | *0.394* | *`64.29%` (`9/14`)* | *`50.00%` (`7/14`)* | *+$434.53* |
| **`TOXIC_DISCOUNT (ask <= 0.36)`** | **14** | `0.268` | **`35.71%`** (`5/14`) | **`21.43%`** (`3/14`) | **-$929.16** |
| **Combined Guards (`ask >= 0.37` & `rv12 >= 3.0 bps`)** | **18** | `0.435` | **`66.67%`** (`12/18`) | **`61.11%`** (`11/18`) | **+$781.91** |

- On `2026-11-06 & 11-07`, only `6/28` (`21.4%`) trades fell into `ask <= 0.36`; on `2026-11-08`, **`8/16` (`50.0%`)** of trades fell into `ask <= 0.36` (winning only `1/8 = 12.5%`).

---

## 4. Implemented Engineering Guards (`v14r3`) / 已落地的双保险工程防护 (`v14r3`)

1. **Guard A — Configurable `min_buy_price` (`tools/argon4_forward/account.py` & `execution.py`)**:
   - `_policy_risk_limits` now reads `min_buy_price` (default `0`, valid range `[0, 0.500]`, recommended `0.37` for trial/live).
   - `PaperExecutor._adapt`, `Account._validate_price_skip_receipt`, and `Account.place_paper_buy` enforce `min_buy_price <= price <= max_buy_price`. When all positive asks on the orderbook are outside `[min_buy_price, 0.515]`, the cohort stream is cleanly skipped as `PRICE_LIMIT_NOT_MET` (`min_buy_price` recorded in the skip receipt) and unlocks lower-ranked streams in the waterfall.
2. **Guard B — `min_rv12_bps` Ultra-Low Volatility Filter (`tools/argon4_forward/model_v14.py` & `config/argon4_forward_policy.json`)**:
   - `apply_volatility_regime_guard` in `tools/argon4_forward/model_v14.py` now supports both `max_rv12_bps` (`ABOVE_MAX_RV12_BPS`) and `min_rv12_bps` (`BELOW_MIN_RV12_BPS`).
   - `config/argon4_forward_policy.json` pre-populates `"min_rv12_bps": {"BTC_5m": 3.0, "ETH_5m": 3.0, "SOL_5m": 3.0}` under `"volatility_regime_guard"`.
3. **Enhanced Settlement & Adverse-Selection Analyzer (`scripts/analyze_argon4_settlement_alignment.py`)**:
   - Automatically computes and reports `by_ask_bucket`, `by_adverse_selection_zone`, `by_rv12_bucket`, and `combined_guards_simulation` alongside Chainlink TWAP vs. Binance 5m point/VWAP alignment.
4. **Live Runtime Activation & `revisions.py` Fastpath Fix (`v14r4`)**:
   - **Activated Both Guards in `config/argon4_forward_policy.json`**: Set `"min_buy_price": 0.37` and `"volatility_regime_guard": {"enabled": true, ...}` in the active policy, and updated `tools/argon4_forward/monitor_logger.py` and `execution.py` to log explicit `VOLATILITY_REGIME_GUARD` and `ADVERSE_SELECTION_GUARD` audit entries.
   - **Fixed `_is_valid_observations_prefix` & Chained Delta in `tools/argon4_forward/revisions.py`**: Live `observations.sqlite3` (`982 MB`) contains only the `snapshots` table (whereas test fixtures used `decisions`). Querying actual tables from `sqlite_master` and tracking `delta_max_id` across chained hardlinked revisions reduces `_backup_sqlite` from `5.24s` (which previously caused `FAST_REVISION_COHORT_GUARD_FAILED` at `BEFORE_CUTOVER`) down to `< 2ms` (`0 bytes` extra disk usage per revision).
5. **Multi-Level CLOB Orderbook `best_ask < min_buy_price` Fix & Immediate Waterfall Unlock (`v14r5`)**:
   - **Multi-Level Orderbook Zero-Capacity Enforcement (`execution.py` & `account.py`)**: Real Polymarket CLOB `/book` responses contain resting ask levels from `best_ask` up to `0.99` (e.g., `asks = [0.25, 0.32, 0.42, 0.50]`). Whenever `min(positive_prices) < min_limit` (`best_ask < 0.37`), `_adapt` immediately zeroes out the entire orderbook capacity (`capacity = Decimal(0)`), `_validate_price_skip_receipt` validates `not (min_limit <= min(positive) <= skip_limit)`, and `place_paper_buy` refuses to sweep resting asks above `0.37` when `best_ask < min_buy`.
   - **Immediate Waterfall Skip Without `WAITING_PRICE_RECHECK` (`account.py` & `execution.py`)**: When `best_ask < min_buy_price` (`< 0.37`), `finish_book_look` sets `next_retry_at = None` and `process_cohort` immediately records `PRICE_LIMIT_NOT_MET` on the first look (`attempt = 0`) without waiting for 3 price rechecks (`7.5s`), immediately unlocking lower-ranked healthy streams in the same window.
6. **Persistent 5-Losing-Day Cooldown Latch Across Offline Days & SQLite C-Engine `ATTACH + EXCEPT` 100% Row Audit (`v14r6`)**:
   - **Persistent Cooldown Latch Across Offline Days (`account.py`)**: `Account.ensure_cohort` checks all historical settlement dates (`daily`) for any 5-consecutive-losing-day streak (`unsettled_five_loss_end`) and any prior `COOLDOWN_STATE_REQUIRED` cohort; cooldown remains latched (`COOLDOWN_STATE_REQUIRED` or `COOLDOWN_ACTIVE`) even if the machine is powered off for arbitrary days with zero database entries, until an explicit `cooldown_until` after the 5th losing day has been configured and completed.
   - **SQLite C-Engine `ATTACH DATABASE ... EXCEPT` Full-Row Prefix Audit (`revisions.py`)**: `_is_valid_observations_prefix` uses SQLite's native C-engine `ATTACH DATABASE ? AS arch_db` + `SELECT 1 FROM (SELECT * FROM arch_db."table" EXCEPT SELECT * FROM main."table" WHERE id<=?) LIMIT 1` to verify 100% of all historical rows and columns (including middle rows such as `id=25000` in `>50,000`-row databases) in ~120ms, paired with a 100-byte SQLite header change-counter cache key.



