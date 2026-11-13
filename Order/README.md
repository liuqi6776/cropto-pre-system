# ARGON4 External Live Executor / ARGON4 外部实盘监听与执行系统

> [!IMPORTANT]
> **Physical Isolation Guarantee / 物理隔离保障**
> This repository (`c:\Users\liuqi\Order`) is completely isolated from the upstream paper simulation system (`c:\Users\liuqi\crypto-predict_system`). It holds all private keys and API credentials locally in `c:\Users\liuqi\Order\.env`, never touches or locks upstream's `account.sqlite3`, strictly executes `OrderType.FAK` (Fill-And-Kill), and only allows CTF `redeemPositions` back to Polymarket Proxy Cash (`withdrawal_permission: false`).
> 本仓库 (`c:\Users\liuqi\Order`) 与上游模拟盘仓库 (`c:\Users\liuqi\crypto-predict_system`) 完全物理隔离。所有私钥与 API 密钥仅保存在本地 `c:\Users\liuqi\Order\.env` 中，绝不读取或锁死上游 `account.sqlite3`，订单类型锁死为 `OrderType.FAK`（即时成交剩余撤销），且仅允许通过 `redeemPositions` 将中奖份额赎回至 Polymarket 可用现金余额（严禁提现）。

---

## 1. Only 2 Keys Required in `.env` / `.env` 极简 2 项密钥配置

Open [`c:\Users\liuqi\Order\.env`](file:///c:/Users/liuqi/Order/.env) and fill in **only these 2 items**:
请直接打开并编辑 [`c:\Users\liuqi\Order\.env`](file:///c:/Users/liuqi/Order/.env)，**只需填入以下 2 项**：

```dotenv
# 1. 您登录 Polymarket 使用的 MetaMask 钱包私钥（0x 开头的 64 位十六进制字符串）
POLY_PRIVATE_KEY=0x您的MetaMask钱包私钥

# 2. 您的 Polymarket Relayer API Key（对应网页 Settings -> Relayer API Keys 下的“API 密钥”）
RELAYER_API_KEY=您的Relayer_API_Key
```

### Automatic Parameter Derivation / 系统自动派生机制（无需手动配置）
Once you fill in `POLY_PRIVATE_KEY` and `RELAYER_API_KEY`, the system **automatically derives and caches** everything else at startup:
当您填入上述 2 项后，系统在启动时会自动根据您的 `POLY_PRIVATE_KEY` 推算并配置其余所有参数：
1. **Wallet Address = Signer Address = `RELAYER_API_KEY_ADDRESS`**: Automatically computed from `POLY_PRIVATE_KEY` (`EthAccount.from_key(POLY_PRIVATE_KEY).address`) and attached to every Relayer request header (`RELAYER_API_KEY_ADDRESS`).
   **`钱包地址 = 签名者地址 = RELAYER_API_KEY_ADDRESS` 自动推算**：直接从您的 `POLY_PRIVATE_KEY` 推算出 `0x...` 地址，并自动作为请求头 `RELAYER_API_KEY_ADDRESS` 发送给 Polymarket Relayer。
2. **Signature Type (`signature_type = 2`) & Polymarket Safe Wallet (`funder_address`)**: Deterministically computed from your MetaMask EOA address via Polymarket's official Gnosis Safe Factory (`0xaacFeEa03eb1561C4e67d661e40682Bd20E3541b`).
   **签名类型 (`signature_type = 2`) 与资金代理地址 (`funder_address`) 自动推算**：通过您的钱包地址自动计算出 Polymarket Gnosis Safe 代理地址。
3. **CLOB L2 API Credentials (`api_key` / `api_secret` / `api_passphrase`)**: Automatically derived via L1 EIP-712 signature (`create_or_derive_api_creds()`) on startup and cached locally in `runtime/.derived_clob_creds.json`.
   **CLOB 下单 L2 API 密钥三元组自动派生**：启动时自动通过私钥签名向 Polymarket 派生，并缓存在本地 `runtime/.derived_clob_creds.json`，无需手动填写。

---

## 2. Two-Stage Capital Scaling (`live_scale_factor`) / 两阶段资金缩放配置

Configured in [`c:\Users\liuqi\Order\config\live_config.json`](file:///c:/Users/liuqi/Order/config/live_config.json) under `capital_and_scaling.live_scale_factor`:
在 [`c:\Users\liuqi\Order\config\live_config.json`](file:///c:/Users/liuqi/Order/config/live_config.json) 的 `capital_and_scaling.live_scale_factor` 中配置：

1. **Week 1 Trial (`"live_scale_factor": 0.10`, Current Default / 当前默认)**:
   - Live capital basis / 实盘基准本金: **`$1,000 USDC`** (`10%` of `$10,000` paper account)
   - Single order cap (`max_order_usd`) / 单笔上限: **`$20`**
   - Total open risk cap (`max_open_risk_usd`) / 总敞口上限: **`$100`**
   - Single underlying cap (`max_underlying_open_risk_usd`) / 单币种上限: **`$50`**
   - 5-minute cohort budget cap (`max_tick_budget_usd`) / 单窗口总预算: **`$120`**
2. **Full Deployment After Week 1 (`"live_scale_factor": 1.00`) / 一周对账无误后切换全额规模**:
   - Edit `config/live_config.json` and change `"live_scale_factor": 0.10` to `"live_scale_factor": 1.00` (or launch with `-Scale 1.00`), no code changes required.
   - 只需将 `config/live_config.json` 中的 `"live_scale_factor"` 从 `0.10` 改为 `1.00`（或启动时传入 `-Scale 1.00`），无需修改任何代码即可切换至 `$10,000` 规模（单笔 `$200`、总敞口 `$1,000`、单币种 `$500`）。

---

## 3. Quick Start & Commands / 启动与常用命令

```powershell
# 1. Run Pre-flight Verification (checks isolation, upstream 8897, local signing speed, and 3 .env keys)
# 1. 运行实盘前自检（核验物理隔离、上游连通性、本地零延迟签名速度及 .env 3 项密钥状态）
$env:PATH = "C:\Users\liuqi\anaconda3\envs\min_env;C:\Users\liuqi\anaconda3\envs\min_env\Library\bin;$env:PATH"
c:\Users\liuqi\Order\.venv\Scripts\python.exe c:\Users\liuqi\Order\scripts\preflight_check.py

# 2. (Optional) Preview your auto-derived MetaMask Safe address & L2 API credentials
# 2. （可选）填好 .env 后，查看自动推导出的 Polymarket 代理钱包地址并预热缓存 L2 密钥
c:\Users\liuqi\Order\.venv\Scripts\python.exe c:\Users\liuqi\Order\scripts\derive_api_creds.py

# 3. Start Background Daemon in Shadow Mode or Live Mode
# 3. 后台启动实盘监听与执行进程（演练用 -Mode shadow，正式实盘用 -Mode live）
powershell -ExecutionPolicy Bypass -File c:\Users\liuqi\Order\scripts\start_live_executor.ps1 -Mode live -Scale 0.10

# 4. Stop Background Daemon
# 4. 停止后台实盘进程
powershell -ExecutionPolicy Bypass -File c:\Users\liuqi\Order\scripts\stop_live_executor.ps1
```

---

## 4. Audit Logs & Paper Reconciliation Files / 实盘对账日志文件路径

All logs are written to `c:\Users\liuqi\Order\runtime\`:
所有结构化对账日志均输出在 `c:\Users\liuqi\Order\runtime\` 目录下：
- `runtime\live_audit.sqlite3`: SQLite database (`WAL` mode) storing `signal_claims`, `live_orders`, `live_quota_snapshots`, and `live_settlements`.
- `runtime\live_orders.jsonl`: Per-order execution & on-chain fill log with timing (`detection_lag_ms`, `clob_rtt_ms`), `paper_benchmark`, `live_execution`, and `diff_vs_paper` (Spec 5.1).
- `runtime\live_quota_snapshots.jsonl`: Remaining quota & capital snapshots recorded immediately after every order and settlement/redeem event (Spec 5.2).
- `runtime\live_settlements.jsonl`: Official resolution & Relayer auto-redeem comparison log (Spec 5.3).
