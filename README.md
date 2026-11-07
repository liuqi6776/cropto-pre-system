# ARGON4 — Polymarket 本地预测与模拟盘

这是当前 ARGON4 系统的源码导出：行情采集、特征计算、本地模型训练、七条预测流、分阶段容量分配、模拟成交与结算、Monitor 页面及测试。仅支持研究、公开行情读取和本地模拟；实盘配置为禁用模板，没有启用真实订单。

当前配置版本为 `argon4-local311-independent65-20261107-v12r1`，本次配置从 `2026-11-07T08:20:00Z` 的未来窗口生效。第二阶段启用 **BTC 15m → ETH 5m → SOL 15m → SOL 5m → BTC 5m**，ETH 15m 与 BNB 15m 保留模型和代码但暂停分配。阶段按已保存本地研究结果的原始胜率排序；这些历史结果不代表未来收益。模型、特征公式和阈值不变；**盘口成交价上限仍为 0.515，手续费另计并占用原资金上限**。

v12 在 v11 的有界并发采集和完整历史特征快照基础上，将最早输入检查提前至校准时间下界达到开窗后 **2.5 秒**，窗口附近每 **0.2 秒**轮询。采集仍止于 T+14 秒，模拟成交仍必须早于 T+20 秒。只有合规盘口确实高于限价时，才允许首次检查加最多三次窗口内重检；不会重算预测、补历史订单或放宽风险额度。

v12r1 修正数据修订确认后等待未来窗口时仍可能进入 10 秒慢轮询的问题：完成热身且时钟有效时，恢复等待也按普通期 1 秒、关键窗口 0.2 秒检查。该修复不消除交易所行情修订或时钟校准失败，原检查和跳过修订窗口的规则继续生效。

Monitor 将严格盘口切片与实际接收时刻的有效深度分开显示，也区分窗口进行中和已到期等待官方结算的仓位。待结算成本与费用仍占用原风险额度，不能提前当作可用资金。改动与局限见 [v12 执行和观测说明](docs/ARGON4_EXECUTION_V12.md)、[当前修正版验证记录](docs/EXPORT_VALIDATION.md)；[初版 v12 验证快照](docs/EXPORT_VALIDATION_V12_INITIAL.md)、[v11 工程说明](docs/ARGON4_LATENCY_V11.md)及[原 v11 验证快照](docs/EXPORT_VALIDATION_V11.md)保留供对照。初版观察不算作修正版部署结果，工程测试不证明成交量或收益会增加。

## 项目结构

| 目录 | 内容 |
|---|---|
| `scripts/` | 启停、服务、采集、训练、复现与候选评估入口 |
| `tools/argon4_forward/` | 采集、模型、分配、执行、账本、时钟、修订处理等核心模块 |
| `dashboard/argon4_forward/` | 本地 Monitor 的 HTML、CSS、JavaScript |
| `config/` | 当前模拟策略配置、未配置的实盘准备模板 |
| `research/argon4_clean_room_20261105/original/` | 三份冻结原始研究/特征源码与校验清单 |
| `tests/` | ARGON4 功能、时序、数据完整性与隔离测试 |
| `archive/catalog/` | 空数据注册表模板，不能视为已有数据通过验证 |
| `reports/` | 历史审查/测速脚本源码；不含原报告、日志或账户快照 |
| `docs/` | 架构、文件清单、恢复说明与本次导出验证记录 |
| `research_data/`、`paper_runtime/` | 用途说明，数据和账本不上传 |

详细文件列表见 [docs/FILE_INDEX.md](docs/FILE_INDEX.md)，运行关系见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 安装与只读观察

验证环境：Windows、Python 3.12.11。核心运行代码要求 Python 3.11 及以上；其他系统未在此次导出中验证。PowerShell 启停脚本仅用于 Windows。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m pytest tests -q
.\.venv\Scripts\python.exe -B scripts/argon4_forward_server.py --port 8898
```

浏览器打开 `http://127.0.0.1:8898/`。上述启动方式没有 `--enable-paper`，只做观察。使用 8898 是为了避免与已有 8897 服务冲突；服务只绑定本机地址。公开行情仍需要网络连通。

这份仓库**没有模型、历史行情、当前模拟账本或完整数据验证凭据**，克隆后不能直接恢复正在运行的模拟账户。模型缺失或数据检查失败时，新单应保持关闭；不要绕过校验。

恢复模拟盘前请阅读 [docs/RESTORE.md](docs/RESTORE.md)。`scripts/start_argon4_forward.ps1` 固定使用 8897，会读取策略中的 `paper_trading_active` 并决定是否启用模拟；它不是上述只读观察命令的等价替代。

## 数据和模型约束

- 只从实际观测构建输入；不补造 OHLCV、mark candles 或 funding 结算事件。
- 行情来源、时间与修订需留存证据；过期/缺失输入停止新单，不补历史订单。
- 使用本地训练模型。Gen4 尾部加权、稀疏网络属于候选研究代码，未因导出而切换到当前运行模型。
- 运行模型和候选源码存在 SHA256 绑定。`.gitattributes` 禁止换行自动转换；修改冻结文件或自动格式化会影响校验。
- `py-clob-client` 不是当前必需依赖；当前实现使用复用的只读 HTTP 连接及独立行情/执行观察路径。
- 模拟成交使用盘口、手续费和时效检查，不保证与真实成交相同，也不保证盈利。
- `scripts/argon4_live_preflight.py` 只核对禁用状态、同一策略配置与本地模型哈希；没有真实订单执行器或签名功能。

个人地区、账户、钱包与计划投入金额已从实盘模板清除。不要将密钥、助记词、私人账本或账户凭据提交到仓库。
