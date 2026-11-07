# 系统结构

```mermaid
flowchart TD
    B[公开标的行情与结算观测] --> O["online.py / revisions.py<br/>采集、来源记录、缺口与修订检查"]
    P[Polymarket 公开市场信息和盘口] --> T["polymarket_transport.py<br/>复用连接与分离观察路径"]
    T --> PM["polymarket.py<br/>市场映射、盘口和费用"]
    C["clock_sync.py<br/>时钟偏差与有效窗口"] --> PL["pipeline.py<br/>实时调度与输入验证"]
    O --> PL
    M[本地模型与冻结特征源码] --> MOD["model.py<br/>特征计算、模型校验、预测"]
    PL --> MOD
    MOD --> S["stage.py / execution.py<br/>ETH5 → BTC15 → BTC5 容量阶梯"]
    PM --> S
    S --> A["account.py<br/>本地模拟成交、风险上限、账本"]
    A --> MON["monitor.py / argon4_forward_server.py"]
    MON --> UI["dashboard/argon4_forward<br/>127.0.0.1 Monitor"]
    R[训练与候选研究 scripts] --> M
```

`execution.py` 只记录本地模拟。限价、风险上限、流启用状态以 `config/argon4_forward_policy.json` 为准。没有信号、尚未到期或经确认的价格不满足可以跳过；缺失、失效或不一致的证据必须停止相应分配，不能猜测容量。已批准但未完成的上层目标与有效跳过是不同状态。

实盘准备文件仅描述禁用状态，不提供钱包签名和真实下单通道。当前仓库也不配置交易所账户密钥。

运行入口：`scripts/argon4_forward_server.py`；前端通过本地只读 API 获取状态、账本和权益。训练入口：`scripts/train_argon4_forward.py`。`train/evaluate_argon4_gen4*` 和组合场景脚本为独立候选研究。
