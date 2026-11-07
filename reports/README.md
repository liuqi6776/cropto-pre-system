# 历史辅助脚本说明

这些脚本保留原路径，供查看过去测速、研究和工作区验证过程。它们不属于实时启动链。

- client_latency：当前连接复用相关测速与历史工作区检查入口。
- external_backtest：外部行情数据试下载及历史核查工具。
- gen4_candidate、waterfall、mark_revision_repair、latency：候选评估、阶梯分配、修订与窗口延迟审查辅助代码。
- clean_room/reproduction_code_snapshot：五份与对应主入口/测试逐字节相同的历史副本，保留供归档比较；不要在快照目录直接运行，根目录推导仍沿用原入口假设。

所有现成 JSON 报告、日志、模型、账户账本、抓取结果和测试临时目录均未上传。历史 runner 会引用原大型研究工作区的通用构建脚本；这些无关框架不在本仓库内。请勿把历史 runner 当成新仓库的安装/启动命令。

benchmark_connections 和 API 比较需要本地数据库中的有效市场与时钟记录；profile_readonly_features 需要模型及真实观测。缺少输入时不能用编造快照替代，更不能补做历史模拟订单。
