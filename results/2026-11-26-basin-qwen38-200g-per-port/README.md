# Qwen3.8-27B Basin 原生 200 Gbps/口实测

2026 年 11 月 26–27 日在两台各四张 Ascend 910B2 的 Coder 容器上完成五轮正式测量。模型为 Qwen3.8-27B BF16，部署为 2P2D，每实例 TP2、DP1。双侧 APC、双侧就近 Store、D 原生重算和代理原生路由均启用。每个 rank 的 Store 为 8 GiB，两机合计 64 GiB。P→D KV 交接使用 MooncakeConnectorV1，共享池使用 AscendStoreConnector。五轮开始和排空后的八个端口均读回 **200 Gbps/口**。 [模型、软件和配置指纹](manifest.json)

负载沿用 Basin 窗口 v1。每个窗口取 Terminal-Bench 公开轨迹里连续四或六次调用，保留各次调用当时完整可见的输入历史。后继等待其前驱完成，再按冻结的工具间隔释放。引擎使用模型默认停止规则和原有输出预算。每轮从同样的初始化协议开始，先跑 32 条预热。正式吞吐用成功输出 token 除以首条计划投放至最后完成的时长，包含排空。TTFT 从客户端请求就绪计到首 token。 [原负载矩阵](../../configs/basin-windows-v1/matrix.json)

| 条件 | 正式成功数 | 输出 token | 含排空时长（秒） | 输出 token/s | TTFT p99（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 稳态 | 304 | 915,983 | 2,576.99 | 355.45 | 217.66 |
| +33% 突发 | 316 | 912,525 | 2,711.17 | 336.58 | 235.85 |
| +100% 突发 | 340 | 937,361 | 2,856.63 | 328.13 | 233.48 |
| 五倍突发，自然排空 | 448 | 1,219,302 | 3,217.90 | 378.91 | 322.06 |
| 短会话突发，自然排空 | 400 | 1,118,685 | 3,234.60 | 345.85 | 391.10 |

共完成 **1,808 条正式请求**和 **160 条预热**，正式阶段输出 **5,103,856 token**。1,512 条自然停止，296 条达到原输出预算。61 次原生重算接续均归回对应正式请求，吞吐分子采用客户端最终报告的输出量。 [逐轮吞吐和分位数](throughput.csv)

| 文件 | 内容 |
| --- | --- |
| [throughput.csv](throughput.csv) | 五轮整轮吞吐、输出量、排空时长、TTFT 分位数和同一基础流的吞吐。 |
| [request-phases.csv.gz](request-phases.csv.gz) | 1,808 条正式请求的长度、完成原因、到达和会话依赖、首次实际路由与各阶段耗时。 |
| [attempt-phases.csv.gz](attempt-phases.csv.gz) | 1,869 次原生派发及重算接续的实际去向、P 首块前等待、D 分配和 KV 就绪时钟。 |
| [handoff-ranks.csv.gz](handoff-ranks.csv.gz) | 3,738 条 TP rank 交接路径，保留入队、方法执行、提交和返回的时间，以及载荷和描述符数。 |
| [port-summary.csv](port-summary.csv) · [port-counters.csv.gz](port-counters.csv.gz) | 40 条逐口汇总及 62,730 条原始计数采样，覆盖双向 MAC 字节与流控相关计数。 |
| [engine-samples.csv.gz](engine-samples.csv.gz) · [store-calls.csv.gz](store-calls.csv.gz) | 引擎生成和缓存遥测，以及正式阶段的 Store 调用名、完成时刻、耗时和逻辑载荷。 |
| [profile-summary.json](profile-summary.json) | 各阶段分布、P 排队与其他工作的重叠、D 可运行工作、突发及排空窗口的端口统计。 |

各 `_s` 时间戳以当轮正式客户端单调时钟起点为零，持续时间单位为秒。P 跨机时间轴按 HTTP 边界对齐，误差区间宽度见 `P_clock_bound_width_ms`。`attempt` 从 1 起算，`source_lane` 是同一 P 副本下的握手端点编号。遥测保留边界前后的采样，重算正式流量时按 `first_planned_offer_s` 至 `last_completed_s` 裁切相邻样本区间。P/D 两侧峰值来自约 2–3 秒间隔的 MAC 计数差分。Store 逻辑字节、P→D 描述符载荷和实际端口流量分别计量。[阶段与采样口径](../../docs/data.md)

五轮均为原生链路速率实测。P 侧最热单口发送峰值为 **19.92–30.03 Gbps**，D 侧为 **4.52–6.09 Gbps**。这些结果提供 200 Gbps/口的基线，较低带宽的吞吐变化仍需同配置实测。[逐口汇总](port-summary.csv)
