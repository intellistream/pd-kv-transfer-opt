# r69 双机各四卡的 MatmulAllReduce 对照

2026-11-26 在两台各四张 Ascend 910B2 上运行 Qwen3.5-35B-A3B BF16。P0、P1、D0、D1 各使用 Attention TP2、Expert EP2、DP1，保留双侧 APC、D 原生重算调度、原生选路和自然 EOS。每 rank 贡献 8 GiB 就近分配 Store，两侧合计 64 GiB。两个带宽档均限制 P/D 两端出站，每端聚合 100/200 Gbps 分别对应 **25/50 Gbps/口**。融合臂只在 P0/P1 启用 MatmulAllReduce。 [配置和对时界](metadata.json) · [逐口读回、恢复与源归档指纹](bandwidth-readback.json)

负载来自固定版 Terminal-Bench 2.1 的十个任务，提取 53 个相邻双轮窗口，每轮 106 条请求。每个请求保留源调用的完整输入历史，窗口从原轨迹中段选取。根计划集中在 0–4.32 秒，后继计划集中在 56–58.65 秒并等待其前驱完成，每条生成预算 8,192 token。正式轮从冷缓存协议启动。 [历史选择](../2026-11-22-terminal-source-restore-short-guard/workload.json) · [本组工作量指纹与请求元数据](workload.json)

| 同档配对 | 含排空吞吐，原生 → 融合（token/s） | 吞吐变化 | TTFT p99，原生 → 融合（秒） |
| --- | ---: | ---: | ---: |
| 100 Gbps 正序 | 143.92 → 109.48 | −23.9% | 134.26 → 103.27 |
| 200 Gbps 正序 | 131.21 → 138.87 | +5.8% | 122.59 → 109.64 |
| 200 Gbps 反序 | 138.87 → 125.89 | −9.3% | 117.26 → 126.07 |
| 100 Gbps 反序 | 139.39 → 125.02 | −10.3% | 112.69 → 121.45 |

八轮全部完成，共 848 条请求，846 条自然停止，两条触及生成预算。吞吐以正式输出量除以首条计划投放至最后完成的时长。TTFT 从实际就绪起计，分位数用 nearest-rank。四对均未达到 20% 吞吐提升。100 Gbps 正序的两条触顶请求与不同的完工链重合，仅替换这两条，条件回放中的吞吐变化从 −23.9% 变为 +29.6%。另外三个配对在固定逐请求输出量后仍为一正两负。 [逐轮工作量和 TTFT](throughput.csv) · [分阶段与固定输出量归因](output-attribution.md)

| 文件 | 内容 |
| --- | --- |
| [request-phases.csv.gz](request-phases.csv.gz) | 848 条正式请求的计划、实际去向、输出和阶段时间。P 派发与代理边界使用因果约束给出跨机对时区间，区间中点用于估计派发前等待。 |
| [sse-delivery.csv.gz](sse-delivery.csv.gz) | 161,127 个 SSE 块的累计输出量与相对首 token 的时间，可复核短输出回放取点。 |
| [port-counters.csv.gz](port-counters.csv.gz) | 5,206 个逐口硬件计数读数。P 侧时间使用 Unix 锚对齐，D 侧使用与客户端同机的 monotonic 时钟。端口用角色加序号标识。 |
| [comparison.json](comparison.json) · [phase-summary.json](phase-summary.json) · [output-replay.json](output-replay.json) | 四组配对、实测压尾和回放结果。回放沿用观测组批、选路和生成驻留模型。 |
| [manifest.json](manifest.json) | 导出文件的 SHA-256。 |
