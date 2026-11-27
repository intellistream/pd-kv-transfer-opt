# r69 1P1D 的 MatmulAllReduce 和 FlashComm1 对照

2026-11-26 在两台各两张 Ascend 910B2 上运行 Qwen3.5-35B-A3B BF16。P0 与 D0 各用 Attention TP2、Expert EP2、DP1、双侧 APC、D 原生重算调度和 32 GiB 双侧就近 Store。三臂分别从冷缓存启动，使用实验环境原有网络设置。按顺序各跑一次原生、仅 P 开启 MatmulAllReduce、仅 P 开启 FlashComm1。FlashComm1 启动时按 TP2 去掉图捕获列表的 1 token 尺寸。 [有效配置及指纹](metadata.json)

从 r69 原计划按首轮输入长度等距选取 16 个双轮窗口，共 32 条请求、3,390,275 个提示 token。保留每次调用完整输入、计划到达、前驱依赖、原生成预算和自然 EOS。每臂 32 条成功，全部自然停止。 [数据集与原计划](../2026-11-22-terminal-source-restore-short-guard/workload.json) · [32 请求清单](workload.json)

| P 侧方案 | 实际输出 token | 含排空时长（秒） | 含排空吞吐（token/s） | TTFT p99（秒） |
| --- | ---: | ---: | ---: | ---: |
| 原生 | 5,593 | 306.173 | 18.267 | 164.516 |
| MatmulAllReduce | 5,890 | 198.921 | 29.610 | 106.439 |
| FlashComm1 | 5,631 | 213.455 | 26.380 | 124.202 |

三个条件均由同一会话的两个后继相依调用决定完工。该会话两次 P HTTP 合计耗时分别为 293.926、185.115、199.933 秒。该计时边界覆盖从代理发起 P 请求到返回，包含排队和执行。32 条请求的 nearest-rank p99 取最大值。此组为 1P1D 小样本，2P2D 复测的吞吐和 TTFT 随配对顺序变化。 [2P2D 八轮](../2026-11-26-r69-tp2-matmul-allreduce/README.md)

[逐臂吞吐和分位数](throughput.csv) · [96 条逐请求阶段](request-phases.csv.gz) · [三臂配对](comparison.json) · [原生与融合的两轮阶段归因](phase-attribution.json) · [校验清单](manifest.json)
