# 双侧 APC 的原生 2P2D 基线

从 [r43 启动配置](../configs/ascend-2p2d-decode-apc.example.json) 生成站点配置，填写两台实验机的地址、模型目录和 CPU 亲和性。r43 给两个 D 实例开启本地 APC 和原生重算调度器，P 侧 APC 已开启。保留原生 MooncakeConnectorV1 的 P→D 交接，以及 AscendStoreConnector 的 D→P 回读和 D 主机 128 GiB 池。

固定 vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、vLLM-Ascend `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd` 和 `mooncake-transfer-engine-npu 0.3.11.post1`。固定版读取 `--additional-config` 顶层 `recompute_scheduler_enable`，为 D 安装原生重算调度器。重算调度器在混合 KV 的 D 接收路径按缓存组查找本地前缀。实机以有效调度器、实际本地命中和短输出校验为准。[固定版配置读取](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/ascend_config.py#L128-L131) · [固定版 D 调度](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/core/recompute_scheduler.py#L447-L476)

| 项目 | 本实验 | 官方 v0.30.0 A2 §5.6 |
|---|---|---|
| APC 和 D 调度器 | P/D 均开 APC，D 启用重算调度器，固定版顶层键 | P/D 均开 APC，D 启用重算调度器，键位于 `scheduler_config` |
| 权重和并行 | Qwen3.5-35B-A3B BF16，无 MTP，双机 910B2 2P2D，每实例 Attention TP4、Expert EP4、DP1 | Qwen3.5-397B-A17B W4A8，MTP-3，双机 A2 1P1D，每实例 TP8、EP、DP1 |
| 调度和图策略 | P 原生图策略，D `FULL_DECODE_ONLY`，两侧 4,096 分块、16 槽、0.8 显存，最大上下文 262,144 | P eager，D `FULL_DECODE_ONLY`，P/D 批 token 预算 16,384/300、槽数 64/200、显存 0.90，最大上下文 133,120 |
| KV 路径 | 原生交接加共享池，记 P→D 和 D→P 两方向 | MooncakeConnectorV1 和 Pull 代理 |

对齐项为双侧 APC 与 D 重算调度器。其他实验参数沿用 r42 部署。[官方 A2 配方](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/Qwen3.5-397B-A17B.html#56-prefill-decode-disaggregation-a2)

## 原生带宽对照

四臂共享 [r42 原始回放清单](../configs/workload-r42-natural-sustained.json)。该清单包含同一数据版本的 48 个长会话、254 个请求，根投放至 92.5 秒，保留原始提示指纹、到达、前驱和输出预算。复用哈希一致的物化输入。自然 EOS 和原生路由按当前运行产生。每臂从排空引擎、空 Store、隔离的 P/D APC 命名域及初始代理开始。

同一套引擎按 `100→200→200→100` Gbps 测 D 聚合出站限额，逐口读回 12.5 或 25 Gbps 并恢复原值。记录每臂实际输出 token、首个计划投放至最后完成的含排空时间、总吞吐及同档波动。详细 profiling 单独运行，先对齐 100 档的热点端口、D→P 同步 GET、P 批次和交付、D 本地 APC、P→D 字节、D 运行与尾部，再以 200 档复核。

优化实验以两项实机证据为前提。100 档持续 RDMA 阻塞要进入供给或完工路径，正反序下总吞吐也要明显低于 200 档。用真实 EOS 输出量、路由和压尾会话解释整轮差。压力不足时，先据实际外部恢复需求增加同源自然长会话的强度或持续时间，再用相同新计划测两个档位。
