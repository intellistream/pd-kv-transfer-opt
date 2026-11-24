# 双侧 APC 的原生 2P2D 基线

当前基线沿用 r43 的双侧 APC、D 原生重算调度器和原生 P→D Pull，恢复 P/D 双侧共享 Store。从模板生成站点配置，填写两台实验机的地址、模型目录和 CPU 亲和性。ascend-112 的 P0/P1 和 ascend-180 的 D0/D1 各用 0–3、4–7 两组卡，Master 保持在 D 主机。[当前配置模板](../configs/ascend-2p2d-decode-apc.example.json) · [历史 r43 配置及结果](r43-native-decode-apc.md)

固定 vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、vLLM-Ascend `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd` 和 `mooncake-transfer-engine-npu 0.3.11.post1`。固定版读取 `--additional-config` 顶层 `recompute_scheduler_enable`，为 D 安装原生重算调度器。重算调度器在混合 KV 的 D 接收路径按缓存组查找本地前缀。实机以有效调度器、实际本地命中和短输出校验为准。[固定版配置读取](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/ascend_config.py#L128-L131) · [固定版 D 调度](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/core/recompute_scheduler.py#L447-L476)

| 项目 | 本实验 | 官方 v0.30.0 A2 §5.6 |
|---|---|---|
| APC 和 D 调度器 | P/D 均开 APC，D 启用重算调度器，固定版顶层键 | P/D 均开 APC，D 启用重算调度器，键位于 `scheduler_config` |
| 权重和并行 | Qwen3.5-35B-A3B BF16，无 MTP，双机 910B2 2P2D，每实例 Attention TP4、Expert EP4、DP1 | Qwen3.5-397B-A17B W4A8，MTP-3，双机 A2 1P1D，每实例 TP8、EP、DP1 |
| 调度和图策略 | P 原生图策略，D `FULL_DECODE_ONLY`，两侧 4,096 分块、16 槽、0.8 显存，最大上下文 262,144 | P eager，D `FULL_DECODE_ONLY`，P/D 批 token 预算 16,384/300、槽数 64/200、显存 0.90，最大上下文 133,120 |
| KV 路径 | MooncakeConnectorV1 负责 P→D 交接，AscendStoreConnector 使用双侧共享池 | MooncakeConnectorV1 和 Pull 代理 |

对齐项为双侧 APC 与 D 重算调度器。保留同步 GET、GET 并发度 1、D 的 `consumer_is_to_put=true`、`kv_load_failure_policy=fail` 和每实例独立 lookup IPC。P 原生图捕获尺寸最大为 4096。每实例 4 个 API 进程，代理用 8 个 worker、原生路由、上游连接复用和 `max_retries=1`。自然 EOS、现有 CPU 亲和性及 OMP 限制沿用已核对配置。[官方 A2 配方](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/Qwen3.5-397B-A17B.html#56-prefill-decode-disaggregation-a2)

## 共享池布局与验证

布局标识为 `two-sided-8gib-local-first/v1`。P/D 的每个模型 worker 贡献 **8 GiB**，每实例四个模型 worker，每主机两个实例，共 **P 64 GiB + D 64 GiB = 128 GiB**。API 进程和每个 worker 的 32 MiB 本地 staging 单列。两侧 Store 均配置 `preferred_segment=true` 与 `prefer_alloc_in_same_node=true`，启用就近分配。历史 r43 的 128 GiB 全在 D 主机，历史结果按原布局解释。

两侧用 `serve --dry-run` 核对生成的 `store_config` 和 `store_pool_layout`，实机以对应的 `state.json` 和 Master 的 `segment_total_capacity_bytes` 为证据。先确认 P/D 各八个 8 GiB 的已注册 segment，再用少量自然请求核对实际缓存复用和来源。遇到跨实例复用时一并记录 APC 命中、对象所在 segment 及实际选中的读源。启动状态记录传入 worker 的 `HCCL_RDMA_TIMEOUT` 环境值，缺省时有效值待实机确认。Master 驱逐水位和驱逐比例沿用固定版行为，当前未采到实际值。

## 带宽对照

Basin、r69 和 r43 的新对照使用双向限速。P、D 各八口出站同时设为每端聚合 100 或 200 Gbps。两端分别核对并保存每口实际值，跑完恢复原值。 [三个实验计划](terminal-throughput-plan.md#双向限速)

### r43 历史口径

历史 r43 对照只设置了 D 出站限额。 [r43 历史结果](r43-native-decode-apc.md)

四臂共享 [r42 原始回放清单](../configs/workload-r42-natural-sustained.json)。该清单包含同一数据版本的 48 个长会话、254 个请求，根投放至 92.5 秒，保留原始提示指纹、到达、前驱和输出预算。复用哈希一致的物化输入。自然 EOS 和原生路由按当前运行产生。每臂从排空引擎、空 Store、隔离的 P/D APC 命名域及初始代理开始。

同一套引擎按 `100→200→200→100` Gbps 测 D 聚合出站限额，逐口读回 12.5 或 25 Gbps 并恢复原值。记录每臂实际输出 token、首个计划投放至最后完成的含排空时间、总吞吐及同档波动。详细 profiling 单独运行，先对齐 100 档的热点端口、按来源分类的同步 Store GET、P 批次和交付、D 本地 APC、P→D 交接字节、D 运行与尾部，再以 200 档复核。[双侧池字节口径](data.md#事件口径)

优化实验以两项实机证据为前提。100 档持续 RDMA 阻塞要进入供给或完工路径，正反序下总吞吐也要明显低于 200 档。用真实 EOS 输出量、路由和压尾会话解释整轮差。压力不足时，先据实际外部恢复需求增加同源自然长会话的强度或持续时间，再用相同新计划测两个档位。
