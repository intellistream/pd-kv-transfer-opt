# 跨领域机制与 2P2D 的判别实验

2026-11-20。并行筛选 D 准入反馈、P→D 流水、暂存和纯网络四类改动。每类先找可独立改变的依赖或资源，使用同字节、同到达和同输出的机理对照。面向原长会话的最终指标仍为含排空输出吞吐。[归因与模型](critical-path-attribution.md#r28-原生长会话的完工路径) · [离线结果指纹](../results/2026-11-20-cross-domain-screen/manifest.json)

后续 r31 以 100 Gbps 为主判定，原长会话的 P 物理块分配零失败。暂存和信用保留给真实页短缺工况。r32 的大块 HIXL READ 在 100 档已接近单口限速，小片才有明显的同字节合批机会。两项结果给下文的离线候选限定适用范围。[r31/r32 实测](results.md#r31-原长会话的物理页回压) · [流计算拓展](stream-processing-related-work.md#无-p-页阻塞后的分支)

## D 消费预测与简单信用

用 r26 和 r27c 的真实方法事件做影子决策。取 P 首个非空 chunk 派发和 D 入口两个决策时点。判断样本时，只使用在该时点已经转发给目标 D 的请求、已经开始流式生成的时间、输入配置里的固定生成预算、最近 15 秒的 D 生成计数与运行槽，以及前 16 条同 P 已完成的服务。估计器按过去的每槽 token 速率估计占用剩余时间，给待发请求排一个假设的 D 槽，并扣除尚需的 P 服务。早决策点以两机事件锚点换算，并留 150 ms 的时间保护。历史 D 交接入队作为标签，实际序列始终按原 trace 运行。已有页分配、在途负载和具体消费顺序会改变新策略的结果。

| r27c 在 P 首派发时的规则 | 实际 D 入队等待至少 5 秒，命中 | 误触发 | 漏掉 |
|---|---:|---:|---:|
| D 已运行 16 条 | 361 / 404 | 22 | 43 |
| 同一 D 已收到且未完成至少 16 条 | 378 / 404 | 22 | 26 |
| 用生成进度和 P 服务估计余量 | 363 / 404 | 21 | 41 |

保护加到 300 ms 后，简单规则仍命中 377 条，预测仍命中 363 条。等到 D 入口估计，预测命中 401 条，简单在途数命中 404 条，此时 P 已经持页。在 15 秒的标签下，P 首派发时预测少误触发 11 条，同时少命中 28 条。r26 的 88 条都低于 1 秒，5 秒标签下三种规则均未触发。r27c 使用固定输出，影子预测已能看到配置预算。保留 EOS 的自然负载还需解决真实剩余工作量的估计。当前优先以简单在途限制和交换实际 Hybrid 目的页、运行槽及在途字节的接收端信用作为对照。复杂预测暂降级，等找到能稳定区分简单规则的样本再投入。Flink 的接收缓冲 credit 和 Homa 的接收端授权给出了已知的控制原语。[影子重算](../results/2026-11-20-cross-domain-screen/downstream-credit.json) · [Flink credit](https://flink.apache.org/2019/06/05/a-deep-dive-into-flinks-network-stack/) · [Homa](https://people.csail.mit.edu/alizadeh/papers/homa-sigcomm18.pdf)

## 固定总带宽，单独改变端口分配

Varys 用同一批关联流的共同完成解释数据流工作。OpenUCX 给出跨 rail 条带化的工程先例。当前 HIXL 路径还未证实可重新安排任意源页的端口。可以先固定原生路由和实际字节，只改变八口限速分布，检验是否由最慢口限制 rank 或批的 join。[Varys](https://istc-cc.cmu.edu/publications/papers/2014/varys-sigcomm14_abs.shtml) · [OpenUCX 多 rail](https://github.com/openucx/ucx/blob/master/docs/source/faq.md#does-ucx-support-multi-rail)

同一批负载下，给端口 $p$ 字节 $W_p$、限速 $B_p$ 和总额 $B$，纯载荷时间下界为

$$
T_{\mathrm{wire}}\ge\max_p \frac{8W_p}{B_p},
\qquad \sum_p B_p=B .
$$

r15 原生六轮的最大口都为 D 口 3，发送量约为八口均值的 1.21 倍。用 100 Gbps 首轮的 MAC TX 各口比例生成固定权重。总额保留 100,000 Mbps，按口 0–7 设置 `11711 12373 12649 15130 13037 12237 11466 11397`，口 3 相比等额多 2.63 Gbps。它们都在现有驱动接口接受的配置范围内，实机仍需核对配置和当前读回。

| 物理计数对照 | 八口等额下界 | 使用 r15 首轮权重的下界 |
|---|---:|---:|
| r15 100 Gbps 首轮 | 89.79 s | 74.18 s |
| r15 100 Gbps 两轮留出 | 各 89.79 s | 各 74.80 s，约低 16.7% |
| r26 四轮，同总额 | 2.75–5.70 s | 比等额更长约 4.3%–6.7% |
| r27c 两轮，同总额 | 34.82–69.09 s | 比等额更长约 6.7%–9.0% |
| r28 两次 100 Gbps | 8.84 / 8.78 s | 8.96 / 8.98 s |

下界适用于同样的端口字节和给定限速，含端口统计中的控制流量。r15 数据显示按原热点加权有可验证空间，r26 至 r28 的热点不同。r28 的 100 Gbps 一秒最大单口速率 10.41/9.86 Gbps，较长会话的整轮基本由其他阶段决定。固定权重的结果只用于选择实机干预。自动选路或跨 rail 仍需可选物理路径、页归属和读取并发证据。[逐口重算](../results/2026-11-20-cross-domain-screen/transport.json)

**固定总额实机协议。** 用 r14/r15 的两条 223,256/238,600 token 热前缀、固定路由、每臂 96 个请求、16 请求/s 和逐条 16 个输出 token。两侧 APC 关闭，保留原生 Store GET 和 P→D 交接。先以原端口设置完成两条独立预热，接着在同一引擎做 100 Gbps 等额、100 Gbps 加权、100 Gbps 等额。逐口保存 `requested_port_Mbps`、实际读回和收发计数。核对每臂一致的 GET 和交接描述符字节、rank 阶段、完整结束数、输出和整轮吞吐。只有在高负载确实有稳定热点、加权使 GET 的最迟完成和服务完工都提前时才继续。清理时读回八口实验前的配置。[r14 部署](deployment.md#r14-硬限速诊断复现)

```bash
python -B -m pd_kv_transfer_opt.experiments.run_suite \
  --config configs/ascend-r14.local.json --run-dir runs/r14-D \
  --workload data/r14-pair.jsonl --port-rates-mbps 11711 12373 12649 15130 13037 12237 11466 11397
```

默认预览顺序。实机在确认两机空闲、自有 2P2D 服务已就绪后使用 `--apply`，现有控制器读回和收尾负责恢复每口原值。方法归因的 P/D 服务使用 `--observe`，本对照无需启动独立 CANN profiler。常规脱敏入口保留新增的端口请求和回执，`analyze_full_rdma` 识别该对照并重算三个整轮结果。

**实机复核。** r29 的旧权重将真实瓶颈从口 2 移到口 7，整轮低于等额均值 3.02%。r30 只从 r29 首个等额臂得到新权重，三轮仍按等额、加权、等额执行，所有字节和输出一致，加权相对等额均值高 1.98%。端口线速下界从约 80.15 降至 74.26 秒，实际排空只缩短约 1.72 秒。固定分配只在当前放置与稳定流量下有可验证收益。需另行测量原长会话和在线分配。[r29/r30 结果](results.md#r29r30-固定总额的八口限速实测)

## 有限暂存解除 P 页持有

Flink Hybrid Shuffle 倾向把消费者能及时取走的分片留在快速路径，剩余部分使用有界保存。SQL Server 的远端内存扩展在异步写回时可借助预注册缓冲，尽早回收原缓冲页。可对应到 P HBM 在 D 尚不能接收时的占用。[FLIP-235](https://cwiki.apache.org/confluence/spaces/FLINK/pages/211884657/FLIP-235%2BHybrid%2BShuffle%2BMode) · [SQL Server RDMA](https://www.microsoft.com/en-us/research/publication/accelerating-relational-databases-by-leveraging-remote-memory-and-rdma-2/)

r27c 的 200 档在 34–48 秒，P0/P1 分别有 31/52 个已回复且 D 还未就绪的目标，约 125/125 GB 的逻辑交接载荷。用同一组 trace，在每次 P 回复时若目标 D 的已转发未完成数至少为 16，且本机预算够用，则模拟一次 P HBM→本机 DRAM 复制。每个 P 同时只有一笔，假定 8 GiB/s，D 各请求仍在历史位置入队。每个 P 的暂存从开始复制计到原 D KV 就绪，复制早于原 D 入队结束时才记作可提前还页。

| 每个 P 的有界暂存 | r27c 100 / 200 Gbps 在 D 入队前完成的复制 | 逻辑待交接页持有积分下降 |
|---|---:|---:|
| 4 GiB | 16 / 10 | 2.77% / 2.06% |
| 8 GiB | 31 / 28 | 6.35% / 5.58% |
| 16 GiB | 50 / 45 | 13.02% / 12.60% |
| 32 GiB | 63 / 62 | 20.44% / 22.96% |

r26 的阴性对照零激活。该回放统计逻辑交接字节乘等待时间，同一物理页可能有多个引用。未测 D2H 速率、NPU 内存通路争用、真实空闲页和随策略变化的 D 供给。第一实机试验只需一条明确等待的请求，先记录实际阻塞的 allocator 分支、持页引用与另一条可推进的请求。临时方案需准备已注册的 host 内存，源和版本在所有 rank 复制完成后发布，P 旧页只有在所有引用安全关闭时才释放。与真实接收端信用比较。若 P 派发提前只改善非尾部路径，按资源效果报告。

当前 vLLM-Ascend 已有 P HBM→D DRAM→D HBM 的 `MooncakeD2RHConnectorV1`。实验固定 `5cb98caa` 的 P/D connector 目录未包含它。官方部署示例在 Atlas 850E/UBoE 验证，须单独核对同版本 910B2、Hybrid、RoCE 支持。该路径和 P 本机暂存应在同页预算下作对照。已有两跳分段方案覆盖基本机制。[D2RH 设计](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/disaggregated_prefill.html#mooncake-d2rh-connector) · [验证平台](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md) · [暂存模型结果](../results/2026-11-20-cross-domain-screen/transport.json)

## 可重叠的 P→D 交接与提交成本

**生产者流水。** MPI 4.1 的分片通信区分各片就绪和整个操作完成。固定版 Ascend 已有 `MooncakeLayerwiseConnector`，用源侧缓存写事件提交逐层传输，Mamba align 也有专门状态选择。实际 2P2D 加 Store 的组合需校验，现成路径应先作为强对照。可在 D 已有合法目标页的请求里，用同一分配和路由比较最后统一提交与早完成的 Attention 片提前提交，所有 rank 的必要 Attention 与 recurrent 状态完整后才放行 D。源不可变时点、D 最终页可写时点、额外提交和完整消费都要分别观测。[MPI 分片通信](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node86.htm) · [固定源码的发送事件](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py#L1690-L1819)

按最后完成 rank 拆原生 P→D 调用。r27c 两档在提交前 handler 排队的 p50 已有 **258/279 ms**，描述符准备 p50 **1.43/2.04 ms**，提交到结束 p50 **36.65/37.58 ms**。D 入口后 250 ms 内即可入队的分别有 35/36 条。在这些样本中固定各 rank 的提交时点，将物理服务归零，最后 rank 就绪的理想提前 p50 **31.32/34.37 ms**，p95 **106.72/93.53 ms**。决定 r27c 完工的尾请求提交耗时约 17–18 ms，源片提前与实物可释放时间仍需另采。此流水适合入队快且交接在关键路径的独立试验。

**同字节提交与网络拥塞。** 用预注册的同一对 endpoint、地址和总字节，扫每批 1/4/16/64 个描述符及少量在途。分开记录 Python/host 准备、HIXL 提交至完成、MAC TX、各 rank ready。FaSST 的批量 doorbell 和 ATC 的 RDMA 设计指南可作为对照。Mooncake 公开 v0.3.11 的异步执行器已经批量提交，`ASCEND_USE_ASYNC_TRANSFER` 按变量是否存在判定，`=0` 也会启用，且 transport buffer pool 和 async 互斥。已装 post1 需单独核对，底层异步也要验证上层同步 Store GET、worker 和发布是否提前。r15 实测 MAC TX 高于有效 GET 载荷的部分仅约 1.4%–1.51%，它包含其他控制流量。优先查提交与最慢端口两条路径，包头比例和 RoCE 拥塞需更直接的计数。只有接收暂停、ECN/CNP 或重传与关键读时段共同出现，才推进拥塞控制参数试验。[RDMA 设计指南](https://www.usenix.org/conference/atc16/technical-sessions/presentation/kalia) · [FaSST](https://www.usenix.org/system/files/conference/osdi16/osdi16-kalia.pdf) · [Mooncake 配置源码](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11/mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/transfer_executor_base.cpp#L94-L114) · [DCQCN](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/dcqcn-sigcomm15.pdf)

## 重算

```bash
python -B -m pd_kv_transfer_opt.profiling.screen_downstream_credit \
  results/2026-11-19-r27c-native-sustained/summary.json \
  results/2026-11-19-r27c-native-sustained/r27c-P.jsonl.gz \
  results/2026-11-19-r27c-native-sustained/r27c-D.jsonl.gz runs/r27c-credit.json --d-slots 16
python -B -m pd_kv_transfer_opt.profiling.screen_transport_limits results runs/transport-branches.json
```
