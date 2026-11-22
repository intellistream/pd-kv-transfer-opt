# 有状态流处理与推理 KV 交接

2026-11-20。作者线索核实为华中科技大学计算机科学与技术学院的张书豪（Shuhao Zhang）。本页按原论文、公开实现和证据边界记录可借鉴的机制。工程对象仍是两台各 8 张 910B2 的 2P2D，Qwen3.5-35B-A3B BF16，官方 vLLM、vLLM-Ascend 和 Mooncake/HIXL。早期的流计算系统用来形成对照。正式效果用原长会话、自然 APC 与 EOS、原生路由、高压到达，按首条计划投放到最后一条完成的总输出 token/s 评判。100 Gbps 是优化效果的主判定工况，200 Gbps 用于检查放宽限速后的机制响应。[作者主页](https://me.sage.org.ai/) · [作者论文档案](https://me.sage.org.ai/publications.html) · [归因](critical-path-attribution.md) · [工作负载](workloads.md)

## 从已观测的回压出发

- **r27c 给出需要验证的因果链。** 200 Gbps 档 34–48 秒有 101 条 P 在途，P0/P1 各有 31/52 条已完工且等待 D 末 rank 收妥，D0/D1 均为 16/16 运行槽，期间缺少新 P 返回和 D 就绪。约 67.30 秒，两条来自不同 P 的此类请求在 D1 就绪，约 67.38 秒 P 恢复非空派发。时序与物理页回压一致，日志与交接块量用于定位待查窗口。需要在 allocator 中直接确认触发的是哪项容量检查、真实可分配页有多少，以及哪次释放让哪条请求获准派发。r27c 关闭本地 APC 并预置 Store 命中，是高恢复诊断。[r27c 的 P 页持有与 D 供给](critical-path-attribution.md#p-页持有与-d-供给)
- **r28 给出约束。** 原生 30 条闭环长会话四轮的 P KV 用量峰值不超过 17.7%。两轮 100 Gbps 的八口聚合接近上限的区间很短，且反序尾会话的输出长度有明显分叉。r27c 的物理页回压候选机制仍需在自然高压会话里出现，并与最终排空的 D 路由相接。r14 的短输出强带宽响应可复核 GET 机制，无法代替这一验证。[r28 的同会话完工分析](critical-path-attribution.md#r28-原生长会话的完工路径) · [r14 至 r28 模型](problem-model.md)
- **在固定版本对齐物理页语义。** 各 Hybrid cache group 的 manager 使用同一个实例级 `BlockPool`。按 group 分解所需块，按真实共用池聚合后做容量判断。`allocate_slots` 有整序列 fit 和当前步扣除 `reserved_blocks` 后的 fit 两个检查。零引用 APC 块仍可在 free queue 中作为驱逐候选，分配量按 free queue、实际新引用和预留计算。目前原生 P→D connector 走请求级延迟释放，scheduler 在处理 `finished_sending` 后才释放请求持有。由此可分别检查 D 何时准入、硬件何时读完、P 何时真正返还页。[KV coordinator](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/kv_cache_coordinator.py#L82-L184) · [allocate_slots](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/kv_cache_manager.py#L338-L418) · [BlockPool](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/block_pool.py#L401-L441) · [P connector](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1882-L1928) · [scheduler](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L2221-L2248)

## 张书豪及合作者的工作

### TStream 与 MorphStream

**TStream，ICDE 2020。** Dual-Mode Scheduling 在事件准备阶段收集状态访问操作，切换到批次状态处理后再给出结果。Dynamic Restructuring 把事务拆成操作，在保留同键顺序及跨状态依赖的前提下执行独立链，在这一批处理完后回收临时版本。公开 benchmark 的 `SLBolt_ts` 在 marker 到来时触发 `start_evaluate`。可复用的是把准备、重负载执行、可见结果各自计时，并找出到底是哪条依赖阻止批次进展。实验先比较固定大小打包、按通信完成选择和按真实消费者就绪选择。只有在页和版本已合法、存在两个以上可选任务、选择结果能提前实际派发时，才有继续研究调度的机会。批处理和就绪操作并行在 TStream 已有，跨 NPU 的可访问性、HIXL 完成和 P 物理页返还还需独立建模。[原文 §IV-B、§IV-C](https://arxiv.org/html/1904.03800#S4) · [固定版执行入口](https://github.com/Xtra-Computing/briskstream/blob/98477d6e3088d2762d36809871f3a8f86b96bef0/BriskBenchmarks/src/main/java/applications/bolts/transactional/sl/SLBolt_ts.java#L42-L87)

**MorphStream，SIGMOD 2023、TKDE 2025。** TPG 记录事务内逻辑依赖、访问同一状态的时间顺序，以及跨状态的参数依赖。结构化调度的 BFS 按层同步，无结构调度在依赖就绪后把操作交给队列。还有任务粒度与回滚处理的策略维度。扩展论文讨论动态状态访问和窗口多版本。这个机制适合模拟器从同一批 trace 检查哪些等待边可以拆掉。我们的边应来自 Attention、SSM/conv 对齐、rank 末端事件、设备计算及可写页。先确认被挡住的后继是否真的独立，再用相同可用描述符重放 BFS 式屏障与就绪触发。r19/r20 的候选常仍在等前一 chunk 结果，按现有边重排没有收益。一般的 DAG、计数器和 ready-queue 属于明确先例。[原文 §3–5](https://me.sage.org.ai/contents/research_papers/2023/2023_morphstream_sigmod_2023.pdf) · [扩展版](https://me.sage.org.ai/contents/research_papers/2025/2025_morphstream_tkde.pdf) · [BFS 实现](https://github.com/DataSysResearch/MorphStream/blob/6f729e64d3c472b5f75cd92363156d4d5f2dea24/morph-core/src/main/java/intellistream/morphstream/engine/txn/scheduler/impl/og/structured/AbstractOGBFSScheduler.java#L18-L34) · [非结构调度实现](https://github.com/DataSysResearch/MorphStream/blob/6f729e64d3c472b5f75cd92363156d4d5f2dea24/morph-core/src/main/java/intellistream/morphstream/engine/txn/scheduler/impl/og/nonstructured/OGNSScheduler.java#L69-L110) · [本项目前序依赖检查](critical-path-attribution.md)

### Spacker 与 RTSFaaS

**Spacker，ICDCS 2025。** 把状态迁移拆成键优先次序、分块推进粒度和本地重载或跨机传输的选择。图 5 的控制消息只针对受影响的数据路径，旧任务等待相关控制消息后迁出，新任务缓冲受影响键的记录并在迁完后恢复处理，无关键继续执行。公开 Flink fork 的 planner 有 key-group 排序、按 `syncKeys` 分批及 replica 配置。可以借鉴只挡住受影响 group 的推进协议，比较全请求一起通知和按安全分片通知，再观测有多少 P 页的最后引用真的随之解除。渐进迁移、优先顺序、目的端本地重载属于已有机制。P/D 的额外约束是设备 DMA、持页引用和完整 Hybrid bundle 的消费语义。[原文 §III–V、图 5](https://me.sage.org.ai/contents/research_papers/2025/2025_spacker_icdcs_2025.pdf) · [Flink fork planner](https://github.com/sane-lab/Spacker/blob/bf85b36ac05bdf6358765695625b14393ecdbcc5/Spacker-on-Flink/flink-runtime/src/main/java/org/apache/flink/runtime/spector/controller/impl/StateMigrationPlannerImpl.java#L232-L288)

**RTSFaaS，USENIX ATC 2025。** 用数据与函数亲和性决定最初的对象租约，然后从全局事务图选择跨 worker 的租约交接次序。远端执行者可经 one-sided RDMA 取得对象，按 lease flag 验证访问权，用完后归还。适合借鉴把数据复制、所有权和最后引用收尾分成不同事件，给受控分片一个版本标识，明确重试和取消时谁仍能访问源页。其可变对象互斥与事务一致性问题和 APC 共享的不可变快照不同，复制后立即返还 P 需结合设备端的最后一次访问与源引用确认。公开 repo 可核对 RDMA 管理和 region token 交换，租约算法据论文核对，源码目录本身未用于证明 NPU 实现兼容。[原文 §3.3–3.4、图 6](https://www.usenix.org/system/files/atc25-zhao-jianjun.pdf) · [公开 RDMA 实现](https://github.com/CGCL-codes/RTSFaaS/blob/89d7b65b58cd65b6aa69fe2f429793810e7ba766/morph-core/src/main/java/intellistream/morphstream/common/io/Rdma/RdmaDatabaseManager.java#L128-L152)

### BriskStream、FineStream 与 SAGE

本项目已将 FineStream 的实测服务曲线用于 D 重新组批，按实际宽度与上下文估计 token 步频。MorphStream 的可运行操作集合对应已接纳、未发首块的 P 等待请求与已合法接收 KV 的 D 请求。按依赖完成再释放下一轮，在相同 P token 预算内评价重排及让位的净效应。r33 同模型原序与全队列 SJF 为 **405.21→446.65 token/s**，分窗排序反而降低吞吐，说明关键动作受队列覆盖范围及会话释放约束。[应用结果](cold-prefill-order.md) · [FineStream](https://www.usenix.org/conference/atc20/presentation/zhang-feng) · [MorphStream](https://doi.org/10.1145/3588913)

**BriskStream，SIGMOD 2019。** RLAS 用生产者和消费者之间的相对 NUMA 位置估算每条边的处理速率、内存与带宽代价，再联合做算子复制及放置。实现将受限下游速率反馈到上游速率估算。可借鉴逐 P→D 路由、rank/peer 和端口校准，以局部资源余量判定可重叠或改变组批。r29/r30 的端口重配对时变流量敏感。对照应使用同一批次同字节的逐口服务时长，看最晚相关 rank 和实际 P 可派发时点；提升非限制端口吞吐只作诊断。按拓扑优化放置及通过边反推生产供给已有先例。[原文 RLAS](https://arxiv.org/html/1904.03604) · [BackPressure 实现](https://github.com/Xtra-Computing/briskstream/blob/21220007589bd5352b6a8d9053a4b5ba2921b927/BriskBenchmarks/src/main/java/brisk/optimization/model/BackPressure.java#L32-L113) · [端口实验](cross-domain-transport.md)

**FineStream，USENIX ATC 2020。** 对共享内存的集成 CPU/GPU 做算子级在线 profiling，模型根据各算子在不同设备和窗口负载下的代价选择运行位置。可迁移的是用少量同构 profiling 校准 CPU 描述符准备、设备复制和 NPU 计算的服务曲线，以及验证并行提交时服务时间是否变化。两机 HIXL 具有独立注册域、端口上限和目标设备地址，需实测实际设备重叠及传输竞争。其细粒度设备调度和动态窗口本身属于已有方法。[原文作者版 §3–4](https://me.sage.org.ai/contents/research_papers/2020/2020_finestream.pdf) · [USENIX 页面](https://www.usenix.org/conference/atc20/presentation/zhang-feng)

**SAGE，ICML 2026。** 从逻辑推理工作流生成应用级 DAG，各 replica 有有界输入队列，满时阻塞上游，配合算子契约、运行遥测和可替换的准入策略。公开主线已经包含工作流、batch 和 serving 接口，论文描述的有界队列与 trace 校准按 §4、附录 H/J 引用。适合把应用前驱、P 计算、D 接收和 EOS 组成大图，记录背压是在发端校门之前还是卡在发端持页状态。把 queue size 换成 D 最大 running 也属于通用反馈控制。若相同队列深度下真实页缺口和最终供给不同，再比较用实际可分配容量及 D 许可的决策能否胜过常量队列。[会议与全文](https://proceedings.mlr.press/v306/liu26co.html) · [公开运行时和 serving 代码](https://github.com/RIDE-Lab/SAGE/tree/29f446b9c9eb91363cfe0f46d88da2cf19d25020/src/sage)

### 直接收紧创新边界的近期工作

- **BidKV，SC26。** 面向活跃请求的 KV 紧张，用回收量与干扰代价排序，代价考虑重算、完成比例及历史抢占，再调用原生抢占。公开实现给 SGLang 提供 radix-tree private-token 分支，vLLM 路径按申请的 token 规模做近似。这个工作已经包含“按真实或可估回收量排序”的近邻。我们的对照要落在已完成 P 计算、因在途交接仍持页的请求上，量化改变最后引用的解除顺序，追踪被阻塞的请求和完整会话。优先比较请求计数、朴素字节/块计数和按物理引用折合共同池可分配量。[会议题录](https://sc26.conference-program.com/presentation/?id=pap1108&sess=sess210) · [作者原文 §4](https://me.sage.org.ai/contents/research_papers/2026/2026_bidkv_sc_2026.pdf) · [两支评分实现](https://github.com/CGCL-codes/BidKV/blob/3f3a0b31051cedeea63aab24a6f0bcb2ac28f0a2/src/bidkv/baselines/bidkv_strategy.py#L100-L155) · [vLLM scheduler hook](https://github.com/CGCL-codes/BidKV/blob/3f3a0b31051cedeea63aab24a6f0bcb2ac28f0a2/src/bidkv/adapters/vllm/scheduler_hook.py#L1-L24)
- **StateAxis 的 Hybrid state groups、会话生命周期、跨实例状态与 host-tier。** Hybrid MOD 已声明 full-KV/recurrent 组联合准入、原子发布及 generation-qualified release，当前公开证据为软件/契约，正式真实设备用例存在启动阻碍。会话 MOD 已把代际、会话关闭时的独占状态释放和租用状态的驱逐排除写进契约。workflow-disaggregated MOD 只做请求边界上的远端候选判定和本地恢复，公开数据面禁用远端分配、传输和长期所有权。host-tier 已定义预算和 generation/location epoch，另保留 Qwen2.5-14B/910B2 历史顺序预取负结果，候选排队约 570 ms、设备重叠为零。版本化状态、本机有界暂存、会话生命周期和全组联合发布均要引用这些机制。差别应定位到原生 P/D 数据面的物理释放与 D 可消费阶段，以及完整会话是否提前排空。[Hybrid MOD 契约](https://github.com/intellistream/stateaxis-typed-hybrid-state-groups/blob/d8d1c6c807aa9e898cfff7773e815d94d9b703e8/README.md#L5-L14) · [Hybrid MOD 证据](https://github.com/intellistream/stateaxis-typed-hybrid-state-groups/blob/d8d1c6c807aa9e898cfff7773e815d94d9b703e8/EVIDENCE.md) · [会话 MOD 契约](https://github.com/intellistream/stateaxis-session-lifecycle/blob/8d5098fb07a1621d1562e135a9ef7eaa7da185bb/src/stateaxis_session_lifecycle/vllm-hust-extension-v0.3.json#L69-L83) · [跨实例 MOD 的执行范围](https://github.com/intellistream/stateaxis-workflow-disaggregated-residency/blob/786dcfeebec2652b537d0f09544fc20e952d35e7/README.md#L5-L16) · [host-tier 历史对照](https://github.com/intellistream/stateaxis-host-tier-residency/blob/d8f28487f5cf8d76b70991991eb8fd30d70dc920/README.md#L14-L34)
- **Dual-Memory Request Scheduling，2026 预印本。** 已直接形式化 P 完工后持 KV、D 准入反过来释放 P 内存，理论用已知请求与瞬时交接优化完成时间之和，实测用代理近似 D 容量和原生 GPU 数据面。实验应保留 D 内存/slot 准入为强对照。可能产生区别的是在线可观测的实际物理 ref、两方向传输及尾会话闭环。仅使用 P/D 双重预算没有贡献区分。[原文 §2、附录 H.2](https://arxiv.org/html/2610.15579v1) · [详细相关工作](related-work.md)

## 其他流系统和 HPC 的直接对照

- **Flink Credit-Based Flow Control 和 FLIP-183。** 接收者按可用网络缓冲发 credit。独占 buffer 和浮动 buffer 分开，网络 buffer debloating 按观测消费速率估计合适的缓冲时长。适合作为 D 接收容量和有限在途数的基础对照，特别是分开 D 初始目的页余量、解码运行槽、暂存地址和网络服务额度。在同样 D 限流下，量化 P 页的实际返还。用预测消费率调预算本身已有直接先例，而且消费速率估错会使下游断粮。[Flink 项目设计](https://flink.apache.org/2019/06/05/a-deep-dive-into-flinks-network-stack/) · [FLIP-183](https://cwiki.apache.org/confluence/spaces/FLINK/pages/184616569/FLIP-183%2BDynamic%2Bbuffer%2Bsize%2Badjustment)
- **Naiad/Timely、Megaphone、Meces。** Naiad/Timely 的 capability 与 outstanding count 证明一类逻辑时间之后不会再收到更早的事件，Megaphone 用这个前沿规划分片迁移，Meces 按下游正在或即将处理的热键按需拉取。可借鉴为明确的 generation/rank 进度和有界等待，检查 consumer 需要的片与持有上游 allocator 的片是否同一组。迟到 ACK、重复 ACK、失败和取消应保留可验证的最后使用协议。提前准备、按需优先与版本前沿已经有先例。逻辑进度要通过设备和 allocator 事件才能对应到页的复用条件。客户端的会话完成仍由真实前驱、工具间隔和 EOS 定义。[Naiad 原文 §3](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/naiad_sosp2013.pdf) · [Timely 进度模型](https://timelydataflow.github.io/timely-dataflow/chapter_5/chapter_5_2.html) · [Megaphone 原文 §3–4](https://www.vldb.org/pvldb/vol12/p1002-hoffmann.pdf) · [Meces 原文](https://www.usenix.org/system/files/atc22-gu-rong.pdf)
- **MPI 4.1 分片通信与原生 HIXL。** 通信端可提前初始化同一持久请求，生产者分片报就绪，接收者可查部分到达。传输批次与可见性粒度可以独立设计。固定版 vLLM-Ascend 已有按层数据/事件发送的 `MooncakeLayerwiseConnector`，也有 Mamba cache align 模式的代码路径，实际构建和 Hybrid 配置仍需核对。Mooncake 的公开 v0.3.11 HIXL transport 已按远端 context 整理 slice，进一步减少的是上层同端点提交次数、跨层就绪阻塞或返还事件的合并延迟。比较现有 layerwise 与现有 context 批提交，先分离主机 CPU、MAC 字节和实际设备时间，异步 API 返回只作为开始测量的边界。[MPI 标准 §5](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node86.htm) · [Ascend 逐层发送](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py#L1690-L1819) · [Mamba 模式选择](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py#L291-L320) · [Mooncake HIXL 分发](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11/mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/slice_dispatcher.cpp#L64-L98)

## 三个可独立反证的设计方向

以下都可服务于同一候选想法，即按实际可分配的状态推动跨机交接，目标是缩短卡住会话排空的依赖。先在 trace 上确认每条边，再选择有证据的机制做单变量对照。阶段收益要求能传到正式整轮吞吐，才支持纳入主设计。

### 1. 用页缺口决定准入与交接优先级

对每个实际池取 `free` 与原生 `allocate_slots` 在当前检查分支的 `need`，按相同预留口径算缺口。对已合法交接的候选集合，用稳定物理 block ID 合并共享块，计算完成交接后能变成零引用且能进 free queue 的页数。每块只计一次，全部最后引用都解除时才计入。若一笔最快或最大交接还不足以填缺口，就检查在同样 D 预算下，是否有另一组合能越过缺口并解除关键后继。调度看当前 D 目标页/运行槽与端口预算，D 的页占用到真实释放时更新，在途额度可在传输结束后独立归还。已接纳且持有源页的交接保留独立的回收进展通路。

令 $F_p(t)$ 是 P 实例此时的实际 free queue 块数，$q_i(t)$ 及 $c_i(t)$ 分别是请求 $i$ 在命中原生分支时的需求与预留，则缺口是 $d_i(t)=[q_i(t)+c_i(t)-F_p(t)]_+$。令 $L_b(t)$ 表示一个池内物理块 $b$ 的未解除引用集合。若一组安全交接 $S$ 解除的引用集合为 $U(S,t)$，其即时贡献可按下面的式子做影子比较。

$$
R_p(S,t)=\left|\left\{b\in\text{pool}_p: L_b(t)\ne\varnothing,\ L_b(t)\subseteq U(S,t),\ b\text{ 可重新分配}\right\}\right|.
$$

只有 $R_p(S,t)\ge d_i(t)>0$，该组释放才可能直接让该次容量检查通过。把有向口、D 页、D 槽和其他领取者保留为单独约束，按事件重算谁先拿到这些块。

**对照与停止条件。** 先用同一 trace 比较原生、固定在途上限、D 页与槽 credit，以及增加源物理缺口的策略。筛选至少有两种合法动作，且同计数/同载荷附近的动作只有一种能使 `allocate_slots` 恢复。检查 P 下一批、D 就绪和最后会话链的改变，重新模拟受影响的组批与资源。若原生没有真实分配失败，或者成功后仅把等待移到同样饱和的最终 D，优先停掉该策略。只有影子差异反复出现才做 100 Gbps 同带宽配对，200 Gbps 用来复核实际端口压力改变了哪项依赖。

### 2. 分片完成和按实际源引用提前返还

对已稳定的片，区分源版本可读、D 目的分配、同一版本所有相关 rank/设备完成、P 原请求何时允许返还该片。把 D 的完整 bundle 发布保留为原生消费门槛。在固定 D 接收顺序与跨机字节的第一对照里，仅把 P 的某些确认与合法页返还移早。若硬件搬运结束到释放之间确有批级空档，且足够数量的片能够独立终止最后引用，才实现逐片确认。若真正的阻塞发生在 D HBM 接纳前，独立比较有限 D DRAM 和 P 本机 host 暂存。每项按同一容量预算核对额外 DMA、注册、重试和持有量。较新版 Ascend D2RH 已覆盖 P HBM→D DRAM→D HBM，固定版和设备支持须按版本另查。

**对照与停止条件。** 比较原生请求级、可用的固定逐层发送、分片完成后的源返还，必要时只单独增加单片暂存。若缓存命中变动导致重算或 D→P 字节变化，同时记账，避免把丢失 APC 可用性当作零成本。只在实际页进入 free queue、唤醒指定 P 分配且在较早的末会话完成链中被消费时计作进展。若 P 最后一次读与 D 末片就绪总是相邻，或提前返还的页仍有其他引用，停止此项。[D2RH 官方设计](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/disaggregated_prefill.html#mooncake-d2rh-connector) · [本项目已有暂存对照](cross-domain-transport.md)

### 3. 按就绪依赖控制聚合与提交时间

限定同端点、同 cache 版本及同地址合法性的已就绪描述符。用固定字节上限控制聚合，比较固定打包与就绪触发额外的早提交条件。只有新片恰好是能派发的下一批的最后未完成输入，或能立即触发一笔足以解决 P 缺口的返还，而且现有路径还有设备和端口余量，才提前 flush。其余继续合批，保证先前批次和既有 D 交付可以推进。发送聚合、设备完成及 D 最终发布各有独立的观测边界。

**对照与停止条件。** 在同组批和输出发布次序下，先拆准备、提交队列、设备活跃、最后依赖消失和首个可用 chunk，比较固定字节打包、空档即早发、依赖触发早发。带宽和请求级字节保持一致，检查前批拖慢多少及后继最终得到多少。已有 r20/r21 的改善常停留在局部，r27c 尾请求的原生传输约 17–22 ms/rank。找不到提交或通信等待决定派发与最终排空的自然样本时，保留必要的工程打包即可。[既有流水和排空对照](critical-path-attribution.md)

## 实验证据与下一轮查源

| 观测 | 要确认的因果边 |
| --- | --- |
| P scheduler 在实际检查分支记录请求、批次、按 group 的 `need`、共用池的实际 `free` 和预留、结果与原因 | 把容量不足、队列等待、先前 worker future 未就绪和每步 token 预算区分开 |
| 交接中同一份状态的 request/版本、每 rank 完成与失败、P 处理通知、引用降至零或回到 free queue、其后谁分配成功 | 拆开 D 接收、传输完成、调度器收取与物理页可再分配，识别由哪批页解除哪个门槛 |
| 每条固定策略下的 D 运行/接收队列、目的块分配与 slot、开始和结束、`idle-and-eligible`、实际输出计数 | 判断 P 的提前是否被最后排空的 D 消费，区分给饱和 D 的积压与本来能用的空档 |
| 方向、端口、payload/MAC、HIXL handler/设备、P 和 D 的本机单调钟和同一组对时锚 | 区分 D→P 恢复、P→D 交接和纯 host 控制延迟，跨机先后给出对时误差 |
| 原会话前驱和工具间隔、预定根到达、客户端最后结束、逐请求终止原因与实际 token | 保留闭环和 EOS，用反序复测重建每轮实际排空链，报告输出总量与时长对吞吐的共同影响 |

下一轮优先查公开实现是否在 P→D 的分片确认后实际返还源页，验收条件包括 Hybrid rank join、当前 910B2 支持和目的槽不足时的进展路径。其次查下游 credit 覆盖发送端共享内存而且有真实回收事件的系统，寻找能区分方向 1 与固定在途限制的既有方法。对网络与提交机制，只有在自然高压 trace 中看到 host submit、传输或 ACK 的独立等待占住同一条排空链再补深。每条来源保留版本、可核对代码或证据所在文件、不同执行环境的边界，以及一项能判别作用的对照。三条方向在这里是候选假设，待上述 allocator 与最终会话链的数据选择。[跨领域网络筛选](cross-domain-transport.md) · [问题模型](problem-model.md) · [主相关工作](related-work.md)

## 无 P 页阻塞后的分支

r31 同 cohort 先跑 200 Gbps，再跑 100 Gbps，两臂均完成 170 条，观察到的 `allocate_slots` 检查全部成功。以下按归档重算。把 D 目标页分配至完整 KV 可见的区间设为零时，固定当轮 EOS、准入和其他实测驻留时长。阶段敏感性只用于筛选能影响末会话的区间。两臂选择出不同末会话，实际生成量也不同，该次顺序的两档观测用于归因。[r31 摘要](../results/2026-11-20-r31-physical-pages/summary.json) · [负载与配置](../results/2026-11-20-r31-physical-pages/manifest.json)

| r31 原生自然会话观测 | 100 Gbps | 200 Gbps |
| --- | ---: | ---: |
| P0/P1 观察到的分配检查 | 466/608 | 420/528 |
| 检查时 P0/P1 最小余量（块） | 2,407/2,404 | 2,460/2,532 |
| D 入口到目的实际分配最长（毫秒） | 2.60 | 197.88 |
| D 八口一秒聚合 TX 峰值（Gbps） | 49.21 | 73.88 |
| 最后完成的会话起始请求 | `natural-crack-7z-hash:31` | `natural-winning-avg-corewars:126` |
| 完整排空时间（秒） | 307.749 | 419.937 |
| 理想移除全部 D 分配→KV 可见后提前的整轮秒数 | 0.328 | 0.758 |
| 折合固定输出量的阶段敏感性 | +0.107% | +0.181% |

100 Gbps 主判定臂共观察至少 1,074 次分配检查，0 次失败。D 八口聚合 TX 达到档位 85% 的样本时间为零。两臂的 D 均存在有后继 P 工作却无就绪工作的窗口，其中与 D 出口接近限速相交的采样时间均为零。分配检查与敏感性结果把页准入、提前返还、仅 P→D 逐层交接降为条件性机制。r31 归档没有 Store GET 的逐笔 descriptor 或设备提交事件，仍需从相同自然工况补齐 D→P 的可用、提交和消费链。

### 两个方向的临界事件

| 数据路径 | 已有实现及门槛 | 优化时需要独立观测 |
| --- | --- | --- |
| **P→D，当前原生 P-first** | 基线的 `MooncakeConnectorV1` 由 P 完工返回源页 `remote_block_ids`，D 有可接收的页以后拉取，P 处理结束通知再释放。 | P 返回、D 入口、D 实际目标页分配、每 rank 读完成、D 完整 KV 可见、首 token 与真实 EOS。源页等待只在挡住真实 P 分配或影响最慢会话时有吞吐机会。[基线接法](experiment-audit.md) · [P 源页保持及公布](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1882-L1928) |
| **P→D，Ascend 已有 D-first 逐层** | 示例代理先提交 D。D 在 `update_state_after_alloc` 生成本地目的页和通道信息，通过 metaserver 转发给 P。P 按 `reshape_cache_event` 送出已产出的层，同头比为 1 时已有源/目的连续块合并，由发送线程按 session 做同步批写。需要 D 先保有合法目的页；末层且 chunk 完成才走最终接收回调。P 侧 `request_finished_all_groups` 返回无需走原生整请求的延迟释放分支。 | 同时记早保留 D 页和接纳被拖延的时间、P 计算与真实 WRITE 的交集、D 完整可见与末会话 EOS。该对照以不同的接纳顺序运行，按实际 APC、路由和目的页预算核算。Dynamo/SGLang 的 D 目标页预分配也是已有机制。[D-first 代理](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py#L423-L475) · [分配后公布 D 目标页](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py#L928-L976) · [连续段](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py#L367-L392) · [批写与末层回调](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py#L489-L531) · [P 结束返回](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py#L1102-L1124) · [Dynamo/SGLang 握手](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/disaggregation) |
| **D→P，历史状态的 Store GET** | P 查 APC 与外部前缀后，对本地可写目标页做恢复，按请求把缺失的 Hybrid group 组成 key/地址/长度，调用已支持多键多缓冲的同步 GET。数据是历史快照，源要有合法副本/租约，目标 P 的块及写权限必须有效。当前基线配的是普通同步 Store。`load_async` 是独立的原生请求路径，具体工作量、目标页分配与这一配置的正确性先查。固定版本另有 `use_layerwise` Store 流水，已做跨层共享描述符、按 byte/block 上限解组和逐层等待，文档的支持域是 memcache、MLA/SFA，明确排除 Mooncake 和多 group Hybrid。 | 记录 store 键和合法源版本、最早合法目的页、组准备及提交、设备 GET/每 rank、forward 依赖与最后缺失输入。保留原 P 本地命中、自然会话及其他计算依赖。P→D 的早分配落在 D 侧，D→P 的控制权在 P scheduler/worker 和 Store。[固定普通与异步分支](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L734-L845) · [Store 现有逐层预提交](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1314-L1396) · [固定版逐层支持范围](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/docs/source/user_guide/feature_guide/layerwise_kv_pool.md#L204-L241) |

**合并在哪一层。** Mooncake v0.3.11 公开 Store 实现先对本次 keys 做 `BatchQuery`，再构造 scatter slices。`Client::BatchGet` 现有默认提交路径按 key 提交并等待结果，另有启用 `prefer_alloc_in_same_node` 时按传输 endpoint 向 TransferEngine 批送的分支。这个参数和 HIXL 最终分发形状以实际 wheel 的运行事件为准。合并跨请求描述符时，现有 Store slices map 按对象 key 建立。同 key 对应两个目的地址时，保留两组目标和版本，确认运行路径能逐项跟踪，再核对每份状态写入。新一些的 Mooncake 主线已经公开 session-based ranged GET，一次 Master 查询和 lease 可为后续多次偏移读取复用，固定 tag 的公开接口没有这组方法。端点批送、Master metadata 复用和逐层读取均已有近邻。[v0.3.11 的 key 查询](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11/mooncake-store/src/real_client.cpp#L4742-L4779) · [slices map](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11/mooncake-store/src/real_client.cpp#L4882-L4908) · [endpoint 分支](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11/mooncake-store/src/client_service.cpp#L915-L1025) · [普通分支](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11/mooncake-store/src/client_service.cpp#L1033-L1152) · [主线 ranged GET](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/api-reference/python/mooncake-store.md#session-based-ranged-multi-buffer-transfer)

### 可直接借鉴的合批和软件流水机制

- **TStream 的标点触发与平台期。** Progress Controller 周期性发 punctuation，事务模式靠两个屏障结束一批。放宽间隔能增加并行性、摊薄同步成本。Figure 12 的 GS 一旦到达吞吐平台，继续增大间隔就增加排队时延。本任务可看同字节下 HIXL 每次提交节省随 group size 的边际变化，同时记录首个合法描述符的队龄。若已经到平台，提前收集只会挪动等待。MorphStream 则说明跨状态依赖积累和任务 ready 列表要单独记，见上文实现。[TStream §IV-B、§VI-F](https://arxiv.org/html/1904.03800#S6.SS6)
- **Beam 与 Spark 的大小和队龄控制。** Beam `GroupIntoBatches` 以 key 分组，可按元素数/字节或首项开始的处理时间到期 flush。Beam `BatchElements` 的项目文档明确称跨 bundle 有状态缓冲可以成为自身的吞吐瓶颈。Das 等把 Spark Streaming 端到端时间拆成凑批、排队和服务三部分，通过反馈调 batch interval。少凑批时间但批服务跑不过到达会积累队列。两者适合做同字节、同队龄的固定/自适应 batch 控制，并给出窗口期内何时会积压。[Beam GroupIntoBatches](https://beam.apache.org/releases/javadoc/2.48.0/org/apache/beam/sdk/transforms/GroupIntoBatches.html) · [Beam BatchElements](https://beam.apache.org/documentation/patterns/batch-elements/) · [Das 等 §2、§5](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2014/EECS-2014-133.pdf)
- **Timely 的进度更新累积。** 它可合并普通进度更新，只在有能力改变全局 frontier 的负更新或 eager 模式才发送。原作者在合入说明中指出，延迟发可能让敏感计算从一次 worker step 传播完毕变成需要多轮。这个实现说明了控制消息减少和关键路径推进之间的取舍。候选触发器应单独标记它解除的真实消费边，立即发与进度触发在同样负载下比较。SnailTrail 的窗口级 critical participation 可用于找候选时间段，最终仍从自然 EOS 反追完整会话的最后路径。[Timely PR 说明](https://github.com/TimelyDataflow/timely-dataflow/pull/228) · [Timely 发送条件](https://github.com/TimelyDataflow/timely-dataflow/pull/228/files) · [SnailTrail §3–4](https://www.usenix.org/system/files/conference/nsdi18/nsdi18-hoffmann.pdf)
- **FaSST 的零额外等待合批与软件流水下界。** FaSST opportunistic doorbell batching 组合同一时刻已有的发送工作，减少主机/NIC 通知，线上包数由协议及发送形态决定。把这个做法用作零等待的工程对照。Rau 的 modulo scheduling 同时按资源占用和跨迭代依赖给出启动间隔下界。把这一点用于推理时，要实测传输与计算重叠后每份服务变慢多少，还要检查在途/目的页预算与原生 D decode 的有效启动和排空。下游持续占满且保持相同的生成速率时，前移传输会增加就绪队列。r20/r21 已出现局部重叠而全轮改善很小的情况。[FaSST §3.2](https://www.cs.cmu.edu/~dga/papers/fasst_osdi.pdf) · [Rau 原文](https://web.eecs.umich.edu/~mahlke/courses/583f12/reading/rau_micro94.pdf) · [本项目流水实测](critical-path-attribution.md)

- **HiCache 的实际软件流水反例。** 在上游问题报告的 H100 TP8、仅 prefill 测量中，已有逐层设备事件，但 CPU 在 scheduler 上串行准备全部层的 H2D，设备复制与 forward 重叠仅 0.04%。异步入队改动在该报告的配对中提升总吞吐，TTFT p99 持平，涉及的实现是提议。SGLang 主线已合入 PP Prefetch Ticket，让各 stage 收到提前通知后独立完成 KV 与 sidecar 分配并发起 Mooncake GET，再汇总就绪。由此优先查本机是否也有 `CPU prepare → worker launch` 的顺序边、真实可写页和后继可用事件。分阶段提早暴露描述符、保留事件保护和逐 stage 提前读都有先例。[上游测量和改动提议](https://github.com/sgl-project/sglang/issues/38470) · [已合入的 PP Ticket](https://github.com/sgl-project/sglang/pull/36700)

### 先用同到达的 trace 决定是否值得上机

**r32 独立组件探针。** D→P 在 100 Gbps 档，64 个 12 KiB 片由 64 次独立提交并成 1 次，6.536→1.221 ms。840 个 512 KiB 片按每次 32/128/256/840 个提交都约 287.5 ms，约 12.25 Gbps，接近该单口 12.5 Gbps。200 档的对应大载荷约 143.7 ms、24.51 Gbps，仍接近单口 25 Gbps。反向同样使用 READ，D 发起获取，P→D 为有效载荷方向。P 端口维持 200 Gbps 硬上限，四种大载荷组形约 21.56–22.99 ms。可继续核查自然工况的小片合批是否移动真实门槛，大载荷增加每次向量大小已经遇到带宽平台。这些测量隔离了两种载荷方向的 HIXL READ。D-first 的 WRITE、Store 元数据、Hybrid 可见性和整轮 D 消费需各用对应事件验证。[探针实现](../pd_kv_transfer_opt/profiling/hixl_batch_probe.py) · [r32 实测结果](results.md#r32-同字节的-hixl-批提交)

**2026-11-20 更新。暂停 Store GET 完成边界方案的原型和实机验证。** 现有证据尚未确认原长会话里存在可独立推进的 forward 被无关读取挡住，并因此推迟整轮排空。以下保留架构候选和识别这一现象所用的判据。流计算和 HPC 机制可作迁移启发，论文贡献与 LLM 推理领域的直接相关工作比较。[七步法状态](research-seven-steps.md#4-what-is-the-key-idea)

原假设是**用真实消费依赖划分 Store GET 的提交与完成边界**。对同一时刻已具备合法源、目的页和队列位置的描述符，同物理端点且不冲突时可零等待合批。只有实际能独立解除 worker 或下一 chunk 执行门槛的读取，才有独立发布的意义。每次运算仍以所需的全部 group、rank 和前序结果为依赖。如果凑批需要额外等待，在同队龄上限下核算提交开销、队龄及最慢片拖尾。

对一个真实消费事件 $v$，令 $J_v$ 是需要的传输片和合法发布条件，$g_j$ 是片 $j$ 达到可消费的时间，$u_v$ 是原生其他依赖和资源到位的最早时点，则 $v$ 的门槛是 $a_v=\max(u_v,\max_{j\in J_v}g_j)$。用每笔的原生目标页、准备和发布事件还原 $a_v$。候选只改 $g_j$ 时比较 $a_v-a'_v$，保留负值和方差，必要时重算共享 stream、batch、D 端队列及下一轮会话依赖。把两种极端理想干预分别做一次，即去掉全体合法描述符的组批/提交等待，以及把该方向整段归零。在固定工作量与当轮其他实测成本的敏感性下，若后者也不会移动末会话，就停止细化该方向。在 r31 两档的 P→D 上，归零对应约 0.11%/0.18% 的整轮吞吐改善量级。

若独立消费及排空传导后来获得直接证据，原方案的三个对照为逐请求原生、同端点零等待合批，以及同预算下保留独立消费门槛的合批。机理试验固定组批、已观测路由、有效载荷与发布约束。正式配对保持原生路由策略、APC 与 EOS，记录实际去向和组批变化。Nuska 的 deadline/laxity 作用于阶段流，AMPD 的 D→P 已实现多轮 prefill 重排和跨请求读算重叠。Store 异步和 P→D D-first 逐层是既有的 LLM 系统对照。MPI 分片提供跨领域启发。论文增量须定位到实际提交或完成边界、解释本领域强对照的局限，并验证整轮差异。[Nuska §4–5](https://arxiv.org/html/2603.17456v2#S4.SS4) · [AMPD 实现](https://arxiv.org/html/2602.14516v2#S6) · [已有并行对照](prefetch-selection-claim.md)

1. **现象门槛。** 在 APC/EOS 的自然会话原生 trace 中找到至少两个独立窗口。每个窗口有至少两项合法可执行的候选输入，在同路由同端点内具备有效源版本和目标页，且不同送法确实会改变哪个实际消费门槛何时打开。实际设备时序可区分于观测误差及无改动复测噪声。若外部前缀几乎都本地命中、只有单个大 GET、两个输入始终一起消费或 worker 仍在等自己的前序结果，维持当前批处理。
2. **机理门槛。** 用相同地址形状、同组描述符数和字节的隔离 HIXL READ 探针标定提交与设备代价，再按方向、已有合并程度和 rank 归约回放实际 trace。100/200 是 D 八口的出口上限，READ/WRITE 本身也有不同方向的应答与排队成本，按 payload 的实际发送端和 MAC 方向解释两类探针。r27c 高压诊断两档已观察到 268/316 个 worker 恢复分组，其中 172/204 组含多个请求。两档各 1,056 笔非空 GET，单次 p50 为 56.5 个 Store key、550.35 MiB，每个请求已有多键提交。Store key 数与底层描述符数分别记录。裸探针从单 descriptor 开始的节省只能填入实际还能合并的调用差，扣掉凑批、局部复制、更慢 rank、前批推迟和任何额外 ACK。要求预测实际 P forward、D 可见或后继轮的提前，D 已有足够就绪工作时按实际资源边扣去。[r27c P 方法归档](../results/2026-11-19-r27c-native-sustained/r27c-P.jsonl.gz)
3. **整轮门槛。** 重放原生及反序配对的关键请求和最晚会话。若只改变 TTFT 或只把 P 等待搬到 D 队列，作为组件记录。隔离试验过关后，从原长会话补一轮 100 Gbps 的原生逐笔 trace。沿同一个 P worker 串联目标页可写、`worker_load_begin`、实际 Store 与 HIXL 提交和完成、`forward_enter`，再连接 D 的 KV 可见和会话 EOS。在 100 Gbps 下与原生做同带宽对照，形成正式主判定。200 Gbps 用来复核放宽限速的响应，其真实端口压力由计数确认。正式轮报原始计划起点到最后 EOS 的总输出 token/s，逐请求记实际 token 数、结束原因和耗时。整轮结果须方向可重复，能在不同顺序对照下解释实测生成量与耗时的差异，并有从实际消费门槛到最后完成的事件链。稀缺卡时优先用零时长敏感性和同机裸探针筛掉收益已被下游遮蔽的分支。
