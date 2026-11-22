# r14–r37 关键路径归因

归因基于两台八卡 910B2 的 2P2D 实验。r14–r17 固定两个约 223K/239K 的长前缀、关闭 P 本地 APC、每条输出 16 token。r18 使用 Terminal-Bench 2.1 原有 30 个长会话、170 条请求，开启 P 本地 APC。r19/r20 在同一数据里挑选 10 条高恢复量后继，保留 18 条真实前驱、P/D 路由和逐请求输出预算。计时臂固定输出 3,971 token，其中两条各输出 1,024 token。100/200 Gbps 均指 D 八口合计发送上限。r17 的原档指八口各 200 Gbps 的配置。 [负载与固定数据版本](workloads.md)

## 带宽限制如何放大成排队

r14 每轮 96 条、16 请求/s 到达，跨机 GET 固定 **914.164 GB**。100→200 Gbps 时，观测 TX 峰值统计从 92.77 到 182.11 Gbps，最慢 rank GET p50 从 1.707 到 0.828 s，输出吞吐从 **16.85 到 33.11 token/s**，TTFT p99 从 **85.97 到 41.06 s**。固定字节下，纯载荷排空下界为 73.13→36.57 s。端到端平均 TTFT 少 26.87 s，可拆成 P 引擎排队少 **12.678 s**、P 派发到末批结果少 **8.046 s**、P HTTP 引擎外少 **6.066 s**，其他阶段合计少约 0.08 s。代理发起到引擎接纳的约 5.78 s 改善已包含在引擎外时间里。 [r14 阶段与抓栈](../results/2026-11-18-r14/summary.json) · [载荷下界与端口](bottleneck-and-validation.md)

按固定到达槽位对齐第一组 100/200 Gbps 的第 095 条请求。该请求两臂均在 P1、238,600 token，只执行一个计算批。下表各列单位为秒，P 阶段按 Unix 事件锚点拆分，末位有测量与舍入误差。

| 速率 | 代理发 P 至引擎接纳 | 引擎排队 | P 派发至批结果 | 批结果至 P HTTP 回包 | P/D 代理衔接 | D 开始至首块 | 客户端及代理交界余项 | 客户端 TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 17.371 | 51.984 | 12.792 | 1.307 | 0.046 | 0.740 | 0.070 | **84.310** |
| 200 | 4.558 | 29.929 | 3.532 | 0.327 | 0.046 | 0.664 | 0.068 | **39.124** |

请求自身四 rank 最慢 GET 只从 0.949 降到 0.518 s。各 rank 同批其他请求串行 GET 累计由 **10.681–11.245 s 降到 0.891–0.998 s**。同一请求在两个观测批分别处在恢复序列第 9/9 和 3/3，rank 0 的 worker→forward 从 12.217 到 1.529 s，forward 本身从 0.203 到 0.149 s。D 传输 handler 窗口约 0.141→0.139 s。100 Gbps 诊断捕获中，P0 rank 0 的设备 `hccl_batchGet` 覆盖匹配 GET 墙钟的 99.73%。抓栈同时看到 EngineCore 等 worker、worker 等同步 GET。带宽压低后，批服务变长，先前批、同批较早的 GET 和引擎取件把影响传给后续请求。限速因果在这组数据上成立。重传增量为零，PFC 计数没有 pause duration，尚无法把各次等待拆成线路发送、端点服务和网络暂停。 [正式 P trace](../results/2026-11-18-r14/r14-P.jsonl.gz) · [CANN 诊断](../results/2026-11-18-r14/profiler-summary.json)

## 更改服务时间后，瓶颈移到哪里

| 实验 | 实际改变 | 归因 |
|---|---|---|
| r15，同头单读加直接分发 | 每轮 GET 914.164→460.173 GB，吞吐在 100/200 Gbps 分别 +53.27%/+13.84% | 较快档已暴露约 0.42 s 的本机分发，低载相邻页可合并，并发下分散页导致数千次 HCCL 广播。 |
| r16，直接分发改为打包 | 跨机 GET 逐轮固定 460.173 GB，广播 p50 约 3,700–3,780→18.5，rank 复制 p50 约 414–419→138 ms | 100/200 Gbps 吞吐分别 +15.96%/+33.87%。每档配对 288 条后，TTFT 平均少 3.59/4.56 s，代理发 P 至最早 worker load 少 2.50/3.90 s，四 rank 的恢复区间少 1.19/0.82 s；逐请求最慢 GET 平均多 74/55 ms。收益主要来自本机复制和其后的等待缩短。 |
| r17，固定打包在不同带宽下的交叉 | 100 Gbps 的 48 对请求中，自身最慢 GET 平均省 1.054 s，四 rank 恢复区间省 8.906 s。原档的 48 对仍省 0.135 s GET，恢复区间却增加 0.149 s | 原档两个排空吞吐对照方向相反，当省下的 GET 不足以覆盖本机工作时，减量在恢复阶段已失效。 |
| r18，APC 开启的原会话 | 48 个有复制的批中只有 1 个恢复两条请求 | 双恢复批的 `video-processing:30` 和 `fix-ocaml-gc:31` 的 P 引擎队列为 5.794/5.289 s，同批 GET 并集 21.4 ms，copy 并集 31.7 ms，前一笔 copy 阻塞下一 GET 约 2.0 ms。在 46 个可用批的固定重放里，两种择路都加读 9.532 GB，按批就绪排序仅比按通信完工排序额外提前 28.55 µs。 |
| r19，APC 开启的高恢复波次 | 每臂实际 GET 41.832→21.238 GB，D MAC TX 约 42.47→21.57 GB | 100/200 Gbps 的 P0 GET 活跃并集少 2.01/1.06 s，加约 0.58/0.57 s 本机分发后，同步 load 净省 1.48/0.53 s。TTFT p99 分别低 21.88%/9.90%，排空吞吐 +5.71%/+3.65%。 |

r15–r17 的代理发起到 worker 区间含引擎接纳前、引擎排队和执行器取件，四 rank 恢复区间也包含同批其他请求。表中各方法的事件区间可能重叠，用于衡量暴露程度。r18 固定重放使用观测 GET 服务时间和释放时点，候选反例仍需实机。 [各轮设置、原始事件与聚合入口](results.md)

## 关键方法与两个尾部请求

下表固定 P0 的恢复顺序 `winning-avg-corewars:131 → train-fasttext:69 → regex-chess:106 → count-dataset-tokens:45`。每臂取最晚进入 host forward 的 rank，并从该 rank 自己进入 worker 起算。单位 ms。准备是 load 入口到首笔 GET，包含描述符和 preflight 配对。复制阻塞只计同步 copy 或异步提交加 fence，异步 copy 在途时间与 GET 有相交。首笔准备、GET、配对、复制与其他列互斥。其他包括 worker 到 load、load 结束到 forward 及未归类的笔间隙。

| 轮次与速率 | 方法 | rank | 准备 | Mooncake GET | 读后同头配对 | 复制阻塞 | 其他 | worker→forward |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r19 首对 100 | 原生 | 2 | 23.90 | 2,760.51 | 0 | 0 | 76.65 | **2,861.06** |
| r19 首对 100 | 同头单读同步打包 | 1 | 68.35 | 1,154.90 | 107.49 | 333.40 | 69.24 | **1,733.38** |
| r19 首对 200 | 原生 | 3 | 25.27 | 1,337.85 | 0 | 0 | 73.66 | **1,436.78** |
| r19 首对 200 | 同头单读同步打包 | 0 | 70.24 | 636.07 | 7.53 | 313.53 | 67.09 | **1,094.45** |
| r20 首对 100 | 同步打包 | 2 | 74.48 | 1,143.50 | 26.97 | 315.05 | 75.86 | **1,635.87** |
| r20 首对 100 | 同批异步打包 | 2 | 89.57 | 1,239.44 | 48.66 | 68.50 | 66.69 | **1,512.87** |
| r20 首对 200 | 同步打包 | 0 | 62.41 | 611.87 | 13.73 | 315.23 | 70.31 | **1,073.55** |
| r20 首对 200 | 同批异步打包 | 1 | 149.23 | 409.99 | 54.52 | 236.30 | 67.21 | **917.25** |

- **Mooncake 与 Gloo。** r19 的主要变化在原生 Mooncake 调用。r20 100 Gbps 首对节省了 246.55 ms 复制阻塞，其中 95.94 ms 被更长的 GET 抵消，另有 21.69 ms 被配对抵消。读后配对中的 peer GET 等待由 19.92 到 41.88 ms，两端都已进入 Gloo 后的时间由 6.85 到 6.33 ms。增加的是 rank 返回时差。200 Gbps 首对还出现一次更长的读前配对，最晚前向 rank 的 preflight 配对为 1.64→66.39 ms，已包含在准备列，和读后配对分开。相同字节和顺序仍会出现 GET 服务变化，现有 r20 时间戳不足以在 Mooncake 内进一步定位端口、完成队列或共享资源。 [逐请求 GET、配对、copy 事件](../results/2026-11-19-r20-packed-overlap/restore-worker-costs.json)
- **HCCL 分发与 fence。** 100 Gbps 首对 rank 2 的四笔 copy 设备时长约为 114/38/109/38 ms，改异步后各自变化不足 0.2 ms。前三笔已在下一笔 GET 内完成，末笔仍留约 35.2 ms fence。200 Gbps 首对的异步最后到 rank 1 有两笔较长 copy 在下一 GET 返回后分别还等 66.6/101.5 ms。低带宽提供了更长的同批隐藏窗口，同时拉长本身在关键路径上的 GET。 [r20 运行实现](../results/2026-11-19-r20-packed-overlap/runtime/same_head_kv_r20.py) · [设备重叠与 fence](../results/2026-11-19-r20-packed-overlap/summary.json)
- **TTFT 最慢 `regex-chess:106`。** r20 四对的 P 首派发→末批结果均减少，变化依次为 −118/−140/−135/−23 ms。整条 TTFT 分别变化 **−367/+32/−123/+81 ms**。100 Gbps 反序的 P HTTP 引擎外增 147 ms，200 Gbps 反序的 P 引擎外增 91 ms、D 进入传输前增 55 ms。每臂 10 条的 p99 是该臂最慢值。 [客户端到两侧引擎的可加分解](results.md#逐轮阶段复盘)
- **最后排空 `winning-avg-corewars:131`。** r20 八臂均由它最后完成。首 token 后还要 17.04–17.43 s 输出，第二慢的 `raman-fitting:28` 同样输出 1,024 token，只早结束 0.35–1.34 s。四对第一条的首 token 分别提前 262/263/185/900 ms，首 token 后输出又多用 86/84/76/306 ms，整轮实际排空只早 176/179/109/594 ms。最后一对还在异步操作前形成了不同批次，批构成会影响首步和输出。 [逐请求输出与批形](../results/2026-11-19-r20-packed-overlap/stage-audit.json)

## 调度、准备与跨批流水

**首选验证跨批 GET 与上一批 prefill 的重叠。** r20 同步打包四臂中，`video-processing:28` 是同一个 3,941-token 后续批里唯一的新恢复，四 rank 合计 1.407 GB。下表以新批派发为零。旧 GET 完成到新 worker 进入相差 **461–470 ms**，两套 P 实例在此区间没有其他正式 GET 事件。前一批单 rank 最长的 host forward 观测窗口为 **147–165 ms**，可与该请求实际四 rank GET 宽度 **100 Gbps 约 142–149 ms，200 Gbps 约 70–75 ms**比较。主机窗口只能筛选机会。当前没有同时证明新页可写、传输额度可用及 NPU 上实际重叠的证据。 [可重算时间戳](../results/2026-11-19-r20-packed-overlap/restore-worker-costs.json)

| r20 同步对照 | 最后旧 GET 返回 | 最晚旧 host forward 启动 | 首个新 worker 进入 | 首笔新 GET | 最后一份新本机补齐返回 |
|---|---:|---:|---:|---:|---:|
| 100 Gbps，第 2 对，ms | 1,635.94 | 1,680.36 | 2,097.22 | 2,127.88 | 2,311.62 |

`regex-chess:106` 的两个 2,048-token 连续 chunk 分别和前批、后批共执行，后批之后还有最后 495 token。提前 `video-processing:28` 的读可能推进最慢 TTFT 请求的后继批。排空请求 `winning-avg-corewars:131` 的唯一 P chunk 在前批，提前后批 GET 没有直接的排空收益路径，还需检查是否反过来拖慢前批。即使整笔 GET 被覆盖，这个固定服务样本的直接时间收益也在 GET 的约 0.07–0.15 s 量级。

**调度与准备有各自的触发条件。** r19/r20 各有 88 个正式 P 批，31/29 个无自身 GET 的批在等 worker 时遇到旧 GET。这些批都混有旧结果尚未交付的请求，对应两轮各 79 个后续 chunk，观察期内没有一个已收到其前序结果且还在旧 GET 内等待。批次重排要先找到真的独立、已就绪且能被放行的计算。P 同步打包的描述符和选读计划常为数毫秒，明显小于上述请求 worker 前的 1.4–2.1 s。大型四恢复批可以花约 60–72 ms 准备，值得作为单独对照。页分配、清零、物理地址可绑定等更早的生命周期事件目前没有时间戳，仍可能是提前送出的有效约束。 [跨批审计](../results/2026-11-19-r20-packed-overlap/cross-batch-audit.json) · [源代码的 RPC 取件](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/executor/multiproc_executor.py#L892-L916)

下次实机先采 RPC 入队和取件、源对象版本或 pin、生效的目标页归属和清零、描述符可用、后端提交及端口额度、GET 完成、同头补齐、页发布、前向和批结果。重点核对待写页与 `new_block_ids_to_zero` 的交集和设备清零完成时刻。固定版 Ascend runner 引用上游的清零路径，需在运行时确认它覆盖哪些实际目标。以这些事件的最晚合法时点作为最早可提交时刻，同时保留实际原批次的计算和发布依赖。 [调度输出与清零列表](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L851-L889) · [Ascend 状态更新](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/worker/model_runner_v1.py#L681-L695) · [上游清零](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/worker/gpu_model_runner.py#L1071-L1074)

1. 同步打包是这次时序实验的优化对照。准备对照只提前构造可复用的描述符，原时点才提交 GET。两组在真正提交前都获取当时有效的目标地址。
2. 若一笔读已具备合法目标、旧批 GET 已返回、相关端口有额度，在相同负载与字节下提前提交这笔读，计算与发布仍由原批次依赖控制。在 100/200 Gbps 分别比较 GET 提前量、后续批结果、最慢 TTFT、前批前向和 D TPOT。若需要 staging，单独计其 HBM 占用和额外 copy，以免把重叠收益转为本机成本。
3. 若耗时集中在准备对照，继续拆页初始化与描述符。若只在提前提交后改善，继续验证 connector 与 executor 的串行耦合。若端口已经忙或早发拖慢前批，回到可执行路径的完成时刻，测等待读取、异步读取与重算的交叉点。重算应使用相同前缀的实际恢复起点和额外 prefill 批次测量，计入它挤占的 P 计算。r14–r20 尚无同形状重算对照，不能从 GET 字节直接推断选择阈值。 [重算对照与建模](bottleneck-and-validation.md) · [七步法](research-seven-steps.md)

## 多恢复批选读的扩展重放

重放覆盖 Terminal-Bench 2.1 的 r18 原始 30 会话、170 请求，以及同一数据集高恢复组的 r19、r20 和 r21 同步 trace。r21 正式组包含同步 A 和未触发干预的臂，排除四 rank 实际早发的 `c-b3-cap100`。保留原批次、每 rank 字节、GET 发布顺序和同头 rendezvous。每次只早发一笔，比较最早可行、最长可行和让后继四 rank 最晚 host forward 入口提前最多的选择。主要口径使用观测 GET 时长、4 ms 准备、每 rank 384 MiB 有界暂存和 8.56 ms 回填。窗口扣除两 P 实例已经观测到的 GET。字节上限按有效载荷筛选，实际页租约、对齐填充和 D 源端口分布仍需通过运行时验证。

| 同步 trace | P 批 | 多恢复批 | 后继 worker 前至少两笔可选 | 最早原始 GET 前至少两笔可选 |
|---|---:|---:|---:|---:|
| r18 原长会话，原带宽配置 | 875 | 1 | 1 | 1 |
| r19，100/200 Gbps | 44 | 10 | 2 | 3 |
| r20，100/200 Gbps | 44 | 7 | 0 | 1 |
| r21 正式与补轮，100/200 Gbps | 77 | 18 | 6 | 6 |
| r21 较早独立同步记录，100/200 Gbps | 22 | 4 | 0 | 0 |
| 合计 | 1,062 | 40 | 9 | 11 |

9 批中有 3 批在前批四 rank 共享 host forward 窗口内也有双候选。后继 worker 前的全部 9 批里，两种选择没有增量差异。r18 唯一多恢复批在原配置下有两笔可选，两种策略都选择 `fix-ocaml-gc:31`，相对同步局部提前 1.53 ms。

r21 的 `count-dataset-tokens:45 → video-processing:28 → financial-document-processor:60` 三恢复批在同一回放负载下重复出现。补轮 100 和 200 Gbps 样本各有约 399 ms 后继 worker 前空档，三笔均通过时间及载荷筛选。100 Gbps 样本 D 八口各有至少 274.8 ms 计数覆盖，窗口内计数增量为零，两策略均选 `video-processing:28`，模型相对同步提前 161.00 ms。200 Gbps 样本各有至少 293.8 ms 覆盖，八口总 TX 约 1.26 MB，两策略均选 `count-dataset-tokens:45`，模型提前 73.99 ms。端口计数按两端各留 42 ms 后取完整区间。这些局部提前量支持流水的潜在收益，两种选择的增量均为 0 ms。

放宽到后继首笔原始 GET 前，在 r19 `b-b1-cap100` 的 `6eee9c5a…` 出现一次差异。最长选 `train-fasttext:69`，观测四 rank GET 跨度 156.04 ms。关键路径选 `count-dataset-tokens:45`，跨度 155.95 ms。前者的跨度包含非关键组约 29 ms 的更早发起点。最后到达的是 rank 1，关键同头组上的两笔 GET 分别约 126.68 和 141.28 ms。模型中两选项相对同步分别提前 118.12 和 132.72 ms，差 **14.60 ms**。后继 worker 前原空档仅 102.23 ms，扣准备后约 98.23 ms，低于两笔约 1.43 GB 有效载荷的 100 Gbps 总速率下界 114 ms。放宽窗也只剩约 5.4 ms 完成余量。同档第二轮关键 rank 变为 3，Count 的观测跨度约 166.19 ms，超出 163.91 ms 的放宽净空档，失去第二个候选。将服务时间乐观缩短 25% 时，r21 的 200 Gbps 样本只在约 70 ms 共享 host 窗口内出现一次 1.87 ms 额外收益，该窗口无计数覆盖。

重放固定前批进度，前批延迟若一对一传给后批，14.60 ms 额外前批延迟就会抵消唯一的名义选读收益。r21 已运行的单笔早发对照曾观察到前批 rank 0 设备时长多 5.35 ms、派发到结果多 35.44 ms，这两个观测口径提供量级参照，无法估计新选择的扰动。本轮止于离线，未得到选择策略的实机 TTFT、吞吐或前批影响。[逐批重放](../results/2026-11-19-prefetch-holdout/choice-screen.json) · [r20 重放](../results/2026-11-19-r20-packed-overlap/cross-batch-choice-audit.json)

## P 侧请求接纳等待

现有 `p_chunk_arrival` 记录进入 `Scheduler.add_request`，`p_chunk_ready` 记录进入 `Scheduler.update_from_output`。先用 D 单调时钟算 P HTTP 往返时长，再扣掉 P 同机的调度等待和首个派发到末块结果的时长，得到调度器区间之外的残差。用两端 Unix 锚点把该残差估计分为代理发起到接纳、末块结果到 HTTP 回复。前批结果开始更新到接纳的间隔直接用 P 同机单调时钟。

| 请求集 | 估计接纳等待超过 300 ms | 其中在同实例前批结果开始更新后 10 ms 内接纳 | 结果更新到接纳中位数 | 同实例 GET 覆盖接纳等待的中位比例 |
|---|---:|---:|---:|---:|
| r20 四对 A/B | 54/80 | 54/54 | 0.512 ms | 45.4% |
| r21 正式六臂 | 44/60 | 44/44 | 0.533 ms | 43.5% |

对长等待的两端各裁掉 42 ms 后，与同实例批从派发到结果更新区间的重叠比例中位数仍为 100%。`winning-avg-corewars:131` 在 100 Gbps 同步 A 和未触发的 B 间，估计接纳等待由 869.44 降到 147.39 ms，前批 GET 重叠由 410.29 降到 0 ms。200 Gbps 同步 A 和未触发的 C 间分别为 651.72 到 143.58 ms、191.92 到 0 ms。两例的慢臂都在前批结果开始更新后约 0.3 ms 接纳该请求。前批 GET 的变化会和组批、其他完成事件共同改变等待，140 条请求的时间对齐给出应查的边界。

这批旧 trace 的近邻时间关系用于定位采样边界。r24 用 P 同机时钟实测队列入取与前批结果，再独立对照提前接纳。[逐请求接纳审计](../results/2026-11-19-prefetch-holdout/admission-audit.json)

## r24 同机时钟确认接纳阻塞

沿用 r21 计时负载。原数据集的 18 个前驱先按原会话关系完成，再按 50 ms 间隔发相同 10 个后继。每臂在新缓存命名域准备，逐请求路由、输入与输出预算一致，计时输出合计 3,971 token。P 仍启用 APC、4096 分块和同步同头打包。A 保持原输入队列取件。对每条请求用同一个 P 单调时钟串起 HTTP 请求体、API render、IPC ADD 发出至反序列化、入队与取出、调度器接纳、派发和 rank 0 实际进入 worker。前批的 future 结果等待和 scheduler 更新在同一 P 上配对，D 的客户端和限速独立计时。[负载](workloads.md#r19-高远端恢复的自然后继) · [采样方法](deployment.md#r24-的-p-请求接纳实验)

| 第一组 A 臂的逐请求阶段 | 100 Gbps | 200 Gbps |
|---|---:|---:|
| HTTP 进入到 ADD 入队中位数 | 33.28 ms | 38.26 ms |
| 其中 API render 中位数 | 0.303 ms | 0.302 ms |
| IPC 调用进入到接收端反序列化完成中位数 | 3.93 ms | 3.90 ms |
| ADD 在输入队列的中位数 | 520.25 ms | 326.31 ms |
| 取出到 Scheduler.add_request 中位数 | 0.056 ms | 0.049 ms |
| 接纳到首块派发中位数 | 40.77 ms | 157.84 ms |
| 首块派发到 rank 0 进入 worker 中位数 | 758.80 ms | 83.35 ms |
| 证实在前批 future 等待时已入队，更新结束后才取出的请求 | 7/10 | 8/10 |

15 个确认样本的前批等待开始时，最老 future 已从 deque 取出，仍在回收结果，另外一个 future 在 deque，两个槽均占用。两档在结果更新结束到 ADD 取出的间隔中位数分别为 0.352/0.393 ms，最大均低于 1 ms。队列等待与 P 同实例 GET 区间重叠的比例中位数为 53.3%/30.9%。该比例表示实际同时发生的时间，前批还包含设备执行、RPC 回传和其他处理。

100 Gbps 的尾请求 `regex-chess:106` 在 P 接到 HTTP 后 88.28 ms 入队，前批 future 在 127.49–771.31 ms 等待，771.67 ms 完成结果更新，772.01 ms 取出 ADD，772.05 ms 接纳。首块在 812.76 ms 派发，rank 0 到 1,672.99 ms 才进入 worker。单请求清楚地暴露了输入队列与派发和 worker 入口三道不同的等待。

同为 100 Gbps，第一组 B 让该尾请求的输入排队少 652.63 ms，最早 worker 晚 110.95 ms，D 首步多 120.70 ms，最终 TTFT 多 166.20 ms。反序 B 的同形首批晚 316.91 ms 入 worker，GET 包络短 290.88 ms，最晚 host forward 仅晚 8.11 ms，D 首步多 71.94 ms，TTFT 多 105.37 ms。同形及同字节的批仍有明显的执行时序波动。

200 Gbps 第一对的尾请求也为 `regex-chess:106`。B 把其输入队列时间从 431.60 ms 减到 6.41 ms，HTTP 进入到首块派发缩短 51.71 ms，首块最早 worker 进入却晚 2.16 ms。此后首批 11.247 GB GET 包络缩短 89.67 ms，最晚 rank 的 host forward 提前 87.51 ms。同一 D 时钟上的 P HTTP 和 D 首步各缩短 61.22/63.34 ms，解释了该请求 TTFT 缩短 122.92 ms 的主要部分。

反序 200 Gbps 的 `regex-chess:106` 首批形状变化。最早 worker 比 A 晚 252.18 ms，P HTTP 多 37.55 ms，D 首步少 95.64 ms，TTFT 仍少 53.49 ms。两次实现的 p99 都下降，首块更早开始执行的机制在这两个尾请求上均未出现。

第一对 200 Gbps 的排空请求 `winning-avg-corewars:131` 与该尾请求共首批。首 token 提前 172.61 ms，随后 D 输出多用 216.73 ms，整轮排空多用 44.12 ms。前端接纳、首步和排空在相同负载下沿不同的关键路径。

前批分阶段对比中，100 Gbps 反序的 P0 `extract-moves-from-video:133` 同形前批变化最大。GET 字节保持 2,769,715,200 B，派发到最早 worker 从 28.10 变为 27.87 ms，worker 到最晚 host forward 从 486.80 变为 533.18 ms，最晚 host forward 到结果更新从 181.96 变为 456.17 ms。GET 包络由 352.05 变为 396.97 ms，rank 0 设备前向由 168.73 变为 210.46 ms。主要变化出现在最晚 host forward 之后。该窗口仍包含各 rank 的设备执行、收取结果及其他收尾，现有事件能够定位区间。[逐请求及逐前批数据](../results/2026-11-19-r24-prefill-admission/summary.json) · [四对端到端结果](results.md#r24-p-接纳与首块实际启动)

## r27c 原生路径的带宽与排空归因

r27c 两档均为各 rank 同步直读的原生路径。每档在同一到达计划和路由下完成 264 条请求及 718,872 个输出 token。以下时间以各档第一条计划投放为零点。端口和 D 阶段用 D 主机的单调时钟，P 方法耗时用 P 主机的单调时钟。跨机事件的先后对齐用归档的时钟锚点，时钟误差影响毫秒级差值。[负载与配置](results.md#r27c-原生持续高压的输出平台) · [P 事件](../results/2026-11-19-r27c-native-sustained/r27c-P.jsonl.gz) · [D 事件](../results/2026-11-19-r27c-native-sustained/r27c-D.jsonl.gz)

### 同进度对齐的带宽压力

| 指标 | 100 Gbps | 200 Gbps |
|---|---:|---:|
| D 驱动读回（每口） | 12.5 Gbps | 25.0 Gbps |
| 0–10 秒 D 八口 MAC TX 均值 | 75.90 Gbps | 130.13 Gbps |
| 10–18 秒 D 八口 MAC TX 均值 | 78.69 Gbps | 154.64 Gbps |
| 发送首 400 GB 的时刻 | 39.36 s | 22.03 s |
| P 同步 GET 活跃并集 | 78.39 s | 40.28 s |
| 最慢单次 GET 的逐请求 p50 | 389.60 ms | 188.41 ms |
| 0–18 秒已完成的 P 返回 | 57 条 | 101 条 |

首 400 GB 的时刻由各端口累计 TX 字节按该口采样时刻插值，求八口增量之和首次达到 400 GB 的时刻。窗口均值也按各口时间轴积分。修正前把各口估出的速率放进了采样组的时间区间，轮询耗时变化时会使窗口均值偏低。GET 配对剔除一条字节不同的请求后，263 条的最慢调用配对中位数减少 226.18 ms。D 全量端口快照中，两档 RDMA 重试和发出 CNP 增量均为零，接收 PFC 为 0/7 帧，发出 PFC 为 436,768/242,684 帧。证据定位到硬限速下 D→P 同步恢复变慢。P 侧和交换机的拥塞计数时序仍缺采样。[修正后的窗口与逐请求](../results/2026-11-19-r27c-native-sustained/summary.json) · [限速与计数](../results/2026-11-19-r27c-native-sustained/r27c-D.jsonl.gz)

到达时的远端 GET 需求按整批平均为 190.6 Gbps，足以给 100 Gbps 档施压。每输出 token 只需约 1.09 MB 远端净载荷，按双 D 的输出平台约 1,154 token/s 折算，相当于两侧每秒消费约 10.1 Gbps 的恢复载荷。这个量级只用来判断工作比，瞬时就绪和热点端口由事件决定。作为对照，r14 每条只输出 16 token，远端载荷约 595 MB/输出 token，因此远端读取的提速可以持续传到整轮吞吐。[r14 原生对照](results.md#r14-远端恢复限速)

### P 页持有与 D 供给

高档 34–48 秒同时出现 101 条 P 在途、零 P 返回、零 D KV 就绪和零 MAC TX。P0、P1 各对同一待派发请求成功匹配外部前缀 649 次，零非空批次派发，也没有任何同步 GET 区间穿过该窗口。D0、D1 的 16 个运行位置全部占用，输出仍有 1,158.4 token/s。这段 P 等待发生在发 GET 前。

P 原始日志的 572 条 `Delaying free` 与准备及正式的 572 条 P 返回请求逐条匹配。用 P 已返回且 D 最后一 rank 尚未完成接收筛选。200 Gbps 档 34、41、48 秒三次快照均为 P0 的 31 条、11,724 个交接块和 P1 的 52 条、11,616 个交接块。对应的输入规模约 299 万和 294 万 token，量级接近两实例启动日志各自的 309 万 token KV 容量。输入长度与交接块数是持有压力的观测，精确空闲块数仍需 allocator 事件。约 67.30 秒，两个 P 各有一条此类请求在 D1 就绪，随后约 67.38 秒恢复非空批次派发。[清洗事件](../results/2026-11-19-r27c-native-sustained/r27c-P.jsonl.gz)

原生 `request_finished()` 会保留 P 上尚需交接的块，scheduler 收到 `finished_sending` 才释放。等待队列在匹配外部前缀之后还要通过 `allocate_slots()`，成功后才能派发。日志、持有量和恢复时序支持 D 接纳较慢引起的 P 页持有回压。现有观测记录了未派发和恢复时点，具体阻断分支及各缓存组的空闲块仍待记录。[Mooncake 固定版本的延迟释放](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1882-L1928) · [vLLM 固定版本的释放](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L2221-L2248)

从 18 秒到最后 P 返回，针对仍有 P 工作的路由，D0 已交付未结束的请求少于 16 的时间为 69.16→24.16 秒。两档这些区间的 D 八口发送均没有达到各自上限的 85%，平均仅 1.31→3.02 Gbps。D1 的该项缺口两档均为零。两路各 132 条，但 D1 负责 422,496 个输出 token，占全轮 58.77%。D0 最后一条在 575.53→559.69 秒结束，D1 在 810.62→809.26 秒结束。D0 的供给改善发生在全轮结束前已排空的一路。[逐请求和解码槽采样](../results/2026-11-19-r27c-native-sustained/summary.json)

### 决定整轮吞吐的尾请求

两档最后完成的都是 P0→D1 的 `rep12-natural-mailman:23`，原输出 4,811 token。客户端测得的结束时刻只提前 1.36 秒。下表用秒列出代理和 D 方法边界。

| 阶段 | 100 Gbps | 200 Gbps | 变化 |
|---|---:|---:|---:|
| P 开始到 P 返回 | 484.545 | 445.193 | −39.352 |
| P 返回到 D 入口 | 0.012 | 0.021 | +0.010 |
| D 入口到首 rank 交接入队 | 111.881 | 223.020 | +111.139 |
| 首 rank 入队到最后 rank KV 就绪 | 0.810 | 0.030 | −0.779 |
| KV 就绪到客户端首 token | 77.841 | 0.171 | −77.670 |
| 首 token 到请求结束 | 103.226 | 108.515 | +5.289 |

尾请求的 P 首块排队为 478.01→444.45 秒，自身最慢 GET 为 190.27→95.98 ms。P 侧获得的大部分提前量出现在首块派发前。D 入队前等待期间，D1 有效采样中运行数一直是 16，完成了其他 21→36 条请求。四个 rank 的 `enqueue→handler_begin` 为 755–779→0.87–1.26 ms，原生 `submit→handle_end` 为 17.19–20.29→17.76–22.08 ms，每 rank 发送载荷均为 372,613,120 B。两档真正提交后的耗时接近。P 提前交付后，D 入口到首 token 的净等待增加约 32.69 秒，生成增加约 5.29 秒，最终整轮吞吐仅从 886.81 增至 888.31 token/s。本轮是一对带宽干预，D 队列顺序和输出时间仍有运行间变动。[逐请求与 rank 方法事件](../results/2026-11-19-r27c-native-sustained/r27c-D.jsonl.gz) · [汇总](../results/2026-11-19-r27c-native-sustained/summary.json)

50 Gbps 八口均分需每口 6.25 Gbps。实机 `hccn_tool -shaping` 帮助和当前控制器均给出每口 10 Gbps 的下限，现有同拓扑均分硬限速最低 80 Gbps。复核时两机 16 卡被其他任务使用，更低带宽尚无实测。[控制器范围](../pd_kv_transfer_opt/experiments/hardware.py)

## r28 原生长会话的完工路径

从清洗归档重算四轮 680 条成功请求。客户端和代理均在 D 主机，用同一单调时钟连接请求就绪、发 P、P HTTP 回复、发 D、客户端首 token 和完成。逐请求流式 token 数与 API usage 全部相等。用原会话前驱和工具间隔重放，全部请求的就绪和完成时刻最大误差小于 1 微秒。r28 保留了这些边界和端口计数，具体 GET、页分配、D 交接入队和 NPU kernel 使用 r14、r26、r27c 的独立方法事件核对。[可重算结果](../results/2026-11-19-r28-natural-bandwidth/closed-session-profile.json)

### 请求与端口同时序

| 观测 | 正序 100 | 正序 200 | 反序 200 | 反序 100 |
|---|---:|---:|---:|---:|
| 30 条根请求的 P HTTP p50（秒） | 46.39 | 40.64 | 38.29 | 40.66 |
| 140 条后继的 P HTTP p50（秒） | 0.85 | 0.71 | 0.67 | 0.78 |
| 最后一条 P 回复（秒） | 382.21 | 394.83 | 364.17 | 331.34 |
| 最后一条 P 回复到整轮结束（秒） | 92.09 | 79.59 | 99.51 | 216.11 |
| D0 / D1 全轮平均运行请求 | 3.26 / 3.46 | 4.35 / 4.03 | 3.63 / 3.69 | 2.92 / 2.70 |
| D0 / D1 引擎指标采样覆盖（秒） | 465.54 / 463.90 | 474.42 / 474.42 | 463.67 / 463.67 | 547.45 / 547.45 |

会话起点集中在 4.56 秒内，之后每个会话等前驱完整结束才发下一条。30 个会话最多同时有 30 个未完成请求，D 两实例总计有 32 个运行槽。在 0–30 秒，四轮均已观测到 P 排队，D 八口 MAC TX 在 0–10 与 10–30 秒窗口的均值均低于 0.013 Gbps。根请求的 P HTTP 明显长于后继。该区间可定位到 P 内部或入口的耗时，进一步区分冷 prefill、派发等待与 GET 需要 P 同机方法事件。四轮 P 和 D 的 KV usage 采样峰值分别不超过 17.7% 和 38.9%，与 r27c 的高占用交接积压属于不同运行点。

100 Gbps 两轮八口一秒峰值为 **67.13 和 68.29 Gbps**。聚合 TX 达到档位的 85% 约 **0.45 和 0.40 秒**。逐口也做了复查，至少一口达到其上限 85% 的区间并集为 **2.11 和 4.04 秒**，与 D 无运行无排队且有最终将交给该 D 的 P 工作的采样区间，仅在反序 100 的 D0 相交约 **0.405 秒**。该轮最后完工在 D1。按约 0.4 秒端口采样和约 1.6 秒引擎采样解释这些交集，它们可定位供给证据，精确的 GET 等待、端口排队和 D 页分配仍需方法事件。[逐端口与引擎数据](../results/2026-11-19-r28-natural-bandwidth/closed-session-profile.json)

### 为什么整轮吞吐差约两成

设整轮输出为 $N$，含排空时长为 $T$，则 $\log(X_{200}/X_{100})=\log(N_{200}/N_{100})+\log(T_{100}/T_{200})$。正序两项为 **0.25674 与 −0.00025**，反序为 **0.06434 与 0.16608**。反序约 72% 的对数比来自时长项。正序实际生成多 43,369 token，反序多 10,251 token。配对后 P 和 D 去向都相同的请求分别有 47 和 40 条，其中仍有 42 和 38 条输出长度不同。两个 100 档中，反序的尾请求在正序 100 档生成 3,600 token 后就已停止。归档记录了 token 数与结束原因，输出序列分叉的时点及原因尚无观测。上游把 batch invariance 作为独立的数值可复现性配置，是否适用于实验的固定版本与 Hybrid 2P2D 需单独校验。[vLLM-Ascend 官方说明](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/batch_invariance.html)

进一步逐个替换 200 档的实测驻留时间，在会话的前驱关系上传递到完成并重新找最后会话。下表按所有阶段替换顺序取对称平均，正数表示该阶段使观察到的 $T_{100}-T_{200}$ 增大。阶段替换冻结实测驻留时间，包含当轮既有的并发、路由和排队，量化的是这组观测时间的分解。它不建立调度或带宽的因果效果。

| 匹配阶段对完工差的贡献（秒） | 正序 | 反序 |
|---|---:|---:|
| P HTTP | +11.19 | +2.30 |
| D 入站到首 token | +0.79 | −0.26 |
| 首 token 到完成 | −12.09 | +81.74 |
| 其余客户端和代理边界 | −0.004 | −0.003 |
| 含排空时长差，100 减 200 | −0.12 | +83.77 |

- **正序为同一条完工会话链。** `crack-7z-hash:31–36` 共六轮。第一轮 P HTTP 为 56.86→45.75 秒，全链 P HTTP 60.93→49.75 秒。全链首 token 后时长 407.84→419.93 秒，抵消了 P 的提前。最后一轮均生成 8,192 token，P HTTP 均约 0.54 秒。
- **反序发生完工会话切换。** 100 档由 `gpt2-codegolf:26–31` 结束，全链 P HTTP 为 67.31 秒，首 token 后累计 476.00 秒。最后一轮在 218.18 秒收到首 token，随后生成 24,098 token，到 547.45 秒触及预算。其余 169 条在 445.04 秒前已完工。同一请求在 200 档由 `stop` 结束，共 3,431 token，首 token 后 57.68 秒。其 D 入口到首 token 只差约 15.5 ms，按首 token 到结束除以后续 token 数衡量的有效 TPOT 是 13.66→16.82 ms。100 档此请求每 token 用时更短，同时有约七倍的实际生成量。
- **只替换 P HTTP 时长的敏感性。** 把反序 200 档逐请求实测 P HTTP 移到 100 档，固定其余成本并传播会话依赖，排空从 547.45 变为 546.47 秒。将反序 100 档的所有 P HTTP 时长理想归零，排空仍为 480.13 秒。只替换首 token 后时长，排空为 466.98 秒且切换到 `crack-7z-hash` 会话。三项属于阶段敏感性，实际改动可能改变并批、资源占用和 EOS。若固定 100 档末请求的工作量、首 token 时间和其他请求，仅缩短这条生成需将 TPOT 从 13.66 降至约 10.19 ms 才能达到 200 档的整轮 463.67 秒。[逐请求与会话链](../results/2026-11-19-r28-natural-bandwidth/closed-session-profile.json)

### 实测方法与工程改动能解释到哪一层

| 对照 | 实测响应 | 对整轮输出的含义 |
|---|---|---|
| **r14 至 r27c，带宽作用于同步 GET。** | 从 100 档的最慢 rank GET 以 $S/b$ 近似预报 200 档，r14、r26 两对、r27c 的预测分别比实测高约 3.1%、4.9%、4.7%、3.4%。r14 的独立 CANN 采样中，P0 rank 0 的 `hccl_batchGet` 覆盖 GET 99.73%，同期设备计算重合约 0.022 ms。 | r14 高恢复、每条 16 token 的整轮吞吐提高 96.5%。r26 长输出两对提高 2.06%/2.22%，r27c 提高 0.17%。相近的 GET 加速对应不同的 D 供给和排空响应。[r14](results.md#r14-远端恢复限速) · [r26](results.md#r26-原生-baseline-的高压带宽对照) · [r27c](results.md#r27c-原生持续高压的输出平台) |
| **r15/r16，同头单读与本机打包。** | r15 减少 49.66% 远端字节，直接分发在并发下产生约 3,800 次小广播。r16 维持相同远端字节，实测约 3,700–3,780→18–19 次，分发 p50 约 414–419→138 ms。 | r15 相对原生两档 +53.27%/+13.84%。r16 相对直接分发 +15.96%/+33.87%。适用域是预热、高恢复、固定短输出，原档 r17 的整轮收益跨配对方向不同。[r15/r16](results.md#r16-同头-kv-的打包分发) · [r17](results.md#r17-固定恢复路径短筛选) |
| **r19，开启 P APC 的同头单读和同步打包。** | 高恢复子组少传 49.23%，P0 同步 load 净减 1.48/0.53 秒。 | 相对原生整轮 +5.71%/+3.65%。目标是筛出的十条，保持相同的 3,971 个输出 token。[r19](results.md#r19-apc-开启的高远端恢复-ab) |
| **r20/r21，流水。** | r20 同批设备重叠约 278–316 ms，P0 同步 load 净减约 99–180 ms。r21 有效的 100 档早发使后继真实门槛早 178 ms，前批结果慢 35 ms。 | r20 整轮 +0.53% 到 +2.98%，最大值还包含介入前的批形变化。r21 实际激活的 100 档整轮约 +0.05%。200 档回落样本不可计入早发效果。[r20](results.md#r20-同批分发与下一笔-get-的异步重叠) · [r21](results.md#r21-固定字节的跨批提前读) |
| **r18/r24，调度代理指标。** | 固定 trace 时 r18 的批次就绪选择比通信完工只多 28.55 微秒。r24 早接纳使 ADD 少等 345–562 ms，真实 worker 启动没有稳定提前。 | r24 整轮 −0.21% 到 +0.43%。接纳、组批与可执行依赖需分开验证。[r18](results.md#r18-原长会话打包分发-trace) · [r24](results.md#r24-p-接纳与首块实际启动) |

r28 四臂全部是原生路径，以上改动的效果来自其各自的对照。将 r19 的提升外推到 r28，需要相同自然负载上的原生与优化配对以及实际激活记录。r28 数据支持针对生成结果的可比性和首波 P 阶段进一步取证，r27c 支持针对交接准入与 P 页释放找优化窗口。

### r26 与 r27c 的交接准入机会

重算 D 入口到第一 rank 交接入队。r26 四臂共 88 条，最大值均低于 0.28 秒。r27c 两档的 p50 为 **91.26/116.34 秒**，等待超过 5 秒的分别有 **208/264 与 196/264 条**。高档尾请求在 P 提前 39.35 秒时，D 实际交接入队反而发生在 700.55 秒，低档为 628.75 秒。早期增加的 P 供给伴随长时间的待交接驻留，D1 仍决定排空。两档同一请求的四 rank 原生传输提交到完成约 20 ms。[r27c 尾请求](#决定整轮吞吐的尾请求)

以每请求实际交接描述符 GB 乘 D 入口到交接入队的秒数，量化待交接负担。r27c 两档为约 **99,187 和 102,246 GB·s**。如果事后准确知道 D 何时可入队，逐条保留 5 秒提前量，落在这个余量以外的负担为约 **95,888 和 99,011 GB·s**。这是逻辑载荷按请求计重的代理量，可用来筛选窗口。物理页可共享，实际可消除的持有量与可提前释放的时刻必须由 allocator、页版本和完成通知验证。实际 D 入队时刻是受队列影响的结果，不能直接作为在线调度的输入或承诺收益。[准入统计和反向窗口](../results/2026-11-19-r28-natural-bandwidth/closed-session-profile.json) · [对应模型](problem-model.md#从-d-消费时点反推恢复)

## r33 原生 100 Gbps 从 P HTTP 到 D 排空

r28 两次 100 Gbps 和 r31 一次 100 Gbps 的最终会话中，P HTTP 累计 **60.93、67.31、55.98 秒**。其中冷首轮分别占 56.86、40.67、51.70 秒。r28 反序轮另有同会话第二条占 20.54 秒。旧 trace 有代理阶段、端口及物理页边界，缺这些请求的 IPC、计算批次及原生传输内层计时。r33 原样回放一次 100 Gbps，捕获所缺事件。请求、路由和停止条件可从归档重算，实际 EOS 仍随运行产生。[r28 会话链](../results/2026-11-19-r28-natural-bandwidth/closed-session-profile.json) · [r33 归档](../results/2026-11-20-r33-native-prefill-path/manifest.json)

实机两侧用 `--observe --profile-kv-lifetime`。P 另加 `--profile-prefill-admission` 和 `--adxl-observer`。ADXL 计时库在目标容器使用实际 CANN 9.1.0 的 `include` 构建，编译参数为 `-O2 -shared -fPIC -std=c++17`，链接 `-ldl`。加载前核对导出的 CANN 符号和软件指纹。计时库只包裹原始传输与建连调用。P 使用 rank 0 设备事件，D 保留目的页接收和客户端事件。[探针源码](../pd_kv_transfer_opt/profiling/adxl_observability.cpp) · [启动入口](../pd_kv_transfer_opt/experiments/serve.py)

### 接纳、派发和分块

客户端和代理在 D 同机。P API、IPC、scheduler 和 worker 在 P 同机。P HTTP 墙钟取 D 代理的发出至回包。同一请求在 P 的 HTTP 进入到保留交接页用 P 单调时钟拆分，代理侧及回程为两段持续时间之差。170/170 请求完整匹配。下表均用秒，rank 0 forward 位于派发后的一段。

| 请求 | P HTTP | P API 至接纳 | 接纳至首批 | 首批至保留 | 自身 rank 0 forward | 最慢 rank 自身 GET |
|---|---:|---:|---:|---:|---:|---:|
| `crack-7z-hash:31`，冷首轮 | 53.535 | 0.068 | **50.974** | 2.433 | 1.855 / 10 批 | 0 |
| `crack-7z-hash:33`，实际回读 | 0.568 | 0.0245 | 0.010 | 0.469 | 0.162 / 1 批 | 0.280 |
| `gpt2-codegolf:26`，另一次旧轮的尾会话首轮 | 45.361 | 1.152 | **39.365** | 4.743 | 3.626 / 18 批 | 0 |
| `gpt2-codegolf:27`，实际回读 | 19.173 | 0.091 | **17.701** | 1.196 | 0.504 / 2 批 | 0.300 |

- `crack-7z-hash:31` 的 API 处理 13.06 ms，ADD 提交到队列 4.31 ms，IPC 驻留 50.29 ms，取出到 scheduler 接纳 0.026 ms。IPC 驻留中 49.11 ms 处在前批 future 等待。真正长的是接纳后的队列。P1 在该段完整执行 185 个已有批次，全来自其他首轮，176 批用满 4,096 token，169 批仅含一个请求。已派发至结果的区间并集覆盖 50.974 秒，其中 rank 0 forward 时间为 41.55 秒。同期本实例无 GET。原生 scheduler 先给 running 请求分配 token，再用剩余预算处理 waiting，请求先后和整批预算共同决定首块。该实例的已覆盖物理分配检查零失败。[固定版 scheduler](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L378-L545) · [r33 事件](../results/2026-11-20-r33-native-prefill-path/r33-events.jsonl.gz)
- 旧轮曾决定排空的另一个会话在 r33 仍显示相同主因。`gpt2-codegolf:26` 接纳后的 39.36 秒包含 136 个前批和 31.69 秒 rank 0 forward，无同期 GET。`:27` 有 2.75 GB 自身回读，等待首批的 17.70 秒中有 14.58 秒前序 rank 0 forward，同期 Store GET 区间并集仅 0.230 秒。全轮 30 条根请求没有 Store GET，接纳后排队 p50 为 42.44 秒，派发到保留 p50 为 4.53 秒。140 条后继的相应值为 8.46 ms 和 0.515 秒。全部 912 个 P 批观察到 rank 0 PIECEWISE replay，其中 910 个有 forward 设备计时。[审计摘要](../results/2026-11-20-r33-native-prefill-path/summary.json)

### 同步 GET 的主机准备和传输服务

58 条后继触发 232 次原生 Store GET，全部能依据进程、调用边界和字节匹配到 9,904 次 ADXL READ，未观测到失败或字节差。每次读的墙钟中位 160.13 ms，p95 为 808.66 ms。worker load 内到提交的准备中位 1.47 ms，Store 入口到首个 ADXL READ 中位 0.92 ms，ADXL `TransferSync` 区间并集 158.50 ms，末次调用结束到 Store 返回 0.32 ms。各中位数独立统计。原生区间覆盖端点调度、设备传输和完成确认，单独的线上逐包服务仍未采集。MAC 计数用约 0.4 秒的窗口核对物理发送。全轮 GET 载荷 100.29 GB，D MAC TX 102.74 GB。D 一秒聚合峰值 55.63 Gbps，达到 85 Gbps 的时间约 0.40 秒，至少一个口达自身上限 85% 的区间并集约 0.81 秒。[字节与调用分布](../results/2026-11-20-r33-native-prefill-path/summary.json) · [ADXL 计时实现](../pd_kv_transfer_opt/profiling/adxl_observability.cpp)

P 的后台 Store 保存有 12,502 次 PUT、106.59 GB 的去往 D 主机池的 ADXL WRITE。它使已计算的状态可以作为后继 Store GET 的来源。该流量与 D 为本轮 decode 发起的 P→D 交接分别入账。`crack-7z-hash:31` 的 P HTTP 窗口有 3.97 秒同 P 实例后台 PUT 调用重合。这是同时发生的活动，因果图上要通过存储就绪、共享资源或引用依赖评估影响。当前观测的长等待由前序批执行覆盖。

### D 从就绪到最后输出

对 170 条请求，从 D 代理发出到 scheduler 入口、到实际分配、到完整 KV 就绪、到客户端首 token 的各段中位分别是 **70.20、1.44、81.30、168.91 ms**。计算供给时按目的实例计数 KV 已就绪且客户端尚未结束的请求，配合 D 运行/等待指标。该计数可用于确认完全缺供，KV 就绪后还需满足实际调度条件。

| 观测（从首条计划投放计） | D0 | D1 |
|---|---:|---:|
| 首次 / 最后一次 KV 就绪（秒） | 7.25 / 299.41 | 10.16 / 286.52 |
| 启动时有目的为该实例的 P HTTP 在途且尚无 KV 就绪（秒） | 7.04 | 9.94 |
| 首次就绪后相同意义的缺供（秒） | **0.49** | **2.15** |
| 最后完工时刻（秒） | **419.70** | 411.33 |
| 最后请求 / 实际输出（token） | `crack-7z-hash:36` / 8,192 | `extract-elf:16` / 10,479 |
| 最后请求首 token 后、同一 D 的其他请求均已结束（秒） | 120.16 | 122.95 |

`crack-7z-hash:31` 在 56.83 秒 KV 就绪，56.95 秒产出首 token，接着生成 181.68 秒才释放会话后继。整条会话链首 token 后累计 **356.89 秒，占整轮约 85%**。最后一轮自身 P HTTP 仅 0.494 秒，D 页分配到完整就绪 36.72 ms。最后一次 P 回复发生于 299.33 秒，剩余 120.37 秒由 D 产出。按客户端的秒桶，在 `t ≥ 300` 时仍有 16,290 个流式 token。P 提前交付能否缩短排空，需要核对其对 D 实际生成进度的影响。[客户端与 D 事件](../results/2026-11-20-r33-native-prefill-path/r33-events.jsonl.gz) · [逐段成本](../results/2026-11-20-r33-native-prefill-path/summary.json)

### 整轮敏感性

每一行独立从同一观测的路由、输出及阶段驻留出发，改变表中一项，在 30 个会话上重新递推前驱并找最后完工。rank 0 forward 的行采用同实例 rank 0 的设备计时，等待窗口内只取完整落入的前批。将它按比例从目标请求驻留扣减，假设这些节省可以传到相应的派发和完工。资源竞争、消费时间与走向在真实干预后可能变化。

| 独立改变 | 排空提前（秒） | 固定输出的整轮吞吐增幅 |
|---|---:|---:|
| API、ADD 和 IPC 至 scheduler 完全归零 | 0.178 | 0.04% |
| 每条 P HTTP 窗口中观测的 Store GET 占用完全归零 | 0.283 | 0.07% |
| D 分配到 KV 就绪完全归零 | 0.326 | 0.08% |
| 仅请求自身的 rank 0 forward 成本缩短 10% | 0.463 | 0.11% |
| 自身和等待中的前序 rank 0 forward 成本均缩短 10% | **4.618** | **1.11%** |
| 全部首 token 后驻留缩短 10% | **35.689** | **9.29%** |

在 P 阶段，实测指向冷首轮积压，以及整批预算被既有工作占用。优化 HTTP 接纳、同步 GET 提交或 D 的交接流水，从观测排空路径上只看到亚秒量级。整轮对 D 生成时间最敏感，两实例分别有长会话压尾。当前 D 主机事件和客户端输出只能定位到解码驻留，能改善的设备阶段及实际端到端增量需要相同 100 Gbps 的证据。这轮用于确定待验证瓶颈，论文 key idea 仍待定。[问题模型及适用范围](problem-model.md#瓶颈归因与模型校验)

## r34 原生 100 Gbps 的 D 步进与排空

同一份 Terminal-Bench 2.1 自然会话在 r33、r34 最后均由 `crack-7z-hash:31–36` 完工。P HTTP 累计为 57.88→56.94 秒，首 token 后累计为 356.89→422.83 秒。主要变化来自 `:35` 的实际输出 3,335→8,192 token，以及相应的首 token 后驻留 50.24→119.83 秒。两轮都是 100 Gbps 的观测，完整吞吐用于记录实际工作量和排空。r34 D 出站一秒峰值 87.94 Gbps，聚合达到 85 Gbps 的采样窗约 1.23 秒。[r33 归档](../results/2026-11-20-r33-native-prefill-path/summary.json) · [r34 归档](../results/2026-11-20-r34-native-decode-path/summary.json)

### Decode 步进成本

D 用 `--observe --profile-kv-lifetime --profile-decode-steps`。以批内 token 偏移每 16 步采样，同一 D 主机上连接 scheduler、四个 TP worker、NPU 事件和输出更新。共匹配 2,788 批和 11,152 份 worker 记录。标定服务时排除 CANN 启停窗口。D0 排空段内采样的步间隔中位在 CANN 期间为 16.90 ms，窗口外为 13.59 ms。两次早段 CANN 停止请求被分发到不同 API 进程并返回 HTTP 500，worker 仍停止并保存了原始捕获。D0 的前端补停成功，D1 前端的停止状态未能核实。后续启动配置的 D 步进模式只启用 worker profiler。所有统计保留 CANN 的窗口与覆盖宽度。[步进探针](../pd_kv_transfer_opt/profiling/decode_step_observability.py) · [触发方式](../pd_kv_transfer_opt/profiling/capture_decode_windows.py) · [采样边界](../results/2026-11-20-r34-native-decode-path/cann-summary.json)

| 单请求排空段的中位数 | D0 | D1 |
|---|---:|---:|
| Scheduler 计算 | 0.174 ms | 0.174 ms |
| Rank 0 worker 相邻步 | **13.59 ms** | **13.19 ms** |
| Rank 0 worker 入口到 host forward | 7.91 ms | 8.09 ms |
| 最慢 rank 设备 forward | **11.59 ms** | **11.67 ms** |
| Rank 0 forward 后同流事件间隔 | 0.282 ms | 0.280 ms |
| Rank 0 采样 host 调用 | 1.062 ms | 1.071 ms |
| EngineCore 结果更新 | 0.089 ms | 0.088 ms |
| 派发至结果更新 | 27.99 ms | 26.65 ms |

两实例采用两槽、异步 EngineCore 批队列。各段有重叠，稳定步频用相邻 worker 入口与实际 token 前进校验。D0 `:36` 有 508 个独跑样本，D1 `:35` 有 256 个。全部在四 rank 回放 FULL 图，pad 宽度均为 1。以终点间实际 token 偏移校准，设备 forward 估计占独跑驻留的 **80.9% / 86.5%**。仅用各自前半独跑样本的中位步间隔预测后半连续 token 的时间，偏差为 −5.15% / −0.91%。它验证步频锚点并量化本轮时间波动。[步频、形状与阶段包络](../results/2026-11-20-r34-native-decode-path/summary.json)

### Collective 等待的来源

D0 排空 CANN 捕获的 48 步，rank 0 的设备任务并集 592.07 ms。矩阵乘占 164.62 ms，其中 MatMulV2 97.86 ms、GroupedMatmul 66.54 ms。机内 collective 占 160.48 ms，其中 AllReduce 159.09 ms。全注意力占 38.78 ms。逐步重复 81 次 AllReduce 和一次 AllGather。AllReduce 本身中位 22.26 µs，p95 34.42 µs，前两次分别为 696.34 和 503.07 µs。Profiler 报的同步时间共 95.47 ms，前两次占 63.03 ms。该捕获记录了等待、同步及任务时间，AllReduce 字节没有可用的传输报告。D0 多请求时，矩阵乘与 collective 占设备任务时间 30.7% 和 17.1%。D1 多请求时为 24.5% 和 16.7%。[逐算子统计](../results/2026-11-20-r34-native-decode-path/cann-summary.json)

按 AllGather 结束对齐同一 D0 的四个 rank，共 48 步，锚点时差中位 0.018 ms。第一和第二次 AllReduce 的各 rank 到达时差中位分别 **1.012 / 0.555 ms**，完成时差分别 **0.018 / 0.017 ms**。从最晚 rank 到达到最晚完成的中位分别 **0.030 / 0.028 ms**。Rank 0 的早到量与 profiler 同步时长相关系数为 0.985 / 0.959。设备时间线上每步首个任务的起点只差 0.019 ms。第一笔前，各 rank 实测任务并集约 0.26 ms，墙钟为 3.99–4.86 ms。第一笔完成后进入第二笔前，四 rank 实测任务约 0.11–0.12 ms，rank 0/1 的未覆盖时间为 0.105/0.096 ms，rank 2/3 为 0.555/0.531 ms。这些间隔的 host 和设备事件在 r35 继续对齐。[四 rank 事件对齐](../results/2026-11-20-r34-native-decode-path/collective-alignment.json)

这轮最后 D1 请求 `:35` 的 EOS 或达到预算是 D0 尾请求 `:36` 释放的前提。条件重放将两条请求独跑段的设备 forward 分别缩短 10%，按实测步进占比转成驻留缩短，整轮提前 **14.37 秒**。整轮吞吐增幅为 **3.05%**，保持输出、路由和其余成本。当前判据是缩短最晚 rank 在对应 join 前的实际进度，逐段检查下一个设备步及后续会话是否推进。论文 key idea 保留到可复现的整轮增量之后。

## r35 D0 的 host 提交与图内输入路径

复用 r34 的 CANN 捕获，以前一步 logits AllGather 结束为锚，将 CANN enqueue、HCCL host API、设备实际进入 collective 串起来。图的 host `aclmdlRIExecuteAsync` 以 `connection_id` 关联实际 `MODEL_EXECUTE`。47 个带上一步锚点的完整四 rank 步均可对齐。下表用每一步对应 collective 的最晚和最早设备 rank 做差，再求中位。[逐步原始事件的重算](../results/2026-11-20-r35-decode-graph-input/r34-host-submission.json)

| 阶段 | 四 rank 的 r34 晚到差中位 |
|---|---:|
| 第一笔图外 embedding AllReduce 实际进入 | **0.985 ms** |
| host enqueue 时已形成的差 | **0.887 ms** |
| enqueue 至 HCCL host 提交 | 0.001 ms |
| HCCL host 提交至设备进入 | 0.024 ms |
| 第二笔图内 AllReduce 实际进入 | **0.549 ms** |
| host 图提交时已形成的差 | **0.541 ms** |
| 图提交至设备开始 | −0.008 ms |
| 设备图开始至第二笔进入 | 0.003 ms |

第一笔之前可见的前台阻塞同步差中位为零。第二笔的晚 rank 在第一笔完成后约 **0.623 ms** 才提交图，其中可见的前台阻塞同步中位 **0.018 ms**。早 rank 的设备流等待随第一笔归约结束解除。两笔的晚到主要随 host 墙钟到达移动，CANN 已记录的队列调度、host 同步 API 和图内前导任务只解释很小的 rank 差。host 入队前的具体 Python 方法和 OS 调度占比仍未拆分。这个短窗定位的是 D0 机内 TP 的依赖。[分解与边界](problem-model.md#图外输入准备与图提交)

### 图内输入改变了等待落点

r35 B 仅在 D 开启 vLLM 的 `--language-model-only`。每臂正式回放后另取同一纯文本长前缀，D0 单并发、同为生成 112 token 时请求启动 CANN。A 的第一笔是图外 embedding，B 的 156 个 rank-step 全在 graph start 后进入第一笔。A/B 的最后归约完成锚点中位为 **4.273/5.030 ms** 和 **5.100/5.170 ms**。每步末 logits AllGather 的完成间隔为 **16.674/16.630 ms**，相邻两步最晚图起点的间隔为 **16.630/16.569 ms**。采样同输入、生成深度与并发，384-token 输出序列的哈希不同。[设备对照明细](../results/2026-11-20-r35-decode-graph-input/summary.json) · [原生参数](https://docs.vllm.ai/en/latest/configuration/engine_args/#multimodalconfig)

B 第一笔的实际 rank 到达差为 **4.297 ms**。按同一对早晚设备 rank 分解，差在 host `aclmdlRIExecuteAsync` 时已约 **4.315 ms**，提交到设备图开始的差约 **−0.003 ms**，图开始到第一笔的差小于 **0.001 ms**。四 rank 在图内第一笔汇合后，第二笔到达差收敛到 **0.016 ms**。这解释了第二笔的对齐改善和整步时长几乎持平可以同时出现。可见的主流同步接口只贡献小量时差，精确的提交前 host 方法仍需更多细粒度观测。[B 四 rank 设备与 host 时序](../results/2026-11-20-r35-decode-graph-input/b-graph-steps.json)

### 含排空吞吐的归因

最晚会话从 A 的 `feal-linear-cryptanalysis` 切换为 B 的 `crack-7z-hash`。`crack-7z-hash:35` 的首 token 后驻留 **18.12 → 126.58 秒**，自然完成从 **256.23 → 373.15 秒**。两臂 `:36` 都生成 **8,192 token**，其首 token 后驻留 **127.28 → 111.10 秒**，局部快了 **16.18 秒**。前驱晚释放 **116.93 秒**，已超过这段局部节省。输出长度相同的请求为 **32/170**，P 和 D 路由都相同的请求为 **46/170**。整轮只用观测值评价，阶段替换重放用来追哪条会话把时长推后，并保留实际 EOS 和路由带来的交互。[两臂尾链逐请求](../results/2026-11-20-r35-decode-graph-input/summary.json)

纯文本切换副作用是 D 可分配 KV 从约 **30.55 GiB** 升到 **31.67–31.68 GiB**，两臂观测的最高 D cache 占用均低于 **39%**，累计抢占为零。D 所有端口的聚合出站一秒峰值为 **69.14/82.70 Gbps**，高于 85 Gbps 的总采样时长最多约 **0.001 秒**。当前负向结果的主要时长变化出现在尾链生成阶段。[启动容量](../results/2026-11-20-r35-decode-graph-input/manifest.json) · [资源采样](../results/2026-11-20-r35-decode-graph-input/summary.json)

### 算子融合的定位

r34 首笔之前可见多次 Cast、索引和掩码操作，slot mapping 已有 `_compute_slot_mapping_kernel`。可共享输入的图外小操作可以研究成批写入捕获图所需缓冲区，候选价值是减少最晚 rank 的准备和提交。第一笔与第二笔之间有 embedding 到第一层计算的数据依赖，各 rank 的已记录设备任务时间并集仅 **0.11–0.12 ms**。单纯压缩这部分设备计算，直接节省量受此区间限制。r35 的上游参数改动检验的是输入路径与等待落点。真正的融合若减少 host 提交或同步，还要分别量最晚 rank、下一步和整轮排空的变化。[r34 算子摘要](../results/2026-11-20-r34-native-decode-path/cann-summary.json)

## r37 原生尾请求的单步分解

以相邻两步 logits AllGather 的最后 rank 完成为边界，分别处理 r34 D0 原 CANN 的 47 个完整对齐步和 r37 的 43 步。两份捕获都是同一 45,900-token 输入的单请求 FULL 路径。每步有 81 次 AllReduce 和一次 AllGather。对每次 collective，先从前一笔最后完成切到本笔最晚到达，取最晚到达 rank 的 NPU kernel、设备拷贝、主流可见 `EVENT_WAIT` 的区间并集，再单列从最晚到达到最后完成的暴露时间。同 rank 的重叠只计一次，所有区间逐步闭合到实测周期。HCCL 结束段含通信实现和完成开销，通信报告的 `Transit Time` 在这一平台不提供可分离的真实线速传输时间。[原 CANN 分解](../results/2026-11-20-r37-decode-critical-step/r34-breakdown.json) · [轻量 CANN 分解](../results/2026-11-20-r37-decode-critical-step/r37-breakdown.json) · [CANN 事件字段](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0027.html)

两种捕获的步间隔中位为 **17.389 / 15.193 ms**。实测 NPU 算子覆盖均值 **9.011 / 8.724 ms**，到齐后的机内 collective 均值 **1.946 / 1.893 ms**，拷贝和主流可见事件等待约 **0.170 ms**。前两次归约之后的未覆盖间隙均值 **0.888 / 0.990 ms**。轻量短窗同一客户端的区间中位步频为开采集前 **13.205 ms**、采集稳定区间 **15.167 ms**、采集结束后 **13.927 ms**。用两份捕获的设备与后段成分分别校准 r34 原生 **13.591 ms**，得到前段间隙残余 **1.577 / 1.814 ms**。它包含 host 准备、提交和运行时空档，现有观测可校准该区间的合计，精确的逐方法拆分需要带设备锚点的原生采样。[耗时口径和原生校准](../results/2026-11-20-r37-decode-critical-step/sensitivity.json)

轻量捕获中，上一拍最后 AllGather 到本拍控制 rank 的上一拍 `argmax` 完成中位为 **0.041 ms**。从该完成到本拍第一笔 embedding 归约最晚到达约 **3.39 ms**。第一笔、第二笔在各 rank 都到齐后中位约 **22 / 22 µs** 完成。对应最晚 rank 在可见 host 提交前的未覆盖间隙约 **2.21 / 0.41 ms**，临界区内可见 host 同步调用的占用中位约 **0 / 0.016 ms**。这两笔的长等待主要来自最晚 rank 尚未到达。图一旦运行，后续 80 次归约与其前序算子把当前 token 推到 logits，输出经采样供下一个 token。两个采集档下，扣除前两笔到达前区间后的剩余串行周期均值为 **11.54 / 11.32 ms**。这段是最大持续成本，前段释放也是一个较小的真实门槛。计量保留设备任务和主机提交的交叠。[四 rank 逐步事件](../results/2026-11-20-r37-decode-critical-step/r37-breakdown.json)

r34 D0 单请求首尾实测偏移覆盖 **8,160 步**，用时 **117.52 秒**，对应 **14.40 ms/token**。13.59 ms 是非采集样本的局部中位数。条件回放以实测 484.680 秒为基准，用固定工作量和先后关系传播每步节省值。计算段缩短 10% 和 20% 时，逐 collective 的最晚 rank 可以变化。仅作用于 D0，整轮分别节省 **7.17 / 14.31 秒**，含排空吞吐分别增加 **1.50% / 3.04%**。若 D1 已覆盖的 4,080 步也达到同样每步改善，则为 **2.27% / 4.64%**，该项用 D0 结果外推。host 的比例灵敏度保留为采集内统计，原生回放用实际传到门槛的绝对节省，如 **0.5 ms/步 → 4.08 秒、+0.85%**。收益需要局部优化实际推进下一步，模型冻结工作量与其他阶段，保持实测路由和 EOS。[模型](problem-model.md#单步分段与排空敏感性) · [逐场景回放](../results/2026-11-20-r37-decode-critical-step/sensitivity.json)

## 重算

r37 的分解输入是同一捕获的 rank 0–3 下四个 `ASCEND_PROFILER_OUTPUT` 目录。冻结 r34 输出和会话依赖时，第二条命令使用已归档的分解结果。

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_decode_critical_step \
  RANK0_OUTPUT RANK1_OUTPUT RANK2_OUTPUT RANK3_OUTPUT runs/r37-breakdown.json
python -B -m pd_kv_transfer_opt.profiling.replay_decode_tail \
  results/2026-11-20-r34-native-decode-path \
  results/2026-11-20-r37-decode-critical-step/r37-breakdown.json runs/r37-sensitivity.json \
  --old-capture results/2026-11-20-r37-decode-critical-step/r34-breakdown.json
```

r35 的整轮与短窗从带校验和的归档复算。`analyze_decode_graph_input` 的四个输入是 rank 0–3 各自包含 `trace_view.json` 和 `communication.json` 的 `ASCEND_PROFILER_OUTPUT` 目录。

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_decode_graph_input \
  RANK0_OUTPUT RANK1_OUTPUT RANK2_OUTPUT RANK3_OUTPUT runs/d0-graph-steps.json
python -B -m pd_kv_transfer_opt.profiling.analyze_decode_graph_experiment \
  results/2026-11-20-r35-decode-graph-input runs/r35-summary.json
```

r34 的步进账本可由同一自然回放重算。

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_decode_steps \
  results/2026-11-20-r34-native-decode-path/r34-events.jsonl.gz \
  results/2026-11-20-r34-native-decode-path/manifest.json runs/r34-decode-steps.json
```

短窗从原生 CANN `kernel_details.csv` 和 `communication.json` 导出，依照目录中的 `profile-windows-D*.jsonl` 对齐。分别运行 [CANN 摘要](../pd_kv_transfer_opt/profiling/summarize_decode_cann.py) 和 [四 rank 算子对齐](../pd_kv_transfer_opt/profiling/analyze_decode_collective_alignment.py)。后者按 rank 0–3 顺序接收四个 `communication.json` 以及输出路径，从相邻 `kernel_details.csv` 复核设备前序。

r33 按清洗归档核验哈希，再独立复算每项敏感性、三个状态域和整轮数据。

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_native_prefill_path \
  results/2026-11-20-r33-native-prefill-path/r33-events.jsonl.gz \
  results/2026-11-20-r33-native-prefill-path/manifest.json runs/r33-prefill-path.json
```

r28 会话链、阶段替换与历史交接统计使用现有脱敏归档。

```bash
python -B -m pd_kv_transfer_opt.profiling.profile_closed_sessions \
  results/2026-11-19-r28-natural-bandwidth runs/r28-closed-session-profile.json \
  --handoff-summary results/2026-11-19-r26-native-pressure/summary.json \
  --handoff-summary results/2026-11-19-r27c-native-sustained/summary.json
```

```bash
python -B -m pd_kv_transfer_opt.profiling.audit_restore_worker_costs \
  results/2026-11-18-r19-natural-ab/r19-P.jsonl.gz \
  results/2026-11-18-r19-natural-ab/r19-D.jsonl.gz \
  results/2026-11-19-r20-packed-overlap/r20-P.jsonl.gz \
  results/2026-11-19-r20-packed-overlap/r20-D.jsonl.gz \
  runs/restore-worker-costs.json

python -B -m pd_kv_transfer_opt.profiling.audit_prefetch_holdout \
  results/2026-11-18-r18-natural-packed results/2026-11-18-r19-natural-ab \
  results/2026-11-19-r21-cross-batch runs/prefetch-holdout.json

python -B -m pd_kv_transfer_opt.profiling.audit_prefill_admission \
  results/2026-11-19-r20-packed-overlap results/2026-11-19-r21-cross-batch \
  runs/prefill-admission.json
```
