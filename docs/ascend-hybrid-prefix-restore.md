# 910B2 上按状态来源恢复 Hybrid 前缀

## 研究候选

**让换 P 的会话从同机上一轮 P 的 HBM 恢复可验证的 Attention 历史，D 侧 Store 经 RoCE 只补齐缺页及该恢复边界所需的 rank 私有状态。** 两种路径直接写本轮 P 已分配的最终页，沿用原生恢复与发布语义。这项改动针对 D→P 的实际跨机字节和 P worker 的恢复等待。P/D 实例选择由原生代理完成。主方案验证独立关闭同头打包。

两台机器的只读拓扑均显示八张 64 GiB 的 910B2 任意卡对为 HCCS，各卡对分属四组 CPU affinity。这个观测给出了可用的物理拓扑。运行中的 HCCL/ADXL 流选路、共享端点及实际并发速率须按对应方法确认。CANN 支持 Atlas A2 的跨进程设备内存导入与 peer access。原生 HIXL 在这两台机器已验证跨机注册设备缓冲的 READ。当前 D Store 的常驻副本在主机池，二者的源内存不同。Mooncake Fabric Memory 的远端主机直访面向 A3。 [CANN IPC](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/appdevgapi/aclcppdevg_03_1934.html) · [A2/A3 传输](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/transport/ascend_direct_transport.md) · [r32](results.md#r32-同字节的-hixl-批提交)

## 单独核对跨机字节与整轮完工

| 已有对照 | 100 Gbps 整轮输出吞吐 | 200 Gbps 整轮输出吞吐 | 观测到的限制 |
|---|---:|---:|---|
| r14 原生，高回读短输出 | 16.85 | 33.11 | 96 个请求各输出 16 token，跨机读 914.164 GB，带宽约束传到排空。 |
| r19 原生，APC 开启的高回读后继 | 174.19 | 188.16 | 10 条目标固定 3,971 token，准备的 18 条前驱单列；100 档 P0 同步 GET 并集 4.087 s，200 档 2.073 s。 |
| r19 同头打包 | 184.13 | 195.03 | 41.832→21.238 GB，P0 分发另占约 0.58 s。100 档达到原生 200 吞吐的 97.86%，达到同策略 200 的 94.41%。 |
| r26 高到达原生，固定原长输出 | 345.33 / 346.78 | 352.44 / 354.49 | 1–6 s 在 200 档出 token 快 62%–90%，D 得到足量工作后的长生成吸收大部分差异。 |
| r28 自然长会话原生 | 312.38 / 281.78 | 403.72 / 354.80 | 正序差主要是实际生成量，反序差主要是生成尾部。 |
| r39 固定工作量的原生归因 | 406.84 / 412.12 | 410.55 / 407.15 | 固定 164,662 token 和实际路由，GET 变快约 62–66 ms，整轮差异两次方向相反。 |

吞吐单位为 token/s，均含排空。r14 和 r19 的诊断边界由各自的计时工作量决定。r28 是部署 baseline，r39 是固定工作量的归因对照。 [r14–r39 吞吐与请求](results.md) · [r39 方法归因](bottleneck-verification.md#归因对照-r39)

在 r33，56 条换 P 后继贡献 97.055/100.287 GB 原生读取。逐个 GET 描述符核对时，按当前 P 的实际本地命中 $L$、外部命中 $E$ 和上轮 P 输入的逐 token 公共前缀 $K$，可检查潜在同机 Attention 对象。TP4 下每 rank 一个 1,024-token 的完整 Attention 页组为 10 MiB。换 P 的请求 $i$ 按下式求乐观上界，其中 $A_i$ 为原生读取中 Attention 部分的实际字节。

$$
S_i=\min\left(A_i,\ 4\cdot10\,{\rm MiB}\,[\lfloor\min(K_i,L_i+E_i)/1024\rfloor-\lfloor L_i/1024\rfloor]_+\right).
$$

| 记录 | 实际远端 GET | 理想留在 RoCE 上 | 可转移至同机的上限 |
|---|---:|---:|---:|
| r19 原生，四臂同载荷 | 41.832 GB | 0.644 GB | 98.46% |
| r33 100 | 100.287 GB | 6.838 GB | 93.18% |
| r39 100，正序 / 反序 | 97.539 / 103.369 GB | 6.774 / 6.774 GB | 93.05% / 93.45% |
| r39 200，正序 / 反序 | 104.901 / 103.746 GB | 6.838 / 6.774 GB | 93.48% / 93.47% |

输入覆盖给出可搬运字节的上限。源 P 从前驱持页到后继发 GET 的中位间隔在 r33 为 14.88 s，在 r39 的 100 档为 16.93 / 13.06 s。r19 的前驱统一准备，上一 P 回复到后继 P 入口的年龄中位为 95.47–141.21 s。已返回的 APC 页可以被复用为新物理页，必须观测原代次是否仍在以及能否取得引用。固定的 DeepSeek 会话输入按本次前驱完工释放，下一轮输入没有用本次 Qwen 输出重写。这些记录的换 P 读范围均落在上一 P 的 token 公共前缀内。D checkpoint 是合法的 Store 状态来源，记录并不表明它由本次 D 生成推进。 [输入前缀](../configs/workload-natural-r10.json) · [r33 方法事件](../results/2026-11-20-r33-native-prefill-path/manifest.json) · [r19](../results/2026-11-18-r19-natural-ab/manifest.json)

**原长会话的限制在可移动的时间。** 冻结实际工作量和其他驻留，r33 将观测到的所有 GET 同批阻塞消除，整轮仅提前 0.283 s（392.33→约 392.60 token/s）。r39 两个 100 档仅提前 0.094 / 0.080 s（406.84→约 406.93、412.12→约 412.21 token/s）。这两组 10% 吞吐需要提前约 36–38 s。在 r28 的自然记录中，即使把全部 P HTTP 驻留归零，两个 100 档也只到 358.43 / 321.28 token/s，低于同对 200 的 403.72 / 354.80。这里固定了各 100 档实际输出，说明这段观察差无法解释为恢复路径可以收回的吞吐。 [长会话模型](critical-path-attribution.md#整轮敏感性) · [固定工作量 100/200](bottleneck-verification.md#归因对照-r39)

## 硬件路径的收益条件

对本 rank 某次恢复，令 $C_R$ 为原生 RoCE 服务含排队到完成，$C_R'$ 为剩余缺页与状态的 RoCE 成本，$C_H$ 为使用目标真实页布局测得的 HCCS 读写成本，$\delta$ 为查询、pin、提交、完成通知和合并。两路径能并行时，低阶估计为 $\delta+\max(C_H,C_R')$。共享目标 HBM 或提交串行时，按实际有重叠的资源和事件重算。批次的消费者由真实最晚 rank 和后续 collective 约束。P 的提前用 D 就绪批宽、会话释放和最终完工传播。

设可移到 HCCS 的字节上限为 $S$，本次实际移动 $w\le S$。需要最小化两条路径的最后完成。以实测有效服务速率 $b_R,b_H$ 拟合一段线性成本时，估计 $\max(t_R^0-w/b_R,\ t_H^0+w/b_H)$，其中 $t_H^0$ 含 HCCS 路径的提交固定成本。等时点裁剪到合法区间并按整页取整，再检查是否实际推进批次和供者。10 GB/s/rank 情景下，每笔只移动 75% 候选字节，100 / 200 为 190.27 / 196.18 token/s。200 档此时比全部搬上 HCCS 的 194.90 更高，说明就绪时间与 RoCE 减量需要同时建模。真实的源页覆盖不均匀，关键批上是否命中不能用全轮百分比代替。 [分覆盖率的条件回放](../results/2026-11-20-peer-prefix-screen/summary.json)

持续阶段还要看工作量构成。若每输出 token 对应 $q$ 字节远端恢复，D 在供给充足时的输出能力为 $X_D$，则整体 100 Gbps 载荷上限成为稳态瓶颈需 $qX_D$ 接近 12.5 GB/s，单热点端口则用该口的份额分别检查。r33 全轮平均 $q$ 约 0.61 MB/token，r19 目标波约 10.53 MB/token。早到合集可以局部受限，一段早期高出站不一定能拖慢 D 生成尾部。保留 P APC、EOS 和原生路由，预先在同源独立会话中量实际换 P 与读强度，选能保持 D 负载并有可恢复等待的留出窗口。只增加同种低恢复强度的到达通常先碰到 P/D 算力约束。 [阶段供给证据](results.md#r26-原生-baseline-的高压带宽对照)

r19 的固定工作量便于给出量级。100 档的最后请求在 P0 同一批中，前后还有另外三笔 GET。只消除末请求自己的 GET，条件吞吐分别约为 184.80 / 185.24 token/s，低于各自的原生 200（188.86 / 187.46）。四笔都归零时，正序的完工者换为 P1 的 raman-fitting，会限制整轮增益。两边的十笔都归零时，两个 100 档条件吞吐为 196.60 / 196.73，提前 2.748 / 2.464 秒。D 生成驻留保持约 17 秒。两边都可能成为最后完成的实例，不能只算最快 P0。

| r19 同一批 3,971 token 的条件回放 | 100 Gbps | 200 Gbps |
|---|---:|---:|
| 原生实测轮均 | 174.19 | 188.16 |
| 完整命中、每 rank 有效 HCCS 7.5 GB/s、每笔固定开销 10 ms | 189.42 | 193.16 |
| 完整命中、每 rank 有效 HCCS 8.5 GB/s、每笔固定开销 10 ms | 190.20 | 193.97 |
| 完整命中、每 rank 有效 HCCS 10 GB/s、每笔固定开销 10 ms | 191.09 | 194.90 |

这一回放用 P0/P1 各 rank 的实际 GET 顺序减去读时差并重算批次最晚 forward，保留其余驻留与实际输出。保守串行同机读和跨机剩余读时，100 档追平各自原生 200 所需约 7.23 / 5.89 GB/s/rank，并行约 6.98 / 5.73 GB/s/rank。7.5 GB/s 情景相对原生约为 +8.7% / +2.7%，100 达到同策略 200 的 98.1%。10 GB/s 情景约 +9.7% / +3.6%。后者在 100 正序离追平门槛仅约 0.24 s。原 trace 中供者 P forward 与受者 GET 确有重叠，应在双向四 rank、供者继续执行的条件下测实际有效速率和干扰。该约束和页驻留需要真机标定。 [r19 阶段和两档结果](results.md#r19-apc-开启的高远端恢复-ab) · [离线复算入口](../pd_kv_transfer_opt/profiling/screen_peer_prefix.py)

按已结束的正式 rank 事件重算每笔机内打包分发的报告载荷除以墙钟。r16 六臂每臂 384 份，p50 为 8.607–8.622 GB/s。r19 四臂每臂 40 份，p50 为 8.535–8.561 GB/s。这是现有同一 TP 实例内的 HCCL 打包、广播和散写路径，给出了本机搬运可达到的量级。分源需要另量跨 P 实例、双向四对并发和供者仍在 prefill 时的有效速率，使用实际观测通路和页布局。 [r16 清洗事件](../results/2026-11-18-r16-packed-copy/r16-P.jsonl.gz) · [r19 清洗事件](../results/2026-11-18-r19-natural-ab/r19-P.jsonl.gz)

## 实现边界与对照

1. **同一请求的合法读计划。** P 的原生 APC、Store 匹配及最终目的页分配完成后，在对应 Attention 描述符上查询机内 peer P。校验模型、TP rank、块位置、cache salt、分配代次和内容来源。使用前一轮实际 P→D 交接与 D Store 写入出处建立来源关系。Store 去重或重算可导致同逻辑键从别次执行产生，需独立字节诊断或可证的同一代次。对匹配的不可变源页取得引用，所有 rank 准备失败时走原生读。保留原生对齐点的 GDN/conv 以及未命中的 Attention。 [Hybrid 对齐与数值行为](https://github.com/vllm-project/vllm-ascend/issues/16451)
2. **对应 rank 的设备搬运。** 以两个原生 TP4 组的对应 rank 为端点，测实际注册页的 HCCS 路径、分散页提交、隐含的主机 bounce 和两方向争用。采用独立完成事件，使用原生目标页并仅在本 rank 完整有效后发布。跨进程 IPC 使用授权导入，源引用持有到设备完成。失败后等已经提交的访问结束再重试或释放。避免为该路径长期预留完整历史的副本，记录持页对供者可用容量的影响。首个实机可串行走两源以分清各段，随后按两条真实完成线验证可重叠份额。
3. **带宽压力门槛。** 在实际端口及 GET 等待表明恢复受限、同机替代有净时间预算时切换。低压力快速沿原生路径，不先发跨进程逐页探测。看源 P 受到的 HBM 和 HCCS 干扰，也看 D 是否已饱和。200 Gbps 按实测约束判定是否有余量。比较同档的原生和改动，并分别给出 100 改动比 200 原生、100 改动比 200 改动。阶段数据用于说明吞吐变化。
4. **分源的独立增量。** 与原生 D 全读、合法时 peer P 全 Hybrid、分源三臂比较。必要时加入 peer P 历史加重算及原生本地会话亲和，以区分直接搬运和路由增量。若同机足以提供整份合法状态，单源 peer 已是对应强对照。已有 r19 可检查搬运机制和临界工期；同源自然高压留出会话需真实换 P、真实驻留，且 GET 等待影响 D 供给或关键会话。随后验证前驱生成确实进入后继输入或证实 GDN 缺失而 Attention 尚存的负载，才能量出两源设计的增量。每个性能对照均保留 100/200 Gbps。

## 先例与创新边界

| 先例 | 已有的机制与本候选的比较 |
|---|---|
| vLLM 双向 NIXL 与 llm-d P2P | 前者在当前 P APC 之后从 D 恢复历史，已有异步 READ 和重算阈值。后者已经从 peer CPU tier 为 P 拉取前缀，v0.30 P2P 协议每请求只提供一个 fetch source。应以单源恢复和缓存亲和为对照。 [双向 NIXL](https://vllm-project.github.io/2026/10/12/bidirectional-kvxfer-multiturn-agentic-workload.html) · [llm-d](https://github.com/llm-d/llm-d/blob/main/guides/p2p-kv-cache-sharing/README.md) · [vLLM P2P 协议](https://docs.vllm.ai/en/v0.30.0/features/kv_offloading_usage/#orchestration-layer-protocol) |
| SmartGen 与 SwiftCache | 前者在 D 侧已有同一请求按本地/远端建立两条路径并加载被选中的 KV，后者已有 NVLink 邻卡 HBM 作为缓存层。可检验的新设计范围是保持原路由时把同源的 P HBM 历史和 D 的精确 Hybrid 末段组装起来，并在 910B2 的不同物理链路上满足同一次恢复的发布边界。 [SmartGen](https://arxiv.org/html/2607.28150v1#S4.SS2) · [SwiftCache](https://arxiv.org/html/2606.16135v1#S4) |
| Evolving Infra 与 Hybrid 共同边界 | Evolving Infra 已支持跨 P/D 的 peer host-memory 恢复。Ascend #16451 与 CN122088718B 已公开 Hybrid Attention 和循环状态的恢复边界、验证或回收机制。固定 vLLM 的 Mamba manager 会释放跳过的较早状态页，提示存在同一历史仍留 Attention 而缺少兼容 GDN 的窗口。用真实 pin 命中证明此窗口和相对单源的收益。 [Evolving Infra](https://arxiv.org/html/2611.12304v1#S4.SS2) · [Ascend RFC](https://github.com/vllm-project/vllm-ascend/issues/16451) · [CN122088718B](https://patents.google.com/patent/CN122088718B/zh#claims) · [固定版回收](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/single_type_kv_cache_manager.py#L1018-L1046) |
| PPD、AscendStore #19786 与 MultiWrite | PPD 允许把增量 prefill 留在 D，改变计算放置。#19786 覆盖同一池对象分片读和机内聚合，在其 GQA 对照未打开。MultiWrite 在双机 910B4 验证机内/跨机流量取舍。这里需测同一请求的有效状态来自不同副本后，是否减掉原本需要经过 RoCE 的字节，并提供同一个固定 HCCS 局部策略作比较。 [PPD](https://arxiv.org/html/2603.13358v1) · [AscendStore PR](https://github.com/vllm-project/vllm-ascend/pull/19786) · [MultiWrite](https://arxiv.org/html/2605.22428v1#S6) |

两次独立复核对可移走的 Attention 字节、r19 完工路径和自然长会话的容量上界得到一致结论。候选具有可测试的 910B2 物理路径与高回读子负载的收益区间。原 170 请求的吞吐主张仍缺少可移动的 RDMA 关键时间。单源 peer 的合法驻留和双源的独立收益也是论文成立条件。 [脱敏数据的复算](../results/2026-11-20-peer-prefix-screen/summary.json)

## 引擎侧的独立对照

r34/r37 在原生单步 13.59 ms 的尺度上，设备算子约 8.72–9.01 ms，最后 rank 进入后的机内 collective 约 1.89–1.95 ms。两笔早期归约到齐后各约 22 微秒，HCCL 标记同步还包含前序等待。以整个设备验证周期和实际最晚 rank 估计收益。按 r34 固定 167,284 token 的条件回放，D0 尾步算子快 20% 时 345.14→355.65 token/s，纯 collective 完成段快 20% 时到 347.37。r40 两原生轮的可确认全部独跑窗口快 20%，条件增益为 3.75% / 5.08%。 [D 单步与条件吞吐](decode-execution-options.md)

MegaGDN 的 910B2 PTO 数据支持作为 P prefill 的算子对照，发布的插件集成以 0.18/0.19 为基础。PRESERVE 已在 910B 上使用 CANN 图把访存预取与通信重叠。A2 原生模型内 MTP 可以验证更少主模型串行步数的效果。Twig 和 vLLM-Ascend tree speculative RFC 已讨论 Hybrid 验证及接受状态修复，TableTune 已用成本和接受率选择草稿长度。固定版 A2 EP4 源码选的是 ALLGATHER。D 侧的这些引擎对照有各自的支持范围和效果。 [MegaGDN](https://github.com/huawei-csl/megagdn-pto#performance-highlights) · [PRESERVE](https://arxiv.org/abs/2501.08192) · [Twig](https://arxiv.org/abs/2608.01651) · [A2 tree RFC](https://github.com/vllm-project/vllm-ascend/issues/18797) · [TableTune](https://arxiv.org/html/2610.38968v1) · [固定版 MoE 选路](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/ascend_forward_context.py)
