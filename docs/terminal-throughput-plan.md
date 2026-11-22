# Terminal-Bench 原生 2P2D 的吞吐验证路线

2026-11-23

## 主场景

用 r43 的原生 2P2D 在 **D 出站聚合 200 Gbps** 下研究含排空总吞吐。D 八口各读回 25 Gbps，P 维持已归档的原档。这个选择沿用 r43 两次高档与同配置方法轮的真实对照，并降低为了制造压力而把其他资源瓶颈掩盖的风险。100 Gbps 留作 RDMA 因果验证与最终候选的压力回归。是否有网络余量按实际方向、逐口压力、待发工作与关键路径判定。若分析发现 P→D 也受限，先用同一负载定位该方向的服务约束。 [部署与原始档](r43-native-decode-apc.md)

两台 8 卡 910B2、Qwen3.5-35B-A3B BF16、每实例 Attention TP4、Expert EP4 和 DP1。使用两侧 APC、D `AsyncRecomputeScheduler`、P 原生分块图、D `FULL_DECODE_ONLY`、4096 分块预算、16 槽和 262,144 上下文。D 通过 `MooncakeConnectorV1` 从 P 拉取交接 KV，P 通过同步 `AscendStoreConnector` 从 D 的 128 GiB Store 恢复缺页。保持原生路由、停止规则和已经核对的版本。使用双 APC 专用模板。官方 A2 示例展示同类 Pull、APC 和 D 重算机制，给出的是 397B W4A8/MTP、1P1D/TP8 的参考部署。客户生产拓扑及服务版本仍以现场为准。 [本地配置模板](../configs/ascend-2p2d-decode-apc.example.json) · [固定版部署说明](decode-apc-baseline.md) · [官方 A2 参考](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/Qwen3.5-397B-A17B.html#56-prefill-decode-disaggregation-a2) · [官方 Store 指南](https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/feature_guide/kv_cache_pool_ascend_store.html)

保留 r43 的 **48 个完整会话和 254 条原请求**。前 12 个根请求沿用 0–1.60 秒的偏移，其余 36 个从第 5 秒起每 2.5 秒投放，最后一个在第 92.5 秒。后继等待实际前驱完成，用原输入和原生成预算，服务照常判定 EOS。两个 200 档各有 246 条以 EOS 停止、8 条达到原预算。提示中位数为 45,490 token，输出中位数 433/458 token。这是同类长上下文 multi-turn Agent 的可复算服务负载。输入来自另一个模型的终端任务轨迹，后继沿用其历史工具结果，回放的工具间隔为零，根到达由实验计划给定。对真实客户的负载分布和端到端 Agent 行为，仍需客户数据。 [冻结清单](../configs/workload-r42-natural-sustained.json) · [来源和投放](r42-natural-sustained.md)

r45 是 170 条、30 会话的短投放复核，r48 是同 254 条输入的突发投放复核。两者历史上为双向限速，r45 的输入也是 r43 的子集。若将其用于确认通用候选，在主协议下重建原生对照。跨输入的外推使用预先冻结的独立原会话。r69/r78 的短生成和集中恢复用于验证 RDMA 问题，首 token 放行、强制输出长度、关 APC 和附加流量留作机制对照。 [突发对照](restore-workload-screen.md) · [放行敏感性](simulator.md#首-token-放行如何放大完工差)

## 已有归档指向的瓶颈

按逐请求事件和真实前驱重放全轮。每臂冻结其输出、路由、其余阶段，并在改变一个阶段后重找最后完成链。D 批宽 1–4 的采样只用两端运行数相同、无等待、输出计数前进且整窗为同一组可见客户端流的区间。以下增益是对这些观测区间的条件敏感性。 [重算入口](../pd_kv_transfer_opt/profiling/screen_terminal_throughput.py) · [完整数字与输入 SHA](../results/2026-11-23-terminal-throughput/summary.json)

| r43 的 200 Gbps | 实际输出 / 含排空 / 吞吐 | 最后会话链 P HTTP / D 首 token 后 | 全 P HTTP 少 10% | 全 D 首 token 后少 10% | 仅采样支持的 D 宽度 1–4 少 20% |
|---|---:|---:|---:|---:|---:|
| 正序正式 | 209,884 / 504.80 s / 415.78 token/s | 19.31 / 433.81 s | +0.38% | +9.40% | +9.48% |
| 反序正式 | 210,866 / 379.79 s / 555.21 token/s | 110.08 / 257.33 s | +2.99% | +7.27% | +2.54% |
| 独立方法轮 | 217,675 / 423.42 s / 514.09 token/s | 15.90 / 361.00 s | +0.38% | +9.32% | +4.54% |

两次正式 200 的 D 出站一秒峰值为 68.31/63.08 Gbps，任一 D 口达自身上限 85% 的采样时间均为零。同档输出总量只差 982 token，末链和完工时长仍相差 125.00 秒。压尾链中有多轮输出触及原生成预算。r45/r48 历史高档也看到 D 首 token 后的阶段敏感性，短尾 r69/r78 中 P 与 D 的排序可以变化。**对主负载先找长会话前驱链上可加速的 D 有效 token 步进**。当前证据定位到生成驻留和下一轮释放，具体设备算子、同步等待、宽度与路由的影响仍待同配置 profiling。P 冷首轮可能成为第二条限制路径，尤其是反序臂。

### 先验证 D，再定优化手段

1. 在 200 Gbps 原生路径补同档 A/A，并记录每条的实际输出长度、终止原因、逐轮路由及前驱和尾链。正式只按实际输出 token 除以首条计划投放至最后完成的时间验收。冷算、稳态和排空分段只用于解释。保持 EOS 和原 max_tokens。对照时分报输出量与时长。新 A/A 的波动和正式客户端开销先核对。
2. 从原观察确定要采的**整条长会话链**和并行 D 进度。用既有 D 批宽观测与 CANN 短窗覆盖宽度 1、2–4、5–8、9–16 及相应上下文。对齐每有效 token 的 host 提交、图实际执行、计算/访存、最晚 rank 到齐、到齐后机内通信、KV 可运行时刻和首 token 后驻留。P 同时量实际冷首块等待、分块执行与 P→D 交接余量。r34/r37 的旧 100 档拆分用于放置探针。
3. 同栈兼容时用**原生一草稿 MTP**做单因素强对照。Hybrid Qwen3.5 的 P/D 草稿数相同。保留严格验证，核对实际图、KV 交接、拒绝后状态与输出质量。用完整周期 $c$ 和接受率 $p$ 比较单请求的每有效 token 时间 $c/(1+p)$，多请求以每轮有效输出计。覆盖宽度 2–4，量宽批和 P 的代价，再做 200 的交替正式原生对照。局部没有有效 token 提速、宽批或 P 开销抵消了收益、或全轮差分不开同档波动时，换到新 trace 验明的资源。MTP 的收益用于工程参照，论文贡献待实际瓶颈定位。 [已有 D 分解](critical-path-attribution.md#r37-原生尾请求的单步分解) · [固定版 MTP 说明](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/docs/source/user_guide/feature_guide/speculative_decoding.md#common-configuration)
4. 若新 trace 指向到齐前的晚 rank，与同步发起和图执行一起调整；若指向设备访存/算子，先量关键步的单项变化；若指向窄批持续缺供或 P 冷队列，先用实测按宽度/上下文划分的服务曲线与合法就绪依赖，分别以原生和 SJF 为调度对照。FineStream 及 MorphStream/SAGE 的服务曲线和依赖判据继续用于这部分。网络减量、合批或搬运只在逐口服务与实际缺页、发布依赖相交的窗口重评。 [已有跨领域调研](stream-processing-related-work.md) · [P 调度对照](cold-prefill-order.md)

## 最后一轮 RDMA 收益验证

1. 冻结 r69 的 **53 个会话、106 条原请求**，沿用该轮的计划到达、原前驱、EOS 与双 APC。不增加后台流量、强制路线或同头打包。每臂统一初始缓存，正式计时与详细观测分开。P 保持原档，只设置 D 八口出站总上限，按 **100→200→200→100 Gbps** 回切。读回每口 **12.5 / 25 Gbps**，随后恢复原档。原实测参考 **225.27→245.16、220.91→243.66 token/s**，低档低 **8.11% / 9.34%**。 [冻结到达](../results/2026-11-22-terminal-source-restore-short-guard/workload.json) · [原实测与同字节方法](restore-workload-screen.md#r69-用实际根进度校准后继投放时间)
2. 用相邻 **400 Gbps 聚合参考**核对 200 的剩余约束，八口各 50 Gbps，其他方向仍为原档。若高档继续加快匹配 GET 或降低整轮时间，保留 200 局部受限结论。如需确认平台，再配一次高档反序和 200 参考。高档单点只提供余量诊断。r78 的同量正序用作已有交叉证据。
3. 主结果逐臂报实际输出、含排空秒数、token/s、两种序的相对差和同档变化。归因沿同 P 去向、同 rank 布局和同描述符字节的 GET，记录逐口服务、P 结果、D 已就绪/可运行与无供给、决定完工的 EOS 链。同步记录实际 P/D 路由和复用/字节。匹配输出量和阶段的回放只用于归因。若热点与关键 GET 等待解除后总吞吐持平，确认现负载的带宽收益小；若统一出现匹配 GET 缩短、D 供给和整轮提升，就按实测幅度报告。验收使用原生负载的配对结果。历史 r19 同头打包作为网络工程项的旁证，实际键未知的跨请求减量仍为条件估计。 [历史强压边界](simulator.md#传输主导和解码主导的实测对照) · [r69 逐口条件回放](simulator.md#r69-p1-第-228-批的五条请求与逐口争用)

~~~powershell
python -B -m pd_kv_transfer_opt.profiling.screen_terminal_throughput `
  --output runs/terminal-throughput.json
~~~
