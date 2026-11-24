# 原生 2P2D 的 NPU 实验计划

2026-11-24

## 共同部署

r43 常规长会话、r69 集中回读和 Basin 窗口 v1 共用两台各 8 卡 910B2 的原生 2P2D、Qwen3.5-35B-A3B BF16、双侧 APC、D 原生重算调度器、自然 EOS 和原生路由。P/D 每侧八个 rank 各注册 8 GiB Store，共 128 GiB，启用就近分配。 [固定版部署](decode-apc-baseline.md#共享池布局与验证) · [历史 NPU 实测](../results/2026-11-24-native-npu-reference/README.md)

## 上机顺序

1. 首批使用 `--group r43` 完成 **r43** 的四轮正式吞吐对照。
2. 接着完成 **r69** 的四轮正式吞吐对照。
3. 两组正式对照完成后执行 **Basin 窗口 v1**，先跑十轮自然完成，取消短测通过后加两轮终止对照。

统一入口、预置包和独立采集的启动方式见 [上机执行](native-npu-run.md)。

r43 和 r69 使用正式计时所需的常规遥测，合计先预留约 1.5–2 小时。r43 的四 rank Decode profiling 在 Basin 批次后单独执行。r69 的 200/400 Gbps 余量诊断暂缓。

## 双向限速

每轮给 P、D 主机各自的八个 RDMA 出站端口设置同一档，带宽按每台主机的八口合计。

| 档位 | P 出站合计 | D 出站合计 | 两端每口设置 |
| --- | ---: | ---: | ---: |
| 100 Gbps | 100 Gbps | 100 Gbps | 12.5 Gbps |
| 200 Gbps | 200 Gbps | 200 Gbps | 25 Gbps |
| 400 Gbps（r69 余量诊断） | 400 Gbps | 400 Gbps | 50 Gbps |

整轮吞吐用正式请求的实际输出 token 数，除以首条计划投放到最后一条完成的时长。每轮同时记录完成数和结束原因。两端逐口采集 TX/RX。P→D 交接与 Store GET 分开计量，GET 按实际读源分为同机、D→P、P→D 和来源未知。结合 P 批次、D 等待和最后完成的会话，判断哪个方向的传输等待传到了整轮。 [事件口径](data.md#事件口径)

## r43 常规长会话

重放冻结的 48 个会话、254 条请求。前 12 个根沿用 0–1.6 秒偏移，余下 36 个从第 5 秒起每 2.5 秒投放，最后一个在第 92.5 秒。后继等实际前驱完成，再使用原输入及原生成预算继续。 [冻结负载](../configs/workload-r42-natural-sustained.json) · [历史结果](r43-native-decode-apc.md)

正式顺序为双向 **100→200→200→100 Gbps**。分别比较正序和反序的整轮吞吐，再核对同档波动。结合实际输出长度、各方向 KV 字节、D 运行槽和压尾会话判断常规负载是否受限。四 rank Decode profiling 单独用双向 **200 Gbps** 跑，覆盖批宽 1–4 与 9–16，核对设备计算、机内通信和步间空隙。 [采集方案](r43-decode-profiling.md)

## r69 集中回读

重放冻结的 53 个双轮窗口、106 条请求。根请求按原计划在约 4.33 秒内投放，后继从第 56 秒起按计划时点及实际前驱完成释放。每臂从同一干净初态开始，保留原生路由、双侧 APC 和自然 EOS。用当前双侧池实测这批后继还能产生多少跨机 GET。 [冻结负载](../results/2026-11-22-terminal-source-restore-short-guard/workload.json) · [历史负载与实测](restore-workload-screen.md#r69-用实际根进度校准后继投放时间)

正式顺序为双向 **100→200→200→100 Gbps**。核对两组配对中的命中比例、同字节 GET、P→D 交接、P 交付及 D 供给，再看整轮吞吐差能否复现。需要判断 200 档还有多少带宽余量时，另配同负载的双向 200/400 Gbps 诊断。

## Basin 窗口 v1

用 Terminal-Bench 长上下文会话验证突发到达时如何积压并恢复。每个条件用双向 100/200 Gbps 各跑一轮。 [限速矩阵](../configs/basin-windows-v1/matrix.json)

1. 共享每档的一次 `steady`，其余自然完成条件为 +33%、+100%、五倍突发和短会话自然排空。按冻结的先后顺序，每个条件各测一次 **100/200 Gbps**，合计 **10 轮**。
2. 若运行短会话终止组，在它的首轮前用少量请求检查一次原生取消和资源释放。现场确认后加两轮，分别测 **100/200 Gbps**，在第 150 秒只终止尚未完成的突发会话。
3. 每轮以同一初始状态开始，32 次预热后正式发出 304–448 条计划调用。突发在第 60–150 秒，基础根到达至第 360 秒，至少观察到第 600 秒并排空。用逐口计数、缓存命中、D 等待和完工链解释结果。 [负载、取消与数据口径](basin-windows-v1-review.md)

## 上机准备和换轮

- **模型启动**在两台机器加载已有权重，完成图和连接器初始化，等待 P0/P1/D0/D1 与 Store 就绪。首次启动时核对实际 segment 和部署状态。
- **清缓存**发生在两轮之间。先确认上一轮请求和 KV 搬运均已结束，再恢复四个实例的本地 APC、双侧共享 Store 和代理计数的初始状态。能核实这些状态一致时复用常驻服务。当前固定版的 vLLM 本地前缀重置和 connector 重置是两步，`MultiConnector` 的成功返回本身无法证明 AscendStore 已空。现场若无法核实整个初态，再从空实验 Store 和清洁进程启动。 [调度器重置](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L1813-L1844) · [MultiConnector 重置](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py#L602-L604) · [固定版 AscendStore](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py)
- **端口核对**在 P、D 各保存八口原值。用现有 `Shaper` 在两端换到同一档，逐口核对 `configured_Mbps` 与 `current_Mbps`。两端均匹配后才开始负载。每轮完成排空、异常退出或任一侧设置失败时，两端各自恢复并读回原值。每轮归档 P、D 两份限速读回和两份恢复回执。 [带宽工具](../pd_kv_transfer_opt/experiments/hardware.py) · [Basin 执行入口](basin-windows-v1-review.md#上机计量和归档)
- **取消诊断**在终止组开始前执行一次，检查客户端断开后，原生调度器是否接受取消并归还槽位和 KV。先预留 5–10 分钟，用三种在途阶段的少量请求核对，确认结果后决定是否执行终止组。 [取消步骤](basin-windows-v1-review.md#取消臂的现场核对)

## 机时预算

下表为 Basin 窗口 v1 的排期。

| 执行范围 | 历史单限 D 的 CPU 负载估算，含预热与排空 | 排期按需冷启 |
| --- | ---: | ---: |
| 10 轮自然完成 | 4.52–4.94 小时 | 约 6–7 小时 |
| 12 轮，含终止组 | 5.38–5.87 小时 | 约 7–9 小时 |

完整 12 轮暂按历史负载预留 6–7.5 小时。若现场全程可验证常驻服务的干净初态，完整矩阵的墙钟时间约 **6.5–8 小时**。冷启按约 5 分钟/轮预留，12 轮合计约 1 小时。预留按权重和依赖已备好计算，两台机器同时占用。首对双向限速实测记录启动、换轮、预热及实际 EOS 排空时长，用来更新预算。 [离线负载预算](../configs/basin-windows-v1/matrix.json)
