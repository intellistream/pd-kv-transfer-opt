# Basin 窗口 v1 的 NPU 实验计划

2026-11-24

## 本次范围

本次上机验证 Terminal-Bench 长上下文窗口在突发到达下的拥塞与恢复。采用两台各 8 卡 910B2 的原生 2P2D、Qwen3.5-35B-A3B BF16、双侧 APC、D 原生重算调度器、自然 EOS 和原生路由。P/D 每侧八个 rank 各注册 8 GiB Store，共 128 GiB，启用就近分配。r69、r43 和 r78 的历史 NPU 结果用来校准和对照。 [固定版部署](decode-apc-baseline.md#共享池布局与验证) · [历史 NPU 实测](../results/2026-11-24-native-npu-reference/README.md)

### Basin 突发的实机顺序

**两档均使用双向限速。** 100 档将 P 出站和 D 出站各设为 100 Gbps，200 档将两端各设为 200 Gbps。每端八口分别读回 12.5 或 25 Gbps。P 出站约束 P→D 的 KV 交接，D 出站约束从 D 侧 Store 发送的 D→P 回读。 [限速矩阵](../configs/basin-windows-v1/matrix.json)

1. 共享每档的一次 `steady`，其余自然完成条件为 +33%、+100%、五倍突发和短会话自然排空。按冻结的先后顺序，每个条件各测一次 **100/200 Gbps**，合计 **10 轮**。
2. 若运行短会话终止组，在它的首轮前用少量请求检查一次原生取消和资源释放。现场确认后加两轮，分别测 **100/200 Gbps**，在第 150 秒只终止尚未完成的突发会话。
3. 每轮以同一初始状态开始，32 次预热后正式发出 304–448 条计划调用。突发在第 60–150 秒，基础根到达至第 360 秒，至少观察到第 600 秒并排空。正式 TPS 按首条计划投放到最后完成的时长和实际输出计算。用逐口计数、缓存命中、D 等待和完工链解释结果。 [冻结矩阵](../configs/basin-windows-v1/matrix.json) · [负载、取消与数据口径](basin-windows-v1-review.md)

## 上机准备和换轮

- **模型启动**在两台机器加载已有权重，完成图和连接器初始化，等待 P0/P1/D0/D1 与 Store 就绪。首次启动时核对实际 segment 和部署状态。
- **清缓存**发生在两轮之间。先确认上一轮请求和 KV 搬运均已结束，再恢复四个实例的本地 APC、双侧共享 Store 和代理计数的初始状态。能核实这些状态一致时复用常驻服务。当前固定版的 vLLM 本地前缀重置和 connector 重置是两步，`MultiConnector` 的成功返回本身无法证明 AscendStore 已空。现场若无法核实整个初态，再从空实验 Store 和清洁进程启动。 [调度器重置](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L1813-L1844) · [MultiConnector 重置](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py#L602-L604) · [固定版 AscendStore](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py)
- **端口核对**在 P、D 各保存八口原值。两端换到同一档，逐口读回匹配后开始负载。结束或异常退出时两端各自恢复并读回原值。 [限速和恢复步骤](basin-windows-v1-review.md#上机计量和归档)
- **取消诊断**在终止组开始前执行一次，检查客户端断开后，原生调度器是否接受取消并归还槽位和 KV。先预留 5–10 分钟，用三种在途阶段的少量请求核对，确认结果后决定是否执行终止组。 [取消步骤](basin-windows-v1-review.md#取消臂的现场核对)

## 机时预算

| 执行范围 | 历史单限 D 的 CPU 负载估算，含预热与排空 | 排期按需冷启 |
| --- | ---: | ---: |
| 10 轮自然完成 | 4.52–4.94 小时 | 约 6–7 小时 |
| 12 轮，含终止组 | 5.38–5.87 小时 | 约 7–9 小时 |

完整 12 轮暂按历史负载预留 6–7.5 小时。若现场全程可验证常驻服务的干净初态，完整矩阵的墙钟时间约 **6.5–8 小时**。冷启按约 5 分钟/轮预留，12 轮合计约 1 小时。预留按权重和依赖已备好计算，两台机器同时占用。首对双向限速实测记录启动、换轮、预热及实际 EOS 排空时长，用来更新预算。 [离线负载预算](../configs/basin-windows-v1/matrix.json)
