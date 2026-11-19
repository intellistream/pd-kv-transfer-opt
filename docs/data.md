# 结果与事件契约

[r14 清单](../results/2026-11-18-r14/manifest.json) 给出提交文件的校验值和源导出指纹。每轮数据在 [rounds.csv](../results/2026-11-18-r14/rounds.csv)，复测和阶段分解在 [summary.json](../results/2026-11-18-r14/summary.json)，CANN 对齐在 [profiler-summary.json](../results/2026-11-18-r14/profiler-summary.json)。

| 项 | 口径 |
|---|---|
| 发送上限 | D 侧八个 NPU 端口的硬件 egress 上限之和，单位十进制 Gbps |
| 实际发送 | D 端 MAC TX octet 差值除以采样时间，包含网络开销 |
| GET 有效字节 | 原生 Store API 提交的描述符载荷，P→D 交接另记 |
| 整轮输出吞吐 | 全部正式请求的实际输出 token 除以首条计划投放至最后完成的时长，包含排空 |
| TTFT | 客户端 ready 到首个含内容或 token IDs 的响应时刻 |
| P GET 区间 | 同一 rank 的批量 GET 墙钟区间并集 |
| 设备 overlap | 同 rank 且同一捕获窗口内，设备时间区间与 host GET 的交集 |

事件导出为 gzip JSONL，每行有 `kind` 和 `data`。两侧用代理 UUID 关联请求。保持原时间单位，跨主机用 Unix 锚点对齐，本机耗时用 monotonic clock 差值。通用 host span 字段见 [事件 schema](../schema/rdma-lab-events-v1.json)。

通用客户端分析的 `finite_cohort.output_tokens_per_s` 使用整轮口径，覆盖所选完整会话；缺失、失败或 token 计数错误的轮次该值为空。主负载从 `--window-start 0` 采完整会话。分别保存输出上限、EOS 策略、实测结束原因、实际 token 数、客户端排队和计划投放偏差。[分析入口](../pd_kv_transfer_opt/experiments/analyze.py)

正式比较应保留工作负载及其版本、软硬件、卡组、带宽读回、每轮成功数、排空范围、采样覆盖与可重算证据。整理新结果时可运行 `pd_kv_transfer_opt.profiling.public_bundle` 提取白名单字段。请求正文、全量设备捕获、完整进程栈和机器配置存于外部实验归档。

## 归档与清理

每轮收尾时核对工作负载、限速读回、输出合同和清洗 trace 的重算。有效的正、负结果保留可重算证据。确认失败或无效的轮次只保留失败阶段、原因、关键配置和收尾回执，并及时清理原始日志、部分结果和重复导出。已通过校验的正式归档作为核对重复文件的依据。优先使用回收站移出已确认无效的生成数据。若某次不完整采集仍能支持明确定义的诊断，记录用途和覆盖范围后保留所需事件。

## 事件口径

r19 的请求标识是 `arm~natural-<source-id>`。代理保留符号路由 `P0/P1/D0/D1`，用于核对请求实际去向。`p_port_snapshot` 是 P 侧计时窗口两端的八口快照，`cap_port_sample` 是 D 侧该窗口内的多次采样。D MAC TX 与 P MAC RX 用于交叉核对恢复方向。D RX、P TX 和 `native_transfer_submit` 用于核对反向交接。采样间隔以逐端口的 monotonic 时间差为准，PFC 帧计数不提供暂停总时长。GET 活跃时间按各 rank GET 区间并集计算，按实例和全局范围分别记录。[r19 重算](results.md#r19-apc-开启的高远端恢复-ab)

r20 向后延迟 copy 完成事件。使用实例、进程、线程、批次及 `call_ordinal` 将它关联到原 GET。`r20_copy_wait` 记录 copy 提交完成、后继 GET 边界的 NPU 事件查询状态和最终 wait。两次查询的前后 host 时间用于夹定设备事件窗口，聚合时在每个实例内对重叠区间取并集。待完成的 copy 从提交到晚些确认的 host span 可能覆盖整个 GET，设备事件下界另算。[r20 重算](results.md#r20-同批分发与下一笔-get-的异步重叠)
