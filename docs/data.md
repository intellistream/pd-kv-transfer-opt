# 结果与事件契约

[r14 清单](../results/2026-11-18-r14/manifest.json) 给出提交文件的校验值和源导出指纹。每轮数据在 [rounds.csv](../results/2026-11-18-r14/rounds.csv)，复测和阶段分解在 [summary.json](../results/2026-11-18-r14/summary.json)，CANN 对齐在 [profiler-summary.json](../results/2026-11-18-r14/profiler-summary.json)。

| 项 | 口径 |
|---|---|
| 发送上限 | D 侧八个 NPU 端口的硬件 egress 上限之和，单位十进制 Gbps |
| 实际发送 | D 端 MAC TX octet 差值除以采样时间，包含网络开销 |
| GET 有效字节 | 原生 Store API 提交的描述符载荷，P→D 交接另记 |
| 排空吞吐 | 成功完成的输出 token 或请求数除以从释放起点到最终完成的时长 |
| TTFT | 客户端 ready 到首个含内容或 token IDs 的响应时刻 |
| P GET 区间 | 同一 rank 的批量 GET 墙钟区间并集 |
| 设备 overlap | 同 rank 且同一捕获窗口内，设备时间区间与 host GET 的交集 |

事件导出为 gzip JSONL，每行有 `kind` 和 `data`。两侧用代理 UUID 关联请求。保持原时间单位，跨主机用 Unix 锚点对齐，本机耗时用 monotonic clock 差值。通用 host span 字段见 [事件 schema](../schema/rdma-lab-events-v1.json)。

正式比较应保留工作负载及其版本、软硬件、卡组、带宽读回、每轮成功数、排空范围、采样覆盖与可重算证据。整理新结果时可运行 `pd_kv_transfer_opt.profiling.public_bundle` 提取白名单字段。请求正文、全量设备捕获、完整进程栈、机器配置和无效轮存于外部实验归档。
