# r43 原生 200 Gbps 的四 rank Decode profiling

沿用 r43 的 Terminal-Bench 48 个会话、254 条请求和原始投放计划。新一轮使用两台各 8 卡 910B2、P/D 各 64 GiB Store 的当前基线。保留双侧 APC、D 原生重算调度器、自然 EOS、原生路由和 D `FULL_DECODE_ONLY`。profiling 单独运行，P、D 出站各限 200 Gbps，两端分别读回八口各 25 Gbps。正式实验按首条计划投放到最后请求完成的输出 token/s 验收。 [双向限速与正式对照](terminal-throughput-plan.md#r43-常规长会话) · [r43 历史归档](r43-native-decode-apc.md)

## 运行入口

CPU 复算读仓库归档。第一条核对 r34 的 scheduler/worker 事件和 r34/r37 两份四 rank 分解，第二条重放原生 r43 的两份 200 Gbps 正式记录。

~~~bash
python -B -m pd_kv_transfer_opt.profiling.analyze_r43_decode archive \
  --output runs/r43-decode-archive-smoke.json
python -B -m pd_kv_transfer_opt.simulator.decode_step_gain \
  --output runs/r43-decode-step-gain.json
~~~

有卡后从空 Store 和隔离 APC 命名域启动独立诊断轮。P 用当前基线启动，D 增加原生 worker profiler 和步进采样。四步采样一次，使用固定版 profiler 自带的 CPU/NPU、Level1 和 `PipeUtilization`。保留同一批的图 replay 标识，供 CANN 和 scheduler 事件关联。 [固定版 profiler](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/profiler/torch_npu_profiler.py#L33-L69) · [CANN 9.1 采集配置](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0033.html)

~~~bash
python -B -m pd_kv_transfer_opt.experiments.serve --config "$R43_SITE" --role D \
  --run-id "$R43_RUN_ID" --run-dir "$R43_D_RUN" \
  --decode-native-profiler --profile-decode-steps --decode-step-stride 4
~~~

核对并归档 P、D 两份读回后，在 D 主机上随同一份物化负载启动采集控制器。`--shaping` 传入 D 的实际读回，可先用 `--dry-run` 检查控制器配置。正式采集在 D0 和 D1 分别等待出现批宽 1–4 和 9–16，连续采样确认仍有生成后触发约一秒短窗。默认在第 93 秒后采窄批，以覆盖更多排空段。一次只开一个窗口，初次窗口未覆盖的实例和批宽可在另一个独立方法轮单独采。触发时的运行数只用来筛窗，最终宽度以四 rank 对上的实际派发为准。

~~~bash
python -B -m pd_kv_transfer_opt.profiling.r43_decode_capture --config "$R43_SITE" \
  --workload-manifest "$R43_WORKLOAD/manifest.json" --shaping "$R43_ARM/shaping.json" \
  --run-dir "$R43_D_RUN" --output "$R43_ARM/decode-windows.json"

python -B -m pd_kv_transfer_opt.profiling.analyze_r43_decode capture \
  --run-dir "$R43_D_RUN" --receipt "$R43_ARM/decode-windows.json" \
  --workload-manifest "$R43_WORKLOAD/manifest.json" --output "$R43_ARM/decode-breakdown.json"
~~~

CANN 的 CSV/JSON 尚未导出时，在已有 Ascend 工具环境给分析命令加 `--parse`。采集回执记录触发宽度、启停和每个 rank 的目录。分析只接受四 rank 共享同一批标识、实际请求已经进入生成、且使用 FULL 图的单 token 步。当前解析已核对 r34/r37 的每步 81 次 AllReduce 和末尾 logits AllGather，遇到不同 collective 序列会记录拒绝原因。诊断窗口、正式计时和原始 CANN 文件各有自己的保留范围。 [采集脚本](../pd_kv_transfer_opt/profiling/r43_decode_capture.py) · [解析脚本](../pd_kv_transfer_opt/profiling/analyze_r43_decode.py)

## 时间拆分和局部反事实

- 对齐相邻 logits AllGather，依次计算每次 collective 的**最晚到达之前**与**到齐后至最后完成**。前者按当次控制 rank 切分，后者记为机内通信。提前到达 rank 的等待归到前驱的一侧。
- 在控制 rank 上用 CANN 的 `memory_bound` 或同核 MTE2、Cube、Vector 比值给关键算子标记算术或访存候选。输出不重叠的候选算子墙钟、重叠或未分类墙钟、额外设备拷贝和设备事件等待。PMU 计数说明执行特征，算术与数据搬运可在同一算子内重叠。 [CANN 字段定义](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0067.html)
- host 项只从前两次归约的提交线程取可见同步 API，减去同期已经计入的设备活动。未解释空档保留原值。每组同时报告 CANN 窗外的原生步频，以免把采集扰动当成可消除的 host 成本。
- 对可见算术、访存候选、机内通信和 host 同步分别缩短 10%、20%、30%，逐次重新取最晚 rank。某一 rank 自身更快时，只有汇合和下一有效 token 都提前的部分进入吞吐回放。算子家族各自取并集，家族之间可能重叠。

## 无卡复核与优化量级

r34 归档重算得到 **2,788 个四 rank 步**，11,152 条 worker 记录均完整关联。纯 FULL 步中，批宽 1–4 有 **1,949 步**，9–16 有 **325 步**。另有排除 CANN 窗口后的原生步频统计。r34/r37 的四 rank 分解各有 **47/43 步**。原始四 rank CANN 文件未保存在当前本地，旧档可重放分解和核算，新的原始文件入口用两组批宽的合成输入验过。访存归类等待同配置 PMU 输出。 [归档复核](../results/2026-11-23-r43-four-rank-prep/archive-smoke.json)

| 旧四 rank 证据，单请求 | r34 | r37 |
|---|---:|---:|
| 前两次归约到齐后合计 p50 | 0.061 ms | 0.044 ms |
| 前两次归约前可见 host 同步合计 p50 | 0.017 ms | 0.016 ms |
| 全步计算/访存段缩短 20% 时的净步时节省 p50 | 1.81 ms | 1.75 ms |
| 全步到齐后通信缩短 20% 时的净步时节省 p50 | 0.39 ms | 0.38 ms |

首查大块设备算子的算术和访存暴露。旧 r34 的 rank 0 矩阵乘并集按观察到的 AllGather 折算，单请求窗约 **3.43 ms/步**，两个多请求窗约 **4.16–5.63 ms/步**。若这整段在新 trace 中仍是关键路径并能压缩 20%，量级为 **0.69–1.13 ms/步**。新采样要检验最长部分来自哪一组矩阵乘或注意力算子、访存是否限制它们，以及最晚 rank 是否真的因此提前。只缩短前两笔 HCCL 的到齐后成本，或只缩短旧 trace 已见的 host 同步，达不到 1 ms/步。 [四 rank 原归因](critical-path-attribution.md#r37-原生尾请求的单步分解)

含排空增益用 $X'=N/T'$ 和 $X'/X-1=T/T'-1$ 计算，$T'$ 由原会话完成依赖重新求解。两臂的归档基线是 **209,884 token / 504.80 秒 / 415.78 token/s** 和 **210,866 token / 379.79 秒 / 555.21 token/s**。每臂分别固定到达、EOS 和路由。下表取两臂增益的较小值，范围由冻结客户端流宽度和重算 D 组批两种回放给出。

| 具有净步时节省的批宽 | 每步省 1 ms | 每步省 1.5 ms | 每步省 2 ms |
|---|---:|---:|---:|
| 1–4 | +2.15–2.46% | +3.25–3.50% | +4.38–4.60% |
| 9–16 | +0.84–0.93% | +1.27–1.41% | +1.70–1.85% |
| 1–4 与 9–16 | +3.03–3.45% | +4.61–5.15% | **+6.24–6.91%** |
| 1–16 | +3.78–4.20% | +5.78–6.25% | **+7.86–8.43%** |

直接代入旧捕获中 20% 设备算子缩短对应的 **1.75–1.81 ms/步**，若两组目标批宽均能得到这项净节省，两臂共同收益为 **+5.43–6.22%**。目标批宽省 2 ms 时，冻结宽度的两臂为 **454.19 / 589.85 token/s**，严格只计稳定计数窗口时共同收益为 **+1.83%**。这说明计入哪些真实批次会改变结果。此前按阶段百分比的模型会给宽批更大的绝对节省，当前表统一使用固定毫秒。获得同配置四 rank 分解后，按实测可省的毫秒和覆盖宽度选择一个改动做 A/B，正式验收只看含排空总吞吐。 [逐臂结果](../results/2026-11-23-r43-four-rank-prep/step-gain.json)
