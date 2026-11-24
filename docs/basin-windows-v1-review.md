# Basin 长上下文窗口负载评审

**修改后通过。** 本次 NPU 计划执行 Basin 窗口 v1，检验负载变化和拥塞恢复。先跑 10 轮自然完成。原生取消和资源释放通过一次现场短测后，再加两轮短会话终止。r69 和 r43 作为历史实测参考。 [执行矩阵](../configs/basin-windows-v1/matrix.json) · [准备与机时](terminal-throughput-plan.md#机时预算)

## 冻结工作量与带宽矩阵

表中的 100 和 200 Gbps 都是**每端的聚合出站限额**。两台机器同时使用同一档，P 和 D 各八口。原始输入、预热和到达表的指纹由矩阵引用。 [执行矩阵](../configs/basin-windows-v1/matrix.json)

每臂从清洁缓存开始，32 个初始窗口各预热一条源调用，正式阶段继续各自后五条。24 个新基础窗口各六条调用，根到达窗口为第 0–360 秒。突发根到达窗口为第 60–150 秒。正式阶段至少观察到第 600 秒并排空。分别分析 **150–360 秒**的持续基础流与第 360 秒后的排空。首轮冻结 seed 216，另一个种子用于 CPU 敏感性。给定根到达数后用均匀时刻生成条件泊松到达，使用约 4 会话/分钟作为名义基础率。论文按部署标定的半过载率需另行实测。 [Basin §3.3 残余会话](https://arxiv.org/html/2611.02543v1#S3.SS3) · [§7.1 实验方法](https://arxiv.org/html/2611.02543v1#S7.SS1)

| 条件 | 额外窗口 | 预热 / 正式计划调用 | 带宽先后 | 状态 |
| --- | ---: | ---: | --- | --- |
| `light/steady` | 0 | 32 / 304 | 100 → 200 | 每档一个共享控制 |
| `fivefold/burst-drain` | 24 | 32 / 448 | 200 → 100 | 五倍名义率 |
| `light/plus33` | 2 | 32 / 316 | 100 → 200 | 轻度升压 |
| `light/plus100` | 6 | 32 / 340 | 200 → 100 | 中度升压 |
| `short/burst-drain` | 24 | 32 / 400 | 100 → 200 | 四调用窗口自然排空 |
| `short/burst-terminate` | 24 | 32 / 400 | 200 → 100 | 待原生取消诊断通过，第 150 秒终止残余突发会话 |

独立逐项核对了 480 个 token 哈希、400 个完整相邻前缀长度、源 ZIP 中的文件哈希和全部八个加载臂。三组共享相同基础输入、到达时刻、工具间隔、预热和自然 EOS 预算。每个带宽可共用对应的 `steady`，前提是实际软件、路由、初始化、预热和两侧驻留可比。预热 token 数、实际命中及初始化差异要随结果存档。一次带宽配对用于初步响应，后续正式复测有独立预算。 [提交方案和输入](../results/2026-11-23-basin-window-review/submitted/final-design.json) · [归档哈希](../results/2026-11-23-basin-window-review/archive.json)

窗口起点保留完整源历史提示，实际回放切成连续六调用或四调用。相邻提示复用约 91.5%–93.0%，工具时钟由原始调用间隔统一缩放到中位数 0.42 秒，p95 为 9.239 秒。真实工具输出继续引用原轨迹。这里能观察短时拥塞、缓存换出和终止后的资源归还。论文的小时级自维持缓存陷阱需要更长的持续基础流、真实会话寿命和一组相同输入且有长期恢复轨迹的实测证据。

## 独立复算和瓶颈判断

原提交的 96 个条件已逐行核对工作量、TPS 口径和基础流完整性，代表性 12 臂独立复算到提交结果。以下历史 CPU 结果只调了 D 出站，使用 seed 216、r43 拟合 1、已知 Qwen EOS 优先且未知项使用源长度。每格按 D 端 100 / 200 Gbps 排列。双向限速的影响由新实机配对测定。 [96 条提交记录](../results/2026-11-23-basin-window-review/submitted/simulation-receipt.json) · [独立复算及敏感性](../results/2026-11-23-basin-window-review/independent-replay.json)

| 条件 | 正式输出 token | 含排空秒数 | 总 TPS | 共同基础 TPS |
| --- | ---: | ---: | ---: | ---: |
| steady | 852,642 / 852,642 | 1235.74 / 1239.89 | 689.99 / 687.67 | 689.99 / 687.67 |
| +33% | 872,986 / 872,986 | 1300.63 / 1265.77 | 671.20 / 689.69 | 655.56 / 673.62 |
| +100% | 898,024 / 898,024 | 1264.20 / 1274.32 | 710.35 / 704.71 | 674.45 / 669.09 |
| 五倍自然排空 | 1,107,810 / 1,107,810 | 1413.29 / 1422.49 | 783.85 / 778.78 | 603.30 / 599.40 |
| 短窗自然排空 | 1,033,932 / 1,033,932 | 1364.39 / 1357.00 | 757.80 / 761.93 | 624.93 / 628.33 |
| 短窗终止，含模型部分输出 | 867,110 / 867,113 | 1228.63 / 1264.05 | 705.76 / 685.98 | 693.98 / 674.53 |

五倍组增加工作量和解码批宽，总 TPS 比稳态高，共同基础 TPS 却低约 12.6%–12.8%。100 档的基础平均 TTFT 从 2.328 秒升到 27.843 秒，其中 D 入队到分配从 1.349 秒升到 26.700 秒，约 **99.4%** 的增量在该段。P 的执行时间含 GET 从 0.477 秒升到 0.549 秒。模拟里稳态已经发生约 **89.8 GB 主机池驱逐**，五倍约 **262.6 GB**，同时出现 290 个模型 HBM 页换出。应核对这些驱逐实际有没有增加重算或链路等待。 [提交的 TTFT 分解](../results/2026-11-23-basin-window-review/submitted/effect-attribution.json) · [独立逐门控结果](../results/2026-11-23-basin-window-review/independent-replay.json)

固定版代理先等待 P 响应，取得交接描述符再选择 D。D 调度器在请求进入后分配接收页并可进 `WAITING_FOR_REMOTE_KVS`，实际生成逐步申请页，页不足能触发原生重算路径。模型近似批宽步时，把正在拉取的请求计入准备占槽，并在接纳时预留最终输出的容量。独立复算去掉未来输出的 HBM 占用，保留生成时间及 Store 发布，代表组的 D 排队和完工保持一致。五倍组 100 档约 **95.4% 的累计 D 请求等待时间**落在该实例有 16 条正在运行请求的区间，其余 4.6% 来自含待 KV 的槽，模型中没有页容量门控。D 步时乘 0.8 或 1.2 时，基础流平均 D 排队约 **17.4 或 37.4 秒**。现场量实际批宽、空闲块、准入失败、在途 KV、重算与等待队列，检查上述归因。 [固定代理的 P/D 顺序](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L834-L871) · [D 原生调度器](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/core/recompute_scheduler.py#L552-L620) · [vLLM 页分配](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/kv_cache_manager.py#L323-L362)

480 条中 158 条有 Qwen 自然 EOS 观测。在历史优先长度代理里，其余 322 条的源长度占约 **92.9%** 的总输出工作量。生成队列的量级仍需真实 EOS 校准。模型 HBM 每实例约 309 万 token 等效容量来自旧校准，需用当前四实例启动的分组、块数和对齐数据核对。双侧 Store 的对象落点尚待逐 segment 采样；20.97 GB/s 来自旧同机单 rank 微测。半速同机读和偏置驱逐敏感性改变读量与重算，代表性五倍组的平均 D 排队仍在约 26.5–27.3 秒。五倍组跨所有提交组合的 200 相对 100 总 TPS 差约 **−1.1% 至 +2.6%**，代表组突发窗口内模型八口没有达到各自配置的 85%。这组数据支持短时拥塞与回收的检验，RDMA 收益另用实测链路压力与关键路径配对。 [独立复算范围](../results/2026-11-23-basin-window-review/independent-replay.json)

## 取消臂的现场核对

短窗的 300 秒筛选来自空闲模型，最慢入选为 296.2 秒。主矩阵用“四调用窗口”描述负载。若报告健康短会话资格，用目标 Qwen 在健康基础流上单独注入待验证窗口，计从根就绪至最后自然 EOS，按实际结果给每个窗口资格。正式机时表另计这一诊断。

固定版代理取消监听会取消 handler。初次 P 请求等待期间，`assign_instances` 用 `except Exception` 回收预占评分，而 Python 的 `CancelledError` 会绕过该分支。取消发生在返回流之前时，代理计数或缓存保留有可能拖到之后才清理。D 流阶段还有独立的 generator 退出、调度器 abort 和源页面释放路径。因此先用正常请求分别命中 **P 请求已发出、D 等待 KV、D 生成中** 的短诊断，核对代理在请求全部完成后恢复到静默计数，并对上真实释放。固定代理在 D 返回 `recomputed` 时的重试按文本拼接，正式计划送 token IDs，需记录实际重算和重试失败，出现失败的样本按客户端质量规则告警。 [固定代理取消监听](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L706-L724) · [P 选路与回收](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L812-L869) · [D 流与重算](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L928-L1012)

CPU 确认了九个取消时刻，覆盖未下发、P 排队与在途、交接、D 等待与生成。实机诊断在四个实例启用轻量 `--observe-abort`，正式比较的各臂使用一致的观测设置。按客户端调用 ID、代理 UUID、引擎请求 ID 和 P 源请求 ID 关联事件，跨主机保存各自时钟锚点和误差。记录代理 `/healthcheck` 的 `request_num` 在诊断前以及完成事件排空后的值。正式终止臂记录第 150 秒仍在跑的请求数、实际取消时刻、资源归还和恢复基础流的时间，每条在途请求都标归自然完成、实际取消或未确认。 [九个模拟时刻](../results/2026-11-23-basin-window-review/independent-replay.json) · [代理活动计数](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L322-L329) · [vLLM 请求取消](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/engine/async_llm.py#L545-L553) · [调度器终态与延迟释放](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L1674-L1751)

- 客户端记录 `burst_termination`、目标会话、在途请求与未发出的后继，基础流全部成功且取消范围一致。
- 代理记录 `prefill_upstream_cancelled` 或 `decode_upstream_closed`。引擎 `engine_abort_accepted` 证明调度器接收终止，`engine_abort_kv_release` 佐证页释放；P 先正常完成时核对 `engine_source_kv_held` 和 `engine_source_kv_release`。另外核对槽位和在途 Pull/GET 已排空。
- 引擎侧输出 token 只诊断取消时已生成的量。整组只有拿到可靠的实际交付部分 token 才计算取消臂总吞吐，缺失则留空，同一基础流单列。对提前结束本就自然完成的调用按实际终态计。

## 从公开源物化输入

归档的输入和源 ZIP 可按清单指纹恢复。也可从公开 Terminal-Bench 快照重取 80 个任务的 240 个源文件，以同一 Qwen tokenizer 和 chat template 物化 480 条调用。把 tokenizer 目录放在 `data/basin-windows-v1/tokenizer`，其指纹见归档清单。物化时逐条核对 token SHA 和相邻前缀，保存新压缩文件的 SHA。CPU 复算还需要同一清单列出的历史 Qwen EOS 账本。 [公开源版本](https://huggingface.co/datasets/openguardrails/terminal-bench-2.1-deepseek-v4-flash-trajectories/tree/3f5b93ff1c5ea84b2c94cf479d0c942227191b4a) · [归档清单](../results/2026-11-23-basin-window-review/archive.json)

```bash
python -m pip install -e '.[workloads,client]'
python -m pd_kv_transfer_opt.workloads.fetch_trajectories \
  --spec configs/basin-windows-v1/catalog.json --output data/basin-windows-v1/source
python -m zipfile -c data/basin-windows-v1/source.zip data/basin-windows-v1/source/trajectories
python -m pd_kv_transfer_opt.workloads.basin_burst_inputs \
  --catalog configs/basin-windows-v1/catalog.json \
  --source-archive data/basin-windows-v1/source.zip \
  --tokenizer data/basin-windows-v1/tokenizer \
  --output data/basin-windows-v1/inputs.jsonl.gz \
  --receipt data/basin-windows-v1/materialized-inputs.json
```

## 上机、计量和归档

首轮只在两台机器确认可用后启动。核对实际 P0/P1、D0/D1 的 `state.json`、版本与 Master 注册，P/D 各八个 8 GiB segment。两侧继续 TP4/EP4/DP1、双 APC、D 原生重算、自然 EOS、原生路由和原图策略。每臂用相同清洁初态及预热，记录实际预热输出和 Store/APC 状态。用现有 `Shaper` 在 P、D 主机各保存八口原值，按当轮档位为每口设置 12.5 或 25 Gbps。等两端读回都匹配再开始。结束或任一侧设置失败时，两端恢复已保存的各自原值并读回，保存各端恢复回执。 [部署](decode-apc-baseline.md) · [带宽工具](../pd_kv_transfer_opt/experiments/hardware.py)

唯一负载执行入口沿用 `basin_burst_validation`，每臂显式传 `--arm`。将 P 和 D 主机各自用 `Shaper.read()` 取得的八口读回写入两个 JSON。每个文件用 `role` 标识 P 或 D，用 `aggregate_Gbps` 标注该主机目标档位，用 `readback` 保存端口 0–7 的记录。执行入口核对两端角色和档位，每口的 `configured_Mbps` 和 `current_Mbps` 都须匹配。两份原件和 SHA 写入本轮目录，manifest 记录 `symmetric-egress/v1`。冻结输入和源 ZIP 位于 Git 忽略目录 `data/basin-windows-v1`。原交接代码、源账本、长度代理、tokenizer 和来源 SHA 在 `archive.json` 备案。 [入口代码](../pd_kv_transfer_opt/experiments/basin_burst_validation.py) · [外部依赖和指纹](../results/2026-11-23-basin-window-review/archive.json)

```bash
python -m pd_kv_transfer_opt.experiments.basin_burst_validation \
  --plan configs/basin-windows-v1/fivefold.json \
  --workload data/basin-windows-v1/inputs.jsonl.gz --arm burst-drain --verify-only

python -m pd_kv_transfer_opt.experiments.basin_burst_validation \
  --plan "$BASIN_PLAN" --workload data/basin-windows-v1/inputs.jsonl.gz \
  --arm "$BASIN_ARM" --endpoint "$NATIVE_PROXY" --model "$SERVED_MODEL" \
  --run-id "$ARM_ID" --output "$FRESH_RUN_DIR" --cap "$VERIFIED_CAP" \
  --p-shaping-readback "$P_CAP_READBACK" --d-shaping-readback "$D_CAP_READBACK" --timeout 600

python -m pd_kv_transfer_opt.simulator.basin_window_review \
  --known-eos data/basin-windows-v1/provenance/terminal-known-native-eos-through-r78.json \
  --submitted results/2026-11-23-basin-window-review/submitted/simulation-receipt.json \
  --output "$NEW_CPU_AUDIT"
```

每个条件每档单跑一轮，记录实际输出 token、结束原因、成功/取消/失败数、首条计划投放至最后完成的含排空时长和整轮 TPS。用本轮采集的端口计数、服务端队列数据、TTFT 及 `common_base` 做归因。单次配对测得的小差异只记录观测幅度。

两端逐口采集 TX/RX 和上限占用。已有读源事件按同机、D→P、P→D 及来源未知记录 Store GET 字节，同时记 P→D 交接。端口计数含其它流量和协议开销。保存客户端、代理和引擎的时钟锚点与请求 ID 对照，以及初始化、段注册、两端读回和恢复回执。 [事件口径](data.md)

历史单限 D 的 CPU 负载估算为前十轮 **4.52–4.94 小时**、完整 12 轮 **5.38–5.87 小时**，均含每轮预热和排空。双向限速下按首对的加载、缓存复位、真实 EOS 和尾部耗时校准。当前状态为等待可用 NPU。 [排期预算](terminal-throughput-plan.md#机时预算)
