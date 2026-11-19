# 工作负载与数据源

数据源是 `openguardrails/terminal-bench-2.1-deepseek-v4-flash-trajectories`，固定 revision `3f5b93ff1c5ea84b2c94cf479d0c942227191b4a`。从 ATIF 的可见历史还原 system、user、assistant、已观察的工具调用与返回，使用 Qwen3.5-35B-A3B 的模板开启 thinking 并加生成提示。历史里的命令与工具结果只作为文本输入。[固定数据版本](https://huggingface.co/datasets/openguardrails/terminal-bench-2.1-deepseek-v4-flash-trajectories/tree/3f5b93ff1c5ea84b2c94cf479d0c942227191b4a)

## 硬限速诊断

[固定输入清单](../configs/workload-r14.json) 记录源文件、分词器、原始 JSONL 与 token IDs 的哈希。两个完整前缀分别是 `regex-chess:120` 的 223,256 token 与 `winning-avg-corewars:145` 的 238,600 token。源任务按长度和 1024 token 对齐余量筛出，保留完整前缀，每条请求固定 16 输出 token。两条前缀各预热一次，交替固定到 P0/D0 与 P1/D1。

- 高负载采用固定间隔 16 请求/s，每轮 96 条，完整排空。四档为 100、200、400 Gbps 和原始八口各 200 Gbps，三个区组轮换顺序。
- 低负载采用 0.5 请求/s，每轮 12 条，在 100/400 Gbps 各重复两次。
- 独立采样在 100、400 Gbps 和原始设置各重放 24 条请求，P0/D0 每实例采四个 rank，离线导出两侧 rank 0。

```bash
python -m pd_kv_transfer_opt.workloads.fetch_trajectories --spec configs/workload-r14.json --output data/terminalbench
python -m pd_kv_transfer_opt.workloads.materialize --spec configs/workload-r14.json --traces data/terminalbench --model /models/Qwen3.5-35B-A3B --output data/r14-pair.jsonl
```

物化程序逐条核对 token 数和 uint32 小端序哈希。重新物化的 JSONL 包装字段可能不同，保留内容指纹与新文件哈希。

## 自然长会话

[r10 清单](../configs/workload-natural-r10.json) 固定 30 个源任务、170 条请求及窗口内的前驱关系。每条保留完整输入历史，长度约 32.9K–207.7K token，历史 assistant 和 tool 文本来自原轨迹。会话从选中的长上下文轮次开始。起点按实验生成的到达序列在约 18.23 秒内释放，前驱全流结束后才释放后继。原工具耗时缺少实测，回放间隔设为 0。输出预算按源长度限制在 128–1,024 token，125 条的上限为 1,024。r10 允许 EOS，实测 39 条以 `length` 停止，实际输出达到 1,024 token 的共有 31 条。r11/r12 固定的是这组带上限的 Qwen 实测输出量，用于控制带宽竞争对照的计算量。

需要时用同一个 fetcher 下载完整固定版本，并将 materializer 的 `--spec` 指向 r10 清单。用 `loadgen` 的 `--workload` 参数读取物化文件，`--window-start 0` 配合覆盖全部会话起点的窗口。普通自然回放使用代理的 `--routing native`。r11/r12 配对回放使用[固定路由和输出清单](../configs/replay-natural-fixed-outputs.json)，将该文件同时传给 materializer 的 `--fixed-output` 和代理的 `--routing fixture --route-fixture`，客户端加 `--ignore-eos`。

两类回放分开报告。热前缀小工作集有意让远端读取占据关键路径。自然会话保留任务、上下文和完成依赖的变化。

部署验收回放从原轨迹增加独立会话，用外部会话起点维持高压。配对保持会话起点、工具间隔、路由策略与两侧缓存配置一致，后继到达随前驱实际完成。P 开启 APC，保留冷输入、同 P 的本地复用和换 P 后的自然缺页。逐请求输出长度按原数据分布核对，实际服务的采样参数、输出上限和 EOS 停止条件一起记录。需要固定源输出长度时单列等工作量对照。从首条正式请求开始计时，同轮产生的缓存按顺序使用，记录实际 P 本地命中 token、Store 匹配及实际读取字节、未命中计算 token 和 P→D 交接字节。D 本地 APC 的适用性和实际命中按目标部署固定。原生 r11 的 170 条中有 48 条 D→P 实读，可用作历史命中构成的参照。[部署代表性复查](experiment-audit.md) · [r11/r18 回放](research-seven-steps.md#r18-原长会话的-trace-判据)

## r19 高远端恢复的自然后继

[r19 清单](../configs/workload-r19-natural-restore.json) 从上述 170 条请求中选 10 个不同会话的真实后继。筛选依据是 r11/r18 的实测 D→P 描述符字节。各目标都是该会话第一次路由到另一 P 实例，开启 APC 时仍有实测远端恢复。两份历史 trace 中，这 10 条对应的原生描述符分别为 **41.748 GB** 和 **38.057 GB**。保留原始 token IDs、前驱关系和 r11 逐请求路由及输出预算。[原固定路由和输出](../configs/replay-natural-fixed-outputs.json) · [原始 trace 与口径](results.md#r18-原长会话打包分发-trace)

每臂先按真实因果链完整执行所需的 18 条前驱，全部排空后开始计时。10 条目标按清单顺序从 0 到 0.45 秒以 50 ms 间隔提交，共 3,971 个固定输出 token。P0/P1 分别承接 7/3 条，D0/D1 承接 2/8 条。该回放用原数据和依赖构造一次可复现的恢复波次，外部到达由实验施加。同步恢复的实测提交仍可能被 P 的排队平摊，报告客户端 `offered`、`dispatch` 和实际 GET 时间。前驱准备时长单列，完整排空吞吐和 TTFT p99 都只计 10 条目标。每轮样本少，尾指标结合逐轮配对解释。

实测目标提示长度为 67,315–206,695 token。八个实验臂的目标均未命中目的 P 的本地 APC，且全部命中 D 侧可恢复前缀。前驱中延续在原 P 的轮次仍可复用本地 APC。计时结果对应开启 APC 时发生 P 切换的高恢复波次。[r19 实测](results.md#r19-apc-开启的高远端恢复-ab)

## r25 原输出长度的连续多波

从同一物化清单中，每个会话取 r18 同步打包 GET 载荷最大且至少为 700 MB 的跨 P 后继。保留每条记录完整的提示 token、立即前驱及 r11 固定 P/D 路由，得到 22 个不同会话。每臂先重放 22 条立即前驱的完整提示历史，待原始输出结束和 Store 提交稳定后释放后继。每臂使用独立缓存命名域，P 本地 APC 保持开启。

四臂共用明确到达时刻。每 3.5 秒一波，波内两条间隔 50 ms，共 11 波，最后一条在 35.05 秒释放。目标按源数据中的 `source_completion_tokens` 生成，共 59,906 token，最长一条 9,919 token。22 条前驱同样保留源输出计数，共 78,207 token。该读法将源记录的 token 数作为 Qwen 输出长度，设置 `ignore_eos` 保证两档输出可比。默认测量区间为 10.5–35 秒，另保留到达末段 21–35 秒和到达后的 35–70 秒。[逐请求计划与输入指纹](../configs/workload-r25-multiwave.json)

## r26 原生 baseline 的高到达率

从 r25 的冻结清单继承 22 个高恢复的跨 P 后继、直接前驱、输入指纹、P/D 路由和逐请求输出计数。P APC 开启，目标实测本地命中为零。原生各 rank 同步直读使用 11 波，每 0.25 秒一波，波内两条仍间隔 50 ms。最后一次投放为 2.55 秒。100/200 Gbps 使用同一个高到达率计划，正式顺序为 100、200、200、100。固定分析窗为 1–6、0–10、10–35、35–70 秒。另报告从首次到达到最后一次 P 返回及最后一次 D KV 就绪的交付速率。该子组刻画换 P 后碰到已知热 Store 的负载。[高压重放清单](../configs/workload-r26-native-pressure.json)

## r27c 持续投放的原生高压负载

沿用 r26 的 22 个目标及其直接前驱，保留输入指纹、源输出计数和全部 P/D 路由。每臂独立准备一遍 22 条直接前驱，保留前驱完整输入和 78,207 个源输出 token，并等待 Store 提交。目标复用该臂已准备的外部前缀，P 侧用原生参数关闭本地 APC，使各轮重复仍走跨机读取。正式阶段同一批目标连续回放 12 遍，每 0.25 秒投放两条，波内间隔 50 ms，共 132 波，最后一条在 32.80 秒投放。每臂 264 条计时目标，共 718,872 个输出 token。[负载展开规则](../configs/workload-r27-native-sustained.json)

该轮为 D→P 回读压力诊断。目标轮的 P/D 本地 APC 均关闭，已有前缀驻留在 D 主机内存的 Store。两档各有 264 条实际从 Store 恢复，后 11 遍在两档均读 722.322 GB，占全轮约 91.9%。这个比例描述重复投放产生的流量，开启 APC 后可省的字节需另测。D→P GET 与 P→D 交接分别计量。固定主测窗口为 18–32 秒，期间持续以约 8 请求/s 到达。输出平台按七个两秒子窗判定，要求子窗 token/s 的变异系数和主窗两半平均速率的相对差各不超过 10%。另设 0–10 秒、10–18 秒和停止到达后 34–48 秒的阶段窗口。到达计划对应约 21,784 输出 token/s 的请求需求。排空从最后一次计划到达开始另计，在途队列的增长用于解释输出速率的平台性质。[分析口径](../pd_kv_transfer_opt/profiling/analyze_multiwave.py)
