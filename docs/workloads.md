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

[r10 清单](../configs/workload-natural-r10.json) 固定 30 个源任务、170 条请求及其前驱关系。保留完整历史，输入长度约 32.9K–207.7K token，外部会话起点在约 18.23 秒内释放，前驱全流结束后才释放后继。历史 tool gap 缺少实测时间，回放设置为 0。该组的输出预算按源轨迹设置，r10 允许 EOS，r11/r12 固定该组实际输出量用于带宽竞争对照。

需要时用同一个 fetcher 下载完整固定版本，并将 materializer 的 `--spec` 指向 r10 清单。用 `loadgen` 的 `--workload` 参数读取物化文件，`--window-start 0` 配合覆盖全部会话起点的窗口。普通自然回放使用代理的 `--routing native`。r11/r12 配对回放使用[固定路由和输出清单](../configs/replay-natural-fixed-outputs.json)，将该文件同时传给 materializer 的 `--fixed-output` 和代理的 `--routing fixture --route-fixture`，客户端加 `--ignore-eos`。

两类回放分开报告。热前缀小工作集有意让远端读取占据关键路径。自然会话保留任务、上下文和完成依赖的变化。

## r19 高远端恢复的自然后继

[r19 清单](../configs/workload-r19-natural-restore.json) 从上述 170 条请求中选 10 个不同会话的真实后继。筛选依据是 r11/r18 的实测 D→P 描述符字节。各目标都是该会话第一次路由到另一 P 实例，开启 APC 时仍有实测远端恢复。两份历史 trace 中，这 10 条对应的原生描述符分别为 **41.748 GB** 和 **38.057 GB**。保留原始 token IDs、前驱关系和 r11 逐请求路由及输出预算。[原固定路由和输出](../configs/replay-natural-fixed-outputs.json) · [原始 trace 与口径](results.md#r18-原长会话打包分发-trace)

每臂先按真实因果链完整执行所需的 18 条前驱，全部排空后开始计时。10 条目标按清单顺序从 0 到 0.45 秒以 50 ms 间隔提交，共 3,971 个固定输出 token。P0/P1 分别承接 7/3 条，D0/D1 承接 2/8 条。该回放用原数据和依赖构造一次可复现的恢复波次，外部到达由实验施加。同步恢复的实测提交仍可能被 P 的排队平摊，报告客户端 `offered`、`dispatch` 和实际 GET 时间。前驱准备时长单列，完整排空吞吐和 TTFT p99 都只计 10 条目标。每轮样本少，尾指标结合逐轮配对解释。

实测目标提示长度为 67,315–206,695 token。八个实验臂的目标均未命中目的 P 的本地 APC，且全部命中 D 侧可恢复前缀。前驱中延续在原 P 的轮次仍可复用本地 APC。计时结果对应开启 APC 时发生 P 切换的高恢复波次。[r19 实测](results.md#r19-apc-开启的高远端恢复-ab)
