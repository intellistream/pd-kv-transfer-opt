# 原生 2P2D 上机执行

从仓库根目录运行。批次入口使用已冻结的输入和双端出站限速。部署、实验次序与机时见 [实验计划](terminal-throughput-plan.md)。

## 冻结配置

| 项目 | 上机取值 |
| --- | --- |
| 硬件与并行 | ascend-112 跑 P0/P1，ascend-180 跑 D0/D1。两台各 8 张 910B2，每台按卡 0–3、4–7 分组。Qwen3.5-35B-A3B BF16，Attention TP4、Expert EP4、DP1。 |
| 软件 | vLLM `0fc695fc`，vLLM-Ascend `5cb98caa`，Mooncake NPU `0.3.11.post1`。部署前核对完整提交、模型配置、分片索引、tokenizer 以及两端实际分发文件。 |
| 调度和图 | 双侧本地 APC，D 原生重算调度器。P 原生图策略，配置捕获尺寸 1/2/4/8/16/1024/2048/3072/4096，D `FULL_DECODE_ONLY`，配置 1/2/4/8/16。eager 关闭。两侧每实例 16 槽、4,096 批 token、262,144 最大上下文、0.8 显存利用率，KV block 128，Mamba `align`。 |
| 交接和 Store | `MooncakeConnectorV1` 负责 P→D 交接，`AscendStoreConnector` 负责共享池。同步 GET，并发 1，D 可以 PUT，加载失败策略为 `fail`。P/D 每个模型 rank 各 8 GiB，每侧 64 GiB，总 128 GiB。两侧启用就近分配，staging 每模型 rank 32 MiB，Master 在 D。加载时显式锁定脚本生成的 Master 和容量。 |
| 代理与生成 | 每实例 4 个 API 进程，代理 8 个 worker，保持原生选路和上游 keepalive。客户端从 D 侧 loopback 投放。`temperature=0`，自然 EOS，每请求保留原输出上限。正式轮走原生执行路径，同头打包、投机解码和自定义调度均关闭。 |
| 带宽与计量 | P、D 同时执行同一档的各自出站限额。每个主机的八口合计 100/200 Gbps，对应每口 12.5/25 Gbps。200 Gbps 是配对档，按两端热点口和供给等待验证余量。 |

Store 的就近设置用于 PUT 分配。GET 的实际来源按读出的证据记账。r43/r69 使用紧凑 JSON，延续历史请求体的编码方式，原生代理以请求体大小计算路由分数。 [固定版 Store 配置](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L181-L189) · [固定版代理打分](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L843-L846)

## 已备好的输入和入口

| 批次 | 负载 | 执行参数 |
| --- | --- | --- |
| r43 | 48 会话，254 条请求，双端 100→200→200→100 Gbps | `--group r43` |
| r69 | 53 个双轮窗口，106 条请求，同一正反序 | `--group r69` |
| Basin 自然完成 | 稳态、+33%、+100%、五倍突发和短会话自然排空，每条件两档，共 10 轮 | `--group basin` |
| Basin 取消短测 | 4 条预热和 27 条正式计划调用，200 Gbps，收集代理和引擎取消证据 | `--group cancel-check` |
| Basin 短会话终止 | 每档一轮，在 150 秒取消剩余突发会话，现场短测通过后执行 | `--group basin-termination` |
| r43 设备采集 | 双端 200 Gbps，分别采 D0、D1 的四个 rank，目标批宽 1–4 和 9–16 | `--group r43-profile` |

`--group formal` 依次执行前三行的 18 轮。每轮从四个新引擎和空 Store 启动。Basin 在正式计时前走同一份 32 请求预热计划。正式臂沿用轻量计数和事件记录，CANN 设备采集在单独一轮启动。r43/r69 使用原始 token、到达、依赖和输出预算，Basin 使用冻结的逐轮工具间隔。所有正式臂均保留自然 EOS 和原生路由。

r43 根到达窗口为 0–92.5 秒，r69 使用冻结的集中双波诊断。Basin 每轮 304–448 个正式计划调用，分批独立投放新会话，保留相邻轮的工具间隔和前驱依赖。三组对照都以本组内同一负载比较带宽。

本机准备好的 r43/r69 原始输入和核对收据在 `data/native-npu-inputs/`，Basin 输入在 `data/basin-windows-v1/inputs.jsonl.gz`。如需重建 r43/r69，可用 `python -m pd_kv_transfer_opt.workloads.native_replay_inputs`，参数指向已归档的 Terminal-Bench 源 ZIP 和 Qwen tokenizer。r43 整份输入与历史工作量 SHA256 一致，r69 用冻结的逐请求 token 指纹校验。原始输入和机器配置均在 Git 忽略目录或 `*.local.json`。

## 从控制端启动

`configs/native-npu-site.example.json` 可直接做 CPU dry-run，打印 18 轮及输入指纹。

```powershell
python -B -m pd_kv_transfer_opt.experiments.native_npu_suite --site configs/native-npu-site.example.json
```

现场配置 `configs/native-npu-site.local.json` 从历史部署收据生成。获准使用两台机器后核对对应容器名、挂载路径和驱动工具位置。`host_repo` 和 `runtime_repo` 需指向同一套挂载内容，`runtime_prefix` 使用已备好的固定版容器。首批仅运行 r43 的四轮。用同一提交工作树离线打包，再在两端空闲时执行入口。入口分发并核对文件、固定版本和模型指纹，然后开始第一轮。

```powershell
python -B -m pd_kv_transfer_opt.experiments.native_npu_suite --site configs/native-npu-site.local.json --group r43 --prepare-bundles data/native-npu-stage-r43
python -B -m pd_kv_transfer_opt.experiments.native_npu_suite --site configs/native-npu-site.local.json --group r43 --stage-bundle data/native-npu-stage-r43 --execute --run-id native-r43-first --output runs/native-r43-first
```

打包范围由 `--group` 决定。已有分发包覆盖的批次可用新的 `--run-id` 和 `--output` 继续执行，省略 `--stage-bundle`。首次执行后续组或更新代码、部署、输入时，用对应批次和新目录离线打包。容器或挂载变化时更新本机现场配置。

入口先起 D 的 Store Master，再启动两侧引擎并核对 `state.json`。新池必须报告零 key，并实际注册 P/D 各八个 8 GiB segment。原生代理就绪后，P/D 各自保存八口原值、写入当轮限额并逐口读回。两侧就绪才从 D 容器的 loopback 启动客户端。退出时恢复两端原值并验证读回，然后分别收回日志与采集包。先核对首轮实际输出、APC 和远端读源覆盖，再解释后续对照。

## 采集与收尾

- 自然完成的整轮 TPS 读取 `summary.json` 的 `finite_cohort`，按正式请求实际输出 token 和首条计划投放至最后完成的时长计算。检查客户端计数、自然 EOS、实际路由和压尾请求。两端的逐口 TX/RX 与应用层 KV 事件分别存档。Store GET 的对象来源缺失时归为来源未知。
- 固定版原生代理在收到 `recomputed` 时按文本拼接重试，遇到 token ID 输入可能中断。任一正式请求失败时整轮吞吐记为无效，并用客户端终态和代理原始日志查明原因。 [固定版重试路径](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L965-L983)
- 短测查看 `proxy-before.json`、`proxy-after.json`、客户端终态、代理关闭事件和引擎 abort/KV 释放。按 [取消诊断口径](basin-windows-v1-review.md#取消臂的现场核对) 核实 P 在途、D 等待 KV、D 生成三个阶段。现场通过后，把短测 `run_id`、`covered_stages` 数组 `['prefill', 'decode_kv_wait', 'decode_generation']` 和 `server_abort_and_resource_release_verified: true` 写入复核 JSON，用 `--termination-qualification` 传给终止批次。活动请求和资源无法核实的阶段留作未确认。部分输出 token 无可靠计数时，终止臂总 TPS 留空。
- `cancel-check` 用 [短测计划](../configs/basin-smoke-short.json)。已有输入为 `data/basin-smoke-inputs.jsonl.gz`。如需从公开源重建，用 [短测来源清单](../configs/basin-smoke-catalog.json) 下载到独立目录，再按 Basin 的源归档和物化流程生成短测输入。 [物化流程](basin-windows-v1-review.md#从公开源物化输入)
- `r43-profile` 在 D 一侧的独立进程采设备窗口，收集目录和缺失窗口记录在 `decode-windows.json`。复用现有 [离线四 rank 分析脚本](r43-decode-profiling.md) 做计算、通信和步间归因。
- 每臂归档 `P.tar.gz`、`D.tar.gz`、逐口限速与恢复、就绪状态和客户端摘要。设备与原始日志保留在 `runs/`。核对有效归档后按 [数据清理口径](data.md#归档与清理) 处理远端和本地临时数据。
