# 双机 2P2D 部署与复现

## 运行域

r14 使用两台各八卡的 910B2、CANN 9.1、Python 3.12、vLLM 0.23.0、vLLM-Ascend 0.23.0.post1 和 `mooncake-transfer-engine-npu==0.3.11.post1`。当前基线沿用固定版环境，两侧使用 Qwen3.5-35B-A3B BF16 和相同分词器。硬件启动需要兼容的 Ascend 驱动、可用的 NPU RoCE 网络及预装的依赖和权重。CPU 复核可独立执行。[vLLM 0fc695f](https://github.com/vllm-project/vllm/tree/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665) · [vLLM-Ascend 5cb98ca](https://github.com/vllm-project/vllm-ascend/tree/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd)

每个角色两个 vLLM 实例，每实例四卡，Attention TP4、Expert EP4、DP1。P 使用平台默认图模式，捕获尺寸扩至 4096。D 使用 `FULL_DECODE_ONLY`。两侧分块预算 4096，最大上下文 262,144，`mamba-cache-mode=align`。`MultiConnector` 组合 `MooncakeConnectorV1` 的 P→D 交接和 `AscendStoreConnector` 的同步 Store 恢复。

将双侧 APC 模板复制到 `configs/ascend.local.json`，设置两侧地址、模型路径和 CPU 亲和性。默认布局为 P/D 每个模型 worker 各 8 GiB，每侧各 64 GiB，保留双侧就近分配和每个 worker 的 32 MiB 本地 staging。每个实例使用自己的 API 端口号作为本地 Store lookup IPC 标识。先用 `serve --dry-run` 核对四个实例，启动后对照两侧 `state.json` 与 Master 的实际注册。[当前双侧 APC 基线](decode-apc-baseline.md) · [配置模板](../configs/ascend-2p2d-decode-apc.example.json)

常规回放的 `proxy` 默认使用上游调度并保留上游 HTTP 连接复用。`--routing fixture --route-fixture` 用于重放已记录路由，`--no-upstream-keepalive` 用于复现关闭复用的诊断。`serve --observe` 开启逐方法事件与按需 CANN 入口。整轮吞吐计时可用默认启动模式，详细 profiling 用同配置的独立轮。

## r14 硬限速诊断复现

历史 r14 关闭两侧 APC，使用 P 贡献 0、D 每个模型 worker 贡献 16 GiB 的单侧池。原样复现应在归档匹配的代码版本上运行。当前代码可使用同一诊断协议验证双侧池，结果记录该布局身份。按所选版本将示例另存为 `configs/ascend-r14.local.json`，把 P、D 的 `enable_prefix_caching` 都设为 `false`。从仓库根目录预览命令，再在对应主机分别移除 `--dry-run` 启动。[历史 r14 清单](../results/2026-11-18-r14/manifest.json)

```bash
python -m pd_kv_transfer_opt.experiments.serve --config configs/ascend-r14.local.json --role P --run-id r14-repro --run-dir runs/r14-P --observe --dry-run
python -m pd_kv_transfer_opt.experiments.serve --config configs/ascend-r14.local.json --role D --run-id r14-repro --run-dir runs/r14-D --observe --dry-run
```

D 角色启动时会起 Mooncake master。两边控制器记录 `state.json` 和每个实例的日志。在 D 侧另起代理。参数指向与安装版本一致的官方示例脚本。

```bash
python -m pd_kv_transfer_opt.experiments.proxy --config configs/ascend-r14.local.json --run-dir runs/r14-D --routing r14 --upstream /path/to/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
```

按 [工作负载说明](workloads.md) 物化输入。四个 API 健康后，从 D 侧运行实验。省略 `--apply` 可预览顺序。加入 `--profiles` 会在全部计时轮之后采三个诊断窗口。根据驱动权限使用 `--sudo` 或管理员环境，`--hccn-tool` 可以覆盖工具路径。

```bash
python -m pd_kv_transfer_opt.experiments.run_suite --config configs/ascend-r14.local.json --run-dir runs/r14-D --workload data/r14-pair.jsonl --sudo --profiles --apply
```

suite 在 D 主机对八个端口读回原值、设置每档上限并在结束或可处理的中断后恢复。保留 `hard-cap-suite/shaping-restore.json`，必要时用同一 `hccn_tool -i <port> -shaping -g` 从驱动复核。

## 同头 KV 读取原型

在 P 启动命令上增加 `--same-head-kv`，并使用 r14 的 TP4、两 KV head、同步 non-layerwise Store 和关闭本地 APC 的设置。D、代理与 P→D 交接保持同一配置。P 运行目录中的 `same-head-mode` 控制该原型。首次启动默认基线。用 `baseline` 或 `enabled` 在相同 P 进程中回切。等上一轮请求排空后用临时文件和原子改名更新模式文件。

`verify` 模式在同头 HCCL 分发后抽样，将结果同一次独立原生 GET 对比。该模式产生额外 GET，另跑 `enabled` 检查纯原型路径的输出，再开始性能计时。r15 正式轮只用 `baseline` 和 `enabled`，从每轮前后的 D 侧端口差值计算 TX/RX。P 的 `same_head_local_copy` 记录本机收到的逻辑字节、分发墙钟、设备流计时和配对等待。配对等待可能包含对端剩余 GET。复制流计时包含两次 HCCL 提交之间的空档。原型只合并 Attention，SSM/conv 保留原有读取。

r16 用 `profile` 运行直接分发并记录页连续性。`same_head_copy_layout` 分别记录条带与张量列边界、只在本 rank 断开的地址、只在对端断开的地址、两端均断开的地址，以及可在两端合并的区间。正式 r16 比较 `enabled` 与 `packed`。后者在每个 P rank 复用 128 MiB NPU 缓冲，用固定版本的批量内存复制算子打包，执行 HCCL 广播后散写到目标页。用 `verify-packed` 做独立 GET 抽样，并用 `packed` 验证纯原型前向。上述两种路径复用同一个远端选读方案，等待本机分发完成后才交给模型。

### r19 的 APC 恢复实验

r19 的 P 侧使用归档中的 [实际 APC 适配器](../results/2026-11-18-r19-natural-ab/runtime/same_head_kv_r19.py)。它调用 r16 的同头读与打包原语，在 APC 后核对原生目的描述符。按归档的模块映射把适配器、`same_head_kv_r16` 和观测器放在同一个 Python 加载目录，保持观测器的导入名称一致。用 `same_head_kv_r19.InstallFilter` 替换 r16 的过滤器，在 P 保留原有同步 Store 及分块观测，开启 P APC。通过 `ISSUE23_SAME_HEAD_MODE` 指定 P 上的模式文件，排空后以原子替换切换 `baseline` 和 `packed`。通用 `serve --same-head-kv` 仍启动 r16 的无 APC 实验路径。 [模块映射与原始指纹](../results/2026-11-18-r19-natural-ab/runtime/provenance.json)

使用 [r19 清单](workloads.md#r19-高远端恢复的自然后继) 切出准备和计时输入，并按同一个固定映射给代理设置两阶段路由。每个实验臂使用独立 `cache_salt`，该臂的全部前驱和后继用相同值。使用专用空 Store 运行前驱，在原档等待客户流和 Store PUT 提交排空。限速只应用于计时后继，100/200 Gbps 对应 D 侧八口各 12.5/25 Gbps。正式对照前用两个不同 P 路径的样本完成首个打包校验。关闭 EOS 提前结束，保留逐条原输出预算，目标的提交间隔为 50 ms。

### r20 的同批分发重叠实验

沿用 r19 部署、原始数据和符号路由。P 过滤器换成 [实测 r20 适配器](../results/2026-11-19-r20-packed-overlap/runtime/same_head_kv_r20.py) 的 `same_head_kv_r20.InstallFilter`，在同一 Python 加载目录提供 [packed_copy_pipeline.py](../results/2026-11-19-r20-packed-overlap/runtime/packed_copy_pipeline.py)。正式计时在 `packed` 和 `async-packed` 之间回切，前驱准备仍使用 `baseline`。两种打包模式分别对两个 P 实例做一次独立原生 GET 对照抽样，校验读位于计时外。始终在同一 NPU 流使用原本的 128 MiB wire buffer，计时中只有一个分发可以在途，目标区间有别名时沿用同步路径。

### r21 的跨批读取实验

沿用 r20 负载及路由。P 在同一加载目录配置 `same_head_kv_r21c.InstallFilter`、`r21_prefetch_c.py`、`r21_stage.py` 及 r20 下层观测与同头打包模块。保持 P 本地 APC 和同步 Store 配置。前驱准备使用 `baseline`，计时切到 `packed`、`prepare-only` 或 `early-get`。C 只把单恢复批内确定的 `video-processing:28` 提前读到 staging，保持原有 KV 发布位置。准备阶段先做原生对照小样，核对正式期物理调用和字节一致。该实验使用固定的 P0 和目标请求长度进行诊断选择。[实测源文件指纹](../results/2026-11-19-r21-cross-batch/runtime/provenance.json) · [r21 结果与时间定义](results.md#r21-固定字节的跨批提前读)

### r24 的 P 请求接纳实验

在 r21 同步打包日志过滤器安装完成后调用 `prefill_admission_observability.install_admission()`。将仓库包路径加入 P 的现有 Python 搜索路径，保留容器的 Ascend 模块路径。P 设置 `ISSUE23_P_ADMISSION_OBSERVABILITY=1` 和 `ISSUE23_P_ADMISSION_MODE_FILE`，后者指向 P 同主机控制文件。原分块事件继续记录调度器接纳、派发和设备事件。新事件记录 ASGI 请求体、API render、IPC `ADD` 发送和接收、引擎输入队列入取、worker future 等待与结果更新。同一 P 主机的进程通过单调时钟对齐。

保持 r21 原组的准备和固定路由回放。前驱置 `baseline` 与 `blocked`，计时两臂都置同步 `packed`。A 使用 `blocked`，B 使用 `drain`。排空后用原子替换更新模式，D 进程和原生配置持续复用。B 只在既有的本地 worker 响应等待中由 EngineCore 主线程处理可接纳的队首 `ADD`，实际批派发、结果更新顺序和批队列深度沿用原实现。以 `p_admission_early_add` 区分实际处理与仅开启模式。按八口读回和恢复回执核对 D 的 100/200 Gbps。

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_live_prefill_admission \
  runs/r24-P.jsonl.gz runs/r24-D.jsonl.gz runs/r24-summary.json \
  --workload configs/workload-r19-natural-restore.json
```

### r26 的原生高压实验

使用原生 `rdma_full_observability.InstallFilter` 启动两侧。两台机器的可编辑安装指向干净的上游 `0fc695f` 和 `5cb98ca`，安装元数据分别为 `vllm 0.23.0+empty`、`vllm-ascend 0.23.0`，Mooncake NPU wheel 为 `0.3.11.post1`。模型参数复用 r25 的已核对配置，包括 P APC 开启、4096 token 分块、D APC 关闭和 `FULL_DECODE_ONLY`。固定路由代理只用于重放，P worker 调用原生同步 Store 方法，每次 GET 调用并发度为 1。

[高压计划](../configs/workload-r26-native-pressure.json) 从 r25 清单展开。四个臂各用独立缓存命名域，从完整提示和源输出计数准备直接前驱，等待 Store 提交后限速并回放目标。D 的 100/200 Gbps 分别由八口各 12.5/25 Gbps 的读回确认。正式轮保存主机阶段事件、短间隔端口计数、D 引擎状态和客户端输出计数，CANN 设备 profiler 保持关闭。每臂退出后恢复该轮打开时的端口设置。[负载和指标](results.md#r26-原生-baseline-的高压带宽对照)

### r27c 的持续高压实验

先尝试为独立命名域扩大原生 D 主机池。r27 的 1,152 GiB 和 r27b 的 512 GiB 均能挂载，随后都在不限速预热发生 HCCL 建连错误，两次均在应用带宽档位前终止。两次控制器均已停止，D 端口读回每口 200 Gbps。这两次在本节保留故障记录。r27c 使用 r26 的完整 D 配置及每 rank 16 GiB、共 128 GiB 的原生 D 主机池。P 使用原有模型参数，采用原生开关关闭本地 APC。每档独立准备完整前驱并等 Store 提交。固定高压到达期间持续复用该臂的外部前缀，正式轮记录同步 GET、短采样端口计数、D 工作量和客户端实际输出。[持续高压负载](workloads.md#r27c-持续投放的原生高压负载)

## 采样与导出

启动至 ready 后可在 P 侧运行 `python -m pd_kv_transfer_opt.profiling.cpu_snapshots --run-dir runs/r14-P`。它在诊断窗口等待 P0 原生 GET 开始后，用 `py-spy` 同时抓 EngineCore 和 worker。工具需要本机已安装 py-spy 及相应的进程采样权限。

可用 `python -m pd_kv_transfer_opt.experiments.collect --help` 采 NPU、Prometheus 与 HCCN。输出到各侧运行目录的 `host-telemetry` 或 `fabric-telemetry`，便于事件导出识别。suite 自身始终保存各轮 D 端口计数。

记录限速恢复后，停止代理和两个控制器。对控制器发送 Ctrl-C 或在各自运行目录写入 `STOP`。随后在对应主机导出事件及 CANN。离线 CANN 导出使用该环境的 torch_npu profiler，耗时取决于捕获大小。

```bash
python -m pd_kv_transfer_opt.profiling.export_events --run-dir runs/r14-P
python -m pd_kv_transfer_opt.profiling.export_cann --run-dir runs/r14-P
```

D 侧换成 `runs/r14-D`。把两侧压缩导出放到统一分析目录，然后用 [README](../README.md) 的离线入口关联。发布时按 [事件契约](data.md) 清洗。

仓库迁移已完成 CPU 校验、生成命令检查和实测事件重算。具体站点的硬件启动由下一次授权机时验证。历史 r14 的 CANN API 停止错误和恢复方式见 [结果](results.md)。
