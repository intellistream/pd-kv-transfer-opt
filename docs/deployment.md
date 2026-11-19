# 双机 2P2D 部署与复现

## 运行域

r14 使用两台各八卡的 910B2、CANN 9.1、Python 3.12、vLLM 0.23.0、vLLM-Ascend 0.23.0.post1 和 `mooncake-transfer-engine-npu==0.3.11.post1`。运行源固定在 [vLLM 0fc695f](https://github.com/vllm-project/vllm/tree/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665) 和 [vLLM-Ascend 5cb98ca](https://github.com/vllm-project/vllm-ascend/tree/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd)。硬件启动使用兼容的 Ascend 环境、NPU 设备、配套驱动、可用的 NPU RoCE 网络和已经安装的 wheel。模型为 Qwen3.5-35B-A3B BF16，两侧使用一致的模型和分词器。依赖和权重预先装入运行环境；CPU 复核可独立执行。

每个角色两个 vLLM 实例，每实例四卡，attention TP4、expert EP4、DP1。P 使用平台默认图模式，捕获尺寸扩至 4096。D 使用 `FULL_DECODE_ONLY`。两侧分块预算 4096，最大上下文 262,144，`mamba-cache-mode=align`。r14 关闭两侧本地 APC。D 侧八个 worker 每个贡献 16 GiB 池容量，P 贡献零，D 保存 KV。P 使用同步 GET。原生 P→D 交接和跨机 Store 恢复分别经过 `MooncakeConnectorV1` 和 `AscendStoreConnector`，两者用 `MultiConnector` 组合。

复制 [配置示例](../configs/ascend-2p2d.example.json) 到 `configs/ascend.local.json` 后设置两侧地址、模型路径和需要的 CPU 亲和性。示例 IP 是文档地址。P 与 D 在各自启动环境内使用相同的已编辑配置。下面的命令从仓库根目录执行。

## 启动和回放

先生成命令预览。配置正确后，在两台主机分别执行对应角色的启动命令，移除 `--dry-run`。

```bash
python -m pd_kv_transfer_opt.experiments.serve --config configs/ascend.local.json --role P --run-id r14-repro --run-dir runs/r14-P --dry-run
python -m pd_kv_transfer_opt.experiments.serve --config configs/ascend.local.json --role D --run-id r14-repro --run-dir runs/r14-D --dry-run
```

D 角色启动时会起 Mooncake master。两边控制器记录 `state.json` 和每个实例的日志。在 D 侧另起代理。参数指向与安装版本一致的官方示例脚本。

```bash
python -m pd_kv_transfer_opt.experiments.proxy --config configs/ascend.local.json --run-dir runs/r14-D --upstream /path/to/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
```

按 [工作负载说明](workloads.md) 物化输入。四个 API 健康后，从 D 侧运行实验。省略 `--apply` 可预览顺序。加入 `--profiles` 会在全部计时轮之后采三个诊断窗口。根据驱动权限使用 `--sudo` 或管理员环境，`--hccn-tool` 可以覆盖工具路径。

```bash
python -m pd_kv_transfer_opt.experiments.run_suite --config configs/ascend.local.json --run-dir runs/r14-D --workload data/r14-pair.jsonl --sudo --profiles --apply
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
