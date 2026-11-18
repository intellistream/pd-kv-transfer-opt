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
