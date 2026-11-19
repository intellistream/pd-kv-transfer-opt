# P/D KV Transfer Optimization

研究 Prefill/Decode 分离推理中的远端 KV 恢复、RDMA 带宽瓶颈和引擎排队，配套实机实验、分阶段 profiling 与 CPU 模拟组件。

硬限速诊断 r14 用双机 2P2D、Qwen3.5-35B-A3B 和两条 223K/239K 热前缀。每请求同步从远端读取约 9.52 GB，固定输出 16 token。D→P 八口发送上限从 100 提升到 200 Gbps，三轮平均吞吐从 16.85 提升到 33.11 token/s。具有自然缓存命中构成与部署生成条件的高压吞吐验收仍待实测。[适用范围](docs/experiment-audit.md) · [实验报告](docs/results.md)

## 项目路径

| 路径 | 内容 |
|---|---|
| [experiments](pd_kv_transfer_opt/experiments/) | 双机启动、代理路由、会话负载、D 端硬件限速和端口采集 |
| [profiling](pd_kv_transfer_opt/profiling/) | P/D 插桩、CANN 导出、CPU 抓栈、逐请求关联及数据清洗 |
| [simulator](pd_kv_transfer_opt/simulator/) | 动态多瓶颈传输、KV 生命周期、布局账本和轨迹条件回放 |
| [configs](configs/) | 部署示例、固定数据源及请求清单、模拟示例 |
| [results](results/) | 有效实测的摘要、可重算事件、SHA-256 清单 |
| [docs](docs/) | 部署、负载、数据、结果和模拟器说明 |

## CPU 快速开始

在仓库根目录使用 Python 3.11 或更新版本。先创建 `runs/` 存放本地输出。以下入口使用标准库。

```text
python -B -m unittest discover -s tests -v
python -B -m pd_kv_transfer_opt.simulator.network_sim --input configs/transport-demo.json --output runs/transport-demo.json
python -B -m pd_kv_transfer_opt.profiling.analyze_full_rdma results/2026-11-18-r14/r14-P.jsonl.gz results/2026-11-18-r14/r14-D.jsonl.gz --output runs/r14-analysis.json
python -B -m pd_kv_transfer_opt.profiling.analyze_full_rdma_profiler runs/r14-analysis.json results/2026-11-18-r14/r14-profiler-P.jsonl.gz results/2026-11-18-r14/r14-profiler-D.jsonl.gz --output runs/r14-profiler.json
```

实机操作见 [2P2D 复现](docs/deployment.md)，输入来源和准备见 [工作负载](docs/workloads.md)。[数据契约](docs/data.md) 说明带宽、吞吐和时间边界。[模拟器](docs/simulator.md) 记录当前实现与校准范围。[AGENTS.md](AGENTS.md) 提供可随研究更新的 AI 开发说明。

研究进度和当前设计见 [七步法](docs/research-seven-steps.md)。证据与细节见 [关键路径归因](docs/critical-path-attribution.md)、[瓶颈定位与补测](docs/bottleneck-and-validation.md)、[问题建模](docs/problem-model.md)、[直接相关工作核查](docs/related-work.md) 和 [优化机制](docs/optimization-research.md)。
