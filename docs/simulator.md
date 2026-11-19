# CPU 模拟器

仓库包含可独立运行的模拟组件，模型工作由真实的阶段锚点和可重放的因果关系驱动。

| 模块 | 当前能力 |
|---|---|
| [network_sim.py](../pd_kv_transfer_opt/simulator/network_sim.py) | 离散事件、动态多资源公平共享、方向独立或显式端点共享、FIFO 并发限制与字节守恒 |
| [lifecycle.py](../pd_kv_transfer_opt/simulator/lifecycle.py) | 物理池分配、在途容量占用、源 pin、完整 hybrid bundle 的可见性与 lease 释放 |
| [geometry.py](../pd_kv_transfer_opt/simulator/geometry.py) | 从 Qwen 配置推导 attention 与 recurrent 状态的逻辑尺寸和聚合存储量 |
| [trace_replay.py](../pd_kv_transfer_opt/simulator/trace_replay.py) | 继承的四 FIFO 条件回放，按前驱的完整结束时刻释放会话后继 |

从 [传输示例](../configs/transport-demo.json) 运行 `python -m pd_kv_transfer_opt.simulator.network_sim --input configs/transport-demo.json --output runs/transport-demo.json`。示例使用人工容量和到达。旧四 FIFO 入口读取 measurements、protocol 和 cost-model，工作负载、已接纳缓存和路径固定，用于重构时序与灵敏度检查。CPU 用例给出最小调用示例。

## 使用实测锚点

r14 的 [逐轮样本](../results/2026-11-18-r14/rounds.csv) 和 [逐阶段摘要](../results/2026-11-18-r14/summary.json) 可用于固定条件标定。四档每请求约 9.52 GB 远端有效载荷。D→P 的八口限速和 MAC 计数应单独输入，P→D 方向另记。当同步 GET 完成后，同一 worker 才能继续 forward。EngineCore 批队列满或没有可发批时等 worker 结果，会连带影响收件。将新 token 计算和本地前缀复用留为各自的成本与事件。

模拟器现阶段提供条件回放和分组件模型。闭环 910B 服务预测需接入调度、缓存与传输事件，使用独立实测留出验证。校准输入可使用真实字节、到达和组件服务成本。真实排队、最终命中和未来完成时刻用于核对输出。图执行、TP/EP、模型与后端变化使用新的兼容域。

闭环扩展的事件依赖、缓存可见性与校准口径见 [问题模型](problem-model.md)，与 ServeTwin 等系统的对照见 [相关工作](optimization-research.md)。

## 可复用的相关工作

- ServeTwin 提供有状态分离推理、D 侧容量准入及 ragged batch 成本的设计参照。[原文](https://arxiv.org/abs/2610.02732)
- Frontier 的 P/D、MoE 和多网络 backend 可作为后续集成底座。前期核对的源码点为 `db5a85b63fb42436da2e3c21cbb42cc53b22a842`。[代码与许可](https://github.com/NetX-lab/Frontier/tree/db5a85b63fb42436da2e3c21cbb42cc53b22a842)
- Simthesizer 与 SimServing 分别提供动态组件 DAG 和复用原生执行语义的参照，用于拆开状态生命周期与框架事件漂移。[Simthesizer](https://arxiv.org/abs/2608.24650) · [SimServing](https://conf.researchr.org/details/ase-2026/ase-2026-research-track/32/SimServing-Native-Execution-Simulation-for-Evolution-Resilient-LLM-Serving-Configura)
- InferaSim 可用于设计小规模标定与留出。[官方说明](https://rocm.blogs.amd.com/software-tools-optimization/inferasim/README.html)

论文里的误差对应各自测试环境。本项目按 910B 的 trace 校准和报告实际误差。
