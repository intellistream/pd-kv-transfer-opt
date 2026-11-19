# 远端 KV 读取瓶颈与优化研究

## 判断

优先在 AscendStore 数据路径验证同头 Attention KV 的跨机读合并，再在传输字节和命中率固定时验证请求级异步恢复。前者减少 D→P 必须传送的字节，后者减少一个慢 GET 对同一 P 引擎其他请求的阻塞。路由、准入和选读策略用于有本地命中、路由差异或排队差异的场景。

r14 每轮 96 条请求，使用两个 223,256 和 238,600 token 的完整热前缀，每条固定生成 16 token，P 本地 APC 关闭。D 侧八口限速从 100 提升到 200 Gbps，完整集合吞吐由 1.053 提升到 2.069 请求/s，TTFT p99 由 85.97 降至 41.06 秒，最慢 rank GET p50 由 1.707 降至 0.828 秒。400 Gbps 和原始设置分别达到 3.523 和 5.355 请求/s。100 Gbps 的诊断窗口中，P0 rank 0 的 `hccl_batchGet` 覆盖同步 GET 的 99.73%，与设备计算重合 0.022 ms。抓栈同时看到 EngineCore 等 worker、worker 等 GET。100→200 的平均 TTFT 改善中，12.68 秒来自 P 调度排队，8.05 秒来自 P 调度到批次结果，5.78 秒来自 P 调度器接收前等待。P forward 与 P→D 交接对各档变化小。因果链是 D→P 可用服务速率降低，随后 GET 变慢，引擎阻塞与排队放大。0.5 请求/s 时 100 和 400 Gbps 的完成率相近，TTFT 仍下降约 0.7 秒，体现负载较轻时传输服务时间的直接效应。[逐轮与抓栈](results.md#r14-远端恢复限速) · [工作负载](workloads.md#硬限速诊断) · [CANN 结构化结果](../results/2026-11-18-r14/profiler-summary.json)

D 侧重试计数增量为零。100、400 Gbps 和原始设置的 P0 rank 0 诊断窗口中，平均每次 HIXL 批量读为 54.97、12.13 和 6.05 ms，流同步为 29.12、5.62 和 1.51 ms，`hccl_batchGet_` 提交 API 约 0.88–1.03 ms。这些 API 与设备区间有包含或重叠。未限速时测得的 511.05 Gbps 是各轮最高采样区间均值的中位数，八口各 200 Gbps 是端口设置。两者之间的空间还须用同形状主动 GET、每口有效载荷和在途数区分引擎供给不足、存储地址放置、端点/DMA 限制及网络拥塞。自然会话的 r11/r12 中 P 同步 GET 只占各 rank 窗口的 0.23%–0.29%，后台竞争主要拉低 P→D 交接带宽，整轮吞吐下降 0.87%。对甲方所见现象的外推应以其远端命中、每请求有效字节、活跃端口和方向为前提。[r14 profiler](../results/2026-11-18-r14/profiler-summary.json) · [r14 与自然会话对照](results.md)

## 最接近的工作与可迁移机制

现有实现的重合程度以 [r14–r16 的直接相关工作](related-work.md) 为准。SGLang Mooncake Direct Linker 对全组复制的 MLA 实现了直接读取一次、分层打包并广播。该路径提供 r15/r16 的机制对照。Evolving Infra 在多轮 P/D 上进行了共享缓存恢复与本机分发的端到端测量。实验仍以当前设备和模型上的 [r15/r16 对照](results.md) 为准。

| 工作 | 原文机制和证据 | 在 910B 上的用法 |
|---|---|---|
| [When the KV Cache Is Slower Than the GPU，§3、§5.1–5.4、§6.2–6.3](https://arxiv.org/html/2610.21500v1) | 8× B300，GLM-5.3 FP8 MLA，PCP8 / EP8，LMCache 与 Valkey。独立 prefill 回放保留 683 条约 16K 的生产基准请求的发送时刻，性能分解使用其中 450 条高命中请求，L2 供给 98.5% 输入 token。§5.1 的 rank 0 主机拷贝热点让其他 rank 等集合通信。原先由一个 rank 存取，再由全部 rank 消费，作者改为按块确定所属 rank、每 rank 一个远端分片、协商最长可用前缀后按同一顺序广播。§6.2 的聚合 GET 从单实例约 1.71 提升到 10.76 GB/s。§6.3 相同命中口径下，先做分片的同步路径 p99 由 83.32 降至 0.80 秒，加入异步后到 0.73 秒，每个条件仅运行一次。§5.4 记录提交和注册 future 间的竞态会吞掉完成通知，缺失 key 的解析也曾卡住读取。 | 将负载 `字节/秒`、单 rank 阻塞和引擎排队分别建模。参考按块分摊 owner、带代数的就绪通知和跨 rank 一致的失败/回退。该实现分的是 MLA/PCP token 块与 Valkey 服务线程，当前项目分的是 GQA 同头副本与 Ascend RDMA 端口。按每个同头组分散读取者，避免单 rank 代读全部 KV。 |
| [KVDirect，§4](https://arxiv.org/html/2501.14743v1) · [HAWK，§4–5](https://arxiv.org/html/2610.29873v1) | KVDirect 用张量描述符合并传输并发出异步读，把设备地址直接交给传输层。HAWK 在 GPU→NPU 的场景针对目的张量布局配置 RDMA 缓冲区，将部分转换合入传输。 | 比较直接写入最终 KV 页与中间缓冲再拷贝，用字节计数验证隐藏的 HBM / host DRAM 搬运。两者均针对 P→D 交接，当前 D→P 的后端和张量约束须重新验证。 |
| [Mooncake FAST 2025，§4](https://www.usenix.org/system/files/fast25-qin.pdf) · [DistServe OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) · [NetKV，§VI](https://arxiv.org/html/2606.03910v1) | Mooncake 用前缀复用和服务时间估计指导放置与准入。DistServe 按互连带宽做阶段放置，以 TTFT/TPOT SLO 定义 goodput。NetKV 在模拟中使用网络状态为 P→D 交接选择 decode 实例。 | 路由同时估计 P 本地可复用前缀、远端读取字节与排队、重算时间、D 首步预算。r14 的 P/D 路由共享 D 主机出口，单纯重排固定字节的请求可改变尾延迟与公平性，整体供给仍受同一带宽约束。 |
| [DualPath，§4–6](https://arxiv.org/html/2602.21548v2) | 用闲置 D 存储网口从独立存储读取，经计算网回送 P，配合 NIC 队列优先级和动态双路径分配。其收益依赖存储网与计算网的资源隔离。 | r14 的对象已在 D 侧 Store，受控点就是 D→P 发送路径。若发现独立且可用的资源，比较源端口放置与分流，并同时测对 EP/TP collective 的影响。 |
| [TENT，§3–5](https://arxiv.org/html/2604.00368v1) · [ARK，§3–4](https://arxiv.org/html/2610.41160v1) | TENT 在 H800 评估基于遥测的多路径切片、调度和故障切换。ARK 在包级 RDMA 模拟中评估长流的路径冲突和路径预留。 | 先验证已有可用且闲置的物理路径，并测同一描述符、端点和注册内存的吞吐上限。当前 wheel 经 Ascend HIXL/HCCL 路径，TENT 的 H800 收益与 ARK 的模拟收益不作为 910B 的预测。 |
| [CacheGen，§5.2–5.3、§7.5](https://arxiv.org/html/2310.07240) | 按 token 块编码 attention KV，网络受限时在不同精度和重算之间切换，解码与传输流水执行。 | 同头合并先保留原始 BF16 内容。若减量后 D→P 仍限制服务，再比较压缩节省字节、解码开销和输出质量，并按模型状态组确定适用范围。 |

模拟器沿用 ServeTwin 的有状态 P/D 容量准入和分阶段校准思路，增加 D→P 与 P→D 两个资源方向、TP rank 就绪屏障、池对象版本和 pin、本机拷贝事件及引擎收件阻塞。对候选方案分别改变字节、服务并发和可运行条件，留出不同带宽与负载交叉验证。ServeTwin 的设备成本与验证误差对应其环境。[ServeTwin，§3–5](https://arxiv.org/pdf/2610.02732) · [仓库模拟器](simulator.md)

## 优先实现同头跨机读合并

固定版本的 worker 已区分 Store 键的所有权与目标张量分配。TP4 且 KV head 数为 2 时，Attention 组的键使用 `head_or_tp_rank=tp_rank//2`，Mamba align 组用 `tp_rank`。同步加载按 rank 各自组装本地地址、大小和键并调用 `m_store.get`。因此现有键空间可以共享 Attention 对象，而两份本地 Attention 副本仍可能各产生一次 D→P 读取。r14 按 BF16 Attention 布局与描述符估算，每轮 914.164 GB 中有 907.983 GB Attention 数据。如果所有同头副本都通过一次跨机读取加一次本机分发，理论可少发 453.991 GB，即 49.66%。它是读取字节上界，实际重复实例、放置和节省量需按描述符和物理端口核对。[固定模型配置](../configs/qwen35-config.json) · [AscendStore 键的生成](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L174-L237) · [同步加载](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L782-L850)

1. **确认可替换的读取。** 对齐请求、前缀块和组 ID，记录模型及缓存命名域、版本或 lease、布局、层、dtype、物理大小、head、源 replica、实际 D 端口、P 目标 rank 和本地页。相同 key 应配合可用范围、位级一致性及目的张量视图判断。Mamba/SSM 与卷积态各 rank 保留原有读取和对齐检查。
2. **避免在 D 或 P 端重新造热点。** 先比较每个同头组固定 owner、同组分块条带化 owner、现有四 rank 各读。按源段位置记录四类映射，分别是 D 端口、P 端口、P owner rank、P 接收副本 rank。仅切换 P owner 可能依然由同一个 D 物理段送出。先用现有 Store placement 做隔离对照，确认热点后再改变 D 的块放置。实测 NPU peer/HCCS 拷贝的时延、带宽及其对前向和 HCCL 集合通信的影响。
3. **发布完整 Hybrid 状态。** 全部必要 Attention 和 align 状态及本机拷贝完成后再进入可执行状态。所有 rank 使用同样的通信顺序、失败状态和缓存前缀。源对象和目的页在读及拷贝完成前固定，取消后延迟回收地址直至事件完成。先用原生 GET 路径作回退，失败重试仍计入跨机字节。

相同请求流下，以跨机有效载荷、每口流量、最慢 rank GET、同头拷贝时间、完整集合吞吐和 TTFT p99 判断收益。使用原生路径作同机回切，分别在 100/200 Gbps 与原始设置下跑高命中负载，另用 APC 开启的自然会话检查迁移性。成功率、各 rank 的缓存块内容和确定性输出短检通过后开始计时。若字节下降而最慢 rank 变慢，按单端口、源段和 HCCS 重叠定位。该机制的研究增量在已共享存储对象上消除重复的跨机读取，同时保留必要的 Hybrid 状态与实际链路并行度。NIXL 已有跳过冗余 MLA producer 的先例，兼容矩阵也区分了 Mamba 的并行布局限制。[vLLM NIXL 兼容矩阵](https://github.com/vllm-project/vllm/blob/main/docs/features/nixl_connector_compatibility.md)

## 再验证同步屏障解耦

已安装环境是 vLLM 0.23.0、vLLM-Ascend 0.23.0.post1、Mooncake NPU 0.3.11.post1。上游 #12390 的标题和说明范围是 AscendStore KV Pool v0.23.0，列出 Qwen3.5/3.6 Hybrid 不支持池的 `use_layerwise=true`，并记录 layerwise 路径会降级到 `PIECEWISE`。当前 2P2D 在 11 月 17 日尝试了请求级开关 `load_async=true`，两条请求超时，调度器重复生成加载规格，worker 未进入实际 GET。现象来自这组 Hybrid、MultiConnector 和固定版本，需先查事件投递与完成回流。11 月 18 日在 37K/57K 前缀上做同批四 GET 并发的 A/B/A，API 调用累计时间与区间并集比由 1.0 到约 3.02，各 rank 中位 GET 的均值由约 28 升至 82 ms，完整集合排空未改善。该轮等待所有 GET 后才前向。两次实验的内部日志索引分别见 `CONTINUATION_RESULTS_20261117.md` 与 `PROFILING_ANALYSIS_20261118.md`。[部署版本](deployment.md) · [AscendStore v0.23.0 已知问题第 5–7 项](https://github.com/vllm-project/vllm-ascend/issues/12390) · [AscendStore 设计 RFC](https://github.com/vllm-project/vllm-ascend/issues/12234)

较新的 v0.30 文档已列出普通 Mooncake layerwise pooling 的 Hybrid 支持，要求 `mamba_cache_mode="align"` 并按完整对齐状态发布。合并 PR 用 Qwen3.5-35B W8A8、TP1、16K 实测。KVPP 叠加池化的支持范围仍为非 Hybrid MLA/SFA。BF16 910B2、TP4、双机 2P2D 长上下文的兼容性和性能待实机确认。[Hybrid 文档](https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/feature_guide/mooncake_hybrid_attention.html) · [合并与实测](https://github.com/vllm-project/vllm-ascend/pull/17287) · [KVPP 支持范围](https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/feature_guide/kvpp.html)

保留 non-layerwise 的传输粒度，做有界的请求级恢复。P 为一个待执行请求固定目的页并提交 I/O，EngineCore 可继续接纳或执行已就绪请求，等所有 rank 完成后才允许运行该请求。复用 vLLM 的等待和完成接口，检查 MultiConnector 下每个请求 ID 携带的 connector 路由、完成归并条件和请求取消路径。提交前注册完成事件，用请求 ID 和本次分配的版本识别过期事件。记录 submit、start、device complete、rank ready、engine resume。最大在途请求数先取 1，并给字节和 HBM 设置上限。对部分前缀、GET 失败和取消记录一致的失败边界，超时回退前处理仍在访问目的页的传输，避免回退重算与已废弃的读取争抢资源。[固定 vLLM 调度器等待入口](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L799-L812) · [完成处理](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/sched/scheduler.py#L2154-L2245) · [远端缓存完成事件竞态](https://arxiv.org/html/2610.21500v1#S5.SS4)

异步对照保持命中和跨机字节相同，先看 P worker 有效前向与 GET 的重叠、无关请求的 EngineCore 等待、p99 和全部请求的完成率。一旦 100 Gbps 继续饱和，高命中轮的总吞吐仍受字节预算约束。混合冷/热会话与更高带宽下更可能有可重叠的工作，应并列观察。若原型有效，再将读合并同异步恢复组成 2×2 对照。

## 调度与传输层的后续判据

路由比较可执行路径的预计完成时间。用端口排队、远端服务、必要的本机拷贝和全 rank 就绪时间估算 GET，结合实际分块 prefill 时间选择复用或重算。对同一会话优先检查 APC，按真实依赖释放后续轮次。远端候选使用在途字节与端口 backlog 做准入，输出同时报告被接受、回退或超时的请求和 TTFT/TPOT goodput。热前缀轮中不同请求之间的读合并依赖对象 pin、目的页归属及同一前缀的实际重复次数，应与 rank 同头读合并分别报告。[自然会话与 r14 输入差异](workloads.md) · [远端读取和重算的成本模型](https://arxiv.org/html/2610.21500v1#S4)

若未限速时同地址和同描述符的传输微基准能超过工作负载供给，沿实际 pool 的 GET 提交、服务端排队、逐端口吞吐、CQ/stream 同步查给定并发下的空档。若负载能驱动连续在途传输且只有少数端口接近各自上限，再处理放置、rail 映射和分片。若多口稳定接近控制上限，按请求减少跨机字节的收益最直接。先把 `HCCL batchGet` host/device 时序、MAC 有效服务、PFC 或错误计数和 dtype 布局的同轮证据对齐，再决定是否下沉到传输层修改。更换引擎、压缩或直接搬用已有网络调度器需要重新建立同等的兼容和性能对照。[r14 profiler](../results/2026-11-18-r14/profiler-summary.json) · [TENT 多路径设计](https://arxiv.org/html/2604.00368v1)
