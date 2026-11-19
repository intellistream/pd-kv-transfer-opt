# 关键实验结果

## r20 同批分发与下一笔 GET 的异步重叠

### r19 trace 提供的机会

r19 的四个打包臂各有 11 个计时 P 批次，均对齐四 rank 前向入口及 rank 0 设备时长。每臂 6 个恢复批，含多笔恢复的批次依次为 2、2、2、4。已有的 P0 分发与同实例 GET 时间相交约 87–149 ms，来自同批另一个 head pair。同一 rank 和同头 peer 的 GET 与分发交集均为 0，同实例跨批也为 0。P0 的分发与同批另一个 head pair 的 host forward 相交约 33–80 ms，与 P1 的 host forward 相交约 112–258 ms。设备事件只有时长，因此 prefill 的相交只能定位到 host 前向窗口。[逐批重算](../results/2026-11-19-r20-packed-overlap/r19-opportunity.json)

在两个 100 Gbps 打包臂，决定 TTFT 和排空的两个请求落在同一个四恢复 P0 批。前三笔分发分别约 119–140、40 和 114–115 ms，各笔下一 GET 在前笔 copy 返回后约 1 ms 才启动。按实测服务时长重放，保留待完成分发的 fence、同头汇合和末次发布门槛，给每次提交预留 0–10 ms，估计该批最晚 rank host forward 可提前 **246–282 ms**。200 Gbps 首臂的对应批可提前 **178–188 ms**，反序臂的两个尾请求自然拆批，估计各可提前 **64/71 ms**。估计固定已有批次与 GET、copy 服务时间，设备竞争由实机检验。[计算方法](../pd_kv_transfer_opt/profiling/analyze_r19_copy_opportunities.py)

### 固定字节的实机对照

在同一 2P2D 进程切换同步打包和同批一笔在途分发。后一笔 GET 是同一 worker 启动的原生同步调用，使用与同步臂相同的物理描述符。后一笔同头汇合与 forward 前都等分发完成，保留同一通信流、固定缓冲和 KV 发布语义。两模式各在计时前用两个 P 实例共 8 个 rank 做 4 KiB 独立 GET 抽样。运行沿用 r19 的固定输入、前驱、路由、到达、APC 与逐请求输出预算。[运行代码与准备流程](deployment.md#r20-的同批分发重叠实验)

设备重叠用前后事件查询和设备经过时间给出保守下界。对各 rank 取同批 copy 事件跨度与下一笔原生 GET 调用区间的交集，然后在实例内取并集。这个区间包含设备流中等待和提交产生的空档，细粒度的执行单元和链路活动另需 CANN 抓取。`pending` 墙钟区间还包含晚些时候才观察到完成的时间，单独保留。[Ascend PyTorch 事件接口测试](https://gitee.com/ascend/pytorch/blob/65f5cf4ae87eb8e1a894f7556176242c8e4ca159/test/npu/test_torch_npu.py)

100/200 Gbps 各做 A/B 和 B/A，共 **80/80 条计时请求及 144/144 条准备请求成功**。每臂 D Store 在准备前读回 0 键。每个计时臂的 40 笔 GET 都是 **21,237,923,840 B**，逐请求字节一致。P→D 描述符均为 42.964 GB，P 调度均为 23,540 token。D MAC TX 约 21.56–21.57 GB。同步和异步模式的 16 次 4 KiB 抽样均在计时外完成。新增反序案例时用相同路由生成新请求标识。两端各两个模型实例持续运行，D 代理在两组之间重启一次。[归档、事件与回执](../results/2026-11-19-r20-packed-overlap/manifest.json) · [运行代码指纹](../results/2026-11-19-r20-packed-overlap/runtime/provenance.json)

| D 出站设置 | 实际顺序 | P0 / P1 新增设备窗重叠下界（ms） | P0 同步 load 并集，同步 → 异步（s） | TTFT p99（s） | 排空吞吐（token/s） |
|---|---|---:|---:|---:|---:|
| 100 Gbps | 同步先 | 301.1 / 76.0 | 2.505 → 2.325 | 5.367 → 5.000（−6.85%） | 185.12 → 186.65（+0.83%） |
| 100 Gbps | 异步先 | 292.7 / 72.6 | 2.529 → 2.365 | 5.268 → 5.300（+0.61%） | 185.63 → 187.20（+0.84%） |
| 200 Gbps | 异步先 | 316.4 / 45.7 | 1.605 → 1.433 | 4.287 → 4.164（−2.87%） | 192.55 → 193.57（+0.53%） |
| 200 Gbps | 同步先 | 278.2 / 0 | 1.557 → 1.458 | 4.208 → 4.289（+1.92%） | 193.36 → 199.12（+2.98%） |

100 Gbps 首对，P0 的重叠区间来自三条依赖边。`winning-avg-corewars:131` 分发与 `train-fasttext:69` 的 GET 相交约 163 ms，`train-fasttext:69` 分发与 `regex-chess:106` 的 GET 相交约 34 ms，`regex-chess:106` 分发与 `count-dataset-tokens:45` 的 GET 相交约 104 ms。200 Gbps 反序轮的 P1 自然形成三个单 GET 批，分发都先完成后进前向。四组均有 P0 重叠，且 P0 同步 load 净省 **99–180 ms**。多恢复 rank 的地址检查中位数约 10–13 ms，正式轮无地址别名回退或探针矛盾。[逐 rank 和逐请求数据](../results/2026-11-19-r20-packed-overlap/summary.json)

两档轮均排空吞吐分别增加 **0.84% / 1.76%**。每臂各回放同一批 10 条后继，nearest-rank p99 是该轮最慢值。每档合并 20 次观测的 p99 为 5.367 → 5.300 s 与 4.287 → 4.289 s，两档逐对方向均混合。八臂的最慢 TTFT 都为 `regex-chess:106`。200 Gbps 反序时，P0 整体 load 降了约 99 ms，但这条请求的代理 P 阶段增加约 17 ms，P 结束到 D 首 chunk 增加约 72 ms，端到端 p99 增加 81 ms。决定排空的 `winning-avg-corewars:131` 则提早进入约 17 秒的 D 输出尾段。两种指标因而可以在同一对试验向不同方向变化。

两机模型进程已停止，16 卡均确认空闲，活跃配置逐字节恢复，D 侧八口读回原来的 200 Gbps/口。[校验和清理回执](../results/2026-11-19-r20-packed-overlap/manifest.json)

### 逐轮阶段复盘

**预期和对照。** 同批异步的直接预期是缩短相同恢复批从最晚 worker 进入到最晚 rank 开始 host forward 的时长。用每对 r20 的同步 trace 和固定的 GET、copy 服务时间重放，给每笔提交预留 10 ms。100 Gbps 两对及 200 Gbps 首对的 P0 尾请求都落在同一顺序的四恢复批，预计提前 **245、245、186 ms**，实测提前 **123、227、215 ms**。100 Gbps 两对中 rank 2 的累计 GET 分别比同步多 **96、108 ms**，所以隐藏的复制等待只部分传到前向入口。首对 rank 2 的笔间空档减少 224 ms，同期设备 copy 时间近似持平。相同 GET 字节和顺序下的时长变化可直接观测，当前数据只能定位到 GET 调用区间。200 Gbps 反序的异步臂拆成两个恢复批，应按请求追踪整条链路。[逐 rank、批次和请求](../results/2026-11-19-r20-packed-overlap/stage-audit.json)

**最慢 TTFT 请求。** 八个臂均为 `regex-chess:106`。下表是异步减同步，负数表示缩短。P HTTP 依次包括引擎入队等待、首派发到末批结果、引擎观察窗两侧的剩余时间。D 分列从 decode 开始到最早传输入队、最早入队到最晚 handler 结束、末次结束到首块。余项包含客户端派发、代理入口和 P 至 D 代理衔接。P 引擎外合量用两边各自的单调时钟时长相减，D 各分段共用同一台主机的单调时钟。单位 ms，四舍五入后各列可能有 1 ms 差。

| 档位与配对 | P 引擎队列 | P 派发至末批结果 | P HTTP 引擎外 | D 传输前 | D handler 窗口 | D 传输后至首块 | 余项 | TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 首对 | −46 | −118 | −166 | −50 | −4 | +2 | +14 | **−367** |
| 100 反序 | −16 | −140 | +147 | −28 | +19 | +19 | +31 | **+32** |
| 200 首对 | −59 | −135 | +30 | +19 | +18 | +12 | −9 | **−123** |
| 200 反序 | −51 | −23 | +91 | +55 | +2 | +16 | −10 | **+81** |

- **100 首对** 的 −367 ms 包含 P 引擎外减少 166 ms，以及 D 查池返回到首笔传输入队减少 53 ms。尾延迟降幅包含多处变化，同批前向门槛实测只省 123 ms。
- **100 反序** 的 P 派发至末批结果省 140 ms。P HTTP 引擎外增加 147 ms，客户端实际派发也晚 25 ms。按 Unix 事件锚点拆 P 引擎外区间，增加的主要部分落在末批结果事件到 HTTP 返回，估计约 +253 ms，发送到引擎接纳则减少约 106 ms。结果事件位于 `update_from_output` 入口，该测量暂时只能定位到 scheduler 更新、输出交付及 API 返回的组合区间。
- **200 首对** 的 P 批阶段省 135 ms，D 首步前后的三个阶段合计增加约 50 ms，最终 TTFT 省 123 ms。
- **200 反序** 的 P 引擎外增加 91 ms，其中发送到 P 引擎接纳估计增加约 79 ms。D 传输前增加的 55 ms 几乎都发生在查池方法开始前，传输 handler 窗口只增加 2 ms。该轮定位到两侧的接纳、首步与交付路径。有待采集 D 接纳和首轮调度的细分事件。

**排空请求。** 八个臂最后结束的都是 `winning-avg-corewars:131`。四对首 token 分别提前 **262、263、185、900 ms**，首 token 后的输出耗时分别增加 **86、84、76、306 ms**，净排空缩短 **176、179、109、594 ms**。200 Gbps 反序的排空请求所在首批总量从 4096 变为 2048 token，rank 0 前向设备时间约 391 → 206 ms。以该异步臂 P0 首个计时批的派发为零点，这个新共批在 +147 ms 已提交，首个允许下一 GET 提前的 copy 要到 +876 ms 才提交完成。最大的一次吞吐增幅因此包含批形和接纳时序差异。同批流水的直接收益以实际 overlap 和匹配批门槛为准。[逐轮阶段数据](../results/2026-11-19-r20-packed-overlap/stage-audit.json)

### 跨批等待证据

分别复核 r19、r20 的 88 个正式 P 批。48/88 和 50/88 在调度器已有输出、worker 尚未进入时与旧批 GET 相交。去掉自身有 GET 的部分，在各自 40、38 个无 GET 批里剩 **31 和 29** 个相交。所有这些批均含旧恢复批里的请求，且其先前结果在 dispatch 时尚未交给调度器。两个 run 各有 79 个相交的无 GET 后续 chunk，记录中均没有在先前结果已交给调度器、仍处于旧 GET 时段且尚未进 worker 的样本。比如 r20 的 P1 中 `raman-fitting:28` 和 `gpt2-codegolf:27` 的续块派发后等了 900 ms，有 647 ms 与旧 GET 相交，两个请求上一块在约 905 ms 才出现结果接收事件。scheduler 接收结果事件晚于设备产生或发布结果的可能时点。这些数据支持先查清依赖完成和 worker 排队的边界。[跨批统计与具体样本](../results/2026-11-19-r20-packed-overlap/cross-batch-audit.json)

另一条可诊断的等待是**已经调度的请求何时能安全发自己的 GET**。r19 有 20/80、r20 有 24/80 个首 chunk 的 dispatch 到入 worker 等待与其他请求的旧 GET 相交，且自己也有远端恢复。r20 最长候选 `video-processing:28` 派发后约 1.91 s 才发出自己的 GET，其中与旧批 GET 相交约 1.35 s。判定可提前的前提是找到目的页分配与租约生效、请求的 connector job 就绪、可用传输容量三个时点，并区分 executor 入队与轮到 worker。现有事件可以定位候选，安全提前量和收益仍需这些埋点才能估算。[跨批事件定义](../results/2026-11-19-r20-packed-overlap/cross-batch-audit.json)

### 重算

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_r19_copy_opportunities \
  results/2026-11-18-r19-natural-ab/r19-P.jsonl.gz \
  results/2026-11-18-r19-natural-ab/r19-D.jsonl.gz \
  runs/r19-copy-opportunities.json
python -B -m pd_kv_transfer_opt.profiling.analyze_packed_overlap \
  results/2026-11-19-r20-packed-overlap/r20-P.jsonl.gz \
  results/2026-11-19-r20-packed-overlap/r20-D.jsonl.gz \
  configs/workload-r19-natural-restore.json \
  runs/r20-packed-overlap.json
python -B -m pd_kv_transfer_opt.profiling.audit_packed_overlap_stages \
  results/2026-11-19-r20-packed-overlap/r20-P.jsonl.gz \
  results/2026-11-19-r20-packed-overlap/r20-D.jsonl.gz \
  results/2026-11-19-r20-packed-overlap/summary.json \
  runs/r20-stage-audit.json
python -B -m pd_kv_transfer_opt.profiling.audit_prefill_batch_waits \
  results/2026-11-18-r19-natural-ab/r19-P.jsonl.gz \
  results/2026-11-18-r19-natural-ab/r19-D.jsonl.gz \
  results/2026-11-19-r20-packed-overlap/r20-P.jsonl.gz \
  results/2026-11-19-r20-packed-overlap/r20-D.jsonl.gz \
  runs/r19-r20-batch-wait-audit.json
```

## r19 APC 开启的高远端恢复 A/B

沿用 r11/r18 的同版本 Terminal-Bench 2.1 轨迹。从实际 D→P 读量较高的会话中选 10 条后继，每条首次访问目标 P。每臂在原档完整执行 18 条真实前驱，等待 Store PUT 排空，再按 50 ms 间隔提交目标。每臂使用独立缓存命名域，保留原 P/D 路由、提示 token 和逐条输出预算，共 3,971 个输出 token。P APC 开启，分块与图捕获预算沿用 4096。同一进程回切原生各 rank 读和同头单读加打包，限速只施加在目标窗口。正式窗口前在两个 P 实例完成打包抽样校验。[负载](workloads.md#r19-高远端恢复的自然后继) · [实测运行域](deployment.md#r19-的-apc-恢复实验)

两档各两次相反顺序，**80/80 条计时请求、144/144 条准备请求成功**。每臂准备前专用 D Store 非强制清空并读回 0 键。目标观测到的 P 本地命中均为 0，外部匹配总量均为 1,005,568 token，P 实际调度总量均为 23,540 token。两臂的 P→D 描述符固定为 42.964 GB。客户端实际 dispatch 最迟比计划晚 51.5 ms，D 代理入口最多晚 56.3 ms，分布记录在归档。四臂打包共 160 次 rank 决策均走打包，计时内无额外校验 GET 或负返回。[逐请求和环境读回](../results/2026-11-18-r19-natural-ab/summary.json)

| D 出站设置 | 回切块 | 排空吞吐，直读 → 打包（token/s） | TTFT p99，直读 → 打包（s） |
|---|---|---:|---:|
| 100 Gbps | A/B | 173.05 → 184.00（+6.33%） | 6.980 → 5.341（−23.47%） |
| 100 Gbps | B/A | 175.33 → 184.26（+5.10%） | 6.843 → 5.452（−20.33%） |
| 200 Gbps | B/A | 188.86 → 191.65（+1.48%） | 4.912 → 4.234（−13.80%） |
| 200 Gbps | A/B | 187.46 → 198.41（+5.84%） | 4.845 → 4.425（−8.66%） |

轮均吞吐在 100 Gbps 为 **174.19 → 184.13 token/s（+5.71%）**，在 200 Gbps 为 **188.16 → 195.03 token/s（+3.65%）**。每轮只有 10 条，nearest-rank p99 是该轮最慢值。每档合并每臂的 20 次观测后，p99 分别为 **6.980 → 5.452 s（−21.88%）** 和 **4.912 → 4.425 s（−9.90%）**。两次回放的是同一批后继，结果描述该受控恢复波次。

**实际减量为每轮 20.594 GB（49.23%）。** 四轮原生都是 41.832 GB，四轮打包都是 21.238 GB。打包臂同一已掩码计划的原生等效字节逐请求完全匹配，因此差异归于同头消重。D MAC TX 的轮中位数在两档均约 42.47 → 21.57 GB，P MAC RX 也约 42.47 → 21.57 GB。D→P GET 和反向交接用独立描述符与 MAC 方向核对。[字节、每口计数与限速回执](../results/2026-11-18-r19-natural-ab/summary.json)

| D 出站设置 | P0 GET 活跃时间并集（s） | P0 同步 load 占用并集（s） | P0 打包分发活跃时间（s） |
|---|---:|---:|---:|
| 100 Gbps | 4.087 → 2.081 | 4.123 → 2.640 | 0.583 |
| 200 Gbps | 2.073 → 1.010 | 2.110 → 1.577 | 0.572 |

**同步 GET 是可干预的 P 侧瓶颈。** 在原生载荷完全相同时，带宽翻倍将 P0 GET 活跃窗口从 4.09 s 降到 2.07 s，轮均排空吞吐提高 8.0%。打包在 100/200 Gbps 分别省约 2.01/1.06 s GET 活跃时间，P0 实际的同步 load 净省约 1.48/0.53 s。第二档仍有同样的跨机减量，分发约 0.57 s，收益相应变小。100 Gbps 原生两轮的 2 s 提交峰值都是 101.28 Gbps，按热点 D 口 MAC TX 和限速计算的串行下界约 3.76–3.91 s，接近跨 rank GET 活跃并集 4.25–4.30 s。粗粒度计数与 PFC 帧用于核对服务压力，无法量出子秒饱和或暂停时间。捕获窗口内 CNP 和重传增量均为 0。

**首 token 和排空走不同的尾部依赖。** 八个实验臂的最慢 TTFT 都是 `regex-chess:106`。100 Gbps 首对中，这条请求的代理 P 阶段 6.166 → 4.550 s，P 已接收后的首次调度等待 120 → 80 ms，所经批次 GET 并集 3.422 → 1.629 s，增加 0.520 s 的本机分发，并使批次 load 并集 3.453 → 2.083 s。P 结束到 D 首 chunk 为 0.755 → 0.736 s。批次区间可能包含同批其他请求，区间可以重叠。八臂始终由 `winning-avg-corewars:131` 排空，其 D 首 token 后输出约 17.11–17.42 s。减少远端读首先提高 P 阶段和 TTFT，影响整轮吞吐的幅度还取决于最后一条的 D 输出尾部。[模型](problem-model.md#减量如何传到服务指标)

两机 16 张卡均确认空闲，活跃配置逐字节恢复，D 侧八口均读回原来的 200 Gbps/口。[归档校验与清理回执](../results/2026-11-18-r19-natural-ab/manifest.json)

### 重算

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_natural_restore_ab \
  results/2026-11-18-r19-natural-ab/r19-P.jsonl.gz \
  results/2026-11-18-r19-natural-ab/r19-D.jsonl.gz \
  configs/workload-r19-natural-restore.json \
  runs/r19-restore-ab.json
```

## r18 原长会话打包分发 trace

沿用 r11 的 30 个闭环会话、170 条请求、原固定路由和逐请求输出，P APC 开启、分块为 4096，D 八口均保持原档 200 Gbps，全程无注入流量。只补固定打包一轮。P worker 在提交前核对同头描述符，共有 49 条实际远端恢复、196 个 rank 打包操作，全走打包。两个 P 实例共 8 个 rank 的首个缺页抽样用独立 GET 校验通过。排除额外校验读，实际 D→P 描述符 **33.920 GB**，同一读计划的原生描述符合计 **64.686 GB**，减量 **47.56%**。后者由本轮各 rank 的未筛选描述符汇总。[负载](workloads.md#自然长会话) · [r18 摘要](../results/2026-11-18-r18-natural-packed/summary.json)

**本机分发会挡住本 rank，串行积压稀少。** 875 个 P 批次均匹配到四个 rank 的 host forward。48 个含分发批次里仅 1 个恢复两条请求，第一笔 copy 完成前同 worker 无法提交第二笔 GET，各 rank 第一笔 copy 约 1.97–2.12 ms。该批派发到 P 返回结果为 746.17 ms。排除两个短检批后，188 条 copy 墙钟 p50 为 **3.66 ms**、p95 **66.41 ms**、最大 **119.43 ms**。98 个同头组 copy 的 host 窗口，同 P 实例内不同组重叠 37 对，跨实例重叠 2 对。重叠不能给出链路争用量。固定 GET、配对与释放，并理想去掉 copy 服务时，46 批的最晚 rank host forward 可提前中位 **4.01 ms**，最大 **119.31 ms**。这是软件启动门槛的估计。

**三策略重放。** 固定打包使用原观测。另两种方案每批最多让一个同头恢复组改走原生双读，以 r11 的同请求、P 实例、rank、原生字节的观测耗时作为备选。94 个合法候选中 78 个匹配，覆盖 46 批中的 38 个完整候选集。按名义耗时得到下表。

| 策略 | 改变的批次 | 额外 D→P 字节 | 46 批的最晚 rank host forward 累计提前 |
|---|---:|---:|---:|
| 固定打包 | 0 | 0 | 0 |
| 只看通信完工 | 38 | 9.532 GB | 158.396 ms |
| 看批次就绪 | 38 | 9.532 GB | 158.425 ms |

两种选择只在 1 批不同，批次就绪额外得到 **28.55 µs**。对原生耗时作 ±25% 敏感性检查，差值不变。限定在客户端有完整终点的 45 个恢复批次，差值也一样。两种预测均保持其余读服务和 rank 释放固定，累计启动提前只用于筛选。真实 collective 等待和端到端收益需要另测。现有差异未达到实机策略 A/B 的证据条件。[七步法的当前判断](research-seven-steps.md#4-what-is-the-key-idea)

P 批次和 D 代理 `decode_end` 覆盖 170 条。SSH 瞬断触发清理时，客户端仅持久化了 168 条成功终点，逐请求输出均符合固定预算。该轮作为 P 侧机制 trace 使用。两侧控制器均停止，16 卡空闲，活跃配置逐字节恢复，D 八口读回仍为原档。[脱敏事件与核对数据](../results/2026-11-18-r18-natural-packed/summary.json)

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_packed_readiness \
  results/2026-11-18-r18-natural-packed/r18-P.jsonl.gz \
  results/2026-11-18-r18-natural-packed/r18-D.jsonl.gz \
  results/2026-11-18-r18-natural-packed/r11-native-P.jsonl.gz \
  results/2026-11-18-r18-natural-packed/r11-native-D.jsonl.gz \
  runs/r18-readiness.json
```

## r17 固定恢复路径短筛选

复用 r16 的同一套 2P2D、模型、图策略和 4096 分块配置，仍关闭 P 本地 APC。原生各 rank 直读与 r16 固定条带轮换读者加打包，在同一次 P 启动中做 A/B 与 B/A 回切，两臂各预热 2 条。使用同一对 223,256 和 238,600 token 前缀，16 请求/s 到达，每轮 24 条、每条产出 16 token。192 条正式请求全部成功。每轮 GET 原生为 **228.541 GB**，打包为 **115.043 GB**。P→D 描述符均为 **229.548 GB**。[设置](research-seven-steps.md#6-what-is-the-experiment-plan) · [摘要、每口计数与回执](../results/2026-11-18-r17-restore-baselines/summary.json)

| D 出站设置 | 配对 | 排空吞吐，直读 → 打包（token/s） | TTFT p99（s） | 四 worker 同步恢复窗口 p50（s） |
|---|---:|---:|---:|---:|
| 100 Gbps | 1 | 13.14 → 22.19 | 29.03 → 16.78 | 19.76 → 9.88 |
| 100 Gbps | 2 | 12.92 → 23.18 | 28.94 → 16.17 | 18.16 → 9.46 |
| 原档 8×200 Gbps | 1 | 47.31 → 49.62 | 7.17 → 6.78 | 1.45 → 1.62 |
| 原档 8×200 Gbps | 2 | 52.44 → 49.69 | 6.14 → 6.53 | 1.48 → 1.70 |

恢复窗口以同一请求对应的四个 worker 最早进入同步 load 到最晚退出计算。窗口包含同批其他恢复。四 worker 均成功匹配，统计按请求计，同一批可能出现多次。原档下，原生 GET p50 约 218–222 ms，打包路径的 GET p50 约 63–73 ms，P 本机分发 p50 约 135–137 ms。低带宽打包缩短同步恢复窗口，原档两组均拉长。原档 TTFT p50 两组也分别由 4.81、4.51 增至 5.27、5.09 s，而排空吞吐与 p99 的差值一正一负。24 条短轮的 p99 接近样本极值。该组用于检验固定打包的失效条件，当前相反方向出现在恢复阶段，端到端劣势仍未稳定出现。[证伪步骤与判据](research-seven-steps.md#3-why-does-existing-work-fail)

原档是端口配置。D 侧 MAC TX 的最高采样区间原生只有约 221–244 Gbps，不能用 1,600 Gbps 当作实际服务速率。原档第一对两臂，P 八口 TX 约 233.06→233.04 GB，RX 232.02→116.82 GB，分别与 D 八口的对向计数一致。打包额外完成的约 113.5 GB 本机分发没有出现在这些外部 MAC 的差值中。八轮均有恢复读回，结束后端口恢复为每口 200 Gbps，两机 16 卡空闲，活跃配置恢复。[计数与回执](../results/2026-11-18-r17-restore-baselines/summary.json)

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_same_head_ab results/2026-11-18-r17-restore-baselines/r17-P.jsonl.gz results/2026-11-18-r17-restore-baselines/r17-D.jsonl.gz --interleaved --output runs/r17-comparison.json
```

## r16 同头 KV 的打包分发

继续使用 r15 的同头选读和同步 2P2D 设置。先在 200 Gbps 记录直接分发的相邻页地址，比较 0.5 请求/s 的两个长前缀与 16 请求/s 的 48 请求短轮。直接分发只有在两个 P rank 的相邻描述符地址都连续时，才合并成同一次广播。条带和张量列的边界还会独立切分。

| 每 rank 单次直接分发，p50 | 低载 | 并发 |
|---|---:|---:|
| 原始描述符 | 4,510 | 4,510 |
| 条带和张量列边界 | 290 | 290 |
| 两端地址都断开的相邻描述符 | 60 | 3,850 |
| HCCL 广播 | 350 | 4,130 |
| 分发墙钟 | 137.4 ms | 462.1 ms |

192 个并发 rank 事件逐个核对，广播次数等于结构边界和地址断点的合计，其中约 88.6% 的广播起点来自地址断点。表中各项取各自中位数。固定版本的 vLLM 从空闲队列取块，释放的块按传入的淘汰顺序插回队列。逻辑相邻的页在地址上可能分散。r15 正式轨迹中，固定前缀和接收字节后，广播数与分发墙钟相关系数仍在 0.966–0.986。[固定版块分配器](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/block_pool.py) · [布局与逐轮数据](../results/2026-11-18-r16-packed-copy/summary.json)

每个 P rank 复用一个 128 MiB NPU 缓冲。用批量设备复制把同一 owner 的分散片段打包，再按块做 HCCL 广播，接收后散写到原目标页。每个分块按有效字节发送，整个分发流完成后交给前向。223K 和 238K 两条快速校验中，各 rank 的 8 个 4 KiB 样本都与独立 GET 逐字节一致，纯打包路径的 16 个输出 token ID 也和直接分发一致。[实现](../pd_kv_transfer_opt/experiments/same_head_kv.py) · [固定版批量内存复制算子](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/ops/triton/batch_memcpy.py)

100/200 Gbps 各三组回切，每轮 96 条，16 请求/s 到达，1,152 条正式请求全部成功。两种路径在相同 P 进程内运行。每轮均有 384 次远端 GET，D→P 描述符载荷精确相等，均为 **460,172,820,480 字节**。P 本机收到的载荷均为 453,991,464,960 字节，P→D 交接均为 918,190,817,280 字节。D 端口 MAC TX 每轮约 467.27 GB，两条路径的差值小于 0.002%。

| D 聚合限速 | 每 rank 广播 p50 | 分发墙钟 p50 | 分发墙钟 p95 | TTFT p99 | 吞吐 |
|---|---:|---:|---:|---:|---:|
| 100 Gbps | 3,700 → 18.5 | 413.8 → 137.5 ms | 555.7 → 140.0 ms | 55.52 → 46.87 s（−15.59%） | 25.19 → 29.21 token/s（+15.96%） |
| 200 Gbps | 3,780 → 18.5 | 419.0 → 137.5 ms | 616.5 → 139.5 ms | 36.75 → 25.43 s（−30.80%） | 36.79 → 49.25 token/s（+33.87%） |

表中吞吐取三轮均值，各延迟取三轮相应百分位的中位数。打包后两个前缀实测分别为 18 和 19 次广播。分发墙钟包含描述符准备、打包、HCCL 和散写，配对等待单独计量。两档分发 p50 均减少约 67%，100 Gbps 的整轮收益较小。该档打包后原生 GET p50 约 657 ms，200 Gbps 约 291 ms。原生 GET 时长仍会随同批并发变化，两组对照保持的是实际字节和限速。[逐轮指标、端口和校验摘要](../results/2026-11-18-r16-packed-copy/summary.json)

从脱敏事件重算 A/B 及布局 profiling。

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_same_head_ab results/2026-11-18-r16-packed-copy/r16-P.jsonl.gz results/2026-11-18-r16-packed-copy/r16-D.jsonl.gz --interleaved --output runs/r16-comparison.json
```

## r15 同头 Attention KV 单次跨机读取

在 r14 的双机 16 卡、2P2D、每实例 attention TP4 / expert EP4 条件下，P 同头组按 16 个逻辑块交替选读者，读后经同主机 HCCL 写入另一 rank 的目标 KV 页。SSM/conv 保留按 rank 的原生 GET。两种模式在相同运行进程内回切。APC 关闭，P 图捕获和分块预算均沿用 4096，D 保持 `FULL_DECODE_ONLY`，每条固定输出 16 token。每档三组 A/B，各轮固定两条原始长前缀、96 请求和 16 请求/s 到达，正式 1,152 条全部成功。[负载](workloads.md#硬限速诊断) · [原型部署](deployment.md#同头-kv-读取原型)

| D 侧聚合限速 | 基线 → 原型吞吐 | 变化 | 基线 → 原型 TTFT p99 | 变化 |
|---|---:|---:|---:|---:|
| 100 Gbps | 15.89 → 24.36 token/s | +53.27% | 91.30 → 57.47 s | −37.05% |
| 200 Gbps | 31.07 → 35.37 token/s | +13.84% | 43.57 → 37.89 s | −13.04% |

吞吐取三轮各自排空吞吐的均值，TTFT p99 取三轮 p99 的中位数。100 Gbps 的逐组吞吐增幅为 51.51%、55.14%、53.14%，200 Gbps 为 15.04%、18.04%、8.42%。跨机 GET 描述符每轮 **914.164 → 460.173 GB，减少 453.991 GB（49.66%）**。P 侧本机收到的 Attention 副本恰好补齐这部分，SSM/conv 载荷和 P→D 918.191 GB 交接载荷在逐轮核对中保持一致。[逐轮结果和端口计数](../results/2026-11-18-r15-same-head/summary.json)

D 侧 MAC TX 每轮约 928.15 → 467.27 GB，降幅 49.66%。以下为 100 Gbps 各口三轮中位数，200 Gbps 在两位小数精度上接近。D MAC RX 合计约 932.25 → 932.17 GB。两档第一组的 P 侧 MAC TX 基本持平，MAC RX 随远端 GET 载荷下降。

| D 端口 | 基线 TX（GB） | 原型 TX（GB） |
|---|---:|---:|
| 0 | 108.60 | 54.44 |
| 1 | 114.73 | 57.50 |
| 2 | 117.29 | 59.05 |
| 3 | 140.30 | 70.81 |
| 4 | 120.89 | 61.10 |
| 5 | 113.47 | 57.27 |
| 6 | 106.32 | 53.69 |
| 7 | 106.55 | 53.42 |

223K 短测中，原型把单请求跨机 GET 从 9.208 降至 4.636 GB，P 本机分发 4.572 GB，四 rank 分发墙钟 127.0–127.8 ms。各 rank 的 8 个 4 KiB 样本与独立原生 GET 逐字节一致。纯原型请求与基线的 16 个输出 token ID 一致。238K 前缀的 KV 抽样也一致。其输出 token ID 在额外执行独立原生 GET 的对照和纯原型中都有变化，逐 token 一致性结论限于 223K 短测。

正式并发下，每次 rank 分发的墙钟 p50 约 418 ms，p95 在 605–609 ms。单次的 HCCL 广播操作数由 223K 短测的 280 增为正式轮中位数约 3,800–3,890。此原型在两端页地址均连续时才合并广播区间，操作数反映实际分配下可合并的区间变少。100 Gbps 的原生 GET p50 从 1,834 降到 665 ms，200 Gbps 从 897 降到 219 ms，约 418 ms 的分发在 200 Gbps 的读取恢复路径上占比较大。这些分位按不同事件单独统计。另有配对等待 p95 约 198 ms 和 78 ms，包含对端剩余 GET 与配对交换。[原型实现](../pd_kv_transfer_opt/experiments/same_head_kv.py) · [逐轮事件](../results/2026-11-18-r15-same-head/summary.json)

从脱敏事件重算正式 A/B。

```bash
python -B -m pd_kv_transfer_opt.profiling.analyze_same_head_ab results/2026-11-18-r15-same-head/r15-P.jsonl.gz results/2026-11-18-r15-same-head/r15-D.jsonl.gz --interleaved --output runs/r15-comparison.json
```

## r14 远端恢复限速

双机各八卡 910B2，2P2D，Qwen3.5-35B-A3B BF16，attention TP4 / expert EP4。16 请求/s 的 12 轮共完成 1,152 条请求。另有 48 条低负载请求和 72 条诊断请求。每个正式轮 96 条，每条固定输出 16 token，每轮共 1,536 个输出 token，远端读取 914.16 GB。测试协议和输入指纹见 [负载](workloads.md)。

| D 侧八口发送上限 | 实测 TX Gbps | 请求/s | 输出 token/s | TTFT p99 | 最慢 rank GET p50 |
|---|---:|---:|---:|---:|---:|
| 100 Gbps | 92.77 | 1.053 | 16.85 | 85.97 s | 1.707 s |
| 200 Gbps | 182.11 | 2.069 | 33.11 | 41.06 s | 0.828 s |
| 400 Gbps | 310.49 | 3.523 | 56.37 | 21.82 s | 0.408 s |
| 原始设置 8 × 200 Gbps | 511.05 | 5.355 | 85.68 | 12.61 s | 0.185 s |

吞吐为三轮均值，延迟为三轮各自百分位的中位数。TX 取各轮最高采样区间均值的中位数。100→200 Gbps 的三个吞吐增幅为 101.5%、97.0% 和 91.1%，均值口径提高 96.5%。0.5 请求/s 时两档均维持约 0.5 请求/s，100/400 Gbps 的 TTFT p50 约 2.35–2.43/1.67–1.68 秒。

100→200 Gbps 的平均 TTFT 缩短 26.87 秒。P 调度排队缩短 12.68 秒，调度到批次结果缩短 8.05 秒，调度器接收前缩短 5.78 秒。P 设备 forward p50 在 154–201 ms，P→D 交接 p50 在 119–122 ms。逐请求关联把 P 的同步 Store GET 与 CANN 的设备 `hccl_batchGet` 对齐。100 Gbps 窗口内批量读覆盖 P0 rank 0 GET 墙钟的 99.73%，与设备计算重合 0.022 ms。并行抓到 EngineCore 等 worker 返回及 worker 等 GET 返回。带宽限制首先拉长远端读取，引擎取件和排队进一步放大。

D0 API 侧 profiler 停止曾返回 HTTP 500。三档模型 worker 的捕获均关闭并离线解析，P0/D0 rank 0 各自对齐每档 12 条请求。正式轮结束后才采诊断窗口。所有 GET 成功，D 侧重试计数增量为零。实验结束后端口恢复为每口 200 Gbps，两机共 16 卡健康空闲。

[逐轮指标](../results/2026-11-18-r14/rounds.csv) · [阶段与抓栈](../results/2026-11-18-r14/summary.json) · [CANN 对齐](../results/2026-11-18-r14/profiler-summary.json) · [校验清单](../results/2026-11-18-r14/manifest.json)

## 有效的支持性对照

- r13 为验证性限速实验，100→200→400→100 的回切展示吞吐随带宽往返变化。[r13 摘要](../results/2026-11-18-supporting/hard-cap-pilot.json)
- r1/r3/r4 在 128 GiB 池上对自然会话做 P 本地 APC 回切。开启 APC 时的请求完成率为 0.762 请求/s，关闭 APC 的两轮为 0.397 和 0.483 请求/s。三轮各完成 170 条，保留实际 EOS 造成的输出量差异。[本地 APC 对照](../results/2026-11-18-supporting/natural-apc.json)
- r9/r10 的 170 条自然会话使用 P 本地 APC。P 侧图捕获档位从 16 扩到 4096，满批 rank 0 forward p50 从约 361 ms 降到 210 ms，整轮 TTFT p99 从 97.23 秒降到 63.38 秒。实际输出量随 EOS 变化，同时保存请求完成率。[图捕获对照](../results/2026-11-18-supporting/graph-capture.json)
- r11/r12 保持自然回放的 170 条请求、路由和逐请求输出长度一致。双向原生后台 READ 将 P→D 单 rank 有效速率 p50 从 173.34 降到 97.06 Gbps。完整集合吞吐下降 0.87%，P 同步 GET 仅占各 rank 窗口约 0.23%–0.29%。该对照体现远端等待占比对吞吐效应的影响。[自然回放带宽竞争](../results/2026-11-18-supporting/natural-rdma-competition.json)

r14 的吞吐近倍增来自高远端复用、小输出、P 本地 APC 关闭的受控负载。r19 补充了 APC 开启后，D→P 恢复占据首 token 关键路径且 D 长输出主导排空的条件。解释实际服务时，应同时核对真缺页、每方向的端口服务、批次等待和输出分布。
