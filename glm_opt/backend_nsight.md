**这轮 A/B 已经得到一个明确的工程结论：在你当前测试的四种负载下，应继续保留 TRTLLM MoE 后端，CUTLASS 没有带来加速，尤其单请求 decode 退化明显。**

你观察到的 stream 占比变化是另一类信息：**它反映“记录到的 GPU 工作如何分布到不同 stream”，不是 GPU 利用率，也不是 shared/routed 两个分支的固定占比。** 不同场景中编号变化、占比变化，需要分别解释。

下面先量化 A/B，再说明这些 stream 数字能支持什么结论。

## 一、A/B 结果：CUTLASS 的主要退化出现在生成阶段

以下全部根据你**本次提供的 `timing` 行**计算；箭头统一表示 **TRTLLM → CUTLASS**。

| 场景 | TTFT，ms | TPOT，ms/token | E2E，ms | 输出吞吐，tokens/s |
|---|---:|---:|---:|---:|
| **P8192** | 287.19 → 302.24 | 不适用 | 287.56 → 302.68 | 3.473 → 3.298，**−5.04%** |
| **D512** | 49.04 → 49.93 | 7.275 → 9.632，**+32.40%** | 507.40 → 656.82 | 126.05 → 97.37，**−22.75%** |
| **B8** | 280.38 → 302.94 | 11.039 → 12.880，**+16.67%** | 975.90 → 1114.40 | 523.41 → 458.35，**−12.43%** |
| **SHARED8_HOT** | 257.43 → 283.56 | 11.538 → 12.861，**+11.47%** | 984.36 → 1093.88 | 518.41 → 466.32，**−10.05%** |

这里的百分比均以 TRTLLM 为基准。P8192 只输出一个 token，评估 prefill 更适合看 TTFT，或者新输入处理吞吐：后者约由 **28,451 降到 27,016 输入 tokens/s**。

### 1. D512 是最值得继续解释的对照

它的输入、输出和并发都最简单，而且：

```text
TTFT 只增加：        0.891 ms
TPOT 每 token 增加： 2.357 ms
E2E 总共增加：      149.418 ms
```

64 个输出 token 对应首 token 之后的 63 个生成间隔：

\[
63\times(9.632182-7.274923)
=\boxed{148.507\text{ ms}}
\]

也就是说，**约 99.4% 的 E2E 增量出现在首 token 之后**，而不是初始 prefill。

这使下一步的问题非常明确：

> **CUTLASS 路径为什么使单请求生成阶段平均每 token 多花约 2.36 ms？**

但这仍是客户端时间分解，**不能直接把 2.36 ms 全部算到两条专家 GEMM 上**；量化、路由、finalize、同步、Graph 执行及输出处理都还需要按时间线核对。

### 2. 并发八请求也退化，但不能与 D512 简单归为同一种缩放规律

B8 每条请求增加约 **138.49 ms**，其中约 **115.93 ms**来自首 token 之后的间隔增长。

SHARED8_HOT 每条请求增加约 **109.52 ms**，其中约 **83.38 ms**来自首 token 之后。

两组 HOT 的命中率都是 **0.9375**，因此，**本次 HOT 对照的差异不能解释为“一组没有成功命中缓存”**。

“单请求相对退化更明显”支持优先检查小规模执行路径，但**目前还不能断言原因一定是小 M GEMM，也不能证明 CUTLASS 在更大 batch 下会反超**。

### 3. 这是一轮有效的否定结果，不是白做了

这次已经验证：

> **“将 MoE 后端换成 CUTLASS 会改善当前工作负载”的假设，未得到支持。**

不过，目前每个后端只提供了一轮汇总，单请求只有3条、并发场景24条，尚无跨轮波动和质量结果。因此可用于当前后端选择，**不能扩展成“CUTLASS 在所有模型和规模上都比 TRTLLM 慢”。**

还应从两份现成 `server.log` 确认：Graph/eager 模式、autotune、额外诊断包装是否保持一致。你此前确实遇到过 CUTLASS 调优缓存命中在不同 rank 上不一致的问题；若后来为了启动关闭了 autotune，这次比较的就不仅是后端名称，还包括最终采用的调优配置。粘贴的文本 (1)

---

## 二、不要用这轮 `trace` 的请求耗时替代上述结论

你这次的 P8192 是两组的第一个 trace：

```text
TRTLLM：timing TTFT 287.19 ms，trace TTFT 2336.92 ms
CUTLASS：timing TTFT 302.24 ms，trace TTFT 1227.61 ms
```

若只看 trace，会误以为 CUTLASS 的 prefill 快了约47%；但 timing 恰好相反。

**这个首轮 trace 明显包含额外延迟，不能拿来给后端性能排序。** 当前数据不能单独证明额外延迟具体来自哪一步。

B8 和 SHARED8_HOT 也存在抵消现象：

| 场景 | CUTLASS 相对 TRTLLM 的 trace TTFT | trace TPOT |
|---|---:|---:|
| B8 | **减少35.13 ms** | **增加1.128 ms/token** |
| SHARED8_HOT | **减少52.53 ms** | **增加1.183 ms/token** |

因此，trace 的 E2E 差距看起来只有约3.1%和1.8%，**不代表实际退化只有这么多**。

继续沿用我们的分工：**timing 给当前性能对照，trace 解释执行路径。** 最终验收再用完全不挂 profiler 的运行；Nsight 的 node 级 Graph tracing 本身也可能增加采集开销。[NVIDIA Docs](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html)

---

## 三、stream 编号为什么会变？它不等于固定的功能模块

### 1. `19` 和 `29445` 首先只是标识符

不能这样解释：

```text
19 比较小     → 必定是默认流
29445 比较大  → 必定是专家流／通信流
29445         → 当前创建了29445条活跃stream
```

Nsight/CUPTI 的 stream ID 用来标识执行所关联的 stream，**编号不编码这些业务含义，默认 stream 的工具侧编号也不是跨版本固定值**。[NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/meaning-of-various-ids-in-the-exported-json-for-nsight-systems/176041?utm_source=chatgpt.com)

因此，应把之前的局部判断限定清楚：

> 在你曾贴出的那一段 MoE 时间线中，Stream 19 执行了 shared expert 链，另一条执行了 routed expert 及主体链路。**这不意味着所有 case 中这两个数字都永久承担同样的角色。**

### 2. 你的 CUDA Graph 会进一步影响执行流的展示

你之前成功启动的日志同时记录了 **Breakable CUDA Graph、PIECEWISE capture 和 FULL capture**，说明运行环境并非只有一张固定执行图。粘贴的文本 (1)

CUDA Graph 捕获的是操作和依赖关系。Replay 时由 CUDA 调度，驱动还可能创建额外的内部 stream；**源码里使用的 stream 对象与 Nsight 展示的所有执行 stream，不一定一一对应**。[NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

结合你的环境，一个合理解释是：

```text
不同实际 batch／不同阶段
    ↓
使用不同的 Graph 实例、执行路径或图外操作
    ↓
关联到不同的内部执行 stream
    ↓
报告中出现不同的大编号
```

**这是解释编号变化的机制，不是仅凭 `29445` 就已经确认了它的创建来源。** 要确认具体身份，应看该 stream 上的 kernel 家族，以及它们关联的 `cudaGraphLaunch`，而不是追数字大小。

---

## 四、stream 占比的准确含义：累计事件时间，不是墙钟占用率

NVIDIA 对这个行首百分比的定义是：某个 stream 的运行时间，占所在 context 总运行时间的比例；总运行时间按该 context 中的 kernel 和 memory operation 的 Duration 累加。[NVIDIA Docs](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

可写成：

\[
P_s=
\frac{\sum_{\text{stream }s}\operatorname{Duration}(\text{kernel/memory op})}
{\sum_{\text{该context全部stream}}\operatorname{Duration}(\text{kernel/memory op})}
\times100\%
\]

因此：

**两条 stream 合计约98%，表示约98%的已统计事件累计时长落在这两条流上。** 它不表示 GPU 利用率98%、两流重叠98%，也不表示只剩2%的优化空间。

举一个纯示意例子：

```text
Stream A 累计执行10 ms
Stream B 累计执行 5 ms
```

无论它们完全串行、总共用15 ms，还是充分重叠、总共约10 ms，按上述累计口径，A/B都可能显示约 **66.7%／33.3%**。

**占比相同，墙钟时间和重叠效果却可以完全不同。** 所以不应追求“两条 stream 各50%”，而应追求目标步骤更早完成。

---

## 五、你这七个 case 的变化，应分三层解释

### 第一层：是否把 prefill／mixed／decode 混在一起统计了？

这是目前最应先核对的因素。**如果表格来自每份报告整体的 stream 行首百分比，它就不是“稳定纯 decode 的 stream 占比”。**

你已经亲自对过 SHARED8 的输入计算量：

```text
SHARED8_COLD：65536 个新输入 tokens
SHARED8_HOT：  4096 个新输入 tokens
```

两者后面都生成64个 token，但开头的工作量和 mixed step 组成差很多。

对于同一逻辑流分组，整个窗口的占比本质上是阶段加权结果：

\[
P_s=
\frac{D_{\mathrm{prefill}}P_{s,\mathrm{prefill}}
+D_{\mathrm{mixed}}P_{s,\mathrm{mixed}}
+D_{\mathrm{decode}}P_{s,\mathrm{decode}}}
{D_{\mathrm{prefill}}+D_{\mathrm{mixed}}+D_{\mathrm{decode}}}
\]

这里的 \(D\) 是各阶段**累计事件时间**，不是阶段墙钟时间。

因此，**即使 Cold/Hot 的稳定 decode 完全一样，整个报告的 stream 占比仍然可以明显不同。** 如果你确实已经分别限制到同等的纯 decode 窗口，那么这一层解释就应排除，转看下面两层。

### 第二层：框架本来就可能按输入规模选择不同重叠策略

vLLM 的 `SharedExperts` 实现会根据后端是否内部支持重叠、输入行数、辅助 stream 是否存在及安全条件，选择内部重叠、独立 stream 重叠或不重叠。**不是所有阶段都强制采用同一个双流安排。**[vLLM](https://docs.vllm.ai/en/v0.29.0/api/vllm/model_executor/layers/fused_moe/runner/shared_experts/?utm_source=chatgpt.com)

公开配置中有：

```text
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
```

文档列出的默认值为256，但**本机版本及环境变量以实际配置为准**。判断对象是 MoE 入口的 token 行数，不是客户端并发上限。[vLLM](https://docs.vllm.ai/en/v0.30.0/configuration/env_vars/?utm_source=chatgpt.com)

例如，在使用这个外部多流分支、阈值为256的条件下：

```text
纯 decode：1、8、32 行，可能满足重叠条件
大 prefill／mixed：数千至上万行，可能不满足
```

这可以改变工作分布在哪条执行流上。**但 B8 与 B32 都低于256，所以不能拿这个阈值直接解释它们之间的差异。**

### 第三层：同为纯 decode，不同工作量也会改变分子、分母

并发变化会改变本步有效 token 数、每专家分配、GEMM 形状、Attention 工作量和通信消息规模；不同算子的耗时不必同比增长。

这里还要看实际执行 batch，而不是只看 C 上限。Graph 也可能按 uniform decode、prefill/mixed 等条件使用不同模式。[vLLM](https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/?utm_source=chatgpt.com)

结合这些边界，你的表可以这样读：

| 对照 | 观察 | 当前合理解释及限制 |
|---|---|---|
| **D512、D8192_HOT、D32768_HOT** | 72.4%、69.3%、71.5%，都约70% | C=1下分布相近；历史长度及剩余prefill造成的差异仍可能存在。不能据此认定Attention成本不随长度变化 |
| **B8 → B32** | 47.7% → 61.6% | 实际batch、专家负载和kernel选择改变后，两条链的成本不同比例增长；也要排除Graph执行流映射差异。不是“并发越大占比越高”的普遍规律 |
| **SHARED8_COLD → HOT** | 24.7% → 61.9%，相差**37.2个百分点** | **若统计整份报告，阶段混合是首要解释候选**。Cold大量prefill/mixed会改变分母、执行路径及stream分布 |
| **B8 → SHARED8_HOT** | 47.7% → 61.9% | 两者虽然都是C=8，但逻辑历史是512与8192，输入内容、到达／调度过程也不同；若限定纯decode后仍不同，再检查kernel/tactic和流归属 |

**仅凭这七个百分比，无法唯一算出每个差异的具体来源。** 但我们已经知道该查什么：阶段混合、执行图映射、各类工作绝对耗时，而不是“stream编号为什么这么大”。

尤其注意，**SHARED8_HOT 重新计算512个输入 token，不等于 decode 的历史长度只有512**。不要用这两种512把它与B8当作完全相同的工作负载。

---

## 六、接下来如何最小化地查明“CUTLASS 为什么慢”？

**先只看 D512 的两份 nsys 报告，不重跑，不同时研究全部七个场景。** 它的性能差距最大，而且没有八请求之间的 mixed 调度干扰。

### 1. 先确认比较的是相同执行条件

从现成日志核对两组都已恢复正常计分状态：同一版本、相同 Graph/eager 设置、相同采样参数，没有抓取快照的同步／落盘逻辑。记录各自 autotune 是否完成即可，不要求 tactic 相同——选择不同 tactic 本来就是后端差异的一部分。

同时，客户端若保存了 `generated_token_ids`，可以检查两边是否明显分叉。固定输出长度只能保证token数量相同，**不能保证数值差异后每一步的MoE路由完全相同**；不要把这一点误当作必然的测量错误，但应记录。

### 2. 各选五个完整的纯 decode 步，按“功能”汇总

使用对应的：

```text
execute_context_0(0)_generation_1(1)
```

关联到GPU工作，限制同一GPU、相同数量的完整步骤。**用过滤后的 Events View 中的 Start 范围核验，不只看高亮或缩放后的行首百分比。**

这次需要的对照表是：

| 对象 | TRTLLM | CUTLASS | 想回答什么 |
|---|---:|---:|---|
| 五步推进时间／每步周期 | 记录 | 记录 | 客户端TPOT差距是否在GPU执行中重现？ |
| Router＋路由准备 | 记录 | 记录 | 是否新增排序、重排或准备工作？ |
| FC1＋激活／量化＋FC2 | **完整链路** | **完整链路** | 是矩阵运算本身变慢，还是融合边界不同？ |
| Finalize／输出整理 | 记录 | 记录 | 收尾成本是否增加？ |
| Shared expert 分支 | 记录 | 记录 | 重叠是否保留？ |
| 归约与可见空洞 | 记录 | 记录 | 是否更多时间在等前序工作或其他rank？ |

**不能要求两个后端必须出现相同的 kernel 名和数量。** 一个融合kernel对应另一个后端的多个kernel时，应比较相同逻辑计算的完整链，而不是只比其中一条GEMM。

### 3. 用一个量级检查，防止追错热点

D512 的平均生成间隔差是约 **2.357 ms/token**。假设差异主要来自75个MoE层，平均相当于：

\[
2.357\text{ ms}/75
\approx\boxed{31.4\ \mu s/\text{MoE层}}
\]

**这是“待解释差额”的预算，不是已经测出的每层退化。**

它可以帮助判断候选是否足够大：

- 若CUTLASS的完整MoE链在很多层上确实多出几十微秒，这个解释量级合理。
- 若每层只多出1～2 μs，就不足以解释全部差距，应继续查Graph、同步和其他路径。
- 若kernel累计时间相近，但步骤变长，就重点检查重叠和依赖，而不是继续追逐stream占比。

---

## 可以直接用于汇报的阶段性结论

> 在当前GLM-5.2-NVFP4、TP4/EP配置和四组负载下，TRTLLM MoE后端的timing结果均优于CUTLASS。CUTLASS在D512、B8、SHARED8_HOT中的平均TPOT分别增加32.4%、16.7%、11.5%；其中D512的额外端到端时间几乎全部发生在首token之后，后续应优先解释单请求生成路径的差异。
>
> Nsight中的stream行首百分比是累计GPU事件时间的分布，不是硬件利用率或并行效率。不同场景的百分比还受到prefill/mixed/decode组成、Graph内部执行流映射及kernel规模变化影响。不能把大编号stream固定等同于MoE流，也不能以两条stream是否各占50%判断优化效果。

**当前决策是保留TRTLLM；下一步只用D512两份已有trace，把约2.36 ms/token的差距落实到完整MoE链、重叠或同步上。** 这样这轮A/B就从“知道谁更快”，推进到“知道具体该改哪里”，而不是继续比较难以归因的百分比。
