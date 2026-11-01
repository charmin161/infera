# ForgeKernel 技术调研：原理、输入输出、依赖及多硬件适配

调研时间：2026 年 10 月 31 日｜主要依据：OpenBMB 官方仓库、技术文档与公开实现

## 一、ForgeKernel 到底是什么？

我查到的是 ForgeKernel，由 OpenBMB 开源的 AI 驱动 GPU Kernel 自动生成与性能优化系统。

官方仓库：[OpenBMB/ForgeKernel](https://github.com/OpenBMB/ForgeKernel?utm_source=chatgpt.com)

它并不是类似 Triton、CUTLASS 的新编程语言，也不是 vLLM、SGLang 那样的推理框架。

更准确地说：

ForgeKernel = AI 编程 Agent + GPU 性能评测工具链 + 自动优化循环 + 硬件知识库。

它的主要功能是：给定算子的数学定义、输入规模、正确性约束和已有实现，让 AI 自动编写更高性能的 GPU Kernel，并通过反复编译、执行、验证、Profiling 和改进寻找更优实现。

例如，你有一个使用 PyTorch 实现的 RMSNorm，想把它优化成高效的 B200 Kernel。

传统方法是工程师自己分析内存访问、线程块布局、寄存器使用、Warp 分工等，然后手写 CUDA 或 CuTe DSL。

ForgeKernel 则试图自动完成这套工作。

用户提供

算子定义 + 输入 Shapes + Reference Kernel + 目标 GPU + 性能要求

ForgeKernel 自动优化系统

Agent 分析算子

生成 CUDA / CuTe DSL

GPU 执行与正确性验证

Benchmark / Profiling

根据结果继续优化

保存最佳版本

最终输出

优化后的 Kernel 源码 + 正确性结果 + 性能报告 + 优化过程记录

官方公开实现以 NVIDIA H100（Hopper）和 B200（Blackwell）为主要目标。其 README 声称，在 27 个算子的实验中，达到了或超过了 22 个专家库 Kernel 中的 18 个。这里应当理解为项目报告的特定基准测试结果，而不是所有输入 Shape、精度格式和硬件上的普遍结论。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

OpenBMB/ForgeKernel · GitHub+1



还有一个值得注意的区别：它既能从算子定义出发生成 Kernel，也能从现有高性能 Kernel 出发继续优化；后者是项目目前更有证据支持的强项。

## 二、用户需要输入什么？ForgeKernel 输出什么？

这里要区分两个层次：

- ForgeKernel 作为工具的输入输出：输入优化任务，输出优化后的 Kernel 和评测报告。
- 生成的 Kernel 本身的输入输出：输入具体 GPU Tensor，输出计算结果，例如 RMSNorm 的 BF16 Tensor。

### 2.1 ForgeKernel 的任务输入

官方采用文件驱动的任务定义，主要有三个文件和一个可选的 Reference 实现。

| 输入     | 文件               | 具体作用                                 |
| ------ | ---------------- | ------------------------------------ |
| 算子数学定义 | `description.md` | 输入输出签名、Shape、Dtype、数学公式、合法数值范围       |
| 测试工作负载 | `shapes.json`    | 需要优化的 Shape 集合，可以附带真实输入数据            |
| 优化配置   | `task.toml`      | 目标 GPU、编程语言、Reference、性能目标、Agent 和预算 |
| 原始实现   | `reference/`     | PyTorch、已有 Kernel 或可调用的黑盒实现；也可以不提供   |

举一个仓库中真实存在的 RMSNorm 例子。

① `description.md`：描述算子做什么

```
# 计算语义示意# x: [m, n], BF16# w: [n], BF16# y: [m, n], BF16y = x * rsqrt(mean(x*x, dim=-1) + 1e-6) * w
```

注意其官方精度契约是统计计算至少使用 FP32，最终输出才转换为 BF16。

② `shapes.json`：指定优化哪些输入规模

```
[
  {"name": "prefill", "axes": {"m": 8192, "n": 4096}},
  {"name": "decode",  "axes": {"m": 64, "n": 4096}}
]
```

这意味着它不是仅为一个固定 Tensor 优化，而是要同时满足两个 Shape 的正确性和性能要求。每个 Shape 还可以通过 `inputs/` 和 `safetensors` 文件提供具体数值分布。

③ `task.toml`：指定硬件和优化目标

```
[task]
language = "cutedsl"
arch = "sm100"

[reference]
run = "reference/run.py:run"
source = "reference"

[target]
ratio = 1.2
timing = "device"

[budget]
hours = 2
max_meta = 1
```

这是基于官方 RMSNorm 示例加上显式性能目标写出的配置。这里 `ratio=1.2` 代表目标吞吐性能是 Reference 的 1.2 倍，即目标运行时间不超过原来的约 83.3%。

④ `reference/run.py`：提供一个可以执行的基准实现

```
import torchdef run(x, w):    xf = x.float()    y = (        xf        * torch.rsqrt(            xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6        )        * w.float()    )    return y.to(x.dtype)
```

Reference 的用途不仅是对照正确性，还用于测量基线耗时、分析 Kernel 的硬件行为和判断是否达到优化目标。

以上文件内容均可在官方 [RMSNorm 示例目录](https://github.com/OpenBMB/ForgeKernel/tree/main/tasks/rmsnorm?utm_source=chatgpt.com) 中找到。

[image](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+4



### 2.2 最终输出的交付文件

正常执行后，输出保存在 `.artifacts/<op>_<id>/` 中。

.artifacts/rmsnorm\_\<id>/

best/kernel.py

最终选中的最佳 Kernel 源码

report.md

最终性能对比、正确性和优化结果

check/

生成的 PyTorch Judge、测试输入组和正确性配置

operators/rmsnorm/

Milestone、Benchmark、Profiling 与迭代记录

ws/

Coding Agent 的工作区及代码版本

run.json / run.log

任务快照、运行状态与日志

其中 `best/kernel.py` 最重要。它是优化后的计算实现，不是重新训练出来的模型权重，也不是自动生成的 vLLM 插件。

要用于实际推理框架，还需要将它的调用接口、Tensor Layout、Launch 方式与模型的执行链集成。

官方给出的 B200 RMSNorm 示例中，最佳 Kernel 的 Device Median 为 0.0307 ms，相对 PyTorch Reference 达到 5.2363×。这是作者报告的示例结果，不能直接外推为整个模型推理快 5.2 倍。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



## 三、它的核心工作原理是什么？

ForgeKernel 的关键设计不是让 LLM 不断随机修改代码，而是将优化拆成两个不同职责的 Agent，以及一套 GPU 实测反馈机制。

### 3.1 两个 Agent 分工

| 组件                    | 职责                                | 核心输出           |
| --------------------- | --------------------------------- | -------------- |
| Arch Agent（Teacher）   | 分析 Reference、决定总体算法架构、识别瓶颈、规划优化步骤 | `milestone.md` |
| Coding Agent（Student） | 根据架构规划实现、修改和优化 Kernel             | `kernel.py`    |
| GPU Evaluator         | 执行候选实现、检查结果、计时、Profiling          | 正确性和性能指标       |
| Hardware KB           | 保存经过硬件实验验证的通用优化机制                 | 可跨算子复用的硬件知识    |

最大的设计特点是 Architecture-first（架构先行）。

Teacher 首先确定 M0 阶段的关键架构决策，包括 Kernel 如何拆分、Tile Shape、Warp 分工、数据流、共享内存和寄存器预算，以及计算和数据搬运如何重叠。

Student 在 M1、M2 等后续阶段优化这个架构。如果实测证明结构本身形成了性能上限，Teacher 才重新设计 M0。

这样能够避免 Agent 一开始选择了错误架构，后面只在错误方向上反复微调。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



### 3.2 优化如何形成闭环？

Teacher：分析 Reference，形成 M0 架构方案

Student：实现或修改 Kernel

Correctness

Tensor 结果是否满足数值容差

Performance

是否比 Reference 快

通过检查且性能改善

&#x20;KEEP

正确性失败或未获认可

&#x20;REVERT

持续迭代 / Teacher 重新规划

遇到性能瓶颈时重新评估优化空间，必要时修改 M0

保存最佳 Kernel，必要时晋升为下一代 Reference

它还设计了几个非常重要的工程机制。

第一，Reference 可以是黑盒。 Teacher 可以运行、计时、Profiling，甚至分析编译后的 SASS，而不需要看到 Reference 的源代码；Student 始终不能直接读取 Reference 的实现源码。这让它能够在受控条件下优化和对比已有专家实现。

[image](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1



第二，性能比较尽量放在同一个 GPU Job 中完成。 普通任务采用预热、交错测量 Reference 和 Candidate 的机制，减少不同 GPU 温度、时钟状态、设备节点带来的误差。

第三，正确性检查不只是比较一组随机 Tensor。 Teacher 根据数学描述生成 PyTorch Judge；普通任务每个 Shape 至少有三组输入。最终 Correctness Monitor 还测试新随机种子、极端数值、零块、NaN/Inf 和可重复性。

但它使用的是数值容差判定，不保证 bit-exact。最终 Monitor 的告警也不等价于每一次都自动否决候选实现。对低比特数值规格验证而言，仍然需要额外构造严格测试。

[image](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1



### 3.3 为什么叫 Self-Evolving？

它不满足于一次优化成功。

例如：

Reference 0

Kernel 1

Kernel 2

Kernel 3

每次优化成功后，新的 Kernel 成为下一轮的 Reference。下一轮不是再去超过最初的 PyTorch，而是必须超过上一轮最佳 Kernel。

对应入口是：

```
python scripts/self_evolve.py tasks/<op> \
  --step 1.05 --max-gen 5
```

这意味着每一代的目标可设为相对上一代提升 5%。不过目标不是保证值；优化可能失败，性能提升低于实测噪声门槛时会停止。当前官方说明中，这个自动晋升脚本仅支持 CuTe DSL 任务。

### 3.4 Hardware Knowledge Base 的作用

这是 ForgeKernel 区别于普通 Coding Agent 的另一处重要设计。

例如某轮优化通过硬件实验确认了一种特定的 TMA 数据搬运方式或异步 MMA 同步机制，系统可以将其总结为通用知识，供以后优化其他算子时复用。

知识库按编程语言和硬件架构组织，避免把不同 GPU 的指令细节混在一起。

但这里的“学习”不是更新 LLM 权重，而是积累可读取的代码示例、实验记录和硬件优化规则。知识条目在正式纳入前还需要人工审查；官方文档也明确说明，Agent 自己编写的验证 Probe 并不构成独立正确性保证。

[image](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1



## 四、ForgeKernel 依赖哪些软件和硬件？

它采用控制端与 GPU 执行端分离的设计。

控制端：Linux / macOS

Python + ForgeKernel + Claude Code + LLM API

负责调度 Agent、管理任务、处理实验结果

SSH 发送任务 / 回传性能数据

GPU 执行端：H100 / B200

CUDA + PyTorch + CuTe DSL + Nsight Compute

负责编译、执行、正确性测试和 Profiling

### 4.1 具体依赖

| 组件                     | 是否必要                 | 用途                         |
| ---------------------- | -------------------- | -------------------------- |
| Python ≥ 3.11          | 必需                   | ForgeKernel 主程序            |
| Git                    | 必需                   | 管理任务快照、Kernel 版本及回滚        |
| Claude Code CLI        | 当前 V1 必需             | 执行 Teacher 和 Student Agent |
| LLM API / Claude 登录    | 必需                   | 提供模型推理能力                   |
| PyTorch CUDA 版本        | GPU 端必需              | Tensor 输入、正确性验证、Benchmark  |
| CUDA Toolkit / Driver  | NVIDIA 端必需           | 编译和运行生成的 Kernel            |
| `nvidia-cutlass-dsl`   | CuTe DSL 任务的 GPU 端必需 | 编译 CuTe DSL Kernel         |
| Nsight Compute (`ncu`) | 推荐                   | 获取硬件计数器、访存与执行瓶颈            |
| SSH                    | 远程 GPU 模式必需          | 控制端和 GPU 端通信               |
| tmux                   | 部分模式需要，其他模式推荐        | 长时间运行、断线恢复                 |

这里有三个重要细节。

1\. Agent 并不是任意模型都能直接替换。

当前 V1 公开实现的 Agent 后端只有 `backend = "claude"`，即 Claude Code CLI。`global_config/models.toml` 可以配置模型；`backends.toml` 支持指定兼容 Anthropic 协议的 API 端点，但不等于已经支持任意 Coding Agent 或任意模型 API。

[image](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+2



2\. Nsight Compute 不可用时还能运行。

缺少 `ncu`，或者服务器没有性能计数器权限时，ForgeKernel 会退化成仅依赖 Timing 的优化流程，但无法完成部分 Stall 分析和硬件机制定位。

3\. 不要直接使用 `pip install forgekernel`。

我发现 PyPI 上已有一个同名的、用于 CAD 几何计算的 `forgekernel` 包，它并不是 OpenBMB 的 GPU 优化项目。正确做法是从 OpenBMB 官方 GitHub 仓库克隆后执行 `pip install -e .`。

[image](https://www.google.com/s2/favicons?domain=https://pypi.org\&sz=32)

PyPI+1



### 4.2 官方验证过的环境

|          | NVIDIA H100      | NVIDIA B200        |
| -------- | ---------------- | ------------------ |
| GPU 架构   | Hopper / SM90a   | Blackwell / SM100a |
| Driver   | 550.90.07        | 580.105.08         |
| CUDA     | 12.8（另有 13.1 环境） | 13.0 / 13.1        |
| PyTorch  | 2.6 / 2.10       | 2.10 / 2.13        |
| CuTe DSL | 4.4.2 / 4.6.x    | 4.4.2 / 4.6.x      |
| Profiler | Nsight Compute   | Nsight Compute     |

这是项目报告中使用的环境组合，不代表其他组合必然不兼容。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

OpenBMB/ForgeKernel · GitHub+1



## 五、它对不同硬件到底是怎么适配的？

这是我认为最需要厘清的问题。

ForgeKernel 的跨硬件能力主要来自优化工作流的抽象，而不是一套能直接编译到所有芯片的统一 Kernel IR。

### 5.1 当前硬件支持情况

| 硬件                   | 公开状态              | 编程/工具链            |
| -------------------- | ----------------- | ----------------- |
| NVIDIA H100          | 已公开实现与实验          | CuTe DSL、CUDA、NCU |
| NVIDIA B200          | 已公开实现与实验          | CuTe DSL、CUDA、NCU |
| 华为 Ascend A3         | 已报告 CANN-Bench 成绩 | 开源仓库尚无完整适配交付      |
| Ascend 950PR / 950DT | 已报告 CANN-Bench 成绩 | 开源仓库尚无完整适配交付      |
| AMD MI 系列            | 未发现官方开箱即用支持       | 需要新后端，如 HIP/ROCm  |
| 其他 NVIDIA GPU        | 非当前主要验证范围         | 需检查架构及运行兼容性       |

OpenBMB 官方 README 确实宣称其在华为 CANN-Bench 榜单取得第一名，媒体也报道了 A3、950PR、950DT 的成绩。但官方 GitHub 目前仍有一个关于开放 Ascend 端代码的 [Issue #1](https://github.com/OpenBMB/ForgeKernel/issues/1?utm_source=chatgpt.com)，尚未看到公开的完整 Ascend Harness 和对应 Kernel。

因此，团队完成过昇腾适配与你今天克隆仓库就能直接在昇腾上运行是两回事。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+2



### 5.2 NVIDIA H100 和 B200 怎么切换？

最直接的配置是：

```
[task]
language = "cutedsl"

# H100
arch = "sm90"

# B200 则改成
# arch = "sm100"
```

`arch` 会影响编译目标 `CUTE_DSL_ARCH` 和匹配的 Hardware KB 目录。

但这不等于同一份生成的 Kernel 源码可以不修改就高效地运行于两代 GPU。

例如：

| 优化维度               | H100 / Hopper     | B200 / Blackwell           |
| ------------------ | ----------------- | -------------------------- |
| 矩阵计算               | WGMMA             | TCGen05                    |
| 数据搬运               | TMA、Shared Memory | TMA、Shared/Tensor Memory 等 |
| MMA 执行组织           | Warpgroup 路径      | 新一代 Tensor Core 路径         |
| 最优 Tile 和 Pipeline | 需要针对 H100 实测      | 需要针对 B200 重新实测             |
| 低比特能力              | 受 Hopper 硬件格式限制   | 支持更多原生低比特 MMA 形式           |

NVIDIA 的 CUTLASS 文档明确区分 SM90 的 WGMMA 与 SM100 的 TCGen05 机制。

[image](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com\&sz=32)

NVIDIA CUTLASS Documentation+2



所以 ForgeKernel 迁移到另一个架构，本质上是：

同一个算子数学问题，使用目标硬件对应的编程接口、知识库和实测反馈，重新探索合适的 Kernel 实现。

另外有个值得注意的源码文档细节：当前 Harness 不会严格检查 `task.toml` 中的 `arch` 是否与 GPU 真实架构一致。因此不能把 `arch` 配置当成自动设备识别，使用前必须自己确认。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 5.3 华为和 AMD 需要怎样适配？

从工程架构看，可以把需要替换的内容概括成下面几层。

| 层次          | NVIDIA            | 华为昇腾（移植方向）     | AMD（移植方向）       |
| ----------- | ----------------- | -------------- | --------------- |
| Agent 优化循环  | Teacher / Student | 理论上复用          | 理论上复用           |
| Kernel 语言   | CUDA / CuTe DSL   | Ascend C 等     | HIP / Triton    |
| 编译运行        | CUDA              | CANN           | ROCm            |
| Profiling   | Nsight Compute    | msprof 等       | rocprofv3 等     |
| 设备 Tensor   | PyTorch CUDA      | torch_npu 等    | PyTorch ROCm    |
| Hardware KB | SM90 / SM100      | Ascend 专属      | CDNA/gfx 专属     |
| 正确性和性能评测    | CUDA Judge        | 需要适配 NPU Judge | 需要适配 ROCm Judge |

上表的华为、AMD 栏是合理的移植技术路线，并不是声称 ForgeKernel 已将这些后端全部公开实现。昇腾 Ascend C 和 AMD ROCprofiler 的作用分别可以在厂商文档中确认。

[image](https://www.google.com/s2/favicons?domain=https://www.hiascend.com\&sz=32)

hiascend.com+2



因此移植一个新架构，至少涉及四项工程工作：

1. 让 Agent 能生成目标芯片支持的 Kernel 语言，并正确编译和运行。
2. 让 Evaluator 能在目标设备上检查结果、计时和采集性能指标。
3. 把 GPU 架构、访存、计算指令和优化机制整理成对应 Hardware KB。
4. 根据新硬件实际测量结果修改架构规划与优化策略。

更换 NCU 为 msprof 或 rocprofv3 只是其中一环，而不是整个适配过程。

真正跨硬件复用的是优化方法与实验反馈机制；生成的高性能 Kernel 通常仍然是硬件专用的。

## 六、用户到底怎么运行它？

下面使用官方 RMSNorm 任务演示完整流程。

### 安装

```
git clone https://github.com/OpenBMB/ForgeKernel.git
cd ForgeKernel

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
pip install nvidia-cutlass-dsl

forge --version
```

### 配置 GPU

```
cp local.env.example local.env
```

本机 GPU 模式：

```
FK_DELIVERY=local
FK_GPU=0
```

远程 GPU 模式：

```
FK_DELIVERY=ssh
FK_GPU=0
REMOTE_HOST=gpu-b200
REMOTE_WORKDIR=/data/forgekernel_jobs
```

SSH 主机名需要对应真实可免交互登录的服务器。远程端须预先具备 CUDA、PyTorch 和相应的编译环境，可通过 `REMOTE_ENV_FILE` 加载 CUDA 和 Python 虚拟环境。

### 创建、运行、查看结果

```
# 仅检查任务并输出执行计划
forge new tasks/rmsnorm --dry-run

# 创建任务实例，记录生成的具体目录名
forge new tasks/rmsnorm

# 启动优化，把 <id> 换成实际生成的 ID
forge run .artifacts/rmsnorm_<id>

# 查看进度
forge status .artifacts/rmsnorm_<id>

# 导出最佳 Kernel 源码
forge export .artifacts/rmsnorm_<id> \
  --out optimized_rmsnorm.py
```

长时间运行建议放在 tmux 会话里。同一个 `forge run` 命令也可以用于中断后的恢复。

对于 FlashInfer-Bench / SOL-ExecBench 等排行榜任务，ForgeKernel 还有专门的 `forge init --leaderboard` 模式，使用排行榜自身的输入、容差和计时规则，最终可导出 `solution.json`，而不是普通任务的源代码文件。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



## 七、结合你的低比特与推理 Kernel 工作，是否值得使用？

我认为值得做一个小规模验证，尤其是在 B200 上。

不过，需要区分它适合的场景和不适合的场景。

| 场景                         | 适合程度       | 判断                               |
| -------------------------- | ---------- | -------------------------------- |
| RMSNorm、RoPE、SwiGLU Fusion | 高          | 数学定义清楚、易自动验证                     |
| FP8 / NVFP4 GEMM           | 高          | 可以探索 Tile、MMA、Pipeline 和访存优化     |
| MLA Decode / Prefill       | 高，但复杂      | 需要严格定义 Cache Layout、Shape 和 Mask |
| 特定模型的 Kernel Fusion        | 高，但复杂      | 适合从已存在的正确实现继续优化                  |
| FP4 量化误差建模                 | 有条件        | 可优化 Q/DQ Kernel，但数值标准需自行规定       |
| Tensor Core bit-exact 验证   | 不宜直接替代原有工具 | 默认正确性 Judge 不保证 bit-exact        |
| 自动完成 vLLM / SGLang 集成      | 不直接支持      | 生成 Kernel 后仍需工程集成                |

官方已公开 FP8 GEMM、FP8×FP4 Grouped GEMM、MLA Paged Decode/Prefill 等算子实现，可以从这些结果出发研究其优化能力。

[image](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub



对于你的单机 4×B200 环境，最小实验只需要其中 1 张 B200。可以在可联网的 Linux 控制机运行 ForgeKernel 和 Claude Code，通过 SSH 将 GPU Job 交给无公网的 B200 服务器。GPU 端不一定需要访问模型 API，但必须提前安装好所需运行依赖。

我建议验证顺序为：先复现官方 RMSNorm，确认编译、Profiling 和迭代闭环；再选择 FP8×FP4 Grouped GEMM 或 MLA Decode，使用已有正确实现作 Reference，并补充低比特数值容差、极值输入、不同 Shape 的测试。这样才能判断其收益来自真实架构优化，而不是过拟合单一 Benchmark 或放宽数值标准。

## 八、最终判断

ForgeKernel 的创新重点不在于发明一种新的 Kernel 编程语言，而在于把过去高度依赖专家经验的算子优化过程，变成一个能够自动设计架构、写代码、做硬件实验、接受或回滚优化、积累知识的工程系统。

但当前还存在明显边界：公开版本以 NVIDIA Hopper 和 Blackwell 为主；非 NVIDIA 后端的完整代码尚未公开；优化成本包含 GPU 时间和大模型调用；正确性检查并非形式化证明；Benchmark 性能也不必然转换为端到端模型加速。

它更适合被定位为 GPU Kernel 研发和优化 Agent，而不是通用自动编译器或跨芯片 Kernel 翻译器。

最后，值得分开阅读两份研究资料：

- [ForgeKernel: Self-Evolving Kernels Catching and Surpassing Experts and Agents](https://openreview.net/pdf?id=EtV3f5BPD4\&utm_source=chatgpt.com)：直接对应 ForgeKernel 架构和结果，目前标注为 ICLR 2027 在审论文。
- [JustGoal: With Only a Goal, the AI Baseline Is Above the Expert Kernels](https://arxiv.org/abs/2610.36434?utm_source=chatgpt.com)：研究不预置专用优化知识和流程时，通用 Agent 的 Kernel 优化能力，属于相关但不同的研究。

两者的区别很值得研究：JustGoal 关注基础模型本身能够做到多好；ForgeKernel 则进一步探索架构规划、硬件知识积累和多代自进化能否带来额外收益。

[image](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1



核心参考文档： [V1 架构与任务协议](https://github.com/OpenBMB/ForgeKernel/blob/main/docs/design/v1-launch.md?utm_source=chatgpt.com) · [Teacher/Student 原理](https://github.com/OpenBMB/ForgeKernel/blob/main/docs/design/milestone-distillation.md?utm_source=chatgpt.com) · [已发布 Kernel 与性能数据](https://github.com/OpenBMB/ForgeKernel/blob/main/kernels/README.md?utm_source=chatgpt.com)


# ForgeKernel vs 伯克利 Evolving Agent Teams：原理、性能与大模型依赖

我找到了你说的伯克利研究，正式名称是 Evolving Agent Teams（EAT），来自 UC Berkeley Sky Computing Lab，论文于 2026 年 10 月 27 日发布。

两篇研究的方向非常接近，都在探索：能不能让 AI Agent 自主生成、验证并持续优化 GPU Kernel，最终达到甚至超过人类专家手写 Kernel 的性能？

但它们解决的核心问题不一样。

OpenBMB

### ForgeKernel

核心是 Kernel 自进化

固定 Teacher/Student 分工，通过架构规划、硬件 Profiling 和 Reference 晋升，不断改进算子实现。

[GitHub 仓库](https://github.com/OpenBMB/ForgeKernel?utm_source=chatgpt.com)

UC Berkeley

### Evolving Agent Teams

核心是 Agent 团队自进化

Supervisor 动态创建 Agent 角色、调整协作关系，让优化过程能突破局部最优。

[GitHub 仓库](https://github.com/caoshiyi/EAT?utm_source=chatgpt.com)

先回答你最关心的两个问题：

第一，两个系统目前都依赖大模型。 公开实现主要使用 Claude Code 作为 Agent 执行环境，由 Claude 系列模型完成分析、代码生成和优化决策。GPU 负责编译、运行、Profiling 和性能反馈；这些过程并不是由大模型直接执行计算。

第二，基座模型会明显影响优化效果。 模型决定能否理解硬件架构、写对复杂 CUDA Kernel、分析错误以及提出有效优化策略。但伯克利实验表明，即使使用同一个模型，不同的 Agent 组织方法也能带来非常大的性能差异。

这意味着 GPU Kernel 自动优化的效果至少取决于三个因素：

\\[ \text{优化效果}=f(\text{基座模型},\text{Agent工作流},\text{实验预算与反馈}) \\]

而不是单纯依赖一个足够强的大模型。

## 一、整体对比

| 对比维度       | ForgeKernel                        | Evolving Agent Teams                 |
| ---------- | ---------------------------------- | ------------------------------------ |
| 研究目标       | 持续超越已有优秀 Kernel                    | 解决 Coding Agent 优化陷入局部最优的问题          |
| 主要创新       | Architecture-first + Reference 自进化 | Supervisor 动态重组 Agent 团队             |
| Agent 架构   | Teacher + Student                  | Supervisor + 动态 Worker Team          |
| 角色是否动态变化   | 固定 Teacher/Student 职责              | 可以自动创造、删除和调整角色                       |
| 优化对象       | GPU Kernel                         | 任意可执行程序，重点验证 GPU Kernel              |
| 硬件知识积累     | 跨算子 Hardware KB                    | 跨迭代 Persistent Memory                |
| GPU 性能验证   | 正确性 Judge + Benchmark + Profiling  | 官方 Evaluator + Benchmark + Profiling |
| Kernel 语言  | CUDA、CuTe DSL                      | 随任务使用 CUDA、CuTe DSL、HIP 等            |
| 公开硬件实验     | H100、B200；另有未完整开源的昇腾成果             | H100、B200、AMD MI350X、Apple M4 Pro 等  |
| 主要模型       | 论文使用 Claude Fable 5.1 + Opus 5     | 主实验 Claude Opus 4.6                  |
| 是否需要模型推理服务 | 是                                  | 是                                    |
| 部署定位       | GPU Kernel 专用优化平台                  | 通用长程 Coding Agent 编排框架               |

资料来自 ForgeKernel 官方 README 与 EAT 论文的 Method、Experiments 章节。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

OpenBMB/ForgeKernel · GitHub+2

下面重点看二者为什么能取得性能提升，以及它们的实测数字究竟能说明什么。

## 二、核心原理：到底在进化什么？

### 2.1 ForgeKernel：改进 Kernel 的设计和实现

ForgeKernel 采用相对固定的两层组织。

Architecture Agent（Teacher）

分析算子与硬件瓶颈，制定 M0 架构及 Milestone

Coding Agent（Student）

实现 Kernel、调试、优化 Tile / Pipeline

GPU Evaluator

Correctness + Timing + NCU Profiling

接受优化 / 更新架构 / 晋升 Reference

下一轮尝试超越当前最佳 Kernel

这里最重要的是 M0 设计。

例如，优化 MLA Decode 时，Teacher 可能选择 WGMMA 路线，并定义共享内存 Layout、Tensor Core 数据流、Register Budget 等。Student 之后主要沿这个方向实现和微调。

当发现 WGMMA 的某个设计确实无法继续提升时，Teacher 可以重新制定 M0。

这种方法的优势是搜索方向比较明确，且能够依赖 Hardware KB 复用历史经验。

它的限制是 Teacher 和 Student 的组织方式不会随任务自动产生新的角色，比如专门新增一个 Debugger 或 Compiler Specialist。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

OpenBMB/ForgeKernel · GitHub



### 2.2 EAT：不仅优化代码，还优化写代码的 Agent 团队

伯克利的核心贡献是把 Agent 组织方式本身也变成优化变量。

论文用三个元素描述一个 Agent Team：

\\[ T=(R,\Pi,M) \\]

其中：

- \\(R\\)：Roles，团队有哪些 Agent 角色。
- \\(\Pi\\)：Coordination，角色之间怎样协作。
- \\(M\\)：Memory Management，如何保存和传递经验。

传统 Agent 的 \\(T\\) 是固定的。EAT 在优化 Kernel 的同时也不断更新 \\(T\\)。

Supervisor Agent

分析优化历史、当前性能瓶颈与失败原因

动态设计当前 Agent Team

Architect

提出新计算架构

Kernel Coder

编写计算 Kernel

Debugger

定位 Correctness Bug

Profiling Scientist

分析 GPU 硬件指标

示意角色，实际角色数量与职责由 Supervisor 生成

执行、正确性验证、Benchmark

将结果、代码和 Agent 日志持久化

Supervisor 重新设计团队

根据瓶颈增减角色、重排任务或修改知识管理方式

其优化实际上是两个嵌套的循环：

\\[ \begin{aligned} \text{Inner: }& s\_{k+1}=\operatorname{Optimize}(s_k,T_k)\\\ \text{Outer: }& T\_{k+1}=\operatorname{Evolve}(T_k,K_k,F_k) \end{aligned} \\]

其中 \\(s_k\\) 是当前代码，\\(K_k\\) 是累计的实验知识，\\(F_k\\) 是性能和正确性反馈。这是对论文机制的简化表示，不是其实际源代码。

一个具体例子很能说明为什么它有用。

### 2.3 MLA Decode 案例：为什么要让 Agent 团队进化？

伯克利在 H100 上优化 DeepSeek-V3 的 MLA Paged Decode，观察到了一种典型的失败模式。

最初单 Agent 尝试使用 WGMMA，但由于 Shared Memory Layout、Descriptor 等细节，反复出现正确性问题。

Agent 最后放弃 WGMMA，转向更容易完成的 WMMA。虽然 WMMA 代码能够跑通，但最终计算架构的性能上限较低。

EAT 的 Supervisor 则做了不同的决策：

1. 发现 Agent 过早放弃了 WGMMA，增加 Disciplinarian 角色，要求保留 WGMMA 设计约束。
2. 为复杂的 WGMMA 正确性问题增加独立 Debugger，同时允许其他 Agent 继续优化。
3. 发现早期试验中的重要知识没有传递给后续团队，增加 Knowledge Integrator。
4. 在核心架构成功运行后，重新引入 Profiling 专家，研究 L2 Prefetch 和计算调度。

最终成功推进了更高性能的 WGMMA 实现，而不是长期停留在 WMMA 架构上。

这个案例说明：局部最优不一定是优化算法本身的问题，也可能是 Agent 团队因为调试困难、信息隔离而放弃了正确方向。

EAT 的方法就是通过改变协作组织来突破这种瓶颈。论文中该案例经历了超过 100 种 Team 配置。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+2



## 三、两个项目的实际效果怎么样？

### 3.1 EAT：最有说服力的是同预算对照实验

先看 H100 上的 MLA Decode。

伯克利让不同 Agent 系统使用同一个 Claude Opus 4.6 模型、相同的 800 美元模型推理预算，并在相同 47 个测试负载上重新评测。

相同模型与 800 美元预算下的 MLA Decode 性能

CUDAForge

相对 FlashInfer 加速比0.4×

0×0.2×0.4×0.6×0.8×CUDAForgeK-SearchEAT

大于 1.0× 才表示超过生产 FlashInfer；以下均未超过。

取论文 Table 2 中每种系统两次运行的较好结果。三者均通过全部 47 项正确性检查。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

注意，CUDAForge 和 ForgeKernel 是两个不同的项目。这个实验没有直接比较 EAT 与 OpenBMB 的 ForgeKernel。

伯克利另外还做了 EAT 与普通 Claude Code 的比较。在相同模型和匹配预算条件下，MLA Decode 的相对性能如下：

| 方法                  | 相对生产 FlashInfer 性能 |
| ------------------- | ------------------ |
| Claude Code 单 Agent | 约 0.51×            |
| EAT 动态团队            | 约 0.74–0.75×       |

也就是说，在这个任务上，EAT 达到的相对性能约比单 Agent 高 45%—47%。

这一对照能更直接地支持一个结论：在不单纯更换基座大模型的情况下，改进 Agent 工作流可以显著提升结果。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



### 3.2 EAT 增加预算之后的成果

| 算子 / 任务          | 硬件         | 最终报告性能 | 对比基线            |
| ---------------- | ---------- | ------ | --------------- |
| MLA Paged Decode | H100       | 1.21×  | FlashInfer 生产评测 |
| GDN Prefill      | B200       | 4.06×  | FlashInfer      |
| GEMM + AllReduce | H100       | 1.56×  | cuBLAS + NCCL   |
| KDA Prefill      | AMD MI350X | 3.01×  | vLLM HIP        |
| KDA Prefill      | B200       | 3.44×  | FlashKDA        |

这些都是论文报告的指定工作负载上的结果，不代表任意 Shape 的性能。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1



其中两个数字特别需要解释。

MLA 的 1.21× 并不代表核心计算 Kernel 比 FlashInfer 的纯 `run()` 快 21%。

论文图 1 另外以排除 `plan()` 开销的 FlashInfer `run()` 为 1.0×，最终 EAT 只有约 0.96×。1.21× 是相对其生产评测路径的加速比，其中包含不同的规划/调度成本口径。

GDN Prefill 的 4.06× 也包含了规划开销差异的收益。

因为在很多小 Shape 下，FlashInfer 的 Planning/Preprocessing 开销较突出，而 EAT 生成的实现不需要相同的规划步骤。即使延长优化后，100 个负载中仍有 37 个比 FlashInfer 基线慢。

因此，把 4.06× 理解成“新 CUDA 核心计算比 FlashInfer 快四倍”是不准确的。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+2



### 3.3 ForgeKernel 的成果

ForgeKernel 的官方报告覆盖 27 个算子，其中与 22 个专家库 Kernel 的对比中，有 18 个达到或超越专家实现。

专家 Kernel 对比

# 18 / 22

达到或超过基线

相对专家基线

# +39.4%

论文报告的几何平均提升

比较有代表性的 Kernel 包括：

| 算子                     | 硬件   | 性能提升    | 对比对象             |
| ---------------------- | ---- | ------- | ---------------- |
| MLA Paged Decode       | H100 | +170.6% | K-Search         |
| GQA Attention Forward  | H100 | +29%    | FlashAttention-4 |
| GDN Prefill Forward    | H100 | +100%   | FlashInfer       |
| FP8 GEMM               | H100 | +0.8%   | DeepGEMM         |
| FP8 × FP4 Grouped GEMM | B200 | +0.6%   | DeepGEMM         |

需要注意，`+170.6%` 这一 MLA 数字比较的是 K-Search，而不是 FlashInfer。不能据此推断 ForgeKernel 直接比 EAT 的 MLA Kernel 快多少。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 3.4 谁更强？目前还不能从论文数字直接判断

从现有证据看：

- ForgeKernel 在比较广泛的 GPU Kernel 集合上展示了稳定的专家基线对比，尤其适合研究已有高性能 Kernel 的继续优化。
- EAT 在相同模型、相同预算的 Agent 对照实验上提供了更直接的证据，说明动态团队重组能够提升复杂任务的搜索能力。
- EAT 还展示了在 AMD、Apple 等平台上优化不同类型程序的能力，适用问题范围更广。

不过，两篇论文并没有在相同算子、Shape、GPU、Baseline、正确性门槛和模型预算下直接对比 ForgeKernel 与 EAT。

此外，EAT 四项主实验通常只运行两次，额外场景不少只运行一次；一些性能成果使用了更长的搜索预算。因此，现阶段的结果证明了方法有潜力，但还不足以断言哪一个系统整体性能更强。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



## 四、它们是否都依赖大模型 API？

是的，当前公开实现都需要 LLM 推理能力，但不一定必须直接购买按 Token 计费的商业 API。

要区分三种东西。

LLM（基座大模型）

Claude / GPT / 本地部署的开源模型等

负责推理、生成代码、理解错误和规划优化

Coding Agent / Harness

Claude Code + ForgeKernel 或 EAT

管理上下文、工具调用、文件修改和优化循环

GPU Toolchain

CUDA / CuTe / 编译器 / Profiler / Evaluator

执行实际计算并提供客观反馈

大模型通常不直接跑 GPU Kernel。它生成和修改源码，工具链编译运行代码，再把测量结果返回给模型。

这里也不是在训练或者微调大模型。所谓 Self-Evolving 主要是更新 Kernel、Prompt、Team 配置、实验历史或硬件知识库，基座模型的权重通常保持不变。

### 4.1 实际支持哪些模型？

| 能力                | ForgeKernel             | EAT                        |
| ----------------- | ----------------------- | -------------------------- |
| 当前 Agent Backend  | Claude Code             | Claude Code                |
| 论文主要模型            | Claude Fable 5.1、Opus 5 | Opus 4.6                   |
| 后续实验或默认配置         | 默认 Opus 5.5             | Opus/Sonnet 5.5            |
| 分别配置 Agent 模型     | Teacher / Student 可独立配置 | Supervisor / Workers 可独立配置 |
| Claude 账号登录       | 支持                      | 支持                         |
| Anthropic API Key | 支持                      | 当前启动脚本优先要求登录或 OAuth Token  |
| 第三方兼容 API         | 提供 Anthropic 协议网关配置     | 无同等的通用后端配置承诺               |
| 本地部署开源模型          | 理论上可适配                  | 理论上可适配                     |
| 任意模型开箱即用          | 否                       | 否                          |

ForgeKernel 的配置文件是 `global_config/models.toml` 和 `global_config/backends.toml`。

EAT 则提供 `SUPERVISOR_MODEL`、`AGENT_MODELS` 等环境变量，允许在 Claude Code 支持的模型中选择 Supervisor 和 Worker 模型。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+2



因此，如果你用自己服务器上的 Qwen、DeepSeek 等开源模型替代 Claude，技术上可以探索，但需要解决 Claude Code 协议、Tool Calling、Context 管理和模型能力兼容问题，不能认为当前两个仓库已经原生支持任意本地模型。

还有一个很重要的区别：优化完成的 CUDA/CuTe Kernel 在实际推理时不再需要调用这些 Agent 或 LLM API。 API 消耗属于 Kernel 研发阶段的成本，不是每个推理请求的额外成本。

## 五、更换基座模型，效果会有多大变化？

影响可能很大，但不仅仅取决于模型的代码生成能力。

针对 GPU Kernel 优化，我认为以下能力尤其重要。

| 模型能力                       | 具体影响                                |
| -------------------------- | ----------------------------------- |
| CUDA / PTX / CuTe DSL 熟悉程度 | 能否生成正确的 WGMMA、TMA、TCGen05 实现        |
| 系统级架构推理                    | 能否从数据流、Tile、Warp、SMEM 层面重新设计 Kernel |
| Debugging                  | 能否定位线程同步、Layout、寄存器溢出和数值错误          |
| 长程规划                       | 多轮优化后是否会陷入局部最优                      |
| Tool Calling               | 能否正确使用编译器、NCU、测试脚本                  |
| 上下文与记忆能力                   | 能否保留重要实验发现，避免反复尝试失败路线               |
| Token 效率与速度                | 固定成本内能完成多少次有效实验                     |

EAT 的研究特别有价值，因为它提供了一个部分受控的例子：相同的 Claude Opus 4.6，加上不同的 Agent 工作流，MLA 性能差距仍然很明显。

但也要谨慎看待跨模型的数字。

论文还报告了 GPT-5.5 xHigh 的 Codex 单 Agent 在 MLA 上取得约 0.28×。这不意味着 GPT-5.5 作为基座模型比 Claude 差。 因为两者除了模型以外，还有 Agent 环境、上下文长度、任务执行路径等不同。

另外，论文中固定四角色团队的模型版本描述存在一处不完全一致：实验设置写的是 Opus 4.7，而其他描述又将它与 Opus 4.6 的动态团队视为同模型对照。因此，`0.31× → 0.74×` 不能被当成完全隔离模型变量的严格实验。

真正缺少的是如下形式的系统性消融：

| 模型                | 单 Agent | ForgeKernel | EAT |
| ----------------- | ------- | ----------- | --- |
| Claude Opus 4.6   | 待测      | 待测          | 待测  |
| Claude Opus 5.5   | 待测      | 待测          | 待测  |
| Claude Sonnet 5.5 | 待测      | 待测          | 待测  |
| 其他兼容模型            | 待测      | 待测          | 待测  |

只有固定相同硬件、算子、初始代码、正确性测试、Token 预算和测量方法，才可能分离出模型提升与 Agent 工作流提升的贡献。

### 5.1 API 成本会不会很高？

可能非常高，特别是长程优化。

EAT 报告的 AMD MI350X KDA Prefill 优化案例：

EAT — KDA Prefill / MI350X

Agent 迭代

# 73

轮

总运行时长

# 186

小时

模型调用预算

# $3,000

论文记录费用

最终性能相对 vLLM HIP

## 3.01×

使用 Claude Opus 5.5，模型费用按作者记录口径；这里没有将 GPU 算力、运维等成本等同计算进去。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

ForgeKernel 同样会消耗数小时 GPU 时间及前沿模型调用费用，其默认单轮预算为 4 小时，但项目没有提供一个可以直接对标上述 EAT 案例的总成本数字。

如果使用按 Token 计费的 API，Agent 的额外反思、角色创建、并行实验都会带来更多开销。

所以多 Agent 并非天然更经济。更合理的评价指标是：

\\[ \text{优化效率} = \frac{\text{有效性能收益}} {\text{LLM成本}+\text{GPU实验成本}} \\]

同时还要关注正确率、结果方差以及是否出现 Benchmark Overfitting。

## 六、对于你的 B200 和低比特推理实验，怎么选择？

结合你正在做的 NVFP4、MXFP8、MLA Decode 等工作，我更建议将二者当成两种不同阶段的优化工具，而不是简单二选一。

先用 ForgeKernel 验证已有算子优化

比如拿 B200 上已有的 FP8×FP4 Grouped GEMM，定义清楚 Shape、Scale Layout、量化误差容限和性能目标，再检查能否自动改善 Tile、Pipeline、SMEM 利用率及 Tensor Core 调度。

因为 ForgeKernel 有已经生成的相关 Kernel，启动成本相对可控。

再用 EAT 探索复杂的算法或架构调整

比如 MLA Decode 的 QK 混合精度、不同 KV Layout、算子融合和 CUDA/Tensor Core 协同，问题可能不在于某个 Tile Parameter，而是必须改变整个计算路径。

这时 EAT 动态生成架构师、调试专家和 Profiler 等角色的机制更值得研究。

数值正确性一定要采用你自己的独立测试

对 FP4/MXFP8 不能仅依赖论文中的默认数值容差。必须根据实际规格验证 Scale 计算、舍入、Overflow、Subnormal、累加精度，以及关键路径的 bit-exact 或 ULP 约束。

EAT 在 MI350X KDA 任务中使用的相对 L2 误差阈值为 2%，就不适合直接拿来判定低比特硬件数值行为。

如果你的目标是研究两种 Agent 架构本身，我建议从一个完全一致的 B200 Kernel 任务进行对照：同一个初始 Kernel、同一套测试 Shape、同样的正确性标准、相同模型和相同预算，分别交给单 Claude Code、ForgeKernel 和 EAT。

再进一步分别改变 Teacher/Supervisor 模型与 Coding Agent 模型，这样能同时回答两个问题：

- 强模型是否一定让最终 Kernel 更快？
- 强模型用在架构规划上，还是用在实际写 Kernel 上，收益更大？

这比直接比较论文里的 Speedup 更有研究价值。

## 七、最终结论

我对这两套系统的判断是：

ForgeKernel 更像一个专用的自动 Kernel 优化研发平台；EAT 更像一个可以自主调整研发团队组织方式的通用优化系统。

它们并不矛盾，实际上存在结合空间。比如保留 ForgeKernel 的 Hardware KB、Correctness Judge、Profiling Harness、Reference 晋升策略，但让 EAT 的 Supervisor 动态决定何时需要架构重构、并行调试、知识整理等角色。

而你关于基座大模型的理解是对的：LLM 决定候选方案的质量，Agent Harness 决定如何有效搜索，GPU Evaluator 决定哪些方案真正值得保留。 三者缺一不可。

我认为未来值得关注的方向不是单纯让更强的模型反复生成 CUDA，而是把硬件知识、可验证实验和动态搜索策略结合起来，并在相同预算下证明其收益。

主要一手资料：

- [Evolving Agent Teams — 伯克利论文全文](https://arxiv.org/html/2610.36848?utm_source=chatgpt.com)
- [EAT GitHub：Agent 编排源码](https://github.com/caoshiyi/EAT?utm_source=chatgpt.com)
- [EAT Gym：可复现实验任务与评测环境](https://huggingface.co/datasets/eva98/eat-gym?utm_source=chatgpt.com)
- [ForgeKernel GitHub：官方实现及模型配置](https://github.com/OpenBMB/ForgeKernel?utm_source=chatgpt.com)
- [ForgeKernel：Kernel 性能与源码索引](https://github.com/OpenBMB/ForgeKernel/blob/main/kernels/README.md?utm_source=chatgpt.com)






