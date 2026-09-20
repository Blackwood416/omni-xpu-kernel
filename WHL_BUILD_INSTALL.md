# omni_xpu_kernel Windows WHL 构建与 Portable 安装

本文只描述当前 Windows 构建合同：

```text
Python 3.13.14
PyTorch 2.13.0+xpu
Intel oneAPI DPC++/C++ Compiler 2026.0
oneDNN 3.11.2 / package release 2026.0.0
Intel Arc Pro B70 / intel_gpu_bmg_g31
OMNI_XPU_DEVICE=bmg
Windows wheel tag: cp313-cp313-win_amd64
omni-xpu-kernel: 0.2.0b2+torch213.bmg
```

A770 兼容 profile 另行验证了以下组合：

```text
Python 3.13 / PyTorch 2.13.0+xpu
Intel oneAPI DPC++/C++ Compiler 2026.1.0
oneDNN 3.11.2 native API/runtime（由 Windows wheel 内置）
Intel Arc A770 / DG2-G10，驱动 32.0.101.8860
OMNI_XPU_DEVICE=a770（规范化为 dg2）
Windows wheel: omni_xpu_kernel-0.2.0b1+torch213.dg2-cp313-cp313-win_amd64.whl
upstream base: ce0ccb928b1aa59f019a8c27b54fbb82d01c332c
```

### A770/DG2 SDP 现状

DG2 wheel 从 `omni_xpu_kernel` 0.2.0b1 起打包 `lgrf_sdp` sidecar，使用
DG2 原生 ESIMD kernel（`flash.attn.b.mha.dg2.h`），A770 上需显式设置
`OMNI_ATTN_BACKEND=esimd` 启用；默认仍走 PyTorch SDPA。

#### 已保留的负面结果与根因

- Xe2 原版 lgrf kernel 的 `dpas.hf.hf.8.8`（fp16 S×V 累加器）被 DG2 VISA
  校验器拒绝；改为 fp32 累加器后仍会
  `UR_RESULT_ERROR_DEVICE_LOST`。
- 根因（通过独立 SYCL harness + Windows LiveKernelEvent 141 日志确认）：
  **ESIMD work-group 大小 512 会在 A770（驱动 32.0.101.8860）触发 GPU
  TDR**，即使 kernel 只做 `slm_init` + 一次 store。Xe2 kernel 家族虽然
  WG=16，但同时使用 2D LSC、约 17 KB spill 与 fp16 DPAS 累加，同样不稳。
- 结论：A770 上 ESIMD attention 必须使用小 work-group（当前实现 WG=32，
  每线程一个 query row，全 head-dim 走寄存器），并避免 2D LSC 与
  named barrier。

#### 当前验收范围

- 正确性：`B=1, H=1..8, D∈{64,128}, fp16/bf16`，含 q/kv 非 32 倍数与
  1 token 边界，全部通过（fp16 max_abs ≤ 2.4e-4，bf16 ≤ 2e-3，对照
  Torch SDPA）。
- 稳定性：连续调用确定性一致；回归门禁
  `tests/repro_a770_sdp_device_lost.py` 通过（有限输出 + 误差阈值）。
- 性能（A770, driver 32.0.101.8860, oneAPI 2026.1, wall median,
  D=128, fp16, H=32）：v4.1 DPAS 变体
  （`flash.attn.b.mha.dg2.dpas4.h`）在扩展形状上反超 Torch SDPA。
  非 fused 路径先把 Q 打包成 DPAS A 操作数布局（每 QK 操作数一个 256B
  块），RPT=8 达到 100% XMX 行且无 spill；小形状保留 fused RPT=4/BN=64
  单 kernel；宿主传原始 kv_len，padding 由行号掩码处理，不再写 kvZero。
  40 样本交错结果：L=512 0.54 vs 0.60 ms，L=2048 2.75 vs 3.29 ms，
  L=4096 8.8 vs 12.6 ms，L=8192 34.7 vs 41.4 ms；1024x4096 2.91 vs
  3.20 ms、1024x1024 H48 1.43 vs 1.49 ms、512x512 H48 0.64 vs 0.65 ms。
  D64 仍走 v1 FMA。
- 已记录的负面结果（同一驱动/编译器栈）：RPT=6/8 只要有编译器 spill
  就触发 `UR_RESULT_ERROR_DEVICE_LOST`；RPT=6/BN=128 编译无 spill 但实机
  DEVICE_LOST；WG=64 的非 fused attn 单 tile 输出错误；fused BN=128 数值
  错误、fused RPT=6 在 BN=64/32 均 spill；N=8 下 fp16 DPAS 累加被
  dpas.hpp 拒绝；packedV 不经 SLM 直接读全局比 SLM staging 慢约 60%。

#### H3 INT8 长序列 offload 诊断（2026-08-12）

#### DG2 ConvRot 融合状态（2026-08-15）

#### DG2 D64 attention v4 移植状态（2026-08-16）

`flash.attn.b.mha.dg2.dpas4.d64.h` 是 D128 v4.1 的 D64 移植（生成脚本
`scripts/gen_dg2_d64_header.py` + 两处语义修复）：

- 修复：K pack 的 chunk 映射（D128 每 lane 2 chunk → D64 每 lane 1
  chunk）、非融合 packed-Q A 操作数按 RPT 存储、RPT=8（满 XMX 行）。
- 正确性：fp16/bf16 D64（H=32）全 seq 1..20683 通过，fp16 max_abs
  ≤1.5e-3，bf16 ≤1.6e-2；无 DEVICE_LOST（含 20x 压力）。
- 性能：仍然输给 torch SDPA（L=4096/8192/20683 为 0.84-0.92x，L=1797
  为 0.57x）。因此 ComfyUI-OmniXPU 的 `dg2_torch_d64_fp16` fallback
  gate 保留；kernel 保留供其他目标/驱动升级后复测。
- 已记录的负面结果：debug 构建下非融合路径对 `qState=nullptr` 解引用
  （dump 代码让 qChunkAll 保持存活）会 DEVICE_LOST；release 构建无此问题。

`int8_convrot_quant_dg2.cpp`（SLM 版）与 `int8_convrot_quant_esimd.cpp`
（寄存器版）共同替代 `rotate_convrot` 的缓存 Hadamard matmul，并融合
rowwise INT8 量化：

- 最终采用：寄存器版 ESIMD 蝶形（PTL-H 设计，DG2 上实测
  `20685x14336` 约 4.3ms vs matmul+quantize 约 7.3ms，1.7x 加速；
  bf16 约 93% 与 matmul 路径逐元素一致，其余相差 ≤1 个 INT8 LSB，
  scale 偏差 ≤0.4%；50 次 TDR 敏感形状压力无 DEVICE_LOST）。
- 路由：`OMNIXPU_DG2_CONVROT_FUSED=1` 默认开启（A/B 时设 0），
  `int8_linear`/H3 streaming 路径优先走
  `quantize_int8_convrot_fused_esimd`，SLM 版保留为后备。
- 已记录的负面结果：SLM 版（三 kernel，WG=256/每 subgroup 一组）在
  DG2 上只有 ~80GB/s，17ms 级别慢于生产路径；无效 subgroup 在 barrier
  前提前 return 会 DEVICE_LOST（已修复为 clamp+mask）；蝶形后的原地
  bf16 round pass 会被 DPC++ 折叠掉（移除 round、把 1/sqrt(G) 折叠进
  scale/quant_inv）。独立复现保留在
  `benchmarks/dg2_convrot_standalone.cpp` 与
  `tests/test_int8_convrot_fused_dg2.py`。

#### H3 INT8 长序列 offload 诊断（2026-08-12）

`comfy-info8.log` 的两次完整 workflow（int8_convrot 权重、内置 UNet
Loader）单次约 559-569 s。前 200 个 `seq=20683/head=56/D=128` block 占
约 440 s；后续 3780 个 `seq=1797/head=32/D=64` block 仅约 90-100 s。

- 已打点的 `int8_linear (20683,5376)x(28672,5376)` 实测只有约 35 ms
  （oneDNN `jit:gemm:any`，sync-each wall median），但 ComfyUI 日志中该
  事件到下一个 RMSNorm 的间隔约 0.85 s。慢的部分不在 OmniXPU kernel。
- 同一 block 的 `fc2`（SwiGLU + 14336 宽线性）没有 `int8_linear` kernel
  日志：vbar/offload cast 把 TensorWise INT8 权重反量化成 bf16，然后走
  bf16 `F.linear`。离线复算该回退约 41 ms（反量化 8.5 ms + SwiGLU/bf16
  linear 40.6 ms），仍远小于 0.85 s，剩余时间来自 vbar page-in/transfer。
- A/B 建议：`OMNIXPU_INT8_DIRECT_CAST=1` 时，offloaded TensorWise INT8
  模块直接拷 qdata/scale 上 XPU 并返回设备端 `QuantizedTensor`，绕过
  bf16 反量化（vbar 与普通 lowvram 均适用）。qdata 搬运实测约 8-27 ms
  （36-110 MiB），int8 kernel
  13-36 ms，合计约 20-60 ms/offloaded projection。
- 复现探针：`benchmarks/dg2_int8_phase0_probe.py`（真实 H3 phase-0 形状：
  qkv/fc1/fc2/out + CPU 搬运）。

#### H3 主循环瓶颈归属（comfy-info14，AIMDO draft PR 生效后）

`comfy-info14.log` 连跑两次完整 workflow：`314.73 s` / `259.84 s`，H3
主模型阶段（`seq=20683`，200 block）两次均为约 160 s，VAE 阶段约 92-120 s。
相比 `comfy-info8/13`（H3 约 440-495 s）是 AIMDO draft PR
（`pr-windows-usm-free-hang`：保留原生 XPU allocator、Level Zero tracing、
VBAR 边界预回收 + active VBAR 保护）带来的，不是 OmniXPU kernel 改动。

`benchmarks/h3_main_phase_a770.py`（真实 H3 shape、sync-each）实测：

- attention `(1,20683,56,128) bf16`：omni ESIMD v4 `405-410 ms`
  （约 30 TFLOPS，接近 A770 bf16 峰值）；torch SDPA `431-451 ms`。
  adapter 的 `BHLD->permute+contiguous->BLHD` 三份拷贝实测被隐藏
  （406 vs 408 ms），不是 0.51 s 间隔的来源。
- int8 oneDNN（tensorwise 真实量化权重）：qkv `29-35 ms`、fc1 `38-46 ms`、
  out `13-15 ms`、fc2(SwiGLU+convrot) `33 ms`；torch bf16 反量化线性
  分别约 `48-62 / 64-84 / 17-22 ms`。
- RMSNorm `(20683,5376) bf16`：`1.4 ms`。
- 完整 block（rms+qkv+attn+out+rms+fc1+fc2）串行实测 `556 ms/block`，
  即纯 kernel 部分约 `111 s / 200 block`。日志阶段 160 s，差额约
  `245 ms/block`（约 49 s/run）来自 `mixed_precision.Linear` 的
  dispatch 开销：`QuantizedTensor.from_float` 先把激活量化，再经 torch
  dispatch 反量化回 bf16，随后我们的 kernel 重新 rowwise 量化；
  qkv/out/fc1 每次约 90-96 ms（dispatch->kernel 间隔）。

- fc2 在无 LoRA/offload 状态下实际走 `linear_input_act` 的 registry
  XPU 路径（`comfy_kitchen.backends.xpu.int8_linear` -> omni int8），
  因此没有 `int8_linear` kernel 调试日志；日志缺失不代表 bf16 回退。
  之前 info8 记录的 fc2 bf16 回退属于 vbar/offload/LoRA 状态。

对应修复：`ComfyUI-OmniXPU/adapters/fp8_gemm.py` 的
`mixed_precision.Linear` 增加 INT8 快路径——满足
（XPU + `int8_tensorwise` + TensorWiseINT8Layout + 无 LoRA function +
 权重已驻留 XPU + 无 pre_quant_scale/input_scale）时直接调用
`omni_int8.int8_linear`，跳过 from_float/dequant/dispatch 往返。
数值路径与 `linear_input_act` 的 fc2 一致；其余条件一律回退原逻辑。
调试日志标记 `backend=omni_dg2_compat_fast`。可用
`OMNIXPU_INT8_FAST_FORWARD=0` 关闭快路径做 A/B。

实测（info17）快路径首次未生效的原因：H3 权重在 vbar 下驻留 CPU，
`self.weight.device != input.device` 导致跳过；真实工作流里 qkv/out/fc1/fc2
四个 Linear 的权重都在 CPU，按需 stream 进 GPU。因此快路径增加
`OMNIXPU_INT8_FAST_FORWARD_COPY=1`（默认开）兜底：qdata/scale 不在
XPU 时先 `.to(device)` 再跑同一个 omni kernel（与原路径每调用搬运的数据量
相同，但跳过 from_float/dequant/dispatch）。设
`OMNIXPU_INT8_FAST_FORWARD_COPY=0` 恢复严格 device 条件。

info18（copy=on）A/B：H3 主模型阶段约快 10 s（attention 间隔
0.746→0.694 s），但 VAE 明显变慢（int8_linear 链间隔 22→40 ms）：VAE
权重在 CPU，每次调用多一次 H2D 拷贝（4-33 MiB），而 VAE kernel 只有
1-3 ms，拷贝开销反而占主导。修复：拷贝兜底只在大激活时启用
（`OMNIXPU_INT8_FAST_FORWARD_COPY_MIN_ELEMS`，默认 16 Mi 元素，约
32 MiB bf16），VAE 小 Linear 回退原路径；权重已在 XPU 时不受限。

info19（阈值生效）：H3 三个 Linear 全部 `omni_dg2_compat_fast`，VAE
回到原路径，总时长 301.26 s / 254.06 s（info17 为 310.96 / 257.56）。

info20（PR #4 + 新版 ComfyUI + D64→torch）：255.02 s / 220.79 s，无
DEVICE_LOST。VAE attention 全部 `dg2_torch_d64_fp16`；H3 主模型
int8 全部 `omni_dg2_compat_fast`。

#### 工作流级 profiling 结论（2026-08-13，headless 复现 224.6/190.5 s）

自跑链路：`benchmarks/run_h3_workflow.ps1`（comfy-cli + 种子随机化，
可 `-Verbose` / `-NoManager`）、`benchmarks/profile_h3_workflow.ps1`
（py-spy 采样）、`benchmarks/vtune_h3_workflow.ps1`（VTune + 工作流）。

- VTune gpu-hotspots：attention kernel 81.4 s、oneDNN GEMM 12.8 s、
  其他 42.5 s；GPU 占用 75.9%。H2D 传输 151.8 GB 主要来自 VAE 逐 tile
  的权重搬运（约 3780 tile × 4 Linear），不是 H3。
- py-spy：`_int8_qdata_cached` 的高采样是首轮 200 个模块一次性 H2D
  拷贝（这也解释了第二次运行快 ~20 s）；真正的 host 时间分散在 torch
  dispatch（~38%）、cast_bias_weight 机制、asyncio/manager，没有单一
  可下手热点。
- A/B 均无收益并回退/未采用：VAE 小 Linear 快路径（首跑 297 s）、
  norm cast 绕过（权重在 CPU 时不生效）、norm 参数缓存、关 manager、
  wrapper→native 直连。qdata 缓存本身确认无 churn、命中正常。
- 每 block ~680 ms vs standalone GPU 下限 ~537 ms，剩余 ~140 ms 是
  ComfyUI/torch 调度与同步间隙，属架构固有；attention kernel 366 ms
  已贴峰值。当前实际可用成绩：首跑 ~214-225 s、第二次 ~182-192 s。

#### VTune：H3 attention 已贴峰值（2026-08-13）

`benchmarks/dg2v4_h3_vtune_driver.cpp`（bf16、L=KV=20683、H=56、D=128，
直接调 sidecar `sdp_bf16io`）实测纯 kernel 366 ms（约 33.5 TFLOPS）；
VTune gpu-hotspots 显示 attn 占 99.4% GPU 指令、packK 0.5%、packQ 0.08%。
kernel 侧没有可挖空间。工作流内每 block ~0.7-0.8 s 的剩余来自 host 侧：
每个 block 重复 H2D 拷贝 qkv/out/fc1/fc2 的 qdata（约 386 MiB）。
`OMNIXPU_INT8_QDATA_CACHE=1`（默认开）按模块身份+存储身份缓存 XPU
qdata/scale 副本（LRU，上限 12），消除重复拷贝；权重原地变更会失效，
LoRA 路径（weight_function 非空）本就不走快路径。可用
`OMNIXPU_INT8_QDATA_CACHE=0` 关闭 A/B。

info21（qdata cache 默认开）：首跑 254 s / 217 s，与 info20 基本持平；
逐事件时间线显示每 block ~840 ms，其中 GPU 计算下限约 537 ms
（standalone 真实权重全 block 流水线实测），剩余 ~240 ms/block 是
ComfyUI host/VRAM 开销（vbar cast、run_every_op、rope/qnorm、日志等），
不是 kernel。fc2 原本仍走 `linear_input_act` 的 cast_bias_weight +
registry int8（每 block 重复 77 MiB 拷贝 + vbar page-in），现也纳入
快路径：`comfy.ops.linear_input_act` 被补丁接管，TensorWise INT8 +
swiglu + 大激活时直接走 cached-qdata omni kernel
（debug 标记 `omni_dg2_compat_fast_fc2`）。

#### DG2 D64 attention 实测回归（2026-08-13）

同步测出 DG2 的 ESIMD D64 kernel 远慢于 torch SDPA：
`(1,1797,32,64) fp16` 10.57 ms vs 1.22 ms（约 8.7x）；`(1,20683,32,64)`
1.45 s vs 0.10 s（约 14x）。该 D64 sidecar 路径继承自 Xe2 调优、未在
A770 重新测量；VAE 阶段约 53 s/run 都耗在这里。适配器新增
`dg2_torch_d64_fp16` 路由：DG2 + fp16 + D64 + 无 mask 直接走 torch
SDPA（BHLD 输入零拷贝），预计 VAE 阶段省 ~45-50 s/run。D128 bf16
主模型保持 ESIMD（385 vs 391 ms，基本持平）。

本文不把 ComfyUI Portable 当作编译环境。编译环境位于项目目录内，
Portable 只用于最终安装和运行测试，避免修改其他项目的 Python 环境。

> [!IMPORTANT]
> Torch、Python ABI 和 GPU AOT 目标都属于 wheel 身份的一部分。不同
> Python ABI、Torch minor 或 GPU 架构必须分别构建，不能通过重命名 wheel
> 互换。Torch 2.13 仅在上述 Windows DG2/A770 组合中完成验证。

Portable 只用于安装和运行。原生扩展应在独立、可复用的项目构建环境中完成，
以便后续继续构建 Kitchen、AIMDO 和 kernel。

## 1. 当前依赖

| 组件 | 当前要求 |
|---|---|
| Windows | Windows 10/11 x64 |
| GPU | Intel Arc Pro B70，AOT target `bmg` |
| Visual Studio | Build Tools 2022，Desktop development with C++ |
| Windows SDK | Visual Studio 提供的 Windows 10/11 SDK，或第 4 节的项目内 fallback |
| Intel compiler | oneAPI DPC++/C++ Compiler 2026.0 |
| oneDNN development files | oneAPI oneDNN 2026.0，native ABI 3.11.2 |
| Python | 3.13.14 |
| Torch | `2.13.0+xpu` |
| sycl-tla | `2fc09973bfdf15755090fcb0e3b6ad236408a992` |

Torch 2.13 Windows 构建必须使用 matched oneDNN 3.11.2 headers、
`dnnl.lib` 和 `dnnl.dll`。`setup.py` 会检查 header/runtime ABI，并把
`dnnl.dll`、许可证和第三方 notices 放入 wheel。

## 2. Windows wheel 组成

如果 Visual Studio 已经安装 Windows 10/11 SDK 和 Universal CRT，通常不需要
NuGet fallback。

### 1.2 独立 Python 构建环境

| 包 | 精确版本 | 用途与获取地址 |
|---|---:|---|
| CPython | `3.13.12` | 必须与目标 Portable 的 `cp313` ABI 一致；[Python 3.13.12](https://www.python.org/downloads/release/python-31312/) |
| uv | `0.11.21` | 仅用于在项目目录管理 Python 和 venv；[uv 0.11.21](https://pypi.org/project/uv/0.11.21/)、[安装文档](https://docs.astral.sh/uv/getting-started/installation/) |
| pip | `26.1.2` | Python 包安装器；[PyPI](https://pypi.org/project/pip/26.1.2/) |
| setuptools | `78.1.0` | wheel 构建后端；[PyPI](https://pypi.org/project/setuptools/78.1.0/) |
| wheel | `0.47.0` | wheel 打包；[PyPI](https://pypi.org/project/wheel/0.47.0/) |
| torch | `2.12.0+xpu` | 编译所针对的原生 ABI；[官方 XPU wheel index](https://download.pytorch.org/whl/xpu/torch/)、[PyTorch Intel GPU 指南](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) |
| numpy | `2.5.1` | 测试环境数值依赖；[PyPI](https://pypi.org/project/numpy/2.5.1/) |
| pytest | `9.1.1` | 可选，运行源码测试；[PyPI](https://pypi.org/project/pytest/9.1.1/) |

当前验证到的 Windows `onednn==2025.3.0`/`onednn-devel==2025.3.0`
安装只包含 metadata、文档或许可证，不提供构建所需的 `dnnl.lib` 和运行时
`dnnl.dll`。构建仍然需要同一个 oneAPI oneDNN development install 中的
`oneapi/dnnl/dnnl.hpp`、`dnnl.lib` 和 `dnnl.dll`。`setup.py` 会校验三者对应
oneDNN `3.9.1`，并把 DLL 和 redistribution notices 打进 Windows wheel。
Windows Torch 2.13 DG2 profile 使用同一 SYCL 2026 ABI 的 oneDNN `3.11.2`；
不要给该组合混入依赖 `sycl8.dll` 的 2025.3 oneDNN runtime。

Torch XPU 在本次解析出的关键原生传递依赖如下。通常不应逐项手工安装，
而应让 `torch==2.12.0+xpu` 解析它们：

| 包 | 已验证版本 | 获取地址 |
|---|---:|---|
| dpcpp-cpp-rt | `2025.3.2` | [PyPI](https://pypi.org/project/dpcpp-cpp-rt/2025.3.2/) |
| intel-sycl-rt | `2025.3.2` | [PyPI](https://pypi.org/project/intel-sycl-rt/2025.3.2/) |
| intel-opencl-rt | `2025.3.2` | [PyPI](https://pypi.org/project/intel-opencl-rt/2025.3.2/) |
| intel-openmp | `2025.3.2` | [PyPI](https://pypi.org/project/intel-openmp/2025.3.2/) |
| intel-cmplr-lib-rt | `2025.3.2` | [PyPI](https://pypi.org/project/intel-cmplr-lib-rt/2025.3.2/) |
| intel-cmplr-lib-ur | `2025.3.2` | [PyPI](https://pypi.org/project/intel-cmplr-lib-ur/2025.3.2/) |
| intel-cmplr-lic-rt | `2025.3.2` | [PyPI](https://pypi.org/project/intel-cmplr-lic-rt/2025.3.2/) |
| intel-pti | `0.16.0` | [PyPI](https://pypi.org/project/intel-pti/0.16.0/) |
| mkl | `2025.3.1` | [PyPI](https://pypi.org/project/mkl/2025.3.1/) |
| tbb | `2022.3.1` | [PyPI](https://pypi.org/project/tbb/2022.3.1/) |
| tcmlib | `1.4.1` | [PyPI](https://pypi.org/project/tcmlib/1.4.1/) |
| umf | `1.0.3` | [PyPI](https://pypi.org/project/umf/1.0.3/) |
| triton-xpu | `3.7.1` | [PyTorch XPU index](https://download.pytorch.org/whl/xpu/triton-xpu/) |

### 1.3 已验证的 ComfyUI Portable 运行环境

| 包 | 精确版本 |
|---|---:|
| Python | `3.13.12` |
| torch | `2.12.0+xpu` |
| torchvision | `0.27.0+xpu` |
| torchaudio | `2.11.0+xpu` |
| onednn | `2025.3.0`（旧环境残留；新 Windows wheel 不依赖该包） |
| omni-xpu-kernel | `0.1.0b9.dev1+torch212.bmg` |
| comfy-kitchen | `0.2.26`，Intel XPU fork commit [`f7250fa4...`](https://github.com/xiangyuT/comfy-kitchen-xpu/commit/f7250fa44cb6f593969ba869be803e7d03c80ec8) |
| ComfyUI | `0.30.0`，commit [`b1693ecb...`](https://github.com/Comfy-Org/ComfyUI/commit/b1693ecba9f5b65f8c80ab36b195ab963ec92413) |
| comfyui-frontend-package | `1.47.12` |
| comfyui-workflow-templates | `0.11.28` |
| comfyui-embedded-docs | `0.5.9` |
| comfy-aimdo | `0.4.11` |
| comfyui-manager | `4.2.2` |

`torchvision 0.27.0+xpu` 可从
[PyTorch XPU torchvision index](https://download.pytorch.org/whl/xpu/torchvision/)
获取。当前 XPU index 中 torchaudio 的可用最新版是 `2.11.0+xpu`；本次与
Torch 2.12 的导入测试通过，但 `omni_xpu_kernel` 本身不依赖 torchvision
或 torchaudio。

完整 Omni runtime 会从 ComfyUI requirements 中省略官方 `comfy-kitchen`
依赖，并由 Intel XPU 部署流程单独安装 fork。原因、安装和更新流程见
[`docs/WINDOWS_PORTABLE.md`](../docs/WINDOWS_PORTABLE.md)。

## 2. Windows wheel 的组成与限制

Windows 构建产出两个原生扩展，并把核心扩展直接依赖的 oneDNN runtime
及其 redistribution notices 内置到同一个 wheel：

```text
omni_xpu_kernel/_C.cp313-win_amd64.pyd
omni_xpu_kernel/lgrf_uni/lgrf_sdp.cp313-win_amd64.pyd
omni_xpu_kernel/cute/cute_fmha_torch.cp313-win_amd64.pyd
omni_xpu_kernel/.libs/dnnl.dll
omni_xpu_kernel/.libs/onednn/LICENSE
omni_xpu_kernel/.libs/onednn/THIRD-PARTY-PROGRAMS
omni_xpu_kernel/.libs/onednn/VERSION
```

- `_C` 提供 norm、FP8、GGUF、SVDQ、INT8、rotary 和 oneDNN 算子。
- `lgrf_sdp` 提供 ESIMD SDP sidecar。
- `cute_fmha_torch` 同时导出 CUTE FMHA 和 Sol-Attn。
- Windows 默认不构建 CUTE；当前完整 wheel 必须显式设置
  `OMNI_XPU_REQUIRE_CUTE=1`。
- 编译开关不会改变 ComfyUI runtime policy。运行时仍需显式设置
  `OMNI_ATTN_BACKEND=cute`。

## 3. 准备持久构建环境

以下 PowerShell 示例把构建环境放在仓库同级目录，避免污染 Portable，并可在
后续构建中继续复用：

```powershell
$repoRoot = (Resolve-Path "<llm-scaler-repository-root>").Path
$kernelRoot = Join-Path $repoRoot "omni\omni_xpu_kernel"
$workspaceRoot = Split-Path $repoRoot -Parent
$buildRoot = Join-Path $workspaceRoot ".omni-portable-build"
$venvRoot = Join-Path $buildRoot "venv"
$buildPython = Join-Path $venvRoot "Scripts\python.exe"

$env:UV_PYTHON_INSTALL_DIR = Join-Path $buildRoot "python"
$env:UV_CACHE_DIR = Join-Path $buildRoot "uv-cache"

New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null

if (-not (Test-Path $buildPython)) {
    uv python install 3.13.14
    uv venv --seed --python 3.13.14 $venvRoot
}

& $buildPython -m pip install --upgrade pip setuptools wheel
& $buildPython -m pip install `
    "torch==2.13.0+xpu" `
    --index-url "https://download.pytorch.org/whl/xpu"
& $buildPython -m pip install pytest numpy
& $buildPython -m pip check
```

确认构建解释器没有引用 Portable 或其他项目环境：

```powershell
& $buildPython -c @"
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch XPU runtime:", torch.version.xpu)
print("XPU available:", torch.xpu.is_available())

assert torch.__version__ == "2.13.0+xpu"
"@
```

## 4. Windows SDK fallback

如果构建报错 `assert.h`、`windows.h` 或 UCRT 头文件缺失，首选通过 Visual
Studio Installer 添加 Windows 10/11 SDK。无法修改系统安装时，可以在
`$buildRoot` 中准备项目内 SDK：

```powershell
$sdkRoot = Join-Path $buildRoot "windows-sdk-nuget"
$sdkPackageVersion = "10.0.26100.3916"
$sdkPackages = @(
    "Microsoft.Windows.SDK.CPP",
    "Microsoft.Windows.SDK.CPP.x64",
    "Microsoft.Windows.SDK.BuildTools"
)

New-Item -ItemType Directory -Force -Path $sdkRoot | Out-Null

foreach ($package in $sdkPackages) {
    $archive = Join-Path $sdkRoot "$($package.ToLowerInvariant()).$sdkPackageVersion.nupkg"
    $destination = Join-Path $sdkRoot $package.ToLowerInvariant()
    Invoke-WebRequest `
        -Uri "https://www.nuget.org/api/v2/package/$package/$sdkPackageVersion" `
        -OutFile $archive
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    tar.exe -xf $archive -C $destination
}
```

包版本为 `10.0.26100.3916`，解压后的 SDK 文件目录为
`10.0.26100.0`。第 5 节的构建 shell 只在系统 SDK 不完整时需要添加这些
`INCLUDE`、`LIB` 和 `PATH`。

## 5. 准备 sycl-tla

使用干净 checkout，不能直接修改 sycl-tla。Windows LLP64、host scalar 和
BMG remainder-mask 修复由 kernel build 在临时 include overlay 中应用。

```powershell
$syclTlaRoot = Join-Path $buildRoot "sycl-tla"
$syclTlaCommit = "2fc09973bfdf15755090fcb0e3b6ad236408a992"

if (-not (Test-Path (Join-Path $syclTlaRoot ".git"))) {
    git clone --filter=blob:none --no-checkout `
        "https://github.com/intel/sycl-tla.git" `
        $syclTlaRoot
}

git -C $syclTlaRoot fetch --depth 1 origin $syclTlaCommit
git -C $syclTlaRoot checkout --detach $syclTlaCommit

if ((git -C $syclTlaRoot status --short)) {
    throw "sycl-tla checkout must be clean"
}
```

## 6. 构建 BMG CUTE/Sol-Attn wheel

在同一个 `cmd.exe` 中初始化 MSVC、oneAPI、oneDNN 和项目 venv。oneAPI
`setvars.bat` 在部分 Windows 安装上不能正确调用 component `vars.bat`，
因此下面直接设置当前 2026.0 安装布局：

```bat
@echo off

set "REPO_ROOT=<llm-scaler-repository-root>"
set "KERNEL_ROOT=%REPO_ROOT%\omni\omni_xpu_kernel"
set "BUILD_ROOT=<persistent-build-root>"
set "BUILD_PYTHON=%BUILD_ROOT%\venv\Scripts\python.exe"
set "CUTLASS_SYCL_ROOT=%BUILD_ROOT%\sycl-tla"

call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1

set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"
set "ONEAPI_COMPILER_ROOT=%ONEAPI_ROOT%\compiler\2026.0"
set "DNNLROOT=%ONEAPI_ROOT%\dnnl\2026.0"

set "PATH=%ONEAPI_COMPILER_ROOT%\bin;%ONEAPI_COMPILER_ROOT%\bin\compiler;%ONEAPI_ROOT%\ocloc\2026.0\bin;%BUILD_ROOT%\venv\Library\bin;%BUILD_ROOT%\venv\Lib\site-packages\torch\lib;%DNNLROOT%\bin;%PATH%"
set "INCLUDE=%ONEAPI_COMPILER_ROOT%\include;%ONEAPI_COMPILER_ROOT%\include\sycl;%INCLUDE%"
set "LIB=%ONEAPI_COMPILER_ROOT%\lib;%ONEAPI_COMPILER_ROOT%\opt\compiler\lib;%BUILD_ROOT%\venv\Lib\site-packages\torch\lib;%DNNLROOT%\lib;%LIB%"

set "OMNI_XPU_DEVICE=bmg"
set "OMNI_XPU_REQUIRE_CUTE=1"
set "MAX_JOBS=8"

where cl
where icx
where llvm-foreach
where ocloc
"%BUILD_PYTHON%" -c "import torch; assert torch.__version__ == '2.13.0+xpu'; print(torch.__version__, torch.version.xpu)"
sycl-ls --verbose

cd /d "%KERNEL_ROOT%"
if not exist "%BUILD_ROOT%\wheels\kernel-solattn" mkdir "%BUILD_ROOT%\wheels\kernel-solattn"

"%BUILD_PYTHON%" -m pip wheel . ^
  --wheel-dir "%BUILD_ROOT%\wheels\kernel-solattn" ^
  --no-build-isolation ^
  --no-deps
```

B70 的 `sycl-ls --verbose` 应报告：

```text
Architecture: intel_gpu_bmg_g31
```

如果使用第 4 节的项目内 Windows SDK，在 `pip wheel` 前添加：

```bat
set "SDK_ROOT=%BUILD_ROOT%\windows-sdk-nuget"
set "SDK_VERSION=10.0.26100.0"
set "SDK_CPP=%SDK_ROOT%\microsoft.windows.sdk.cpp\c"
set "SDK_X64=%SDK_ROOT%\microsoft.windows.sdk.cpp.x64\c"
set "SDK_TOOLS=%SDK_ROOT%\microsoft.windows.sdk.buildtools"

set "INCLUDE=%SDK_CPP%\Include\%SDK_VERSION%\ucrt;%SDK_CPP%\Include\%SDK_VERSION%\shared;%SDK_CPP%\Include\%SDK_VERSION%\um;%SDK_CPP%\Include\%SDK_VERSION%\winrt;%SDK_CPP%\Include\%SDK_VERSION%\cppwinrt;%INCLUDE%"
set "LIB=%SDK_X64%\ucrt\x64;%SDK_X64%\um\x64;%LIB%"
set "PATH=%SDK_TOOLS%\bin\%SDK_VERSION%\x64;%PATH%"
```

`setup.py` 会把 Windows 核心扩展的多个 translation unit 并行编译。默认
最多 8 个并行任务；需要手动控制时设置 `OMNI_XPU_BUILD_JOBS`（或
`MAX_JOBS`），例如 `set OMNI_XPU_BUILD_JOBS=8` 后再执行 `pip wheel`。
实测 DG2/Torch 2.13 全量 wheel 构建约 4-5 分钟，并行开关不会改变产物
身份。

`--no-build-isolation` 是必需的：构建必须读取当前 venv 中已安装的
Torch XPU 头文件、库和版本。`--no-deps` 避免打包过程改变环境。

已验证输出：

```text
.venv-win-py313-torch212\wheelhouse\patched\
  omni_xpu_kernel-0.1.0b9.dev1+torch212.bmg-cp313-cp313-win_amd64.whl
```

已验证 artifact：

```text
size:   25,185,658 bytes
SHA256: E112C1720ACA4AF975501470A77F654656D6A4A3CF919A36A2EFBC8B1F4F0795
```

体积增加来自 wheel 内置的匹配 oneDNN Windows runtime。构建只复制与
已校验 `dnnl.lib` 同一个安装根下的 `bin\dnnl.dll`，不会把 Torch、SYCL、
Unified Runtime 或完整 oneAPI SDK 重复打进 wheel。

该 artifact 使用与 Linux 相同的 RMSNorm、LayerNorm 和 fused Add+RMSNorm
GS dispatch ladder；Windows SDP loader 同时解析 sidecar 导出的 D64、D128
和 FP16 fast-path 符号。

项目的 `scripts\build.bat` 可用于原地安装或开发安装，但需要发布或复制
artifact 时，应使用上面的 `pip wheel` 命令。

## 6. 检查 wheel 内容和元数据

```powershell
$wheelRoot = Join-Path $buildRoot "wheels\kernel-solattn"
$kernelWheel = Get-ChildItem $wheelRoot `
    -Filter "omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-cp313-win_amd64.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $kernelWheel) {
    throw "kernel wheel not found"
}

Get-FileHash -Algorithm SHA256 -LiteralPath $kernelWheel.FullName
& $buildPython -m zipfile -l $kernelWheel.FullName
```

当前验收 artifact：

```text
omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-cp313-win_amd64.whl
SHA256: 054B5AD9B7AC046153446A249ADBBAED56C95F00CC82FB40C5AF595F6345183A
```

zip listing 必须包含第 2 节列出的三个 `.pyd`、`dnnl.dll` 和 oneDNN
redistribution notices。缺少 CUTE `.pyd` 表示构建没有实际启用
`OMNI_XPU_REQUIRE_CUTE=1`。

## 8. 安装到 Portable

先关闭所有使用目标 `python_embeded` 的进程，然后验证基础包身份：

```powershell
$portableRoot = (Resolve-Path "<ComfyUI_windows_portable-root>").Path
$embeddedPython = Join-Path $portableRoot "python_embeded\python.exe"

& $embeddedPython -c @"
import torch
import torchvision
import torchaudio

assert torch.__version__ == "2.13.0+xpu"
assert torchvision.__version__ == "0.28.0+xpu"
assert torchaudio.__version__ == "2.11.0+xpu"
assert torch.xpu.is_available()
"@
```

安装时保留 `--no-deps`：

```powershell
& $embeddedPython -m pip install `
    --force-reinstall `
    --no-deps `
    $kernelWheel.FullName

& $embeddedPython -m pip check
```

## 9. 验收

所有安装态检查都应离开 kernel source checkout，避免本地源码遮蔽 Portable
中的 wheel：

```powershell
Set-Location $portableRoot

& $embeddedPython -c @"
from importlib import metadata
from pathlib import Path

import torch
import omni_xpu_kernel as omni
from omni_xpu_kernel import cute

print("torch:", torch.__version__)
print("kernel:", metadata.version("omni-xpu-kernel"))
print("module:", Path(omni.__file__).resolve())
print("target:", omni.__xpu_target__, omni.core_aot_target())
print("capabilities:", omni.native_capabilities())
print("CUTE:", cute.is_available())
print("Sol-Attn:", cute.supports_sol_attn())

assert torch.__version__ == "2.13.0+xpu"
assert metadata.version("omni-xpu-kernel") == "0.2.0b2+torch213.bmg"
assert omni.__xpu_target__ == "bmg"
assert omni.core_aot_target() == "bmg"
assert omni.is_available()
assert cute.is_available()
assert cute.supports_sol_attn()
"@
```

最小 D128 CUTE correctness：

```powershell
@'
import torch
from omni_xpu_kernel import cute

q = torch.randn(1, 256, 8, 128, device="xpu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

actual = cute.sdp(q, k, v)
expected = torch.nn.functional.scaled_dot_product_attention(
    q.permute(0, 2, 1, 3),
    k.permute(0, 2, 1, 3),
    v.permute(0, 2, 1, 3),
).permute(0, 2, 1, 3)

torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
torch.xpu.synchronize()
print("Windows BMG CUTE D128: PASS")
'@ | & $embeddedPython -
```

源码侧当前回归入口：

```powershell
Set-Location $kernelRoot
& $buildPython -m pytest -q `
    tests\test_packaging.py `
    tests\test_cute_sol_attn_api.py
```

## 10. 常见错误

### 找不到 oneDNN 3.11.2

确认 `DNNLROOT` 指向完整 oneAPI oneDNN 2026.0 安装：

```bat
dir "%DNNLROOT%\include\oneapi\dnnl\dnnl.hpp"
dir "%DNNLROOT%\lib\dnnl.lib"
dir "%DNNLROOT%\bin\dnnl.dll"
```

不要混用不同安装根的 headers、import library 和 runtime DLL。自定义布局应
同时设置 `ONEDNN_INCLUDE`、`ONEDNN_LIB`、`ONEDNN_RUNTIME` 和
`ONEDNN_LICENSE_DIR`。

### `LNK1104: cannot open file 'libircmt.lib'`

把 oneAPI compiler library 加入同一个构建 shell：

```bat
set "LIB=%ONEAPI_COMPILER_ROOT%\lib;%ONEAPI_COMPILER_ROOT%\opt\compiler\lib;%LIB%"
```

### 找不到 `llvm-foreach.exe`、`sycl-post-link.exe` 或 `ocloc.exe`

确认 `PATH` 包含：

```text
compiler\2026.0\bin
compiler\2026.0\bin\compiler
ocloc\2026.0\bin
```

### sycl-tla overlay 校验失败

#### SeedVR2 A770 优化与负面结果（2026-08-19/20）

目标：`seedvr2_3b_int8_upscale_video.json`（输入 124 帧 1280x2304，3B
INT8 ConvRot 模型，A770 / torch 2.13 / oneAPI 2026.1 / driver 8860）。

实测端到端：**681 s → 607 s**（decode 415→362 s，encode 162→146 s，
sampling 82→78 s）。

- DG2 启用 SeedVR2 cat-pad（`[1,128,4,512,512]` temporal-major + 连续
  prefix）与 SeedVR group-norm（`[4,128,512,512]` temporal-interleaved）：
  standalone 分别 7.2 vs 9.3 ms、2.95 vs 5.4 ms（vs torch），输出与
  torch 一致（cat-pad 逐位一致，group-norm 在 fp16 噪声内）。
- Attention dispatch：A770 D128 bf16/fp16 在 `q_len∈[1024,2048)∪
  (2048,4096)` 区间 esimd 比 torch SDPA 慢 1.1-1.6x，`q_len==2048` 与
  `q_len>=4096` 才占优；插件按此回退 torch。

已记录的负面结果：

- **单次 USM 分配上限约 4 GiB**（3.9 GiB 成功、4.0 GiB 失败，即使空闲
  14+ GiB）；`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` 可解除（4.0 /
  4.12 / 5.27 / 8.0 GiB 均成功）。未设置时 SeedVR2 decode 的 4.12 GiB
  fp32 结果会 OOM，ComfyUI-OmniXPU 提供 CPU staging fallback。
- VTune（attach decode 300 s）：GPU Time 99.7%，XVE Array
  Stalled/Idle 68.1% —— decode 是 GPU 满负荷、conv3d 访存/占用受限，
  host 不是瓶颈；oneDNN 仍是该形状的最强 conv 基线。
- spatial tile 512→1024：decode 峰值内存超过 16 GiB，直接 OOM。
- temporal_size 64→125（外层 3→2 chunk）：decode 365 vs 362 s，无收益
  —— 总帧数决定成本，外层 chunk 数不是瓶颈。
- INT8 fast-path copy 阈值 16Mi→4Mi：中型 int8 linear 接入快路径但
  采样无提升（77 vs 78 s），默认阈值保持 16Mi。

## 10. Torch 2.13 后续阶段

### CUTE `.pyd` 存在但无法加载

确认 Python tag 是 `cp313`，Torch 是 XPU build，wheel target 是 `bmg`，并且
Portable 的 `python_embeded\Library\bin` 和
`python_embeded\Lib\site-packages\torch\lib` 位于 `PATH`。

### 测试完成后 Python 进程不退出

部分 XPU 测试会在所有输出和断言完成后卡在 interpreter teardown。先确认测试
结果已经完整打印，再终止具体 PID；不要删除被进程锁定的 `.pyd`。

## 11. 当前边界

- Windows CUTE/Sol-Attn 当前只验收 BMG；PTL-H 需要独立构建与验收。
- 当前 ComfyUI CUTE route 以已验证 D128 contract 为主；不支持的
  dtype、layout、mask、head dimension 或 GQA contract 回退到 dense
  attention。
- CUTE 是 build-time 和 runtime 双重 opt-in。
- 旧 Sol custom node 和实验 gate 已退出新集成；使用
  [ComfyUI 原生 Model Sparse Attention](../docs/SPARSE_ATTENTION.md) 及匹配的
  Kitchen XPU provider 和完整 sparse API。本文历史 Windows wheel 验收
  不等于新版原生 SOL/SLA/VSA 的 Windows 验收。
