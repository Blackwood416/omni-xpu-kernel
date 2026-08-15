# omni_xpu_kernel Windows WHL 构建与 Portable 安装

本文记录 `omni_xpu_kernel` 在 Windows x64 上的独立构建、wheel
检查、ComfyUI Portable 安装和验收流程。

当前已实际验证的组合是：

```text
Python 3.13.12
PyTorch 2.12.0+xpu
Intel oneAPI DPC++/C++ Compiler 2025.3.3
oneDNN 3.9.1 native API/runtime（由 Windows wheel 内置）
Intel Arc Pro B70 / intel_gpu_bmg_g31
OMNI_XPU_DEVICE=bmg
Windows wheel tag: cp313-cp313-win_amd64
llm-scaler source: b9b0c4c900f1a1ef3ec987fe6be5aef26b22e3c8
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

`int8_convrot_quant_dg2.cpp` 实现了 radix-4 SLM 蝶形变换替代
`rotate_convrot` 的缓存 Hadamard matmul，并融合 rowwise INT8 量化：

- 最终形态：三 kernel（rotate+group-max → 行归约 → rotate+quantize），
  WG=256、每 subgroup 一个 group、无效 subgroup clamp 而非提前 return。
- 正确性：bf16 各 H3 shape 约 92-94% 与 matmul 路径逐元素一致，其余
  相差 ≤2 个 INT8 LSB，scale 偏差 ≤0.6%；f16 约 99%。
- 稳定性：无 DEVICE_LOST（之前两个版本的 DEVICE_LOST 根因是无效
  subgroup 在 barrier 前提前 return；已修复）。
- 编译器坑：蝶形变换后紧跟的原地 bf16 round pass（读取并重写全部 SLM）
  会被 DPC++ 折叠掉，kernel 退化为“返回原始输入”；因此 round pass 被
  移除，`1/sqrt(G)` 折叠进 scale/quant_inv（`q=round(raw*127/row_max)`）。
- 性能：A770 上仍比现有 `rotate_convrot`(matmul) +
  `quantize_int8_rowwise_fused` 慢（20685x14336：fused ~17.3ms vs
  ~7.3ms），主要是 SLM barrier kernel 在 DG2 上的带宽远低于 matmul 路径。
- 结论：保持 `OMNIXPU_DG2_CONVROT_FUSED=1` opt-in（默认关闭）；后续可
  尝试寄存器/ESIMD 蝶形或驱动升级后复测。负面结果与独立复现保留在
  `benchmarks/dg2_convrot_standalone.cpp`、`tests/test_int8_convrot_fused_dg2.py`
  与 kernel 文件注释中。

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

## 1. 已验证版本矩阵

### 1.1 系统工具链

| 组件 | 已验证版本 | 说明与获取地址 |
|---|---:|---|
| Windows | Windows 11 Pro x64，`10.0.26200` | Windows 10/11 x64；这里记录的是本次验证主机，而不是硬性最低版本 |
| Intel Arc Pro 驱动 | `32.0.101.8515` | 本机验证版本，不代表硬性最低版本；从 [Intel Arc Pro Windows 驱动页](https://www.intel.com/content/www/us/en/download/741626/intel-arc-pro-graphics-windows.html) 获取当前驱动 |
| Visual Studio Build Tools 2022 | `17.14.36` | 安装 `Desktop development with C++`；参见 [Microsoft C++ Build Tools 安装文档](https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc) |
| MSVC v143 x64/x86 | `14.42.34433`，`cl 19.42.34444` | 本次构建显式使用 `-vcvars_ver=14.42`；工作负载组件见 [Microsoft Build Tools component IDs](https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools) |
| Intel oneAPI DPC++/C++ Compiler | `2025.3.3`，build `20260319` | [编译器下载页](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html)；[2025 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html) |
| Intel oneAPI oneDNN development install | `2025.3` | 必须包含 oneDNN `3.9.1` 的头文件、`dnnl.lib`、`dnnl.dll` 和 redistribution notices；可随 [Intel oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit.html) 安装 |
| Windows SDK NuGet package | `10.0.26100.3916` | NuGet 包内头文件版本为 `10.0.26100.0`；仅在系统没有 Windows SDK/UCRT 时需要项目内 fallback |

本次使用的三个 Windows SDK NuGet 包：

- [Microsoft.Windows.SDK.CPP 10.0.26100.3916](https://www.nuget.org/packages/Microsoft.Windows.SDK.CPP/10.0.26100.3916)
- [Microsoft.Windows.SDK.CPP.x64 10.0.26100.3916](https://www.nuget.org/packages/Microsoft.Windows.SDK.CPP.x64/10.0.26100.3916)
- [Microsoft.Windows.SDK.BuildTools 10.0.26100.3916](https://www.nuget.org/packages/Microsoft.Windows.SDK.BuildTools/10.0.26100.3916)

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
omni_xpu_kernel/.libs/dnnl.dll
omni_xpu_kernel/.libs/onednn/LICENSE
omni_xpu_kernel/.libs/onednn/THIRD-PARTY-PROGRAMS
omni_xpu_kernel/.libs/onednn/VERSION
```

- `_C` 包含 norm、FP8、GGUF、SVDQ、INT8、rotary 和 oneDNN 等核心算子。
- `lgrf_sdp` 是独立的 ESIMD SDP sidecar。
- CUTE FMHA 当前是 Linux-only，Windows 必须设置
  `OMNI_XPU_REQUIRE_CUTE=0`。
- Windows ComfyUI 默认保留 PyTorch SDPA；ESIMD sidecar 仅在显式设置
  `OMNI_ATTN_BACKEND=esimd` 时由 Custom Node 使用。
- `OMNI_XPU_DEVICE=bmg` 会把核心扩展和 sidecar 都 AOT 编译为 BMG
  `spir64_gen` 镜像。
- PTL-H 必须单独使用 `OMNI_XPU_DEVICE=ptl-h` 构建，不能安装 BMG wheel。

项目当前识别 Torch XPU 2.10、2.11 和 2.12。识别某个 minor 不代表所有
组合都已经验收；本文只对 Torch `2.12.0+xpu`、Python 3.13、BMG 作出验证
声明。

## 3. 准备项目内独立构建环境

以下命令在 PowerShell 中执行。先把尖括号占位符替换为
`omni_xpu_kernel` 源码目录：

```powershell
$kernelRoot = (Resolve-Path "<omni_xpu_kernel-source-directory>").Path
$buildRoot = Join-Path $kernelRoot ".venv-win-py313-torch212"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $buildRoot "python"
$env:UV_CACHE_DIR = Join-Path $buildRoot "cache"

Set-Location $kernelRoot

uv python install 3.13.12
uv venv --seed --python 3.13.12 (Join-Path $buildRoot "venv")

$buildPython = Join-Path $buildRoot "venv\Scripts\python.exe"

& $buildPython -m pip install `
    "pip==26.1.2" `
    "setuptools==78.1.0" `
    "wheel==0.47.0"

& $buildPython -m pip install `
    "torch==2.12.0+xpu" `
    --index-url "https://download.pytorch.org/whl/xpu"

& $buildPython -m pip install `
    "numpy==2.5.1" `
    "pytest==9.1.1"
```

检查环境，不要依赖其他 Python：

```powershell
& $buildPython -c @"
import torch
print("torch:", torch.__version__)
print("torch XPU runtime:", torch.version.xpu)
print("XPU available:", torch.xpu.is_available())
print("devices:", [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())])
"@

& $buildPython -m pip check
```

预期 Torch 版本为 `2.12.0+xpu`，`torch.version.xpu` 为 `20250302`。

## 4. 没有系统 Windows SDK 时的项目内 fallback

如果编译探针或正式构建报错：

```text
fatal error: 'assert.h' file not found
```

说明 MSVC/Intel 编译器环境没有取得 Windows SDK/UCRT 头文件。首选方案是
通过 Visual Studio Installer 安装 Windows 10/11 SDK 和 Universal CRT。
如果不希望修改系统安装，可以把已验证的 NuGet SDK 放进 `$buildRoot`。

```powershell
$sdkRoot = Join-Path $buildRoot "windows-sdk-nuget"
New-Item -ItemType Directory -Force -Path $sdkRoot | Out-Null

$sdkPackages = @(
    "Microsoft.Windows.SDK.CPP",
    "Microsoft.Windows.SDK.CPP.x64",
    "Microsoft.Windows.SDK.BuildTools"
)
$sdkPackageVersion = "10.0.26100.3916"

foreach ($package in $sdkPackages) {
    $fileName = "$($package.ToLowerInvariant()).$sdkPackageVersion.nupkg"
    $packageFile = Join-Path $sdkRoot $fileName
    $extractDir = Join-Path $sdkRoot $package.ToLowerInvariant()
    $downloadUrl = "https://www.nuget.org/api/v2/package/$package/$sdkPackageVersion"

    Invoke-WebRequest -Uri $downloadUrl -OutFile $packageFile
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    tar.exe -xf $packageFile -C $extractDir
}
```

本次解压后的关键目录是：

```text
windows-sdk-nuget\
  microsoft.windows.sdk.cpp\c\Include\10.0.26100.0\
    ucrt\
    shared\
    um\
    winrt\
    cppwinrt\
  microsoft.windows.sdk.cpp.x64\c\
    ucrt\x64\
    um\x64\
  microsoft.windows.sdk.buildtools\bin\10.0.26100.0\x64\
```

NuGet 包版本 `10.0.26100.3916` 与包内 SDK 文件版本
`10.0.26100.0` 不同，这是正常的。

## 5. 初始化编译环境并构建 wheel

建议打开普通 `cmd.exe`，显式初始化 MSVC 和 oneAPI，然后调用项目 venv
中的 Python。下面的命令适用于本次已验证机器：

```bat
@echo off

set "KERNEL_ROOT=<omni_xpu_kernel-source-directory>"
set "BUILD_ROOT=%KERNEL_ROOT%\.venv-win-py313-torch212"
set "BUILD_PYTHON=%BUILD_ROOT%\venv\Scripts\python.exe"

call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 -vcvars_ver=14.42
if errorlevel 1 exit /b 1

call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" --force
if errorlevel 1 exit /b 1

set "DNNLROOT=%ProgramFiles(x86)%\Intel\oneAPI\dnnl\2025.3"
set "OMNI_XPU_DEVICE=bmg"
set "OMNI_XPU_REQUIRE_CUTE=0"

set "PATH=%BUILD_ROOT%\venv\Library\bin;%BUILD_ROOT%\venv\Lib\site-packages\torch\lib;%DNNLROOT%\bin;%PATH%"
```

如果使用第 4 节的项目内 Windows SDK，继续在同一个 `cmd.exe` 设置：

```bat
set "SDK_NUGET=%BUILD_ROOT%\windows-sdk-nuget"
set "SDK_CPP=%SDK_NUGET%\microsoft.windows.sdk.cpp\c"
set "SDK_X64=%SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c"
set "SDK_TOOLS=%SDK_NUGET%\microsoft.windows.sdk.buildtools"
set "SDK_FILE_VERSION=10.0.26100.0"

set "INCLUDE=%SDK_CPP%\Include\%SDK_FILE_VERSION%\ucrt;%SDK_CPP%\Include\%SDK_FILE_VERSION%\shared;%SDK_CPP%\Include\%SDK_FILE_VERSION%\um;%SDK_CPP%\Include\%SDK_FILE_VERSION%\winrt;%SDK_CPP%\Include\%SDK_FILE_VERSION%\cppwinrt;%INCLUDE%"
set "LIB=%SDK_X64%\ucrt\x64;%SDK_X64%\um\x64;%LIB%"
set "PATH=%SDK_TOOLS%\bin\%SDK_FILE_VERSION%\x64;%PATH%"
```

先检查当前 shell 确实使用预期工具链：

```bat
where cl
where icx
cl /Bv
icx --version
"%BUILD_PYTHON%" -c "import torch; print(torch.__version__, torch.version.xpu)"
sycl-ls --verbose
```

在 B70 上，`sycl-ls --verbose` 应包含：

```text
Architecture: intel_gpu_bmg_g31
```

正式构建：

```bat
cd /d "%KERNEL_ROOT%"
if not exist "%BUILD_ROOT%\wheelhouse\patched" mkdir "%BUILD_ROOT%\wheelhouse\patched"

"%BUILD_PYTHON%" -m pip wheel . ^
  --wheel-dir "%BUILD_ROOT%\wheelhouse\patched" ^
  --no-build-isolation ^
  --no-deps
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
$wheelPath = Join-Path $buildRoot `
    "wheelhouse\patched\omni_xpu_kernel-0.1.0b9.dev1+torch212.bmg-cp313-cp313-win_amd64.whl"

& $buildPython -m zipfile -l $wheelPath
Get-FileHash -Algorithm SHA256 -LiteralPath $wheelPath
```

应至少看到：

```text
omni_xpu_kernel/_C.cp313-win_amd64.pyd
omni_xpu_kernel/.libs/dnnl.dll
omni_xpu_kernel/.libs/onednn/LICENSE
omni_xpu_kernel/.libs/onednn/THIRD-PARTY-PROGRAMS
omni_xpu_kernel/.libs/onednn/VERSION
omni_xpu_kernel/lgrf_uni/lgrf_sdp.cp313-win_amd64.pyd
omni_xpu_kernel/csrc/kitchen_rms_rope_sycl.cpp
omni_xpu_kernel-0.1.0b9.dev1+torch212.bmg.dist-info/METADATA
```

metadata 应包含精确的 `torch==2.12.0` 要求。`onednn==2025.3.0` 只保留
Linux x86_64 条件依赖，Windows 不再依赖不能提供 DLL 的 Python 包。

## 7. 安装到 ComfyUI Portable

下面仅以本次测试 Portable 为例。安装前先关闭所有正在使用该 Portable
Python 的 ComfyUI/Python 进程，并记录原环境：

```powershell
$portableRoot = (Resolve-Path "<ComfyUI_windows_portable-root>").Path
$embeddedPython = Join-Path $portableRoot "python_embeded\python.exe"

& $embeddedPython -m pip list
& $embeddedPython -c "import torch; print(torch.__version__)"
```

如果 Portable 原来不是 Torch 2.12，先安装匹配组合：

```powershell
& $embeddedPython -m pip install --force-reinstall `
    --index-url "https://download.pytorch.org/whl/xpu" `
    "torch==2.12.0+xpu" `
    "torchvision==0.27.0+xpu"
```

本次 Portable 保留了可用的 `torchaudio==2.11.0+xpu`。它不是
`omni_xpu_kernel` 的依赖；如果目标 Portable 不使用音频节点，可以不额外
安装 torchaudio。

安装 wheel；oneDNN runtime 已包含在 wheel 内：

```powershell
& $embeddedPython -m pip install --force-reinstall --no-deps $wheelPath
& $embeddedPython -m pip check
```

安装本地 wheel 时必须保留 `--no-deps`，避免 pip 从其他 index 替换已确认的
Torch XPU build。预期 `pip check` 输出：

```text
No broken requirements found.
```

如果 Torch 降级后 pip 报告 `Ignoring invalid distribution ~orch`，检查
`python_embeded\Lib\site-packages` 是否留下旧版本的
`~orch-*.dist-info` 临时目录。仅删除这个已经确认的临时目录，不要删除
`torch` 或有效的 `torch-2.12.0+xpu.dist-info`。

## 8. 验收

所有安装后测试都应在源码目录之外运行，否则源码 checkout 可能遮蔽
Portable 中真正安装的 wheel。

### 8.1 包身份和 XPU

```powershell
Set-Location $portableRoot

& $embeddedPython -c @"
from pathlib import Path
import importlib.metadata as metadata
import torch
import omni_xpu_kernel as omni

print("torch:", torch.__version__)
print("torch XPU runtime:", torch.version.xpu)
print("devices:", [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())])
print("package:", metadata.version("omni-xpu-kernel"))
print("module:", Path(omni.__file__).resolve())
print("metadata target:", omni.__xpu_target__)
print("core AOT target:", omni.core_aot_target())
print("available:", omni.is_available())
print("capabilities:", omni.native_capabilities())

assert torch.__version__ == "2.12.0+xpu"
assert torch.xpu.is_available()
assert omni.__xpu_target__ == "bmg"
assert omni.core_aot_target() == "bmg"
assert omni.is_available()
"@
```

### 8.2 最小原生 kernel correctness smoke

```powershell
@'
import torch
from omni_xpu_kernel import norm, sdp

for dtype in (torch.float16, torch.bfloat16, torch.float32):
    x = torch.randn(8, 2048, device="xpu", dtype=dtype)
    weight = torch.randn(2048, device="xpu", dtype=dtype)
    actual = norm.rms_norm(weight, x, eps=1e-6)
    x32 = x.float()
    expected = (
        x32
        / torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + 1e-6)
        * weight.float()
    ).to(dtype)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance
    )

for dtype in (torch.float16, torch.bfloat16):
    q = torch.randn(1, 64, 8, 128, device="xpu", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    actual = sdp.sdp(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
    ).permute(0, 2, 1, 3).contiguous()
    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance
    )

torch.xpu.synchronize()
print("native kernel smoke: PASS")
'@ | & $embeddedPython -
```

2026-08-05 的 MiniMax H3 Windows 定点验证通过：

- Kernel packaging、device dispatch 和 platform source：33 passed、3 skipped；
- `ComfyUI-OmniXPU` attention control flow：70 passed；
- Kitchen XPU suite：57 passed；backend suite：20 passed；
- Kernel H3 cached RMS-RoPE：每张 B70 各 3 passed；
- Kernel H3 INT8：每张 B70 各 2 passed；
- Kitchen H3 RMS-RoPE、INT8 和 fullgraph：每张 B70 各 4 passed；
- `ZE_AFFINITY_MASK=0` 和 `ZE_AFFINITY_MASK=1` 都只暴露目标 B70，
  上述每卡 9 项定点测试全部通过；
- MiniMax H3 INT8 safetensors 可读取，ComfyUI 模型检测得到 50 layers、
  56 heads、head dim 128；
- `pip check` 无 broken requirements。

H3 低层 packed-QKV RMS-RoPE 测试固定 XPU 随机种子，避免 BF16 随机输入只在
极少数元素上跨过绝对容差边界；固定后在两张 B70 上分别重复 5 次，10/10
通过。该修改只影响测试复现性，不改变 kernel。

当前 BMG core-only Windows wheel 的 standalone SDP 实机验收范围是
`head_dim=64/128` 的 FP16、BF16。Windows 与 Linux 使用相同 sidecar
kernel source 和参数；平台分支只负责分别通过 `GetProcAddress` 和 `dlsym`
解析相同的五个导出符号。

### 8.3 ComfyUI 启动 smoke

```powershell
& $embeddedPython (Join-Path $portableRoot "ComfyUI\main.py") `
    --windows-standalone-build `
    --disable-auto-launch `
    --quick-test-for-ci `
    --database-url "sqlite:///:memory:" `
    --log-stdout `
    --verbose INFO
```

本次测试中该命令返回码为 `0`，ComfyUI `0.30.0` 正确识别：

```text
pytorch version: 2.12.0+xpu
Device: xpu:0 Intel(R) Arc(TM) Pro B70 Graphics
Device: xpu:1 Intel(R) Arc(TM) Pro B70 Graphics
```

启动日志还应显示 `comfy-kitchen 0.2.26`、六个 MiniMax H3 workflow 模板、
`Using pytorch attention`，以及 `ComfyUI-OmniXPU` 成功导入。本文的 quick
test 证明 Portable 基础启动、设备发现和 H3 集成可加载；正式发布仍应使用
目标模型和 workflow 完成端到端验收。

## 9. 常见错误

### `assert.h` 或 Windows/UCRT 头文件缺失

安装 Visual Studio 的 Windows SDK/UCRT，或使用第 4 节的项目内 NuGet SDK，
并确认 `INCLUDE` 和 `LIB` 是在同一个已初始化的构建 shell 中设置的。

### Torch 头文件出现 `min`/`max` 宏冲突

当前 Windows build 已定义 `NOMINMAX` 和 `WIN32_LEAN_AND_MEAN`。旧 checkout
缺少这些定义时，会在 Torch 模板头文件中出现看似无关的语法错误。

### 找不到 oneDNN headers 或 `dnnl.lib`

确认：

```bat
set "DNNLROOT=%ProgramFiles(x86)%\Intel\oneAPI\dnnl\2025.3"
dir "%DNNLROOT%\include\oneapi\dnnl\dnnl.hpp"
dir "%DNNLROOT%\lib\dnnl.lib"
dir "%DNNLROOT%\bin\dnnl.dll"
dir "%DNNLROOT%\share\doc\dnnl\LICENSE"
dir "%DNNLROOT%\share\doc\dnnl\THIRD-PARTY-PROGRAMS"
```

不要混用不同版本的 oneDNN header、import library 和 runtime DLL。非标准
目录布局可以同时设置 `ONEDNN_INCLUDE`、`ONEDNN_LIB`、`ONEDNN_RUNTIME`，并
用 `ONEDNN_LICENSE_DIR` 指向包含两个 notices 的目录。

### `c10::xpu::XPUStream` 等链接错误

Windows 核心扩展必须链接 `torch_xpu.lib` 和 `c10_xpu.lib`。当前 `setup.py`
已经包含它们；出现错误通常意味着安装的不是 XPU Torch，或 Torch
`Lib\site-packages\torch\lib` 没有进入链接搜索路径。

### `lgrf_sdp.pyd` 找不到

Windows wheel 使用 ABI 后缀，例如
`lgrf_sdp.cp313-win_amd64.pyd`。当前 loader 会扫描 `lgrf_sdp*.pyd`；
固定查找无 ABI 后缀文件的旧 wheel/旧源码需要重新构建。

### wheel 可以安装但 import 失败

依次检查：

1. Python tag 是否一致，例如目标必须能使用 `cp313`；
2. Torch public version 是否为构建时的 `2.12.0`；
3. wheel target 是否与设备一致，例如 B70 使用 `bmg`；
4. `omni_xpu_kernel\.libs\dnnl.dll`、`python_embeded\Library\bin` 和
   `torch\lib` 是否可见；
5. 测试 cwd 是否离开源码 checkout。

## 10. Torch 2.13 后续阶段

Torch 2.13 不能直接复用本文的 `torch212.bmg` wheel。原 Portable 的
Torch 2.13 runtime 使用 2026 系列 SYCL/oneDNN，而本文已验证编译器是
2025.3.3。下一阶段应至少：

1. 在新的项目内目录创建独立 Torch 2.13 build environment；
2. 使用与 Torch 2.13 runtime 对齐的 2026 系列 DPC++ compiler；
3. 明确并校验对应 oneDNN header/library/runtime 版本；
4. 在 `_version.py` 和 packaging metadata 中增加 Torch 2.13 映射；
5. 重新构建 `torch213.bmg` wheel；
6. 重跑本文全部 kernel correctness 和 ComfyUI smoke。

在以上步骤完成前，不应把 Torch 2.13 标记为 Windows 已支持。
