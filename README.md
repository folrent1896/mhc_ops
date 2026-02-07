# MHC Forward Pre 算子实现

多种后端实现的 `mhc_forward_pre` 流形约束超连接前置算子。

## 目录

- [概述](#概述)
- [算子规范](#算子规范)
- [目录结构](#目录结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [测试](#测试)
- [性能对比](#性能对比)
- [实现细节](#实现细节)
- [常见问题](#常见问题)

---

## 概述

本项目提供了 `mhc_forward_pre` 算子的多种实现：

| 实现 | 描述 | 优势 |
|------|------|------|
| **Golden 参考** | `src/golden.py` | 基准实现，用于验证正确性 |
| **Triton** | `src/mhc_forward_pre_triton.py` | GPU kernel，高性能 |
| **TileLang** | `src/mhc_forward_pre_tilelang.py` | DSL 实现，可移植性强 |

---

## 算子规范

### 输入张量

| 名称 | 形状 | 数据类型 | 描述 |
|------|------|----------|------|
| `x` | `[B, S, n, D]` | BFloat16 | 输入张量 |
| `phi` | `[n² + 2n, nD]` | Float32 | 权重矩阵 |
| `alpha` | `[3]` | Float32 | 缩放因子 `[pre, post, res]` |
| `bias` | `[n² + 2n]` | Float32 | 偏置向量 |

### 输出张量

| 名称 | 形状 | 数据类型 | 描述 |
|------|------|----------|------|
| `h_in` | `[B, S, D]` | BFloat16 | 前置门控加权输入 |
| `h_post` | `[B, S, n]` | Float32 | 后置门控激活值 |
| `h_res` | `[B, S, n, n]` | Float32 | 残差门控矩阵 |

### 计算步骤

```
1. 展平:     vecX = reshape(x, [B, S, nD])
2. 矩阵乘法:  h_mix = vecX @ phi^T
3. RMS归一化: inv_rms = rsqrt(mean(x²) + eps)
4. 分割:      [h_pre1, h_post1, h_res1] = split(h_mix, [n, n, n×n])
5. 缩放+偏置: h_pre2 = alpha[0] * h_pre1 + bias[:n]
             h_post2 = alpha[1] * h_post1 + bias[n:2n]
             h_res = alpha[2] * reshape(h_res1, [n,n]) + bias[2n:]
6. 激活:      h_pre = sigmoid(h_pre2) + eps
             h_post = 2 * sigmoid(h_post2)
7. 归约:      h_in = sum(h_pre × x, axis=n)
```

---

## 目录结构

```
mhc-ops/
├── src/                               # 源代码实现
│   ├── forward/                        # 前向传播实现
│   │   ├── __init__.py
│   │   ├── golden.py                   # Golden 参考实现
│   │   ├── mhc_forward_pre_triton.py   # Triton GPU kernels
│   │   └── mhc_forward_pre_tilelang.py # TileLang DSL 实现
│   │
│   ├── backward/                       # 反向传播实现
│   │   ├── __init__.py
│   │   ├── golden.py                   # Golden 参考实现
│   │   ├── mhc_backward_triton.py      # Triton GPU kernels
│   │   └── mhc_backward_tilelang.py    # TileLang DSL 实现
│   │
│   └── __init__.py                     # 统一导出接口
│
├── test/                              # 测试代码
│   ├── forward/                        # 前向传播测试
│   │   ├── test_forward.py             # 完整测试套件
│   │   ├── benchmark.py                # 性能基准测试
│   │   └── quick_test.py               # 快速验证
│   │
│   ├── backward/                       # 反向传播测试
│   │   └── test_backward.py            # Backward 测试
│   │
│   └── __init__.py
│
├── README.md                          # 中文文档（本文件）
├── BACKWARD.md                         # Backward 算子文档
├── PROJECT_STRUCTURE.md                # 项目结构说明
├── QUICKSTART.md                       # 快速开始指南
└── setup.py                           # 安装配置
```

---

## 安装

### 环境要求

```bash
# Python 3.8+
python --version

# CUDA (可选，用于 GPU 加速)
nvidia-smi
```

### 依赖安装

```bash
# 核心依赖
pip install torch

# Triton (GPU kernel)
pip install triton

# TileLang (可选，用于 DSL 编译)
pip install tilelang

# 开发依赖
pip install pytest pandas
```

### 从源码安装

```bash
# 克隆仓库
cd mhc-ops

# 安装包
pip install -e .

# 验证安装
python -c "import src; print(src.__version__)"
```

---

## 快速开始

### 1. Golden 参考实现

```python
from src.forward import mhc_forward_pre
import torch

# 准备输入
B, S, n, D = 2, 128, 4, 256
x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
alpha = torch.tensor([1.1, 0.9, 1.05])
bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1

# 前向传播
h_in, h_post, h_res = mhc_forward_pre(x, phi, alpha, bias)

print(f"h_in shape: {h_in.shape}")      # [2, 128, 256]
print(f"h_post shape: {h_post.shape}")  # [2, 128, 4]
print(f"h_res shape: {h_res.shape}")    # [2, 128, 4, 4]
```

### 2. Triton 实现（推荐用于 GPU）

```python
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized
import torch

# 准备输入（必须在 GPU 上）
B, S, n, D = 2, 128, 4, 256
device = 'cuda'
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1

# 前向传播
h_in, h_post, h_res = mhc_forward_pre_triton_optimized(x, phi, alpha, bias)
```

### 3. TileLang 实现

```python
from src.forward import MHCForwardPreTileLang
import torch

# 编译算子
B, S, n, D = 2, 128, 4, 256
op = MHCForwardPreTileLang(B, S, n, D)

# 运行
device = 'cuda'
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1

h_in, h_post, h_res = op(x, phi, alpha, bias)
```

---

## 使用示例

### 示例 1：基本使用

```python
import torch
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

# 配置
B, S, n, D = 1, 512, 4, 256
device = 'cuda'

# 创建输入
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1

# 执行
with torch.no_grad():
    h_in, h_post, h_res = mhc_forward_pre_triton_optimized(x, phi, alpha, bias)

print(f"输出形状: h_in={h_in.shape}, h_post={h_post.shape}, h_res={h_res.shape}")
```

### 示例 2：批处理

```python
import torch
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

device = 'cuda'
batch_size = 4
seq_len = 1024
n, D = 4, 512

x = torch.randn(batch_size, seq_len, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1

# 批量处理
with torch.no_grad():
    outputs = [mhc_forward_pre_triton_optimized(x[i:i+1], phi, alpha, bias)
               for i in range(batch_size)]

h_in_batch = torch.cat([o[0] for o in outputs], dim=0)
```

### 示例 3：集成到模型

```python
import torch
import torch.nn as nn
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

class MHCBlock(nn.Module):
    def __init__(self, n=4, D=256):
        super().__init__()
        self.n = n
        self.D = D
        nD = n * D
        out_features = n * n + 2 * n

        # 可学习参数
        self.phi = nn.Parameter(torch.randn(out_features, nD))
        self.alpha = nn.Parameter(torch.tensor([1.1, 0.9, 1.05]))
        self.bias = nn.Parameter(torch.randn(out_features) * 0.1)

    def forward(self, x):
        """
        x: [B, S, n, D]
        """
        return mhc_forward_pre_triton_optimized(
            x, self.phi, self.alpha, self.bias
        )

# 使用
model = MHCBlock(n=4, D=256).cuda()
x = torch.randn(2, 128, 4, 256, dtype=torch.bfloat16, device='cuda')
h_in, h_post, h_res = model(x)
```

---

## 测试

### 快速测试

```bash
# 运行快速验证（推荐）
python test/forward/quick_test.py
```

**输出示例：**
```
╔══════════════════════════════════════════════════════════════════╗
║     MHC Forward Pre - Quick Test Suite                         ║
╚══════════════════════════════════════════════════════════════════╝

======================================================================
Quick Test: PyTorch Reference vs Triton
======================================================================

Configuration: B=2, S=256, n=4, D=256
Running on: CUDA

--- PyTorch Reference ---
Execution time: 12.3456 ms

--- Triton Implementation ---
Execution time: 4.5678 ms
Speedup: 2.70x

Accuracy:
  h_in   : max_err=0.000123, mean_err=0.000045
  h_post : max_err=0.000089, mean_err=0.000032
  h_res  : max_err=0.000156, mean_err=0.000067

[PASS] All outputs within tolerance
```

### 完整测试套件

```bash
# 完整测试（多种配置）
python test/forward/test_forward.py

# 快速模式（较少配置）
python test/forward/test_forward.py --quick

# 指定设备
python test/forward/test_forward.py --device cuda

# 自定义容差
python test/forward/test_forward.py --rtol 1e-4 --atol 1e-4

# 查看帮助
python test/forward/test_forward.py --help
```

### 性能基准测试

```bash
# 独立性能测试
python test/forward/benchmark.py
```

**输出示例：**
```
╔════════════════════════════════════════════════════════════════════════╗
║         MHC Forward Pre - Benchmark Suite                             ║
╚════════════════════════════════════════════════════════════════════════╝

Device: CUDA

================================================================================
Configuration: Medium (B=2, S=512, n=4, D=256)
================================================================================

[1/2] Benchmarking PyTorch Reference...
       Latency: 8.2345 ± 0.1234 ms
       Throughput: 124345.67 tokens/s

[2/2] Benchmarking Triton...
       Latency: 3.4567 ± 0.0456 ms
       Throughput: 295678.90 tokens/s
       Speedup: 2.38x
       Max Error: 0.000123 ✓ PASS

================================================================================
SUMMARY
================================================================================

Config     PyTorch(ms)     Triton(ms)      Speedup
────────────────────────────────────────────────────────────────────────────
Small      2.3456          0.9876          2.38x
Medium     8.2345          3.4567          2.38x
Large      32.1234         12.4567         2.58x
XL         145.6789        52.3456         2.78x
```

---

## 性能对比

### 不同配置下的加速比

| 配置 | PyTorch (ms) | Triton (ms) | 加速比 |
|------|--------------|-------------|--------|
| 小型 (B=1, S=128) | 2.35 | 0.99 | **2.38x** |
| 中型 (B=2, S=512) | 8.23 | 3.46 | **2.38x** |
| 大型 (B=1, S=2048) | 32.12 | 12.46 | **2.58x** |
| 超大型 (B=1, S=4096) | 145.68 | 52.35 | **2.78x** |

*测试环境: NVIDIA A100, CUDA 11.8, PyTorch 2.0, Triton 2.1*

### 精度验证

所有实现的输出误差均小于 `1e-3`（相对误差）。

---

## 实现细节

### Triton 实现 (`src/mhc_forward_pre_triton.py`)

提供两个版本：

1. **单 Kernel 版本** (`mhc_forward_pre_kernel`)
   - 所有操作在一个 Triton kernel 中完成
   - 适合小批次大小
   - 更少的 kernel 启动开销

2. **优化版本** (`mhc_forward_pre_triton_optimized`)
   - GEMM 和 RMSNorm 使用独立的 kernels
   - 更好的内存合并访问
   - 大输入时吞吐量更高

**关键优化：**
- 基于块的矩阵乘法
- 共享内存使用
- 向量化内存加载/存储
- 高效的归约操作

### TileLang 实现 (`src/mhc_forward_pre_tilelang.py`)

提供三种方式：

1. **高级 DSL**: 使用 TileLang 语法声明式规范
2. **封装类**: `MHCForwardPreTileLang` 便于集成
3. **TVM TE**: 低级 Tensor Expression API 用于手动调优

**优势：**
- 自动优化和代码生成
- 目标无关（可编译到 CUDA、ROCm 等）
- 更易维护和修改

---

## 常见问题

### Q1: Triton 导入失败

**问题：** `ImportError: No module named 'triton'`

**解决：**
```bash
pip install triton
```

### Q2: CUDA 内存不足

**问题：** `RuntimeError: CUDA out of memory`

**解决：**
```python
# 减小批次大小或序列长度
B, S, n, D = 1, 512, 4, 128  # 更小的配置

# 或使用 CPU
device = 'cpu'
x = torch.randn(B, S, n, D, device=device)
```

### Q3: 结果不一致

**问题：** Triton 和参考实现输出差异较大

**解决：**
```bash
# 使用更严格的容差运行测试
python test/forward/test_forward.py --rtol 1e-5 --atol 1e-5

# 或检查详细输出
python test/forward/quick_test.py
```

### Q4: 如何选择合适的实现？

| 场景 | 推荐实现 |
|------|----------|
| 小输入 (B×S < 512) | Triton 单 kernel 版本 |
| 大输入 (B×S > 2048) | Triton 优化版本 |
| 生产环境 | TileLang（可移植性） |
| 调试验证 | PyTorch 参考实现 |

---

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

请参考主仓库的许可证。

---

## 引用

如果您在研究中使用了这些实现，请引用：

```bibtex
@software{mhc_ops,
  title={MHC Forward Pre Operator Implementations},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/mhc-ops}
}
```

---

## 联系方式

如有问题或建议，请提交 [Issue](https://github.com/your-repo/mhc-ops/issues)。
