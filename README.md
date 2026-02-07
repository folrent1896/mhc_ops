# MHC Forward Pre 算子实现

多种后端实现的 `mhc_forward_pre` 流形约束超连接前置算子。

---

## 📋 目录

- [概述](#概述)
- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [使用示例](#使用示例)
- [API 参考](#api-参考)
- [测试](#测试)

---

## 概述

本项目提供了 `mhc_forward_pre` 算子的多种高性能实现：

| 实现 | 描述 | 优势 |
|------|------|------|
| **Golden** | PyTorch 参考实现 | 验证正确性的基准 |
| **Triton** | GPU kernel 实现 | 高性能 GPU 加速 |
| **TileLang** | DSL 可移植实现 | 跨平台优化 |

---

## 特性

✨ **前向传播 (Forward)**
- GEMM 矩阵乘法
- RMSNorm 归一化
- Sigmoid 激活函数
- 支持可变批次大小和序列长度

✨ **反向传播 (Backward)**
- 完整的梯度计算
- 支持 `dx`, `dphi`, `dalpha`, `dbias`, `dgamma`
- 与前向传播无缝集成

✨ **多种后端**
- **Golden**: 纯 PyTorch，易于调试
- **Triton**: 高性能 GPU kernel
- **TileLang**: 跨平台可移植

---

## 安装

### 环境要求

```bash
Python >= 3.8
CUDA >= 11.8 (可选，用于 GPU 加速)
```

### 安装依赖

```bash
# 基础依赖
pip install torch

# Triton (GPU 加速)
pip install triton

# TileLang (可选)
pip install tilelang tvm
```

### 从源码安装

```bash
git clone https://github.com/folrent1896/mhc_ops.git
cd mhc_ops

# 开发模式安装
pip install -e .
```

---

## 快速开始

### 1. 前向传播 (Forward)

```python
from src.forward import mhc_forward_pre
import torch

# 准备输入
B, S, n, D = 2, 128, 4, 256
x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1

# 前向传播
h_in, h_post, h_res = mhc_forward_pre(x, phi, alpha, bias)

print(f"h_in shape: {h_in.shape}")      # [2, 128, 256]
print(f"h_post shape: {h_post.shape}")  # [2, 128, 4]
print(f"h_res shape: {h_res.shape}")    # [2, 128, 4, 4]
```

### 2. 使用 Triton 加速

```python
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized
import torch

# 在 GPU 上运行
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1

# 前向传播 (GPU 加速)
h_in, h_post, h_res = mhc_forward_pre_triton_optimized(x, phi, alpha, bias)
```

### 3. 反向传播 (Backward)

```python
from src.forward import mhc_forward_pre
from src.backward import mhc_backward_manual
import torch

# 前向传播 (需要中间值)
h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
    x, phi, alpha, bias, outflag=True
)

# 准备梯度
dh_in = torch.randn_like(h_in)
dh_post = torch.randn_like(h_post)
dh_res = torch.randn_like(h_res)
gamma = torch.randn(n, D)

# 反向传播
dx, dphi, dalpha, dbias, dgamma = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)
```

---

## 目录结构

```
mhc-ops/
├── src/
│   ├── forward/                    # 前向传播
│   │   ├── golden.py              # Golden 参考
│   │   ├── mhc_forward_pre_triton.py       # Triton
│   │   └── mhc_forward_pre_tilelang.py     # TileLang
│   │
│   ├── backward/                   # 反向传播
│   │   ├── golden.py              # Golden 参考
│   │   ├── mhc_backward_triton.py          # Triton
│   │   └── mhc_backward_tilelang.py        # TileLang
│   │
│   └── __init__.py               # 统一导出
│
├── test/
│   ├── forward/                    # 前向测试
│   │   ├── test_forward.py       # 完整测试
│   │   ├── benchmark.py          # 性能测试
│   │   └── quick_test.py         # 快速验证
│   │
│   └── backward/                   # 反向测试
│       └── test_backward.py      # Backward 测试
│
├── README.md                      # 本文档
├── QUICKSTART.md                  # 快速开始
├── BACKWARD.md                    # Backward 文档
├── PROJECT_STRUCTURE.md            # 项目结构
├── requirements.txt                # 依赖列表
└── setup.py                       # 安装配置
```

---

## 使用示例

### 基础使用

```python
from src.forward import mhc_forward_pre

# 输入
B, S, n, D = 1, 256, 4, 256
x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1

# 执行
h_in, h_post, h_res = mhc_forward_pre(x, phi, alpha, bias)
```

### GPU 加速

```python
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

device = 'cuda'
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1

h_in, h_post, h_res = mhc_forward_pre_triton_optimized(x, phi, alpha, bias)
```

### 完整的前向 + 反向

```python
from src.forward import mhc_forward_pre
from src.backward import mhc_backward_manual

# 前向
h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
    x, phi, alpha, bias, outflag=True
)

# 计算损失
loss = h_in.sum() + h_post.sum() + h_res.sum()

# 反向
dh_in = torch.ones_like(h_in)
dh_post = torch.ones_like(h_post)
dh_res = torch.ones_like(h_res)

dx, dphi, dalpha, dbias, dgamma = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)
```

---

## API 参考

### Forward 算子

#### `mhc_forward_pre(x, phi, alpha, bias, outflag=False, norm_eps=1e-6, hc_eps=1e-6)`

Golden 参考实现的前向传播。

**参数:**
- `x` ([B, S, n, D]): 输入张量 (BFloat16)
- `phi` ([n²+2n, nD]): 权重矩阵 (Float32)
- `alpha` ([3]): 缩放因子 [pre, post, res] (Float32)
- `bias` ([n²+2n]): 偏置向量 (Float32)
- `outflag` (bool): 是否返回中间值
- `norm_eps` (float): RMSNorm epsilon
- `hc_eps` (float): Hyper connection epsilon

**返回:**
- `h_in` ([B, S, D]): 前置门控加权输入 (BFloat16)
- `h_post` ([B, S, n]): 后置门控激活值 (Float32)
- `h_res` ([B, S, n, n]): 残差门控矩阵 (Float32)

#### `mhc_forward_pre_triton_optimized(x, phi, alpha, bias, outflag=False, norm_eps=1e-6, hc_eps=1e-6)`

Triton 优化版本的前向传播，性能更高。

**参数与返回**: 同 `mhc_forward_pre`

### Backward 算子

#### `mhc_backward_manual(x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post, dh_in, dh_post, dh_res, gamma, norm_eps=1e-6, hc_eps=1e-6)`

Golden 参考实现的反向传播。

**参数:**
- `x`, `phi`, `alpha`, `bias`: 前向输入
- `inv_rms`, `h_mix`, `h_pre`, `h_post`: 前向中间值
- `dh_in`, `dh_post`, `dh_res`: 输出梯度
- `gamma` ([n, D]): 缩放因子

**返回:**
- `dx` ([B, S, n, D]): x 的梯度
- `dphi` ([n²+2n, nD]): phi 的梯度
- `dalpha` ([3]): alpha 的梯度
- `dbias` ([n²+2n]): bias 的梯度
- `dgamma` ([n, D]): gamma 的梯度

---

---

## 测试

### 快速测试

```bash
# Forward 快速测试
python test/forward/quick_test.py

# Forward 性能测试
python test/forward/benchmark.py

# Backward 测试
python test/backward/test_backward.py
```

### 完整测试套件

```bash
# Forward 完整测试
python test/forward/test_forward.py --quick

# 自定义配置
python test/forward/test_forward.py --device cuda --rtol 1e-4
```

---

## 文档

- **[QUICKSTART.md](QUICKSTART.md)** - 5分钟快速上手
- **[BACKWARD.md](BACKWARD.md)** - Backward 算子详细文档
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 项目结构说明

---

## 引用

如果您在研究中使用了这些实现，请引用：

```bibtex
@software{mhc_ops,
  title={MHC Forward Pre Operator Implementations},
  author={Your Name},
  year={2025},
  url={https://github.com/folrent1896/mhc_ops}
}
```

---

## 许可证

请参考主仓库的许可证。

---

## 联系方式

- GitHub: [https://github.com/folrent1896/mhc_ops](https://github.com/folrent1896/mhc_ops)
- Issues: [提交问题](https://github.com/folrent1896/mhc_ops/issues)
