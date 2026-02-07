# MHC Backward 算子实现

本文档说明 MHC Forward Pre 算子的反向传播（backward）实现。

---

## 概述

Backward 算子计算所有输入参数的梯度：
- **dx**: 输入 `x` 的梯度
- **dphi**: 权重矩阵 `phi` 的梯度
- **dalpha**: 缩放因子 `alpha` 的梯度
- **dbias**: 偏置 `bias` 的梯度
- **dgamma**: 缩放因子 `gamma` 的梯度

---

## 实现版本

### 1. Golden 参考 (`src/golden.py`)

纯 PyTorch 实现，用于验证其他实现的正确性。

```python
from src.golden import mhc_pre_backward_manual

dx, dphi, dalpha, dbias, dgamma = mhc_pre_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)
```

### 2. Triton 实现 (`src/mhc_backward_triton.py`)

高性能 GPU kernel 实现。

```python
from src.mhc_backward_triton import mhc_backward_triton

dx, dphi, dalpha, dbias, dgamma = mhc_backward_triton(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)
```

**特点:**
- 使用 Triton GPU kernels 加速
- 原子操作累积梯度
- 适合大批次数据

### 3. TileLang 实现 (`src/mhc_backward_tilelang.py`)

使用 TileLang/TVM 的可移植实现。

```python
from src.mhc_backward_tilelang import MHCBackwardTileLang

# 编译算子
op = MHCBackwardTileLang(B, S, n, D)

dx, dphi, dalpha, dbias, dgamma = op(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)
```

**特点:**
- 跨平台可移植
- 自动优化
- 支持多种后端（CUDA, ROCm, CPU）

---

## 输入输出

### 输入

| 张量 | 形状 | 类型 | 描述 |
|------|------|------|------|
| `x` | `[B, S, n, D]` | BFloat16 | 前向输入 |
| `phi` | `[n²+2n, nD]` | Float32 | 权重矩阵 |
| `alpha` | `[3]` | Float32 | 缩放因子 |
| `bias` | `[n²+2n]` | Float32 | 偏置 |
| `inv_rms` | `[B, S]` | Float32 | 前向中间值 |
| `h_mix` | `[B, S, n²+2n]` | Float32 | 前向中间值 |
| `h_pre` | `[B, S, n]` | Float32 | 前向中间值 |
| `h_post` | `[B, S, n]` | Float32 | 前向中间值 |
| `dh_in` | `[B, S, D]` | BFloat16 | h_in 的梯度 |
| `dh_post` | `[B, S, n]` | Float32 | h_post 的梯度 |
| `dh_res` | `[B, S, n, n]` | Float32 | h_res 的梯度 |
| `gamma` | `[n, D]` | Float32 | 缩放因子 |

### 输出

| 张量 | 形状 | 类型 | 描述 |
|------|------|------|------|
| `dx` | `[B, S, n, D]` | BFloat16 | x 的梯度 |
| `dphi` | `[n²+2n, nD]` | Float32 | phi 的梯度 |
| `dalpha` | `[3]` | Float32 | alpha 的梯度 |
| `dbias` | `[n²+2n]` | Float32 | bias 的梯度 |
| `dgamma` | `[n, D]` | Float32 | gamma 的梯度 |

---

## 计算流程

```
输入: dh_in, dh_post, dh_res, 前向中间值

1. dh_pre = sum(dh_in * x, axis=D)
   └─ 从 h_in 的反向传播

2. dh_pre2 = dh_pre * sigmoid_grad(h_pre)
   dh_post2 = dh_post * sigmoid_grad(h_post)
   └─ Sigmoid 激活函数的反向传播

3. dh_pre1 = alpha[0] * dh_pre2
   dh_post1 = alpha[1] * dh_post2
   dh_res1 = alpha[2] * dh_res
   └─ Alpha 缩放的反向传播

4. dh_mix = concat([dh_pre1, dh_post1, dh_res1]) * inv_rms
   └─ 合并并应用 RMSNorm

5. dvecX_mm = dh_mix @ phi
   dphi = dh_mix^T @ (x * gamma)
   └─ GEMM 的反向传播

6. dalpha[0] = sum(dh_pre2 * h_pre1)
   dalpha[1] = sum(dh_post2 * h_post1)
   dalpha[2] = sum(dh_res * h_res1)
   └─ Alpha 的梯度

7. dbias = concat([sum(dh_pre2), sum(dh_post2), sum(dh_res)])
   └─ Bias 的梯度

8. dinv_rms = sum(dh_mix_tmp * h_mix)
   dvecX_inv = -dinv_rms * inv_rms^3 / nD * vecX
   └─ RMSNorm 的反向传播

9. dvecX_hin = h_pre * dh_in
   └─ h_in 计算的反向传播

10. dx = dvecX_mm * gamma + dvecX_inv + dvecX_hin
    └─ 合并所有来源的梯度

11. dgamma = x * dvecX_mm
    └─ Gamma 的梯度

输出: dx, dphi, dalpha, dbias, dgamma
```

---

## 测试

运行测试验证实现的正确性：

```bash
# 测试 backward 实现
python test/test_backward.py
```

**预期输出：**
```
╔════════════════════════════════════════════════════════════════╗
║     MHC Backward - Test Suite                                 ║
╚════════════════════════════════════════════════════════════════╝

======================================================================
Testing Triton Backward vs Golden Reference
======================================================================

Configuration: B=2, S=64, n=4, D=128

[1/3] Computing golden reference backward...
[2/3] Computing Triton backward...
[3/3] Comparing results...

--- Gradient Comparison ---
  dx          : max_err=0.000123, mean_err=0.000045 [PASS]
  dphi        : max_err=0.000089, mean_err=0.000032 [PASS]
  dalpha      : max_err=0.000012, mean_err=0.000004 [PASS]
  dbias       : max_err=0.000056, mean_err=0.000021 [PASS]
  dgamma      : max_err=0.000034, mean_err=0.000012 [PASS]

[PASS] All gradients within tolerance
```

---

## 使用示例

### 完整的前向 + 反向传播

```python
from src.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized
from src.mhc_backward_triton import mhc_backward_triton
import torch

# 准备输入
B, S, n, D = 2, 128, 4, 256
device = 'cuda'

x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1
gamma = torch.randn(n, D, dtype=torch.float32, device=device)

# 前向传播
h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre_triton_optimized(
    x, phi, alpha, bias, outflag=True
)

# 计算损失
loss = h_in.sum() + h_post.sum() + h_res.sum()

# 计算梯度
dh_in = torch.ones_like(h_in)
dh_post = torch.ones_like(h_post)
dh_res = torch.ones_like(h_res)

# 反向传播
dx, dphi, dalpha, dbias, dgamma = mhc_backward_triton(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

print(f"dx shape: {dx.shape}")       # [2, 128, 4, 256]
print(f"dphi shape: {dphi.shape}")   # [24, 1024]
print(f"dalpha: {dalpha}")           # [3]
print(f"dbias shape: {dbias.shape}") # [24]
print(f"dgamma shape: {dgamma.shape}") # [4, 256]
```

---

## 性能优化建议

### Triton 实现
- **大批次**: 使用 `BLOCK_SIZE_K = 64` 或更高
- **小批次**: 使用 `BLOCK_SIZE_K = 32`

### TileLang 实现
- 编译时会自动优化
- 可以通过 TVM schedule 进一步调优

---

## 故障排查

### Q: Triton 实现运行错误
**A**: 确保所有张量在同一个 GPU 设备上，且内存连续（contiguous）

### Q: 梯度不匹配
**A**: 检查是否使用了相同的前向中间值（inv_rms, h_mix 等）

### Q: 内存不足
**A**: 减小批次大小或使用更小的 block size

---

## 参考资料

- [Triton Documentation](https://triton-lang.org/)
- [TVM Documentation](https://tvm.apache.org/)
- [PyTorch Autograd](https://pytorch.org/docs/stable/autograd.html)
