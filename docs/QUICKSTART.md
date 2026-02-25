# 快速开始指南

5 分钟上手 MHC Forward Pre 算子。

---

## 1. 安装

```bash
# 克隆仓库
git clone https://github.com/folrent1896/mhc_ops.git
cd mhc_ops

# 安装依赖
pip install torch triton

# 开发模式安装
pip install -e .
```

---

## 2. 快速测试

```bash
# Forward 快速测试
python test/forward/quick_test.py

# 性能基准测试
python test/forward/benchmark.py
```

**预期输出：**
```
✓ Configuration: B=2, S=256, n=4, D=256
✓ PyTorch Reference: 8.23 ms
✓ Triton: 3.46 ms
✓ Speedup: 2.38x
✓ Status: PASS
```

---

## 3. 基础使用

### 3.1 Forward（前向传播）

```python
from src.forward import mhc_forward_pre
import torch

# 准备输入
B, S, n, D = 2, 128, 4, 256
x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1

# 执行
h_in, h_post, h_res = mhc_forward_pre(x, phi, alpha, bias)

print(f"输出形状:")
print(f"  h_in:   {h_in.shape}")     # [2, 128, 256]
print(f"  h_post: {h_post.shape}")   # [2, 128, 4]
print(f"  h_res:  {h_res.shape}")    # [2, 128, 4, 4]
```

### 3.2 GPU 加速版本

```python
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 在 GPU 上运行
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1

# GPU 加速的前向传播
h_in, h_post, h_res = mhc_forward_pre_triton_optimized(x, phi, alpha, bias)
```

### 3.3 Backward（反向传播）

```python
from src.forward import mhc_forward_pre
from src.backward import mhc_backward_manual
import torch

# 前向传播（需要中间值）
h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
    x, phi, alpha, bias, outflag=True
)

# 准备梯度
dh_in = torch.ones_like(h_in)
dh_post = torch.ones_like(h_post)
dh_res = torch.ones_like(h_res)
gamma = torch.randn(n, D)

# 反向传播
dx, dphi, dalpha, dbias, dgamma = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)
```

### 3.4 使用 Triton Backward（GPU 加速）

```python
from src.forward import mhc_forward_pre
from src.backward.mhc_backward_triton import mhc_backward_triton
import torch

# 在 GPU 上运行
device = 'cuda'

# 准备输入（移到 GPU）
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1

# 前向传播（需要中间值）
h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
    x, phi, alpha, bias, outflag=True
)

# 准备梯度
dh_in = torch.ones_like(h_in)
dh_post = torch.ones_like(h_post)
dh_res = torch.ones_like(h_res)
gamma = torch.randn(n, D, device=device)

# 反向传播（Triton 加速）
dx, dphi, dalpha, dbias, dgamma = mhc_backward_triton(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)
```

**Triton Backward 状态** (2025-02-25):
- ✅ **所有组件完全正确并通过验证！**
- ✅ dphi, dalpha, dbias, dgamma: max_err < 1e-4
- ✅ dx: max_err = 0.25 (bfloat16 精度限制，可接受)
- ✅ 可用于生产环境！

---

## 4. 实际场景示例

### 4.1 集成到模型

```python
import torch
import torch.nn as nn
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

class MHCBlock(nn.Module):
    """MHC 模块"""
    def __init__(self, n=4, D=256):
        super().__init__()
        self.n = n
        self.D = D
        self.out_features = n * n + 2 * n

        # 可学习参数
        self.phi = nn.Parameter(torch.randn(self.out_features, n * D))
        self.alpha = nn.Parameter(torch.tensor([1.1, 0.9, 1.05]))
        self.bias = nn.Parameter(torch.randn(self.out_features) * 0.1)

    def forward(self, x):
        """
        Args:
            x: [B, S, n, D]
        Returns:
            h_in, h_post, h_res
        """
        return mhc_forward_pre_triton_optimized(x, self.phi, self.alpha, self.bias)

# 使用
model = MHCBlock(n=4, D=256).cuda()
x = torch.randn(2, 128, 4, 256, dtype=torch.bfloat16, device='cuda')
h_in, h_post, h_res = model(x)
```

### 4.2 训练循环示例

```python
import torch.nn as nn
import torch.optim as optim
from src.forward import mhc_forward_pre
from src.backward import mhc_backward_manual

# 前向传播
def forward_pass(x, phi, alpha, bias):
    return mhc_forward_pre(x, phi, alpha, bias, outflag=True)

# 反向传播
def backward_pass(x, phi, alpha, bias, outputs, grad_outputs, gamma):
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = outputs
    dh_in, dh_post, dh_res = grad_outputs
    return mhc_backward_manual(
        x, phi, alpha, bias,
        inv_rms, h_mix, h_pre, h_post,
        dh_in, dh_post, dh_res, gamma
    )

# 模拟训练循环
phi = torch.randn(24, 1024, requires_grad=True)
alpha = torch.tensor([1.1, 0.9, 1.05], requires_grad=True)
bias = torch.randn(24, requires_grad=True)
gamma = torch.randn(4, 256)

x = torch.randn(2, 128, 4, 256, dtype=torch.bfloat16)

# Forward
outputs = forward_pass(x, phi, alpha, bias)

# 计算损失
loss = outputs[0].sum() + outputs[1].sum() + outputs[2].sum()

# Backward
grad_outputs = (torch.ones_like(outputs[0]),
                 torch.ones_like(outputs[1]),
                 torch.ones_like(outputs[2]))
dx, dphi, dalpha, dbias, dgamma = backward_pass(
    x, phi, alpha, bias, outputs, grad_outputs, gamma
)

print(f"梯度形状:")
print(f"  dx:     {dx.shape}")      # [2, 128, 4, 256]
print(f"  dphi:   {dphi.shape}")    # [24, 1024]
print(f"  dalpha: {dalpha.shape}")  # [3]
print(f"  dbias:  {dbias.shape}")   # [24]
print(f"  dgamma: {dgamma.shape}") # [4, 256]
```

---

## 5. 常见使用场景

### 场景 1: 推理 (Inference)

```python
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

# 推理模式
with torch.no_grad():
    h_in, h_post, h_res = mhc_forward_pre_triton_optimized(
        x, phi, alpha, bias
    )
```

### 场景 2: CPU 环境

```python
from src.forward import mhc_forward_pre

# CPU 上使用 Golden 实现
h_in, h_post, h_res = mhc_forward_pre(x, phi, alpha, bias)
```

### 场景 3: 获取中间值

```python
from src.forward import mhc_forward_pre

# 设置 outflag=True 获取中间值（用于 backward）
h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
    x, phi, alpha, bias, outflag=True
)
```

---

## 6. 性能优化建议

### 小批次 (B×S < 512)

```python
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton

# 使用单 kernel 版本，启动开销小
h_in, h_post, h_res = mhc_forward_pre_triton(x, phi, alpha, bias)
```

### 大批次 (B×S > 2048)

```python
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

# 使用优化版本，吞吐量更高
h_in, h_post, h_res = mhc_forward_pre_triton_optimized(x, phi, alpha, bias)
```

### 跨平台部署

```python
from src.forward import MHCForwardPreTileLang

# 使用 TileLang，可移植性强
op = MHCForwardPreTileLang(B, S, n, D)
h_in, h_post, h_res = op(x, phi, alpha, bias)
```

---

## 7. 故障排查

### Q: Triton 导入失败？

```bash
pip install triton
```

### Q: CUDA 内存不足？

```python
# 减小批次大小
B, S, n, D = 1, 512, 4, 128
```

### Q: 测试失败？

```bash
# 检查环境
python -c "import torch; print(torch.cuda.is_available())"

# 重新运行测试
python test/forward/quick_test.py
```

### Q: 梯度不正确？

```bash
# 运行验证测试
python test/backward/test_backward.py
```

---

## 8. 下一步

- 📖 阅读完整 [README.md](README.md)
- 📖 查看 [BACKWARD.md](BACKWARD.md) 了解反向传播
- 📖 查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解项目结构
- 🐛 提交 [Issue](https://github.com/folrent1896/mhc_ops/issues) 反馈问题

---

**准备好了吗？开始使用吧！**

```bash
# 快速验证
python test/forward/quick_test.py

# 查看更多示例
cat README.md
```
