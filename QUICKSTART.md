# 快速开始指南

## 1. 安装依赖

```bash
# 基础依赖
pip install torch

# Triton (可选，用于 GPU 加速)
pip install triton

# 安装项目（开发模式）
pip install -e .
```

## 2. 快速测试

```bash
# 方式一：运行测试脚本
./run_tests.sh

# 方式二：直接运行 Python
python test/quick_test.py
```

## 3. 使用示例

### 基础使用

```python
# 导入实现
from src.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized
import torch

# 准备输入
B, S, n, D = 2, 128, 4, 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
alpha = torch.tensor([1.1, 0.9, 1.05], device=device)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1

# 执行
h_in, h_post, h_res = mhc_forward_pre_triton_optimized(x, phi, alpha, bias)
```

### 性能测试

```bash
python test/benchmark.py
```

### 完整验证

```bash
python test/test_implementations.py --quick
```

## 4. 预期输出

```
╔══════════════════════════════════════════════════════════════╗
║     MHC Forward Pre - Quick Test Suite                      ║
╚══════════════════════════════════════════════════════════════╝

Configuration: B=2, S=256, n=4, D=256
Running on: CUDA

--- PyTorch Reference ---
Execution time: 12.3456 ms

--- Triton Implementation ---
Execution time: 4.5678 ms
Speedup: 2.70x ✓ PASS
```

## 5. 下一步

- 阅读完整文档: `README.md`
- 查看项目结构: `PROJECT_STRUCTURE.md`
- 查看英文文档: `README_IMPLEMENTATIONS.md`

## 常见问题

**Q: CUDA 内存不足？**
```python
# 减小配置
B, S, n, D = 1, 256, 4, 128
```

**Q: Triton 导入失败？**
```bash
pip install triton
```

**Q: 测试失败？**
```bash
# 使用 CPU 测试
python test/quick_test.py  # 会自动检测设备
```
