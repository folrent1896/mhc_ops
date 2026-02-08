# MHC Ops 项目知识库

## Conda 虚拟环境

### 环境信息
- **环境名称**: `mhc_ops`
- **环境路径**: `/home/huan1178/miniconda3/envs/mhc_ops`
- **Python 版本**: 3.10.19
- **CUDA 版本**: 12.4 (Runtime 12.4.1, 版本 13.1)

### 环境配置

#### 核心依赖
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4 支持)
- **Triton**: 3.2.0
- **NumPy**: 2.2.6 (pip) + 2.2.5 (conda base)

#### CUDA 库
- cuda-cudart: 12.4.127
- cuda-cupti: 12.4.127
- cuda-nvrtc: 12.4.127
- libcublas: 12.4.5.8
- libcufft: 11.2.1.3
- libcusolver: 11.6.1.9
- libcusparse: 12.3.1.170
- libcurand: 10.4.1.81
- torchtriton: 3.1.0

#### 数学库
- MKL: 2025.0.0
- intel-openmp: 2025.0.0

### 启动方式

#### 方式 1: 使用 conda run（推荐）
```bash
# 运行测试
conda run -n mhc_ops python test/forward/quick_test.py

# 运行性能基准测试
conda run -n mhc_ops python test/forward/benchmark.py

# 运行后向测试
conda run -n mhc_ops python test/backward/test_backward.py
```

#### 方式 2: 激活环境
```bash
# 激活环境（需要先运行 conda init）
conda activate mhc_ops

# 然后直接运行命令
python test/forward/quick_test.py
```

注意：如果 `conda activate` 不可用，需要先运行 `conda init` 并重启终端。

### 快速测试

#### Forward 测试
```bash
conda run -n mhc_ops python test/forward/quick_test.py
```

**预期输出**：
- ✅ PyTorch Reference vs Triton 比较
- ✅ 多配置性能测试
- ✅ 功能正确性验证
- ✅ 所有输出误差应接近 0

#### 性能基准测试
```bash
conda run -n mhc_ops python test/forward/benchmark.py
```

### 依赖安装

如需重新安装依赖：
```bash
# 安装 PyTorch (CUDA 12.4)
conda run -n mhc_ops pip install torch torchvision triton --index-url https://download.pytorch.org/whl/cu124

# 或安装所有依赖
conda run -n mhc_ops pip install -r requirements.txt

# 以开发模式安装项目
conda run -n mhc_ops pip install -e .
```

### 环境管理

#### 导出环境配置
```bash
conda env export -n mhc_ops > environment.yml
```

#### 从配置文件创建环境
```bash
conda env create -f environment.yml
```

#### 删除环境
```bash
conda remove -n mhc_ops --all
```

### 项目结构
```
mhc_ops/
├── src/
│   ├── forward/      # 前向传播实现
│   └── backward/     # 反向传播实现
├── test/
│   ├── forward/      # 前向测试
│   └── backward/     # 反向测试
├── setup.py          # 安装配置
├── requirements.txt  # 依赖列表
└── QUICKSTART.md     # 快速开始指南
```

### 常见问题

#### Q: 测试时提示 "ModuleNotFoundError: No module named 'src'"
A: 需要以开发模式安装项目：
```bash
conda run -n mhc_ops pip install -e .
```

#### Q: CUDA 内存不足
A: 减小批次大小，例如：
```python
B, S, n, D = 1, 512, 4, 128
```

#### Q: Triton 导入失败
A: 确保安装了正确版本的 Triton：
```bash
conda run -n mhc_ops pip install triton>=2.0.0
```

### 性能优化建议

1. **使用 GPU**: 确保代码在 CUDA 设备上运行
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

2. **选择合适的实现**:
   - 小批次 (B×S < 512): 使用 `mhc_forward_pre_triton`
   - 大批次 (B×S > 2048): 使用 `mhc_forward_pre_triton_optimized`

3. **预热**: 在性能测试前运行多次预热
   ```python
   for _ in range(5):
       _ = model(x)
   torch.cuda.synchronize()
   ```

---

**最后更新**: 2026-02-08
**测试状态**: ✅ 所有快速测试通过
