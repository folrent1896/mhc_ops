# 项目结构说明

## 目录组织

```
mhc-ops/
│
├── src/                               # 源代码目录
│   ├── __init__.py                    # 包初始化文件
│   ├── mhc_forward_pre_triton.py      # Triton GPU kernel 实现
│   │   ├── mhc_forward_pre_kernel()   #   单 kernel 版本
│   │   ├── mhc_forward_pre_triton()   #   单 kernel 包装函数
│   │   ├── gemm_kernel()              #   GEMM kernel (优化版)
│   │   ├── rmsnorm_kernel()           #   RMSNorm kernel (优化版)
│   │   └── mhc_forward_pre_triton_optimized()  # 优化版包装函数
│   │
│   └── mhc_forward_pre_tilelang.py    # TileLang DSL 实现
│       ├── mhc_forward_pre_tilelang() #   TileLang DSL 函数
│       ├── MHCForwardPreTileLang      #   封装类
│       └── mhc_forward_pre_tvm()      #   TVM TE 实现
│
├── test/                              # 测试代码目录
│   ├── __init__.py                    # 包初始化文件
│   ├── test_mhc_pre_grad.py           # PyTorch 参考实现
│   │   ├── mhc_forward_pre()          #   正向传播
│   │   └── mhc_pre_backward_manual()  #   手动反向传播
│   │
│   ├── test_implementations.py        # 完整测试套件
│   │   ├── TestResult                 #   测试结果数据类
│   │   ├── MHCOperatorTester          #   测试器类
│   │   └── main()                     #   主函数
│   │
│   ├── quick_test.py                  # 快速验证脚本
│   │   ├── quick_test_reference_vs_triton()  #   基本对比测试
│   │   ├── quick_test_multiple_configs()     #   多配置测试
│   │   ├── test_functional_correctness()     #   功能正确性测试
│   │   └── main()                     #   主函数
│   │
│   └── benchmark.py                   # 性能基准测试
│       ├── BenchmarkResult            #   基准结果数据类
│       ├── Benchmark                  #   基准测试类
│       ├── pytorch_reference()        #   内嵌参考实现
│       └── run_benchmark_suite()      #   运行基准测试
│
├── README.md                          # 中文文档
├── README_IMPLEMENTATIONS.md          # 英文文档
├── PROJECT_STRUCTURE.md               # 本文件 - 项目结构说明
├── requirements.txt                   # 依赖列表
├── setup.py                           # 安装配置
├── .gitignore                         # Git 忽略文件
└── run_tests.sh                       # 快速测试脚本
```

## 模块说明

### 源代码模块 (`src/`)

#### `mhc_forward_pre_triton.py`
提供两个 Triton 实现版本：

**单 Kernel 版本** - `mhc_forward_pre_kernel`:
- 将所有计算融合到一个 kernel 中
- 适合小批次场景
- 启动开销小

**优化版本** - `mhc_forward_pre_triton_optimized`:
- 分离 GEMM 和 RMSNorm kernels
- 更好的并行化
- 适合大批次场景

#### `mhc_forward_pre_tilelang.py`
提供三种 TileLang 实现方式：

1. **DSL 版本** - `mhc_forward_pre_tilelang()`:
   - 使用高级 DSL 语法
   - 自动优化

2. **封装类** - `MHCForwardPreTileLang`:
   - 面向对象接口
   - 易于集成

3. **TVM TE 版本** - `mhc_forward_pre_tvm()`:
   - 低级 API 控制
   - 手动调度优化

### 测试模块 (`test/`)

#### `test_mhc_pre_grad.py`
PyTorch 参考实现，用于验证其他实现的正确性。

**函数:**
- `mhc_forward_pre()` - 正向传播
- `mhc_pre_backward_manual()` - 手动反向传播（梯度验证）

#### `test_implementations.py`
完整的测试和基准测试框架。

**类:**
- `TestResult` - 存储测试结果
- `MHCOperatorTester` - 测试器主类

**功能:**
- 多配置自动测试
- 精度验证（最大/平均误差）
- 性能测量（延迟、吞吐量）
- CSV 结果导出

#### `quick_test.py`
快速验证脚本，包含三个测试函数：

1. `quick_test_reference_vs_triton()` - 基本对比
2. `quick_test_multiple_configs()` - 多配置扫描
3. `test_functional_correctness()` - 详细值检查

#### `benchmark.py`
独立的性能基准测试工具。

**类:**
- `BenchmarkResult` - 基准结果
- `Benchmark` - 基准测试器

**特点:**
- 内嵌参考实现（无外部依赖）
- 统计数据（均值、标准差、最小/最大值）
- 自动 GPU 内存管理

## 导入示例

### 使用源代码

```python
# 导入 Triton 实现
from src.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

# 导入 TileLang 实现
from src.mhc_forward_pre_tilelang import MHCForwardPreTileLang
```

### 使用测试工具

```python
# 导入参考实现
from test.test_mhc_pre_grad import mhc_forward_pre

# 运行测试
from test import quick_test
quick_test.main()
```

## 开发工作流

1. **修改实现**: 编辑 `src/` 中的文件
2. **运行快速测试**: `python test/quick_test.py`
3. **性能测试**: `python test/benchmark.py`
4. **完整验证**: `python test/test_implementations.py --quick`

## 添加新实现

在 `src/` 中创建新文件，例如 `mhc_forward_pre_custom.py`:

```python
# src/mhc_forward_pre_custom.py

def mhc_forward_pre_custom(x, phi, alpha, bias, **kwargs):
    """你的自定义实现"""
    # ... 实现 ...
    return h_in, h_post, h_res
```

然后在 `src/__init__.py` 中导出:

```python
from .mhc_forward_pre_custom import mhc_forward_pre_custom

__all__ = [
    "mhc_forward_pre_triton",
    "mhc_forward_pre_triton_optimized",
    "mhc_forward_pre_custom",  # 添加新实现
]
```
