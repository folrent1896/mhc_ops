# TileLang 实现状态总结

**日期**: 2025-02-25
**状态**: ❌ **无法运行 - API 不兼容**

---

## 概述

MHC Ops 项目中 TileLang 的 forward 和 backward 实现都使用了 TileLang 内置的 TVM (Tensor Expression) API，但这些 API 与标准 TVM 不同，导致代码无法运行。

---

## 问题根源

### TileLang 内置的 TVM 版本限制

TileLang (v0.1.8) 内置了定制版本的 TVM，该版本与标准 TVM API 有以下差异：

1. **不支持 Python 切片语法**
   - TVM TE 不支持 `tensor[:]` 切片
   - 必须使用 `te.reduce_axis` 显式定义 reduction 轴

2. **缺少 Schedule API**
   - 没有 `te.create_schedule` 函数
   - 需要使用 TileLang 原生的编译 API

3. **Reduction 位置限制**
   - 不能在 `te.compute` lambda 内部嵌套 reduction
   - 必须先创建中间 tensor

4. **不支持幂运算**
   - Tensor 不支持 `**` 运算符
   - 需要使用乘法 `* * *`

---

## Forward 实现问题

### 文件: `src/forward/mhc_forward_pre_tilelang.py`

#### 问题 1: 导入错误 ✅ (已修复)

**错误**:
```python
from tilelang.lang import Tensor  # ❌ ModuleNotFoundError
```

**修复**:
```python
from tilelang.language import Tensor  # ✅
```

#### 问题 2: 切片语法错误 ❌ (无法运行)

**错误位置**: Line 395
```python
h_mix = te.compute(
    [B, S, out_features],
    lambda b, s, j: te.sum(vecX[b, s, :] * phi[j, :]),  # ❌ Slice syntax
    name="h_mix"
)
```

**错误信息**:
```
TypeError: Mismatched type on argument #1 when calling: `tir.ProducerLoad(...)`.
Expected `Array<ir.PrimExpr>` but got `Array[index 2: ffi.OpaquePyObject]`
```

**需要的修复**:
```python
k = te.reduce_axis((0, nD), name="k")
h_mix = te.compute(
    [B, S, out_features],
    lambda b, s, j: te.sum(vecX[b, s, k] * phi[j, k], axis=k),
    name="h_mix"
)
```

**影响范围**: 整个 forward 实现中有数十处类似的切片语法使用

---

## Backward 实现问题

### 文件: `src/backward/mhc_backward_tilelang.py`

#### 问题 1: 导入路径问题 ⚠️ (需更新)

**当前导入**:
```python
from tilelang import tvm as tvm
from tilelang.lang import Tensor  # ❌ Wrong
import tilelang.language as T
```

**应改为**:
```python
from tilelang import tvm as tvm
from tilelang.language import Tensor  # ✅ Correct
import tilelang.language as T
```

#### 问题 2: 切片语法错误 ❌ (多处)

**影响范围**:
- `dh_pre` 计算 (Line 62)
- `dvecX_mm` 计算 (Line 137)
- `dphi` 计算 (Line 159)
- `dbias` 计算 (Line 210-222)
- `dinv_rms` 计算 (Line 245)
- `dgamma` 计算 (Line 289)

#### 问题 3: Schedule API 缺失 ❌

**错误**:
```python
s = te.create_schedule({dx.op, dphi.op, ...})  # ❌ AttributeError
```

**原因**: TileLang 的 TVM 没有此 API

#### 问题 4: Reduction 嵌套错误 ❌

**错误**: 不能在 compute lambda 内使用 `te.sum()`

---

## 当前状态

### Forward

| 组件 | 状态 | 说明 |
|------|------|------|
| **导入** | ✅ 已修复 | `from tilelang.language import Tensor` |
| **运行** | ❌ 失败 | 切片语法错误，无法编译 |
| **测试** | ❌ 已禁用 | 在 `test_forward.py` 中禁用 |

### Backward

| 组件 | 状态 | 说明 |
|------|------|------|
| **导入** | ⚠️ 待修复 | 使用错误的导入路径 |
| **实现** | ❌ 不完整 | 多处切片语法和 API 错误 |
| **调度** | ❌ 失败 | Schedule API 不存在 |
| **测试** | ❌ 已禁用 | 在 `test_backward.py` 中跳过 |

---

## 修复难度评估

### 方案 A: 修复现有 TVM TE 实现

**工作量**: ⭐⭐⭐⭐⭐ (极大)

**需要修复**:
- Forward: ~30+ 处切片语法
- Backward: ~20+ 处切片语法
- Backward: 完全重写调度部分
- 两个文件: 测试所有计算逻辑

**预计时间**: 8-16 小时

**风险**: 高 (可能引入新错误)

---

### 方案 B: 使用 TileLang 原生 API 重写 ⭐ (推荐)

**工作量**: ⭐⭐⭐⭐ (大)

**优势**:
- 使用正确的 API
- 可以利用 TileLang 的优化
- 跨平台支持更好
- 代码更现代化

**劣势**:
- 需要学习 TileLang API
- 完全重写两个实现
- TileLang 文档不完善

**预计时间**: 16-32 小时

---

### 方案 C: 暂时禁用，使用其他实现 ⭐⭐⭐ (当前)

**工作量**: ⭐ (最小)

**已完成**:
- ✅ 禁用 TileLang forward 测试
- ✅ 禁用 TileLang backward 测试
- ✅ Triton forward 工作正常
- ✅ Triton backward 工作正常

**劣势**:
- 没有跨平台 TileLang 实现
- 依赖 CUDA (Triton)

---

## 推荐行动计划

### 短期 (当前)

1. ✅ **已完成**: 禁用 TileLang 测试
2. ✅ **已完成**: 专注于 Triton 实现的正确性和性能
3. ✅ **已完成**: 创建本文档记录问题

### 中期 (可选)

4. **评估需求**: 是否真的需要 TileLang 跨平台支持？
   - 如果只需要 CUDA: Triton 已经足够好
   - 如果需要 CPU/其他平台: 考虑方案 B

5. **如果需要 TileLang**:
   - 学习 TileLang 原生 API (参考官方文档和示例)
   - 使用方案 B 重写 forward 和 backward
   - 充分测试和性能对比

### 长期 (可选)

6. **社区贡献**: 将 TileLang 实现贡献给 TileLang 项目作为示例
7. **文档完善**: 编写 TileLang 实现指南供其他开发者参考

---

## 相关文档

- `docs/TILELANG_BACKWARD_ISSUES.md` - Backward 实现详细问题
- `docs/PERFORMANCE_OPTIMIZATION_PLAN.md` - Triton 性能优化计划
- `docs/CURRENT_STATUS.md` - 当前实现状态

---

## 测试命令

### Triton 测试 (推荐)

```bash
# Forward 测试
conda run -n mhc_ops python test/forward/test_forward.py --quick

# Backward 测试
conda run -n mhc_ops python test/backward/test_backward.py

# 性能基准
conda run -n mhc_ops python test/forward/benchmark.py
conda run -n mhc_ops python test/backward/benchmark.py
```

### TileLang 测试 (已禁用)

```bash
# 当前 TileLang 测试已被禁用
# 如需重新启用，需先修复实现问题
```

---

## 总结

**当前 TileLang 实现（forward 和 backward）都无法使用**。

**核心问题**: 使用了不兼容的 TVM TE API，需要大量修复或完全重写。

**推荐**: 暂时使用 Triton 实现（性能优秀，功能完整），仅在确实需要跨平台支持时考虑重写 TileLang 版本。

---

**最后更新**: 2025-02-25
**状态**: ❌ **需要完全重写**
**推荐方案**: 方案 C (暂时禁用) 或 方案 B (使用 TileLang 原生 API 重写)
