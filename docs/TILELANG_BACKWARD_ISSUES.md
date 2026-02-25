# TileLang Backward 实现问题总结

**日期**: 2025-02-25
**状态**: ❌ **无法运行 - API 不兼容**

---

## 核心问题

TileLang 内置的 TVM 版本与标准 TVM API 不同，当前实现使用了大量标准 TVM TE API，导致无法编译。

---

## 已发现的问题

### 1. TVM TE 切片语法不兼容 ❌ (已修复)

**问题**: TVM TE 不支持 Python 切片语法 `[:]`

**错误**:
```python
dh_pre = te.compute(
    [B, S, n],
    lambda b, s, i: te.sum(
        dh_in[b, s, :].astype(compute_dtype) * x[b, s, i, :].astype(compute_dtype)
    ),
    name="dh_pre"
)
```

**错误信息**:
```
TypeError: Mismatched type on argument #1 when calling: `tir.ProducerLoad(...)`.
Expected `Array<ir.PrimExpr>` but got `Array[index 2: ffi.OpaquePyObject]`
```

**修复**: 使用 `te.reduce_axis`
```python
j = te.reduce_axis((0, D), name="j")
dh_pre = te.compute(
    [B, S, n],
    lambda b, s, i: te.sum(
        dh_in[b, s, j].astype(compute_dtype) * x[b, s, i, j].astype(compute_dtype),
        axis=j
    ),
    name="dh_pre"
)
```

**影响范围**:
- `dh_pre` (Line 59-65)
- `dvecX_mm` (Line 135-139)
- `dphi` (Line 156-162)
- `dalpha` (Line 188-190)
- `dbias` (Line 208-224)
- `dinv_rms` (Line 243-247)
- `dgamma` (Line 286-292)

---

### 2. Reduction 嵌套限制 ❌ (已修复)

**问题**: TVM TE 不允许在 `te.compute` 的 lambda 内部使用 `te.sum()` 的结果

**错误**:
```python
dalpha = te.compute([3],
    lambda k: te.if_then_else(
        k == 0,
        te.sum(dh_pre2 * h_pre1),  # ❌ Not allowed
        ...
    )
)
```

**错误信息**:
```
tvm.error.InternalError: Check failed: (0 == level_) is false:
Reductions are only allowed at the top level of compute.
Please create another tensor for further composition.
```

**修复**: 先创建中间 tensor
```python
dalpha_pre_val = te.compute(
    [],
    lambda: te.sum(dh_pre2[b2, s2, i2] * h_pre1[b2, s2, i2], axis=[b2, s2, i2]),
    name="dalpha_pre_val"
)
dalpha = te.compute([3],
    lambda k: te.if_then_else(
        k == 0,
        dalpha_pre_val(),  # ✅ Call the tensor
        ...
    )
)
```

---

### 3. 幂运算不兼容 ❌ (已修复)

**问题**: TVM TE Tensor 不支持 `**` 运算符

**错误**:
```python
inv_rms[b, s] ** 3  # ❌ TypeError
```

**错误信息**:
```
TypeError: unsupported operand type(s) for ** or pow(): 'TensorSlice' and 'int'
```

**修复**: 使用乘法
```python
inv_rms[b, s] * inv_rms[b, s] * inv_rms[b, s]  # ✅
```

---

### 4. Schedule API 缺失 ❌ (无法修复)

**问题**: TileLang 内置的 TVM 没有暴露 `te.create_schedule` API

**尝试**:
```python
s = te.create_schedule({dx.op, dphi.op, ...})  # ❌ AttributeError
```

**错误信息**:
```
AttributeError: module 'tvm.te' has no attribute 'create_schedule'
```

**检查结果**:
```python
from tilelang import tvm
from tvm import te
print([x for x in dir(te) if 'sched' in x.lower()])
# 输出: []
```

**结论**: TileLang 的 TVM 版本完全移除了 `te.create_schedule` API

---

## TileLang 正确的 API

基于 TileLang 的设计，应该使用 TileLang 的原生 API 而不是 TVM TE API：

```python
import tilelang as tl

# 使用 TileLang 的 API
@tl.jit
def mhc_backward(...):
    # 使用 TileLang 的 IR
    ...
```

或者使用 TileLang 的高级 API：
```python
from tilelang import compile

# 使用 compile 函数
func = compile(
    ...
)
```

---

## 当前实现状态

### 完全无法运行 ❌

由于 Schedule API 缺失，当前实现无法编译。需要完全重写以使用 TileLang 的原生 API。

### 代码质量评估

| 方面 | 评分 | 说明 |
|------|------|------|
| **正确性** | ❌ 0% | 无法运行，无法验证正确性 |
| **API 使用** | ⚠️ 20% | 使用了错误的 TVM TE API |
| **代码结构** | ✅ 80% | 逻辑清晰，结构良好 |
| **可维护性** | ⚠️ 40% | 需要完全重写 |

---

## 建议的修复方案

### 方案 A: 使用 TileLang 原生 API (推荐)

重写实现使用 TileLang 的 API：

```python
import tilelang as tl

def mhc_backward_tilelang(B, S, n, D, ...):
    # 定义 Tensor
    x = tl.Tensor([B, S, n, D], dtype="bfloat16", name="x")
    phi = tl.Tensor([out_features, nD], dtype="float32", name="phi")
    ...

    # 定义计算
    dh_pre = tl.sum(dh_in * x, axis=3)

    # 编译
    return tl.jit(lambda inputs: ...)
```

**优点**:
- 使用正确的 API
- 可以利用 TileLang 的优化
- 跨平台支持

**缺点**:
- 需要完全重写
- 学习曲线陡峭
- TileLang 文档不完善

### 方案 B: 暂时禁用

在测试中跳过 TileLang backward：

```python
@pytest.mark.skipif(True, reason="TileLang backward needs rewrite")
def test_backward_tilelang_vs_golden():
    ...
```

**优点**:
- 简单快速
- 不影响其他测试

**缺点**:
- 没有 TileLang backward 实现

### 方案 C: 纯 PyTorch 实现

创建一个纯 PyTorch 的 backward 实现：

```python
def mhc_backward_pytorch(x, phi, alpha, bias, ...):
    # 使用 PyTorch ops
    dh_pre = torch.einsum('bsd,bsid->bsi', dh_in, x)
    ...
```

**优点**:
- 简单易懂
- 易于维护
- 可以作为 reference

**缺点**:
- 性能可能不如 Triton/TileLang
- 需要 CUDA 才有 GPU 加速

---

## 推荐行动

### 短期 (当前)
1. ✅ **方案 B**: 暂时禁用 TileLang backward 测试
2. 专注于 Triton backward 的正确性和性能

### 中期
3. 如果需要跨平台支持，考虑 **方案 C**: 纯 PyTorch 实现
4. 或者等待 TileLang 文档完善后，使用 **方案 A** 重写

### 长期
5. 评估是否真的需要 TileLang backward 实现
6. 如果 Triton backward 足够好，可以移除 TileLang backward

---

## 修复进度

- [x] 问题 1: 切片语法 (已修复)
- [x] 问题 2: Reduction 嵌套 (已修复)
- [x] 问题 3: 幂运算 (已修复)
- [ ] 问题 4: Schedule API (❌ 无法修复，需要重写)

---

## 总结

**当前 TileLang backward 实现无法使用**。核心问题是使用了错误的 TVM TE API，而 TileLang 内置的 TVM 版本没有这些 API。

**建议**: 暂时禁用 TileLang backward，专注于 Triton backward 的优化。如果未来需要跨平台支持，可以重写使用 TileLang 原生 API 或使用纯 PyTorch 实现。

---

**最后更新**: 2025-02-25
**状态**: ❌ **需要完全重写**
