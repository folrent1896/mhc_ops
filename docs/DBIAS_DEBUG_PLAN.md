# dbias 精度问题定位与测试计划

**日期**: 2025-02-25
**问题**: dbias_res 误差过大（max error = 0.82, 13/16 元素误差 > 0.1）

---

## 问题定位结果

### 错误分布

| 部分 | 索引范围 | 最大误差 | 平均误差 | 状态 |
|------|---------|---------|---------|------|
| dbias_pre | [0:4] | 4.5e-7 | 3.2e-7 | ✅ 完美 |
| dbias_post | [4:8] | 2.4e-7 | 1.2e-7 | ✅ 完美 |
| dbias_res | [8:24] | **0.82** | **0.34** | ❌ **错误** |

**结论**: 问题完全集中在 **dbias_res** 的计算上！

---

## 根本原因分析

### Golden 实现（正确）

```python
# src/backward/golden.py line 85
dbias_res = dh_res.reshape(B, S, n * n).sum(dim=(0, 1))
```

这表示：
- dh_res: [B, S, n, n] = [2, 64, 4, 4]
- 重塑为: [B, S, n*n] = [2, 64, 16]
- 在 dim=(0, 1) 求和 → [16]

### Kernel 1 实现（错误）

```python
# src/backward/mhc_backward_triton.py lines 177-193
for i in range(0, n, BLOCK_SIZE_N):        # 外层循环：4 次
    for j in range(0, n, BLOCK_SIZE_N):    # 内层循环：4 次
        # 每次循环加载 dh_res1_chunk 并累加到 dbias
        dh_res1_chunk = tl.load(...) * a_res
        dbias_offset = 2 * n + (i + i_idx)[:, None] * n + (j + j_idx)[None, :]
        tl.atomic_add(dbias_ptr + dbias_offset, dh_res1_chunk, mask=ij_mask)
```

### 问题所在

**每个 (b, s) 程序都执行完整的嵌套循环，导致每个 dbias_res 元素被累加多次！**

以 n=4 为例：
- 每个 (b, s) 程序执行 4×4 = 16 次循环
- 每次循环都会加载和累加 dh_res1 的一部分
- 结果：每个 dbias_res 元素被累加了 16 次！

**正确的行为应该是**：
- 每个 (b, s, i, j) 计算 dh_res1[b, s, i, j]
- 累加到 dbias_res[i, j]
- 最终：每个 dbias_res 元素被累加 B×S = 128 次（从不同的 b, s）

### 当前错误行为

由于嵌套循环，每个 dbias_res 元素被累加了：
- (B × S) × (n × n) = 128 × 16 = 2048 次！

**实际值 = 期望值 × 16**（因为每个 (b,s) 程序重复计算了 n×n 次）

---

## 修复方案

### 方案 1: 使用已加载的 dh_res_block（推荐）

在 kernel 1 开头已经加载了 dh_res_block（第 117-123 行），应该直接使用它：

```python
# 在第 167 行之后添加
# dbias_res: accumulate dh_res1 over B, S
# dh_res1 = a_res * dh_res (already loaded)
dbias_res_offset = 2 * n + x_off_n[:, None] * n + x_off_n[None, :]
dh_res1_to_accum = dh_res_block * a_res
tl.atomic_add(dbias_ptr + dbias_res_offset, dh_res1_to_accum, mask=dh_res_mask)
```

**优点**：
- 无需重复加载 dh_res
- 代码简洁
- 与 dbias_pre, dbias_post 模式一致

**缺点**：
- 无

### 方案 2: 移除嵌套循环，直接加载所需部分

```python
# 加载当前 (b, s) 的 dh_res 块
dh_res1_chunk = tl.load(...) * a_res

# 只累加一次（移除嵌套循环）
dbias_res_offset = 2 * n + x_off_n[:, None] * n + x_off_n[None, :]
tl.atomic_add(dbias_ptr + dbias_res_offset, dh_res1_chunk, mask=dh_res_mask)
```

**优点**：
- 逻辑清晰
- 避免重复累加

**缺点**：
- 需要重新设计代码结构

---

## 测试计划

### 阶段 1: 创建隔离测试（10分钟）

创建 `test/test_dbias_res_isolated.py`：

```python
# 目的：隔离测试 dbias_res 计算
# 方法：创建简化的 kernel，只计算 dbias_res
# 验证：比较 kernel 输出与 golden

@triton.jit
def test_dbias_res_kernel(
    dh_res_ptr, alpha_ptr, dbias_ptr,
    B, S, n,
    stride_hres_b, stride_hres_s, stride_hres_n1, stride_hres_n2,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    x_off_n = tl.arange(0, BLOCK_SIZE_N)
    dh_res_mask = (x_off_n[:, None] < n) & (x_off_n[None, :] < n)

    # Load dh_res
    dh_res_offset = (b_idx * stride_hres_b + s_idx * stride_hres_s +
                     x_off_n[:, None] * stride_hres_n1 +
                     x_off_n[None, :] * stride_hres_n2)
    dh_res = tl.load(dh_res_ptr + dh_res_offset, mask=dh_res_mask, other=0.0)

    # Load alpha[2]
    a_res = tl.load(alpha_ptr + 2)

    # Compute dh_res1
    dh_res1 = dh_res * a_res

    # Accumulate to dbias (ONCE per (b,s,i,j))
    dbias_offset = 2 * n + x_off_n[:, None] * n + x_off_n[None, :]
    tl.atomic_add(dbias_ptr + dbias_offset, dh_res1, mask=dh_res_mask)
```

### 阶段 2: 验证修复（5分钟）

运行隔离测试，确认：
- dbias_res 最大误差 < 1e-4
- 所有元素误差在可接受范围内

### 阶段 3: 集成测试（5分钟）

运行完整 backward 测试：
```bash
conda run -n mhc_ops python test/backward/test_backward.py
```

验证：
- dbias 总误差 < 1e-3
- dbias_pre, dbias_post 仍然正确
- 没有引入新的错误

### 阶段 4: 多配置测试（10分钟）

测试不同配置确保泛化性：
```python
configs = [
    (2, 64, 4, 128),   # 基准配置
    (1, 256, 4, 256),  # 大 S，大 D
    (4, 512, 8, 128),  # 大 n
]
```

---

## 实施步骤

### 步骤 1: 备份当前代码（1分钟）

```bash
git add -A
git commit -m "backup: before dbias_res fix"
```

### 步骤 2: 修改 kernel 1（5分钟）

在 `src/backward/mhc_backward_triton.py` 中：

**删除第 177-193 行的嵌套循环代码**

**在第 175 行后添加**：
```python
# dbias_res: accumulate dh_res1 over B, S
# Use the already-loaded dh_res_block (line 117-123)
dbias_res_offset = 2 * n + x_off_n[:, None] * n + x_off_n[None, :]
dh_res1_to_accum = dh_res_block * a_res
tl.atomic_add(dbias_ptr + dbias_res_offset, dh_res1_to_accum, mask=dh_res_mask)
```

### 步骤 3: 创建验证脚本（5分钟）

创建 `test/verify_dbias_res.py`：
```python
# 验证 dbias_res 计算是否正确
# 对比 golden 和 triton 输出
```

### 步骤 4: 运行测试（2分钟）

```bash
conda run -n mhc_ops python test/verify_dbias_res.py
conda run -n mhc_ops python test/backward/test_backward.py
```

### 步骤 5: 提交修复（2分钟）

```bash
git add src/backward/mhc_backward_triton.py
git commit -m "fix(backward): Fix dbias_res accumulation bug"
```

---

## 预期结果

修复后应达到：
- dbias_res max error < 1e-4
- dbias 总误差 < 1e-3
- 所有 dbias 组件通过测试

---

## 风险与备选方案

### 风险 1: dh_res_block 尚未加载

**缓解措施**: 确认 dh_res_block 在第 117-123 行已加载

### 风险 2: 索引计算错误

**缓解措施**: 使用隔离测试验证索引正确性

### 风险 3: 性能影响

**评估**: 移除嵌套循环应**提升**性能，而不是降低

---

## 调试命令汇总

```bash
# 分析 dbias 误差分布
conda run -n mhc_ops python test/analyze_dbias.py

# 隔离测试 dbias_res
conda run -n mhc_ops python test/test_dbias_res_isolated.py

# 完整 backward 测试
conda run -n mhc_ops python test/backward/test_backward.py

# 快速验证
conda run -n mhc_ops python test/verify_dbias_res.py
```

---

## 时间估算

- 问题分析：15分钟 ✅ 已完成
- 创建测试计划：15分钟 ✅ 正在进行
- 实施修复：10分钟
- 验证测试：10分钟
- 文档更新：5分钟

**总计**: 约 55 分钟

---

## 关键经验教训

1. **嵌套循环要谨慎**
   - 确保每个元素只被处理一次
   - 检查是否有重复累加

2. **利用已加载的数据**
   - dh_res_block 已加载，应直接使用
   - 避免重复内存访问

3. **分段验证很重要**
   - dbias_pre, dbias_post 正确
   - 只有 dbias_res 错误
   - 快速定位问题所在

---

**状态**: 问题已定位，等待修复实施
**预计修复时间**: 30分钟
**难度**: 低（逻辑清晰，修改简单）
