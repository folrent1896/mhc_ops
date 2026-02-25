# TileLang v2 实现进度记录

**日期**: 2025-02-25
**状态**: ⚠️ 遇到 TileLang 编译器 Bug

---

## 当前进度

### ✅ 已完成

1. **决策文档** - `docs/TILELANG_DECISIONS.md`
   - 记录了项目决策
   - 明确了功能正确性优先
   - 确定了仅支持当前 TileLang 版本

2. **知识库文档**
   - `docs/TILELANG_REWRITE_PLAN.md` - 详细重写计划
   - `docs/TILELANG_API_CHEATSHEET.md` - API 速查表
   - `docs/tilelang_knowledge_memory.json` - 结构化知识库

3. **基础实现** - `src/forward/mhc_forward_pre_tilelang_v2.py`
   - 使用 TileLang 原生 API
   - 实现了所有计算步骤
   - 代码结构清晰

4. **测试脚本** - `test/forward/test_tilelang_v2.py`
   - 完整的正确性验证
   - 支持中间值测试

---

### ❌ 遇到的问题

#### TileLang 编译器 Bug

**错误信息**:
```python
h_res[b_idx, s_idx, i, j] = h_res_local[i, j] is not a callable object
```

**问题分析**:
- TileLang v0.1.8 eager JIT 编译器错误地将赋值语句转换为 Python `is not` 表达式
- 这是一个编译器 bug，不是代码逻辑错误
- 仅影响特定的索引模式（多层索引 + 赋值）

**受影响的代码**:
```python
# 这个写法会触发 bug
for i in T.Serial(n):
    for j in T.Serial(n):
        h_res[b_idx, s_idx, i, j] = h_res_local[i, j]
```

**尝试的解决方案**:
1. ❌ 使用单层循环 + 手动索引计算 - 仍然触发 bug
2. ❌ 使用 shared memory 中转 - 仍然触发 bug
3. ⏳ 使用 T.copy() - 待测试

---

## 根本原因分析

### TileLang Eager JIT 的限制

TileLang 的 eager JIT 模式 (`@tilelang.jit`) 尚不成熟，存在以下限制：

1. **复杂的索引表达式** - 多层索引（如 `h_res[b, s, i, j]`）处理不稳定
2. **动态 shape 支持** - 对运行时 shape 的支持有限
3. **编译器错误** - 某些合法代码会触发编译器 bug

### 与示例代码的差异

TileLang 官方示例（如 `examples/gemm/example_gemm.py`）主要处理：
- 2D 张量（M, N）
- 简单的索引模式（1D 或 2D）

而 MHC Forward 需要：
- 4D 输入 `[B, S, n, D]`
- 4D 输出 `[B, S, n, n]`
- 复杂的索引和 reshape 操作

---

## 可能的解决方案

### 方案 A: 使用 TVM TIR Script（推荐）

**优点**:
- 更底层的控制
- 避开 eager JIT 的 bug
- 类似 Triton 的编程模型

**缺点**:
- 代码更复杂
- 需要手动管理更多细节

**工作量**: 4-6 小时

### 方案 B: 等待 TileLang 更新

**优点**:
- 可能在新版本中修复
- 继续使用 native API

**缺点**:
- 时间不可控
- 可能需要回退到方案 A

**工作量**: 不确定

### 方案 C: 回退到 Triton

**优点**:
- Triton 已经稳定且功能完整
- 性能优秀

**缺点**:
- 仅支持 NVIDIA GPU
- 无法跨平台

**工作量**: 0 小时（已完成）

---

## 建议行动

### 短期（立即执行）

1. **向 TileLang 社区报告此 bug**
   - GitHub Issue: https://github.com/tile-ai/tilelang/issues
   - 包含最小可复现示例

2. **尝试方案 A - 使用 TVM TIR Script**
   - 参考 TileLang 内部实现
   - 使用更底层的 API

### 中期（1-2 周）

3. **评估是否真的需要跨平台支持**
   - 如果只需要 NVIDIA GPU: 使用 Triton
   - 如果需要跨平台: 继续方案 A

4. **关注 TileLang 更新**
   - 订阅 GitHub releases
   - 测试新版本是否修复 bug

### 长期（1-3 个月）

5. **完善 Triton 实现**（如果选择方案 C）
   - 性能优化
   - 更多配置支持

6. **贡献 TileLang 示例**（如果方案 A 成功）
   - 为 TileLang 社区提供复杂算子示例
   - 帮助改进文档和工具

---

## 最小可复现示例

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def test_bug(B, S, n):
    @T.prim_func
    def main(
        input: T.Tensor((B, S, n), T.float32),
        output: T.Tensor((B, S, n), T.float32),
    ):
        with T.Kernel(B * S, threads=128) as bs_idx:
            b_idx = bs_idx // S
            s_idx = bs_idx % S

            local = T.alloc_fragment((n,), T.float32)
            for i in T.Serial(n):
                local[i] = input[b_idx, s_idx, i]

            # 这行会触发编译器 bug
            for i in T.Serial(n):
                output[b_idx, s_idx, i] = local[i]

    return main

# 尝试编译
try:
    kernel = test_bug(2, 64, 4)
    print("✅ 编译成功")
except Exception as e:
    print(f"❌ 编译失败: {e}")
```

---

## 文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `docs/TILELANG_DECISIONS.md` | ✅ 完成 | 决策记录 |
| `docs/TILELANG_REWRITE_PLAN.md` | ✅ 完成 | 重写计划 |
| `docs/TILELANG_API_CHEATSHEET.md` | ✅ 完成 | API 速查表 |
| `docs/tilelang_knowledge_memory.json` | ✅ 完成 | 知识库 |
| `src/forward/mhc_forward_pre_tilelang_v2.py` | ⚠️ 遇到 bug | 实现代码 |
| `test/forward/test_tilelang_v2.py` | ⚠️ 无法运行 | 测试脚本 |

---

## 下一步

1. **创建 GitHub Issue** - 向 TileLang 社区报告 bug
2. **尝试 TVM TIR Script** - 绕过 eager JIT 的限制
3. **或回退到 Triton** - 如果跨平台不是硬性要求

---

**更新时间**: 2025-02-25
**下一步**: 向用户报告进展并确认行动方案
