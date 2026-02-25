# TVM TIR Script 实施进度和问题记录

**日期**: 2025-02-25
**状态**: ❌ 遇到多个编译问题

---

## 尝试的方案：TVM TIR Script

### 实施思路

使用 TVM TIR Script 直接编写底层代码，绕过 TileLang eager JIT 编译器的 bug。

**理论优势**:
- 更底层的控制
- 避免 eager JIT 的转换问题
- 可以直接生成 CUDA 代码

### 实施内容

创建了 `src/forward/mhc_forward_pre_tilelang_tir.py`：
- 使用 `@T.prim_func` 装饰器
- 使用 `T.match_buffer` 定义 buffers
- 使用 `T.alloc_buffer` 分配内存
- 使用 `T.launch_thread` 定义线程
- 完整实现了所有计算步骤

---

## 遇到的问题

### 问题 1: 缺少 nvcc 编译器

**错误**:
```
RuntimeError: [Errno 2] No such file or directory: 'nvcc'
```

**原因**: TVM 需要使用 nvcc 来编译 CUDA 代码，但系统中没有安装 nvcc。

**解决方案**:
- 尝试回退到 LLVM target (CPU)
- 或者安装 CUDA toolkit

### 问题 2: LLVM Target 编译错误

**错误**:
```
tvm.error.InternalError: Check failed: (it != info_map_.end()) is false:
Load/Store of buffer vecX_shared occurred before its declaration.
```

**原因**: TIR 代码中的 buffer 访问顺序或声明有问题。

**分析**:
- 可能是循环边界问题（使用了 `range(128)` 硬编码）
- 或者 buffer 作用域问题

---

## 根本原因分析

### TVM TIR Script 的复杂性

TVM TIR Script 虽然提供了底层控制，但也带来了新的挑战：

1. **手动内存管理** - 需要手动管理所有 buffer 的生命周期
2. **target 依赖** - CUDA 和 LLVM 有不同的限制和要求
3. **编译错误难调试** - TIR 编译错误信息往往很难理解

### 与 Triton 的对比

| 方面 | Triton | TileLang Eager JIT | TVM TIR Script |
|------|--------|-------------------|----------------|
| 编程模型 | 相对简单 | 类 Python | 低级 TIR |
| 调试难度 | 中等 | 困难（编译器 bug）| 非常困难 |
| 文档质量 | 优秀 | 一般 | 较差 |
| 功能完整性 | ✅ 完整 | ❌ 有 bug | ❌ 复杂 |

---

## 当前状态总结

### 尝试过的方案

1. ✅ **TileLang Native API** - 遇到 eager JIT 编译器 bug
2. ❌ **TVM TIR Script** - 遇到编译和 buffer 访问问题

### 工作量统计

| 任务 | 预估时间 | 实际时间 | 状态 |
|------|----------|----------|------|
| 文档和知识库 | 4h | 4h | ✅ 完成 |
| TileLang Native API 实现 | 4-6h | 5h | ⚠️ 遇到 bug |
| TVM TIR Script 实现 | 4-6h | 3h | ❌ 遇到问题 |
| **总计** | 12-16h | 12h | ⏸️ 暂停 |

---

## 建议的下一步

### 方案 A: 向 TileLang 社区报告问题（推荐）

**行动**:
1. 创建 GitHub Issue: https://github.com/tile-ai/tilelang/issues
2. 包含最小可复现示例（已准备好）
3. 等待社区修复或建议

**优点**:
- 让社区知道这些问题
- 可能在未来版本中修复
- 不需要投入更多时间

**时间成本**: 低（1-2小时）

### 方案 B: 安装 CUDA Toolkit 并继续调试

**行动**:
1. 安装 CUDA Toolkit（包含 nvcc）
2. 修复 TVM TIR 代码中的问题
3. 完成实现和测试

**优点**:
- 可以继续使用 TVM TIR Script

**缺点**:
- 需要 4-8 小时额外工作
- 成功不确定

**时间成本**: 高（4-8小时）

### 方案 C: 使用 Triton 实现（已完成）

**行动**:
- 使用已有的 Triton forward 和 backward 实现
- 功能完整且经过验证

**优点**:
- 立即可用
- 功能完整
- 性能优秀

**缺点**:
- 仅支持 NVIDIA GPU
- 无法跨平台

**时间成本**: 0（已完成）

---

## 文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `docs/TILELANG_DECISIONS.md` | ✅ 完成 | 决策记录 |
| `docs/TILELANG_REWRITE_PLAN.md` | ✅ 完成 | 重写计划 |
| `docs/TILELANG_API_CHEATSHEET.md` | ✅ 完成 | API 速查表 |
| `docs/TILELANG_V2_PROGRESS.md` | ✅ 完成 | v2 实施进度 |
| `docs/TILELANG_TIR_PROGRESS.md` | ✅ 完成 | TIR 实施进度（本文件）|
| `src/forward/mhc_forward_pre_tilelang_v2.py` | ⚠️ 有 bug | Native API 实现 |
| `src/forward/mhc_forward_pre_tilelang_tir.py` | ❌ 编译失败 | TIR Script 实现 |
| `test/forward/test_tilelang_v2.py` | ⚠️ 无法运行 | v2 测试脚本 |
| `test/forward/test_tilelang_tir.py` | ❌ 无法运行 | TIR 测试脚本 |

---

## 经验总结

### 学到的知识

1. **TileLang Native API** - 理解了基本使用方法
2. **TVM TIR Script** - 了解了低级编程模型
3. **TileLang 生态系统** - 发现了一些不成熟的方面

### 遇到的限制

1. **TileLang eager JIT bug** - 无法正确编译某些合法代码
2. **TVM TIR 复杂性** - 需要深入理解底层细节
3. **文档不足** - 缺少复杂算子的示例

### 对项目的启示

1. **Triton 的价值** - Triton 相对成熟且功能完整
2. **跨平台的代价** - 需要投入大量时间和精力
3. **优先级建议** - 功能正确 > 跨平台支持

---

## 建议

基于当前的尝试和经验，我建议：

1. **短期**: 使用 Triton 实现（已完成且功能完整）
2. **中期**: 向 TileLang 社区报告问题，贡献测试用例
3. **长期**: 关注 TileLang 发展，待成熟后再考虑跨平台支持

**如果确实需要跨平台支持**，建议考虑：
- 使用 PyTorch 原生操作（CPU/GPU 通用）
- 或者等待 TileLang 社区修复这些 bug

---

**更新时间**: 2025-02-25
**状态**: ⏸️ 暂停，等待决策
**下一步**: 向用户报告进展并确认行动方案
