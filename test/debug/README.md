# Debug Scripts 目录

本目录包含在 MHC Backward Triton 开发过程中创建的调试、分析和验证脚本。

---

## 目录结构

### 📊 analysis/ - 误差分析脚本

用于分析误差分布、模式和相关性的脚本。

**脚本列表**:
- `analyze_dbias.py` - 分析 dbias 误差，按 pre/post/res 部分分解
- `analyze_dgamma.py` - 分析 dgamma 整体误差分布
- `analyze_dgamma_ratio.py` - 分析 dgamma 的 Triton/Golden ratio

**用途**:
- 快速了解误差规模和分布
- 识别系统性误差模式
- 验证修复效果

---

### 🔍 diagnostic/ - 诊断脚本

用于定位问题的根本原因的诊断脚本。

**脚本列表**:
- `decompose_dx.py` - 分解 dx 的三个组成部分，定位误差来源
- `debug_all_parts.py` - 分析 dvecX_mm 所有部分的详细计算
- `debug_dalpha.py` - dalpha 组件的详细调试
- `debug_dalpha_detailed.py` - dalpha 更详细的调试信息
- `debug_dh_res1_loop.py` - 验证 dh_res1 嵌套循环的计算逻辑
- `debug_discrepancy.py` - 检查计算逻辑的一致性
- `debug_dx.py` - dx 误差分析和模式识别
- `debug_full_backward.py` - 完整 backward 流程调试
- `debug_simple_backward.py` - 简化版 backward 调试
- `diagnose_dvecX_mm_loop.py` - 诊断 nD_start 循环执行情况
- `isolate_dh_res1_effect.py` - 隔离 dh_res1 对 dgamma 的影响
- `simple_dgamma_check.py` - 按 block 检查 dgamma 误差

**用途**:
- 定位问题根源
- 验证假设
- 隔离特定组件或计算

---

### ✅ verify/ - 验证脚本

用于验证特定组件正确性的独立测试脚本。

**脚本列表**:
- `test_dalpha_isolated.py` - 隔离测试 dalpha 组件
- `test_kernel4_isolated.py` - 隔离测试 kernel 4（dgamma）
- `verify_dhpre2_kernel.py` - 验证 dh_pre2 在 kernel 中的计算
- `verify_dvecX_mm.py` - 验证 dvecX_mm 的计算结果
- `verify_dvecX_mm_computation.py` - 验证 dvecX_mm 计算逻辑
- `verify_hmix_kernel.py` - 验证 h_mix 在 kernel 中的计算

**用途**:
- 独立验证单个组件
- 确认修复前后的状态
- 避免其他组件的干扰

---

## 使用指南

### 快速诊断流程

当遇到问题时，可以按以下顺序使用这些脚本：

#### 1. 分析误差（analysis/）

```bash
# 快速了解误差规模
conda run -n mhc_ops python test/debug/analysis/analyze_dgamma.py
conda run -n mhc_ops python test/debug/analysis/analyze_dbias.py
```

#### 2. 隔离组件（verify/）

```bash
# 验证单个组件是否正确
conda run -n mhc_ops python test/debug/verify/test_kernel4_isolated.py
conda run -n mhc_ops python test/debug/verify/test_dalpha_isolated.py
```

#### 3. 诊断问题（diagnostic/）

```bash
# 深入诊断问题根源
conda run -n mhc_ops python test/debug/diagnostic/decompose_dx.py
conda run -n mhc_ops python test/debug/diagnostic/diagnose_dvecX_mm_loop.py
```

---

## 按问题类型查找脚本

### dbias 问题
- **分析**: `analysis/analyze_dbias.py`
- **诊断**: `diagnostic/debug_full_backward.py`（查看 dbias 部分）
- **验证**: `verify/verify_hmix_kernel.py`（验证 h_mix 相关）

### dalpha 问题
- **诊断**: `diagnostic/debug_dalpha.py`
- **诊断**: `diagnostic/debug_dalpha_detailed.py`
- **验证**: `verify/test_dalpha_isolated.py`
- **验证**: `verify/verify_dhpre2_kernel.py`

### dgamma 问题
- **分析**: `analysis/analyze_dgamma.py`
- **分析**: `analysis/analyze_dgamma_ratio.py`
- **诊断**: `diagnostic/simple_dgamma_check.py`
- **诊断**: `diagnostic/isolate_dh_res1_effect.py`
- **验证**: `verify/test_kernel4_isolated.py`

### dx 问题
- **分析**: `diagnostic/debug_dx.py`
- **诊断**: `diagnostic/decompose_dx.py`
- **验证**: `verify/verify_dvecX_mm.py`

### dvecX_mm 问题
- **验证**: `verify/verify_dvecX_mm_computation.py`
- **诊断**: `diagnostic/diagnose_dvecX_mm_loop.py`
- **诊断**: `diagnostic/debug_dh_res1_loop.py`

---

## 脚本历史

这些脚本是在 MHC Backward Triton 开发过程中创建的，记录了以下 bug 的修复过程：

1. **Bug #1: dalpha 精度问题** (Session 4)
   - `verify/verify_dhpre2_kernel.py`
   - `diagnostic/debug_dalpha.py`

2. **Bug #2: dbias 精度问题** (Session 5)
   - `analysis/analyze_dbias.py`
   - `verify/verify_hmix_kernel.py`

3. **Bug #3: dgamma 精度问题** (Session 5)
   - `analysis/analyze_dgamma.py`
   - `analysis/analyze_dgamma_ratio.py`
   - `verify/test_kernel4_isolated.py`
   - `diagnostic/simple_dgamma_check.py`
   - `diagnostic/isolate_dh_res1_effect.py`
   - `diagnostic/debug_dh_res1_loop.py`
   - `diagnostic/debug_all_parts.py`

4. **Bug #4: dx 精度问题** (Session 5)
   - `diagnostic/decompose_dx.py`
   - `diagnostic/debug_dx.py`

---

## 注意事项

1. **运行环境**: 所有脚本都需要在 `mhc_ops` conda 环境中运行
   ```bash
   conda activate mhc_ops
   # 或
   conda run -n mhc_ops python test/debug/...
   ```

2. **依赖**: 这些脚本依赖项目的主要代码
   - `src/forward/golden.py`
   - `src/backward/golden.py`
   - `src/backward/mhc_backward_triton.py`

3. **数据类型**: 注意 CPU/CUDA tensor 的转换
   - Golden 实现在 CPU 上
   - Triton 实现在 CUDA 上

4. **随机种子**: 大多数脚本使用固定随机种子以确保结果可重复
   - `torch.manual_seed(42)` 用于 x, phi, alpha, bias
   - `torch.manual_seed(123)` 用于 dh_*

---

## 维护指南

### 添加新脚本

当创建新的调试脚本时：

1. **根据目的选择目录**:
   - 分析误差 → `analysis/`
   - 诊断问题 → `diagnostic/`
   - 验证组件 → `verify/`

2. **命名规范**:
   - 分析脚本: `analyze_<component>.py`
   - 诊断脚本: `debug_<component>.py` 或 `diagnose_<issue>.py`
   - 验证脚本: `verify_<component>.py` 或 `test_<component>_isolated.py`

3. **文档注释**:
   - 在脚本开头添加目的说明
   - 包含预期输出和解释
   - 注明相关的问题或 bug

4. **测试脚本**:
   - 确保脚本可以独立运行
   - 包含必要的 import 和参数设置
   - 提供清晰的输出

---

## 相关文档

- `docs/BUGFIX_LOG.md` - Bug 修复记录
- `docs/CURRENT_STATUS.md` - 当前实现状态
- `docs/DX_DEBUG_PLAN.md` - dx 调试计划
- `docs/DGAMMA_FIX_PLAN.md` - dgamma 修复计划
- `docs/DBIAS_FIX_SUMMARY.md` - dbias 修复总结

---

**最后更新**: 2025-02-25
**用途**: MHC Backward Triton 开发和调试
**维护**: 如有问题请参考相关文档或提交 issue
