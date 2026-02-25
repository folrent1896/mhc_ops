# TileLang 重写决策记录

**日期**: 2025-02-25
**决策者**: 项目团队
**状态**: ✅ 已批准执行

---

## 决策摘要

经过讨论，项目团队决定**使用 TileLang 原生 API 重写 MHC Forward/Backward 算子**，以支持跨平台 GPU 计算。

---

## 决策点确认

### ✅ 决策 1: 需要跨平台支持

**问题**: 是否真的需要跨平台支持？
**决策**: **是**

**理由**:
- Triton 仅支持 NVIDIA CUDA
- 需要支持 AMD GPU（ROCm）、国产 GPU（沐曦、昇腾等）
- TileLang 提供统一的跨平台抽象

**影响**:
- 增加 TileLang 实现的维护成本
- 但可扩展到更多硬件平台

---

### ✅ 决策 2: 功能正确性优先

**问题**: 性能要求如何？
**决策**: **功能正确性优先，性能可接受即可**

**标准**:
- Forward: max_err < 1e-2（略低于 Triton 的 1e-3）
- Backward: max_err < 1e-2
- 性能不低于 Golden 的 0.5x

**理由**:
- Triton 已经提供高性能实现
- TileLang 主要用于跨平台兼容性
- 首次重写以正确性为首要目标

**计划**:
- Phase 1: Naive 实现（功能正确）
- Phase 2: 性能优化（如有需要）

---

### ✅ 决策 3: 仅支持当前 TileLang 版本

**问题**: 是否需要考虑 API 兼容性？
**决策**: **仅支持 TileLang v0.1.8，不考虑版本兼容**

**理由**:
- TileLang API 仍在快速演进
- 维护多版本兼容成本过高
- 使用时锁定版本：`tilelang==0.1.8`

**措施**:
- 在 requirements.txt 中固定版本
- 文档中明确要求版本
- 测试脚本中检查版本

---

### ✅ 决策 4: 暂不考虑维护成本

**问题**: 维护成本可接受吗？
**决策**: **可接受，优先完成功能**

**理由**:
- 当前目标是验证跨平台可行性
- 如 TileLang 生态成熟，可考虑长期维护
- 如出现问题，可回退到 Triton

**风险缓解**:
- 保留 Triton 实现作为主要方案
- TileLang 作为可选的跨平台方案
- 充分的测试覆盖

---

## 实施优先级

### Phase 1: Forward 基础实现（立即执行）

**目标**: 功能正确的 TileLang Forward 实现

**时间**: 4-6 小时

**内容**:
1. 创建 `src/forward/mhc_forward_pre_tilelang_v2.py`
2. 使用 TileLang 原生 API 实现
3. 验证正确性

**成功标准**:
- ✅ 通过 test_forward.py 测试
- ✅ max_err < 1e-2 vs Golden
- ✅ 支持 BFloat16 输入/输出

---

### Phase 2: Backward 基础实现（Phase 1 完成后）

**目标**: 功能正确的 TileLang Backward 实现

**时间**: 6-8 小时

**内容**:
1. 创建 `src/backward/mhc_backward_tilelang_v2.py`
2. 使用 4-kernel 架构（参考 Triton）
3. 验证所有梯度分量

**成功标准**:
- ✅ 通过 test_backward.py 测试
- ✅ 所有梯度 max_err < 1e-2
- ✅ 梯度检查通过

---

### Phase 3: 性能优化（可选）

**触发条件**: 如果性能 < Golden 的 0.5x

**优化方向**:
1. 使用 Tensor Core GEMM
2. 优化内存访问模式
3. 软件流水线优化

---

## 验收标准

### 功能正确性

| 测试 | Golden | Triton | TileLang | 状态 |
|------|--------|--------|----------|------|
| Forward | ✅ | ✅ | ⏳ | 待实现 |
| Backward | ✅ | ✅ | ⏳ | 待实现 |

### 性能目标

| 实现 | Forward (vs Golden) | Backward (vs Golden) | 状态 |
|------|---------------------|----------------------|------|
| Triton | 2-4x | 0.74-0.86x | ✅ 已完成 |
| TileLang | ≥ 0.5x | ≥ 0.5x | ⏳ 待验证 |

---

## 风险管理

### 已识别风险

| 风险 | 概率 | 影响 | 缓解措施 | 状态 |
|------|------|------|----------|------|
| Sigmoid 不可用 | 中 | 中 | 手动实现为 1/(1+exp(-x)) | ⏳ |
| 编译错误 | 中 | 高 | 参考 TileLang 示例 | ⏳ |
| 性能不达标 | 高 | 低 | 功能正确即可接受 | ✅ |
| API 版本变更 | 低 | 中 | 固定 TileLang 0.1.8 | ✅ |

---

## 依赖项

### 环境要求

```bash
# 固定 TileLang 版本
pip install tilelang==0.1.8

# 或使用 conda
conda install -c tilelang tilelang=0.1.8
```

### 参考资源

- **TileLang GitHub**: https://github.com/tile-ai/tilelang
- **API 速查表**: `docs/TILELANG_API_CHEATSHEET.md`
- **重写计划**: `docs/TILELANG_REWRITE_PLAN.md`
- **知识库**: `docs/tilelang_knowledge_memory.json`

---

## 里程碑

- [x] **M0**: 决策确认（2025-02-25）
- [ ] **M1**: Forward 基础实现完成（预计 6 小时）
- [ ] **M2**: Forward 测试通过（预计 2 小时）
- [ ] **M3**: Backward 基础实现完成（预计 8 小时）
- [ ] **M4**: Backward 测试通过（预计 4 小时）
- [ ] **M5**: 文档更新（预计 2 小时）

**总计**: 预计 22 小时

---

## 成功指标

### 最小可行产品（MVP）

- [ ] TileLang Forward 实现
- [ ] TileLang Backward 实现
- [ ] 基础测试通过
- [ ] 文档完善

### 完整产品（v1.0）

- [ ] 所有测试通过
- [ ] 性能可接受（≥ 0.5x Golden）
- [ ] 支持主流配置（B, S, n, D）
- [ ] 代码可读、可维护

---

## 附录：替代方案考虑

### 方案 A: 继续使用 Triton（未采纳）

**优点**:
- 已完成且功能正确
- 性能优秀
- 社区活跃

**缺点**:
- 仅支持 NVIDIA GPU
- 无法跨平台

**结论**: 不满足跨平台需求，未采纳

### 方案 B: 使用 TileLang 原生 API 重写（已采纳）

**优点**:
- 跨平台支持
- 功能正确性可控
- 参考 Triton 经验

**缺点**:
- 需要额外开发时间
- 维护成本增加

**结论**: 满足需求，已采纳

---

**文档版本**: v1.0
**创建日期**: 2025-02-25
**状态**: ✅ 已批准执行
**最后更新**: 2025-02-25
