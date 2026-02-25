# TileLang 实现重写计划

**日期**: 2025-02-25
**方案**: 方案B - 使用 TileLang 原生 API 重写
**预估工作量**: 16-32 小时
**优先级**: 中（可选任务）

---

## 执行摘要

本计划详细说明如何使用 TileLang 的原生 API（而非 TVM TE）完全重写 `mhc_forward_pre_tilelang.py` 和 `mhc_backward_tilelang.py`，以解决当前实现的 API 不兼容问题。

**核心问题**: 当前实现使用了 TVM TE API，但 TileLang v0.1.8 内置的 TVM 与标准 TVM API 不兼容。

**解决方案**: 使用 TileLang 原生算子接口（`T.gemm`, `T.copy`, `T.Parallel` 等）重写所有计算逻辑。

---

## 学习总结：TileLang 核心概念

### 1. TileLang 编程模型

TileLang 采用基于 **Tile（块）** 的编程模型，核心思想是：
- 将计算分解为固定大小的 tile（如 128x128）
- 显式管理内存层级（HBM → Shared Memory → Fragment/Registers）
- 使用软件流水线重叠计算和内存传输

### 2. 关键 API 类别

#### 2.1 内存分配
```python
T.alloc_shared(shape, dtype)  # 共享内存 (~64KB per SM)
T.alloc_fragment(shape, dtype)  # 寄存器/片段内存（Tensor Core）
T.alloc_local(shape, dtype)  # 线程私有内存
```

#### 2.2 内存操作
```python
T.copy(src, dst)  # 自动优化的内存拷贝（HBM ↔ Shared ↔ Local）
T.fill(buffer, value)  # 填充 buffer
T.clear(buffer)  # 清零 buffer
```

#### 2.3 计算操作
```python
T.gemm(A, B, C, ...)  # 矩阵乘法（dispatch 到 Tensor Cores）
# Element-wise 操作在 Parallel 循环中使用 Python 运算符
for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] * B[i, j]  # 直接使用 * 运算符
```

#### 2.4 归约操作
```python
T.reduce_sum(A, result, dim=1)  # 沿维度求和
T.reduce_max(A, result, dim=1)  # 沿维度求最大值
T.warp_reduce_sum(value)  # warp 内归约
```

#### 2.5 数学函数
```python
T.exp2(x)  # 2^x（从 Flash Attention 示例）
T.rsqrt(x)  # 1/sqrt(x)（从 RMS Norm 示例）
T.max(a, b)  # 最大值
T.min(a, b)  # 最小值
T.log(x)  # 自然对数（通过 tir.call_intrin）
```

**注意**: Sigmoid 可能需要手动实现为 `1.0 / (1.0 + T.exp(-x))`。

#### 2.6 控制流
```python
T.Pipelined(n, num_stages=3)  # 软件流水线循环
T.Parallel(M, N)  # 并行循环（映射到 threads）
T.Serial(n)  # 串行循环
T.Unroll(n)  # 展开循环
```

#### 2.7 Kernel 启动
```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    # bx, by 是 block 索引
    # ...
```

### 3. 编译和执行

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])  # out_idx 指定输出参数索引
def my_kernel(M, N, ...):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float16),
        B: T.Tensor((M, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(...) as (bx, by):
            # ...

    return main

# 使用
kernel = my_kernel(1024, 1024, ...)
output = kernel(input_a, input_b)
```

---

## MHC Forward Pre 重写计划

### 当前实现分析

**文件**: `src/forward/mhc_forward_pre_tilelang.py`

**主要计算流程**:
1. Reshape x: [B, S, n, D] → [B, S, nD]
2. GEMM: h_mix = vecX @ phi.T
3. RMSNorm: inv_rms = rsqrt(mean(x^2) + eps)
4. Split h_mix into h_pre1, h_post1, h_res1
5. Apply alpha and bias
6. Sigmoid activation
7. Compute h_in = h_pre @ x

### 重写策略

#### 步骤 1: Reshape 和 GEMM（步骤 1-2）

```python
@tilelang.jit
def mhc_forward_pre_tilelang(B, S, n, D, block_BS, block_nD, block_K):
    nD = n * D
    out_features = n * n + 2 * n
    dtype = T.bfloat16
    compute_dtype = T.float32

    @T.prim_func
    def main(
        x: T.Tensor((B, S, n, D), dtype),
        phi: T.Tensor((out_features, nD), compute_dtype),
        alpha: T.Tensor((3,), compute_dtype),
        bias: T.Tensor((out_features,), compute_dtype),
        h_in: T.Tensor((B, S, D), dtype),
        h_post: T.Tensor((B, S, n), compute_dtype),
        h_res: T.Tensor((B, S, n, n), compute_dtype),
    ):
        # Grid: 每个 block 处理一个 (B, S) 位置
        with T.Kernel(B * S, threads=128) as bs_idx:
            b_idx = bs_idx // S
            s_idx = bs_idx % S

            # ============================================================
            # Step 1: Reshape x and compute GEMM: h_mix = vecX @ phi.T
            # ============================================================
            # vecX: [nD], phi: [out_features, nD], h_mix: [out_features]

            # 分配 shared memory
            vecX_shared = T.alloc_shared((nD,), dtype)
            phi_shared = T.alloc_shared((out_features, nD), compute_dtype)
            h_mix_local = T.alloc_fragment((out_features,), compute_dtype)

            # 加载 vecX
            for i in T.Serial(n):
                for j in T.Serial(D):
                    vecX_shared[i * D + j] = x[b_idx, s_idx, i, j]

            # 初始化 h_mix
            T.clear(h_mix_local)

            # 分块 GEMM: h_mix = vecX @ phi.T
            # vecX: [nD], phi: [out_features, nD]
            # 结果: h_mix: [out_features]
            for k_start in T.Pipelined(T.ceildiv(nD, block_K), num_stages=2):
                # 加载 phi 的一个 tile
                for out_idx in T.Serial(out_features):
                    for k_idx in T.Serial(block_K):
                        k_global = k_start * block_K + k_idx
                        if k_global < nD:
                            phi_shared[out_idx, k_idx] = phi[out_idx, k_global]

                # 计算外积累积
                for out_idx in T.Serial(out_features):
                    acc = T.alloc_local((1,), compute_dtype)
                    T.clear(acc)
                    for k_idx in T.Serial(block_K):
                        k_global = k_start * block_K + k_idx
                        if k_global < nD:
                            vecX_val = vecX_shared[k_global]
                            phi_val = phi_shared[out_idx, k_idx]
                            acc[0] += vecX_val.astype(compute_dtype) * phi_val
                    h_mix_local[out_idx] += acc[0]

            # ============================================================
            # Step 2-3: RMSNorm
            # ============================================================
            vecX_sq_local = T.alloc_local((nD,), compute_dtype)
            for i in T.Serial(nD):
                vecX_sq_local[i] = vecX_shared[i].astype(compute_dtype) * vecX_shared[i]

            sum_sq = T.alloc_local((1,), compute_dtype)
            T.reduce_sum(vecX_sq_local, sum_sq, dim=0)

            inv_rms = T.rsqrt(sum_sq[0] / nD + 1e-6)

            # 应用 RMSNorm 到 h_mix
            for i in T.Serial(out_features):
                h_mix_local[i] *= inv_rms

            # ============================================================
            # Step 4-6: Split, apply alpha/bias, sigmoid
            # ============================================================
            h_pre = T.alloc_fragment((n,), compute_dtype)
            h_post = T.alloc_fragment((n,), compute_dtype)
            h_res_local = T.alloc_fragment((n, n), compute_dtype)

            a_pre = alpha[0]
            a_post = alpha[1]
            a_res = alpha[2]

            # Split and apply scaling
            for i in T.Serial(n):
                idx_pre = i
                idx_post = n + i
                h_pre[i] = a_pre * h_mix_local[idx_pre] + bias[idx_pre]
                # sigmoid(h_pre) + eps
                h_pre[i] = 1.0 / (1.0 + T.exp(-h_pre[i])) + 1e-6

                h_post[i] = a_post * h_mix_local[idx_post] + bias[idx_post]
                # 2.0 * sigmoid(h_post)
                h_post[i] = 2.0 / (1.0 + T.exp(-h_post[i]))

            for i in T.Serial(n):
                for j in T.Serial(n):
                    idx_res = 2 * n + i * n + j
                    h_res_local[i, j] = a_res * h_mix_local[idx_res] + bias[idx_res]

            # ============================================================
            # Step 7: Compute h_in = h_pre @ x
            # ============================================================
            # h_pre: [n], x: [n, D], h_in: [D]
            h_in_local = T.alloc_fragment((D,), compute_dtype)

            T.clear(h_in_local)
            for i in T.Serial(n):
                for j in T.Serial(D):
                    h_in_local[j] += h_pre[i] * x[b_idx, s_idx, i, j].astype(compute_dtype)

            # ============================================================
            # Step 8: Write back to global memory
            # ============================================================
            for j in T.Serial(D):
                h_in[b_idx, s_idx, j] = h_in_local[j].astype(dtype)

            for i in T.Serial(n):
                h_post[b_idx, s_idx, i] = h_post[i]

            for i in T.Serial(n):
                for j in T.Serial(n):
                    h_res[b_idx, s_idx, i, j] = h_res_local[i, j]

    return main
```

**关键修改点**:
1. ✅ 移除所有 `te.compute`，改用 `T.Kernel` + 手动内存管理
2. ✅ 移除 `te.sum` 切片语法，使用分块 GEMM
3. ✅ 移除 `**` 运算符，使用 `T.rsqrt()`
4. ✅ 手动实现 sigmoid 为 `1.0 / (1.0 + T.exp(-x))`
5. ✅ 使用 `T.exp()`（需要确认 TileLang 是否支持，否则用 `T.exp2()` 和换底公式）

#### 步骤 2: 优化 GEMM（使用 Tensor Cores）

上面的 naive GEMM 实现性能较差。优化版本：

```python
@tilelang.jit
def mhc_forward_pre_tilelang_optimized(B, S, n, D, block_M, block_N, block_K):
    nD = n * D
    out_features = n * n + 2 * n
    dtype = T.bfloat16
    compute_dtype = T.float32

    @T.prim_func
    def main(
        x: T.Tensor((B, S, n, D), dtype),
        phi: T.Tensor((out_features, nD), compute_dtype),
        alpha: T.Tensor((3,), compute_dtype),
        bias: T.Tensor((out_features,), compute_dtype),
        h_in: T.Tensor((B, S, D), dtype),
        h_post: T.Tensor((B, S, n), compute_dtype),
        h_res: T.Tensor((B, S, n, n), compute_dtype),
    ):
        # Grid: 每个 block 处理一个 (B, S) 位置
        with T.Kernel(B * S, threads=128) as bs_idx:
            b_idx = bs_idx // S
            s_idx = bs_idx % S

            # Reshape x to vecX in shared memory
            vecX_shared = T.alloc_shared((nD,), dtype)

            for i in T.Serial(n):
                for j in T.Serial(block_N):
                    if j < D:
                        vecX_shared[i * D + j] = x[b_idx, s_idx, i, j]

            # 使用优化的 GEMM kernel（如果 nD 足够大）
            # 对于小规模的 nD，使用上面的 naive 实现即可
            # ...

    return main
```

#### 步骤 3: 实现 Wrapper 类

```python
class MHCForwardPreTileLang:
    def __init__(self, B, S, n, D, **kwargs):
        self.B = B
        self.S = S
        self.n = n
        self.D = D
        self.kernel = mhc_forward_pre_tilelang(B, S, n, D, **kwargs)

    def __call__(self, x, phi, alpha, bias, outflag=False):
        h_in = torch.empty(x.shape[0], x.shape[1], self.D, dtype=x.dtype, device=x.device)
        h_post = torch.empty(x.shape[0], x.shape[1], self.n, dtype=torch.float32, device=x.device)
        h_res = torch.empty(x.shape[0], x.shape[1], self.n, self.n, dtype=torch.float32, device=x.device)

        self.kernel(x, phi, alpha, bias, h_in, h_post, h_res)

        if outflag:
            # 计算中间值（使用 PyTorch，backward 逻辑复杂）
            B, S, n, D = x.shape
            nD = n * D
            vecX = x.reshape(B, S, nD).float()
            h_mix = torch.matmul(vecX, phi.t())
            inv_rms = torch.rsqrt(vecX.square().mean(-1, keepdim=True) + 1e-6)
            inv_rms = inv_rms.squeeze(-1)
            h_pre_val = h_post / 2.0
            h_pre_val = torch.logit(h_pre_val.clamp(1e-6, 1-1e-6)) + 1e-6
            return h_in, h_post, h_res, inv_rms, h_mix, h_pre_val
        else:
            return h_in, h_post, h_res
```

---

## MHC Backward 重写计划

### 当前实现分析

**文件**: `src/backward/mhc_backward_tilelang.py`

**主要计算流程**:
1. dh_pre = dh_in @ x^T
2. dh_pre2 = dh_pre * sigmoid_deriv
3. dalpha_pre, dbias_pre from dh_pre2
4. dvecX_inv from inv_rms, h_mix, dh_mix
5. dvecX_mm from dh_res
6. dx from dvecX_mm, dvecX_inv, dvecX_hin
7. dphi from dh_mix, x
8. dgamma from x, dvecX_mm

### 重写策略

由于 backward 计算更复杂，建议使用 **多 kernel 架构**（与 Triton 实现类似）：

#### Kernel 1: 计算主要梯度（dalpha, dbias, dvecX_inv, dvecX_hin）

```python
@tilelang.jit
def mhc_backward_kernel1(B, S, n, D, block_BS):
    nD = n * D
    out_features = n * n + 2 * n
    dtype = T.bfloat16
    compute_dtype = T.float32

    @T.prim_func
    def main(
        # Inputs
        x: T.Tensor((B, S, n, D), dtype),
        inv_rms: T.Tensor((B, S), compute_dtype),
        h_mix: T.Tensor((B, S, out_features), compute_dtype),
        h_pre: T.Tensor((B, S, n), compute_dtype),
        dh_in: T.Tensor((B, S, D), compute_dtype),
        dh_post: T.Tensor((B, S, n), compute_dtype),
        dh_res: T.Tensor((B, S, n, n), compute_dtype),
        alpha: T.Tensor((3,), compute_dtype),
        gamma: T.Tensor((nD,), compute_dtype),
        # Outputs
        dalpha: T.Tensor((3,), compute_dtype),
        dbias: T.Tensor((out_features,), compute_dtype),
        dvecX_inv: T.Tensor((B, S, nD), compute_dtype),
        dvecX_hin: T.Tensor((B, S, nD), compute_dtype),
    ):
        with T.Kernel(B * S, threads=128) as bs_idx:
            b_idx = bs_idx // S
            s_idx = bs_idx % S

            # Local accumulators for dalpha and dbias
            dalpha_local = T.alloc_fragment((3,), compute_dtype)
            dbias_local = T.alloc_fragment((out_features,), compute_dtype)
            T.clear(dalpha_local)
            T.clear(dbias_local)

            # ... 计算逻辑（参考 golden.py）

            # Atomic add to global outputs
            for i in T.Serial(3):
                T.atomic_add(dalpha[i], dalpha_local[i])

            for i in T.Serial(out_features):
                T.atomic_add(dbias[i], dbias_local[i])

            # ... 写入 dvecX_inv 和 dvecX_hin ...

    return main
```

#### Kernel 2: 计算 dx

```python
@tilelang.jit
def mhc_backward_dx_kernel(B, S, n, D, block_n, block_D):
    nD = n * D
    dtype = T.bfloat16
    compute_dtype = T.float32

    @T.prim_func
    def main(
        dvecX_mm: T.Tensor((B, S, nD), compute_dtype),
        dvecX_inv: T.Tensor((B, S, nD), compute_dtype),
        dvecX_hin: T.Tensor((B, S, nD), compute_dtype),
        gamma: T.Tensor((nD,), compute_dtype),
        dx: T.Tensor((B, S, n, D), dtype),
    ):
        with T.Kernel(B * S, n, threads=128) as (bs_idx, n_idx):
            b_idx = bs_idx // S
            s_idx = bs_idx % S

            # Reshape dvecX_mm to [n, D] and compute GEMM with gamma
            # ...

    return main
```

#### Kernel 3: 计算 dphi

```python
@tilelang.jit
def mhc_backward_dphi_kernel(B, S, n, D, block_out, block_nD):
    nD = n * D
    out_features = n * n + 2 * n

    @T.prim_func
    def main(
        dh_mix: T.Tensor((B, S, out_features), T.float32),
        x: T.Tensor((B, S, n, D), T.bfloat16),
        gamma: T.Tensor((nD,), T.float32),
        dphi: T.Tensor((out_features, nD), T.float32),
    ):
        # Grid: (out_features,)
        with T.Kernel(out_features, threads=128) as out_idx:
            # 每个 block 计算 dphi 的一行
            # ...

    return main
```

#### Kernel 4: 计算 dgamma

```python
@tilelang.jit
def mhc_backward_dgamma_kernel(B, S, n, D, block_n, block_D):
    nD = n * D

    @T.prim_func
    def main(
        x: T.Tensor((B, S, n, D), T.bfloat16),
        dvecX_mm: T.Tensor((B, S, nD), T.float32),
        dgamma: T.Tensor((nD,), T.float32),
    ):
        # Grid: (n,)
        with T.Kernel(n, threads=128) as n_idx:
            # 每个 block 计算 dgamma 的一行 [D]
            # ...

    return main
```

---

## 验证和测试计划

### 阶段 1: Forward 实现验证

1. **功能正确性测试**
   ```bash
   # 运行 test_forward.py 对比 TileLang vs Golden
   conda run -n mhc_ops python test/forward/test_forward.py
   ```

2. **预期结果**
   - max_err < 1e-3（与 Triton 类似）
   - 通过所有配置测试

3. **性能基准测试**
   ```bash
   conda run -n mhc_ops python test/forward/benchmark.py
   ```
   - 预期：与 Triton 性能相当或更优（TileLang 优化更好）

### 阶段 2: Backward 实现验证

1. **功能正确性测试**
   ```bash
   conda run -n mhc_ops python test/backward/test_backward.py
   ```

2. **预期结果**
   - 所有梯度分量误差 < 1e-3
   - 通过梯度检查（autograd 对比）

3. **性能基准测试**
   ```bash
   conda run -n mhc_ops python test/backward/benchmark.py
   ```

### 阶段 3: 集成测试

```python
# 端到端训练测试
for epoch in range(10):
    # Forward
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre_tilelang(...)

    # Compute loss
    loss = compute_loss(h_in, h_post, h_res)

    # Backward
    dx, dphi, dalpha, dbias, dgamma = mhc_backward_tilelang(...)

    # Update weights
    # ...
```

---

## 实施时间表

| 阶段 | 任务 | 预估时间 | 依赖 |
|------|------|----------|------|
| **Phase 1** | Forward Naive 实现 | 4-6h | - |
| **Phase 2** | Forward 优化（Tensor Core GEMM） | 4-8h | Phase 1 |
| **Phase 3** | Forward 测试和调试 | 2-4h | Phase 2 |
| **Phase 4** | Backward Kernel 1 实现 | 4-6h | - |
| **Phase 5** | Backward Kernel 2-4 实现 | 4-6h | Phase 4 |
| **Phase 6** | Backward 测试和调试 | 4-6h | Phase 5 |
| **Phase 7** | 性能优化和调优 | 4-8h | Phase 3, 6 |
| **总计** | | **26-44h** | |

---

## 风险和缓解措施

### 风险 1: Sigmoid 函数不可用

**概率**: 中
**影响**: 高
**缓解**:
- 使用近似实现：`sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2)`
- 或手动实现为 `1.0 / (1.0 + T.exp(-x))`
- 测试数值精度

### 风险 2: 小规模配置性能不佳

**概率**: 高
**影响**: 中
**缓解**:
- 对于 nD < 128 的配置，回退到 naive 实现
- 添加动态配置选择逻辑
- 参考 Triton 的多 kernel 架构

### 风险 3: 编译错误或运行时错误

**概率**: 中
**影响**: 高
**缓解**:
- 逐步实现，每个 kernel 独立测试
- 使用 `tilelang.testing` 框架进行单元测试
- 参考 TileLang 官方示例

### 风险 4: 性能不如 Triton

**概率**: 中
**影响**: 低（功能正确即可）
**缓解**:
- 使用 autotuner 自动调优 block sizes
- 参考 Flash Attention 的优化技巧
- 必要时回退到 Triton 实现

---

## 参考资料

### TileLang 官方文档

- **GitHub 主页**: https://github.com/tile-ai/tilelang
- **快速开始**: `examples/quickstart.py`
- **GEMM 示例**: `examples/gemm/example_gemm.py`
- **RMS Norm 示例**: `examples/norm/rms_norm.py`
- **Flash Attention**: `examples/flash_attention/example_mha_fwd_bshd_wgmma_pipelined.py`
- **Element-wise**: `examples/elementwise/example_elementwise_add.py`

### 关键 API 文档

- **内存操作**: `tilelang/language/copy_op.py`
- **GEMM 操作**: `tilelang/language/gemm_op.py`
- **归约操作**: `tilelang/language/reduce_op.py`
- **循环控制**: `tilelang/language/loop.py`
- **内存分配**: `tilelang/language/allocate.py`
- **数学函数**: `tilelang/language/math_intrinsics.py`

### 项目内部参考

- **Golden Forward**: `src/forward/golden.py`
- **Golden Backward**: `src/backward/golden.py`
- **Triton Forward**: `src/forward/mhc_forward_pre_triton.py`
- **Triton Backward**: `src/backward/mhc_backward_triton.py`
- **TileLang 状态**: `docs/TILELANG_STATUS.md`

---

## 决策点

在开始实施前，需要确认以下问题：

1. **是否需要跨平台支持？**
   - 如果只需要 CUDA: Triton 已经足够好，无需重写
   - 如果需要 CPU/AMD/国产GPU: 继续本计划

2. **性能要求？**
   - 如果要求与 Triton 持平: 需要投入更多优化时间
   - 如果功能正确即可: 可以接受稍低的性能

3. **维护成本？**
   - TileLang API 仍在快速迭代，可能需要频繁更新
   - Triton 相对稳定

4. **优先级？**
   - 是否有其他更高优先级的任务？
   - 建议完成 Triton backward 优化后再考虑 TileLang

---

## 下一步行动

如果决定继续实施本计划：

1. **准备开发环境**
   ```bash
   # 确认 TileLang 版本
   conda run -n mhc_ops python -c "import tilelang; print(tilelang.__version__)"

   # 安装最新版本（如需要）
   conda run -n mhc_ops pip install --upgrade tilelang
   ```

2. **创建开发分支**
   ```bash
   git checkout -b feature/tilelang-rewrite
   ```

3. **开始 Phase 1**: Forward Naive 实现
   - 创建新文件 `src/forward/mhc_forward_pre_tilelang_v2.py`
   - 实现基础的 tile-based kernel
   - 对比 golden 验证正确性

4. **每周同步进度**
   - 在项目中记录进度和遇到的问题
   - 及时更新本文档

---

**文档版本**: v1.0
**最后更新**: 2025-02-25
**状态**: 待审批
