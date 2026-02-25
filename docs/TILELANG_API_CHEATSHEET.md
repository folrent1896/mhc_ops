# TileLang API 速查表

**用途**: MHC Ops 项目 TileLang 重写参考
**基于**: TileLang 官方文档和示例 (2025-02-25)
**源码**: https://github.com/tile-ai/tilelang

---

## 核心概念

### Tile-Based Programming Model

TileLang 采用基于 **Tile（块）** 的编程模型：
- 将计算分解为固定大小的 tile（如 128x128）
- 显式管理内存层级（HBM → Shared → Fragment）
- 使用软件流水线重叠计算和内存传输

**关键思想**: 大多数深度学习算子都可以分解为：
1. Copy（内存移动）
2. Compute（矩阵乘法/element-wise）
3. Reduce（归约）

---

## 基本结构

### Kernel 定义模板

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])  # 最后一个参数是输出
def my_kernel(M, N, K, block_M, block_N, block_K):
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Grid 配置: (grid_x, grid_y, threads)
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Kernel 逻辑
            pass

    return main

# 使用
kernel = my_kernel(1024, 1024, 1024, 128, 128, 32)
output = kernel(input_a, input_b)
```

---

## 内存分配 API

### T.alloc_shared(shape, dtype)

**用途**: 分配共享内存（~64KB per SM）
**特点**: 线程间协作，用于临时缓存

```python
# 分配 128x32 的共享内存
A_shared = T.alloc_shared((128, 32), T.float16)
```

### T.alloc_fragment(shape, dtype)

**用途**: 分配片段/寄存器内存
**特点**: 线程私有，连接 Tensor Cores

```python
# 分配累加器（用于 GEMM）
C_local = T.alloc_fragment((128, 128), T.float32)
```

### T.alloc_local(shape, dtype)

**用途**: 分配线程私有内存
**特点**: 可能溢出到全局内存

```python
# 临时变量
temp = T.alloc_local((1,), T.float32)
```

---

## 内存操作 API

### T.copy(src, dst)

**用途**: 在内存层级间复制数据
**特点**: 自动优化，支持 coalescing

```python
# HBM → Shared Memory
T.copy(A[by * block_M, k * block_K], A_shared)

# Shared → Fragment
T.copy(A_shared, A_local)

# Fragment → Shared
T.copy(C_local, C_shared)

# Shared → HBM
T.copy(C_shared, C[by * block_M, bx * block_N])

# 支持切片语法
T.copy(A[by * block_M : (by + 1) * block_M, :], A_shared)
```

### T.fill(buffer, value)

**用途**: 填充 buffer 为指定值

```python
T.fill(C_local, 0.0)  # 填充零
```

### T.clear(buffer)

**用途**: 清零 buffer（等价于 `T.fill(buffer, 0)`）

```python
T.clear(C_local)
```

---

## 计算操作 API

### T.gemm(A, B, C, ...)

**用途**: 矩阵乘法（dispatch 到 Tensor Cores）
**参数**:
- `A, B`: 输入矩阵
- `C`: 累加器
- `transpose_B`: 是否转置 B
- `policy`: GEMM 策略

```python
# C += A @ B
T.gemm(A_shared, B_shared, C_local)

# C += A @ B^T
T.gemm(A_shared, B_shared, C_local, transpose_B=True)

# 使用特定策略
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
```

### Element-wise 操作

**用途**: 在 `T.Parallel` 循环中使用 Python 运算符

```python
for i, j in T.Parallel(block_M, block_N):
    # 加法
    C_local[i, j] = A_shared[i, j] + B_shared[i, j]

    # 乘法
    C_local[i, j] = A_shared[i, j] * B_shared[i, j]

    # 比较和条件
    C_local[i, j] = T.max(A_shared[i, j], B_shared[i, j])

    # 复杂表达式
    C_local[i, j] = A_shared[i, j] * scale + bias
```

---

## 归约操作 API

### T.reduce_sum(A, result, dim=1)

**用途**: 沿指定维度求和

```python
# 沿维度 1 求和: [M, N] → [M]
A_sum = T.alloc_fragment((M,), T.float32)
T.reduce_sum(A_local, A_sum, dim=1)
```

### T.reduce_max(A, result, dim=1)

**用途**: 沿指定维度求最大值

```python
# 沿维度 1 求最大值
A_max = T.alloc_fragment((M,), T.float32)
T.reduce_max(A_local, A_max, dim=1, clear=False)
```

### Warp 级归约

```python
# Warp 内求和
reduced = T.warp_reduce_sum(local_value)

# Warp 内求最大值
reduced = T.warp_reduce_max(local_value)
```

---

## 数学函数 API

### 基本数学函数

```python
# 平方根倒数
inv_rms = T.rsqrt(x)

# 指数（以 2 为底）
result = T.exp2(x)

# 最大值/最小值
result = T.max(a, b)
result = T.min(a, b)

# 无穷大
neg_inf = -T.infinity(dtype)
```

### 手动实现常用函数

**Sigmoid**:
```python
def sigmoid(x):
    return 1.0 / (1.0 + T.exp(-x))

# 或使用 tanh 近似
def sigmoid_approx(x):
    return 0.5 + 0.5 * T.tanh(x / 2)
```

**Softmax**:
```python
# 参考 Flash Attention 实现
max_x = T.alloc_fragment((1,), T.float32)
T.reduce_max(x, max_x, dim=0, clear=False)

exp_x = T.alloc_fragment((N,), T.float32)
for i in T.Parallel(N):
    exp_x[i] = T.exp2(x[i] - max_x[0])

sum_exp = T.alloc_fragment((1,), T.float32)
T.reduce_sum(exp_x, sum_exp, dim=0)

for i in T.Parallel(N):
    exp_x[i] /= sum_exp[0]
```

---

## 控制流 API

### T.Pipelined(n, num_stages=3, ...)

**用途**: 软件流水线循环，重叠计算和内存传输
**参数**:
- `n`: 循环次数
- `num_stages`: 流水线阶段数（通常 2-4）
- `order`, `stage`, `group`: 高级控制

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    # Stage 0: 加载数据（异步）
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)

    # Stage 1: 计算（同步）
    T.gemm(A_shared, B_shared, C_local)
```

### T.Parallel(*extents)

**用途**: 并行循环（映射到 threads）

```python
# 二维并行
for i, j in T.Parallel(block_M, block_N):
    C[i, j] = A[i, j] + B[i, j]

# 一维并行
for i in T.Parallel(N):
    C[i] = A[i] * 2.0
```

### T.Serial(n) / T.Unroll(n)

**用途**: 串行/展开循环

```python
for i in T.Serial(n):
    # 顺序执行
    temp[i] = A[i] + B[i]

for i in T.Unroll(4):
    # 完全展开
    C[i] = A[i] * B[i]
```

---

## Kernel 启动 API

### T.Kernel(*grid_size, threads=128)

**用途**: 定义 kernel 启动配置
**参数**:
- `grid_size`: Grid 尺寸（可以是多个维度）
- `threads`: 每个 block 的线程数

```python
# 1D Grid
with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
    # bx: block 索引
    pass

# 2D Grid
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # bx, by: block 索引
    pass

# 3D Grid
with T.Kernel(grid_z, grid_y, grid_x, threads=256) as (bz, by, bx):
    pass
```

### 计算 Block 索引

```python
with T.Kernel(B * S, threads=128) as bs_idx:
    # 从 1D 索引分解为 2D 索引
    b_idx = bs_idx // S
    s_idx = bs_idx % S
```

---

## 高级特性

### Atomic 操作

```python
# 原子加法（用于梯度累积）
T.atomic_add(dalpha[0], local_dalpha)

# 原子最大值
T.atomic_max(output[idx], local_max)
```

### 条件执行

```python
# if-then-else
result = T.if_then_else(condition, true_value, false_value)

for i in T.Parallel(N):
    # 在循环内使用
    C[i] = T.if_then_else(i < M, A[i], 0)
```

### 动态 Shape 支持

```python
# 使用 T.dynamic() 声明动态维度
M = T.dynamic("m")
N = T.dynamic("n")

@tilelang.jit
def dynamic_kernel(M, N):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float16),
        B: T.Tensor((M, N), T.float16),
    ):
        # ...
```

---

## 常见模式

### Pattern 1: 分块 GEMM

```python
@tilelang.jit
def tiled_gemm(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

### Pattern 2: RMS Norm

```python
@tilelang.jit
def rms_norm(M, N, blk_m):
    @T.prim_func
    def main(A: T.Tensor((M, N), T.float), B: T.Tensor((M, N), T.float)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), T.float)
            A_local = T.alloc_fragment((blk_m, N), T.float)
            A_powsum = T.alloc_fragment((blk_m,), T.float)

            T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_shared)
            T.copy(A_shared, A_local)

            # 计算平方和
            A_pow_local = T.alloc_fragment((blk_m, N), T.float)
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]

            T.reduce_sum(A_pow_local, A_powsum, dim=1)

            # 归一化
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)

            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]

            T.copy(A_local, B[bx * blk_m : (bx + 1) * blk_m, :])

    return main
```

### Pattern 3: Element-wise 加法

```python
@tilelang.jit
def elementwise_add(M, N, block_M, block_N):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), T.float32)
            B_shared = T.alloc_shared((block_M, block_N), T.float32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = A_shared[i, j] + B_shared[i, j]

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

---

## 数据类型

### 支持的数据类型

```python
# 浮点类型
T.float16    # FP16
T.bfloat16   # BF16
T.float32    # FP32
T.float64    # FP64

# 整型
T.int8       # 8-bit 整型
T.int32      # 32-bit 整型
```

### 类型转换

```python
# 使用 .astype() 方法
value_fp32 = value_fp16.astype(T.float32)

# 在操作中隐式转换
result = A.astype(T.float32) * B
```

---

## 性能优化技巧

### 1. 软件流水线

```python
# 使用 num_stages 重叠内存传输和计算
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[..., k * block_K], A_shared)  # Stage 0: Load
    T.copy(B[..., k * block_K], B_shared)  # Stage 0: Load
    T.gemm(A_shared, B_shared, C_local)    # Stage 1: Compute
```

### 2. Memory Coalescing

```python
# T.copy() 自动优化，但建议：
# - 连续访问内存
# - 使用合适的 block size
# - 避免跨步访问
```

### 3. Shared Memory 复用

```python
# 对于小规模数据，可以复用 shared memory
temp_shared = T.alloc_shared((block_M, block_N), T.float32)

# 第一次使用
T.copy(A[...], temp_shared)
# ... compute ...

# 第二次使用（覆盖）
T.copy(B[...], temp_shared)
# ... compute ...
```

### 4. 减少 Bank Conflict

```python
# 使用 padding 避免 bank conflict
# 例如：32 → 33
A_shared = T.alloc_shared((block_M, block_N + 1), T.float16)
```

---

## 调试和测试

### 使用 Profiler

```python
# 创建 kernel
kernel = my_kernel(1024, 1024, ...)

# 获取 profiler
profiler = kernel.get_profiler()

# 运行基准测试
latency = profiler.do_bench(warmup=500)
print(f"Latency: {latency:.2f} ms")

# 验证正确性
profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
```

### 查看 CUDA 源码

```python
# 获取生成的 CUDA 代码
cuda_source = kernel.get_kernel_source()
print(cuda_source)
```

---

## 常见问题

### Q: 如何实现 sigmoid?

**A**: 手动实现为 `1.0 / (1.0 + T.exp(-x))` 或使用 `tanh` 近似。

### Q: 如何处理动态 shape?

**A**: 使用 `T.dynamic("name")` 声明动态维度。

### Q: 如何优化小规模配置?

**A**:
- 减小 block size
- 使用 naive 实现而非 Tensor Core
- 避免过度使用 shared memory

### Q: 调试时的常见错误?

**A**:
- Shared memory 超限（减少 block size）
- 类型不匹配（使用 `.astype()` 转换）
- 索引越界（添加 mask 检查）

---

## 参考资料

- **TileLang GitHub**: https://github.com/tile-ai/tilelang
- **Quickstart**: `examples/quickstart.py`
- **GEMM**: `examples/gemm/example_gemm.py`
- **RMS Norm**: `examples/norm/rms_norm.py`
- **Flash Attention**: `examples/flash_attention/example_mha_fwd_bshd_wgmma_pipelined.py`
- **API 文档**: `tilelang/language/__init__.py`

---

**文档版本**: v1.0
**最后更新**: 2025-02-25
**维护者**: MHC Ops 项目
