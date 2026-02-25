# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MHC Ops is a deep learning operator library providing high-performance implementations of the Manifold-constrained Hyper-connection (MHC) Forward Pre operator. The project implements both forward and backward passes with multiple backends:

- **Golden**: Pure PyTorch reference implementations for correctness verification
- **Triton**: High-performance GPU kernels (NVIDIA CUDA)
- **TileLang**: Cross-platform DSL implementations (via TVM)

**Status**: Forward implementations are fully functional. Backward Triton implementation is temporarily disabled due to Triton language limitations.

---

## Conda Environment

### Environment Information
- **Name**: `mhc_ops`
- **Path**: `/home/huan1178/miniconda3/envs/mhc_ops`
- **Python**: 3.10.19
- **CUDA**: 12.4
- **PyTorch**: 2.6.0+cu124
- **Triton**: 3.2.0

### Running Commands

**Always use `conda run` to execute commands in the project environment:**
```bash
# Quick test
conda run -n mhc_ops python test/forward/quick_test.py

# Benchmark
conda run -n mhc_ops python test/forward/benchmark.py

# Full test suite
./run_tests.sh
```

**If using `conda activate`, run `conda init` first and restart terminal.**

---

## Project Architecture

### Directory Structure
```
mhc_ops/
├── src/
│   ├── forward/          # Forward pass implementations
│   │   ├── golden.py              # PyTorch reference
│   │   ├── mhc_forward_pre_triton.py       # Triton GPU kernels
│   │   └── mhc_forward_pre_tilelang.py     # TileLang DSL
│   └── backward/         # Backward pass implementations
│       ├── golden.py              # PyTorch reference
│       ├── mhc_backward_triton.py          # Triton (⚠️ incomplete)
│       └── mhc_backward_tilelang.py        # TileLang
├── test/
│   ├── forward/          # Forward tests
│   └── backward/         # Backward tests
└── run_tests.sh         # Test runner
```

### Core Data Flow

**Forward Pass:**
```
Input x [B,S,n,D] → GEMM → RMSNorm → Sigmoid → Alpha Scaling
                                    ↓
Output: h_in [B,S,D], h_post [B,S,n], h_res [B,S,n,n]
```

**Backward Pass:**
```
Gradients + Forward Intermediates → Gradient Computation → dx, dphi, dalpha, dbias, dgamma
```

### Key Design Patterns

1. **Unified Interface**: All implementations follow the same signature
2. **Intermediate Values**: Set `outflag=True` to get values needed for backward
3. **Optional Dependencies**: TileLang imports wrapped in try/except
4. **Performance-based Selection**: Different kernels for different batch sizes

### Data Type Conventions
- Input `x`: **BFloat16** (memory efficiency)
- Weights `phi`, `alpha`, `bias`: **Float32** (numerical precision)
- Outputs: **Float32** (numerical stability)
- Intermediate computations: **Float32**

---

## Common Issues and Solutions

### Import Path Issues

**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Install in development mode
```bash
conda run -n mhc_ops pip install -e .
```

### Triton Language Limitations

Triton has strict compile-time requirements. **Always follow these rules:**

#### 1. `tl.arange` Must Use Compile-Time Constants
❌ **WRONG**:
```python
indices = tl.arange(0, n * n)  # n * n is runtime value
```

✅ **CORRECT**:
```python
indices = tl.arange(0, BLOCK_SIZE_K)  # BLOCK_SIZE_K is tl.constexpr
mask = indices < n * n  # Use mask for runtime bound
data = tl.load(ptr + indices, mask=mask)
```

#### 2. No `.flatten()` Method
❌ **WRONG**:
```python
flat = tensor.flatten()
```

✅ **CORRECT**:
```python
flat = tensor.reshape(fixed_size)  # fixed_size must be constexpr
# OR avoid reshape entirely
flat = tensor  # Use 2D indexing instead
```

#### 3. `reshape` Parameters Must Be Compile-Time Constants
❌ **WRONG**:
```python
reshaped = tensor.reshape(nD)  # nD is runtime value
```

✅ **CORRECT**:
```python
# Option 1: Use fixed BLOCK_SIZE
reshaped = tensor.reshape(BLOCK_SIZE_K)
# Option 2: Avoid reshape, use 2D operations
result_2d = keep_as_2d(tensor)
```

#### 4. No Slice Operations
❌ **WRONG**:
```python
subset = tensor[i:j]
row = matrix[i, :]
```

✅ **CORRECT**:
```python
# Use 2D indexing with masks
indices = tl.arange(0, BLOCK_SIZE)
mask = indices < size
subset = tl.load(ptr + indices, mask=mask)

# For matrices, use full 2D indexing
row_mask = (row_indices == i) & col_mask
row_data = tl.load(ptr + offset, mask=row_mask)
```

#### 5. `tl.zeros`/`tl.ones` Shape Must Be Compile-Time Constants
❌ **WRONG**:
```python
result = tl.zeros([n, D], dtype=tl.float32)
```

✅ **CORRECT**:
```python
result = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)
# Apply mask to get actual [n, D] subset
actual = result * mask  # or use masked operations
```

#### 6. No Scalar Indexing on Tensors
❌ **WRONG**:
```python
alpha = tl.load(alpha_ptr)
a_pre = alpha[0]  # Can't index scalar tensor
```

✅ **CORRECT**:
```python
a_pre = tl.load(alpha_ptr + 0)
a_post = tl.load(alpha_ptr + 1)
```

#### 7. No Loop Index Assignment
❌ **WRONG**:
```python
for i in range(n):
    result[i] = value
```

✅ **CORRECT**:
```python
# Vectorized with mask
indices = tl.arange(0, BLOCK_SIZE_N)
mask = indices < n
tl.atomic_add(result_ptr + indices, values, mask=mask)
```

### Test Script Issues

**Problem**: Test files moved to `test/forward/` and `test/backward/`
**Solution**: Update imports in test files
```python
# OLD (wrong):
from mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

# NEW (correct):
from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized
```

---

## Testing Guidelines

### Test Structure
- **Forward**: `test/forward/quick_test.py`, `test/forward/benchmark.py`
- **Backward**: `test/backward/test_backward.py`

### Running Tests
```bash
# Quick correctness validation
conda run -n mhc_ops python test/forward/quick_test.py

# Performance benchmarking
conda run -n mhc_ops python test/forward/benchmark.py

# All tests
./run_tests.sh
```

### Test Configurations
Standard test configurations:
```python
(B, S, n, D) = [(2, 64, 4, 128), (2, 256, 4, 256), (4, 512, 4, 512)]
```

### Tolerance Settings
- Relative tolerance (rtol): 1e-3 to 1e-4 for GPU implementations
- Small numerical differences are expected due to different reduction orders

---

## Development Workflow

1. **Make changes** to implementation in `src/`
2. **Reinstall** if needed: `conda run -n mhc_ops pip install -e .`
3. **Run quick tests**: `conda run -n mhc_ops python test/forward/quick_test.py`
4. **Run benchmarks** for performance validation

---

## Known Limitations

### Backward Triton Implementation
**Status**: Temporarily disabled due to Triton language limitations.

**Issues**:
- Cannot use runtime values with `tl.arange`, `reshape`, `zeros`
- No support for tensor slicing or scalar indexing
- Complex gradient computations require redesign

**Workaround**: Use golden backward implementation for now.

---

## File Organization

### Test Files
```
test/
├── forward/
│   ├── quick_test.py      # Quick validation
│   ├── benchmark.py       # Performance tests
│   └── test_forward.py    # Comprehensive tests
└── backward/
    └── test_backward.py   # Backward tests (Triton skipped)
```

### Run Script
- `run_tests.sh`: Runs all forward and backward tests

---

## Performance Characteristics

- **Triton forward**: 2-5x speedup over golden reference
- **Small batches** (B×S < 512): Use single kernel version
- **Large batches** (B×S > 2048): Use optimized version with separated kernels

---

## Troubleshooting

### Problem: CUDA Out of Memory
**Solution**: Reduce batch size or D dimension
```python
B, S, n, D = 1, 512, 4, 128  # Smaller config
```

### Problem: Triton Import Errors
**Solution**: Verify GPU drivers and CUDA installation
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Problem: Test Failures with Large Errors
**Solution**:
- Verify correct implementation is being used
- Check device placement (CPU vs GPU)
- Ensure all tensors on same device

---

## Code Style Guidelines

1. **Match the signature**: Ensure function signatures match reference implementation
2. **Support outflag**: Return intermediate values when `outflag=True`
3. **Handle device placement**: Ensure all tensors on same device
4. **Test first**: Always run `quick_test.py` after changes
5. **Benchmark after correctness**: Use `benchmark.py` to measure performance impact

---

## Adding New Implementations

1. Create new file in `src/forward/` or `src/backward/`
2. Follow existing function signatures
3. Export from `__init__.py`
4. Add tests in corresponding test directory
5. Run tests to verify correctness

---

**Last Updated**: 2025-02-24
**Test Status**: ✅ Forward tests passing | ⚠️ Backward Triton disabled (use golden)
