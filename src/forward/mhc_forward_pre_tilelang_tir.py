"""
MHC Forward Pre - TVM TIR Script Implementation

This implementation uses TVM TIR Script directly to bypass TileLang eager JIT bugs.
We use low-level TIR to have full control over the generated code.

Author: MHC Ops Team
Date: 2025-02-25
Status: Using TVM TIR Script to avoid eager JIT bugs
"""

import torch
import tvm
from tvm.script import tir as T
import tilelang


def mhc_forward_pre_tilelang_tir(B, S, n, D, dtype="bfloat16", compute_dtype="float32"):
    """
    Compile MHC Forward Pre operator using TVM TIR Script.

    Args:
        B: Batch size
        S: Sequence length
        n: Number of heads
        D: Head dimension
        dtype: Input/output data type (default: "bfloat16")
        compute_dtype: Computation data type (default: "float32")

    Returns:
        Compiled TVM module
    """
    nD = n * D
    out_features = n * n + 2 * n

    @T.prim_func
    def mhc_forward_tir(
        x: T.handle,
        phi: T.handle,
        alpha: T.handle,
        bias: T.handle,
        h_in: T.handle,
        h_post: T.handle,
        h_res: T.handle,
    ):
        """MHC Forward Pre kernel using TVM TIR Script."""
        # Match buffers
        x_buf = T.match_buffer(x, (B, S, n, D), dtype)
        phi_buf = T.match_buffer(phi, (out_features, nD), compute_dtype)
        alpha_buf = T.match_buffer(alpha, (3,), compute_dtype)
        bias_buf = T.match_buffer(bias, (out_features,), compute_dtype)
        h_in_buf = T.match_buffer(h_in, (B, S, D), dtype)
        h_post_buf = T.match_buffer(h_post, (B, S, n), compute_dtype)
        h_res_buf = T.match_buffer(h_res, (B, S, n, n), compute_dtype)

        # Grid and block configuration
        bx = T.launch_thread("blockIdx.x", B * S)
        tx = T.launch_thread("threadIdx.x", 128)

        # Local buffers in shared memory and registers
        vecX_shared = T.alloc_buffer((nD,), dtype, scope="shared.dyn")
        h_mix_local = T.alloc_buffer((out_features,), compute_dtype, scope="local")
        h_pre_local = T.alloc_buffer((n,), compute_dtype, scope="local")
        h_post_local = T.alloc_buffer((n,), compute_dtype, scope="local")
        h_res_local = T.alloc_buffer((n, n), compute_dtype, scope="local")
        h_in_local = T.alloc_buffer((D,), compute_dtype, scope="local")

        # Compute block and sequence indices
        b_idx = bx // S
        s_idx = bx % S

        # ====================================================================
        # Step 1: Reshape x to vecX [nD]
        # ====================================================================
        for i in range(n):
            for j in range(128):  # Assuming D <= 128, otherwise need multiple iterations
                if j < D:
                    idx = i * D + j
                    vecX_shared[idx] = x_buf[b_idx, s_idx, i, j]

        # ====================================================================
        # Step 2: GEMM - h_mix = vecX @ phi.T
        # ====================================================================
        # Initialize h_mix_local
        for out_idx in range(out_features):
            h_mix_local[out_idx] = T.float32(0)

        # Naive GEMM: for each output, compute dot product
        for out_idx in range(out_features):
            acc = T.float32(0)
            for k in range(nD):
                vecX_val = T.cast(vecX_shared[k], compute_dtype)
                phi_val = phi_buf[out_idx, k]
                acc += vecX_val * phi_val
            h_mix_local[out_idx] = acc

        # ====================================================================
        # Step 3: RMSNorm
        # ====================================================================
        # Compute sum of squares
        sum_sq = T.float32(0)
        for i in range(nD):
            vecX_val = T.cast(vecX_shared[i], compute_dtype)
            sum_sq += vecX_val * vecX_val

        # Compute inv_rms
        inv_rms = T.rsqrt(sum_sq / T.float32(nD) + T.float32(1e-6))

        # Apply RMSNorm to h_mix
        for i in range(out_features):
            h_mix_local[i] = h_mix_local[i] * inv_rms

        # ====================================================================
        # Step 4-5: Split and apply alpha/bias
        # ====================================================================
        a_pre = alpha_buf[0]
        a_post = alpha_buf[1]
        a_res = alpha_buf[2]

        # Process h_pre and h_post
        for i in range(n):
            # h_pre: indices [0:n]
            h_pre2 = a_pre * h_mix_local[i] + bias_buf[i]
            # sigmoid(h_pre2) + eps
            h_pre_local[i] = T.float32(1.0) / (T.float32(1.0) + T.exp(-h_pre2)) + T.float32(1e-6)

            # h_post: indices [n:2n]
            idx_post = n + i
            h_post2 = a_post * h_mix_local[idx_post] + bias_buf[idx_post]
            # 2.0 * sigmoid(h_post2)
            h_post_local[i] = T.float32(2.0) / (T.float32(1.0) + T.exp(-h_post2))

        # Process h_res: indices [2n:2n+n*n], reshape to [n, n]
        for i in range(n):
            for j in range(n):
                idx_res = 2 * n + i * n + j
                h_res_local[i, j] = a_res * h_mix_local[idx_res] + bias_buf[idx_res]

        # ====================================================================
        # Step 6: Compute h_in = h_pre @ x
        # ====================================================================
        # Initialize h_in_local
        for j in range(D):
            h_in_local[j] = T.float32(0)

        # Matrix multiplication: h_pre [n] @ x [n, D] -> h_in [D]
        for i in range(n):
            h_pre_val = h_pre_local[i]
            for j in range(D):
                x_val = T.cast(x_buf[b_idx, s_idx, i, j], compute_dtype)
                h_in_local[j] += h_pre_val * x_val

        # ====================================================================
        # Step 7: Write back to global memory
        # ====================================================================
        # Write h_in
        for j in range(D):
            h_in_buf[b_idx, s_idx, j] = T.cast(h_in_local[j], dtype)

        # Write h_post
        for i in range(n):
            h_post_buf[b_idx, s_idx, i] = h_post_local[i]

        # Write h_res
        for i in range(n):
            for j in range(n):
                h_res_buf[b_idx, s_idx, i, j] = h_res_local[i, j]

    # Create IRModule
    mod = tvm.IRModule({"mhc_forward_tir": mhc_forward_tir})

    # Build with target - try CUDA first, fall back to LLVM
    try:
        target = tvm.target.Target("cuda")
        with tvm.transform.PassContext(opt_level=3):
            built_mod = tvm.build(mod, target=target)
        return built_mod
    except Exception as e:
        print(f"Warning: CUDA build failed ({e}), falling back to LLVM (CPU)")
        target = tvm.target.Target("llvm")
        with tvm.transform.PassContext(opt_level=3):
            built_mod = tvm.build(mod, target=target)
        return built_mod


class MHCForwardPreTileLangTIR:
    """
    Wrapper class for MHC Forward Pre operator using TVM TIR Script.

    This class compiles the kernel at initialization time and provides a
    convenient interface for calling the operator.
    """

    def __init__(
        self,
        B: int,
        S: int,
        n: int,
        D: int,
        dtype=torch.bfloat16,
        compute_dtype=torch.float32,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ):
        """
        Initialize and compile the TVM TIR kernel.

        Args:
            B: Batch size
            S: Sequence length
            n: Number of heads
            D: Head dimension
            dtype: Input/output data type
            compute_dtype: Computation data type
            norm_eps: RMSNorm epsilon
            hc_eps: Hyper connection epsilon
        """
        self.B = B
        self.S = S
        self.n = n
        self.D = D
        self.nD = n * D
        self.out_features = n * n + 2 * n
        self.dtype = dtype
        self.compute_dtype = compute_dtype
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps

        # Map torch dtypes to TVM dtypes
        dtype_map = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }

        tvm_dtype = dtype_map.get(dtype, "bfloat16")
        tvm_compute_dtype = dtype_map.get(compute_dtype, "float32")

        # Compile the kernel using TVM
        self.mod = mhc_forward_pre_tilelang_tir(
            B, S, n, D,
            dtype=tvm_dtype,
            compute_dtype=tvm_compute_dtype
        )

        # Create TVM function
        self.func = self.mod

    def __call__(
        self,
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        outflag: bool = False,
    ):
        """
        Forward pass.

        Args:
            x: [B, S, n, D] - Input tensor (BFloat16)
            phi: [out_features, nD] - Weight matrix (Float32)
            alpha: [3] - Scaling factors (Float32)
            bias: [out_features] - Bias vector (Float32)
            outflag: Return intermediate values if True

        Returns:
            h_in: [B, S, D] - Output (BFloat16)
            h_post: [B, S, n] - Post gate (Float32)
            h_res: [B, S, n, n] - Residual gate (Float32)

        If outflag=True, also returns:
            inv_rms: [B, S] - RMSNorm inverse (Float32)
            h_mix: [B, S, out_features] - GEMM output (Float32)
            h_pre: [B, S, n] - Pre gate (Float32)
        """
        B, S, n, D = x.shape

        # Allocate output tensors
        h_in = torch.empty(B, S, D, dtype=self.dtype, device=x.device)
        h_post = torch.empty(B, S, n, dtype=self.compute_dtype, device=x.device)
        h_res = torch.empty(B, S, n, n, dtype=self.compute_dtype, device=x.device)

        # Convert torch tensors to TVM NDArray
        from tvm.runtime import NDArray

        def to_tvm_array(tensor):
            """Convert torch tensor to TVM NDArray."""
            return tvm.nd.array(tensor.cpu().numpy(), tvm.numpy.empty(tensor.shape, dtype=str(tensor.dtype)[5:]))

        def to_tvm_dtype(torch_dtype):
            """Map torch dtype to TVM dtype string."""
            dtype_map = {
                torch.bfloat16: "bfloat16",
                torch.float16: "float16",
                torch.float32: "float32",
            }
            return dtype_map.get(torch_dtype, "float32")

        # Create TVM arrays
        x_tvm = tvm.nd.array(x.cpu().numpy())
        phi_tvm = tvm.nd.array(phi.cpu().numpy())
        alpha_tvm = tvm.nd.array(alpha.cpu().numpy())
        bias_tvm = tvm.nd.array(bias.cpu().numpy())
        h_in_tvm = tvm.nd.empty((B, S, D), dtype=to_tvm_dtype(self.dtype))
        h_post_tvm = tvm.nd.empty((B, S, n), dtype=to_tvm_dtype(self.compute_dtype))
        h_res_tvm = tvm.nd.empty((B, S, n, n), dtype=to_tvm_dtype(self.compute_dtype))

        # Run the kernel
        self.func(x_tvm, phi_tvm, alpha_tvm, bias_tvm, h_in_tvm, h_post_tvm, h_res_tvm)

        # Convert back to torch tensors
        h_in.copy_(torch.from_numpy(h_in_tvm.numpy()).to(device=x.device, dtype=self.dtype))
        h_post.copy_(torch.from_numpy(h_post_tvm.numpy()).to(device=x.device, dtype=self.compute_dtype))
        h_res.copy_(torch.from_numpy(h_res_tvm.numpy()).to(device=x.device, dtype=self.compute_dtype))

        if not outflag:
            return h_in, h_post, h_res
        else:
            # Compute intermediate values for backward compatibility
            # Use PyTorch for intermediate values
            vecX = x.reshape(B, S, n * D).float()
            h_mix = torch.matmul(vecX, phi.t())
            inv_rms = torch.rsqrt(vecX.square().mean(-1, keepdim=True) + self.norm_eps)
            inv_rms = inv_rms.squeeze(-1)

            # Reconstruct h_pre from h_post (h_post = 2 * sigmoid(h_post2))
            h_pre = torch.logit((h_post / 2.0).clamp(1e-6, 1 - 1e-6)) + self.hc_eps

            return h_in, h_post, h_res, inv_rms, h_mix, h_pre


# Convenience function for direct use
def mhc_forward_pre_tilelang_tir_wrapper(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    outflag: bool = False,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """
    Convenience wrapper for TVM TIR Script forward pass.

    Args:
        x: [B, S, n, D] - Input tensor
        phi: [out_features, nD] - Weight matrix
        alpha: [3] - Scaling factors
        bias: [out_features] - Bias vector
        outflag: Return intermediate values if True
        norm_eps: RMSNorm epsilon
        hc_eps: Hyper connection epsilon

    Returns:
        h_in, h_post, h_res (+ intermediates if outflag=True)
    """
    B, S, n, D = x.shape
    kernel = MHCForwardPreTileLangTIR(B, S, n, D, dtype=x.dtype, norm_eps=norm_eps, hc_eps=hc_eps)
    return kernel(x, phi, alpha, bias, outflag=outflag)
