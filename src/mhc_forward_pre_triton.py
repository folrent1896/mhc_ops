"""
MHC Forward Pre Operator - Triton Implementation

Input:
    x     : [B, S, n, D]                BFloat16
    phi   : [n^2 + 2n, nD]              Float32
    alpha : [3] -> [pre, post, res]     Float32
    bias  : [n^2 + 2n]                  Float32

Output:
    h_in   : [B, S, D]                     BFloat16
    h_post : [B, S, n]                     Float32
    h_res  : [B, S, n, n]                  Float32
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def mhc_forward_pre_kernel(
    # Input pointers
    x_ptr,
    phi_ptr,
    alpha_ptr,
    bias_ptr,

    # Output pointers
    h_in_ptr,
    h_post_ptr,
    h_res_ptr,

    # Scalar parameters
    B, S, n, D,
    nD,
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_phi_out, stride_phi_in,
    stride_hin_b, stride_hin_s, stride_hin_d,
    stride_hpost_b, stride_hpost_s, stride_hpost_n,
    stride_hres_b, stride_hres_s, stride_hres_n1, stride_hres_n2,

    # Epsilons
    norm_eps: tl.float32,
    hc_eps: tl.float32,

    # Meta parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for MHC Forward Pre operation.

    This kernel computes:
    1. GEMM: h_mix = vecX @ phi^T
    2. RMSNorm on vecX
    3. Apply alpha scaling and bias
    4. Sigmoid activation for h_pre and h_post
    5. Compute h_in = h_pre @ x
    """

    # Program ID maps to batch and sequence
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    # -----------------------------------------------------------
    # Step 1: Load x block [n, D] and compute vecX [nD]
    # -----------------------------------------------------------
    # x shape: [B, S, n, D]
    # Compute offsets for x block
    x_off_n = tl.arange(0, BLOCK_SIZE_N)
    x_off_d = tl.arange(0, BLOCK_SIZE_K)

    x_mask = (x_off_n[:, None] < n) & (x_off_d[None, :] < D)

    x_offset = (b_idx * stride_x_b + s_idx * stride_x_s +
                x_off_n[:, None] * stride_x_n +
                x_off_d[None, :] * stride_x_d)

    x_block = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0).to(tl.float32)

    # Flatten to vecX: [n, D] -> [nD]
    # We'll compute GEMM directly on the 2D block

    # -----------------------------------------------------------
    # Step 2: Compute RMSNorm
    # -----------------------------------------------------------
    # Compute mean of squares: mean(x^2) over n*D
    x_sq = x_block ** 2
    x_sq_sum = tl.sum(x_sq)
    mean_sq = x_sq_sum / (n * D)
    inv_rms = tl.rsqrt(mean_sq + norm_eps)

    # -----------------------------------------------------------
    # Step 3: GEMM - h_mix = vecX @ phi^T
    # phi shape: [n^2 + 2n, nD], need phi^T: [nD, n^2 + 2n]
    # Output: [n^2 + 2n]
    # -----------------------------------------------------------
    out_features = n * n + 2 * n

    h_mix = tl.zeros([out_features], dtype=tl.float32)

    # Accumulate over K (nD) dimension
    for k_start in range(0, nD, BLOCK_SIZE_K):
        k_off = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_off < nD

        # Load vecX chunk (flattened x_block)
        # x_block is [n, D], flatten row-major to [nD]
        vecX_chunk = tl.zeros([BLOCK_SIZE_K], dtype=tl.float32)

        # Load phi chunk: [n^2 + 2n, nD]
        # We need to load phi^T chunk: [nD, n^2 + 2n]
        phi_off_out = tl.arange(0, out_features)
        phi_off_in = k_off

        phi_mask = k_mask[None, :] & (phi_off_out[:, None] < out_features)

        phi_offset = (phi_off_out[:, None] * stride_phi_out +
                      phi_off_in[None, :] * stride_phi_in)

        phi_chunk = tl.load(phi_ptr + phi_offset, mask=phi_mask, other=0.0)

        # Matrix multiply: h_mix += vecX @ phi^T
        # We need to flatten x_block properly
        for i in range(BLOCK_SIZE_N):
            for j in range(BLOCK_SIZE_K):
                if i * BLOCK_SIZE_K + j < BLOCK_SIZE_K:
                    vecX_chunk[j] = x_block[i, j]

        # Accumulate
        h_mix += tl.sum(vecX_chunk[None, :] * phi_chunk, axis=1)

    # -----------------------------------------------------------
    # Step 4: Apply RMSNorm and split
    # -----------------------------------------------------------
    h_mix = h_mix * inv_rms

    # Split into h_pre1, h_post1, h_res1
    h_pre1 = h_mix[:n]
    h_post1 = h_mix[n:2*n]
    h_res1 = h_mix[2*n:]

    # -----------------------------------------------------------
    # Step 5: Load alpha and bias, apply scaling
    # -----------------------------------------------------------
    alpha = tl.load(alpha_ptr)  # [3]
    bias = tl.load(bias_ptr + tl.arange(0, out_features))

    bias_pre = bias[:n]
    bias_post = bias[n:2*n]
    bias_res = bias[2*n:]

    a_pre, a_post, a_res = alpha[0], alpha[1], alpha[2]

    h_pre2 = a_pre * h_pre1 + bias_pre
    h_post2 = a_post * h_post1 + bias_post
    h_res2_flat = a_res * h_res1 + bias_res

    # Reshape h_res2 to [n, n]
    h_res2 = h_res2_flat.reshape(n, n)

    # -----------------------------------------------------------
    # Step 6: Apply sigmoid activation
    # -----------------------------------------------------------
    h_pre = tl.sigmoid(h_pre2) + hc_eps  # [n]
    h_post = 2.0 * tl.sigmoid(h_post2)   # [n]

    # -----------------------------------------------------------
    # Step 7: Compute h_in = h_pre @ x
    # h_pre: [n], x: [n, D] -> h_in: [D]
    # -----------------------------------------------------------
    h_in = tl.zeros([D], dtype=tl.float32)
    for i in range(n):
        h_in += h_pre[i] * x_block[i, :]

    # -----------------------------------------------------------
    # Step 8: Store outputs
    # -----------------------------------------------------------
    # Store h_in [D]
    h_in_off_d = tl.arange(0, BLOCK_SIZE_K)
    h_in_mask = h_in_off_d < D
    h_in_offset = (b_idx * stride_hin_b + s_idx * stride_hin_s +
                   h_in_off_d * stride_hin_d)
    tl.store(h_in_ptr + h_in_offset, h_in, mask=h_in_mask)

    # Store h_post [n]
    h_post_off_n = tl.arange(0, BLOCK_SIZE_N)
    h_post_mask = h_post_off_n < n
    h_post_offset = (b_idx * stride_hpost_b + s_idx * stride_hpost_s +
                     h_post_off_n * stride_hpost_n)
    tl.store(h_post_ptr + h_post_offset, h_post, mask=h_post_mask)

    # Store h_res [n, n]
    h_res_off_n1 = tl.arange(0, BLOCK_SIZE_N)
    h_res_off_n2 = tl.arange(0, BLOCK_SIZE_N)
    h_res_mask = (h_res_off_n1[:, None] < n) & (h_res_off_n2[None, :] < n)
    h_res_offset = (b_idx * stride_hres_b + s_idx * stride_hres_s +
                    h_res_off_n1[:, None] * stride_hres_n1 +
                    h_res_off_n2[None, :] * stride_hres_n2)
    tl.store(h_res_ptr + h_res_offset, h_res2, mask=h_res_mask)


def mhc_forward_pre_triton(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    outflag: bool = False,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6
):
    """
    MHC Forward Pre - Triton implementation.

    Args:
        x: [B, S, n, D] - BFloat16 input
        phi: [n^2 + 2n, nD] - Float32 weight matrix
        alpha: [3] - Float32 scaling factors [pre, post, res]
        bias: [n^2 + 2n] - Float32 bias
        outflag: Whether to return intermediate values
        norm_eps: RMSNorm epsilon
        hc_eps: Hyper connection epsilon

    Returns:
        h_in: [B, S, D] - BFloat16
        h_post: [B, S, n] - Float32
        h_res: [B, S, n, n] - Float32
    """
    B, S, n, D = x.shape
    nD = n * D
    out_features = n * n + 2 * n

    # Ensure inputs are on same device and contiguous
    x = x.contiguous()
    phi = phi.contiguous()
    alpha = alpha.contiguous()
    bias = bias.contiguous()

    # Allocate outputs
    h_in = torch.empty((B, S, D), dtype=x.dtype, device=x.device)
    h_post = torch.empty((B, S, n), dtype=torch.float32, device=x.device)
    h_res = torch.empty((B, S, n, n), dtype=torch.float32, device=x.device)

    # Block sizes
    BLOCK_SIZE_M = 1  # Process one (b, s) at a time
    BLOCK_SIZE_N = triton.next_power_of_2(n)
    BLOCK_SIZE_K = triton.next_power_of_2(min(D, nD))

    # Grid
    grid = (B * S,)

    # Launch kernel
    with torch.cuda.device(x.device.index if x.is_cuda else 0):
        mhc_forward_pre_kernel[grid](
            x_ptr=x,
            phi_ptr=phi,
            alpha_ptr=alpha,
            bias_ptr=bias,

            h_in_ptr=h_in,
            h_post_ptr=h_post,
            h_res_ptr=h_res,

            B=B, S=S, n=n, D=D,
            nD=nD,

            stride_x_b=S * n * D,
            stride_x_s=n * D,
            stride_x_n=D,
            stride_x_d=1,

            stride_phi_out=nD,
            stride_phi_in=1,

            stride_hin_b=S * D,
            stride_hin_s=D,
            stride_hin_d=1,

            stride_hpost_b=S * n,
            stride_hpost_s=n,
            stride_hpost_n=1,

            stride_hres_b=S * n * n,
            stride_hres_s=n * n,
            stride_hres_n1=n,
            stride_hres_n2=1,

            norm_eps=norm_eps,
            hc_eps=hc_eps,

            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

    if not outflag:
        return h_in, h_post, h_res
    else:
        # Recompute intermediate values for backward compatibility
        vecX = x.reshape(B, S, nD).float()
        h_mix = torch.matmul(vecX, phi.t())
        inv_rms = torch.rsqrt(vecX.square().mean(-1, keepdim=True) + norm_eps)
        inv_rms = inv_rms.squeeze(-1)
        h_pre = h_post / 2.0  # Reverse of h_post = 2.0 * sigmoid(...)
        h_pre = torch.logit(h_pre.clamp(0, 1)) + hc_eps
        return h_in, h_post, h_res, inv_rms, h_mix, h_pre


# ============================================================================
# Alternative optimized version using separate kernels for each stage
# ============================================================================

@triton.jit
def gemm_kernel(
    x_ptr,
    phi_ptr,
    output_ptr,
    B, S, n, D, nD,
    out_features,
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_phi_out, stride_phi_in,
    stride_out_b, stride_out_s, stride_out_feat,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """GEMM kernel: output = vecX @ phi^T"""
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    # Output feature dimension
    feat_id = tl.program_id(axis=1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    feat_mask = feat_id < out_features

    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    for k_start in range(0, nD, BLOCK_SIZE_K):
        k = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k < nD

        # Load x chunk [nD]
        x_off = (b_idx * stride_x_b + s_idx * stride_x_s +
                 (k // D) * stride_x_n +
                 (k % D) * stride_x_d)
        x_vals = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)

        # Load phi chunk [out_features, nD]
        phi_off = (feat_id[:, None] * stride_phi_out +
                   k[None, :] * stride_phi_in)
        phi_mask = feat_mask[:, None] & k_mask[None, :]
        phi_vals = tl.load(phi_ptr + phi_off, mask=phi_mask, other=0.0)

        # Accumulate
        acc += tl.sum(x_vals[None, :] * phi_vals, axis=1)

    # Store
    out_off = (b_idx * stride_out_b + s_idx * stride_out_s +
               feat_id * stride_out_feat)
    tl.store(output_ptr + out_off, acc, mask=feat_mask)


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    inv_rms_ptr,
    B, S, n, D, nD,
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_inv_b, stride_inv_s,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    norm_eps: tl.float32,
):
    """RMSNorm kernel: compute inverse RMS for each batch and sequence"""
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    acc = tl.zeros([], dtype=tl.float32)

    # Process in blocks
    for n_start in range(0, n, BLOCK_SIZE_N):
        n_off = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_off < n

        for d_start in range(0, D, BLOCK_SIZE_K):
            d_off = d_start + tl.arange(0, BLOCK_SIZE_K)
            d_mask = d_off < D

            x_off = (b_idx * stride_x_b + s_idx * stride_x_s +
                     n_off[:, None] * stride_x_n +
                     d_off[None, :] * stride_x_d)
            x_mask = n_mask[:, None] & d_mask[None, :]

            x_block = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
            acc += tl.sum(x_block ** 2)

    mean_sq = acc / (nD)
    inv_rms = tl.rsqrt(mean_sq + norm_eps)

    # Store
    inv_off = b_idx * stride_inv_b + s_idx * stride_inv_s
    tl.store(inv_rms_ptr + inv_off, inv_rms)


def mhc_forward_pre_triton_optimized(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    outflag: bool = False,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6
):
    """
    Optimized MHC Forward Pre using separate kernels.
    """
    B, S, n, D = x.shape
    nD = n * D
    out_features = n * n + 2 * n

    x = x.contiguous()
    phi = phi.contiguous()
    alpha = alpha.contiguous()
    bias = bias.contiguous()

    # Step 1: Compute h_mix via GEMM
    h_mix = torch.empty((B, S, out_features), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_N = triton.next_power_of_2(min(out_features, 64))
    BLOCK_SIZE_K = triton.next_power_of_2(min(nD, 64))

    grid = (B * S, triton.cdiv(out_features, BLOCK_SIZE_N))

    with torch.cuda.device(x.device.index if x.is_cuda else 0):
        gemm_kernel[grid](
            x_ptr=x,
            phi_ptr=phi,
            output_ptr=h_mix,
            B=B, S=S, n=n, D=D, nD=nD,
            out_features=out_features,
            stride_x_b=S * n * D,
            stride_x_s=n * D,
            stride_x_n=D,
            stride_x_d=1,
            stride_phi_out=nD,
            stride_phi_in=1,
            stride_out_b=S * out_features,
            stride_out_s=out_features,
            stride_out_feat=1,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

    # Step 2: Compute RMSNorm
    inv_rms = torch.empty((B, S), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_N = triton.next_power_of_2(min(n, 32))
    BLOCK_SIZE_K = triton.next_power_of_2(min(D, 32))

    grid = (B * S,)

    with torch.cuda.device(x.device.index if x.is_cuda else 0):
        rmsnorm_kernel[grid](
            x_ptr=x,
            inv_rms_ptr=inv_rms,
            B=B, S=S, n=n, D=D, nD=nD,
            stride_x_b=S * n * D,
            stride_x_s=n * D,
            stride_x_n=D,
            stride_x_d=1,
            stride_inv_b=S,
            stride_inv_s=1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            norm_eps=norm_eps,
        )

    # Step 3-7: CPU/GPU fused operations
    # Apply RMSNorm, split, alpha/bias, sigmoid, compute h_in
    h_mix_normalized = h_mix * inv_rms.unsqueeze(-1)

    h_pre1 = h_mix_normalized[..., :n]
    h_post1 = h_mix_normalized[..., n:2*n]
    h_res1 = h_mix_normalized[..., 2*n:]

    a_pre, a_post, a_res = alpha[0], alpha[1], alpha[2]

    h_pre2 = a_pre * h_pre1 + bias[:n]
    h_post2 = a_post * h_post1 + bias[n:2*n]
    h_res2 = a_res * h_res1.reshape(B, S, n, n) + bias[2*n:].view(n, n)

    h_pre = torch.sigmoid(h_pre2) + hc_eps
    h_post = 2.0 * torch.sigmoid(h_post2)
    h_res = h_res2

    # h_in = h_pre @ x
    h_in_fp = (h_pre.unsqueeze(-1) * x.float()).sum(dim=2)
    h_in = h_in_fp.to(x.dtype)

    if not outflag:
        return h_in, h_post, h_res
    else:
        return h_in, h_post, h_res, inv_rms, h_mix, h_pre


# ============================================================================
# Test function
# ============================================================================

def test_mhc_forward_pre_triton():
    """Test Triton implementation against reference."""
    import sys
    sys.path.append('/Users/huan1178/Downloads/code-base/mhc-ops')
    from test_mhc_pre_grad import mhc_forward_pre

    # Test parameters
    B, S, n, D = 2, 128, 4, 256

    # Create test data
    torch.manual_seed(42)
    x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device='cuda')
    phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device='cuda')
    alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32, device='cuda')
    bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cuda') * 0.1

    # Reference
    h_in_ref, h_post_ref, h_res_ref = mhc_forward_pre(
        x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu()
    )
    h_in_ref = h_in_ref.cuda()
    h_post_ref = h_post_ref.cuda()
    h_res_ref = h_res_ref.cuda()

    # Triton
    h_in_tri, h_post_tri, h_res_tri = mhc_forward_pre_triton_optimized(
        x, phi, alpha, bias
    )

    # Compare
    print(f"h_in max error: {(h_in_tri.float() - h_in_ref.float()).abs().max().item()}")
    print(f"h_post max error: {(h_post_tri - h_post_ref).abs().max().item()}")
    print(f"h_res max error: {(h_res_tri - h_res_ref).abs().max().item()}")

    assert torch.allclose(h_in_tri.float(), h_in_ref.float(), rtol=1e-3, atol=1e-3)
    assert torch.allclose(h_post_tri, h_post_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(h_res_tri, h_res_ref, rtol=1e-3, atol=1e-3)
    print("All tests passed!")


if __name__ == "__main__":
    test_mhc_forward_pre_triton()
