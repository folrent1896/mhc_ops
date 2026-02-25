"""
MHC Backward Operator - Triton Implementation (FIXED)

This module implements the backward pass for MHC Forward Pre operator using a
multi-kernel architecture for correctness and performance.

FIXES:
1. dvecX_mm: Now correctly handles nD > BLOCK_SIZE_K by writing to global memory
2. dx: Includes GEMM term (dvecX_mm @ gamma.T)
3. dphi: Fully implemented
4. dgamma: Fully implemented
5. Optimized memory access patterns

Architecture:
- Kernel 1: Main backward kernel (computes dalpha, dbias, dvecX_mm, dh_mix)
- Kernel 2: Compute dx (dvecX_mm @ gamma.T + dvecX_inv + dvecX_hin)
- Kernel 3: Compute dphi (dh_mix.T @ (x * gamma))
- Kernel 4: Compute dgamma (sum x * dvecX_mm)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mhc_backward_kernel(
    # Input pointers (forward outputs and gradients)
    x_ptr,
    phi_ptr,
    alpha_ptr,
    bias_ptr,
    inv_rms_ptr,
    h_mix_ptr,
    h_pre_ptr,
    h_post_ptr,
    dh_in_ptr,
    dh_post_ptr,
    dh_res_ptr,
    gamma_ptr,

    # Output pointers (gradients)
    dalpha_ptr,
    dbias_ptr,
    dvecX_mm_ptr,  # Intermediate output
    dvecX_inv_ptr,  # Intermediate output for RMSNorm gradient

    # Scalar parameters
    B, S, n, D,
    nD,
    out_features,

    # Strides
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_phi_out, stride_phi_in,
    stride_hin_b, stride_hin_s, stride_hin_d,
    stride_hpost_b, stride_hpost_s, stride_hpost_n,
    stride_hres_b, stride_hres_s, stride_hres_n1, stride_hres_n2,
    stride_dvecxmm_b, stride_dvecxmm_s, stride_dvecxmm_d,
    stride_dvecinv_b, stride_dvecinv_s, stride_dvecinv_n, stride_dvecinv_d,

    # Epsilons
    norm_eps: tl.float32,
    hc_eps: tl.float32,

    # Block sizes
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Main backward kernel: computes dalpha, dbias, dvecX_mm, dvecX_inv

    This kernel handles:
    1. Gradient computation for dalpha and dbias
    2. dvecX_mm = dh_mix @ phi (written to global memory)
    3. dvecX_inv = RMSNorm gradient (written to global memory)
    """
    # Program ID
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    # ============================================================
    # Step 1: Load forward outputs and gradients
    # ============================================================
    # Load x [n, D]
    x_off_n = tl.arange(0, BLOCK_SIZE_N)
    x_off_d = tl.arange(0, BLOCK_SIZE_K)
    x_mask = (x_off_n[:, None] < n) & (x_off_d[None, :] < D)
    x_offset = (b_idx * stride_x_b + s_idx * stride_x_s +
                x_off_n[:, None] * stride_x_n +
                x_off_d[None, :] * stride_x_d)
    x_block = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0).to(tl.float32)

    # Load inv_rms [1]
    inv_rms_off = b_idx * S + s_idx
    inv_rms = tl.load(inv_rms_ptr + inv_rms_off)

    # Load h_pre [n], h_post [n]
    h_pre_off = (b_idx * stride_hpost_b + s_idx * stride_hpost_s +
                 x_off_n * stride_hpost_n)
    h_pre_mask = x_off_n < n
    h_pre = tl.load(h_pre_ptr + h_pre_off, mask=h_pre_mask, other=0.0)
    h_post = tl.load(h_post_ptr + h_pre_off, mask=h_pre_mask, other=0.0)

    # Load dh_in [D]
    dh_in_off = x_off_d
    dh_in_mask = x_off_d < D
    dh_in_offset = (b_idx * stride_hin_b + s_idx * stride_hin_s +
                     dh_in_off * stride_hin_d)
    dh_in = tl.load(dh_in_ptr + dh_in_offset, mask=dh_in_mask, other=0.0)

    # Load dh_post [n]
    dh_post = tl.load(dh_post_ptr + h_pre_off, mask=h_pre_mask, other=0.0)

    # Load dh_res [n, n] - load once, reuse multiple times
    dh_res_off_n1 = x_off_n
    dh_res_off_n2 = tl.arange(0, BLOCK_SIZE_N)
    dh_res_mask = (dh_res_off_n1[:, None] < n) & (dh_res_off_n2[None, :] < n)
    dh_res_offset = (b_idx * stride_hres_b + s_idx * stride_hres_s +
                     dh_res_off_n1[:, None] * stride_hres_n1 +
                     dh_res_off_n2[None, :] * stride_hres_n2)
    dh_res_block = tl.load(dh_res_ptr + dh_res_offset, mask=dh_res_mask, other=0.0)

    # Load alpha [3]
    a_pre = tl.load(alpha_ptr + 0)
    a_post = tl.load(alpha_ptr + 1)
    a_res = tl.load(alpha_ptr + 2)

    # ============================================================
    # Step 2-4: Compute gradient components
    # ============================================================
    dh_pre = tl.sum(x_block * dh_in[None, :], axis=1)

    s_pre2 = h_pre - hc_eps
    dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
    dh_post2 = dh_post * h_post * (1.0 - h_post / 2.0)

    dh_pre1 = a_pre * dh_pre2
    dh_post1 = a_post * dh_post2
    dh_res1 = a_res * dh_res_block

    # ============================================================
    # Step 5: Accumulate dalpha and dbias
    # ============================================================
    # Load h_mix sections (h_mix from forward is BEFORE inv_rms, so we multiply by inv_rms)
    h_pre1_hmix = tl.load(h_mix_ptr + (b_idx * S * out_features + s_idx * out_features + x_off_n),
                          mask=h_pre_mask, other=0.0) * inv_rms
    h_post1_hmix = tl.load(h_mix_ptr + (b_idx * S * out_features + s_idx * out_features + n + x_off_n),
                           mask=h_pre_mask, other=0.0) * inv_rms

    dalpha_pre_sum = tl.sum(dh_pre2 * h_pre1_hmix)
    tl.atomic_add(dalpha_ptr + 0, dalpha_pre_sum)

    dalpha_post_sum = tl.sum(dh_post2 * h_post1_hmix)
    tl.atomic_add(dalpha_ptr + 1, dalpha_post_sum)

    h_res1_hmix_off_i = x_off_n[:, None]
    h_res1_hmix_off_j = x_off_n[None, :]
    h_res1_hmix_mask = (h_res1_hmix_off_i < n) & (h_res1_hmix_off_j < n)
    h_res1_hmix_off = (b_idx * S * out_features + s_idx * out_features + 2 * n +
                       h_res1_hmix_off_i * n + h_res1_hmix_off_j)
    h_res1_hmix = tl.load(h_mix_ptr + h_res1_hmix_off, mask=h_res1_hmix_mask, other=0.0) * inv_rms

    dalpha_res_sum = tl.sum(dh_res_block * h_res1_hmix)
    tl.atomic_add(dalpha_ptr + 2, dalpha_res_sum)

    dbias_indices = tl.arange(0, BLOCK_SIZE_N)
    dbias_mask = dbias_indices < n

    # dbias_pre: accumulate dh_pre2 over B, S
    tl.atomic_add(dbias_ptr + dbias_indices, dh_pre2, mask=dbias_mask)

    # dbias_post: accumulate dh_post2 over B, S
    tl.atomic_add(dbias_ptr + n + dbias_indices, dh_post2, mask=dbias_mask)

    # FIXED: Use dh_res_block directly instead of nested loops
    dbias_res_offset = 2 * n + x_off_n[:, None] * n + x_off_n[None, :]
    tl.atomic_add(dbias_ptr + dbias_res_offset, dh_res_block, mask=dh_res_mask)

    # ============================================================
    # Step 6: Compute dvecX_inv (RMSNorm gradient)
    # ============================================================
    # dinv_rms = sum(dh_mix * h_mix)
    # dh_mix = concat([dh_pre1, dh_post1, dh_res1_flat]) * inv_rms
    # h_mix is already loaded (already has inv_rms applied), we have dh_pre1, dh_post1, dh_res1
    # So: dh_mix[i] = dh_xxx1[i] * inv_rms
    # And we compute: dh_mix[i] * h_mix[i]
    dinv_rms_pre = tl.sum((dh_pre1 * inv_rms) * h_pre1_hmix)
    dinv_rms_post = tl.sum((dh_post1 * inv_rms) * h_post1_hmix)
    dinv_rms_res = tl.sum((dh_res1 * inv_rms) * h_res1_hmix)
    dinv_rms = dinv_rms_pre + dinv_rms_post + dinv_rms_res

    # dvecX_inv = -(dinv_rms * inv_rms^3 / nD) * x
    dvecX_inv = -(dinv_rms * inv_rms * inv_rms * inv_rms / nD) * x_block

    # Store dvecX_inv to global memory
    dvecX_inv_offset = (b_idx * stride_dvecinv_b + s_idx * stride_dvecinv_s +
                        x_off_n[:, None] * stride_dvecinv_n +
                        x_off_d[None, :] * stride_dvecinv_d)
    tl.store(dvecX_inv_ptr + dvecX_inv_offset, dvecX_inv, mask=x_mask)

    # ============================================================
    # Step 7: Compute dvecX_mm = dh_mix @ phi
    # ============================================================
    # FIX: Write each block to global memory instead of keeping only last block
    for nD_start in range(0, nD, BLOCK_SIZE_K):
        nD_idx = nD_start + tl.arange(0, BLOCK_SIZE_K)
        nD_mask = nD_idx < nD
        acc = tl.zeros([BLOCK_SIZE_K], dtype=tl.float32)

        # Part 1: dh_pre1 @ phi[0:n, :]
        phi_pre_off = (x_off_n[:, None] * stride_phi_out + nD_idx[None, :] * stride_phi_in)
        phi_pre_mask = (x_off_n[:, None] < n) & nD_mask[None, :]
        phi_pre = tl.load(phi_ptr + phi_pre_off, mask=phi_pre_mask, other=0.0)
        acc += tl.sum((dh_pre1 * inv_rms)[:, None] * phi_pre, axis=0)

        # Part 2: dh_post1 @ phi[n:2n, :]
        phi_post_off = ((n + x_off_n)[:, None] * stride_phi_out + nD_idx[None, :] * stride_phi_in)
        phi_post_mask = ((n + x_off_n)[:, None] < 2 * n) & nD_mask[None, :]
        phi_post = tl.load(phi_ptr + phi_post_off, mask=phi_post_mask, other=0.0)
        acc += tl.sum((dh_post1 * inv_rms)[:, None] * phi_post, axis=0)

        # Part 3: dh_res1 @ phi[2n:, :]
        for res_i in range(0, n, BLOCK_SIZE_N):
            for res_j in range(0, n, BLOCK_SIZE_N):
                res_i_idx = tl.arange(0, BLOCK_SIZE_N)
                res_j_idx = tl.arange(0, BLOCK_SIZE_N)
                res_i_mask = (res_i + res_i_idx) < n
                res_j_mask = (res_j + res_j_idx) < n
                res_ij_mask = res_i_mask[:, None] & res_j_mask[None, :]

                phi_row_idx = 2 * n + (res_i + res_i_idx)[:, None] * n + (res_j + res_j_idx)[None, :]
                phi_res_off = (phi_row_idx[:, :, None] * stride_phi_out + nD_idx[None, None, :] * stride_phi_in)
                phi_res_mask = res_ij_mask[:, :, None] & nD_mask[None, None, :]
                phi_res = tl.load(phi_ptr + phi_res_off, mask=phi_res_mask, other=0.0)

                # Try reversing sum order for better precision
                # Original: sum(axis=1) then sum(axis=0)
                # New: sum(axis=0) then sum(axis=0)
                # FIXED: Multiply by inv_rms like Parts 1 and 2
                temp = tl.sum((dh_res1 * inv_rms)[:, :, None] * phi_res, axis=0)
                acc += tl.sum(temp, axis=0)

        # FIX: Write to global memory for each block
        dvecX_mm_offset = (b_idx * stride_dvecxmm_b + s_idx * stride_dvecxmm_s + nD_idx)
        tl.store(dvecX_mm_ptr + dvecX_mm_offset, acc, mask=nD_mask)


@triton.jit
def mhc_backward_dx_kernel(
    # Input pointers
    dvecX_mm_ptr,
    dvecX_inv_ptr,
    gamma_ptr,
    h_pre_ptr,
    dh_in_ptr,

    # Output pointer
    dx_ptr,

    # Scalar parameters
    B, S, n, D, nD,

    # Strides
    stride_dvecxmm_b, stride_dvecxmm_s, stride_dvecxmm_d,
    stride_dvecinv_b, stride_dvecinv_s, stride_dvecinv_n, stride_dvecinv_d,
    stride_gamma_n, stride_gamma_d,
    stride_hin_b, stride_hin_s, stride_hin_d,
    stride_hpost_b, stride_hpost_s, stride_hpost_n,
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,

    # Block sizes
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute dx = dvecX_mm * gamma + dvecX_inv + dvecX_hin

    Each program handles one (b, s, n) slice and computes the full D dimension.
    """
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S
    n_idx = tl.program_id(axis=1)

    # Load h_pre[n_idx]
    h_pre_off = (b_idx * stride_hpost_b + s_idx * stride_hpost_s + n_idx)
    h_pre_val = tl.load(h_pre_ptr + h_pre_off)

    # Load dh_in [D]
    d_off = tl.arange(0, BLOCK_SIZE_K)
    d_mask = d_off < D
    dh_in_offset = (b_idx * stride_hin_b + s_idx * stride_hin_s + d_off)
    dh_in = tl.load(dh_in_ptr + dh_in_offset, mask=d_mask, other=0.0)

    # Compute dvecX_hin = h_pre * dh_in
    dvecX_hin = h_pre_val * dh_in

    # Load dvecX_inv [D] for this n_idx
    dvecX_inv_offset = (b_idx * stride_dvecinv_b + s_idx * stride_dvecinv_s +
                        n_idx * stride_dvecinv_n + d_off * stride_dvecinv_d)
    dvecX_inv = tl.load(dvecX_inv_ptr + dvecX_inv_offset, mask=d_mask, other=0.0)

    # Load dvecX_mm slice for this n_idx: [D]
    dvecX_mm_offset = (b_idx * stride_dvecxmm_b + s_idx * stride_dvecxmm_s +
                       n_idx * D + d_off)
    dvecX_mm_slice = tl.load(dvecX_mm_ptr + dvecX_mm_offset, mask=d_mask, other=0.0)

    # Load gamma row: [D]
    gamma_off = (n_idx * stride_gamma_n + d_off)
    gamma_row = tl.load(gamma_ptr + gamma_off, mask=d_mask, other=0.0)

    # Element-wise multiplication: dx[n_idx, d] = dvecX_mm[n_idx*D+d] * gamma[n_idx, d]
    elem_contrib = dvecX_mm_slice * gamma_row

    # dx = element-wise + dvecX_inv + dvecX_hin
    dx = elem_contrib + dvecX_inv + dvecX_hin

    # Store dx
    dx_offset = (b_idx * stride_x_b + s_idx * stride_x_s +
                 n_idx * stride_x_n + d_off)
    tl.store(dx_ptr + dx_offset, dx, mask=d_mask)


@triton.jit
def mhc_backward_dphi_kernel(
    # Input pointers
    dh_mix_ptr,
    x_ptr,
    gamma_ptr,

    # Output pointer
    dphi_ptr,

    # Scalar parameters
    B, S, n, D, nD,
    out_features,

    # Strides
    stride_dmix_b, stride_dmix_s, stride_dmix_out,
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_gamma_n, stride_gamma_d,
    stride_phi_out, stride_phi_in,

    # Block sizes
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute dphi = dh_mix.T @ (x * gamma)

    dphi[out, in] = sum_{B,S} dh_mix[b, s, out] * (x * gamma)[b, s, in]

    Each program handles one (out, in_block) and accumulates over all (b,s).
    """
    # Grid is (out_features, nD // BLOCK_SIZE_K)
    out_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    out_mask = out_idx < out_features

    # This program handles dphi[out_idx, block_idx*BLOCK_SIZE_K:(block_idx+1)*BLOCK_SIZE_K]
    k = tl.arange(0, BLOCK_SIZE_K)
    nD_idx = block_idx * BLOCK_SIZE_K + k
    nD_mask = nD_idx < nD

    # Convert nD_idx to (n_idx, d_idx)
    n_idx = nD_idx // D
    d_idx = nD_idx % D

    n_mask = n_idx < n
    nd_mask = nD_mask & n_mask

    # Accumulator for this block
    acc = tl.zeros([BLOCK_SIZE_K], dtype=tl.float32)

    # Iterate over all batch and sequence elements
    for bs_idx in range(B * S):
        b_idx = bs_idx // S
        s_idx = bs_idx % S

        # Load dh_mix for this output feature (scalar)
        dh_mix_off = (b_idx * stride_dmix_b + s_idx * stride_dmix_s + out_idx)
        dh_mix_val = tl.load(dh_mix_ptr + dh_mix_off, mask=out_mask, other=0.0)

        # Load x
        x_off = (b_idx * stride_x_b + s_idx * stride_x_s +
                 n_idx * stride_x_n +
                 d_idx * stride_x_d)
        x_vals = tl.load(x_ptr + x_off, mask=nd_mask, other=0.0).to(tl.float32)

        # Load gamma
        gamma_off = (n_idx * stride_gamma_n + d_idx * stride_gamma_d)
        gamma_vals = tl.load(gamma_ptr + gamma_off, mask=nd_mask, other=0.0)

        # x * gamma, accumulate
        x_gamma = x_vals * gamma_vals
        acc += dh_mix_val * x_gamma

    # Atomic add to dphi (in case multiple programs write to the same location - though they shouldn't)
    dphi_off = (out_idx * stride_phi_out + nD_idx * stride_phi_in)
    tl.atomic_add(dphi_ptr + dphi_off, acc, mask=nd_mask)


@triton.jit
def mhc_backward_dgamma_kernel(
    # Input pointers
    x_ptr,
    dvecX_mm_ptr,

    # Output pointer
    dgamma_ptr,

    # Scalar parameters
    B, S, n, D, nD,

    # Strides
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_dvecxmm_b, stride_dvecxmm_s, stride_dvecxmm_d,

    # Block sizes
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute dgamma = sum_{B,S} x * dvecX_mm

    dgamma[n*D + d] = sum_{b,s} x[b, s, n, d] * dvecX_mm[b, s, n*D + d]

    Grid is (n, D // BLOCK_SIZE_K), each program handles one (n, d_block).
    """
    n_idx = tl.program_id(axis=0)
    d_block_idx = tl.program_id(axis=1)

    n_mask = n_idx < n

    d_off = d_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    d_mask = d_off < D

    # Accumulator: [BLOCK_SIZE_K]
    acc = tl.zeros([BLOCK_SIZE_K], dtype=tl.float32)

    # Iterate over all batch and sequence elements
    for bs_idx in range(B * S):
        b_idx = bs_idx // S
        s_idx = bs_idx % S

        # Load x block (single row for this n_idx)
        x_off = (b_idx * stride_x_b + s_idx * stride_x_s +
                 n_idx * stride_x_n +
                 d_off * stride_x_d)
        x_vals = tl.load(x_ptr + x_off, mask=(n_mask & d_mask), other=0.0).to(tl.float32)

        # Load dvecX_mm block (same slice)
        dvecX_mm_off = (b_idx * stride_dvecxmm_b + s_idx * stride_dvecxmm_s +
                        n_idx * D + d_off)
        dvecX_mm_vals = tl.load(dvecX_mm_ptr + dvecX_mm_off, mask=(n_mask & d_mask), other=0.0)

        # Accumulate
        acc += x_vals * dvecX_mm_vals

    # Store to dgamma (1D array with stride 1)
    dgamma_off = (n_idx * D + d_off)
    tl.store(dgamma_ptr + dgamma_off, acc, mask=(n_mask & d_mask))


def mhc_backward_triton(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """
    MHC Backward - Triton implementation with multi-kernel architecture.

    Uses 4 kernels for correct gradient computation:
    1. Main kernel: dalpha, dbias, dvecX_mm
    2. DX kernel: dx = dvecX_mm @ gamma.T + dvecX_hin
    3. Dphi kernel: dphi = dh_mix.T @ (x * gamma)
    4. Dgamma kernel: dgamma = sum x * dvecX_mm
    """
    B, S, n, D = x.shape
    nD = n * D
    out_features = n * n + 2 * n

    # Ensure contiguous
    x = x.contiguous()
    phi = phi.contiguous()
    alpha = alpha.contiguous()
    bias = bias.contiguous()
    inv_rms = inv_rms.contiguous()
    h_mix = h_mix.contiguous()
    h_pre = h_pre.contiguous()
    h_post = h_post.contiguous()
    dh_in = dh_in.contiguous()
    dh_post = dh_post.contiguous()
    dh_res = dh_res.contiguous()
    gamma = gamma.contiguous()

    # Allocate outputs
    dx = torch.zeros_like(x)
    dphi = torch.zeros_like(phi)
    dalpha = torch.zeros_like(alpha)
    dbias = torch.zeros_like(bias)
    dgamma = torch.zeros(n * D, dtype=gamma.dtype, device=gamma.device)

    # Allocate intermediate variables
    dvecX_mm = torch.zeros(B, S, nD, dtype=torch.float32, device=x.device)
    dvecX_inv = torch.zeros(B, S, n, D, dtype=torch.float32, device=x.device)

    # Also need dh_mix for dphi computation (kernel 3)
    # NOTE: This is the gradient dh_mix, NOT the forward h_mix!
    a_pre, a_post, a_res = alpha[0], alpha[1], alpha[2]
    x_fp = x.float()

    # Compute dh_pre, dh_post, dh_res
    h_in_fp_grad = dh_in.float()
    dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)

    hc_eps_tensor = torch.tensor(hc_eps, device=x.device)
    s_pre2 = h_pre - hc_eps_tensor
    dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
    dh_post2 = dh_post * h_post * (1.0 - h_post / 2)

    dh_pre1 = a_pre * dh_pre2
    dh_post1 = a_post * dh_post2
    dh_res1 = a_res * dh_res

    # This is dh_mix for dphi kernel (already has inv_rms applied)
    dh_mix = torch.cat([dh_pre1, dh_post1, dh_res1.reshape(B, S, n * n)], dim=-1) * inv_rms[:, :, None]

    # Block sizes
    # BLOCK_SIZE_N: for n dimension (h_pre, h_post, etc.)
    # BLOCK_SIZE_K: for D dimension (x, dh_in, etc.) - must be >= D!
    BLOCK_SIZE_N = triton.next_power_of_2(n)
    BLOCK_SIZE_K = triton.next_power_of_2(D)  # FIX: Use D, not min(D, nD, out_features)

    # ============================================================
    # Kernel 1: Main backward (dalpha, dbias, dvecX_mm)
    # ============================================================
    grid1 = (B * S,)

    with torch.cuda.device(x.device.index if x.is_cuda else 0):
        mhc_backward_kernel[grid1](
            x_ptr=x,
            phi_ptr=phi,
            alpha_ptr=alpha,
            bias_ptr=bias,
            inv_rms_ptr=inv_rms,
            h_mix_ptr=h_mix,  # Use forward h_mix, not dh_mix!
            h_pre_ptr=h_pre,
            h_post_ptr=h_post,
            dh_in_ptr=dh_in,
            dh_post_ptr=dh_post,
            dh_res_ptr=dh_res,
            gamma_ptr=gamma,

            dalpha_ptr=dalpha,
            dbias_ptr=dbias,
            dvecX_mm_ptr=dvecX_mm,
            dvecX_inv_ptr=dvecX_inv,

            B=B, S=S, n=n, D=D,
            nD=nD,
            out_features=out_features,

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

            stride_dvecxmm_b=S * nD,
            stride_dvecxmm_s=nD,
            stride_dvecxmm_d=1,

            stride_dvecinv_b=S * n * D,
            stride_dvecinv_s=n * D,
            stride_dvecinv_n=D,
            stride_dvecinv_d=1,

            norm_eps=norm_eps,
            hc_eps=hc_eps,

            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

        # DEBUG: Check dalpha after kernel 1 (disabled)
        # torch.cuda.synchronize()
        # print(f"[DEBUG] After kernel 1, dalpha = {dalpha.cpu().numpy()}")

        # ============================================================
        # Kernel 2: Compute dx
        # ============================================================
        # FIXED: grid2 should be (B*S, n) to cover all n_idx values
        # Previous grid: (B*S, 1) only processed n_idx=0
        grid2 = (B * S, n)

        mhc_backward_dx_kernel[grid2](
            dvecX_mm_ptr=dvecX_mm,
            dvecX_inv_ptr=dvecX_inv,
            gamma_ptr=gamma,
            h_pre_ptr=h_pre,
            dh_in_ptr=dh_in,

            dx_ptr=dx,

            B=B, S=S, n=n, D=D, nD=nD,

            stride_dvecxmm_b=S * nD,
            stride_dvecxmm_s=nD,
            stride_dvecxmm_d=1,

            stride_dvecinv_b=S * n * D,
            stride_dvecinv_s=n * D,
            stride_dvecinv_n=D,
            stride_dvecinv_d=1,

            stride_gamma_n=D,
            stride_gamma_d=1,

            stride_hin_b=S * D,
            stride_hin_s=D,
            stride_hin_d=1,

            stride_hpost_b=S * n,
            stride_hpost_s=n,
            stride_hpost_n=1,

            stride_x_b=S * n * D,
            stride_x_s=n * D,
            stride_x_n=D,
            stride_x_d=1,

            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

        # ============================================================
        # Kernel 3: Compute dphi
        # ============================================================
        grid3 = (out_features, triton.cdiv(nD, BLOCK_SIZE_K))

        # torch.cuda.synchronize()
        # print(f"[DEBUG] After kernel 2 (dx), dalpha = {dalpha.cpu().numpy()}")

        # ============================================================
        # Kernel 3: Compute dphi
        # ============================================================
        grid3 = (out_features, triton.cdiv(nD, BLOCK_SIZE_K))

        mhc_backward_dphi_kernel[grid3](
            dh_mix_ptr=dh_mix,
            x_ptr=x,
            gamma_ptr=gamma,

            dphi_ptr=dphi,

            B=B, S=S, n=n, D=D, nD=nD,
            out_features=out_features,

            stride_dmix_b=S * out_features,
            stride_dmix_s=out_features,
            stride_dmix_out=1,

            stride_x_b=S * n * D,
            stride_x_s=n * D,
            stride_x_n=D,
            stride_x_d=1,

            stride_gamma_n=D,
            stride_gamma_d=1,

            stride_phi_out=nD,
            stride_phi_in=1,

            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

        # torch.cuda.synchronize()
        # print(f"[DEBUG] After kernel 3 (dphi), dalpha = {dalpha.cpu().numpy()}")

        # ============================================================
        # Kernel 4: Compute dgamma
        # ============================================================
        grid4 = (n, triton.cdiv(D, BLOCK_SIZE_K))

        mhc_backward_dgamma_kernel[grid4](
            x_ptr=x,
            dvecX_mm_ptr=dvecX_mm,

            dgamma_ptr=dgamma,

            B=B, S=S, n=n, D=D, nD=nD,

            stride_x_b=S * n * D,
            stride_x_s=n * D,
            stride_x_n=D,
            stride_x_d=1,

            stride_dvecxmm_b=S * nD,
            stride_dvecxmm_s=nD,
            stride_dvecxmm_d=1,

            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

    # DEBUG: Check final dalpha before returning
    # print(f"[DEBUG] Final dalpha[0] = {dalpha[0]:.8f}")

    return dx, dphi, dalpha, dbias, dgamma
