"""
MHC Backward Operator - Triton Implementation

This module implements the backward pass for MHC Forward Pre operator.
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
    dx_ptr,
    dphi_ptr,
    dalpha_ptr,
    dbias_ptr,
    dgamma_ptr,

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

    # Epsilons
    norm_eps: tl.float32,
    hc_eps: tl.float32,

    # Block sizes
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for MHC backward pass.

    Computes gradients for: x, phi, alpha, bias, gamma
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

    # Load h_pre [n]
    h_pre_off = (b_idx * stride_hpost_b + s_idx * stride_hpost_s +
                 x_off_n * stride_hpost_n)
    h_pre_mask = x_off_n < n
    h_pre = tl.load(h_pre_ptr + h_pre_off, mask=h_pre_mask, other=0.0)

    # Load h_post [n]
    h_post = tl.load(h_post_ptr + h_pre_off, mask=h_pre_mask, other=0.0)

    # Load dh_in [D]
    dh_in_off = x_off_d
    dh_in_mask = x_off_d < D
    dh_in_offset = (b_idx * stride_hin_b + s_idx * stride_hin_s +
                     dh_in_off * stride_hin_d)
    dh_in = tl.load(dh_in_ptr + dh_in_offset, mask=dh_in_mask, other=0.0)

    # Load dh_post [n]
    dh_post = tl.load(dh_post_ptr + h_pre_off, mask=h_pre_mask, other=0.0)

    # Load h_mix [out_features]
    h_mix_off = tl.arange(0, BLOCK_SIZE_K)
    h_mix_mask = h_mix_off < out_features
    h_mix_offset = (b_idx * S * out_features + s_idx * out_features + h_mix_off)
    h_mix = tl.load(h_mix_ptr + h_mix_offset, mask=h_mix_mask, other=0.0)

    # Load alpha [3]
    alpha = tl.load(alpha_ptr)
    a_pre, a_post, a_res = alpha[0], alpha[1], alpha[2]

    # ============================================================
    # Step 2: Compute dh_pre from dh_in
    # dh_pre = sum(dh_in * x, axis=D)
    # ============================================================
    dh_pre = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for i in range(n):
        dh_pre[i] = tl.sum(dh_in * x_block[i, :])

    # ============================================================
    # Step 3: Backward through sigmoid
    # dh_pre2 = dh_pre * sigmoid_grad
    # sigmoid_grad = s_pre2 * (1 - s_pre2)
    # ============================================================
    s_pre2 = h_pre - hc_eps
    dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)

    dh_post2 = dh_post * h_post * (1.0 - h_post / 2.0)

    # ============================================================
    # Step 4: Backward through alpha scaling
    # ============================================================
    dh_pre1 = a_pre * dh_pre2
    dh_post1 = a_post * dh_post2

    # ============================================================
    # Step 5: Accumulate gradients for alpha and bias
    # ============================================================
    # Load h_mix_tmp (normalized h_mix)
    h_mix_tmp = h_mix * inv_rms
    h_pre1 = h_mix_tmp[:n]
    h_post1 = h_mix_tmp[n:2*n]

    # Accumulate dalpha and dbias using atomics
    # dalpha_pre
    dalpha_pre_val = tl.sum(dh_pre2 * h_pre1)
    tl.atomic_add(dalpha_ptr, 0, dalpha_pre_val)

    # dalpha_post
    dalpha_post_val = tl.sum(dh_post2 * h_post1)
    tl.atomic_add(dalpha_ptr, 1, dalpha_post_val)

    # dalpha_res (need to load dh_res)
    # Load dh_res [n, n]
    dh_res_off_n1 = x_off_n
    dh_res_off_n2 = tl.arange(0, BLOCK_SIZE_N)
    dh_res_mask = (dh_res_off_n1[:, None] < n) & (dh_res_off_n2[None, :] < n)
    dh_res_offset = (b_idx * stride_hres_b + s_idx * stride_hres_s +
                     dh_res_off_n1[:, None] * stride_hres_n1 +
                     dh_res_off_n2[None, :] * stride_hres_n2)
    dh_res_block = tl.load(dh_res_ptr + dh_res_offset, mask=dh_res_mask, other=0.0)

    h_res1 = h_mix_tmp[2*n:].reshape(n, n)
    dalpha_res_val = tl.sum(dh_res_block * h_res1)
    tl.atomic_add(dalpha_ptr, 2, dalpha_res_val)

    # Accumulate dbias
    for i in range(BLOCK_SIZE_N):
        if i < n:
            tl.atomic_add(dbias_ptr, i, dh_pre2[i])
            tl.atomic_add(dbias_ptr, n + i, dh_post2[i])

    # dh_res1 and dbias_res
    dh_res1 = a_res * dh_res_block
    dh_res1_flat = dh_res1.flatten()
    for i in range(n * n):
        tl.atomic_add(dbias_ptr, 2 * n + i, dh_res1_flat[i])

    # ============================================================
    # Step 6: Compute dh_mix
    # ============================================================
    dh_mix_tmp = tl.concatenate([dh_pre1, dh_post1, dh_res1_flat], axis=0)
    dh_mix = dh_mix_tmp * inv_rms

    # ============================================================
    # Step 7: Compute dvecX_mm = dh_mix @ phi
    # ============================================================
    dvecX_mm = tl.zeros([nD], dtype=tl.float32)

    for k_start in range(0, out_features, BLOCK_SIZE_K):
        k = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k < out_features

        # Load phi chunk [out_features, nD]
        phi_off = (k[:, None] * stride_phi_out +
                   h_mix_off[None, :] * stride_phi_in)
        phi_mask = k_mask[:, None] & h_mix_mask[None, :]
        phi_chunk = tl.load(phi_ptr + phi_off, mask=phi_mask, other=0.0)

        # Accumulate: dvecX_mm += dh_mix[k] * phi[k, :]
        for i in range(BLOCK_SIZE_K):
            if k_mask[i]:
                dvecX_mm += dh_mix[i] * phi_chunk[i, :]

    # ============================================================
    # Step 8: Compute dvecX_inv (gradient through RMSNorm)
    # ============================================================
    dinv_rms = tl.sum(dh_mix_tmp * h_mix)
    dvecX_inv = -(dinv_rms * inv_rms * inv_rms * inv_rms / nD) * x_block.flatten()

    # ============================================================
    # Step 9: Compute dvecX_hin (gradient through h_in computation)
    # ============================================================
    dvecX_hin = tl.zeros([nD], dtype=tl.float32)
    idx = 0
    for i in range(n):
        for j in range(D):
            dvecX_hin[idx] = h_pre[i] * dh_in[j]
            idx += 1

    # ============================================================
    # Step 10: Compute dx
    # ============================================================
    dx_flat = dvecX_mm.reshape(n, D) * gamma.reshape(1, D) + dvecX_inv.reshape(n, D) + dvecX_hin.reshape(n, D)

    # Store dx [n, D]
    tl.store(x_ptr + x_offset, dx_flat, mask=x_mask)

    # ============================================================
    # Step 11: Accumulate dphi
    # dphi += dh_mix^T @ (x * gamma)
    # ============================================================
    x_scaled = x_block * gamma.reshape(1, D)
    x_scaled_flat = x_scaled.flatten()

    for out_idx in range(0, out_features, BLOCK_SIZE_K):
        out = out_idx + tl.arange(0, BLOCK_SIZE_K)
        out_mask = out < out_features

        for in_idx in range(0, nD, BLOCK_SIZE_K):
            inp = in_idx + tl.arange(0, BLOCK_SIZE_K)
            in_mask = inp < nD

            # Accumulate: dphi[out, in] += dh_mix[out] * x_scaled[in]
            val = dh_mix[out] * x_scaled_flat[inp]
            dphi_offset = (out[:, None] * stride_phi_out +
                          inp[None, :] * stride_phi_in)
            dphi_mask = out_mask[:, None] & in_mask[None, :]
            tl.atomic_add(dphi_ptr + dphi_offset, val, mask=dphi_mask)

    # ============================================================
    # Step 12: Accumulate dgamma
    # dgamma = x * dvecX_mm
    # ============================================================
    dgamma_block = x_block * dvecX_mm.reshape(n, D)
    for i in range(n):
        for j in range(D):
            tl.atomic_add(dgamma_ptr + i * D + j, dgamma_block[i, j])


def mhc_backward_triton(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """
    MHC Backward - Triton implementation.

    Args:
        x: [B, S, n, D] - BFloat16 input
        phi: [n^2 + 2n, nD] - Float32 weight matrix
        alpha: [3] - Float32 scaling factors
        bias: [n^2 + 2n] - Float32 bias
        inv_rms: [B, S] - Forward intermediate
        h_mix: [B, S, n^2 + 2n] - Forward intermediate
        h_pre: [B, S, n] - Forward intermediate
        h_post: [B, S, n] - Forward intermediate
        dh_in: [B, S, D] - Gradient of h_in
        dh_post: [B, S, n] - Gradient of h_post
        dh_res: [B, S, n, n] - Gradient of h_res
        gamma: [n, D] - Scaling factor
        norm_eps: RMSNorm epsilon
        hc_eps: Hyper connection epsilon

    Returns:
        dx: [B, S, n, D] - BFloat16
        dphi: [n^2 + 2n, nD] - Float32
        dalpha: [3] - Float32
        dbias: [n^2 + 2n] - Float32
        dgamma: [n, D] - Float32
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

    # Allocate outputs (initialize to zero)
    dx = torch.zeros_like(x)
    dphi = torch.zeros_like(phi)
    dalpha = torch.zeros_like(alpha)
    dbias = torch.zeros_like(bias)
    dgamma = torch.zeros_like(gamma)

    # Block sizes
    BLOCK_SIZE_N = triton.next_power_of_2(n)
    BLOCK_SIZE_K = triton.next_power_of_2(min(D, nD, out_features))

    # Grid
    grid = (B * S,)

    # Launch kernel
    with torch.cuda.device(x.device.index if x.is_cuda else 0):
        mhc_backward_kernel[grid](
            x_ptr=x,
            phi_ptr=phi,
            alpha_ptr=alpha,
            bias_ptr=bias,
            inv_rms_ptr=inv_rms,
            h_mix_ptr=h_mix,
            h_pre_ptr=h_pre,
            h_post_ptr=h_post,
            dh_in_ptr=dh_in,
            dh_post_ptr=dh_post,
            dh_res_ptr=dh_res,
            gamma_ptr=gamma,

            dx_ptr=dx,
            dphi_ptr=dphi,
            dalpha_ptr=dalpha,
            dbias_ptr=dbias,
            dgamma_ptr=dgamma,

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

            norm_eps=norm_eps,
            hc_eps=hc_eps,

            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

    return dx, dphi, dalpha, dbias, dgamma
