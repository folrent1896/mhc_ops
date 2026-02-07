"""
MHC Forward Pre - Golden Reference Implementation

This is the reference implementation for correctness verification.
"""

import torch


def mhc_forward_pre(x, phi, alpha, bias,
                    outflag: bool = False, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
    """
    MHC Forward Pre operator - Reference implementation.

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

    # Flatten
    vecX = x.reshape(B, S, nD).float()

    # GEMM
    h_mix = torch.matmul(vecX, phi.t())

    # RMSNorm
    inv_rms = torch.rsqrt(vecX.square().mean(-1, keepdim=True) + norm_eps)
    h_mix_tmp = h_mix * inv_rms

    # Split
    h_pre1, h_post1, h_res1 = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
    h_res2 = h_res1.reshape(B, S, n, n)
    a_pre, a_post, a_res = alpha

    # Apply alpha and bias
    h_pre2 = a_pre * h_pre1 + bias[:n]
    h_post2 = a_post * h_post1 + bias[n:2*n]
    h_res = a_res * h_res2 + bias[2*n:].view(n, n)

    # Sigmoid activation
    h_pre = torch.sigmoid(h_pre2) + hc_eps
    h_post = 2.0 * torch.sigmoid(h_post2)

    # Compute h_in = h_pre @ x
    h_in_fp = (h_pre.unsqueeze(-1) * x.float()).sum(dim=2)
    h_in = h_in_fp.to(torch.bfloat16)

    if not outflag:
        return h_in, h_post, h_res
    else:
        inv_rms = inv_rms.squeeze(-1)
        return h_in, h_post, h_res, inv_rms, h_mix, h_pre


def mhc_pre_backward_manual(x, phi, alpha, bias,
                             inv_rms, h_mix, h_pre, h_post,
                             dh_in, dh_post, dh_res, gamma,
                             norm_eps: float = 1e-6, hc_eps: float = 1e-6):
    """
    Manual backward pass for gradient verification.

    Args:
        x: [B, S, n, D] - BFloat16 input
        phi: [n^2 + 2n, nD] - Float32 weight matrix
        alpha: [3] - Float32 scaling factors
        bias: [n^2 + 2n] - Float32 bias
        inv_rms: [B, S] - Forward pass intermediate
        h_mix: [B, S, n^2 + 2n] - Forward pass intermediate
        h_pre: [B, S, n] - Forward pass intermediate
        h_post: [B, S, n] - Forward pass intermediate
        dh_in: [B, S, D] - Gradient of h_in
        dh_post: [B, S, n] - Gradient of h_post
        dh_res: [B, S, n, n] - Gradient of h_res
        gamma: [n, D] - Optional scaling

    Returns:
        dx: [B, S, n, D] - BFloat16
        dphi: [n^2 + 2n, nD] - Float32
        dalpha: [3] - Float32
        dbias: [n^2 + 2n] - Float32
        dgamma: [n, D] - Float32
    """
    B, S, n, D = x.shape
    nD = n * D
    vecX = x.reshape(B, S, nD).float()
    a_pre, a_post, a_res = alpha

    dx = torch.zeros_like(x)
    dphi = torch.zeros_like(phi)
    dalpha = torch.zeros_like(alpha)
    dbias = torch.zeros_like(bias)

    # Gradient through sigmoid/linear
    x_fp = x.float()
    h_in_fp_grad = dh_in.float()
    dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)

    s_pre2 = h_pre - hc_eps
    dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
    dh_post2 = dh_post * h_post * (1.0 - h_post / 2)

    dh_pre1 = a_pre * dh_pre2
    dh_post1 = a_post * dh_post2
    dh_res1 = a_res * dh_res

    dh_mix_tmp = torch.cat([dh_pre1, dh_post1, dh_res1.reshape(B, S, n * n)], dim=-1)
    dh_mix = dh_mix_tmp * inv_rms[:, :, None]

    # Gradient w.r.t. x
    dvecX_mm = torch.matmul(dh_mix, phi)

    # Gradient w.r.t. phi
    xrs = x_fp.reshape(B*S, n*D) * gamma.reshape(1, n*D)
    dphi = torch.matmul(dh_mix.reshape(B*S, n*n + 2*n).t(), xrs)

    # Gradient w.r.t. inv_rms
    h_mix_tmp = h_mix * inv_rms[:, :, None]
    h_pre1, h_post1, h_res1 = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
    dinv_rms = (dh_mix_tmp * h_mix).sum(dim=-1, keepdim=True)

    # Gradient w.r.t. alpha
    dalpha_pre = (dh_pre2 * h_pre1).sum()
    dalpha_post = (dh_post2 * h_post1).sum()
    dalpha_res = (dh_res.reshape(B, S, n * n) * h_res1).sum()
    dalpha = torch.stack([dalpha_pre, dalpha_post, dalpha_res])

    # Gradient w.r.t. bias
    dbias_pre = dh_pre2.sum(dim=(0, 1))
    dbias_post = dh_post2.sum(dim=(0, 1))
    dbias_res = dh_res.reshape(B, S, n * n).sum(dim=(0, 1))
    dbias = torch.cat([dbias_pre, dbias_post, dbias_res], dim=0)

    # Gradient w.r.t. x (continued)
    dvecX_inv = -(dinv_rms * inv_rms[:, :, None].pow(3) / nD) * vecX
    dvecX_hin = h_pre.unsqueeze(-1) * dh_in.unsqueeze(2)
    dvecX_inv = dvecX_inv.reshape(B, S, n, D) + dvecX_hin

    dgamma = (x_fp.reshape(B * S, n * D) * dvecX_mm.reshape(B * S, n * D)).sum(dim=-2)
    dx = dvecX_mm.reshape(B, S, n, D) * gamma.reshape(1, 1, n, D) + dvecX_inv

    return dx.to(torch.bfloat16), dphi, dalpha, dbias, dgamma
