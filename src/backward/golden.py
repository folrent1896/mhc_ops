"""
MHC Backward - Golden Reference Implementation

This is the reference backward implementation for gradient verification.
"""

import torch


def mhc_backward_manual(x, phi, alpha, bias,
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
