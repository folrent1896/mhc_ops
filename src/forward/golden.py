"""
MHC Forward Pre - Golden Reference Implementation

This is the reference forward implementation for correctness verification.
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

    If outflag=True, also returns:
        inv_rms: [B, S] - Float32
        h_mix: [B, S, n^2 + 2n] - Float32
        h_pre: [B, S, n] - Float32
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
