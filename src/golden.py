import argparse
import torch
import math
import pandas as pd
import omni_training_custom_ops
import pytest

# ============================================================
# Forward 小算子拼接的正向传播
# ============================================================
def mhc_forward_pre(x, phi, alpha, bias, 
                outflag: bool=False, norm_eps: float=1e-6, hc_eps: float=1e-6):
    """
    x     : [B, S, n, D]                BFloat16
    phi   : [n^2 + 2n, nD]              Float32
    alpha : [3] -> [pre, post, res]     Float32
    bias  : [n^2 + 2n]                  Float32

    returns:
        h_in   : [B, S, D]                     BFloat16
        h_post : [B, S, n]                     Float32
        h_res  : [B, S, D, n]                  Float32
    """
    # ---- shape ----
    B, S, n, D = x.shape
    nD = n * D

    # ---- flatten ----
    # [B, S, nD]
    vecX = x.reshape(B, S, nD).float()

    # ---- GEMM ----
    # [B, S, n^2 + 2n]
    h_mix = torch.matmul(vecX, phi.t())

    # ---- RMSNorm (delayed) ----
    # [B, S, 1]
    inv_rms = torch.rsqrt(vecX.square().mean(-1, keepdim=True) + norm_eps)
    # [B, S, n^2 + 2n]
    h_mix_tmp = h_mix * inv_rms
    # [B, S, n], [B, S, n], [B, S, n*n]
    h_pre1, h_post1, h_res1 = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
    # [B, S, n, n]
    h_res2 = h_res1.reshape(B, S, n, n)
    a_pre, a_post, a_res = alpha

    # ---- split ----
    h_pre2  = a_pre * h_pre1    + bias[:n]                 # [B, S, n]
    h_post2 = a_post * h_post1   + bias[n:2*n]              # [B, S, n]
    h_res  = a_res   * h_res2    + bias[2*n:].view(n, n)    # [B, S, n, n]

    # ---- nonlinear ----
    h_pre  = torch.sigmoid(h_pre2) + hc_eps                 # [B, S, n]
    h_post = 2.0 * torch.sigmoid(h_post2)                   # [B, S, n]

    # ---- h_in = h_pre @ x ----
    # [B, S, D] = [B, S, n, 1] * [B, S, n, D]
    h_in_fp = (h_pre.unsqueeze(-1) * x.float()).sum(dim=2)
    h_in = h_in_fp.to(torch.bfloat16)

    if not outflag:
        # [B, S, D],  [B, S, n],  [B, S, n, n]
        return h_in, h_post, h_res 
    else:
        inv_rms = inv_rms.squeeze(-1) # [B, S]
        # ..., [B, S],  [B, S, n^2 + 2n],  [B, S, n]
        return h_in, h_post, h_res, inv_rms, h_mix, h_pre


# ============================================================
# Manual Backward： 人工的反向传播
# ============================================================
def mhc_pre_backward_manual(x, phi, alpha, bias,
                             inv_rms, h_mix, h_pre, h_post,
                             dh_in, dh_post, dh_res, gamma, 
                             norm_eps: float=1e-6, hc_eps: float=1e-6):
    """
    input:
        x     : [B, S, n, D]                BFloat16
        phi   : [n^2 + 2n, nD]              Float32
        alpha : [3] -> [pre, post, res]     Float32
        bias  : [n^2 + 2n]                  Float32
    forward outputs:
        inv_rms : [B, S]                    Float32
        h_mix   : [B, S, n^2 + 2n]          Float32
        h_pre   : [B, S, N]                 Float32
        h_post  : [B, S, N]                 Float32
    grad inputs:
        dh_in   : [B, S, D]                 BFloat16
        dh_post : [B, S, n]                 Float32
        dh_res  : [B, S, D, n]              Float32
    optional input:
        gamma : [n, D]                      Float32

    output:
        dx      : [B, S, N, D]              BFloat16
        dphi    : [n^2 + 2n, nD]            Float32
        dalpha  : [3]                       Float32
        dbias   : [n^2 + 2n]                Float32
        dgamma  : [N, D]                    Float32
    """

    B, S, n, D = x.shape
    nD = n * D
    vecX = x.reshape(B, S, nD).float()
    a_pre, a_post, a_res = alpha

    dx = torch.zeros_like(x)
    dphi = torch.zeros_like(phi)
    dalpha = torch.zeros_like(alpha)
    dbias = torch.zeros_like(bias)

    # ========================================================
    # ---- 计算导数 ----
    # (V0) 反推sigmod/线性变换梯度
    # ========================================================
    x_fp = x.float() # [B, S, n, D]
    h_in_fp_grad = dh_in.float() # [B, S, D]
    dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1) # [B, S, n]
    
    s_pre2 = h_pre - hc_eps                                     # [B, S, n]   
    dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)                  # [B, S, n]
    dh_post2 = dh_post * h_post * (1.0 - h_post / 2)            # [B, S, n]

    dh_pre1 = a_pre * dh_pre2                                   # [B, S, n]
    dh_post1 = a_post * dh_post2                                # [B, S, n]
    dh_res1 = a_res * dh_res                                    # [B, S, n, n]

    dh_mix_tmp = torch.cat([dh_pre1, dh_post1, dh_res1.reshape(B, S, n * n)], dim=-1)   # [B, S, n^2 + 2n]
    dh_mix = dh_mix_tmp * inv_rms[:, :, None]                   # [B, S, n^2 + 2n]

    # ========================================================
    # (C0)
    # ========================================================
    dvecX_mm = torch.matmul(dh_mix, phi)                     # [B, S, nD]

    # ========================================================
    # (C1)
    # ========================================================
    xrs = x_fp.reshape(B*S, n*D) * gamma.reshape(1, n*D)            # [BS, nD]
    dphi = torch.matmul(dh_mix.reshape(B*S, n*n + 2*n).t(), xrs)    # [n^2 + 2n, nD]

    # ========================================================
    # (V1)
    # ========================================================
    h_mix_tmp = h_mix * inv_rms[:, :, None]                         # [B, S, n^2 + 2n]
    h_pre1, h_post1, h_res1 = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
    dinv_rms = (dh_mix_tmp * h_mix).sum(dim=-1, keepdim=True)       # [B, S, 1]

    dalpha_pre = (dh_pre2 * h_pre1).sum()
    dalpha_post = (dh_post2 * h_post1).sum()
    dalpha_res = (dh_res.reshape(B, S, n * n) * h_res1).sum()
    dalpha = torch.stack([dalpha_pre, dalpha_post, dalpha_res])

    dbias_pre = dh_pre2.sum(dim=(0, 1))                         # [n]
    dbias_post = dh_post2.sum(dim=(0, 1))                       # [n]
    dbias_res = dh_res.reshape(B, S, n * n).sum(dim=(0, 1))     # [n*n]
    dbias = torch.cat([dbias_pre, dbias_post, dbias_res], dim=0)

    # ========================================================
    # (V2)
    # ========================================================
    dvecX_inv = -(dinv_rms * inv_rms[:, :, None].pow(3) / nD) * vecX    # [B, S, nD]
    dvecX_hin = h_pre.unsqueeze(-1) * dh_in.unsqueeze(2)                # [B, S, n, D]
    dvecX_inv = dvecX_inv.reshape(B, S, n, D) + dvecX_hin               # [B, S, n, D]

    dgamma = (x_fp.reshape(B * S, n * D) * dvecX_mm.reshape(B * S, n * D)).sum(dim=-2)
    dx = dvecX_mm.reshape(B, S, n, D) * gamma.reshape(1, 1, n, D) + dvecX_inv
    return dx.to(torch.bfloat16), dphi, dalpha, dbias, dgamma

# ============================================================
# PrintMat
# ============================================================
def print_mat(cm, desc):
    pass
    print("Print Tensor: ", desc)
    df = pd.DataFrame(cm[:5, :5].detach().numpy())
    print(df.to_string())

# ============================================================
# Verify
# ============================================================
def verify_result(output, golden, tol = 1e-3):
    # print_mat(golden, "golden")
    # print_mat(output, "npu_out")
    output = output.reshape(-1)
    golden = golden.reshape(-1)

    different_element_results = torch.isclose(output, golden, rtol=tol, atol=tol, equal_nan=True)
    different_element_indexes = torch.where(different_element_results == False)[0]

    for index in range(min(len(different_element_indexes), 10)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]

        print(
            "data index %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6d" % (real_index.item(), golden_data.item(), output_data.item(),
                abs(output_data - golden_data) / golden_data))
    
    error_ratio = float(different_element_indexes.size(0)) / float(golden.size(0))
    print("error ratio: %.6f, tolerance: %.6f" % (error_ratio, tol))
    return error_ratio <= 1e-4

# ============================================================
# Check
# ============================================================
@pytest.mark.resources(device="npu:910B", npus_per_node=1)
def test_mhc_pre_grad_case():
    # 初始化
    B=1
    S=4096
    n=4
    D=2560
    x = torch.randn(B, S, n, D).bfloat16()
    phi = torch.randn(n*n + 2*n, n*D)
    alpha = torch.tensor([1.1, 0.9, 1.05])
    bias = torch.randn(n*n + 2*n) * 0.1
    gamma = torch.randn(n, D)

    x_ = x.detach().clone().requires_grad_(True)
    phi_ = phi.detach().clone().requires_grad_(True)
    alpha_ = alpha.detach().clone().requires_grad_(True)
    bias_ = bias.detach().clone().requires_grad_(True)

    # 随机初始化梯度
    dh_in = torch.randn(B, S, D).bfloat16()
    dh_post = torch.randn(B, S, n)
    dh_res = torch.randn(B, S, n, n)

    # 正向传播
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
        x_, phi_, alpha_, bias_, outflag=True
    )

    # 手动微分
    dx0, dphi0, da0, db0, dgamma0 = mhc_pre_backward_manual(
        x_, phi_, alpha_, bias_,
        inv_rms, h_mix, h_pre, h_post,
        dh_in, dh_post, dh_res, gamma
    )

    res = True
    for i in range(5):
        # 调用npu接口
        dx1, dphi1, da1, db1, dgamma1 = torch.ops.custom.npu_manifold_constrained_hyper_connection_pre_grad(
            x.npu(), phi.npu(), alpha.npu(),
            dh_in.npu(), dh_post.npu(), dh_res.npu(),
            inv_rms.npu(), h_mix.npu(), h_pre.npu(), h_post.npu(), gamma=gamma.npu()
        )
        print(f"dphi1.shape = {dphi1.shape}")
        print("=============dx================")
        resflag_dx = verify_result(dx1.reshape(B*S, n*D).float().cpu(), dx0.reshape(B*S, n*D).float().cpu(), 2**-7)
        print("=============dphi================")
        resflag_dphi = verify_result(dphi1.reshape(n * n + 2 * n, n*D).float().cpu(), dphi0.reshape(n * n + 2 * n, n*D).float().cpu())
        print("=============dalpha================")
        resflag_da = verify_result(da1.float().cpu(), da0.float().cpu())
        print("=============dbias================")
        resflag_db = verify_result(db1.float().cpu(), db0.float().cpu())
        print("=============dgamma================")
        resflag_dgamma = verify_result(dgamma1.float().cpu().reshape(n, D), dgamma0.float().cpu().reshape(n, D))
        if not (resflag_dx and resflag_dphi and resflag_da and resflag_db and resflag_dgamma):
            res = False
    if res:
        assert True
    else:
        assert False

if __name__ == "__main__":
    test_mhc_pre_grad_case()