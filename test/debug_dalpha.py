"""
Debug script for dalpha_pre precision issue

This script helps identify where the dalpha_pre calculation goes wrong
by comparing intermediate values between Golden and Triton implementations.
"""

import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, '.')

from src.forward.golden import mhc_forward_pre
from src.backward.golden import mhc_backward_manual


@triton.jit
def debug_dh_pre2_kernel(
    x_ptr,
    dh_in_ptr,
    h_pre_ptr,
    dh_pre2_ptr,  # Output: [B, S, n]
    hc_eps: tl.float32,
    B, S, n, D,
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_hin_b, stride_hin_s, stride_hin_d,
    stride_hpost_b, stride_hpost_s, stride_hpost_n,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Debug kernel to compute dh_pre2 and output for comparison
    """
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    # Load x [n, D]
    x_off_n = tl.arange(0, BLOCK_SIZE_N)
    x_off_d = tl.arange(0, BLOCK_SIZE_K)
    x_mask = (x_off_n[:, None] < n) & (x_off_d[None, :] < D)
    x_offset = (b_idx * stride_x_b + s_idx * stride_x_s +
                x_off_n[:, None] * stride_x_n +
                x_off_d[None, :] * stride_x_d)
    x_block = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0).to(tl.float32)

    # Load dh_in [D]
    dh_in_off = x_off_d
    dh_in_mask = x_off_d < D
    dh_in_offset = (b_idx * stride_hin_b + s_idx * stride_hin_s +
                     dh_in_off * stride_hin_d)
    dh_in = tl.load(dh_in_ptr + dh_in_offset, mask=dh_in_mask, other=0.0)

    # Load h_pre [n]
    h_pre_off = (b_idx * stride_hpost_b + s_idx * stride_hpost_s + x_off_n)
    h_pre_mask = x_off_n < n
    h_pre = tl.load(h_pre_ptr + h_pre_off, mask=h_pre_mask, other=0.0)

    # Compute dh_pre = sum(x * dh_in, axis=1)
    dh_pre = tl.sum(x_block * dh_in[None, :], axis=1)

    # Compute dh_pre2 = dh_pre * s_pre2 * (1 - s_pre2)
    s_pre2 = h_pre - hc_eps
    dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)

    # Store output
    dh_pre2_offset = (b_idx * S * n + s_idx * n + x_off_n)
    tl.store(dh_pre2_ptr + dh_pre2_offset, dh_pre2, mask=h_pre_mask)


@triton.jit
def debug_h_pre1_hmix_kernel(
    h_mix_ptr,
    inv_rms_ptr,
    h_pre1_hmix_ptr,  # Output: [B, S, n]
    B, S, n, out_features,
    stride_hmix_b, stride_hmix_s, stride_hmix_out,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Debug kernel to compute h_pre1_hmix and output for comparison
    """
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    x_off_n = tl.arange(0, BLOCK_SIZE_N)
    h_pre_mask = x_off_n < n

    # Load inv_rms
    inv_rms_off = b_idx * S + s_idx
    inv_rms = tl.load(inv_rms_ptr + inv_rms_off)

    # Load h_mix[0:n]
    h_mix_off = (b_idx * stride_hmix_b + s_idx * stride_hmix_s + x_off_n)
    h_mix_vals = tl.load(h_mix_ptr + h_mix_off, mask=h_pre_mask, other=0.0)

    # h_pre1_hmix = h_mix * inv_rms
    h_pre1_hmix = h_mix_vals * inv_rms

    # Store output
    h_pre1_hmix_offset = (b_idx * S * n + s_idx * n + x_off_n)
    tl.store(h_pre1_hmix_ptr + h_pre1_hmix_offset, h_pre1_hmix, mask=h_pre_mask)


@triton.jit
def debug_dalpha_pre_contribution_kernel(
    dh_pre2_ptr,
    h_pre1_hmix_ptr,
    contribution_ptr,  # Output: [B, S, n]
    B, S, n,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Debug kernel to compute per-element contribution to dalpha_pre
    contribution[b,s,n] = dh_pre2[b,s,n] * h_pre1_hmix[b,s,n]
    """
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    x_off_n = tl.arange(0, BLOCK_SIZE_N)
    mask = x_off_n < n

    # Load dh_pre2[b, s, :]
    dh_pre2_offset = (b_idx * S * n + s_idx * n + x_off_n)
    dh_pre2_vals = tl.load(dh_pre2_ptr + dh_pre2_offset, mask=mask, other=0.0)

    # Load h_pre1_hmix[b, s, :]
    h_pre1_hmix_offset = (b_idx * S * n + s_idx * n + x_off_n)
    h_pre1_hmix_vals = tl.load(h_pre1_hmix_ptr + h_pre1_hmix_offset, mask=mask, other=0.0)

    # Compute contribution
    contribution = dh_pre2_vals * h_pre1_hmix_vals

    # Store
    contribution_offset = (b_idx * S * n + s_idx * n + x_off_n)
    tl.store(contribution_ptr + contribution_offset, contribution, mask=mask)


def debug_dalpha_pre():
    """Main debug function"""

    print("=" * 70)
    print("dalpha_pre 精度问题诊断")
    print("=" * 70)

    # Setup
    B, S, n, D = 2, 64, 4, 128
    device = 'cuda'
    hc_eps = 1e-6

    torch.manual_seed(42)
    x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
    phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
    alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32, device=device)
    bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1
    gamma = torch.randn(n, D, dtype=torch.float32, device=device)

    # Forward pass
    with torch.no_grad():
        h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
            x, phi, alpha, bias, outflag=True
        )

    # Gradients
    dh_in = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
    dh_post = torch.randn(B, S, n, dtype=torch.float32, device=device)
    dh_res = torch.randn(B, S, n, n, dtype=torch.float32, device=device)

    out_features = n * n + 2 * n

    # ============================================================
    # Step 1: Compute Golden intermediate values
    # ============================================================
    print("\n[1/5] Computing Golden reference intermediate values...")

    # Golden dh_pre2
    x_fp = x.float()
    h_in_fp_grad = dh_in.float()
    dh_pre_gold = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
    hc_eps_tensor = torch.tensor(hc_eps, device=x.device)
    s_pre2_gold = h_pre - hc_eps_tensor
    dh_pre2_gold = dh_pre_gold * s_pre2_gold * (1.0 - s_pre2_gold)

    # Golden h_pre1_hmix
    h_mix_tmp = h_mix * inv_rms[:, :, None]
    h_pre1_hmix_gold = h_mix_tmp[:, :, 0:n]

    # Golden contribution
    contribution_gold = dh_pre2_gold * h_pre1_hmix_gold
    dalpha_pre_gold = contribution_gold.sum()

    print(f"  Golden dh_pre2 shape: {dh_pre2_gold.shape}")
    print(f"  Golden h_pre1_hmix shape: {h_pre1_hmix_gold.shape}")
    print(f"  Golden dalpha_pre: {dalpha_pre_gold:.6f}")

    # ============================================================
    # Step 2: Run debug kernels
    # ============================================================
    print("\n[2/5] Running Triton debug kernels...")

    # Allocate outputs
    dh_pre2_tri = torch.zeros(B, S, n, dtype=torch.float32, device=device)
    h_pre1_hmix_tri = torch.zeros(B, S, n, dtype=torch.float32, device=device)
    contribution_tri = torch.zeros(B, S, n, dtype=torch.float32, device=device)

    # Ensure contiguous
    x = x.contiguous()
    dh_in = dh_in.contiguous()
    h_pre = h_pre.contiguous()
    h_mix = h_mix.contiguous()
    inv_rms = inv_rms.contiguous()

    BLOCK_SIZE_N = triton.next_power_of_2(n)
    BLOCK_SIZE_K = triton.next_power_of_2(D)

    with torch.cuda.device(x.device.index if x.is_cuda else 0):
        # Kernel 1: dh_pre2
        grid1 = (B * S,)
        debug_dh_pre2_kernel[grid1](
            x_ptr=x,
            dh_in_ptr=dh_in,
            h_pre_ptr=h_pre,
            dh_pre2_ptr=dh_pre2_tri,
            hc_eps=hc_eps,
            B=B, S=S, n=n, D=D,
            stride_x_b=S * n * D,
            stride_x_s=n * D,
            stride_x_n=D,
            stride_x_d=1,
            stride_hin_b=S * D,
            stride_hin_s=D,
            stride_hin_d=1,
            stride_hpost_b=S * n,
            stride_hpost_s=n,
            stride_hpost_n=1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

        # Kernel 2: h_pre1_hmix
        grid2 = (B * S,)
        debug_h_pre1_hmix_kernel[grid2](
            h_mix_ptr=h_mix,
            inv_rms_ptr=inv_rms,
            h_pre1_hmix_ptr=h_pre1_hmix_tri,
            B=B, S=S, n=n,
            out_features=out_features,
            stride_hmix_b=S * out_features,
            stride_hmix_s=out_features,
            stride_hmix_out=1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        # Kernel 3: contribution
        grid3 = (B * S,)
        debug_dalpha_pre_contribution_kernel[grid3](
            dh_pre2_ptr=dh_pre2_tri,
            h_pre1_hmix_ptr=h_pre1_hmix_tri,
            contribution_ptr=contribution_tri,
            B=B, S=S, n=n,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

    # ============================================================
    # Step 3: Compare intermediate values
    # ============================================================
    print("\n[3/5] Comparing intermediate values...")

    print("\ndh_pre2 comparison:")
    print(f"  Shape: Golden={dh_pre2_gold.shape}, Triton={dh_pre2_tri.shape}")
    print(f"  Max error:  {torch.abs(dh_pre2_tri - dh_pre2_gold).max().item():.8f}")
    print(f"  Mean error: {torch.abs(dh_pre2_tri - dh_pre2_gold).mean().item():.8f}")
    print(f"  Std (Golden): {dh_pre2_gold.std().item():.8f}")
    print(f"  Std (Triton): {dh_pre2_tri.std().item():.8f}")
    dh_pre2_match = torch.allclose(dh_pre2_tri, dh_pre2_gold, rtol=1e-4, atol=1e-4)
    print(f"  Match: {('✓ YES' if dh_pre2_match else '✗ NO')}")

    print("\nh_pre1_hmix comparison:")
    print(f"  Shape: Golden={h_pre1_hmix_gold.shape}, Triton={h_pre1_hmix_tri.shape}")
    print(f"  Max error:  {torch.abs(h_pre1_hmix_tri - h_pre1_hmix_gold).max().item():.8f}")
    print(f"  Mean error: {torch.abs(h_pre1_hmix_tri - h_pre1_hmix_gold).mean().item():.8f}")
    print(f"  Std (Golden): {h_pre1_hmix_gold.std().item():.8f}")
    print(f"  Std (Triton): {h_pre1_hmix_tri.std().item():.8f}")
    h_pre1_hmix_match = torch.allclose(h_pre1_hmix_tri, h_pre1_hmix_gold, rtol=1e-4, atol=1e-4)
    print(f"  Match: {('✓ YES' if h_pre1_hmix_match else '✗ NO')}")

    print("\ncontribution comparison:")
    print(f"  Shape: Golden={contribution_gold.shape}, Triton={contribution_tri.shape}")
    print(f"  Max error:  {torch.abs(contribution_tri - contribution_gold).max().item():.8f}")
    print(f"  Mean error: {torch.abs(contribution_tri - contribution_gold).mean().item():.8f}")
    contribution_match = torch.allclose(contribution_tri, contribution_gold, rtol=1e-4, atol=1e-4)
    print(f"  Match: {('✓ YES' if contribution_match else '✗ NO')}")

    # ============================================================
    # Step 4: Compare dalpha_pre accumulation
    # ============================================================
    print("\n[4/5] Comparing dalpha_pre accumulation...")

    # Triton: sum over all elements
    dalpha_pre_tri_sum = contribution_tri.sum()

    # Golden: already computed
    print(f"\nGolden dalpha_pre:  {dalpha_pre_gold:.8f}")
    print(f"Triton dalpha_pre:  {dalpha_pre_tri_sum:.8f}")
    print(f"Absolute error:     {abs(dalpha_pre_tri_sum - dalpha_pre_gold):.8f}")
    print(f"Relative error:     {abs(dalpha_pre_tri_sum - dalpha_pre_gold) / abs(dalpha_pre_gold) * 100:.2f}%")

    # ============================================================
    # Step 5: Sample element inspection
    # ============================================================
    print("\n[5/5] Sample element inspection (b=0, s=0, first 4 n values)...")

    for i in range(min(4, n)):
        print(f"\n  n={i}:")
        print(f"    dh_pre2_gold[{i}]      = {dh_pre2_gold[0, 0, i]:.8f}")
        print(f"    dh_pre2_tri[{i}]       = {dh_pre2_tri[0, 0, i]:.8f}")
        print(f"    h_pre1_hmix_gold[{i}]  = {h_pre1_hmix_gold[0, 0, i]:.8f}")
        print(f"    h_pre1_hmix_tri[{i}]   = {h_pre1_hmix_tri[0, 0, i]:.8f}")
        print(f"    contrib_gold[{i}]      = {contribution_gold[0, 0, i]:.8f}")
        print(f"    contrib_tri[{i}]       = {contribution_tri[0, 0, i]:.8f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)

    issues = []
    if not dh_pre2_match:
        issues.append("❌ dh_pre2计算错误")
    if not h_pre1_hmix_match:
        issues.append("❌ h_pre1_hmix加载错误")
    if not contribution_match:
        issues.append("❌ contribution计算错误")

    if len(issues) == 0:
        print("✓ 所有中间值都正确！")
        print("  问题可能在atomic_add的浮点精度上")
        print("  建议: 检查浮点累加顺序或使用更高精度")
    else:
        print("发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n  建议: 优先修复第一个标记为❌的问题")

    print("\n" + "=" * 70)

    return {
        'dh_pre2_match': dh_pre2_match,
        'h_pre1_hmix_match': h_pre1_hmix_match,
        'contribution_match': contribution_match,
    }


if __name__ == "__main__":
    debug_dalpha_pre()
