"""
Test dalpha_pre computation without atomic_add to rule out accumulation issues
"""

import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, '.')

from src.forward.golden import mhc_forward_pre

B, S, n, D = 2, 64, 4, 128
device = 'cuda'
hc_eps = 1e-6
nD = n * D
out_features = n * n + 2 * n

torch.manual_seed(42)
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device='cpu')
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device='cpu')
alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cpu') * 0.1

# Forward pass
with torch.no_grad():
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
        x, phi, alpha, bias, outflag=True
    )

# Gradients
torch.manual_seed(123)
dh_in = torch.randn(B, S, D, dtype=torch.bfloat16, device='cpu')

# Move to CUDA
x_dev = x.to(device)
alpha_dev = alpha.to(device)
inv_rms_dev = inv_rms.to(device)
h_mix_dev = h_mix.to(device)
h_pre_dev = h_pre.to(device)
dh_in_dev = dh_in.to(device)

# Kernel that computes per-program contribution instead of summing with atomic_add
@triton.jit
def dalpha_pre_no_atomic_kernel(
    x_ptr,
    alpha_ptr,
    inv_rms_ptr,
    h_mix_ptr,
    h_pre_ptr,
    dh_in_ptr,
    contribution_ptr,  # Output: [B, S, n] - contribution of each element
    B, S, n, D, nD, out_features,
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_hin_b, stride_hin_s, stride_hin_d,
    stride_hpost_b, stride_hpost_s, stride_hpost_n,
    stride_hmix_b, stride_hmix_s, stride_hmix_out,
    norm_eps: tl.float32,
    hc_eps: tl.float32,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    x_off_n = tl.arange(0, BLOCK_SIZE_N)
    x_mask = x_off_n < n

    # Load dh_in
    dh_in_off = tl.arange(0, BLOCK_SIZE_K)
    dh_in_mask = dh_in_off < D
    dh_in_offset = (b_idx * stride_hin_b + s_idx * stride_hin_s + dh_in_off * stride_hin_d)
    dh_in = tl.load(dh_in_ptr + dh_in_offset, mask=dh_in_mask, other=0.0)

    # Load x
    x_off_d = tl.arange(0, BLOCK_SIZE_K)
    x_mask2 = (x_off_n[:, None] < n) & (x_off_d[None, :] < D)
    x_offset = (b_idx * stride_x_b + s_idx * stride_x_s +
                 x_off_n[:, None] * stride_x_n +
                 x_off_d[None, :] * stride_x_d)
    x_block = tl.load(x_ptr + x_offset, mask=x_mask2, other=0.0).to(tl.float32)

    # Load h_pre
    h_pre_off = (b_idx * stride_hpost_b + s_idx * stride_hpost_s + x_off_n)
    h_pre = tl.load(h_pre_ptr + h_pre_off, mask=x_mask, other=0.0)

    # Load inv_rms
    inv_rms_off = b_idx * S + s_idx
    inv_rms = tl.load(inv_rms_ptr + inv_rms_off)

    # Load alpha
    a_pre = tl.load(alpha_ptr + 0)

    # Compute dh_pre
    dh_pre = tl.sum(x_block * dh_in[None, :], axis=1)

    # Compute dh_pre2
    s_pre2 = h_pre - hc_eps
    dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)

    # Load h_pre1_hmix
    h_pre1_hmix = tl.load(h_mix_ptr + (b_idx * S * out_features + s_idx * out_features + x_off_n),
                          mask=x_mask, other=0.0) * inv_rms

    # Compute contribution for each n (store instead of summing)
    contribution = dh_pre2 * h_pre1_hmix
    contribution_offset = (b_idx * S * n + s_idx * n + x_off_n)
    tl.store(contribution_ptr + contribution_offset, contribution, mask=x_mask)

# Allocate output
contribution = torch.zeros(B, S, n, dtype=torch.float32, device=device)

# Ensure contiguous
x_dev = x_dev.contiguous()
alpha_dev = alpha_dev.contiguous()
inv_rms_dev = inv_rms_dev.contiguous()
h_mix_dev = h_mix_dev.contiguous()
h_pre_dev = h_pre_dev.contiguous()
dh_in_dev = dh_in_dev.contiguous()

BLOCK_SIZE_N = triton.next_power_of_2(n)
BLOCK_SIZE_K = triton.next_power_of_2(D)

# Run kernel
grid = (B * S,)
with torch.cuda.device(device):
    dalpha_pre_no_atomic_kernel[grid](
        x_ptr=x_dev,
        alpha_ptr=alpha_dev,
        inv_rms_ptr=inv_rms_dev,
        h_mix_ptr=h_mix_dev,
        h_pre_ptr=h_pre_dev,
        dh_in_ptr=dh_in_dev,
        contribution_ptr=contribution,
        B=B, S=S, n=n, D=D, nD=nD, out_features=out_features,
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
        stride_hmix_b=S * out_features,
        stride_hmix_s=out_features,
        stride_hmix_out=1,
        norm_eps=1e-6,
        hc_eps=hc_eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

# Sum on CPU (no atomic_add)
dalpha_pre_no_atomic = contribution.sum().item()

# Compute golden
x_fp = x.float()
h_in_fp_grad = dh_in.float()
dh_pre_gold = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps_tensor = torch.tensor(hc_eps)
s_pre2_gold = h_pre - hc_eps_tensor
dh_pre2_gold = dh_pre_gold * s_pre2_gold * (1.0 - s_pre2_gold)
h_mix_tmp = h_mix * inv_rms[:, :, None]
h_pre1_gold = h_mix_tmp[:, :, 0:n]
dalpha_pre_gold = (dh_pre2_gold * h_pre1_gold).sum().item()

print("Comparison:")
print(f"  Golden (manual):         {dalpha_pre_gold:.8f}")
print(f"  No atomic_add (kernel):  {dalpha_pre_no_atomic:.8f}")
print(f"  Error:                   {abs(dalpha_pre_no_atomic - dalpha_pre_gold):.8f}")

if abs(dalpha_pre_no_atomic - dalpha_pre_gold) < 0.001:
    print("\n✓ PASS: dalpha_pre computation without atomic_add is correct!")
else:
    print("\n✗ FAIL: dalpha_pre computation without atomic_add has error!")
