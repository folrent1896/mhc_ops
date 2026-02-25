"""
Isolate and test kernel 4 (dgamma) to identify the issue
"""

import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, '.')

from src.forward.golden import mhc_forward_pre
from src.backward.golden import mhc_backward_manual

B, S, n, D = 2, 64, 4, 128
device = 'cuda'

torch.manual_seed(42)
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device='cpu')
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device='cpu')
alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cpu') * 0.1
gamma = torch.randn(n, D, dtype=torch.float32, device='cpu')

with torch.no_grad():
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
        x, phi, alpha, bias, outflag=True
    )

torch.manual_seed(123)
dh_in = torch.randn(B, S, D, dtype=torch.bfloat16, device='cpu')
dh_post = torch.randn(B, S, n, dtype=torch.float32, device='cpu')
dh_res = torch.randn(B, S, n, n, dtype=torch.float32, device='cpu')

# Golden backward - also compute dvecX_mm
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

# Manually compute dvecX_mm from golden
a_pre, a_post, a_res = alpha[0], alpha[1], alpha[2]
x_fp = x.float()
h_in_fp_grad = dh_in.float()
dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps_tensor = torch.tensor(1e-6, device=x.device)
s_pre2 = h_pre - hc_eps_tensor
dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
dh_post2 = dh_post * h_post * (1.0 - h_post / 2)
dh_pre1 = a_pre * dh_pre2
dh_post1 = a_post * dh_post2
dh_res1 = a_res * dh_res
dh_mix_tmp = torch.cat([dh_pre1, dh_post1, dh_res1.reshape(B, S, n * n)], dim=-1)
dh_mix = dh_mix_tmp * inv_rms[:, :, None]
dvecX_mm_gold = torch.matmul(dh_mix, phi)

# Move to CUDA
x_dev = x.to(device)
dvecX_mm_gold_dev = dvecX_mm_gold.to(device)

# Test kernel 4 in isolation
@triton.jit
def test_dgamma_kernel(
    x_ptr,
    dvecX_mm_ptr,
    dgamma_ptr,
    B, S, n, D,
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_dvecxmm_b, stride_dvecxmm_s, stride_dvecxmm_d,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Test version of dgamma kernel - same logic as kernel 4
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

# Allocate output
dgamma_tri = torch.zeros(n * D, dtype=torch.float32, device=device)

# Block sizes
BLOCK_SIZE_N = triton.next_power_of_2(n)
BLOCK_SIZE_K = triton.next_power_of_2(D)

# Grid
grid = (n, triton.cdiv(D, BLOCK_SIZE_K))

# Ensure contiguous
x_dev = x_dev.contiguous()
dvecX_mm_gold_dev = dvecX_mm_gold_dev.contiguous()

# Run kernel
with torch.cuda.device(device):
    test_dgamma_kernel[grid](
        x_ptr=x_dev,
        dvecX_mm_ptr=dvecX_mm_gold_dev,
        dgamma_ptr=dgamma_tri,
        B=B, S=S, n=n, D=D,
        stride_x_b=S * n * D,
        stride_x_s=n * D,
        stride_x_n=D,
        stride_x_d=1,
        stride_dvecxmm_b=S * n * D,
        stride_dvecxmm_s=n * D,
        stride_dvecxmm_d=1,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

# Compare
print('=' * 70)
print('Isolated Kernel 4 Test (using golden dvecX_mm)')
print('=' * 70)
print('')
print('If this test passes, kernel 4 logic is correct.')
print('If it fails, kernel 4 has a bug.')
print('')

dgamma_tri_cpu = dgamma_tri.cpu()

print(f'dgamma shape: {dgamma_gold.shape}')
print('')

# Reshape to [n, D] for comparison
dgamma_gold_2d = dgamma_gold.reshape(n, D)
dgamma_tri_2d = dgamma_tri_cpu.reshape(n, D)

print('Per-row comparison:')
all_pass = True
for i in range(n):
    row_gold = dgamma_gold_2d[i, :]
    row_tri = dgamma_tri_2d[i, :]
    max_err = torch.abs(row_tri - row_gold).max().item()
    mean_err = torch.abs(row_tri - row_gold).mean().item()
    status = '✅ PASS' if max_err < 0.001 else '❌ FAIL'
    print(f'  n={i}: max_err={max_err:.8f}, mean_err={mean_err:.8f} {status}')
    if max_err >= 0.001:
        all_pass = False

print('')
if all_pass:
    print('✅ Kernel 4 logic is CORRECT when using golden dvecX_mm!')
    print('')
    print('Conclusion: The problem is in dvecX_mm computation, NOT in kernel 4!')
else:
    print('❌ Kernel 4 has a BUG!')
    print('')
    print('Need to fix the kernel logic.')

print('')
print('=' * 70)
print('Overall Statistics')
print('=' * 70)
print(f'Max error:  {torch.abs(dgamma_tri_2d - dgamma_gold_2d).max().item():.8f}')
print(f'Mean error: {torch.abs(dgamma_tri_2d - dgamma_gold_2d).mean().item():.8f}')
