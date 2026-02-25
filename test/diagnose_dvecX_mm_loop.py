"""
Diagnostic kernel to check if nD_start loop executes correctly in dvecX_mm computation
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
norm_eps = 1e-6
nD = n * D
out_features = n * n + 2 * n

torch.manual_seed(42)
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device='cpu')
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device='cpu')
alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cpu') * 0.1

with torch.no_grad():
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
        x, phi, alpha, bias, outflag=True
    )

torch.manual_seed(123)
dh_in = torch.randn(B, S, D, dtype=torch.bfloat16, device='cpu')
dh_post = torch.randn(B, S, n, dtype=torch.float32, device='cpu')
dh_res = torch.randn(B, S, n, n, dtype=torch.float32, device='cpu')

# Move to CUDA
x_dev = x.to(device)
phi_dev = phi.to(device)
alpha_dev = alpha.to(device)
bias_dev = bias.to(device)
inv_rms_dev = inv_rms.to(device)
h_mix_dev = h_mix.to(device)
h_pre_dev = h_pre.to(device)
h_post_dev = h_post.to(device)
dh_in_dev = dh_in.to(device)
dh_post_dev = dh_post.to(device)
dh_res_dev = dh_res.to(device)

@triton.jit
def diagnostic_dvecX_mm_kernel(
    x_ptr,
    phi_ptr,
    alpha_ptr,
    inv_rms_ptr,
    h_pre_ptr,
    h_post_ptr,
    dh_in_ptr,
    dh_post_ptr,
    dh_res_ptr,
    dvecX_mm_ptr,
    loop_counter_ptr,  # Output: [B*S] - count loop iterations per program
    B, S, n, D, nD, out_features,
    stride_x_b, stride_x_s, stride_x_n, stride_x_d,
    stride_phi_out, stride_phi_in,
    stride_hin_b, stride_hin_s, stride_hin_d,
    stride_hpost_b, stride_hpost_s, stride_hpost_n,
    stride_hres_b, stride_hres_s, stride_hres_n1, stride_hres_n2,
    stride_dvecxmm_b, stride_dvecxmm_s, stride_dvecxmm_d,
    norm_eps: tl.float32,
    hc_eps: tl.float32,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Diagnostic version of dvecX_mm computation that counts loop iterations
    """
    # Program ID
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    x_off_n = tl.arange(0, BLOCK_SIZE_N)
    x_off_d = tl.arange(0, BLOCK_SIZE_K)
    x_mask = (x_off_n[:, None] < n) & (x_off_d[None, :] < D)
    x_offset = (b_idx * stride_x_b + s_idx * stride_x_s +
                x_off_n[:, None] * stride_x_n +
                x_off_d[None, :] * stride_x_d)
    x_block = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0).to(tl.float32)

    # Load inv_rms
    inv_rms_off = b_idx * S + s_idx
    inv_rms = tl.load(inv_rms_ptr + inv_rms_off)

    # Load h_pre, h_post, dh_in, dh_post
    h_pre_off = (b_idx * stride_hpost_b + s_idx * stride_hpost_s + x_off_n)
    h_pre_mask = x_off_n < n
    h_pre = tl.load(h_pre_ptr + h_pre_off, mask=h_pre_mask, other=0.0)
    h_post = tl.load(h_post_ptr + h_pre_off, mask=h_pre_mask, other=0.0)

    dh_in_off = x_off_d
    dh_in_mask = x_off_d < D
    dh_in_offset = (b_idx * stride_hin_b + s_idx * stride_hin_s +
                     dh_in_off * stride_hin_d)
    dh_in = tl.load(dh_in_ptr + dh_in_offset, mask=dh_in_mask, other=0.0)

    dh_post = tl.load(dh_post_ptr + h_pre_off, mask=h_pre_mask, other=0.0)

    # Load dh_res
    dh_res_off_n1 = x_off_n
    dh_res_off_n2 = x_off_n
    dh_res_mask = (dh_res_off_n1[:, None] < n) & (dh_res_off_n2[None, :] < n)
    dh_res_offset = (b_idx * stride_hres_b + s_idx * stride_hres_s +
                     dh_res_off_n1[:, None] * stride_hres_n1 +
                     dh_res_off_n2[None, :] * stride_hres_n2)
    dh_res_block = tl.load(dh_res_ptr + dh_res_offset, mask=dh_res_mask, other=0.0)

    # Load alpha
    a_pre = tl.load(alpha_ptr + 0)
    a_post = tl.load(alpha_ptr + 1)
    a_res = tl.load(alpha_ptr + 2)

    # Compute gradients
    dh_pre = tl.sum(x_block * dh_in[None, :], axis=1)
    s_pre2 = h_pre - hc_eps
    dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
    dh_post2 = dh_post * h_post * (1.0 - h_post / 2)

    dh_pre1 = a_pre * dh_pre2
    dh_post1 = a_post * dh_post2
    dh_res1 = a_res * dh_res_block

    # Initialize loop counter
    loop_count = tl.zeros([BLOCK_SIZE_K], dtype=tl.int32)

    # ============================================================
    # KEY: nD_start loop - this is what we're testing
    # ============================================================
    expected_iterations = (nD + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K  # Should be 4

    for nD_start in range(0, nD, BLOCK_SIZE_K):
        # Increment counter (just increment first element)
        loop_count[0] += 1

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
        # Omit for simplicity in diagnostic
        # ...

        # Store for this nD_idx (even if incomplete)
        dvecX_mm_offset = (b_idx * stride_dvecxmm_b + s_idx * stride_dvecxmm_s + nD_idx)
        tl.store(dvecX_mm_ptr + dvecX_mm_offset, acc, mask=nD_mask)

    # Store loop count for this program
    tl.store(loop_counter_ptr + pid, loop_count)

# Allocate outputs
dvecX_mm_diag = torch.zeros(B, S, nD, dtype=torch.float32, device=device)
loop_counter = torch.zeros(B * S, dtype=torch.int32, device=device)

# Block sizes
BLOCK_SIZE_N = triton.next_power_of_2(n)
BLOCK_SIZE_K = triton.next_power_of_2(D)

# Ensure contiguous
x_dev = x_dev.contiguous()
phi_dev = phi_dev.contiguous()
alpha_dev = alpha_dev.contiguous()
inv_rms_dev = inv_rms_dev.contiguous()
h_pre_dev = h_pre_dev.contiguous()
h_post_dev = h_post_dev.contiguous()
dh_in_dev = dh_in_dev.contiguous()
dh_post_dev = dh_post_dev.contiguous()
dh_res_dev = dh_res_dev.contiguous()

# Grid
grid = (B * S,)

# Run diagnostic kernel
with torch.cuda.device(device):
    diagnostic_dvecX_mm_kernel[grid](
        x_ptr=x_dev,
        phi_ptr=phi_dev,
        alpha_ptr=alpha_dev,
        inv_rms_ptr=inv_rms_dev,
        h_pre_ptr=h_pre_dev,
        h_post_ptr=h_post_dev,
        dh_in_ptr=dh_in_dev,
        dh_post_ptr=dh_post_dev,
        dh_res_ptr=dh_res_dev,
        dvecX_mm_ptr=dvecX_mm_diag,
        loop_counter_ptr=loop_counter,
        B=B, S=S, n=n, D=D, nD=nD, out_features=out_features,
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
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

# Check loop counter
loop_counter_cpu = loop_counter.cpu()

print('=' * 70)
print('nD_start Loop Execution Diagnostic')
print('=' * 70)
print('')
print(f'Expected loop iterations: {(nD + 127) // 128} (nD={nD}, BLOCK_SIZE_K=128)')
print('')
print(f'Actual loop iterations per program:')
print(f'  Min: {loop_counter_cpu.min().item()}')
print(f'  Max: {loop_counter_cpu.max().item()}')
print(f'  Mean: {loop_counter_cpu.float().mean().item():.2f}')
print(f'  Unique values: {torch.unique(loop_counter_cpu).cpu().numpy()}')
print('')

if loop_counter_cpu.max().item() == 1:
    print('❌ BUG CONFIRMED: Loop only executes ONCE!')
    print('')
    print('This means only the first block [0:128] is computed,')
    print('and blocks [128:256], [256:384], [384:512] are all ZERO!')
    print('')
    print('This explains why dgamma values are too small.')
elif loop_counter_cpu.max().item() == 4:
    print('✅ Loop executes correctly 4 times')
    print('')
    print('The problem is elsewhere (possibly in dh_res1 nested loop)')
else:
    print(f'⚠️ Unexpected loop count: {loop_counter_cpu.max().item()}')
    print('')
    print('Need to investigate further')

print('')
print('=' * 70)
print('dvecX_mm Value Check')
print('=' * 70)
print(f'dvecX_mm_diag stats:')
print(f'  Mean: {dvecX_mm_diag.mean().item():.8f}')
print(f'  Std: {dvecX_mm_diag.std().item():.8f}')
print(f'  Non-zero elements: {(dvecX_mm_diag.abs() > 1e-6).sum().item()}/{dvecX_mm_diag.numel()}')
print('')

# Check if values are concentrated in first block
first_block = dvecX_mm_diag[:, :, 0:128]
remaining_blocks = dvecX_mm_diag[:, :, 128:]

print(f'First block [0:128] non-zero: {(first_block.abs() > 1e-6).sum().item()}')
print(f'Remaining blocks [128:512] non-zero: {(remaining_blocks.abs() > 1e-6).sum().item()}')
print('')

if (first_block.abs() > 1e-6).sum().item() > 0 and (remaining_blocks.abs() > 1e-6).sum().item() == 0:
    print('❌ CONFIRMED: Only first block is non-zero!')
    print('')
    print('Fix needed: Make nD_start loop work correctly')
