"""
Simple check: Is dvecX_mm only computing the first block?

We'll run the full backward and then manually inspect dvecX_mm
by temporarily modifying the function to return it.
"""

import torch
import sys
sys.path.insert(0, '.')

from src.forward.golden import mhc_forward_pre
from src.backward.golden import mhc_backward_manual
from src.backward.mhc_backward_triton import mhc_backward_triton

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

# Compute golden dvecX_mm for comparison
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

# Move to CUDA and run Triton backward
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
gamma_dev = gamma.to(device)

# Run Triton backward

# Actually, let's use a simpler approach
# Just check if Triton dgamma is systematically smaller

dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = mhc_backward_triton(
    x_dev, phi_dev, alpha_dev, bias_dev,
    inv_rms_dev, h_mix_dev, h_pre_dev, h_post_dev,
    dh_in_dev, dh_post_dev, dh_res_dev, gamma_dev
)

dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

print('=' * 70)
print('Quick Check: Is dvecX_mm only computing first block?')
print('=' * 70)
print('')

# If only first block is computed, dvecX_mm[128:] would be all zeros
# This would cause dgamma to be too small

# Let's check: is dgamma_tri systematically smaller than dgamma_gold?
diff = dgamma_tri.cpu() - dgamma_gold
ratio = dgamma_tri.cpu() / (dgamma_gold + 1e-10)  # Avoid division by zero

print('Statistics:')
print(f'  Golden dgamma mean (abs): {dgamma_gold.abs().mean().item():.8f}')
print(f'  Triton dgamma mean (abs): {dgamma_tri.cpu().abs().mean().item():.8f}')
print(f'  Ratio of means: {dgamma_tri.cpu().abs().mean().item() / (dgamma_gold.abs().mean().item() + 1e-10):.8f}')
print('')

# Check how many elements have Triton < Golden
triton_smaller = (dgamma_tri.cpu() < dgamma_gold).sum().item()
triton_larger = (dgamma_tri.cpu() > dgamma_gold).sum().item()

print(f'Elements where Triton < Golden: {triton_smaller}/{len(dgamma_gold)} ({triton_smaller/len(dgamma_gold)*100:.1f}%)')
print(f'Elements where Triton > Golden: {triton_larger}/{len(dgamma_gold)} ({triton_larger/len(dgamma_gold)*100:.1f}%)')
print('')

if triton_smaller > triton_larger * 2:
    print('✅ Confirmed: Triton dgamma is systematically SMALLER than Golden')
    print('')
    print('This suggests dvecX_mm is incomplete (only first block computed)')
elif triton_larger > triton_smaller * 2:
    print('✅ Confirmed: Triton dgamma is systematically LARGER than Golden')
    print('')
    print('This suggests dvecX_mm has repeated accumulation')
else:
    print('⚠️ No clear pattern: Need deeper investigation')

print('')
print('=' * 70)
print('Block-specific analysis')
print('=' * 70)

# Split dgamma into blocks corresponding to nD_start iterations
# Block 0: dgamma[0:128] corresponds to nD_start=0
# Block 1: dgamma[128:256] corresponds to nD_start=1
# etc.

for block_idx in range(4):
    start = block_idx * 128
    end = start + 128
    block_gold = dgamma_gold[start:end]
    block_tri = dgamma_tri.cpu()[start:end]

    error = torch.abs(block_tri - block_gold).mean().item()

    print(f'Block {block_idx} [{start:3d}:{end:3d}]: mean error = {error:.8f}')

print('')
print('If Block 0 error is much smaller than other blocks,')
print('it confirms only Block 0 is being computed correctly.')
