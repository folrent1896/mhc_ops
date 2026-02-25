"""
Debug all three parts of dvecX_mm computation
"""

import torch
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

# Compute all dh_* values manually
x_fp = x.float()
h_in_fp_grad = dh_in.float()
dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps_tensor = torch.tensor(1e-6, device=x.device)
s_pre2 = h_pre - hc_eps_tensor
dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
dh_post2 = dh_post * h_post * (1.0 - h_post / 2)

a_pre, a_post, a_res = alpha[0], alpha[1], alpha[2]
dh_pre1 = a_pre * dh_pre2
dh_post1 = a_post * dh_post2
dh_res1 = a_res * dh_res

# Compute dh_mix
dh_mix_tmp = torch.cat([dh_pre1, dh_post1, dh_res1.reshape(B, S, n * n)], dim=-1)
dh_mix_gold = dh_mix_tmp * inv_rms[:, :, None]

# Compute full dvecX_mm
dvecX_mm_gold = torch.matmul(dh_mix_gold, phi)

print('=' * 70)
print('dvecX_mm Computation Breakdown')
print('=' * 70)
print('')

b, s = 0, 0

# Part 1: dh_pre1 contribution
part1_gold = torch.matmul((dh_pre1[b, s] * inv_rms[b, s]), phi[0:n, :])
print(f'Part 1 (dh_pre1 @ phi[0:n, :]):')
print(f'  First 5 elements: {part1_gold[:5]}')
print('')

# Part 2: dh_post1 contribution
part2_gold = torch.matmul((dh_post1[b, s] * inv_rms[b, s]), phi[n:2*n, :])
print(f'Part 2 (dh_post1 @ phi[n:2n, :]):')
print(f'  First 5 elements: {part2_gold[:5]}')
print('')

# Part 3: dh_res1 contribution
dh_res1_flat = dh_res1[b, s].reshape(n * n)
part3_gold = torch.matmul(dh_res1_flat, phi[2*n:, :])
print(f'Part 3 (dh_res1 @ phi[2n:, :]):')
print(f'  First 5 elements: {part3_gold[:5]}')
print('')

# Total
total_gold = part1_gold + part2_gold + part3_gold
print(f'Total (Part 1 + Part 2 + Part 3):')
print(f'  First 5 elements: {total_gold[:5]}')
print(f'  Direct dvecX_mm: {dvecX_mm_gold[b, s, :5]}')
print('')

# Now get Triton version
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

from src.backward.mhc_backward_triton import mhc_backward_triton

# We need to temporarily modify the backward function to return dvecX_mm
# Let's use a simpler approach: compute dgamma from golden and compare
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = mhc_backward_triton(
    x_dev, phi_dev, alpha_dev, bias_dev,
    inv_rms_dev, h_mix_dev, h_pre_dev, h_post_dev,
    dh_in_dev, dh_post_dev, dh_res_dev, gamma_dev
)

print('=' * 70)
print('dgamma Analysis')
print('=' * 70)
print('')

# dgamma is computed from x and dvecX_mm
# dgamma[n, D] = sum_{B,S} x[b, s, n, :] * dvecX_mm[b, s, n*D:(n+1)*D]

# Let's manually compute dgamma contribution from (b=0, s=0)
dgamma_contrib_gold = torch.zeros(n, D, dtype=torch.float32)
for n_idx in range(n):
    x_row = x_fp[0, 0, n_idx, :]  # [D]
    dvecX_mm_row = dvecX_mm_gold[0, 0, n_idx*D:(n_idx+1)*D]  # [D]
    dgamma_contrib_gold[n_idx, :] = x_row * dvecX_mm_row

print(f'dgamma contribution from (b=0, s=0):')
print(f'  First row: {dgamma_contrib_gold[0, :5]}')
print(f'  Sum over D for first row: {dgamma_contrib_gold[0, :].sum().item():.6f}')
print('')

# Compare with actual dgamma
print(f'Golden dgamma[0, :]:')
print(f'  First 5 elements: {dgamma_gold[0, :5]}')
print(f'  Sum: {dgamma_gold[0, :].sum().item():.6f}')
print('')

print(f'Triton dgamma[0, :] (from CUDA):')
print(f'  First 5 elements: {dgamma_tri.cpu()[0, :5]}')
print(f'  Sum: {dgamma_tri.cpu()[0, :].sum().item():.6f}')
print('')

error = torch.abs(dgamma_tri.cpu()[0, :] - dgamma_gold[0, :])
print(f'Error in dgamma[0, :]:')
print(f'  Max: {error.max().item():.6f}')
print(f'  Mean: {error.mean().item():.6f}')
