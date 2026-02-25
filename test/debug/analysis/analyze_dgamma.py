"""
Analyze dgamma error in detail to understand the problem
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

# Golden
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

# Triton
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

dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = mhc_backward_triton(
    x_dev, phi_dev, alpha_dev, bias_dev,
    inv_rms_dev, h_mix_dev, h_pre_dev, h_post_dev,
    dh_in_dev, dh_post_dev, dh_res_dev, gamma_dev
)

# Move to CPU for comparison
dgamma_tri_cpu = dgamma_tri.cpu()

print('=' * 70)
print('dgamma Error Analysis')
print('=' * 70)
print('')
print(f'dgamma shape: {dgamma_gold.shape}')  # Should be [n*D]
print(f'Reshaped to [n, D]: [{n}, {D}]')
print('')

# Reshape to [n, D] for easier analysis
dgamma_gold_2d = dgamma_gold.reshape(n, D)
dgamma_tri_2d = dgamma_tri_cpu.reshape(n, D)

print('Overall statistics:')
print(f'  Max error:  {torch.abs(dgamma_tri_2d - dgamma_gold_2d).max().item():.8f}')
print(f'  Mean error: {torch.abs(dgamma_tri_2d - dgamma_gold_2d).mean().item():.8f}')
print(f'  Std error:  {torch.abs(dgamma_tri_2d - dgamma_gold_2d).std().item():.8f}')
print('')

# Analyze each row (each n dimension)
print('=' * 70)
print('Per-row (per n dimension) analysis')
print('=' * 70)

for i in range(n):
    row_gold = dgamma_gold_2d[i, :]
    row_tri = dgamma_tri_2d[i, :]
    row_error = torch.abs(row_tri - row_gold)

    print(f'n={i}:')
    print(f'  Max error:  {row_error.max().item():.8f}')
    print(f'  Mean error: {row_error.mean().item():.8f}')
    print(f'  Golden range: [{row_gold.min().item():.6f}, {row_gold.max().item():.6f}]')
    print(f'  Triton range: [{row_tri.min().item():.6f}, {row_tri.max().item():.6f}]')
    print('')

# Find the element with maximum error
max_err_2d = torch.abs(dgamma_tri_2d - dgamma_gold_2d)
max_idx = max_err_2d.argmax().item()
max_n = max_idx // D
max_d = max_idx % D

print('=' * 70)
print('Maximum Error Element')
print('=' * 70)
print(f'Element [{max_n}, {max_d}]:')
print(f'  Golden value: {dgamma_gold_2d[max_n, max_d].item():.8f}')
print(f'  Triton value: {dgamma_tri_2d[max_n, max_d].item():.8f}')
print(f'  Error:        {max_err_2d[max_n, max_d].item():.8f}')
print(f'  Relative error: {abs(max_err_2d[max_n, max_d].item() / (dgamma_gold_2d[max_n, max_d].item() + 1e-10)) * 100:.2f}%')
print('')

# Check distribution of errors
print('=' * 70)
print('Error Distribution')
print('=' * 70)

errors = torch.abs(dgamma_tri_2d - dgamma_gold_2d).flatten()
print(f'Total elements: {len(errors)}')
print(f'Elements with error < 0.1:  {(errors < 0.1).sum().item()} ({(errors < 0.1).sum().item() / len(errors) * 100:.1f}%)')
print(f'Elements with error 0.1-1.0:  {((errors >= 0.1) & (errors < 1.0)).sum().item()} ({((errors >= 0.1) & (errors < 1.0)).sum().item() / len(errors) * 100:.1f}%)')
print(f'Elements with error > 1.0:  {(errors >= 1.0).sum().item()} ({(errors >= 1.0).sum().item() / len(errors) * 100:.1f}%)')
print('')

# Compare value ranges
print('=' * 70)
print('Value Range Comparison')
print('=' * 70)
print(f'Golden dgamma:')
print(f'  Min: {dgamma_gold.min().item():.8f}')
print(f'  Max: {dgamma_gold.max().item():.8f}')
print(f'  Mean: {dgamma_gold.mean().item():.8f}')
print(f'  Std: {dgamma_gold.std().item():.8f}')
print('')
print(f'Triton dgamma:')
print(f'  Min: {dgamma_tri_cpu.min().item():.8f}')
print(f'  Max: {dgamma_tri_cpu.max().item():.8f}')
print(f'  Mean: {dgamma_tri_cpu.mean().item():.8f}')
print(f'  Std: {dgamma_tri_cpu.std().item():.8f}')
