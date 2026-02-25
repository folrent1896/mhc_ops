"""
Verify dvecX_mm computation in kernel 1 to find the source of dgamma error
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
hc_eps = 1e-6
nD = n * D
out_features = n * n + 2 * n

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

# Compute golden dvecX_mm
a_pre, a_post, a_res = alpha[0], alpha[1], alpha[2]
x_fp = x.float()
h_in_fp_grad = dh_in.float()
dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps_tensor = torch.tensor(hc_eps, device=x.device)
s_pre2 = h_pre - hc_eps_tensor
dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
dh_post2 = dh_post * h_post * (1.0 - h_post / 2)
dh_pre1 = a_pre * dh_pre2
dh_post1 = a_post * dh_post2
dh_res1 = a_res * dh_res
dh_mix_tmp = torch.cat([dh_pre1, dh_post1, dh_res1.reshape(B, S, n * n)], dim=-1)
dh_mix = dh_mix_tmp * inv_rms[:, :, None]
dvecX_mm_gold = torch.matmul(dh_mix, phi)

print('=' * 70)
print('Golden dvecX_mm Computation')
print('=' * 70)
print(f'dvecX_mm shape: {dvecX_mm_gold.shape}')
print(f'dvecX_mm stats:')
print(f'  Mean: {dvecX_mm_gold.mean().item():.8f}')
print(f'  Std: {dvecX_mm_gold.std().item():.8f}')
print(f'  Min: {dvecX_mm_gold.min().item():.8f}')
print(f'  Max: {dvecX_mm_gold.max().item():.8f}')
print('')

# Now let's compute what dgamma would be using golden dvecX_mm
dgamma_from_gold_dvecX = (x_fp.reshape(B * S, n * D) * dvecX_mm_gold.reshape(B * S, n * D)).sum(dim=-2)

print('=' * 70)
print('Check: Does golden dvecX_mm produce golden dgamma?')
print('=' * 70)
# Get golden dgamma from backward
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

print(f'dgamma from manual calc: {dgamma_from_gold_dvecX[0:10].numpy()}')
print(f'dgamma from backward:     {dgamma_gold[0:10].numpy()}')
print(f'Match: {torch.allclose(dgamma_from_gold_dvecX, dgamma_gold, rtol=1e-5, atol=1e-5)}')
print(f'Max diff: {torch.abs(dgamma_from_gold_dvecX - dgamma_gold).max().item():.8f}')
print('')

# Now let's compute dgamma from the actual Triton backward
from src.backward.mhc_backward_triton import mhc_backward_triton

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

print('=' * 70)
print('Triton vs Golden Comparison')
print('=' * 70)
print(f'Triton dgamma: {dgamma_tri.cpu()[0:10].numpy()}')
print(f'Golden dgamma: {dgamma_gold[0:10].numpy()}')
print(f'Error:         {(dgamma_tri.cpu() - dgamma_gold)[0:10].numpy()}')
print('')

# The key question: Is the problem in dvecX_mm or in dgamma calculation?
# We already verified kernel 4 is correct when using golden dvecX_mm
# So if Triton dgamma != golden dgamma, the problem must be in dvecX_mm

print('=' * 70)
print('Conclusion')
print('=' * 70)
print('Since kernel 4 test passed, the problem MUST be in dvecX_mm computation!')
print('')
print('Need to investigate kernel 1 dvecX_mm calculation (lines 210-246)')
print('')
print('Possible issues:')
print('  1. Loop not executing correctly (nD_start loop)')
print('  2. Block accumulation missing some data')
print('  3. dh_res1 nested loop has issues')
print('  4. Memory stride/access pattern problems')
