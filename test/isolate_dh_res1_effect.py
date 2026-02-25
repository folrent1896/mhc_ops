"""
Test dgamma computation without dh_res1 part to isolate the problem
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

# Compute golden dh_mix without dh_res1
a_pre, a_post = alpha[0], alpha[1]
x_fp = x.float()
h_in_fp_grad = dh_in.float()
dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps_tensor = torch.tensor(1e-6, device=x.device)
s_pre2 = h_pre - hc_eps_tensor
dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
dh_post2 = dh_post * h_post * (1.0 - h_post / 2)

dh_pre1 = a_pre * dh_pre2
dh_post1 = a_post * dh_post2

# dh_mix without dh_res1: shape should be [B, S, 2n]
dh_mix_partial = torch.cat([dh_pre1, dh_post1], dim=-1) * inv_rms[:, :, None]

# Need to use the first 2n rows of phi
phi_partial = phi[:2*n, :]  # [2n, nD]

dvecX_mm_partial_gold = torch.matmul(dh_mix_partial, phi_partial)  # [B, S, 2n] @ [2n, nD] = [B, S, nD]

# Compute dgamma (only first 2n elements will be correct)
# We only care about first 2n output elements, so we don't need the rest
x_reshaped = x_fp.reshape(B * S, n * D)  # [B*S, nD]
dvecX_mm_partial_reshaped = dvecX_mm_partial_gold.reshape(B * S, n * D)  # [B*S, nD]
dgamma_partial_gold = (x_reshaped[:, :2*n] * dvecX_mm_partial_reshaped[:, :2*n]).sum(dim=-2)

# Now get full Triton dgamma
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

dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = mhc_backward_triton(
    x_dev, phi_dev, alpha_dev, bias_dev,
    inv_rms_dev, h_mix_dev, h_pre_dev, h_post_dev,
    dh_in_dev, dh_post_dev, dh_res_dev, gamma_dev
)

# Get full golden dgamma
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

print('=' * 70)
print('Test: Is dh_res1 part causing the problem?')
print('=' * 70)
print('')

# Split dgamma into contributions
# First 2*n elements come from dh_pre1 and dh_post1
# Last n*n elements come from dh_res1
first_2n = 2 * n
last_n2 = n * n

dgamma_gold_first = dgamma_gold[:first_2n]
dgamma_tri_first = dgamma_tri.cpu()[:first_2n]

dgamma_gold_last = dgamma_gold[first_2n:]
dgamma_tri_last = dgamma_tri.cpu()[first_2n:]

print('First 2n elements (from dh_pre1 and dh_post1):')
print(f'  Max error: {torch.abs(dgamma_tri_first - dgamma_gold_first).max().item():.8f}')
print(f'  Mean error: {torch.abs(dgamma_tri_first - dgamma_gold_first).mean().item():.8f}')
print('')

print('Last n*n elements (from dh_res1):')
print(f'  Max error: {torch.abs(dgamma_tri_last - dgamma_gold_last).max().item():.8f}')
print(f'  Mean error: {torch.abs(dgamma_tri_last - dgamma_gold_last).mean().item():.8f}')
print('')

# Compare
if torch.abs(dgamma_tri_first - dgamma_gold_first).max().item() < 0.1:
    print('✅ First 2n elements are mostly correct')
else:
    print('❌ First 2n elements have errors')

print('')
if torch.abs(dgamma_tri_last - dgamma_gold_last).max().item() < 0.1:
    print('✅ Last n*n elements are mostly correct')
else:
    print('❌ Last n*n elements have errors')

print('')
print('=' * 70)
print('Conclusion')
print('=' * 70)
print('If first 2n elements have much smaller error than last n*n elements,')
print('then the problem is in dh_res1 computation (nested loop).')
print('Otherwise, the problem affects all parts equally.')
