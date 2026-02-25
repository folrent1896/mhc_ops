"""
Direct comparison of dvecX_mm from golden vs Triton
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

# Compute golden dvecX_mm
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

# Get Triton dgamma
from src.backward.mhc_backward_triton import mhc_backward_triton

# Actually, let's just compute dgamma from both and compare the ratio
# This will tell us if dvecX_mm is systematically too small or too large

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

# Get golden dgamma
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

print('=' * 70)
print('dgamma Ratio Analysis')
print('=' * 70)
print('')
print('If dgamma_tri / dgamma_gold is constant, dvecX_mm has a systematic error')
print('')

dgamma_tri_cpu = dgamma_tri.cpu()

# Compute ratio for elements where golden != 0
mask = torch.abs(dgamma_gold) > 1e-6
ratio = dgamma_tri_cpu[mask] / dgamma_gold[mask]

print(f'Number of non-zero golden elements: {mask.sum().item()}/{len(dgamma_gold)}')
print(f'Ratio statistics:')
print(f'  Mean: {ratio.mean().item():.8f}')
print(f'  Std: {ratio.std().item():.8f}')
print(f'  Min: {ratio.min().item():.8f}')
print(f'  Max: {ratio.max().item():.8f}')
print('')

# Check if ratio is close to constant
if ratio.std().item() / abs(ratio.mean().item()) < 0.01:
    print('✅ Ratio is nearly constant!')
    print(f'   This means dvecX_mm is systematically scaled by {ratio.mean().item():.6f}')
    print('')
    print('Hypothesis: The dvecX_mm loop might not be accumulating all iterations')
else:
    print('❌ Ratio varies significantly')
    print('   This suggests a more complex issue in dvecX_mm computation')

print('')
print('=' * 70)
print('First 10 elements comparison')
print('=' * 70)
for i in range(10):
    g = dgamma_gold[i].item()
    t = dgamma_tri_cpu[i].item()
    r = t / g if abs(g) > 1e-6 else float('inf')
    print(f'[{i:2d}] Golden: {g:12.6f}  Triton: {t:12.6f}  Ratio: {r:8.6f}')
