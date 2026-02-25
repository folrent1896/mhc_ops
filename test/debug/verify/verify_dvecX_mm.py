"""
Verify dvecX_mm computation to identify the source of dgamma error
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

# Golden backward
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

# Compute dvecX_mm in golden (need to add this manually)
# From golden: dvecX_mm = dh_mix @ phi
# Where dh_mix = dh_mix_tmp * inv_rms
# And dh_mix_tmp = cat([dh_pre1, dh_post1, dh_res1])

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

# Compute dvecX_mm
dvecX_mm_gold = torch.matmul(dh_mix, phi)  # [B, S, nD]

print('=' * 70)
print('dvecX_mm Analysis')
print('=' * 70)
print(f'dvecX_mm shape: {dvecX_mm_gold.shape}')
print(f'dvecX_mm stats:')
print(f'  Mean: {dvecX_mm_gold.mean().item():.8f}')
print(f'  Std: {dvecX_mm_gold.std().item():.8f}')
print(f'  Min: {dvecX_mm_gold.min().item():.8f}')
print(f'  Max: {dvecX_mm_gold.max().item():.8f}')
print('')

# Now compute dgamma manually using golden dvecX_mm
x_fp = x.float()
dgamma_manual = (x_fp.reshape(B * S, n * D) * dvecX_mm_gold.reshape(B * S, n * D)).sum(dim=-2)

print('Manual dgamma calculation:')
print(f'  Shape: {dgamma_manual.shape}')
print(f'  Matches golden dgamma: {torch.allclose(dgamma_manual, dgamma_gold, rtol=1e-4, atol=1e-4)}')
print(f'  Max diff: {torch.abs(dgamma_manual - dgamma_gold).max().item():.8f}')
print('')

# Now let's check what Triton dvecX_mm looks like
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

# We need to capture dvecX_mm from the backward function
# Temporarily modify the function to output dvecX_mm
import src.backward.mhc_backward_triton as mhc_bt

# Save original function
original_func = mhc_bt.mhc_backward_triton

def modified_backward(*args, **kwargs):
    # Call original but capture dvecX_mm
    result = original_func(*args, **kwargs)

    # Access dvecX_mm from the function's scope - we can't do this easily
    # Instead, let's run the full backward and compare dgamma
    return result

dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = modified_backward(
    x_dev, phi_dev, alpha_dev, bias_dev,
    inv_rms_dev, h_mix_dev, h_pre_dev, h_post_dev,
    dh_in_dev, dh_post_dev, dh_res_dev, gamma_dev
)

# Compare dgamma
print('=' * 70)
print('dgamma Comparison')
print('=' * 70)
print(f'Golden dgamma[0:10]: {dgamma_gold[0:10].numpy()}')
print(f'Triton dgamma[0:10]: {dgamma_tri.cpu()[0:10].numpy()}')
print(f'Error[0:10]: {(dgamma_tri.cpu() - dgamma_gold)[0:10].numpy()}')
print('')

# Reshape to [n, D] for comparison
dgamma_gold_2d = dgamma_gold.reshape(n, D)
dgamma_tri_2d = dgamma_tri.cpu().reshape(n, D)

print(f'Per-row comparison:')
for i in range(n):
    row_gold = dgamma_gold_2d[i, :]
    row_tri = dgamma_tri_2d[i, :]
    print(f'  n={i}: max_err={torch.abs(row_tri - row_gold).max().item():.8f}, mean_err={torch.abs(row_tri - row_gold).mean().item():.8f}')
