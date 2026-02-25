"""
Debug dx computation to find the error
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

# Get golden backward
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

# Get Triton backward
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

print('=' * 70)
print('dx Error Analysis')
print('=' * 70)
print('')

dx_tri_cpu = dx_tri.cpu()
error = torch.abs(dx_tri_cpu - dx_gold)

print(f'Overall error statistics:')
print(f'  Max error: {error.max().item():.6f}')
print(f'  Mean error: {error.mean().item():.6f}')
print(f'  Elements with error > 1.0: {(error > 1.0).sum().item()}/{error.numel()}')
print('')

# Check which dimensions have the most error
print('Error by (b, s, n) slice:')
for b in range(B):
    for s in range(0, S, S//4):  # Check every 1/4 of S
        for n_idx in range(n):
            slice_error = error[b, s, n_idx, :].max().item()
            if slice_error > 1.0:
                print(f'  [{b}, {s}, {n_idx}]: max_err = {slice_error:.6f}')

print('')
print('dx value comparison (first element):')
print(f'  Golden dx[0, 0, 0, :5]: {dx_gold[0, 0, 0, :5]}')
print(f'  Triton dx[0, 0, 0, :5]: {dx_tri_cpu[0, 0, 0, :5]}')
print(f'  Error: {error[0, 0, 0, :5]}')
print('')

# Let's manually compute dx for (b=0, s=0, n=0)
b, s, n_idx = 0, 0, 0

# Golden computation for this slice
x_fp = x.float()
h_in_fp_grad = dh_in.float()
dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps_tensor = torch.tensor(1e-6, device=x.device)
s_pre2 = h_pre - hc_eps_tensor
dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)

# dvecX_hin for this slice
dvecX_hin_manual = h_pre[b, s, n_idx] * dh_in[b, s, :].float()
print(f'Manual dvecX_hin[{b}, {s}, {n_idx}, :5]: {dvecX_hin_manual[:5]}')

# Check if Triton's dvecX_hin would match
# dvecX_hin = h_pre_val * dh_in
print(f'Expected: h_pre[{b}, {s}, {n_idx}] * dh_in[{b}, {s}, :]')
print(f'  h_pre[{b}, {s}, {n_idx}] = {h_pre[b, s, n_idx].item():.6f}')
print(f'  dh_in[{b}, {s}, :5] = {dh_in[b, s, :5]}')
print(f'  Product: {dvecX_hin_manual[:5]}')
