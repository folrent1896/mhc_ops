"""
Analyze dbias error by section to understand which part has issues
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

# Move triton results to CPU for comparison
dbias_tri_cpu = dbias_tri.cpu()

print('dbias shape:', dbias_gold.shape)
print('')
print('dbias section breakdown:')
print(f'  dbias_pre:   indices [0:{n}]')
print(f'  dbias_post:  indices [{n}:{2*n}]')
print(f'  dbias_res:   indices [{2*n}:{n*n + 2*n}]')
print('')

# Analyze each section
dbias_pre_gold = dbias_gold[0:n]
dbias_post_gold = dbias_gold[n:2*n]
dbias_res_gold = dbias_gold[2*n:]

dbias_pre_tri = dbias_tri_cpu[0:n]
dbias_post_tri = dbias_tri_cpu[n:2*n]
dbias_res_tri = dbias_tri_cpu[2*n:]

print('=' * 70)
print('dbias_pre Analysis')
print('=' * 70)
print(f'First 4 elements:')
print(f'  Golden: {dbias_pre_gold[:4].numpy()}')
print(f'  Triton: {dbias_pre_tri[:4].numpy()}')
print(f'  Errors: {(dbias_pre_tri - dbias_pre_gold)[:4].numpy()}')
print(f'  Max error (dbias_pre):  {torch.abs(dbias_pre_tri - dbias_pre_gold).max().item():.8f}')
print(f'  Mean error (dbias_pre): {torch.abs(dbias_pre_tri - dbias_pre_gold).mean().item():.8f}')
print('')

print('=' * 70)
print('dbias_post Analysis')
print('=' * 70)
print(f'First 4 elements:')
print(f'  Golden: {dbias_post_gold[:4].numpy()}')
print(f'  Triton: {dbias_post_tri[:4].numpy()}')
print(f'  Errors: {(dbias_post_tri - dbias_post_gold)[:4].numpy()}')
print(f'  Max error (dbias_post):  {torch.abs(dbias_post_tri - dbias_post_gold).max().item():.8f}')
print(f'  Mean error (dbias_post): {torch.abs(dbias_post_tri - dbias_post_gold).mean().item():.8f}')
print('')

print('=' * 70)
print('dbias_res Analysis')
print('=' * 70)
print(f'First 4 elements:')
print(f'  Golden: {dbias_res_gold[:4].numpy()}')
print(f'  Triton: {dbias_res_tri[:4].numpy()}')
print(f'  Errors: {(dbias_res_tri - dbias_res_gold)[:4].numpy()}')
print(f'  Max error (dbias_res):  {torch.abs(dbias_res_tri - dbias_res_gold).max().item():.8f}')
print(f'  Mean error (dbias_res): {torch.abs(dbias_res_tri - dbias_res_gold).mean().item():.8f}')
print('')

# Check all elements of dbias_res
dbias_res_errors = torch.abs(dbias_res_tri - dbias_res_gold)
print(f'dbias_res error statistics:')
print(f'  Max error:  {dbias_res_errors.max().item():.8f}')
print(f'  Mean error: {dbias_res_errors.mean().item():.8f}')
print(f'  Std error:  {dbias_res_errors.std().item():.8f}')
print(f'  Elements with error > 0.1: {(dbias_res_errors > 0.1).sum().item()}/{len(dbias_res_errors)}')
print('')

# Find the element with maximum error
max_err_idx = dbias_res_errors.argmax().item()
max_err_val = dbias_res_errors[max_err_idx].item()
print(f'dbias_res max error at index {max_err_idx}:')
print(f'  Golden value: {dbias_res_gold[max_err_idx].item():.8f}')
print(f'  Triton value: {dbias_res_tri[max_err_idx].item():.8f}')
print(f'  Error:        {max_err_val:.8f}')
print('')

print('=' * 70)
print('Summary')
print('=' * 70)
print(f'dbias_pre max error:  {torch.abs(dbias_pre_tri - dbias_pre_gold).max().item():.8f}')
print(f'dbias_post max error: {torch.abs(dbias_post_tri - dbias_post_gold).max().item():.8f}')
print(f'dbias_res max error:  {torch.abs(dbias_res_tri - dbias_res_gold).max().item():.8f}')
print(f'Total max error:       {torch.abs(dbias_tri_cpu - dbias_gold).max().item():.8f}')
