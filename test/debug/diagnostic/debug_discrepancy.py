"""
Debug why Part 1 + Part 2 + Part 3 != Direct dvecX_mm
"""

import torch
import sys
sys.path.insert(0, '.')

from src.forward.golden import mhc_forward_pre

B, S, n, D = 2, 64, 4, 128

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

b, s = 0, 0

# Compute dh_mix correctly
dh_mix_pre1 = dh_pre1[b, s] * inv_rms[b, s]  # [n]
dh_mix_post1 = dh_post1[b, s] * inv_rms[b, s]  # [n]
dh_mix_res1 = (dh_res1[b, s].reshape(n * n)) * inv_rms[b, s]  # [n*n]

dh_mix_full = torch.cat([dh_mix_pre1, dh_mix_post1, dh_mix_res1])  # [2n + n*n]

print('=' * 70)
print('dh_mix Computation Check')
print('=' * 70)
print('')
print(f'dh_mix_pre1 shape: {dh_mix_pre1.shape}')
print(f'dh_mix_post1 shape: {dh_mix_post1.shape}')
print(f'dh_mix_res1 shape: {dh_mix_res1.shape}')
print(f'dh_mix_full shape: {dh_mix_full.shape}')
print('')

# Part 1: dh_pre1 contribution
part1 = torch.matmul(dh_mix_pre1, phi[0:n, :])
print(f'Part 1 (first 5): {part1[:5]}')

# Part 2: dh_post1 contribution
part2 = torch.matmul(dh_mix_post1, phi[n:2*n, :])
print(f'Part 2 (first 5): {part2[:5]}')

# Part 3: dh_res1 contribution
part3 = torch.matmul(dh_mix_res1, phi[2*n:, :])
print(f'Part 3 (first 5): {part3[:5]}')

# Total
total = part1 + part2 + part3
print(f'Total (first 5): {total[:5]}')

# Direct
direct = torch.matmul(dh_mix_full, phi)
print(f'Direct (first 5): {direct[:5]}')

print('')
print(f'Max difference: {torch.abs(total - direct).max().item():.10f}')
print(f'Mean difference: {torch.abs(total - direct).mean().item():.10f}')

# Check if inv_rms is applied correctly
print('')
print('=' * 70)
print('inv_rms Check')
print('=' * 70)
print('')
print(f'inv_rms[{b}, {s}]: {inv_rms[b, s].item():.10f}')
print(f'dh_pre1[{b}, {s}] (first 5): {dh_pre1[b, s, :5]}')
print(f'dh_pre1 * inv_rms (first 5): {(dh_pre1[b, s] * inv_rms[b, s])[:5]}')
print(f'dh_mix_pre1 (first 5): {dh_mix_pre1[:5]}')
