"""
Debug script to understand dh_res1 @ phi[2n:, :] computation
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

# Compute dh_res1 manually
a_res = alpha[2]
dh_res1_gold = a_res * dh_res  # [B, S, n, n]

# Compute golden dh_res1 @ phi[2n:, :] for one (b, s) pair
b, s = 0, 0
dh_res1_flat = dh_res1_gold[b, s].reshape(n * n)  # [n*n]
phi_res = phi[2*n:, :]  # [n*n, nD]

result_gold = torch.matmul(dh_res1_flat.unsqueeze(0), phi_res)  # [1, nD]

print('=' * 70)
print('dh_res1 @ phi[2n:, :] Computation Debug')
print('=' * 70)
print('')
print(f'Input shapes:')
print(f'  dh_res1[{b}, {s}]: {dh_res1_gold[b, s].shape}')
print(f'  phi[2n:, :]: {phi_res.shape}')
print(f'  Result: {result_gold.shape}')
print('')

# Show first few elements
print(f'First 10 elements of dh_res1_flat:')
print(f'  {dh_res1_flat[:10]}')
print('')

print(f'First 10 elements of phi_res (first 10 rows, first 5 columns):')
print(f'  {phi_res[:10, :5]}')
print('')

print(f'First 10 elements of result_gold:')
print(f'  {result_gold[0, :10]}')
print('')

# Now let's simulate what the nested loop does
nD = n * D
BLOCK_SIZE_N = 4  # next_power_of_2(n)
BLOCK_SIZE_K = 128  # next_power_of_2(D)

print('=' * 70)
print('Simulating Triton nested loop computation')
print('=' * 70)
print('')

# Manual simulation of the nested loop
result_sim = torch.zeros(nD, dtype=torch.float32)

for nD_start in range(0, nD, BLOCK_SIZE_K):
    nD_idx = nD_start + torch.arange(0, BLOCK_SIZE_K)
    nD_mask = nD_idx < nD
    acc = torch.zeros(BLOCK_SIZE_K, dtype=torch.float32)

    # Part 3: dh_res1 @ phi[2n:, :]
    for res_i in range(0, n, BLOCK_SIZE_N):
        for res_j in range(0, n, BLOCK_SIZE_N):
            # Compute phi row indices
            i_idx = torch.arange(0, BLOCK_SIZE_N)
            j_idx = torch.arange(0, BLOCK_SIZE_N)

            # These are the phi row indices being loaded
            phi_row_idx = 2 * n + (res_i + i_idx)[:, None] * n + (res_j + j_idx)[None, :]

            # Load phi
            phi_res_block = phi[phi_row_idx[:, :, None], nD_idx[None, None, :]]  # [BLOCK, BLOCK, BLOCK_K]

            # Load dh_res1 block
            dh_res1_block = dh_res1_gold[b, s][res_i:res_i+BLOCK_SIZE_N, res_j:res_j+BLOCK_SIZE_N]

            # Compute
            temp = torch.sum(dh_res1_block[:, :, None] * phi_res_block, dim=1)
            acc += torch.sum(temp, dim=0)

    result_sim[nD_idx[nD_mask]] = acc[nD_mask[:nD_idx[nD_mask].numel()]]

print(f'First 10 elements of result_sim:')
print(f'  {result_sim[:10]}')
print('')

print(f'First 10 elements of result_gold:')
print(f'  {result_gold[0, :10]}')
print('')

error = torch.abs(result_sim - result_gold[0, :])
print(f'Error statistics:')
print(f'  Max error: {error.max().item():.8f}')
print(f'  Mean error: {error.mean().item():.8f}')
print(f'  Elements with error > 0.1: {(error > 0.1).sum().item()}/{len(error)}')
print('')

# Let's check if the issue is in the nested loop structure
print('=' * 70)
print('Checking if nested loop changes result')
print('=' * 70)
print('')

# Direct computation without nested loop
result_direct = torch.zeros(nD, dtype=torch.float32)
for i in range(n):
    for j in range(n):
        phi_row = 2 * n + i * n + j
        result_direct += dh_res1_gold[b, s, i, j] * phi[phi_row, :]

print(f'Max error (sim vs direct): {torch.abs(result_sim - result_direct).max().item():.8f}')
print(f'Max error (sim vs gold): {torch.abs(result_sim - result_gold[0, :]).max().item():.8f}')
print(f'Max error (direct vs gold): {torch.abs(result_direct - result_gold[0, :]).max().item():.8f}')
