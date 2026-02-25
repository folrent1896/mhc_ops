"""
Step 1: 分解 dx 的三个组成部分，定位误差来源

dx = elem_contrib + dvecX_inv + dvecX_hin

其中：
- elem_contrib = dvecX_mm * gamma
- dvecX_inv = -(dinv_rms * inv_rms^3 / nD) * x
- dvecX_hin = h_pre * dh_in
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
print('Step 1: 分解 dx 的三个组成部分')
print('=' * 70)
print('')

# Recompute golden components using the same formulas as golden.py
x_fp = x.float()
nD = n * D

# Recompute dh_* values (same as in golden)
h_in_fp_grad = dh_in.float()
dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps = 1e-6
s_pre2 = h_pre - hc_eps
dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
dh_post2 = dh_post * h_post * (1.0 - h_post / 2)

a_pre, a_post, a_res = alpha[0], alpha[1], alpha[2]
dh_pre1 = a_pre * dh_pre2
dh_post1 = a_post * dh_post2
dh_res1 = a_res * dh_res

# dh_mix
dh_mix_tmp = torch.cat([dh_pre1, dh_post1, dh_res1.reshape(B, S, n * n)], dim=-1)
dh_mix = dh_mix_tmp * inv_rms[:, :, None]

# dvecX_mm
dvecX_mm_gold = torch.matmul(dh_mix, phi)

# dvecX_inv (exact copy from golden.py)
# h_mix is from forward (already has inv_rms applied)
# h_mix shape: [B, S, out_features] where out_features = n*n + 2*n
# Need to apply inv_rms to get h_mix_tmp
h_mix_tmp = h_mix * inv_rms[:, :, None]
h_pre1_h, h_post1_h, h_res1_h = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
dinv_rms = (dh_mix_tmp * h_mix).sum(dim=-1, keepdim=True)  # [B, S, 1]
vecX = x_fp

# Debug: print shapes
print(f'dinv_rms shape: {dinv_rms.shape}')
print(f'inv_rms shape: {inv_rms.shape}')
print(f'vecX shape: {vecX.shape}')

# Apply the formula carefully
# dinv_rms: [B, S, 1], inv_rms: [B, S]
# We need to broadcast inv_rms to match vecX: [B, S, n, D]
inv_rms_broadcast = inv_rms[:, :, None, None]  # [B, S, 1, 1]
dinv_rms_broadcast = dinv_rms[:, :, :, None]  # [B, S, 1, 1]
dvecX_inv_base_gold = -(dinv_rms_broadcast * inv_rms_broadcast.pow(3) / nD) * vecX

# dvecX_hin
dvecX_hin_gold = h_pre.unsqueeze(-1) * dh_in.float().unsqueeze(2)

# Combine dvecX_inv
dvecX_inv_gold = dvecX_inv_base_gold.reshape(B, S, n, D) + dvecX_hin_gold

# elem_contrib
elem_contrib_gold = dvecX_mm_gold.reshape(B, S, n, D) * gamma.reshape(1, 1, n, D)

# Total dx
dx_from_components_gold = elem_contrib_gold + dvecX_inv_gold

print('验证 golden 分解:')
print(f'  dx_gold 与 dx_from_components_gold 匹配: {torch.allclose(dx_gold.float(), dx_from_components_gold, rtol=1e-5)}')
print(f'  Max diff: {torch.abs(dx_gold.float() - dx_from_components_gold).max().item():.10f}')
print('')

# Now analyze specific (b, s, n_idx) slices
test_cases = [
    (0, 0, 0),  # n_idx=0 (should be correct)
    (0, 0, 1),  # n_idx=1 (has error)
    (0, 0, 2),  # n_idx=2 (has error)
    (0, 0, 3),  # n_idx=3 (has error)
]

print('=' * 70)
print('逐 (b, s, n_idx) 分析误差来源')
print('=' * 70)
print('')

dx_tri_cpu = dx_tri.cpu()

for b, s, n_idx in test_cases:
    print(f'--- Slice [{b}, {s}, {n_idx}] ---')
    print('')

    # Golden components
    elem_gold = elem_contrib_gold[b, s, n_idx, :]
    dvecX_inv_gold_slice = dvecX_inv_gold[b, s, n_idx, :]
    dx_gold_slice = dx_gold.float()[b, s, n_idx, :]

    # Triton result
    dx_tri_slice = dx_tri_cpu[b, s, n_idx, :]

    # Compute errors
    dx_error = torch.abs(dx_tri_slice - dx_gold_slice).max().item()

    print(f'  dx max error: {dx_error:.6f}')
    print('')

    # Show first 5 elements of each component
    print(f'  Golden elem_contrib[:5]:     {elem_gold[:5]}')
    print(f'  Golden dvecX_inv[:5]:         {dvecX_inv_gold_slice[:5]}')
    print(f'  Golden dvecX_hin[:5]:        {dvecX_hin_gold[b, s, n_idx, :5]}')
    print(f'  Golden dx[:5]:               {dx_gold_slice[:5]}')
    print('')

    print(f'  Triton dx[:5]:               {dx_tri_slice[:5]}')
    print(f'  Error[:5]:                   {(dx_tri_slice - dx_gold_slice)[:5]}')
    print('')

    # Check if error is in elem_contrib or dvecX_inv
    # We can't separate them from Triton output, but we can check patterns
    print('  分析:')
    if dx_error < 0.1:
        print('    ✅ n_idx={} 无显著误差'.format(n_idx))
    else:
        # Check if error pattern matches dvecX_hin
        dvecX_hin_slice = dvecX_hin_gold[b, s, n_idx, :]
        print(f'    ⚠️ n_idx={n_idx} 有误差')
        print(f'    dvecX_hin 值范围: [{dvecX_hin_slice.min():.4f}, {dvecX_hin_slice.max():.4f}]')
        print(f'    dvecX_inv 值范围: [{dvecX_inv_gold_slice.min():.4f}, {dvecX_inv_gold_slice.max():.4f}]')
        print(f'    elem_contrib 值范围: [{elem_gold.min():.4f}, {elem_gold.max():.4f}]')

        # Check if error is proportional to dvecX_hin
        error_vec = dx_tri_slice - dx_gold_slice
        correlation = torch.corrcoef(torch.stack([error_vec, dvecX_hin_slice]))[0, 1]
        print(f'    误差与 dvecX_hin 的相关系数: {correlation:.4f}')

        if abs(correlation) > 0.5:
            print('    💡 误差与 dvecX_hin 高度相关，问题可能在 dvecX_hin 计算')
        else:
            print('    💡 误差与 dvecX_hin 相关性低，问题可能在其他部分')

    print('')

print('=' * 70)
print('总结')
print('=' * 70)
print('')
print('通过对比不同 n_idx 的误差，可以初步判断：')
print('1. 如果所有 n_idx 的误差都与 dvecX_hin 相关 → dvecX_hin 计算问题')
print('2. 如果只有 n_idx>0 的误差与 dvecX_hin 相关 → h_pre 加载问题')
print('3. 如果误差与 dvecX_inv 相关 → dvecX_inv 传输或计算问题')
print('4. 如果误差与 elem_contrib 相关 → dvecX_mm 或 gamma 乘法问题')
