"""
Detailed comparison of dalpha_pre calculation between Golden and Triton
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

# Forward pass
with torch.no_grad():
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
        x, phi, alpha, bias, outflag=True
    )

# Use SAME gradients for both golden and triton
torch.manual_seed(123)  # Different seed for gradients
dh_in = torch.randn(B, S, D, dtype=torch.bfloat16, device='cpu')
dh_post = torch.randn(B, S, n, dtype=torch.float32, device='cpu')
dh_res = torch.randn(B, S, n, n, dtype=torch.float32, device='cpu')

# Golden backward (on CPU)
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

print("Golden results (CPU):")
print(f"  dalpha: {dalpha_gold.numpy()}")
print(f"  dalpha_pre (dalpha[0]): {dalpha_gold[0]:.8f}")
print(f"  dalpha_post (dalpha[1]): {dalpha_gold[1]:.8f}")
print(f"  dalpha_res (dalpha[2]): {dalpha_gold[2]:.8f}")
print()

# Move to CUDA for Triton
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

# Triton backward
from src.backward.mhc_backward_triton import mhc_backward_triton
dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = mhc_backward_triton(
    x_dev, phi_dev, alpha_dev, bias_dev,
    inv_rms_dev, h_mix_dev, h_pre_dev, h_post_dev,
    dh_in_dev, dh_post_dev, dh_res_dev, gamma_dev
)

print("Triton results (CUDA):")
print(f"  dalpha: {dalpha_tri.cpu().numpy()}")
print(f"  dalpha_pre (dalpha[0]): {dalpha_tri[0]:.8f}")
print(f"  dalpha_post (dalpha[1]): {dalpha_tri[1]:.8f}")
print(f"  dalpha_res (dalpha[2]): {dalpha_tri[2]:.8f}")
print()

print("Comparison:")
print(f"  dalpha_pre error: {abs(dalpha_tri[0] - dalpha_gold[0]):.8f}")
print(f"  dalpha_post error: {abs(dalpha_tri[1] - dalpha_gold[1]):.8f}")
print(f"  dalpha_res error: {abs(dalpha_tri[2] - dalpha_gold[2]):.8f}")
print()

# Detailed analysis for dalpha_pre
print("=" * 70)
print("Detailed analysis of dalpha_pre calculation")
print("=" * 70)

# Golden computation (manual)
hc_eps = 1e-6
x_fp = x.float()
h_in_fp_grad = dh_in.float()
dh_pre = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps_tensor = torch.tensor(hc_eps)
s_pre2 = h_pre - hc_eps_tensor
dh_pre2 = dh_pre * s_pre2 * (1.0 - s_pre2)
h_mix_tmp = h_mix * inv_rms[:, :, None]
h_pre1_gold = h_mix_tmp[:, :, 0:n]
dalpha_pre_gold_manual = (dh_pre2 * h_pre1_gold).sum()

print(f"Manual golden dalpha_pre: {dalpha_pre_gold_manual:.8f}")
print(f"Actual golden dalpha_pre: {dalpha_gold[0]:.8f}")
print(f"Match: {abs(dalpha_pre_gold_manual - dalpha_gold[0]) < 1e-5}")
print()

# Check component values
print("Component statistics:")
print(f"  dh_pre2: mean={dh_pre2.mean():.6f}, std={dh_pre2.std():.6f}")
print(f"  h_pre1_gold: mean={h_pre1_gold.mean():.6f}, std={h_pre1_gold.std():.6f}")
print(f"  inv_rms: mean={inv_rms.mean():.6f}, std={inv_rms.std():.6f}")
print()

# Check for a specific element
b, s = 0, 0
print(f"Sample element (b={b}, s={s}):")
print(f"  dh_pre2[{b}, {s}, :]: {dh_pre2[b, s, :].numpy()}")
print(f"  h_pre1_gold[{b}, {s}, :]: {h_pre1_gold[b, s, :].numpy()}")
print(f"  contribution (dh_pre2 * h_pre1_gold): {(dh_pre2[b, s, :] * h_pre1_gold[b, s, :]).numpy()}")
print(f"  sum: {(dh_pre2[b, s, :] * h_pre1_gold[b, s, :]).sum().item():.8f}")
