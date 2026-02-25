"""
Simple comparison of golden vs triton backward
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

# Forward pass
with torch.no_grad():
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
        x, phi, alpha, bias, outflag=True
    )

# Use SAME gradients for both
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

# Triton backward
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

print("Golden dalpha:     ", dalpha_gold.numpy())
print("Triton dalpha:     ", dalpha_tri.cpu().numpy())
print()
print("Errors:")
print(f"  dalpha[0]: {abs(dalpha_tri[0] - dalpha_gold[0]):.8f}")
print(f"  dalpha[1]: {abs(dalpha_tri[1] - dalpha_gold[1]):.8f}")
print(f"  dalpha[2]: {abs(dalpha_tri[2] - dalpha_gold[2]):.8f}")
print()

# Now let's manually compute dalpha_pre on CUDA and compare
hc_eps = 1e-6
x_fp = x.float()
h_in_fp_grad = dh_in.float()
dh_pre_gold = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps_tensor = torch.tensor(hc_eps)
s_pre2_gold = h_pre - hc_eps_tensor
dh_pre2_gold = dh_pre_gold * s_pre2_gold * (1.0 - s_pre2_gold)
h_mix_tmp = h_mix * inv_rms[:, :, None]
h_pre1_gold = h_mix_tmp[:, :, 0:n]
dalpha_pre_manual = (dh_pre2_gold * h_pre1_gold).sum()

print("Manual verification:")
print(f"  dalpha_pre (manual): {dalpha_pre_manual:.8f}")
print(f"  dalpha_pre (golden): {dalpha_gold[0]:.8f}")
print(f"  dalpha_pre (triton): {dalpha_tri[0]:.8f}")
print()

# The error should be small if everything is correct
if abs(dalpha_tri[0] - dalpha_gold[0]) < 0.01:
    print("✓ dalpha_pre error is acceptable (< 0.01)")
else:
    print(f"✗ dalpha_pre error is too large: {abs(dalpha_tri[0] - dalpha_gold[0]):.8f}")
