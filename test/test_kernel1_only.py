"""
Test kernel 1 with ONLY dalpha_pre computation (no dbias, no dvecX_mm)
"""

import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, '.')

from src.forward.golden import mhc_forward_pre
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

# Gradients
torch.manual_seed(123)
dh_in = torch.randn(B, S, D, dtype=torch.bfloat16, device='cpu')
dh_post = torch.randn(B, S, n, dtype=torch.float32, device='cpu')
dh_res = torch.randn(B, S, n, n, dtype=torch.float32, device='cpu')

# Move to CUDA
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

print("Testing full backward implementation...")
print()

# Run full backward
dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = mhc_backward_triton(
    x_dev, phi_dev, alpha_dev, bias_dev,
    inv_rms_dev, h_mix_dev, h_pre_dev, h_post_dev,
    dh_in_dev, dh_post_dev, dh_res_dev, gamma_dev
)

print(f"dalpha from full backward: {dalpha_tri.cpu().numpy()}")

# Now let's manually verify what dalpha_pre SHOULD be
from src.backward.golden import mhc_backward_manual
dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

print(f"dalpha from golden:       {dalpha_gold.numpy()}")
print()
print(f"dalpha[0] error: {abs(dalpha_tri[0] - dalpha_gold[0]):.8f}")
print()

# Now let's check if the issue is that kernel 1 is computing something wrong
# by looking at what values are being used

# Manual computation
hc_eps = 1e-6
x_fp = x.float()
h_in_fp_grad = dh_in.float()
dh_pre_gold = (h_in_fp_grad.unsqueeze(2) * x_fp).sum(dim=-1)
hc_eps_tensor = torch.tensor(hc_eps)
s_pre2_gold = h_pre - hc_eps_tensor
dh_pre2_gold = dh_pre_gold * s_pre2_gold * (1.0 - s_pre2_gold)
h_mix_tmp = h_mix * inv_rms[:, :, None]
h_pre1_gold = h_mix_tmp[:, :, 0:n]

print("Manual verification:")
print(f"  dalpha_pre (manual): {(dh_pre2_gold * h_pre1_gold).sum().item():.8f}")
print(f"  dalpha_pre (golden): {dalpha_gold[0]:.8f}")
print(f"  dalpha_pre (triton): {dalpha_tri[0]:.8f}")
print()

# Check if maybe the issue is with the way we're calling the kernel
# Let's verify the input parameters are correct

print("Input verification:")
print(f"  h_mix shape: {h_mix_dev.shape}")
print(f"  h_mix dtype: {h_mix_dev.dtype}")
print(f"  h_mix is_contiguous: {h_mix_dev.is_contiguous()}")
print(f"  h_mix stride: {h_mix_dev.stride()}")
print()

print("  inv_rms shape: {inv_rms_dev.shape}")
print(f"  inv_rms dtype: {inv_rms_dev.dtype}")
print(f"  inv_rms is_contiguous: {inv_rms_dev.is_contiguous()}")
print()

# The key insight: we know the computation is correct (from isolated tests),
# but the full implementation gives wrong result. This suggests there's either:
# 1. A bug in how we're passing parameters to the kernel
# 2. A memory alignment or stride issue
# 3. Some other kernel is modifying dalpha
# 4. A synchronization issue

print("Next steps:")
print("  1. Check if maybe the issue is with the BLOCK_SIZE settings")
print("  2. Verify that all pointers are correct")
print("  3. Check if there's any padding or alignment issue")
