"""
Compare isolated kernel vs full kernel 1 for dalpha_pre computation
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

# Use SAME gradients for all tests
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

# Test 1: Isolated kernel
print("=" * 70)
print("Test 1: Isolated kernel (test_dalpha_isolated.py)")
print("=" * 70)

from test_dalpha_isolated import test_dalpha_isolated
# We can't easily run this, so let's skip for now

# Test 2: Full backward
print("\n" + "=" * 70)
print("Test 2: Full backward implementation")
print("=" * 70)

from src.backward.mhc_backward_triton import mhc_backward_triton
dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = mhc_backward_triton(
    x_dev, phi_dev, alpha_dev, bias_dev,
    inv_rms_dev, h_mix_dev, h_pre_dev, h_post_dev,
    dh_in_dev, dh_post_dev, dh_res_dev, gamma_dev
)

print(f"dalpha from full backward: {dalpha_tri.cpu().numpy()}")

# Test 3: Manual computation (golden)
print("\n" + "=" * 70)
print("Test 3: Manual golden computation")
print("=" * 70)

dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
    x, phi, alpha, bias,
    inv_rms, h_mix, h_pre, h_post,
    dh_in, dh_post, dh_res, gamma
)

print(f"dalpha from golden: {dalpha_gold.numpy()}")

print("\n" + "=" * 70)
print("Comparison")
print("=" * 70)
print(f"dalpha[0] error: {abs(dalpha_tri[0] - dalpha_gold[0]):.8f}")
print(f"dalpha[1] error: {abs(dalpha_tri[1] - dalpha_gold[1]):.8f}")
print(f"dalpha[2] error: {abs(dalpha_tri[2] - dalpha_gold[2]):.8f}")

# The mystery: why is dalpha[0] wrong when both dh_pre2 and h_pre1_hmix are correct?
print("\n" + "=" * 70)
print("Investigation")
print("=" * 70)

# Check if maybe the issue is that we need to verify the ACTUAL values
# computed inside kernel 1, not just the inputs

print("Known facts:")
print("  1. dh_pre2 computed by kernel 1 is correct (verified)")
print("  2. h_pre1_hmix loaded by kernel 1 is correct (verified)")
print("  3. Isolated kernel test passes (error < 5e-7)")
print("  4. Full kernel 1 produces wrong dalpha_pre (error = 1.13)")
print("")
print("Hypothesis: There must be some other computation in kernel 1 that")
print("             is interfering with the dalpha_pre calculation.")
print("")
print("Possible causes:")
print("  1. Memory aliasing or shared memory conflicts")
print("  2. Register spilling causing precision loss")
print("  3. Unexpected dependency between computations")
print("  4. Bug in atomic_add accumulation")
