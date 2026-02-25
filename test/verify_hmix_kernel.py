"""
Verify that h_mix is being loaded correctly in kernel 1
"""

import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, '.')

from src.forward.golden import mhc_forward_pre

B, S, n, D = 2, 64, 4, 128
device = 'cuda'
out_features = n * n + 2 * n

torch.manual_seed(42)
x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device='cpu')
phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device='cpu')
alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
bias = torch.randn(n*n + 2*n, dtype=torch.float32, device='cpu') * 0.1

# Forward pass
with torch.no_grad():
    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
        x, phi, alpha, bias, outflag=True
    )

# Move to CUDA
x_dev = x.to(device)
h_mix_dev = h_mix.to(device)
inv_rms_dev = inv_rms.to(device)

# Kernel to load and verify h_pre1_hmix
@triton.jit
def verify_hmix_kernel(
    h_mix_ptr,
    inv_rms_ptr,
    output_ptr,
    B, S, n, out_features,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b_idx = pid // S
    s_idx = pid % S

    x_off_n = tl.arange(0, BLOCK_SIZE_N)
    h_pre_mask = x_off_n < n

    # Load inv_rms
    inv_rms_off = b_idx * S + s_idx
    inv_rms = tl.load(inv_rms_ptr + inv_rms_off)

    # Load h_pre1_hmix = h_mix[:, :, 0:n] * inv_rms
    h_pre1_hmix = tl.load(h_mix_ptr + (b_idx * S * out_features + s_idx * out_features + x_off_n),
                          mask=h_pre_mask, other=0.0) * inv_rms

    # Store output
    output_offset = (b_idx * S * n + s_idx * n + x_off_n)
    tl.store(output_ptr + output_offset, h_pre1_hmix, mask=h_pre_mask)

# Allocate output
output = torch.zeros(B, S, n, dtype=torch.float32, device=device)

BLOCK_SIZE_N = triton.next_power_of_2(n)
grid = (B * S,)

with torch.cuda.device(device):
    verify_hmix_kernel[grid](
        h_mix_ptr=h_mix_dev,
        inv_rms_ptr=inv_rms_dev,
        output_ptr=output,
        B=B, S=S, n=n,
        out_features=out_features,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

# Compute expected value on CPU
h_mix_tmp = h_mix * inv_rms[:, :, None]
h_pre1_expected = h_mix_tmp[:, :, 0:n]

# Compare
print("Comparison of h_pre1_hmix:")
print(f"  Expected (CPU): {h_pre1_expected[0, 0, :].numpy()}")
print(f"  Kernel (CUDA):  {output[0, 0, :].cpu().numpy()}")
print(f"  Max error: {torch.abs(output.cpu() - h_pre1_expected).max().item():.8f}")
print(f"  Mean error: {torch.abs(output.cpu() - h_pre1_expected).mean().item():.8f}")

# Check a few more elements
for b in [0, 1]:
    for s in [0, 1, 63]:
        expected_val = h_pre1_expected[b, s, 0].item()
        kernel_val = output[b, s, 0].cpu().item()
        err = abs(kernel_val - expected_val)
        print(f"  [{b}, {s}, 0]: expected={expected_val:.6f}, kernel={kernel_val:.6f}, err={err:.8f}")
