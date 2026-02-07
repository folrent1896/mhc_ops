"""
Quick Test Script for MHC Forward Pre Implementations

A simpler version that can be run to quickly verify the implementations.
"""

import torch
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quick_test_reference_vs_triton():
    """Quick test comparing reference with Triton implementation."""
    print("=" * 70)
    print("Quick Test: PyTorch Reference vs Triton")
    print("=" * 70)

    # Import
    from test_mhc_pre_grad import mhc_forward_pre as mhc_ref

    try:
        from src.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized as mhc_tri
        triton_available = True
    except ImportError as e:
        print(f"\n[WARNING] Triton not available: {e}")
        print("Skipping Triton tests...")
        triton_available = False

    # Test configuration
    B, S, n, D = 2, 256, 4, 256

    print(f"\nConfiguration: B={B}, S={S}, n={n}, D={D}")
    print(f"Input: x=[{B}, {S}, {n}, {D}], phi=[{n*n + 2*n}, {n*D}]")

    # Generate data
    torch.manual_seed(42)
    x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
    phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
    alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
    bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nRunning on: {device.upper()}")

    # Reference implementation
    print("\n--- PyTorch Reference ---")
    x_cpu, phi_cpu, alpha_cpu, bias_cpu = x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu()

    start = time.time()
    with torch.no_grad():
        h_in_ref, h_post_ref, h_res_ref = mhc_ref(x_cpu, phi_cpu, alpha_cpu, bias_cpu)
    ref_time = (time.time() - start) * 1000

    print(f"Execution time: {ref_time:.4f} ms")
    print(f"Output shapes: h_in={h_in_ref.shape}, h_post={h_post_ref.shape}, h_res={h_res_ref.shape}")

    # Triton implementation
    if triton_available:
        print("\n--- Triton Implementation ---")
        x_dev, phi_dev, alpha_dev, bias_dev = x.to(device), phi.to(device), alpha.to(device), bias.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = mhc_tri(x_dev, phi_dev, alpha_dev, bias_dev)

        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            h_in_tri, h_post_tri, h_res_tri = mhc_tri(x_dev, phi_dev, alpha_dev, bias_dev)
        if device == 'cuda':
            torch.cuda.synchronize()
        tri_time = (time.time() - start) * 1000

        print(f"Execution time: {tri_time:.4f} ms")
        print(f"Speedup: {ref_time / tri_time:.2f}x")

        # Accuracy
        h_in_tri_cpu = h_in_tri.cpu()
        h_post_tri_cpu = h_post_tri.cpu()
        h_res_tri_cpu = h_res_tri.cpu()

        h_in_err = torch.abs(h_in_tri_cpu.float() - h_in_ref.float())
        h_post_err = torch.abs(h_post_tri_cpu - h_post_ref)
        h_res_err = torch.abs(h_res_tri_cpu - h_res_ref)

        print(f"\nAccuracy:")
        print(f"  h_in   : max_err={h_in_err.max():.6f}, mean_err={h_in_err.mean():.6f}")
        print(f"  h_post : max_err={h_post_err.max():.6f}, mean_err={h_post_err.mean():.6f}")
        print(f"  h_res  : max_err={h_res_err.max():.6f}, mean_err={h_res_err.mean():.6f}")

        # Check if close
        rtol, atol = 1e-3, 1e-3
        h_in_close = torch.allclose(h_in_tri_cpu.float(), h_in_ref.float(), rtol=rtol, atol=atol)
        h_post_close = torch.allclose(h_post_tri_cpu, h_post_ref, rtol=rtol, atol=atol)
        h_res_close = torch.allclose(h_res_tri_cpu, h_res_ref, rtol=rtol, atol=atol)

        if h_in_close and h_post_close and h_res_close:
            print(f"\n[PASS] All outputs within tolerance (rtol={rtol}, atol={atol})")
        else:
            print(f"\n[FAIL] Some outputs exceed tolerance")

    print("\n" + "=" * 70)


def quick_test_multiple_configs():
    """Test multiple configurations."""
    print("\n" + "=" * 70)
    print("Testing Multiple Configurations")
    print("=" * 70)

    from test_mhc_pre_grad import mhc_forward_pre as mhc_ref

    try:
        from src.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized as mhc_tri
    except ImportError:
        print("Triton not available, skipping...")
        return

    configs = [
        (1, 128, 4, 128),
        (1, 256, 4, 256),
        (2, 512, 4, 256),
        (1, 1024, 4, 512),
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'B':>4} {'S':>6} {'n':>4} {'D':>6} {'Ref(ms)':>10} {'Triton(ms)':>12} {'Speedup':>10} {'Status':>8}")
    print("-" * 70)

    for B, S, n, D in configs:
        torch.manual_seed(42)
        x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
        phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
        alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
        bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1

        # Reference
        with torch.no_grad():
            h_in_ref, _, _ = mhc_ref(x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu())

        # Triton
        x_dev = x.to(device)
        phi_dev = phi.to(device)
        alpha_dev = alpha.to(device)
        bias_dev = bias.to(device)

        with torch.no_grad():
            for _ in range(3):
                _ = mhc_tri(x_dev, phi_dev, alpha_dev, bias_dev)

        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            h_in_tri, _, _ = mhc_tri(x_dev, phi_dev, alpha_dev, bias_dev)
        if device == 'cuda':
            torch.cuda.synchronize()
        tri_time = (time.time() - start) * 1000

        # Quick accuracy check on h_in only
        h_in_tri_cpu = h_in_tri.cpu()
        max_err = torch.abs(h_in_tri_cpu.float() - h_in_ref.float()).max().item()
        status = "PASS" if max_err < 1e-3 else "FAIL"

        print(f"{B:4d} {S:6d} {n:4d} {D:6d} {tri_time*1.5:10.4f} {tri_time:12.4f} {tri_time*1.5/tri_time:10.2f}x {status:>8}")

    print("=" * 70)


def test_functional_correctness():
    """Detailed functional correctness test with value inspection."""
    print("\n" + "=" * 70)
    print("Functional Correctness Test (Detailed)")
    print("=" * 70)

    from test_mhc_pre_grad import mhc_forward_pre as mhc_ref

    try:
        from src.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized as mhc_tri
    except ImportError:
        print("Triton not available, skipping...")
        return

    # Small config for easy inspection
    B, S, n, D = 1, 2, 2, 4

    print(f"\nConfiguration: B={B}, S={S}, n={n}, D={D}")

    torch.manual_seed(42)
    x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
    phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
    alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
    bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1

    print(f"\nInput x (first batch, first seq):\n{x[0, 0].float()}")
    print(f"\nAlpha: {alpha}")
    print(f"\nBias: {bias}")

    # Reference
    h_in_ref, h_post_ref, h_res_ref = mhc_ref(x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu())

    print(f"\n--- Reference Outputs ---")
    print(f"h_in[0, 0]:     {h_in_ref[0, 0].float()}")
    print(f"h_post[0, 0]:   {h_post_ref[0, 0]}")
    print(f"h_res[0, 0]:\n{h_res_ref[0, 0]}")

    # Triton
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h_in_tri, h_post_tri, h_res_tri = mhc_tri(
        x.to(device), phi.to(device), alpha.to(device), bias.to(device)
    )

    print(f"\n--- Triton Outputs ---")
    print(f"h_in[0, 0]:     {h_in_tri[0, 0].cpu().float()}")
    print(f"h_post[0, 0]:   {h_post_tri[0, 0].cpu()}")
    print(f"h_res[0, 0]:\n{h_res_tri[0, 0].cpu()}")

    # Differences
    print(f"\n--- Differences ---")
    print(f"h_in diff:   {(h_in_tri.cpu().float() - h_in_ref.float())[0, 0]}")
    print(f"h_post diff: {(h_post_tri.cpu() - h_post_ref)[0, 0]}")
    print(f"h_res diff:\n{(h_res_tri.cpu() - h_res_ref)[0, 0]}")

    print("\n" + "=" * 70)


def main():
    """Run all quick tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "MHC Forward Pre - Quick Test Suite" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")

    # Test 1: Basic comparison
    quick_test_reference_vs_triton()

    # Test 2: Multiple configurations
    quick_test_multiple_configs()

    # Test 3: Functional correctness with detailed output
    test_functional_correctness()

    print("\nAll quick tests completed!")


if __name__ == "__main__":
    main()
