"""
Test MHC Backward Implementations

Tests Triton and TileLang backward implementations against golden reference.
"""

import torch
import sys
sys.path.insert(0, '..')

from src.forward.golden import mhc_forward_pre
from src.backward.golden import mhc_backward_manual


def test_backward_triton_vs_golden():
    """Test Triton backward against golden reference.

    Current status (2025-02-25):
    - dphi: ✓ PASS (fully correct)
    - dalpha[1:]: ✓ PASS (dalpha_post, dalpha_res correct)
    - dalpha[0]: ✗ PARTIAL (dalpha_pre has errors)
    - dbias: ✗ PARTIAL (some sections correct, others have errors)
    - dgamma: ✗ PARTIAL (has errors)
    - dx: ✗ PARTIAL (has errors)

    The multi-kernel architecture is working. Remaining issues appear to be
    related to indexing or accumulation in specific sections.
    """
    print("=" * 70)
    print("Testing Triton Backward vs Golden Reference")
    print("=" * 70)
    print("\n[INFO] Multi-kernel Triton implementation")
    print("       Status: dphi PASSING, other components PARTIAL")

    try:
        from src.backward.mhc_backward_triton import mhc_backward_triton
        triton_available = True
    except ImportError as e:
        print(f"\n[WARNING] Triton not available: {e}")
        triton_available = False
        return

    # Test parameters
    B, S, n, D = 2, 64, 4, 128

    print(f"\nConfiguration: B={B}, S={S}, n={n}, D={D}")

    # Create test data
    torch.manual_seed(42)
    x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
    phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
    alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
    bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1
    gamma = torch.randn(n, D, dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
            x, phi, alpha, bias, outflag=True
        )

    # Random gradients
    dh_in = torch.randn(B, S, D, dtype=torch.bfloat16)
    dh_post = torch.randn(B, S, n, dtype=torch.float32)
    dh_res = torch.randn(B, S, n, n, dtype=torch.float32)

    # Golden reference backward
    print("\n[1/3] Computing golden reference backward...")
    dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
        x, phi, alpha, bias,
        inv_rms, h_mix, h_pre, h_post,
        dh_in, dh_post, dh_res, gamma
    )

    # Triton backward
    print("[2/3] Computing Triton backward...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # Compare
    print("[3/3] Comparing results...")

    def compare(name, tri, gold, rtol=1e-3, atol=1e-3):
        tri_cpu = tri.cpu()
        gold_cpu = gold.cpu()
        max_err = torch.abs(tri_cpu - gold_cpu).max().item()
        mean_err = torch.abs(tri_cpu - gold_cpu).mean().item()
        passed = torch.allclose(tri_cpu, gold_cpu, rtol=rtol, atol=atol)
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<12}: max_err={max_err:.6f}, mean_err={mean_err:.6f} [{status}]")
        return passed

    print("\n--- Gradient Comparison ---")
    r1 = compare("dx", dx_tri, dx_gold, rtol=1e-2, atol=1e-2)
    r2 = compare("dphi", dphi_tri, dphi_gold)
    r3 = compare("dalpha", dalpha_tri, dalpha_gold)
    r4 = compare("dbias", dbias_tri, dbias_gold)
    r5 = compare("dgamma", dgamma_tri, dgamma_gold)

    if r1 and r2 and r3 and r4 and r5:
        print("\n[PASS] All gradients within tolerance")
    else:
        print("\n[FAIL] Some gradients exceed tolerance")

    print("\n" + "=" * 70)
    return r1 and r2 and r3 and r4 and r5


def test_backward_tilelang_vs_golden():
    """Test TileLang backward against golden reference."""
    # Commented out - needs API rewrite
    print("\n" + "=" * 70)
    print("Testing TileLang Backward vs Golden Reference")
    print("=" * 70)

    # Test parameters
    B, S, n, D = 1, 32, 4, 64

    print(f"\nConfiguration: B={B}, S={S}, n={n}, D={D}")
    print("\n[INFO] TileLang/TVM compilation may take a while...")

    # Create test data
    torch.manual_seed(42)
    x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
    phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
    alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
    bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1
    gamma = torch.randn(n, D, dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
            x, phi, alpha, bias, outflag=True
        )

    # Random gradients
    dh_in = torch.randn(B, S, D, dtype=torch.bfloat16)
    dh_post = torch.randn(B, S, n, dtype=torch.float32)
    dh_res = torch.randn(B, S, n, n, dtype=torch.float32)

    # Golden reference backward
    print("\n[1/2] Computing golden reference backward...")
    dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = mhc_backward_manual(
        x, phi, alpha, bias,
        inv_rms, h_mix, h_pre, h_post,
        dh_in, dh_post, dh_res, gamma
    )

    # TileLang backward
    print("[2/2] Computing TileLang backward...")

    try:
        from src.backward.mhc_backward_tilelang import MHCBackwardTileLang

        # Compile and run
        op = MHCBackwardTileLang(B, S, n, D)

        dx_tl, dphi_tl, dalpha_tl, dbias_tl, dgamma_tl = op(
            x, phi, alpha, bias,
            inv_rms, h_mix, h_pre, h_post,
            dh_in, dh_post, dh_res, gamma
        )

        # Compare
        print("\n--- Gradient Comparison ---")
        r1 = torch.allclose(dx_tl, dx_gold, rtol=1e-3, atol=1e-3)
        r2 = torch.allclose(dphi_tl, dphi_gold, rtol=1e-3, atol=1e-3)
        r3 = torch.allclose(dalpha_tl, dalpha_gold, rtol=1e-3, atol=1e-3)
        r4 = torch.allclose(dbias_tl, dbias_gold, rtol=1e-3, atol=1e-3)
        r5 = torch.allclose(dgamma_tl, dgamma_gold, rtol=1e-3, atol=1e-3)

        print(f"  dx:     {torch.abs(dx_tl - dx_gold).max().item():.6f} {'PASS' if r1 else 'FAIL'}")
        print(f"  dphi:   {torch.abs(dphi_tl - dphi_gold).max().item():.6f} {'PASS' if r2 else 'FAIL'}")
        print(f"  dalpha: {torch.abs(dalpha_tl - dalpha_gold).max().item():.6f} {'PASS' if r3 else 'FAIL'}")
        print(f"  dbias:  {torch.abs(dbias_tl - dbias_gold).max().item():.6f} {'PASS' if r4 else 'FAIL'}")
        print(f"  dgamma: {torch.abs(dgamma_tl - dgamma_gold).max().item():.6f} {'PASS' if r5 else 'FAIL'}")

        if r1 and r2 and r3 and r4 and r5:
            print("\n[PASS] All gradients within tolerance")
        else:
            print("\n[FAIL] Some gradients exceed tolerance")

        print("\n" + "=" * 70)
        return r1 and r2 and r3 and r4 and r5

    except ImportError as e:
        print(f"[ERROR] TileLang not available: {e}")
        print("[SKIP] Skipping TileLang test")
        print("\n" + "=" * 70)
        return None


def main():
    """Run all backward tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "MHC Backward - Test Suite" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")

    # Test 1: Triton backward
    result_triton = test_backward_triton_vs_golden()

    # Test 2: TileLang backward
    result_tilelang = test_backward_tilelang_vs_golden()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if result_triton is not None:
        print(f"\nTriton:    {'✓ PASS' if result_triton else '✗ FAIL'}")

    if result_tilelang is not None:
        print(f"TileLang:  {'✓ PASS' if result_tilelang else '✗ FAIL'}")
    elif result_tilelang is None:
        print(f"TileLang:  ⊘ SKIPPED (not available)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
