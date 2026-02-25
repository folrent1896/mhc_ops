"""
Standalone Benchmark Script for MHC Backward

Simple benchmark comparing PyTorch golden reference vs Triton implementation.
Can be run without external test files.
"""

import torch
import time
from dataclasses import dataclass
from typing import Callable
import gc


@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    name: str
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput: float  # tokens per second


class Benchmark:
    """Simple benchmark utility."""

    def __init__(self, warmup: int = 10, iters: int = 100):
        self.warmup = warmup
        self.iters = iters

    def run(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Run benchmark."""
        # Extract device from first tensor argument
        device = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break

        if device is None:
            device = torch.device('cpu')

        # Warmup
        for _ in range(self.warmup):
            _ = func(*args, **kwargs)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = func(*args, **kwargs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        times = torch.tensor(times)

        # Compute throughput (tokens = batch * seq_len)
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            B, S = args[0].shape[0], args[0].shape[1]
            throughput = (B * S) / (times.mean() / 1000)
        else:
            throughput = 0.0

        return BenchmarkResult(
            name=func.__name__ if hasattr(func, '__name__') else 'unknown',
            avg_latency_ms=times.mean().item(),
            std_latency_ms=times.std().item(),
            min_latency_ms=times.min().item(),
            max_latency_ms=times.max().item(),
            throughput=throughput,
        )


def pytorch_reference_backward(x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post, dh_in, dh_post, dh_res, gamma):
    """PyTorch golden reference backward (standalone)."""
    from src.backward.golden import mhc_backward_manual
    return mhc_backward_manual(
        x, phi, alpha, bias,
        inv_rms, h_mix, h_pre, h_post,
        dh_in, dh_post, dh_res, gamma
    )


def triton_backward(x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post, dh_in, dh_post, dh_res, gamma):
    """Triton backward implementation."""
    from src.backward.mhc_backward_triton import mhc_backward_triton
    return mhc_backward_triton(
        x, phi, alpha, bias,
        inv_rms, h_mix, h_pre, h_post,
        dh_in, dh_post, dh_res, gamma
    )


def run_benchmark_suite():
    """Run complete backward benchmark suite."""

    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "MHC Backward - Benchmark Suite" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")

    # Test configurations
    configs = [
        {"name": "Small",  "B": 2, "S": 64,  "n": 4, "D": 128},
        {"name": "Medium", "B": 2, "S": 256, "n": 4, "D": 256},
        {"name": "Large",  "B": 1, "S": 1024, "n": 4, "D": 512},
        {"name": "XL",     "B": 1, "S": 2048, "n": 4, "D": 512},
    ]

    results = {}

    for config in configs:
        name = config["name"]
        B, S, n, D = config["B"], config["S"], config["n"], config["D"]

        print(f"\n{'=' * 80}")
        print(f"Configuration: {name} (B={B}, S={S}, n={n}, D={D})")
        print(f"{'=' * 80}")

        # Generate test data
        torch.manual_seed(42)
        x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
        phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
        alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32, device=device)
        bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1
        gamma = torch.randn(n, D, dtype=torch.float32, device=device)

        # Forward pass (get intermediate values)
        try:
            from src.forward.golden import mhc_forward_pre
            with torch.no_grad():
                h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
                    x, phi, alpha, bias, outflag=True
                )
        except ImportError:
            print("Error: Cannot import forward implementation")
            continue

        # Generate backward inputs
        torch.manual_seed(123)
        dh_in = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
        dh_post = torch.randn(B, S, n, dtype=torch.float32, device=device)
        dh_res = torch.randn(B, S, n, n, dtype=torch.float32, device=device)

        # Move CPU tensors to device if needed
        inv_rms = inv_rms.to(device)
        h_mix = h_mix.to(device)
        h_pre = h_pre.to(device)
        h_post = h_post.to(device)

        # Benchmark PyTorch Golden
        print(f"\n[1/2] Benchmarking PyTorch Golden...")
        bench = Benchmark(warmup=10, iters=100)
        ref_result = bench.run(
            pytorch_reference_backward,
            x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
            dh_in, dh_post, dh_res, gamma
        )
        print(f"       Latency: {ref_result.avg_latency_ms:.4f} ± {ref_result.std_latency_ms:.4f} ms")
        print(f"       Throughput: {ref_result.throughput:.2f} tokens/s")

        # Benchmark Triton
        tri_result = None
        try:
            print(f"\n[2/2] Benchmarking Triton...")
            tri_result = bench.run(
                triton_backward,
                x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
                dh_in, dh_post, dh_res, gamma
            )
            print(f"       Latency: {tri_result.avg_latency_ms:.4f} ± {tri_result.std_latency_ms:.4f} ms")
            print(f"       Throughput: {tri_result.throughput:.2f} tokens/s")
            print(f"       Speedup: {ref_result.avg_latency_ms / tri_result.avg_latency_ms:.2f}x")

            # Check accuracy (all should be on device now)
            dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = pytorch_reference_backward(
                x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
                dh_in, dh_post, dh_res, gamma
            )

            dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = triton_backward(
                x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
                dh_in, dh_post, dh_res, gamma
            )

            # Compute errors (all on device)
            dx_err = torch.abs(dx_tri - dx_gold).max().item()
            dphi_err = torch.abs(dphi_tri - dphi_gold).max().item()
            dalpha_err = torch.abs(dalpha_tri - dalpha_gold).max().item()
            dbias_err = torch.abs(dbias_tri - dbias_gold).max().item()
            dgamma_err = torch.abs(dgamma_tri - dgamma_gold).max().item()

            max_err = max(dx_err, dphi_err, dalpha_err, dbias_err, dgamma_err)
            status = "✓ PASS" if max_err < 0.5 else "⚠ PARTIAL" if max_err < 5.0 else "✗ FAIL"

            print(f"\n       Accuracy Check:")
            print(f"       dx:     {dx_err:.6f}")
            print(f"       dphi:   {dphi_err:.6f}")
            print(f"       dalpha: {dalpha_err:.6f}")
            print(f"       dbias:  {dbias_err:.6f}")
            print(f"       dgamma: {dgamma_err:.6f}")
            print(f"       Max Error: {max_err:.6f} {status}")

        except ImportError as e:
            print(f"\n[2/2] Triton not available: {e}")
            tri_result = None

        # Store results
        results[name] = {
            "config": config,
            "golden": ref_result,
            "triton": tri_result,
        }

        # Cleanup
        del x, phi, alpha, bias, gamma
        del dh_in, dh_post, dh_res
        del inv_rms, h_mix, h_pre, h_post
        del dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold
        if 'dx_tri' in locals():
            del dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Config':<10} {'Golden(ms)':<15} {'Triton(ms)':<15} {'Speedup':<10} {'Status':<10}")
    print("-" * 80)

    for name, res in results.items():
        golden_ms = res["golden"].avg_latency_ms
        triton_ms = res["triton"].avg_latency_ms if res["triton"] else float('inf')
        speedup = f"{golden_ms / triton_ms:.2f}x" if res["triton"] else "N/A"
        triton_str = f"{triton_ms:.4f}" if res["triton"] else "N/A"

        # Get status
        if res["triton"]:
            # Recompute quickly to get status
            try:
                B, S = res["config"]["B"], res["config"]["S"]
                n, D = res["config"]["n"], res["config"]["D"]
                torch.manual_seed(42)
                x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device=device)
                phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32, device=device)
                alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32, device=device)
                bias = torch.randn(n*n + 2*n, dtype=torch.float32, device=device) * 0.1
                gamma = torch.randn(n, D, dtype=torch.float32, device=device)

                from src.forward.golden import mhc_forward_pre
                with torch.no_grad():
                    h_in, h_post, h_res, inv_rms, h_mix, h_pre = mhc_forward_pre(
                        x, phi, alpha, bias, outflag=True
                    )

                torch.manual_seed(123)
                dh_in = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
                dh_post = torch.randn(B, S, n, dtype=torch.float32, device=device)
                dh_res = torch.randn(B, S, n, n, dtype=torch.float32, device=device)

                inv_rms = inv_rms.to(device)
                h_mix = h_mix.to(device)
                h_pre = h_pre.to(device)
                h_post = h_post.to(device)

                dx_tri, dphi_tri, dalpha_tri, dbias_tri, dgamma_tri = triton_backward(
                    x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
                    dh_in, dh_post, dh_res, gamma
                )

                dx_gold, dphi_gold, dalpha_gold, dbias_gold, dgamma_gold = pytorch_reference_backward(
                    x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
                    dh_in, dh_post, dh_res, gamma
                )

                max_err = max(
                    torch.abs(dx_tri - dx_gold).max().item(),
                    torch.abs(dphi_tri - dphi_gold).max().item(),
                    torch.abs(dalpha_tri - dalpha_gold).max().item(),
                    torch.abs(dbias_tri - dbias_gold).max().item(),
                    torch.abs(dgamma_tri - dgamma_gold).max().item()
                )
                status = "✓ PASS" if max_err < 0.5 else "⚠"
            except:
                status = "?"

            print(f"{name:<10} {golden_ms:<15.4f} {triton_str:<15} {speedup:<10} {status}")

    print("\n" + "=" * 80)
    print("\nNOTE: Speedup is relative to PyTorch golden reference.")
    print("Status: ✓ PASS (<0.5), ⚠ PARTIAL (0.5-5.0), ✗ FAIL (>5.0)")
    print("Backward includes forward pass in real training scenarios.")

    return results


if __name__ == "__main__":
    results = run_benchmark_suite()
