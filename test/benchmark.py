"""
Standalone Benchmark Script for MHC Forward Pre

Simple benchmark comparing PyTorch reference vs optimized implementations.
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
        device = args[0].device if isinstance(args[0], torch.Tensor) else torch.device('cpu')

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = func(*args, **kwargs)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
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


def pytorch_reference(x, phi, alpha, bias, norm_eps=1e-6, hc_eps=1e-6):
    """PyTorch reference implementation (standalone)."""
    B, S, n, D = x.shape
    nD = n * D

    # Flatten
    vecX = x.reshape(B, S, nD).float()

    # GEMM
    h_mix = torch.matmul(vecX, phi.t())

    # RMSNorm
    inv_rms = torch.rsqrt(vecX.square().mean(-1, keepdim=True) + norm_eps)
    h_mix_tmp = h_mix * inv_rms

    # Split
    h_pre1, h_post1, h_res1 = torch.split(h_mix_tmp, [n, n, n * n], dim=-1)
    h_res2 = h_res1.reshape(B, S, n, n)

    # Alpha and bias
    a_pre, a_post, a_res = alpha

    h_pre2 = a_pre * h_pre1 + bias[:n]
    h_post2 = a_post * h_post1 + bias[n:2*n]
    h_res = a_res * h_res2 + bias[2*n:].view(n, n)

    # Sigmoid
    h_pre = torch.sigmoid(h_pre2) + hc_eps
    h_post = 2.0 * torch.sigmoid(h_post2)

    # h_in = h_pre @ x
    h_in_fp = (h_pre.unsqueeze(-1) * x.float()).sum(dim=2)
    h_in = h_in_fp.to(x.dtype)

    return h_in, h_post, h_res


def run_benchmark_suite():
    """Run complete benchmark suite."""

    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "MHC Forward Pre - Benchmark Suite" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")

    # Test configurations
    configs = [
        {"name": "Small",  "B": 1, "S": 128, "n": 4, "D": 128},
        {"name": "Medium", "B": 2, "S": 512, "n": 4, "D": 256},
        {"name": "Large",  "B": 1, "S": 2048, "n": 4, "D": 512},
        {"name": "XL",     "B": 1, "S": 4096, "n": 4, "D": 2560},
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

        # Get reference output for accuracy check
        with torch.no_grad():
            h_in_ref, h_post_ref, h_res_ref = pytorch_reference(x, phi, alpha, bias)

        # Benchmark PyTorch
        print(f"\n[1/2] Benchmarking PyTorch Reference...")
        bench = Benchmark(warmup=10, iters=100)
        ref_result = bench.run(pytorch_reference, x, phi, alpha, bias)
        print(f"       Latency: {ref_result.avg_latency_ms:.4f} ± {ref_result.std_latency_ms:.4f} ms")
        print(f"       Throughput: {ref_result.throughput:.2f} tokens/s")

        # Try to import and benchmark Triton
        tri_result = None
        try:
            from mhc_forward_pre_triton import mhc_forward_pre_triton_optimized

            print(f"\n[2/2] Benchmarking Triton...")

            def triton_wrapper(x, phi, alpha, bias):
                return mhc_forward_pre_triton_optimized(x, phi, alpha, bias)

            tri_result = bench.run(triton_wrapper, x, phi, alpha, bias)
            print(f"       Latency: {tri_result.avg_latency_ms:.4f} ± {tri_result.std_latency_ms:.4f} ms")
            print(f"       Throughput: {tri_result.throughput:.2f} tokens/s")
            print(f"       Speedup: {ref_result.avg_latency_ms / tri_result.avg_latency_ms:.2f}x")

            # Check accuracy
            with torch.no_grad():
                h_in_tri, h_post_tri, h_res_tri = triton_wrapper(x, phi, alpha, bias)

            h_in_err = torch.abs(h_in_tri.cpu().float() - h_in_ref.cpu().float()).max().item()
            h_post_err = torch.abs(h_post_tri.cpu() - h_post_ref.cpu()).max().item()
            h_res_err = torch.abs(h_res_tri.cpu() - h_res_ref.cpu()).max().item()

            max_err = max(h_in_err, h_post_err, h_res_err)
            status = "✓ PASS" if max_err < 1e-3 else "✗ FAIL"
            print(f"       Max Error: {max_err:.6f} {status}")

        except ImportError as e:
            print(f"\n[2/2] Triton not available: {e}")
            tri_result = None

        # Store results
        results[name] = {
            "config": config,
            "pytorch": ref_result,
            "triton": tri_result,
        }

        # Cleanup
        del x, phi, alpha, bias
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Config':<10} {'PyTorch(ms)':<15} {'Triton(ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    for name, res in results.items():
        pytorch_ms = res["pytorch"].avg_latency_ms
        triton_ms = res["triton"].avg_latency_ms if res["triton"] else float('inf')
        speedup = f"{pytorch_ms / triton_ms:.2f}x" if res["triton"] else "N/A"
        triton_str = f"{triton_ms:.4f}" if res["triton"] else "N/A"

        print(f"{name:<10} {pytorch_ms:<15.4f} {triton_str:<15} {speedup:<10}")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    results = run_benchmark_suite()
