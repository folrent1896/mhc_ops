"""
Test Script for MHC Forward Pre Operator Implementations

Tests both Triton and TileLang implementations against the reference PyTorch implementation.
Measures accuracy (max error, relative error) and performance (latency, throughput).
"""

import torch
import time
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import reference implementation
from src.forward import mhc_forward_pre


@dataclass
class TestResult:
    """Store test results for each implementation."""
    name: str
    h_in_max_error: float
    h_post_max_error: float
    h_res_max_error: float
    h_in_mean_error: float
    h_post_mean_error: float
    h_res_mean_error: float
    passed: bool
    latency_ms: float = 0.0
    throughput: float = 0.0  # sequences per second
    speedup_vs_ref: float = 0.0


class MHCOperatorTester:
    """Tester for MHC Forward Pre operator implementations."""

    def __init__(
        self,
        device: str = "cuda",
        warmup_iters: int = 10,
        benchmark_iters: int = 100,
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ):
        """
        Initialize tester.

        Args:
            device: Device to run tests on ("cuda" or "cpu")
            warmup_iters: Number of warmup iterations before benchmarking
            benchmark_iters: Number of benchmark iterations
            rtol: Relative tolerance for accuracy check
            atol: Absolute tolerance for accuracy check
        """
        self.device = torch.device(device) if torch.device(device).type == "cuda" else torch.device("cpu")
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.rtol = rtol
        self.atol = atol

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")

        # Import implementations
        self.triton_impl = self._load_triton()
        self.tilelang_impl = self._load_tilelang()

    def _load_triton(self):
        """Try to load Triton implementation."""
        try:
            from src.forward.mhc_forward_pre_triton import mhc_forward_pre_triton_optimized
            return mhc_forward_pre_triton_optimized
        except ImportError as e:
            print(f"WARNING: Could not import Triton implementation: {e}")
            return None

    def _load_tilelang(self):
        """Try to load TileLang implementation."""
        try:
            from src.forward.mhc_forward_pre_tilelang import mhc_forward_pre_tvm
            # TileLang returns a compiled function, need wrapper
            def tilelang_wrapper(x, phi, alpha, bias, outflag=False, norm_eps=1e-6, hc_eps=1e-6):
                B, S, n, D = x.shape
                out_features = n * n + 2 * n

                # Compile and run (simplified for testing)
                func = mhc_forward_pre_tvm(B, S, n, D, norm_eps=norm_eps, hc_eps=hc_eps)

                # Allocate outputs
                h_in = torch.empty(B, S, D, dtype=x.dtype, device=x.device)
                h_post = torch.empty(B, S, n, dtype=torch.float32, device=x.device)
                h_res = torch.empty(B, S, n, n, dtype=torch.float32, device=x.device)

                # Run
                func(x, phi, alpha, bias, h_in, h_post, h_res)

                return h_in, h_post, h_res
            return tilelang_wrapper
        except ImportError as e:
            print(f"WARNING: Could not import TileLang implementation: {e}")
            return None

    def generate_test_data(
        self,
        B: int,
        S: int,
        n: int,
        D: int,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random test data."""
        torch.manual_seed(seed)

        x = torch.randn(B, S, n, D, dtype=torch.bfloat16)
        phi = torch.randn(n*n + 2*n, n*D, dtype=torch.float32)
        alpha = torch.tensor([1.1, 0.9, 1.05], dtype=torch.float32)
        bias = torch.randn(n*n + 2*n, dtype=torch.float32) * 0.1

        return x, phi, alpha, bias

    def compute_reference(
        self,
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reference output using PyTorch implementation."""
        with torch.no_grad():
            h_in, h_post, h_res = mhc_forward_pre(
                x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu()
            )
        return h_in, h_post, h_res

    def compute_accuracy(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        name: str,
    ) -> Tuple[float, float, bool]:
        """
        Compute accuracy metrics.

        Returns:
            max_error: Maximum absolute error
            mean_error: Mean absolute error
            passed: Whether result is within tolerance
        """
        # Ensure same device and dtype for comparison
        output = output.float().cpu()
        reference = reference.float().cpu()

        abs_error = torch.abs(output - reference)
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()

        # Check if within tolerance
        close = torch.allclose(output, reference, rtol=self.rtol, atol=self.atol)
        passed = close.item()

        return max_error, mean_error, passed

    def benchmark(
        self,
        func,
        *args,
        **kwargs,
    ) -> float:
        """
        Benchmark a function.

        Returns:
            Average latency in milliseconds
        """
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iters):
                _ = func(*args, **kwargs)

        # Synchronize
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(self.benchmark_iters):
                _ = func(*args, **kwargs)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        avg_latency_ms = (end_time - start_time) / self.benchmark_iters * 1000

        return avg_latency_ms

    def test_implementation(
        self,
        impl_func,
        impl_name: str,
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        h_in_ref: torch.Tensor,
        h_post_ref: torch.Tensor,
        h_res_ref: torch.Tensor,
    ) -> TestResult:
        """Test a single implementation."""
        B, S, n, D = x.shape

        # Move to device
        x_dev = x.to(self.device)
        phi_dev = phi.to(self.device)
        alpha_dev = alpha.to(self.device)
        bias_dev = bias.to(self.device)

        # Run implementation
        try:
            with torch.no_grad():
                h_in, h_post, h_res = impl_func(
                    x_dev, phi_dev, alpha_dev, bias_dev
                )

            # Compute accuracy
            h_in_max, h_in_mean, h_in_pass = self.compute_accuracy(h_in, h_in_ref, "h_in")
            h_post_max, h_post_mean, h_post_pass = self.compute_accuracy(h_post, h_post_ref, "h_post")
            h_res_max, h_res_mean, h_res_pass = self.compute_accuracy(h_res, h_res_ref, "h_res")

            passed = h_in_pass and h_post_pass and h_res_pass

            # Benchmark
            latency_ms = self.benchmark(
                impl_func, x_dev, phi_dev, alpha_dev, bias_dev
            )

            # Compute throughput (sequences per second)
            throughput = (B * S) / (latency_ms / 1000)

            result = TestResult(
                name=impl_name,
                h_in_max_error=h_in_max,
                h_post_max_error=h_post_max,
                h_res_max_error=h_res_max,
                h_in_mean_error=h_in_mean,
                h_post_mean_error=h_post_mean,
                h_res_mean_error=h_res_mean,
                passed=passed,
                latency_ms=latency_ms,
                throughput=throughput,
            )

        except Exception as e:
            print(f"ERROR running {impl_name}: {e}")
            import traceback
            traceback.print_exc()

            result = TestResult(
                name=impl_name,
                h_in_max_error=float('inf'),
                h_post_max_error=float('inf'),
                h_res_max_error=float('inf'),
                h_in_mean_error=float('inf'),
                h_post_mean_error=float('inf'),
                h_res_mean_error=float('inf'),
                passed=False,
                latency_ms=float('inf'),
                throughput=0.0,
            )

        return result

    def run_test_suite(
        self,
        test_configs: List[Dict[str, int]],
    ) -> List[TestResult]:
        """
        Run complete test suite.

        Args:
            test_configs: List of dicts with keys 'B', 'S', 'n', 'D'

        Returns:
            List of TestResult objects
        """
        all_results = []

        print("=" * 80)
        print("MHC Forward Pre Operator - Implementation Comparison")
        print("=" * 80)
        print(f"\nDevice: {self.device}")
        print(f"Warmup iterations: {self.warmup_iters}")
        print(f"Benchmark iterations: {self.benchmark_iters}")
        print(f"Tolerance: rtol={self.rtol}, atol={self.atol}")
        print()

        for config in test_configs:
            B, S, n, D = config['B'], config['S'], config['n'], config['D']

            print("-" * 80)
            print(f"Test Configuration: B={B}, S={S}, n={n}, D={D}")
            print(f"Input shapes: x=[{B}, {S}, {n}, {D}], phi=[{n*n + 2*n}, {n*D}]")
            print("-" * 80)

            # Generate test data
            x, phi, alpha, bias = self.generate_test_data(B, S, n, D)

            # Compute reference
            print("\n[1/4] Computing reference (PyTorch)...")
            ref_latency = self.benchmark(
                mhc_forward_pre, x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu()
            )
            h_in_ref, h_post_ref, h_res_ref = self.compute_reference(x, phi, alpha, bias)
            ref_throughput = (B * S) / (ref_latency / 1000)

            print(f"       Reference latency: {ref_latency:.4f} ms")
            print(f"       Reference throughput: {ref_throughput:.2f} seq/s")

            # Test implementations
            results = []

            # PyTorch reference
            ref_result = TestResult(
                name="PyTorch Reference",
                h_in_max_error=0.0,
                h_post_max_error=0.0,
                h_res_max_error=0.0,
                h_in_mean_error=0.0,
                h_post_mean_error=0.0,
                h_res_mean_error=0.0,
                passed=True,
                latency_ms=ref_latency,
                throughput=ref_throughput,
                speedup_vs_ref=1.0,
            )
            results.append(ref_result)

            # Triton
            if self.triton_impl is not None:
                print("\n[2/4] Testing Triton implementation...")
                triton_result = self.test_implementation(
                    self.triton_impl,
                    "Triton",
                    x, phi, alpha, bias,
                    h_in_ref, h_post_ref, h_res_ref,
                )
                triton_result.speedup_vs_ref = ref_latency / triton_result.latency_ms
                results.append(triton_result)

            # TileLang
            if self.tilelang_impl is not None:
                print("\n[3/4] Testing TileLang implementation...")
                tilelang_result = self.test_implementation(
                    self.tilelang_impl,
                    "TileLang",
                    x, phi, alpha, bias,
                    h_in_ref, h_post_ref, h_res_ref,
                )
                tilelang_result.speedup_vs_ref = ref_latency / tilelang_result.latency_ms
                results.append(tilelang_result)

            # Print results for this configuration
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80)

            self.print_results_table(results)

            all_results.extend(results)

        return all_results

    def print_results_table(self, results: List[TestResult]):
        """Print results in a formatted table."""
        # Header
        print(f"\n{'Implementation':<20} {'Status':<10} {'Max Error':<20} {'Mean Error':<20} {'Latency (ms)':<15} {'Speedup':<10}")
        print("-" * 115)

        for r in results:
            status = "PASS" if r.passed else "FAIL"

            # Max error across all outputs
            max_error = max(r.h_in_max_error, r.h_post_max_error, r.h_res_max_error)
            mean_error = max(r.h_in_mean_error, r_post_mean_error, r.h_res_mean_error)

            print(f"{r.name:<20} {status:<10} {max_error:<20.6f} {mean_error:<20.6f} {r.latency_ms:<15.4f} {r.speedup_vs_ref:<10.2f}x")

        print()

        # Detailed error breakdown
        print("Detailed Error Breakdown:")
        print("-" * 80)
        for r in results:
            if r.name == "PyTorch Reference":
                continue
            print(f"\n{r.name}:")
            print(f"  h_in   : max={r.h_in_max_error:.6f}, mean={r.h_in_mean_error:.6f}")
            print(f"  h_post : max={r.h_post_max_error:.6f}, mean={r.h_post_mean_error:.6f}")
            print(f"  h_res  : max={r.h_res_max_error:.6f}, mean={r.h_res_mean_error:.6f}")

    def save_results_to_csv(self, results: List[TestResult], filename: str = "benchmark_results.csv"):
        """Save results to CSV file."""
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Implementation', 'Status', 'h_in_max_error', 'h_post_max_error', 'h_res_max_error',
                'h_in_mean_error', 'h_post_mean_error', 'h_res_mean_error',
                'Latency_ms', 'Throughput_seq_per_s', 'Speedup_vs_ref'
            ])

            for r in results:
                writer.writerow([
                    r.name, 'PASS' if r.passed else 'FAIL',
                    r.h_in_max_error, r.h_post_max_error, r.h_res_max_error,
                    r.h_in_mean_error, r.h_post_mean_error, r.h_res_mean_error,
                    f"{r.latency_ms:.4f}", f"{r.throughput:.2f}", f"{r.speedup_vs_ref:.2f}"
                ])

        print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Test MHC Forward Pre implementations")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run tests on")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--benchmark", type=int, default=100,
                        help="Number of benchmark iterations")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="Relative tolerance for accuracy check")
    parser.add_argument("--atol", type=float, default=1e-3,
                        help="Absolute tolerance for accuracy check")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test with smaller configurations")
    parser.add_argument("--output", type=str, default="benchmark_results.csv",
                        help="Output CSV filename")

    args = parser.parse_args()

    # Define test configurations
    if args.quick:
        test_configs = [
            {'B': 1, 'S': 128, 'n': 4, 'D': 128},
            {'B': 2, 'S': 256, 'n': 4, 'D': 256},
        ]
        args.benchmark = 20
        args.warmup = 5
    else:
        test_configs = [
            # Small configurations for quick testing
            {'B': 1, 'S': 128, 'n': 4, 'D': 128},
            {'B': 1, 'S': 256, 'n': 4, 'D': 256},
            {'B': 2, 'S': 512, 'n': 4, 'D': 256},

            # Medium configurations
            {'B': 1, 'S': 1024, 'n': 4, 'D': 512},
            {'B': 2, 'S': 2048, 'n': 4, 'D': 512},

            # Large configurations (realistic workloads)
            {'B': 1, 'S': 4096, 'n': 4, 'D': 2560},
            {'B': 2, 'S': 2048, 'n': 8, 'D': 1280},
            {'B': 4, 'S': 1024, 'n': 4, 'D': 2560},
        ]

    # Create tester
    tester = MHCOperatorTester(
        device=args.device,
        warmup_iters=args.warmup,
        benchmark_iters=args.benchmark,
        rtol=args.rtol,
        atol=args.atol,
    )

    # Run tests
    results = tester.run_test_suite(test_configs)

    # Save results
    tester.save_results_to_csv(results, args.output)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Group by implementation
    impl_results = {}
    for r in results:
        if r.name not in impl_results:
            impl_results[r.name] = []
        impl_results[r.name].append(r)

    for impl_name, impl_res_list in impl_results.items():
        if impl_name == "PyTorch Reference":
            continue

        # Average speedup
        avg_speedup = sum(r.speedup_vs_ref for r in impl_res_list) / len(impl_res_list)
        all_passed = all(r.passed for r in impl_res_list)

        print(f"\n{impl_name}:")
        print(f"  All tests passed: {all_passed}")
        print(f"  Average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
