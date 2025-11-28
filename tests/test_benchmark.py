"""Tests for benchmark utilities.

These tests ensure the benchmark functions run without error.
Actual performance validation is done manually via the benchmark script.
"""

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestBenchmarkFunctions:
    """Tests for benchmark helper functions."""

    def test_import_benchmark(self):
        """Test benchmark module can be imported."""
        from benchmark.bench_packed_inference import (
            BenchmarkResult,
            get_gpu_memory_mb,
            measure_peak_memory,
            measure_latency,
            get_param_memory,
        )

    def test_benchmark_result_dataclass(self):
        """Test BenchmarkResult dataclass."""
        from benchmark.bench_packed_inference import BenchmarkResult

        result = BenchmarkResult(
            name="test",
            shape="(32, 64) -> (32, 128)",
            dtype="fp32",
            peak_memory_bytes=1024,
            avg_latency_ms=0.5,
            std_latency_ms=0.1,
            param_memory_bytes=512,
        )
        assert result.name == "test"
        assert result.peak_memory_bytes == 1024

    def test_get_param_memory(self):
        """Test get_param_memory function."""
        from benchmark.bench_packed_inference import get_param_memory
        import torch.nn as nn

        linear = nn.Linear(64, 32, bias=False).cuda()
        mem = get_param_memory(linear)
        expected = 64 * 32 * 4  # fp32
        assert mem == expected

    def test_measure_peak_memory(self):
        """Test measure_peak_memory function."""
        from benchmark.bench_packed_inference import measure_peak_memory

        x = torch.randn(32, 64, device='cuda')
        linear = torch.nn.Linear(64, 128).cuda()

        def fn():
            with torch.no_grad():
                return linear(x)

        mem = measure_peak_memory(fn, warmup=2)
        assert mem > 0

    def test_measure_latency(self):
        """Test measure_latency function."""
        from benchmark.bench_packed_inference import measure_latency

        x = torch.randn(32, 64, device='cuda')
        linear = torch.nn.Linear(64, 128).cuda()

        def fn():
            with torch.no_grad():
                return linear(x)

        avg, std = measure_latency(fn, warmup=2, iters=10)
        assert avg > 0
        assert std >= 0


class TestBenchmarkModules:
    """Tests for module benchmark functions."""

    def test_benchmark_fp_linear(self):
        """Test FP32 linear benchmark."""
        from benchmark.bench_packed_inference import benchmark_fp_linear

        result = benchmark_fp_linear(
            B=8, K=64, N=32,
            dtype=torch.float32,
            warmup=2, iters=5
        )
        assert result.name == "nn.Linear (fp32)"
        assert result.peak_memory_bytes > 0
        assert result.avg_latency_ms > 0

    def test_benchmark_ternary_training(self):
        """Test TernaryLinear benchmark."""
        from benchmark.bench_packed_inference import benchmark_ternary_training

        result = benchmark_ternary_training(
            B=8, K=64, N=32,
            dtype=torch.float32,
            warmup=2, iters=5
        )
        assert "TernaryLinear" in result.name
        assert result.peak_memory_bytes > 0

    def test_benchmark_ternary_inference(self):
        """Test TernaryLinearInference benchmark."""
        from benchmark.bench_packed_inference import benchmark_ternary_inference

        result = benchmark_ternary_inference(
            B=8, K=64, N=32,
            dtype=torch.float32,
            warmup=2, iters=5
        )
        assert "TernaryLinearInference" in result.name
        assert result.peak_memory_bytes > 0

    def test_inference_uses_less_param_memory(self):
        """Test that inference module uses less parameter memory."""
        from benchmark.bench_packed_inference import (
            benchmark_fp_linear,
            benchmark_ternary_inference,
        )

        fp_result = benchmark_fp_linear(
            B=8, K=256, N=128,
            dtype=torch.float32,
            warmup=2, iters=5
        )
        infer_result = benchmark_ternary_inference(
            B=8, K=256, N=128,
            dtype=torch.float32,
            warmup=2, iters=5
        )

        # Inference should use significantly less parameter memory
        ratio = fp_result.param_memory_bytes / infer_result.param_memory_bytes
        assert ratio > 5  # At least 5x smaller (accounting for scale overhead)


class TestFormatBytes:
    """Tests for format_bytes helper."""

    def test_format_bytes(self):
        """Test byte formatting."""
        from benchmark.bench_packed_inference import format_bytes

        assert format_bytes(100) == "100 B"
        assert "KB" in format_bytes(2048)
        assert "MB" in format_bytes(2 * 1024 * 1024)
        assert "GB" in format_bytes(2 * 1024 * 1024 * 1024)
