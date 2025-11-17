#!/usr/bin/env python3
"""
MLX Benchmark Runner

This script automates running MLX model benchmarks with various prompt sizes,
measures performance, and saves results to CSV files.
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load, generate

from config_loader import load_benchmark_config


@dataclass
class BenchmarkResult:
    """Single benchmark result from MLX generation"""

    timestamp: str
    model_name: str
    model_size_gb: float
    params_b: float
    test_type: str
    prompt_size: int
    tokens_per_second: float
    std_dev: float
    quantization: str
    hardware_slug: str
    hardware_make: str
    hardware_model: str
    hardware_cpu: str
    hardware_mem_gb: int
    hardware_gpu: str
    environment_os: str
    model_path: str


class MLXBenchRunner:
    """MLX benchmark runner"""

    def __init__(self, config_path: str):
        """Initialize the benchmark runner with configuration."""
        self.cfg = load_benchmark_config(config_path)
        self.results: List[BenchmarkResult] = []
        self.model = None
        self.tokenizer = None

    def _get_model_size(self) -> tuple[float, float]:
        """Estimate model size and parameters."""
        model_path = Path(self.cfg.target.model_path)

        # Try to get size from disk
        size_gb = 0.0
        if model_path.exists():
            if model_path.is_dir():
                size_gb = sum(
                    f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                ) / (1024**3)
            else:
                size_gb = model_path.stat().st_size / (1024**3)

        # Estimate parameters based on common model sizes
        # This is a rough estimate - actual params depend on architecture
        params_b = size_gb * 1.5  # Rough heuristic

        return size_gb, params_b

    def _create_prompt(self, tokens: int) -> str:
        """Create a prompt with approximately the specified token count."""
        # Average ~1.3 tokens per word in English
        words_needed = int(tokens / 1.3)

        # Create a simple repetitive prompt
        base_text = "The quick brown fox jumps over the lazy dog. "
        words_in_base = len(base_text.split())

        repetitions = max(1, words_needed // words_in_base)
        return base_text * repetitions

    def _benchmark_prompt_processing(
        self, prompt: str, prompt_size: int
    ) -> list[float]:
        """Benchmark prompt processing speed (pp test) - returns list of individual results."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        times = []

        for _ in range(self.cfg.benchmark.iterations):
            mx.eval(self.model.parameters())
            mx.clear_cache()

            start = time.time()

            # Tokenize and process prompt (without generation)
            tokens = self.tokenizer.encode(prompt)
            inputs = mx.array(
                [tokens]
            )  # Run through model to get prompt processing time
            _ = self.model(inputs)
            mx.eval(_)

            elapsed = time.time() - start
            tokens_per_sec = len(tokens) / elapsed if elapsed > 0 else 0
            times.append(tokens_per_sec)

        return times

    def _benchmark_text_generation(self, prompt: str, prompt_size: int) -> list[float]:
        """Benchmark text generation speed (tg test) - returns list of individual results."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        times = []

        for _ in range(self.cfg.benchmark.iterations):
            mx.eval(self.model.parameters())
            mx.clear_cache()

            # Pre-process the prompt (not timed for generation)
            prompt_tokens = self.tokenizer.encode(prompt)

            # Time only the generation phase
            start = time.time()

            # Generate tokens
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.cfg.benchmark.output_tokens,
                verbose=False,
            )

            elapsed = time.time() - start

            # Subtract approximate prompt processing time
            # by doing a quick prompt-only pass to measure it
            mx.clear_cache()
            prompt_start = time.time()
            inputs = mx.array([prompt_tokens])
            _ = self.model(inputs)
            mx.eval(_)
            prompt_time = time.time() - prompt_start

            # Pure generation time
            generation_time = elapsed - prompt_time

            # Calculate tokens/sec for generation only
            generated_tokens = self.cfg.benchmark.output_tokens
            tokens_per_sec = (
                generated_tokens / generation_time if generation_time > 0 else 0
            )
            times.append(tokens_per_sec)

        return times

    def run_benchmark(self) -> bool:
        """Run MLX benchmarks with configured parameters."""
        print(f"\n{'='*80}")
        print(f"MLX Benchmark Runner")
        print(f"{'='*80}\n")
        print(f"Model: {self.cfg.target.model_name}")
        print(f"Quantization: {self.cfg.target.quant}")
        print(f"Hardware: {self.cfg.hardware.device_model}")
        print(f"Prompt sizes: {self.cfg.benchmark.prompt_sizes}")
        print(f"Output tokens: {self.cfg.benchmark.output_tokens}")
        print(f"Iterations per size: {self.cfg.benchmark.iterations}")
        print(f"\n{'='*80}\n")

        # Load model
        print(f"📥 Loading model from {self.cfg.target.model_path}...")
        try:
            self.model, self.tokenizer = load(self.cfg.target.model_path)
            print(f"✓ Model loaded successfully\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        # Get model metadata
        model_size_gb, params_b = self._get_model_size()
        timestamp = datetime.now().isoformat()

        # Warmup
        if self.cfg.benchmark.warmup_iterations > 0:
            print(
                f"🔥 Running {self.cfg.benchmark.warmup_iterations} warmup iterations..."
            )
            warmup_prompt = self._create_prompt(128)
            for i in range(self.cfg.benchmark.warmup_iterations):
                print(f"  Warmup {i+1}/{self.cfg.benchmark.warmup_iterations}")
                try:
                    _ = generate(
                        self.model,
                        self.tokenizer,
                        prompt=warmup_prompt,
                        max_tokens=32,
                        verbose=False,
                    )
                except Exception as e:
                    print(f"  Warning: Warmup {i+1} failed: {e}")
                time.sleep(1)
            print("✓ Warmup complete\n")

        # Run benchmarks for each prompt size
        print(
            f"📊 Running benchmarks for {len(self.cfg.benchmark.prompt_sizes)} prompt sizes...\n"
        )

        for prompt_size in self.cfg.benchmark.prompt_sizes:
            print(f"{'='*80}")
            print(f"Prompt Size: {prompt_size} tokens")
            print(f"{'='*80}")

            prompt = self._create_prompt(prompt_size)

            # Prompt processing (pp) test
            print(f"  Running pp test ({self.cfg.benchmark.iterations} iterations)...")
            try:
                pp_times = self._benchmark_prompt_processing(prompt, prompt_size)

                # Save each iteration as a separate result
                for iteration_tps in pp_times:
                    result = BenchmarkResult(
                        timestamp=timestamp,
                        model_name=self.cfg.target.model_name,
                        model_size_gb=model_size_gb,
                        params_b=params_b,
                        test_type=f"pp{prompt_size}",
                        prompt_size=prompt_size,
                        tokens_per_second=iteration_tps,
                        std_dev=0.0,  # No std_dev for individual iterations
                        quantization=self.cfg.target.quant,
                        hardware_slug=self.cfg.hardware.identifier(),
                        hardware_make=self.cfg.hardware.make,
                        hardware_model=self.cfg.hardware.device_model,
                        hardware_cpu=self.cfg.hardware.cpu,
                        hardware_mem_gb=self.cfg.hardware.memory_gb,
                        hardware_gpu=self.cfg.hardware.gpu,
                        environment_os=self.cfg.environment.os,
                        model_path=self.cfg.target.model_path,
                    )
                    self.results.append(result)

                # Calculate stats for display
                mean_tps = sum(pp_times) / len(pp_times)
                variance = sum((x - mean_tps) ** 2 for x in pp_times) / len(pp_times)
                std_dev = variance**0.5
                print(f"  ✓ pp: {mean_tps:.2f} ± {std_dev:.2f} tokens/sec")
            except Exception as e:
                print(f"  ✗ pp test failed: {e}")

            # Text generation (tg) test
            print(f"  Running tg test ({self.cfg.benchmark.iterations} iterations)...")
            try:
                tg_times = self._benchmark_text_generation(prompt, prompt_size)

                # Save each iteration as a separate result
                # TG test is named after output_tokens (like llama-bench does), not prompt_size
                for iteration_tps in tg_times:
                    result = BenchmarkResult(
                        timestamp=timestamp,
                        model_name=self.cfg.target.model_name,
                        model_size_gb=model_size_gb,
                        params_b=params_b,
                        test_type=f"tg{self.cfg.benchmark.output_tokens}",
                        prompt_size=prompt_size,  # Store actual prompt size used
                        tokens_per_second=iteration_tps,
                        std_dev=0.0,  # No std_dev for individual iterations
                        quantization=self.cfg.target.quant,
                        hardware_slug=self.cfg.hardware.identifier(),
                        hardware_make=self.cfg.hardware.make,
                        hardware_model=self.cfg.hardware.device_model,
                        hardware_cpu=self.cfg.hardware.cpu,
                        hardware_mem_gb=self.cfg.hardware.memory_gb,
                        hardware_gpu=self.cfg.hardware.gpu,
                        environment_os=self.cfg.environment.os,
                        model_path=self.cfg.target.model_path,
                    )
                    self.results.append(result)

                # Calculate stats for display
                mean_tps = sum(tg_times) / len(tg_times)
                variance = sum((x - mean_tps) ** 2 for x in tg_times) / len(tg_times)
                std_dev = variance**0.5
                print(f"  ✓ tg: {mean_tps:.2f} ± {std_dev:.2f} tokens/sec")
            except Exception as e:
                print(f"  ✗ tg test failed: {e}")

            print()

        print(f"\n{'='*80}")
        print(f"Benchmark complete: {len(self.results)} total results collected")
        print(f"{'='*80}\n")

        return len(self.results) > 0

    def save_results(self):
        """Save results to CSV files (separate pp and tg)."""
        if not self.results:
            print("No results to save")
            return

        # Separate pp and tg results
        pp_results = [r for r in self.results if r.test_type.startswith("pp")]
        tg_results = [r for r in self.results if r.test_type.startswith("tg")]

        # Create nested directory structure
        model_slug = (
            self.cfg.target.model_name.lower().replace(" ", "_").replace("-", "_")
        )
        quant_slug = self.cfg.target.quant.lower().replace(" ", "_").replace("-", "_")
        hardware_slug = self.cfg.hardware.identifier()

        results_dir = Path("results") / model_slug / quant_slug / hardware_slug
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{timestamp}__{model_slug}__{quant_slug}__MLX__{hardware_slug}"

        # Define CSV columns
        fieldnames = [
            "timestamp",
            "model_name",
            "model_size_gb",
            "params_b",
            "test_type",
            "prompt_size",
            "tokens_per_second",
            "std_dev",
            "quantization",
            "hardware_slug",
            "hardware_make",
            "hardware_model",
            "hardware_cpu",
            "hardware_mem_gb",
            "hardware_gpu",
            "environment_os",
            "model_path",
        ]

        # Save pp results
        if pp_results:
            pp_filepath = results_dir / f"{base_filename}__mlx_bench_pp.csv"
            with open(pp_filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in pp_results:
                    writer.writerow(result.__dict__)
            print(f"✓ Saved pp results: {pp_filepath}")

        # Save tg results
        if tg_results:
            tg_filepath = results_dir / f"{base_filename}__mlx_bench_tg.csv"
            with open(tg_filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in tg_results:
                    writer.writerow(result.__dict__)
            print(f"✓ Saved tg results: {tg_filepath}")

        print(f"\nResults organization: {model_slug}/{quant_slug}/{hardware_slug}")
        print(f"Total results: {len(self.results)}")

    def print_summary(self):
        """Print summary statistics of benchmark results."""
        if not self.results:
            return

        print(f"\n{'='*80}")
        print("📊 BENCHMARK SUMMARY")
        print(f"{'='*80}\n")

        # Group by test type
        pp_results = [r for r in self.results if r.test_type.startswith("pp")]
        tg_results = [r for r in self.results if r.test_type.startswith("tg")]

        for test_group, test_label in [
            (pp_results, "PP (Prompt Processing)"),
            (tg_results, "TG (Text Generation)"),
        ]:
            if not test_group:
                continue

            print(f"\n{test_label}:")
            print(f"{'-'*80}")
            print(f"{'Prompt Size':<15} {'Mean t/s':<15} {'Std Dev':<15}")
            print(f"{'-'*15} {'-'*15} {'-'*15}")

            for result in test_group:
                print(
                    f"{result.prompt_size:<15} "
                    f"{result.tokens_per_second:<15.2f} "
                    f"{result.std_dev:<15.2f}"
                )

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run MLX benchmarks")
    parser.add_argument(
        "--config",
        default="configs/config_m4max_gemma3_4b.json",
        help="Configuration file path",
    )

    args = parser.parse_args()

    try:
        runner = MLXBenchRunner(args.config)

        if runner.run_benchmark():
            runner.save_results()
            runner.print_summary()
            return 0
        else:
            print("Benchmark failed")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
