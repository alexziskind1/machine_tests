#!/usr/bin/env python3
"""
llama-bench Benchmark Runner

This script automates running llama-bench with various prompt sizes,
parses the output, and saves results to CSV files.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config_loader import load_benchmark_config


@dataclass
class BenchmarkResult:
    """Single benchmark result from llama-bench output"""

    timestamp: str
    model_name: str
    model_size_gb: float
    params_b: float
    backend: str
    threads: int
    test_type: str
    prompt_size: int
    tokens_per_second: float
    std_dev: float

    # Metadata
    quantization: str
    hardware_slug: str
    hardware_make: str
    hardware_model: str
    hardware_cpu: str
    hardware_mem_gb: int
    hardware_gpu: str
    environment_os: str
    model_path: str


class LlamaBenchRunner:
    def __init__(self, config_path: str):
        """Initialize the benchmark runner with configuration."""
        self.cfg = load_benchmark_config(config_path)
        self.results: List[BenchmarkResult] = []

    def _expand_path(self, path: str) -> str:
        """Expand ~ and environment variables in path."""
        return os.path.expanduser(os.path.expandvars(path))

    def _parse_model_size(self, size_str: str) -> float:
        """Parse model size string (e.g., '2.31 GiB') to GB float."""
        match = re.search(r"([\d.]+)\s*GiB", size_str)
        if match:
            return float(match.group(1))
        return 0.0

    def _parse_params(self, params_str: str) -> float:
        """Parse parameters string (e.g., '3.88 B') to billions float."""
        match = re.search(r"([\d.]+)\s*B", params_str)
        if match:
            return float(match.group(1))
        return 0.0

    def _parse_tokens_per_second(self, tps_str: str) -> Tuple[float, float]:
        """Parse tokens/second string (e.g., '1623.32 ± 13.39') to mean and std dev."""
        match = re.search(r"([\d.]+)\s*±\s*([\d.]+)", tps_str)
        if match:
            return float(match.group(1)), float(match.group(2))
        # Try without std dev
        try:
            return float(tps_str.strip()), 0.0
        except ValueError:
            return 0.0, 0.0

    def _parse_test_info(self, test_name: str) -> Optional[Tuple[str, int]]:
        """Parse test name to extract type and size (e.g., 'pp128' -> ('pp', 128), 'tg512' -> ('tg', 512))."""
        # Prompt processing test
        pp_match = re.search(r"pp(\d+)", test_name)
        if pp_match:
            return ("pp", int(pp_match.group(1)))

        # Text generation test
        tg_match = re.search(r"tg(\d+)", test_name)
        if tg_match:
            return ("tg", int(tg_match.group(1)))

        return None

    def _parse_llama_bench_output(self, output: str) -> List[Dict]:
        """Parse llama-bench table output into structured data."""
        lines = output.split("\n")
        results = []

        # Find the table header (look for the separator line with dashes)
        header_found = False
        data_start = 0
        for i, line in enumerate(lines):
            # Look for the separator line with dashes (appears after header)
            if line.strip().startswith("|") and "---" in line:
                header_found = True
                # Data starts on next line
                data_start = i + 1
                break

        if not header_found:
            print("Warning: Could not find llama-bench output table")
            return results

        # Parse data rows
        for line in lines[data_start:]:
            if not line.strip() or not line.startswith("|"):
                continue

            # Stop at build info or empty lines
            if "build:" in line:
                break

            # Split by | and clean up
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 8:
                continue

            # Skip empty rows
            if not parts[1]:
                continue

            try:
                result = {
                    "model": parts[1],
                    "size": parts[2],
                    "params": parts[3],
                    "backend": parts[4],
                    "threads": int(parts[5]) if parts[5].isdigit() else 0,
                    "test": parts[6],
                    "tokens_per_second": parts[7],
                }
                results.append(result)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line}")
                continue

        return results

    def run_benchmark(self) -> bool:
        """Run llama-bench with configured parameters."""
        print("=== llama-bench Benchmark Runner ===")
        print(f"Model: {self.cfg.target.model_name}")
        print(f"Quantization: {self.cfg.target.quant}")
        print(f"Hardware: {self.cfg.hardware.code}")
        print(f"Prompt sizes: {self.cfg.benchmark.prompt_sizes}")
        print(f"Output tokens: {self.cfg.benchmark.output_tokens}")
        print(f"Iterations: {self.cfg.benchmark.iterations}")

        # Expand paths
        llama_bench_path = self._expand_path(self.cfg.target.llama_bench_path)
        model_path = self._expand_path(self.cfg.target.model_path)

        # Validate paths
        if not os.path.exists(llama_bench_path):
            print(f"Error: llama-bench not found at {llama_bench_path}")
            return False

        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return False

        # Build command
        prompt_sizes_str = ",".join(map(str, self.cfg.benchmark.prompt_sizes))

        cmd = [
            llama_bench_path,
            "-m",
            model_path,
            "-p",
            prompt_sizes_str,
            "-n",
            str(self.cfg.benchmark.output_tokens),
        ]

        if self.cfg.target.threads:
            cmd.extend(["-t", str(self.cfg.target.threads)])

        print(f"\nRunning: {' '.join(cmd)}\n")

        # Warmup runs - use small prompt size to just load the model
        if self.cfg.benchmark.warmup_iterations > 0:
            print(
                f"🔥 Running {self.cfg.benchmark.warmup_iterations} warmup iterations (small prompt)..."
            )
            warmup_cmd = [
                llama_bench_path,
                "-m",
                model_path,
                "-p",
                "128",  # Small prompt for quick warmup
                "-n",
                "64",  # Fewer output tokens for warmup
            ]
            if self.cfg.target.threads:
                warmup_cmd.extend(["-t", str(self.cfg.target.threads)])

            for i in range(self.cfg.benchmark.warmup_iterations):
                print(f"  Warmup {i+1}/{self.cfg.benchmark.warmup_iterations}")
                try:
                    subprocess.run(
                        warmup_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=120,  # Shorter timeout for warmup
                        check=True,
                    )
                except subprocess.TimeoutExpired:
                    print(f"  Warning: Warmup {i+1} timed out")
                except subprocess.CalledProcessError as e:
                    print(f"  Warning: Warmup {i+1} failed: {e}")
                time.sleep(1)  # Shorter pause between warmups
            print("✓ Warmup complete\n")

        # Run actual benchmarks
        print(f"📊 Running {self.cfg.benchmark.iterations} benchmark iterations...\n")

        for iteration in range(self.cfg.benchmark.iterations):
            print(f"{'='*80}")
            print(f"Iteration {iteration + 1}/{self.cfg.benchmark.iterations}")
            print(f"{'='*80}")

            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=600,
                    check=True,
                    text=True,
                )

                # Parse output
                parsed_results = self._parse_llama_bench_output(result.stdout)

                if not parsed_results:
                    print("Warning: No results parsed from output")
                    print("\n--- llama-bench output ---")
                    print(result.stdout)
                    print("--- end output ---\n")
                    continue

                # Store results
                timestamp = datetime.now().isoformat()
                for res in parsed_results:
                    test_info = self._parse_test_info(res["test"])
                    if test_info is None:
                        continue  # Skip unparseable tests

                    test_type, test_size = test_info

                    tokens_per_sec, std_dev = self._parse_tokens_per_second(
                        res["tokens_per_second"]
                    )

                    benchmark_result = BenchmarkResult(
                        timestamp=timestamp,
                        model_name=self.cfg.target.model_name,
                        model_size_gb=self._parse_model_size(res["size"]),
                        params_b=self._parse_params(res["params"]),
                        backend=res["backend"],
                        threads=res["threads"],
                        test_type=res["test"],
                        prompt_size=test_size,  # Now stores both pp and tg sizes
                        tokens_per_second=tokens_per_sec,
                        std_dev=std_dev,
                        quantization=self.cfg.target.quant,
                        hardware_slug=self.cfg.hardware.identifier(),
                        hardware_make=self.cfg.hardware.make,
                        hardware_model=self.cfg.hardware.device_model,
                        hardware_cpu=self.cfg.hardware.cpu,
                        hardware_mem_gb=self.cfg.hardware.memory_gb,
                        hardware_gpu=self.cfg.hardware.gpu,
                        environment_os=self.cfg.environment.os,
                        model_path=model_path,
                    )

                    self.results.append(benchmark_result)

                print(
                    f"✓ Iteration {iteration + 1} complete: {len(parsed_results)} results"
                )

            except subprocess.TimeoutExpired:
                print(f"✗ Iteration {iteration + 1} timed out")
                continue
            except subprocess.CalledProcessError as e:
                print(f"✗ Iteration {iteration + 1} failed: {e}")
                print(f"stderr: {e.stderr}")
                continue

            # Brief pause between iterations
            if iteration < self.cfg.benchmark.iterations - 1:
                time.sleep(3)

        print(f"\n{'='*80}")
        print(f"Benchmark complete: {len(self.results)} total results collected")
        print(f"{'='*80}\n")

        return len(self.results) > 0

    def save_results(self) -> list:
        """Save results to CSV files in organized directory structure (separate files for pp and tg)."""
        if not self.results:
            print("No results to save")
            return []

        # Separate results by test type
        pp_results = [r for r in self.results if r.test_type.startswith("pp")]
        tg_results = [r for r in self.results if r.test_type.startswith("tg")]

        # Create nested directory structure: results/<model>/<quant>/<hardware>/
        model_dir = (
            self.cfg.target.model_name.replace("-", "_").replace(" ", "_").lower()
        )
        quant_dir = self.cfg.target.quant.lower()
        hardware_dir = self.cfg.hardware.identifier()

        results_dir = Path("results") / model_dir / quant_dir / hardware_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV fieldnames
        fieldnames = [
            "timestamp",
            "model_name",
            "model_size_gb",
            "params_b",
            "backend",
            "threads",
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

        saved_files = []

        # Save pp results
        if pp_results:
            pp_filename = f"{timestamp}__{self.cfg.target.model_name.replace('-', '_')}__{self.cfg.target.quant}__{self.cfg.target.backend}__{hardware_dir}__llama_bench_pp.csv"
            pp_filepath = results_dir / pp_filename

            with open(pp_filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in pp_results:
                    writer.writerow(
                        {
                            "timestamp": result.timestamp,
                            "model_name": result.model_name,
                            "model_size_gb": result.model_size_gb,
                            "params_b": result.params_b,
                            "backend": result.backend,
                            "threads": result.threads,
                            "test_type": result.test_type,
                            "prompt_size": result.prompt_size,
                            "tokens_per_second": result.tokens_per_second,
                            "std_dev": result.std_dev,
                            "quantization": result.quantization,
                            "hardware_slug": result.hardware_slug,
                            "hardware_make": result.hardware_make,
                            "hardware_model": result.hardware_model,
                            "hardware_cpu": result.hardware_cpu,
                            "hardware_mem_gb": result.hardware_mem_gb,
                            "hardware_gpu": result.hardware_gpu,
                            "environment_os": result.environment_os,
                            "model_path": result.model_path,
                        }
                    )

            saved_files.append(str(pp_filepath))
            print(f"\n=== Prompt Processing (pp) Results Saved ===")
            print(f"File: {pp_filepath}")
            print(f"Results: {len(pp_results)}")

        # Save tg results
        if tg_results:
            tg_filename = f"{timestamp}__{self.cfg.target.model_name.replace('-', '_')}__{self.cfg.target.quant}__{self.cfg.target.backend}__{hardware_dir}__llama_bench_tg.csv"
            tg_filepath = results_dir / tg_filename

            with open(tg_filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in tg_results:
                    writer.writerow(
                        {
                            "timestamp": result.timestamp,
                            "model_name": result.model_name,
                            "model_size_gb": result.model_size_gb,
                            "params_b": result.params_b,
                            "backend": result.backend,
                            "threads": result.threads,
                            "test_type": result.test_type,
                            "prompt_size": result.prompt_size,
                            "tokens_per_second": result.tokens_per_second,
                            "std_dev": result.std_dev,
                            "quantization": result.quantization,
                            "hardware_slug": result.hardware_slug,
                            "hardware_make": result.hardware_make,
                            "hardware_model": result.hardware_model,
                            "hardware_cpu": result.hardware_cpu,
                            "hardware_mem_gb": result.hardware_mem_gb,
                            "hardware_gpu": result.hardware_gpu,
                            "environment_os": result.environment_os,
                            "model_path": result.model_path,
                        }
                    )

            saved_files.append(str(tg_filepath))
            print(f"\n=== Text Generation (tg) Results Saved ===")
            print(f"File: {tg_filepath}")
            print(f"Results: {len(tg_results)}")

        print(f"\nOrganization: {model_dir}/{quant_dir}/{hardware_dir}")
        print(
            f"Total results: {len(self.results)} ({len(pp_results)} pp, {len(tg_results)} tg)"
        )

        return saved_files

    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results:
            print("No results to summarize")
            return

        print(f"\n{'='*80}")
        print("📊 BENCHMARK SUMMARY")
        print(f"{'='*80}\n")

        # Group by prompt size
        from collections import defaultdict

        by_prompt_size = defaultdict(list)

        for result in self.results:
            by_prompt_size[result.prompt_size].append(result.tokens_per_second)

        print(
            f"{'Prompt Size':<15} {'Mean t/s':<15} {'Min t/s':<15} {'Max t/s':<15} {'Samples':<10}"
        )
        print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")

        for prompt_size in sorted(by_prompt_size.keys()):
            values = by_prompt_size[prompt_size]
            mean_tps = sum(values) / len(values)
            min_tps = min(values)
            max_tps = max(values)

            print(
                f"{prompt_size:<15} {mean_tps:<15.2f} {min_tps:<15.2f} {max_tps:<15.2f} {len(values):<10}"
            )

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run llama-bench benchmarks")
    parser.add_argument(
        "--config",
        default="configs/config_m4max_gemma3_4b.json",
        help="Configuration file path",
    )

    args = parser.parse_args()

    try:
        runner = LlamaBenchRunner(args.config)

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
