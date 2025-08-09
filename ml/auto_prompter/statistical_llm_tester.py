#!/usr/bin/env python3
"""
Statistical LLM Performance Tester (refactored for new config schema v3)

This script runs multiple iterations of each prompt to ensure statistical reliability.
It calculates confidence intervals, standard deviations, and identifies outliers.
"""

import json
import time
import csv
import os
import glob
import argparse
import subprocess
import platform
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tiktoken
import statistics
import math
from dataclasses import dataclass
from config_loader import load_benchmark_config


@dataclass
class TestResult:
    """Single test result"""

    timestamp: str
    filename: str
    iteration: int
    prompt_token_count: int
    response_time: float
    success: bool
    response_token_count: int
    tokens_per_second: float
    model: str
    llm_url: str


@dataclass
class StatisticalSummary:
    """Statistical summary for a prompt across multiple runs"""

    filename: str
    prompt_token_count: int
    iterations: int
    successful_runs: int

    # Response time statistics
    response_time_mean: float
    response_time_std: float
    response_time_min: float
    response_time_max: float
    response_time_median: float
    response_time_95_ci_lower: float
    response_time_95_ci_upper: float

    # Tokens per second statistics
    tokens_per_second_mean: float
    tokens_per_second_std: float
    tokens_per_second_min: float
    tokens_per_second_max: float
    tokens_per_second_median: float
    tokens_per_second_95_ci_lower: float
    tokens_per_second_95_ci_upper: float

    # Variability metrics
    response_time_cv: float  # Coefficient of variation
    tokens_per_second_cv: float

    # Outlier detection
    outliers_detected: int
    outlier_threshold: float

    model: str
    llm_url: str


class StatisticalLLMTester:
    def __init__(
        self,
        config_path: str = "configs/config_lm_studio.json",
        hardware_override: Optional[str] = None,
    ):
        self.cfg = load_benchmark_config(config_path)
        if hardware_override:
            self.cfg.hardware.code = hardware_override
        self.results: List[TestResult] = []
        self.statistical_summaries: List[StatisticalSummary] = []
        self.tokenizer = self._initialize_tokenizer()
        self.hardware_mapping = self._load_hardware_mapping()

    def _load_hardware_mapping(self) -> Dict:
        """Load hardware mapping configuration."""
        try:
            with open("hardware_mapping.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load hardware mapping: {e}")
            return {
                "hardware_mappings": {},
                "quantization_mappings": {},
                "model_mappings": {},
            }

    def _detect_hardware(self) -> str:
        """Detect hardware configuration for organizing results."""
        try:
            # Try to detect Apple Silicon first
            if platform.system() == "Darwin":  # macOS
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    cpu_info = result.stdout.strip().lower()

                    if "apple m1" in cpu_info:
                        return "m1"
                    elif "apple m2" in cpu_info:
                        return "m2"
                    elif "apple m3" in cpu_info:
                        return "m3"
                    elif "apple m4" in cpu_info:
                        return "m4"
                except subprocess.TimeoutExpired:
                    pass

            # For other systems, return a generic identifier
            system = platform.system().lower()
            machine = platform.machine().lower()
            return f"{system}_{machine}"

        except Exception:
            return "unknown_hardware"

    def _parse_model_info(self, model_name: str) -> Tuple[str, str]:
        """
        Parse model name to extract base model and quantization.
        Enhanced to handle various model name formats and use hardware mapping.
        """
        model_name = model_name.lower()
        original_model = model_name

        # Extract quantization pattern - enhanced patterns
        quant_patterns = [
            r"q\d+(?:_[a-z0-9]+)*",  # q4_k_m, q8_0, q4km, q3, etc.
            r"fp\d+",  # fp16, fp32
            r"int\d+",  # int4, int8
            r"bf\d+",  # bf16
            r"mlx@?\d*bit",  # mlx4bit, mlx8bit, mlx@4bit
            r"mlx\d+bit",  # mlx4bit, mlx8bit
        ]

        quantization = "unknown"
        base_model = model_name

        # First try to find quantization in the model name
        for pattern in quant_patterns:
            match = re.search(pattern, model_name)
            if match:
                quantization = match.group()
                # Normalize common patterns
                if quantization == "q4km":
                    quantization = "q4_k_m"
                elif "mlx" in quantization:
                    quantization = re.sub(r"mlx@?(\d+)bit", r"mlx\1bit", quantization)

                # Remove quantization from model name to get base model
                base_model = re.sub(f"[-_:@]{re.escape(match.group())}", "", model_name)
                break

        # Clean up base model name
        base_model = re.sub(r"[-_:](instruct|chat|it)$", "", base_model)
        base_model = re.sub(r"[^\w\.]", "_", base_model)

        # Apply model mapping if available
        model_mappings = self.hardware_mapping.get("model_mappings", {})
        for key, mapped_name in model_mappings.items():
            if key in base_model:
                base_model = key
                break

        # Apply quantization mapping if available
        quant_mappings = self.hardware_mapping.get("quantization_mappings", {})
        for key, mapped_quant in quant_mappings.items():
            if key in quantization:
                quantization = key
                break

        return base_model, quantization

    def _normalize_hardware_name(self, hardware: str) -> str:
        """Normalize hardware name using hardware mapping."""
        hardware_lower = hardware.lower()

        # Check direct mappings first
        hardware_mappings = self.hardware_mapping.get("hardware_mappings", {})
        if hardware_lower in hardware_mappings:
            return hardware_lower

        # Check fallback patterns
        fallback_patterns = self.hardware_mapping.get("fallback_patterns", {})
        for pattern, normalized in fallback_patterns.items():
            if pattern in hardware_lower:
                return pattern

        return hardware

    def _initialize_tokenizer(self):
        """Initialize the tokenizer based on the model."""
        try:
            model_name = self.cfg.target.model.lower()
            if any(name in model_name for name in ["gpt-4", "gpt-3.5"]):
                return tiktoken.encoding_for_model("gpt-4")
            elif "gpt" in model_name:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return len(text) // 4

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                # Set default values for statistical testing
                config.setdefault("iterations_per_prompt", 5)
                config.setdefault("outlier_threshold", 2.0)  # Z-score threshold
                config.setdefault("warmup_iterations", 2)
                config.setdefault("cooldown_between_iterations", 2)
                config.setdefault("cooldown_between_prompts", 5)
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {config_path}")

    def load_prompts(self, prompts_dir: str = "prompts") -> List[Tuple[str, str]]:
        """Load all prompts from text files in the specified directory."""
        prompts = []
        prompt_files = glob.glob(os.path.join(prompts_dir, "*.txt"))

        if not prompt_files:
            print(f"No .txt files found in {prompts_dir} directory")
            return prompts

        for file_path in sorted(prompt_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        filename = os.path.basename(file_path)
                        prompts.append((filename, content))
                        print(f"Loaded prompt from {filename}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return prompts

    def send_curl_request(self, prompt: str):
        """Send a curl request to the LLM and measure response time."""
        payload = {
            "model": self.cfg.target.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        # generation params
        for p in [
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
        ]:
            if p in self.cfg.raw:
                payload[p] = self.cfg.raw[p]
        curl_cmd = [
            "curl",
            "-s",
            "-X",
            "POST",
            self.cfg.target.llm_url,
            "-H",
            f"Content-Type: {self.cfg.headers.get('Content-Type','application/json')}",
            "-d",
            json.dumps(payload),
            "--max-time",
            str(self.cfg.request_timeout),
        ]
        start = time.time()
        try:
            r = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.cfg.request_timeout,
            )
            dt = time.time() - start
            if r.returncode == 0:
                txt = r.stdout.strip()
                try:
                    data = json.loads(txt)
                    if isinstance(data, dict) and data.get("object") == "error":
                        return data, dt, False
                    return data, dt, True
                except json.JSONDecodeError:
                    return {"raw_response": txt}, dt, False
            return None, dt, False
        except subprocess.TimeoutExpired:
            return None, self.cfg.request_timeout, False
        except Exception:
            return None, 0, False

    def calculate_tokens_per_second(
        self, response: Dict, response_time: float
    ) -> Tuple[int, float]:
        """Calculate tokens per second from the response."""
        if (
            not isinstance(response, dict)
            or "object" in response
            and response["object"] == "error"
        ):
            return 0, 0.0

        token_count = 0
        token_fields = [
            "eval_count",
            "tokens_evaluated",
            "completion_tokens",
            "output_tokens",
            "tokens",
        ]

        # Check usage information
        if "usage" in response:
            usage = response["usage"]
            if "completion_tokens" in usage:
                token_count = usage["completion_tokens"]
            elif "total_tokens" in usage:
                token_count = response["usage"]["total_tokens"]

        # Try other fields
        if token_count == 0:
            for field in token_fields:
                if field in response:
                    token_count = response[field]
                    break

        # Estimate based on response content
        if token_count == 0:
            response_text = ""
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    response_text = choice["message"]["content"]
                elif "text" in choice:
                    response_text = choice["text"]
            elif "response" in response:
                response_text = response["response"]

            if response_text:
                token_count = self.count_tokens(response_text)

        tokens_per_second = token_count / response_time if response_time > 0 else 0
        return token_count, tokens_per_second

    def warm_up_model(self) -> bool:
        """Send warm-up requests to ensure the model is loaded."""
        print("🔥 Warming up the model...")
        warm_up_prompt = "Hello, please respond with just 'Ready'."

        warmup_iterations = self.cfg.raw.get("warmup_iterations", 2)
        successful = 0
        for i in range(warmup_iterations):
            print(f"  Warm-up {i+1}/{warmup_iterations}")
            _, _, ok = self.send_curl_request(warm_up_prompt)
            if ok:
                successful += 1
            time.sleep(1)

        print(f"✓ Completed {successful}/{warmup_iterations} successful warm-ups")
        time.sleep(2)
        return successful > 0

    def test_prompt_multiple_times(
        self, filename: str, prompt: str, iterations: int
    ) -> List[TestResult]:
        """Test a single prompt multiple times for statistical reliability."""
        print(f"\n--- Testing '{filename}' with {iterations} iterations ---")
        prompt_token_count = self.count_tokens(prompt)
        print(f"Prompt length: {prompt_token_count} tokens")
        results = []
        successful_runs = 0
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}", end=" ")
            response, response_time, success = self.send_curl_request(prompt)
            result = TestResult(
                timestamp=datetime.now().isoformat(),
                filename=filename,
                iteration=i + 1,
                prompt_token_count=prompt_token_count,
                response_time=response_time,
                success=success,
                response_token_count=0,
                tokens_per_second=0,
                model=self.cfg.target.model,
                llm_url=self.cfg.target.llm_url,
            )
            if success and response:
                rtoks, tps = self.calculate_tokens_per_second(response, response_time)
                result.response_token_count = rtoks
                result.tokens_per_second = tps
                successful_runs += 1
                print(f"✓ {tps:.1f} tokens/sec ({response_time:.2f}s)")
            else:
                print(f"✗ Failed ({response_time:.2f}s)")
            results.append(result)
            if i < iterations - 1:
                time.sleep(self.cfg.raw.get("cooldown_between_iterations", 2))
        print(f"  Summary: {successful_runs}/{iterations} successful runs")
        return results

    def calculate_statistics(self, results: List[TestResult]) -> StatisticalSummary:
        """Calculate statistical summary for a set of test results."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            # Return empty stats if no successful runs
            return StatisticalSummary(
                filename=results[0].filename,
                prompt_token_count=results[0].prompt_token_count,
                iterations=len(results),
                successful_runs=0,
                response_time_mean=0,
                response_time_std=0,
                response_time_min=0,
                response_time_max=0,
                response_time_median=0,
                response_time_95_ci_lower=0,
                response_time_95_ci_upper=0,
                tokens_per_second_mean=0,
                tokens_per_second_std=0,
                tokens_per_second_min=0,
                tokens_per_second_max=0,
                tokens_per_second_median=0,
                tokens_per_second_95_ci_lower=0,
                tokens_per_second_95_ci_upper=0,
                response_time_cv=0,
                tokens_per_second_cv=0,
                outliers_detected=0,
                outlier_threshold=self.cfg.raw.get("outlier_threshold", 2.0),
                model=results[0].model,
                llm_url=results[0].llm_url,
            )

        # Extract successful measurements
        response_times = [r.response_time for r in successful_results]
        tokens_per_second = [r.tokens_per_second for r in successful_results]

        # Calculate basic statistics
        rt_mean = statistics.mean(response_times)
        rt_std = statistics.stdev(response_times) if len(response_times) > 1 else 0
        rt_median = statistics.median(response_times)

        tps_mean = statistics.mean(tokens_per_second)
        tps_std = (
            statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0
        )
        tps_median = statistics.median(tokens_per_second)

        # Calculate 95% confidence intervals
        n = len(successful_results)
        if n > 1:
            t_score = 1.96 if n >= 30 else 2.78  # Approximate t-scores
            rt_margin = t_score * (rt_std / math.sqrt(n))
            tps_margin = t_score * (tps_std / math.sqrt(n))
        else:
            rt_margin = 0
            tps_margin = 0

        # Detect outliers using Z-score
        outlier_threshold = self.cfg.raw.get("outlier_threshold", 2.0)
        outliers = 0
        if rt_std > 0:
            for rt in response_times:
                z_score = abs(rt - rt_mean) / rt_std
                if z_score > outlier_threshold:
                    outliers += 1

        # Coefficient of variation (CV) - measure of relative variability
        rt_cv = (rt_std / rt_mean * 100) if rt_mean > 0 else 0
        tps_cv = (tps_std / tps_mean * 100) if tps_mean > 0 else 0

        return StatisticalSummary(
            filename=results[0].filename,
            prompt_token_count=results[0].prompt_token_count,
            iterations=len(results),
            successful_runs=len(successful_results),
            response_time_mean=rt_mean,
            response_time_std=rt_std,
            response_time_min=min(response_times),
            response_time_max=max(response_times),
            response_time_median=rt_median,
            response_time_95_ci_lower=rt_mean - rt_margin,
            response_time_95_ci_upper=rt_mean + rt_margin,
            tokens_per_second_mean=tps_mean,
            tokens_per_second_std=tps_std,
            tokens_per_second_min=min(tokens_per_second),
            tokens_per_second_max=max(tokens_per_second),
            tokens_per_second_median=tps_median,
            tokens_per_second_95_ci_lower=tps_mean - tps_margin,
            tokens_per_second_95_ci_upper=tps_mean + tps_margin,
            response_time_cv=rt_cv,
            tokens_per_second_cv=tps_cv,
            outliers_detected=outliers,
            outlier_threshold=outlier_threshold,
            model=results[0].model,
            llm_url=results[0].llm_url,
        )

    def run_statistical_tests(
        self, prompts_dir: str = "prompts", iterations: int = None
    ) -> None:
        """Run statistical tests on prompts."""
        print("=== Statistical LLM Performance Tester ===")
        print(f"LLM URL: {self.cfg.target.llm_url}")
        print(f"Model: {self.cfg.target.model}")
        if iterations is None:
            iterations = self.cfg.raw.get("iterations_per_prompt", 5)
        print(f"Iterations per prompt: {iterations}")
        print(
            f"Outlier detection threshold: {self.cfg.raw.get('outlier_threshold', 2.0)} standard deviations"
        )
        self.warm_up_model()
        prompts = self.load_prompts(prompts_dir)
        if not prompts:
            print("No prompts found. Please add .txt files to the prompts directory.")
            return
        print(f"\nFound {len(prompts)} prompts to test")
        print(f"Total iterations planned: {len(prompts) * iterations}")
        for i, (filename, prompt) in enumerate(prompts, 1):
            print(f"\n{'='*60}")
            print(f"Prompt {i}/{len(prompts)}: {filename}")
            prompt_results = self.test_prompt_multiple_times(
                filename, prompt, iterations
            )
            self.results.extend(prompt_results)
            stats = self.calculate_statistics(prompt_results)
            self.statistical_summaries.append(stats)
            if stats.successful_runs > 0:
                print("  📊 Statistical Summary:")
                print(
                    f"     Tokens/sec: {stats.tokens_per_second_mean:.1f} ± {stats.tokens_per_second_std:.1f} (CV: {stats.tokens_per_second_cv:.1f}%)"
                )
                print(
                    f"     Response time: {stats.response_time_mean:.2f} ± {stats.response_time_std:.2f}s (CV: {stats.response_time_cv:.1f}%)"
                )
                if stats.outliers_detected > 0:
                    print(f"     ⚠️  {stats.outliers_detected} outlier(s) detected")
            if i < len(prompts):
                cooldown = self.cfg.raw.get("cooldown_between_prompts", 5)
                print(f"  💤 Cooling down for {cooldown} seconds...")
                time.sleep(cooldown)
        self.save_results()
        self.print_final_summary()

    def _canonical_quant(self) -> str:
        q = (self.cfg.target.quant or "").lower()
        mapping = self.hardware_mapping.get("quantization_mappings", {})
        return mapping.get(q, q)

    def _build_result_filename(self, kind: str, timestamp: str) -> str:
        alloc = (
            self.cfg.hardware.vram_allocation.lower()
            if self.cfg.hardware.vram_allocation
            else "dyn"
        )
        alloc_short = {"dynamic": "dyn", "static": "sta"}.get(alloc, alloc[:3])
        vram_part = ""
        if self.cfg.hardware.vram_gb and self.cfg.hardware.vram_gb > 0:
            v = self.cfg.hardware.vram_gb
            if float(v).is_integer():
                vram_part = f"{int(v)}gb"
            else:
                vram_part = f"{str(v).replace('.', 'p')}gb"
        raw_quant = self.cfg.target.quant  # keep original quant in filename
        parts = [
            timestamp,
            self.cfg.target.model.split("/")[-1].replace("-", "_"),
            raw_quant,
            self.cfg.target.backend,
            self.cfg.target.runtime,
            alloc_short + (f"_{vram_part}" if vram_part else ""),
            self.cfg.hardware.identifier(),
            kind,
        ]
        safe = []
        for p in parts:
            p = p.lower()
            p = re.sub(r"[^a-z0-9]+", "_", p)
            p = re.sub(r"_+", "_", p).strip("_")
            safe.append(p)
        return "__".join([p for p in safe if p]) + ".csv"

    def save_results(self) -> None:
        """Save detailed results and statistical summaries to CSV files with nested folder structure."""
        if not self.results:
            print("No results to save")
            return
        base_model = self.cfg.target.model.split("/")[-1]
        quantization = self._canonical_quant()
        hardware_slug = self.cfg.hardware.identifier()
        results_dir = os.path.join("results", base_model, quantization, hardware_slug)
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_file = os.path.join(
            results_dir, self._build_result_filename("detailed", timestamp)
        )
        stats_file = os.path.join(
            results_dir, self._build_result_filename("stats", timestamp)
        )
        detailed_fieldnames = [
            "timestamp",
            "filename",
            "iteration",
            "prompt_token_count",
            "response_time",
            "success",
            "response_token_count",
            "tokens_per_second",
            "model",
            "llm_url",
            "base_model",
            "quantization",
            "backend",
            "runtime",
            "hardware_slug",
            "hardware_make",
            "hardware_model",
            "hardware_cpu",
            "hardware_mem_gb",
            "hardware_gpu",
            "environment_os",
            "schema_version",
        ]
        with open(detailed_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=detailed_fieldnames)
            w.writeheader()
            for r in self.results:
                row = r.__dict__.copy()
                row.update(
                    {
                        "base_model": base_model,
                        "quantization": quantization,
                        "backend": self.cfg.target.backend,
                        "runtime": self.cfg.target.runtime,
                        "hardware_slug": hardware_slug,
                        "hardware_make": self.cfg.hardware.make,
                        "hardware_model": self.cfg.hardware.device_model,
                        "hardware_cpu": self.cfg.hardware.cpu,
                        "hardware_mem_gb": self.cfg.hardware.memory_gb,
                        "hardware_gpu": self.cfg.hardware.gpu,
                        "environment_os": self.cfg.environment.os,
                        "schema_version": self.cfg.schema_version,
                    }
                )
                row = {k: row.get(k, "") for k in detailed_fieldnames}
                w.writerow(row)
        stats_fieldnames = [
            "filename",
            "prompt_token_count",
            "iterations",
            "successful_runs",
            "tokens_per_second_mean",
            "tokens_per_second_std",
            "tokens_per_second_min",
            "tokens_per_second_max",
            "tokens_per_second_median",
            "tokens_per_second_95_ci_lower",
            "tokens_per_second_95_ci_upper",
            "tokens_per_second_cv",
            "response_time_mean",
            "response_time_std",
            "response_time_min",
            "response_time_max",
            "response_time_median",
            "response_time_95_ci_lower",
            "response_time_95_ci_upper",
            "response_time_cv",
            "outliers_detected",
            "outlier_threshold",
            "model",
            "llm_url",
            "base_model",
            "quantization",
            "backend",
            "runtime",
            "hardware_slug",
            "hardware_make",
            "hardware_model",
            "hardware_cpu",
            "hardware_mem_gb",
            "hardware_gpu",
            "environment_os",
            "schema_version",
        ]
        with open(stats_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=stats_fieldnames)
            w.writeheader()
            for s in self.statistical_summaries:
                row = s.__dict__.copy()
                row.update(
                    {
                        "base_model": base_model,
                        "quantization": quantization,
                        "backend": self.cfg.target.backend,
                        "runtime": self.cfg.target.runtime,
                        "hardware_slug": hardware_slug,
                        "hardware_make": self.cfg.hardware.make,
                        "hardware_model": self.cfg.hardware.device_model,
                        "hardware_cpu": self.cfg.hardware.cpu,
                        "hardware_mem_gb": self.cfg.hardware.memory_gb,
                        "hardware_gpu": self.cfg.hardware.gpu,
                        "environment_os": self.cfg.environment.os,
                        "schema_version": self.cfg.schema_version,
                    }
                )
                row = {k: row.get(k, "") for k in stats_fieldnames}
                w.writerow(row)
        print(f"\n=== Results Saved ===")
        print(f"Organization: {base_model}/{quantization}/{hardware_slug}")
        print(f"Detailed results: {detailed_file}")
        print(f"Statistical summaries: {stats_file}")
        print(f"Total prompts tested: {len(self.statistical_summaries)}")
        print(f"Total iterations: {len(self.results)}")

    def print_final_summary(self) -> None:
        """Print a comprehensive final summary."""
        if not self.statistical_summaries:
            return

        print(f"\n{'='*80}")
        print("📈 FINAL STATISTICAL SUMMARY")
        print(f"{'='*80}")

        total_iterations = sum(s.iterations for s in self.statistical_summaries)
        total_successful = sum(s.successful_runs for s in self.statistical_summaries)

        print(f"Total prompts tested: {len(self.statistical_summaries)}")
        print(f"Total iterations: {total_iterations}")
        print(
            f"Total successful runs: {total_successful} ({total_successful/total_iterations*100:.1f}%)"
        )

        # Overall reliability analysis
        reliable_prompts = 0
        for stats in self.statistical_summaries:
            if stats.successful_runs > 0:
                # Consider "reliable" if CV < 20% and no outliers
                if stats.tokens_per_second_cv < 20 and stats.outliers_detected == 0:
                    reliable_prompts += 1

        print(
            f"Reliable prompts (CV<20%, no outliers): {reliable_prompts}/{len(self.statistical_summaries)} "
            f"({reliable_prompts/len(self.statistical_summaries)*100:.1f}%)"
        )

        print(f"\n{'='*80}")
        print("📋 PER-PROMPT RELIABILITY ANALYSIS")
        print(f"{'='*80}")

        for stats in self.statistical_summaries:
            if stats.successful_runs == 0:
                print(f"❌ {stats.filename}: ALL TESTS FAILED")
                continue

            success_rate = stats.successful_runs / stats.iterations * 100
            reliability_indicators = []

            if success_rate < 100:
                reliability_indicators.append(f"❌ {success_rate:.0f}% success rate")

            if stats.tokens_per_second_cv > 20:
                reliability_indicators.append(
                    f"⚠️  High variability (CV: {stats.tokens_per_second_cv:.1f}%)"
                )

            if stats.outliers_detected > 0:
                reliability_indicators.append(f"⚠️  {stats.outliers_detected} outliers")

            if not reliability_indicators:
                reliability_status = "✅ RELIABLE"
            else:
                reliability_status = " | ".join(reliability_indicators)

            print(f"{stats.filename}:")
            print(f"  {reliability_status}")
            print(
                f"  Tokens/sec: {stats.tokens_per_second_mean:.1f} ± {stats.tokens_per_second_std:.1f} "
                f"[{stats.tokens_per_second_95_ci_lower:.1f}, {stats.tokens_per_second_95_ci_upper:.1f}]"
            )
            print()


def main():
    parser = argparse.ArgumentParser(description="Statistical LLM performance testing")
    parser.add_argument(
        "--config",
        default="configs/config_lm_studio.json",
        help="Configuration file path",
    )
    parser.add_argument(
        "--prompts-dir", default="prompts", help="Directory containing prompt files"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of iterations per prompt (overrides config)",
    )
    parser.add_argument(
        "--hardware",
        help="Hardware identifier for organizing results (e.g., 'm3max', 'rtx4090'). If not specified, hardware will be auto-detected.",
    )

    args = parser.parse_args()

    try:
        tester = StatisticalLLMTester(args.config, hardware_override=args.hardware)
        tester.run_statistical_tests(args.prompts_dir, args.iterations)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
