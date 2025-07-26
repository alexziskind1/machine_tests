#!/usr/bin/env python3
"""
Statistical LLM Performance Tester

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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tiktoken
import statistics
import math
from dataclasses import dataclass


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
    def __init__(self, config_path: str = "config.json"):
        """Initialize the statistical performance tester."""
        self.config = self.load_config(config_path)
        self.results: List[TestResult] = []
        self.statistical_summaries: List[StatisticalSummary] = []
        self.tokenizer = self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize the tokenizer based on the model."""
        try:
            model_name = self.config.get("model", "").lower()
            if any(name in model_name for name in ["gpt-4", "gpt-3.5"]):
                return tiktoken.encoding_for_model("gpt-4")
            elif "gpt" in model_name:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"Warning: Could not initialize specific tokenizer: {e}")
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

    def send_curl_request(self, prompt: str) -> Tuple[Optional[Dict], float, bool]:
        """Send a curl request to the LLM and measure response time."""
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        curl_cmd = [
            "curl",
            "-s",
            "-X",
            "POST",
            self.config["llm_url"],
            "-H",
            f"Content-Type: {self.config['headers']['Content-Type']}",
            "-d",
            json.dumps(payload),
            "--max-time",
            str(self.config["request_timeout"]),
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=self.config["request_timeout"],
            )

            end_time = time.time()
            total_time = end_time - start_time

            if result.returncode == 0:
                response_text = result.stdout.strip()
                try:
                    response_data = json.loads(response_text)
                    if (
                        isinstance(response_data, dict)
                        and "object" in response_data
                        and response_data["object"] == "error"
                    ):
                        return response_data, total_time, False
                    return response_data, total_time, True
                except json.JSONDecodeError:
                    return {"raw_response": response_text}, total_time, False
            else:
                return None, total_time, False

        except subprocess.TimeoutExpired:
            return None, self.config["request_timeout"], False
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
                token_count = usage["total_tokens"]

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

        warmup_iterations = self.config.get("warmup_iterations", 2)
        successful_warmups = 0

        for i in range(warmup_iterations):
            print(f"  Warm-up {i+1}/{warmup_iterations}")
            response, response_time, success = self.send_curl_request(warm_up_prompt)
            if success:
                successful_warmups += 1
            time.sleep(1)

        print(
            f"✓ Completed {successful_warmups}/{warmup_iterations} successful warm-ups"
        )
        time.sleep(2)
        return successful_warmups > 0

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
                model=self.config["model"],
                llm_url=self.config["llm_url"],
            )

            if success and response:
                response_token_count, tokens_per_second = (
                    self.calculate_tokens_per_second(response, response_time)
                )
                result.response_token_count = response_token_count
                result.tokens_per_second = tokens_per_second
                successful_runs += 1
                print(f"✓ {tokens_per_second:.1f} tokens/sec ({response_time:.2f}s)")
            else:
                print(f"✗ Failed ({response_time:.2f}s)")

            results.append(result)

            # Cooldown between iterations
            if i < iterations - 1:
                time.sleep(self.config.get("cooldown_between_iterations", 2))

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
                outlier_threshold=self.config.get("outlier_threshold", 2.0),
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
        outlier_threshold = self.config.get("outlier_threshold", 2.0)
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
        print(f"LLM URL: {self.config['llm_url']}")
        print(f"Model: {self.config['model']}")

        if iterations is None:
            iterations = self.config.get("iterations_per_prompt", 5)

        print(f"Iterations per prompt: {iterations}")
        print(
            f"Outlier detection threshold: {self.config.get('outlier_threshold', 2.0)} standard deviations"
        )

        # Warm up the model
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

            # Test the prompt multiple times
            prompt_results = self.test_prompt_multiple_times(
                filename, prompt, iterations
            )
            self.results.extend(prompt_results)

            # Calculate statistics for this prompt
            stats = self.calculate_statistics(prompt_results)
            self.statistical_summaries.append(stats)

            # Print immediate summary
            if stats.successful_runs > 0:
                print(f"  📊 Statistical Summary:")
                print(
                    f"     Tokens/sec: {stats.tokens_per_second_mean:.1f} ± {stats.tokens_per_second_std:.1f} "
                    f"(CV: {stats.tokens_per_second_cv:.1f}%)"
                )
                print(
                    f"     Response time: {stats.response_time_mean:.2f} ± {stats.response_time_std:.2f}s "
                    f"(CV: {stats.response_time_cv:.1f}%)"
                )
                if stats.outliers_detected > 0:
                    print(f"     ⚠️  {stats.outliers_detected} outlier(s) detected")

            # Cooldown between prompts
            if i < len(prompts):
                cooldown = self.config.get("cooldown_between_prompts", 5)
                print(f"  💤 Cooling down for {cooldown} seconds...")
                time.sleep(cooldown)

        self.save_results()
        self.print_final_summary()

    def save_results(self) -> None:
        """Save detailed results and statistical summaries to CSV files."""
        if not self.results:
            print("No results to save")
            return

        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.config.get("output_csv", "performance_results.csv")
        name = os.path.splitext(base_name)[0]

        # Save detailed results
        detailed_file = os.path.join(results_dir, f"{name}_detailed_{timestamp}.csv")
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
        ]

        with open(detailed_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=detailed_fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.__dict__)

        # Save statistical summaries
        stats_file = os.path.join(results_dir, f"{name}_statistics_{timestamp}.csv")
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
        ]

        with open(stats_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=stats_fieldnames)
            writer.writeheader()
            for stats in self.statistical_summaries:
                writer.writerow(stats.__dict__)

        print(f"\n=== Results Saved ===")
        print(f"Detailed results: {detailed_file}")
        print(f"Statistical summaries: {stats_file}")

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
        "--config", default="config.json", help="Configuration file path"
    )
    parser.add_argument(
        "--prompts-dir", default="prompts", help="Directory containing prompt files"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of iterations per prompt (overrides config)",
    )

    args = parser.parse_args()

    try:
        tester = StatisticalLLMTester(args.config)
        tester.run_statistical_tests(args.prompts_dir, args.iterations)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
