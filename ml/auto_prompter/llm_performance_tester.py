#!/usr/bin/env python3
"""
LLM Performance Tester (refactored for new config schema v3)

This script sends prompts from text files to an LLM API and measures performance
including tokens per second. Results are saved to a CSV file for analysis.
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

try:
    import tiktoken  # type: ignore
except ImportError:  # graceful fallback
    tiktoken = None  # noqa: N816
from config_loader import load_benchmark_config


class LLMPerformanceTester:
    def __init__(
        self,
        config_path: str = "configs/config_lm_studio.json",
        hardware_override: Optional[str] = None,
    ):
        """Initialize the performance tester with configuration."""
        self.cfg = load_benchmark_config(config_path)
        self.results = []
        self.tokenizer = self._initialize_tokenizer()
        # Hardware override just changes slug/device_model
        if hardware_override:
            self.cfg.hardware.device_model = hardware_override
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

        Args:
            model_name: Full model name (e.g., "llama3.3:70b-instruct-q4_K_M")

        Returns:
            Tuple of (base_model, quantization)
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
        if tiktoken is None:
            return None
        try:
            model_name = self.cfg.target.model.lower()
            if any(name in model_name for name in ["gpt-4", "gpt-3.5"]):
                return tiktoken.encoding_for_model("gpt-4")
            elif "gpt" in model_name:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base") if tiktoken else None

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text using the appropriate tokenizer."""
        if self.tokenizer is None:
            return len(text) // 4
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text) // 4

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
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
        if "max_tokens" in self.cfg.raw:
            payload["max_tokens"] = self.cfg.raw["max_tokens"]
        for p in [
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
        # Handle error responses or invalid responses
        if not isinstance(response, dict):
            print(f"Warning: Response is not a dictionary, got {type(response)}")
            return 0, 0.0

        # Check if this is an error response
        if "object" in response and response["object"] == "error":
            print("Error response received, cannot calculate tokens")
            return 0, 0.0

        # Try different ways to extract token count from response
        token_count = 0

        # Common fields where token count might be stored
        token_fields = [
            "eval_count",  # Ollama
            "tokens_evaluated",
            "completion_tokens",  # OpenAI-style
            "output_tokens",
            "tokens",
        ]

        # Check if we have usage information (OpenAI/chat completions format)
        if "usage" in response:
            usage = response["usage"]
            if "completion_tokens" in usage:
                token_count = usage["completion_tokens"]
            elif "total_tokens" in usage:
                token_count = usage["total_tokens"]

        # If no usage info, try other fields
        if token_count == 0:
            for field in token_fields:
                if field in response:
                    token_count = response[field]
                    break

        # If no token count found, estimate based on response content
        if token_count == 0:
            response_text = ""

            # Try to extract response text from different formats
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    response_text = choice["message"][
                        "content"
                    ]  # Chat completions format
                elif "text" in choice:
                    response_text = choice["text"]  # Completions format
            elif "response" in response:
                response_text = response["response"]  # Ollama format

            if response_text:
                # Rough estimation: ~4 characters per token
                estimated_tokens = len(response_text) // 4
                token_count = estimated_tokens
                print(f"Estimated token count: {token_count}")

        # Calculate tokens per second
        tokens_per_second = token_count / response_time if response_time > 0 else 0

        return token_count, tokens_per_second

    def select_prompts_interactively(
        self, prompts: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Allow user to interactively select which prompts to run."""
        print("\n=== Available Prompts ===")
        for i, (filename, prompt) in enumerate(prompts, 1):
            # Show a preview of the prompt
            preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            token_count = self.count_tokens(prompt)
            print(f"{i:2d}. {filename} ({token_count} tokens) - {preview}")

        print(f"{len(prompts) + 1:2d}. Run ALL prompts")

        while True:
            try:
                selection = input(
                    f"\nSelect prompt(s) to run (1-{len(prompts) + 1}), comma-separated numbers (e.g., 1,3,5), or 'all': "
                ).strip()

                if not selection:
                    continue

                # Handle "all" option (both by number and by word)
                if selection.lower() == "all" or selection == str(len(prompts) + 1):
                    return prompts

                # Handle single number or comma-separated numbers
                selected_indices = []
                for num_str in selection.split(","):
                    num_str = num_str.strip()
                    if num_str.isdigit():
                        num = int(num_str)
                        if 1 <= num <= len(prompts):
                            selected_indices.append(num - 1)  # Convert to 0-based index
                        else:
                            raise ValueError(f"Number {num} is out of range")
                    else:
                        raise ValueError(f"'{num_str}' is not a valid number")

                if not selected_indices:
                    print("No valid selections made. Please try again.")
                    continue

                # Return selected prompts
                selected_prompts = [prompts[i] for i in selected_indices]
                print(f"\nSelected {len(selected_prompts)} prompt(s):")
                for filename, _ in selected_prompts:
                    print(f"  - {filename}")

                return selected_prompts

            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                return []

    def warm_up_model(self) -> bool:
        """Send a simple warm-up request to ensure the model is loaded in memory."""
        print("🔥 Warming up the model...")

        # Use a simple, short prompt for warm-up
        warm_up_prompt = "Hello, please respond with just 'Ready'."

        try:
            response, response_time, success = self.send_curl_request(warm_up_prompt)

            if success:
                print(f"✓ Model warmed up successfully in {response_time:.2f}s")
                # Wait a moment for the model to settle in memory
                time.sleep(2)
                return True
            else:
                print(f"⚠️  Warm-up request failed, but continuing with tests...")
                return False

        except Exception as e:
            print(f"⚠️  Warm-up failed with error: {e}, but continuing with tests...")
            return False

    def test_prompt(self, filename: str, prompt: str) -> Dict:
        """Test a single prompt and return results."""
        print(f"\n--- Testing prompt from {filename} ---")
        prompt_token_count = self.count_tokens(prompt)
        print(f"Prompt length: {prompt_token_count} tokens")
        print(f"Prompt preview: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        response, response_time, success = self.send_curl_request(prompt)

        result = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "prompt_token_count": prompt_token_count,
            "response_time": response_time,
            "success": success,
            "response_token_count": 0,
            "tokens_per_second": 0,
            "model": self.cfg.target.model,
            "llm_url": self.cfg.target.llm_url,
        }

        if success and response:
            response_token_count, tokens_per_second = self.calculate_tokens_per_second(
                response, response_time
            )
            result.update(
                {
                    "response_token_count": response_token_count,
                    "tokens_per_second": tokens_per_second,
                }
            )

            print(
                f"✓ Success! Response tokens: {response_token_count}, Time: {response_time:.2f}s, Tokens/sec: {tokens_per_second:.2f}"
            )
        else:
            print(f"✗ Failed after {response_time:.2f}s")

        return result

    def run_tests(self, prompts_dir: str = "prompts", interactive: bool = True) -> None:
        """Run tests on all prompts and save results."""
        print("=== LLM Performance Tester ===")
        print(f"LLM URL: {self.cfg.target.llm_url}")
        print(f"Model: {self.cfg.target.model}")
        print(f"Loading prompts from: {prompts_dir}")

        # Warm up the model first
        self.warm_up_model()

        prompts = self.load_prompts(prompts_dir)

        if not prompts:
            print("No prompts found. Please add .txt files to the prompts directory.")
            return

        print(f"Found {len(prompts)} prompts to test")

        # Interactive prompt selection
        if interactive:
            selected_prompts = self.select_prompts_interactively(prompts)
            if not selected_prompts:
                print("No prompts selected. Exiting.")
                return
        else:
            selected_prompts = prompts
            print("Running all prompts in non-interactive mode.")

        print(f"\nStarting benchmark tests on {len(selected_prompts)} prompt(s)...\n")

        for filename, prompt in selected_prompts:
            result = self.test_prompt(filename, prompt)
            self.results.append(result)

            # Small delay between requests to be polite to the API
            time.sleep(1)

        self.save_results()

    def _canonical_quant(self) -> str:
        q = (self.cfg.target.quant or "").lower()
        mapping = self.hardware_mapping.get("quantization_mappings", {})
        return mapping.get(q, q)

    def _build_result_filename(
        self, kind: str = "perf", timestamp: str | None = None
    ) -> str:
        if timestamp is None:
            from datetime import datetime as _dt

            timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
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
        raw_quant = self.cfg.target.quant  # preserve original for filename
        parts = [
            timestamp,
            self.cfg.target.model.split("/")[-1].replace("-", "_"),
            raw_quant,
            self.cfg.target.backend,
            self.cfg.target.runtime,
            alloc_short + (f"_{vram_part}" if vram_part else ""),
            self.cfg.hardware.identifier(),
        ]
        if kind:
            parts.append(kind)
        safe = []
        for p in parts:
            p = str(p).lower()
            p = re.sub(r"[^a-z0-9]+", "_", p)
            p = re.sub(r"_+", "_", p).strip("_")
            safe.append(p)
        return "__".join([p for p in safe if p]) + ".csv"

    def save_results(self):
        if not self.results:
            print("No results to save")
            return
        base_model = self.cfg.target.model.split("/")[-1]
        quant = self._canonical_quant()
        hw_id = self.cfg.hardware.identifier()
        results_dir = os.path.join("results", base_model, quant, hw_id)
        os.makedirs(results_dir, exist_ok=True)
        filename = self._build_result_filename()
        output_file = os.path.join(results_dir, filename)
        fieldnames = [
            "timestamp",
            "filename",
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
            "hardware_make",
            "hardware_model",
            "hardware_cpu",
            "hardware_mem_gb",
            "hardware_gpu",
            "hardware_slug",
            "environment_os",
            "schema_version",
        ]
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.results:
                row = r.copy()
                row.update(
                    {
                        "model": self.cfg.target.model,
                        "llm_url": self.cfg.target.llm_url,
                        "base_model": base_model,
                        "quantization": quant,
                        "backend": self.cfg.target.backend,
                        "runtime": self.cfg.target.runtime,
                        "hardware_make": self.cfg.hardware.make,
                        "hardware_model": self.cfg.hardware.device_model,
                        "hardware_cpu": self.cfg.hardware.cpu,
                        "hardware_mem_gb": self.cfg.hardware.memory_gb,
                        "hardware_gpu": self.cfg.hardware.gpu,
                        "hardware_slug": hw_id,
                        "environment_os": self.cfg.environment.os,
                        "schema_version": self.cfg.schema_version,
                    }
                )
                row = {k: row.get(k, "") for k in fieldnames}
                w.writerow(row)
        print(f"Saved results to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM performance with multiple prompts"
    )
    parser.add_argument(
        "--config", default="configs/config_ollama.json", help="Configuration file path"
    )
    parser.add_argument(
        "--prompts-dir", default="prompts", help="Directory containing prompt files"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run all prompts without interactive selection",
    )
    parser.add_argument(
        "--hardware",
        help="Hardware identifier for organizing results (e.g., 'm3max', 'rtx4090'). If not specified, hardware will be auto-detected.",
    )

    args = parser.parse_args()

    try:
        tester = LLMPerformanceTester(args.config, hardware_override=args.hardware)
        tester.run_tests(args.prompts_dir, interactive=not args.no_interactive)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
