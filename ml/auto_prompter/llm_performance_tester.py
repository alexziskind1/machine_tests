#!/usr/bin/env python3
"""
LLM Performance Tester

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
import tiktoken


class LLMPerformanceTester:
    def __init__(self, config_path: str = "configs/config_ollama.json", hardware_override: Optional[str] = None):
        """Initialize the performance tester with configuration."""
        self.config = self.load_config(config_path)
        self.results = []
        self.tokenizer = self._initialize_tokenizer()
        self.hardware_info = hardware_override or self._detect_hardware()
        self.hardware_mapping = self._load_hardware_mapping()

    def _load_hardware_mapping(self) -> Dict:
        """Load hardware mapping configuration."""
        try:
            with open("hardware_mapping.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load hardware mapping: {e}")
            return {"hardware_mappings": {}, "quantization_mappings": {}, "model_mappings": {}}

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
                        timeout=5
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
            r'q\d+(?:_[a-z0-9]+)*',  # q4_k_m, q8_0, q4km, q3, etc.
            r'fp\d+',                # fp16, fp32
            r'int\d+',               # int4, int8
            r'bf\d+',                # bf16
            r'mlx@?\d*bit',          # mlx4bit, mlx8bit, mlx@4bit
            r'mlx\d+bit',            # mlx4bit, mlx8bit
        ]
        
        quantization = "unknown"
        base_model = model_name
        
        # First try to find quantization in the model name
        for pattern in quant_patterns:
            match = re.search(pattern, model_name)
            if match:
                quantization = match.group()
                # Normalize common patterns
                if quantization == 'q4km':
                    quantization = 'q4_k_m'
                elif 'mlx' in quantization:
                    quantization = re.sub(r'mlx@?(\d+)bit', r'mlx\1bit', quantization)
                
                # Remove quantization from model name to get base model
                base_model = re.sub(f'[-_:@]{re.escape(match.group())}', '', model_name)
                break
        
        # Clean up base model name
        base_model = re.sub(r'[-_:](instruct|chat|it)$', '', base_model)
        base_model = re.sub(r'[^\w\.]', '_', base_model)
        
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
            model_name = self.config.get("model", "").lower()

            # Map common model names to tiktoken encodings
            if any(name in model_name for name in ["gpt-4", "gpt-3.5"]):
                return tiktoken.encoding_for_model("gpt-4")
            elif "gpt" in model_name:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Use cl100k_base as a reasonable default for most modern models
                return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(
                f"Warning: Could not initialize specific tokenizer for model {self.config.get('model', 'unknown')}: {e}"
            )
            print("Using cl100k_base encoding as fallback")
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text using the appropriate tokenizer."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {e}")
            # Fallback to character-based estimation
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

    def send_curl_request(self, prompt: str) -> Tuple[Optional[Dict], float, bool]:
        """Send a curl request to the LLM and measure response time."""
        # Use chat completions format
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        # Add response length control parameters if specified in config
        if "max_tokens" in self.config:
            payload["max_tokens"] = self.config["max_tokens"]

        # Add other generation parameters if specified
        generation_params = [
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
        ]
        for param in generation_params:
            if param in self.config:
                payload[param] = self.config[param]

        # Prepare curl command - removed timing option that was interfering with JSON
        curl_cmd = [
            "curl",
            "-s",  # Silent mode
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

        print(f"Sending request to {self.config['llm_url']}...")
        start_time = time.time()

        try:
            # Execute curl command
            result = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.config["request_timeout"],
            )

            end_time = time.time()
            total_time = end_time - start_time

            if result.returncode == 0:
                # Try to parse the response
                response_text = result.stdout.strip()

                try:
                    # Parse JSON response
                    response_data = json.loads(response_text)

                    # Check if the response contains an error
                    if (
                        isinstance(response_data, dict)
                        and "object" in response_data
                        and response_data["object"] == "error"
                    ):
                        print(
                            f"API Error: {response_data.get('message', 'Unknown error')}"
                        )
                        return response_data, total_time, False

                    return response_data, total_time, True
                except json.JSONDecodeError as e:
                    # If we can't parse JSON, show more detail for debugging
                    print(f"JSON parsing error: {e}")
                    print(f"Response preview: {response_text[:200]}...")
                    return {"raw_response": response_text}, total_time, False
            else:
                print(f"Curl failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return None, total_time, False

        except subprocess.TimeoutExpired:
            print(f"Request timed out after {self.config['request_timeout']} seconds")
            return None, self.config["request_timeout"], False
        except Exception as e:
            print(f"Error executing curl command: {e}")
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
            "model": self.config["model"],
            "llm_url": self.config["llm_url"],
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
        print(f"LLM URL: {self.config['llm_url']}")
        print(f"Model: {self.config['model']}")
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

    def save_results(self) -> None:
        """Save results to CSV file with nested folder structure: /results/<model>/<quantization>/<hardware>."""
        if not self.results:
            print("No results to save")
            return

        # Parse model information
        model_name = self.config.get("model", "unknown_model")
        base_model, quantization = self._parse_model_info(model_name)
        hardware = self._normalize_hardware_name(self.hardware_info)
        
        # Create nested directory structure
        results_dir = os.path.join("results", base_model, quantization, hardware)
        os.makedirs(results_dir, exist_ok=True)

        # Generate timestamped filename
        base_filename = self.config.get("output_csv", "performance_results.csv")
        name, ext = os.path.splitext(base_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(results_dir, f"{name}_{timestamp}{ext}")

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
            "hardware",
        ]

        try:
            with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in self.results:
                    # Add parsed model info to each result
                    enhanced_result = result.copy()
                    enhanced_result.update({
                        "base_model": base_model,
                        "quantization": quantization,
                        "hardware": hardware,
                    })
                    
                    # Only include fields that exist in fieldnames
                    filtered_result = {
                        k: v for k, v in enhanced_result.items() if k in fieldnames
                    }
                    writer.writerow(filtered_result)

            print(f"\n=== Results saved to {output_file} ===")
            print(f"Organization: {base_model}/{quantization}/{hardware}")
            print(f"Total tests: {len(self.results)}")
            successful_tests = sum(1 for r in self.results if r["success"])
            print(f"Successful tests: {successful_tests}/{len(self.results)}")

            if successful_tests > 0:
                avg_tokens_per_sec = (
                    sum(r["tokens_per_second"] for r in self.results if r["success"])
                    / successful_tests
                )
                print(f"Average tokens per second: {avg_tokens_per_sec:.2f}")

        except Exception as e:
            print(f"Error saving results: {e}")


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
