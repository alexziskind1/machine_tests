#!/usr/bin/env python3
"""
Configuration loader for MLX benchmarking suite.
Follows the same schema pattern as the llama_bencher suite.
"""

import json
import platform
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class HardwareConfig:
    """Hardware configuration"""

    make: str
    device_model: str
    code: str
    cpu: str
    memory_gb: int
    is_igpu: bool
    gpu: str
    vram_allocation: str
    vram_gb: int
    notes: str = ""

    def identifier(self) -> str:
        """Generate a unique identifier for this hardware configuration."""
        return self.code.lower().replace(" ", "_").replace("-", "_")


@dataclass
class EnvironmentConfig:
    """Environment configuration"""

    os: str


@dataclass
class TargetConfig:
    """Target model and runtime configuration"""

    model_path: str
    model_name: str
    quant: str
    max_tokens: int = 100
    tokenizer_config: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Benchmark parameters"""

    prompt_sizes: list
    output_tokens: int
    iterations: int = 5
    warmup_iterations: int = 2
    temperature: float = 0.0
    repetition_penalty: float = 1.0


@dataclass
class BenchmarkConfigFull:
    """Full benchmark configuration"""

    schema_version: int
    hardware: HardwareConfig
    environment: EnvironmentConfig
    target: TargetConfig
    benchmark: BenchmarkConfig
    raw: Dict[str, Any]  # Store raw config for additional fields


def load_benchmark_config(config_path: str) -> BenchmarkConfigFull:
    """Load and parse benchmark configuration from JSON file."""
    with open(config_path, "r") as f:
        data = json.load(f)

    # Validate schema version
    schema_version = data.get("schema_version", 1)
    if schema_version != 1:
        print(f"Warning: Expected schema version 1, got {schema_version}")

    # Parse hardware config
    hw_data = data["hardware"]
    hardware = HardwareConfig(
        make=hw_data["make"],
        device_model=hw_data["device_model"],
        code=hw_data["code"],
        cpu=hw_data["cpu"],
        memory_gb=hw_data["memory_gb"],
        is_igpu=hw_data["is_igpu"],
        gpu=hw_data["gpu"],
        vram_allocation=hw_data["vram_allocation"],
        vram_gb=hw_data["vram_gb"],
        notes=hw_data.get("notes", ""),
    )

    # Parse environment config
    env_data = data["environment"]
    environment = EnvironmentConfig(os=env_data["os"])

    # Parse target config
    target_data = data["target"]
    target = TargetConfig(
        model_path=target_data["model_path"],
        model_name=target_data["model_name"],
        quant=target_data["quant"],
        max_tokens=target_data.get("max_tokens", 100),
        tokenizer_config=target_data.get("tokenizer_config"),
    )

    # Parse benchmark config
    bench_data = data["benchmark"]
    benchmark = BenchmarkConfig(
        prompt_sizes=bench_data["prompt_sizes"],
        output_tokens=bench_data["output_tokens"],
        iterations=bench_data.get("iterations", 5),
        warmup_iterations=bench_data.get("warmup_iterations", 2),
        temperature=bench_data.get("temperature", 0.0),
        repetition_penalty=bench_data.get("repetition_penalty", 1.0),
    )

    return BenchmarkConfigFull(
        schema_version=schema_version,
        hardware=hardware,
        environment=environment,
        target=target,
        benchmark=benchmark,
        raw=data,
    )


def detect_system_info() -> Dict[str, str]:
    """Auto-detect system information."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
