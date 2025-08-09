from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Dict
from datetime import datetime

_slug_re = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    t = text.lower()
    t = _slug_re.sub("_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


@dataclass
class HardwareConfig:
    make: str
    device_model: str
    cpu: str
    memory_gb: int
    gpu: str
    is_igpu: bool = False
    vram_allocation: str = "dynamic"
    vram_gb: float = 0
    notes: str = ""
    code: str | None = None  # stable short identifier

    def slug(self) -> str:
        base = f"{self.make}_{self.device_model}"
        return _slugify(base)

    def identifier(self) -> str:
        # If an explicit code is provided, use it verbatim (trimmed) without slugifying.
        if self.code and self.code.strip():
            return self.code.strip()
        return self.slug()


@dataclass
class EnvironmentConfig:
    os: str = "unknown"


@dataclass
class TargetConfig:
    llm_url: str
    model: str
    quant: str
    backend: str
    runtime: str

    def base_model(self) -> str:
        # Take portion after last slash if present
        raw = self.model.split("/")[-1]
        raw = raw.replace("-", "_")
        return _slugify(raw)


@dataclass
class BenchmarkConfig:
    schema_version: int
    headers: Dict[str, Any]
    hardware: HardwareConfig
    environment: EnvironmentConfig
    target: TargetConfig
    request_timeout: int
    output_csv: str
    raw: Dict[str, Any]


def load_benchmark_config(path: str) -> BenchmarkConfig:
    with open(path, "r") as f:
        data = json.load(f)

    hardware_raw = data.get("hardware", {})
    env_raw = data.get("environment", {})
    target_raw = data.get("target", {})

    hardware = HardwareConfig(
        make=hardware_raw.get("make", "unknown"),
        device_model=hardware_raw.get("device_model", "machine"),
        cpu=hardware_raw.get("cpu", "cpu"),
        memory_gb=hardware_raw.get("memory_gb", 0),
        gpu=hardware_raw.get("gpu", "gpu"),
        is_igpu=hardware_raw.get("is_igpu", False),
        vram_allocation=hardware_raw.get("vram_allocation", "dynamic"),
        vram_gb=hardware_raw.get("vram_gb", 0),
        notes=hardware_raw.get("notes", ""),
        code=hardware_raw.get("code"),
    )
    environment = EnvironmentConfig(os=env_raw.get("os", "unknown"))
    target = TargetConfig(
        llm_url=target_raw["llm_url"],
        model=target_raw["model"],
        quant=target_raw["quant"],
        backend=target_raw["backend"],
        runtime=target_raw["runtime"],
    )

    return BenchmarkConfig(
        schema_version=data.get("schema_version", 1),
        headers=data.get("headers", {}),
        hardware=hardware,
        environment=environment,
        target=target,
        request_timeout=data.get("request_timeout", 300),
        output_csv=data.get("output_csv", "results.csv"),
        raw=data,
    )
