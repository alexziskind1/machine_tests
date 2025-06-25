# Benchmarking LLMs with LM Studio

Automate performance benchmarking of large language models (LLMs) via the LM Studio CLI. This directory contains two benchmarking scripts with different capabilities:

## Scripts Overview

### `main.py` - Basic Benchmarking
The original script performs standard LLM benchmarking:

1. Detect machine specs (host, OS, CPU, RAM, GPU names & memory).
2. Start the LM Studio REST API server.
3. Enumerate installed LLMs and extract metadata (parameters, file size, architecture, format).
4. For each model:
   - Unload any loaded models.
   - Load with full GPU offload (`--gpu max`).
   - Send a fixed prompt and record timing stats.
   - Unload the model.
5. Save all results (including machine info) to a CSV file named by host, CPU, GPU details.

### `main2.py` - Enhanced Memory & Context Analysis
An enhanced version with additional capabilities:

**Key Differences from `main.py`:**
- **Multiple Context Length Testing**: Tests each model with different context lengths (default: 256, 2048, 8192 tokens)
- **Memory Usage Tracking**: Monitors RAM and GPU VRAM usage before/after model loading and inference
- **macOS/Apple Silicon Optimization**: Handles Apple Silicon GPU detection gracefully (hardcoded as "N/A")
- **Minimal Test Prompt**: Uses "Just say hello." instead of longer prompts for consistent comparison
- **Enhanced CSV Output**: Includes context length and memory usage columns
- **Resource Delta Reporting**: Shows memory usage changes during inference

**Use Cases:**
- Understanding memory requirements across different context lengths
- Analyzing resource efficiency on Apple Silicon Macs
- Detailed memory profiling for model deployment planning

---

## Contents

- `main.py` — Basic benchmarking script with machine and model metadata.
- `main2.py` — Enhanced script with memory tracking and multiple context length testing.
- `benchmark_<host>_<cpu>_<gpu>_<mem>.csv` — Output file generated per run.
- `README.md` — This documentation.

## Requirements

- **Python 3.8+**
- **LM Studio CLI**: Installed and bootstrapped (`lms bootstrap`).
- **Python packages**:
  - `requests`
  - `psutil`
  - `GPUtil`
  - `py-cpuinfo`

Install dependencies:
```bash
pip install requests psutil GPUtil py-cpuinfo
```

Ensure you have LLMs installed:
```bash
lms install <model-identifier>
```

---

## Usage

### Basic Benchmarking (`main.py`)

```bash
python main.py [options]
```

### Options

- `--first-model`  
  Benchmark only the first discovered model (for faster dev iterations).

- `--gpu-names <names>`  
  Override detected GPU names (semicolon-separated).

- `--gpu-memory <mem>`  
  Override detected GPU memory in GB (semicolon-separated).

### Enhanced Memory & Context Benchmarking (`main2.py`)

```bash
python main2.py [options]
```

### Options

- `--first-model`  
  Benchmark only the first discovered model (for faster dev iterations).

- `--contexts <lengths>`  
  Comma-separated context lengths to test in tokens (default: "256,2048,8192").

### Example

Benchmark all models with auto-detected hardware info:
```bash
python main.py
```

Quick dev run on first model, manually specify GPU mem:
```bash
python main.py --first-model --gpu-memory "24.0"
```

Test memory usage across specific context lengths:
```bash
python main2.py --contexts "512,4096,16384"
```

Run enhanced benchmarking on first model only:
```bash
python main2.py --first-model
```

---

## Output

Both scripts generate a CSV named like:
```
benchmark_<host>_<cpu>_<gpu_names>_<gpu_memory>.csv
```

### `main.py` Columns:

- **Machine info**:
  - `host`: sanitized hostname
  - `system`: OS name
  - `release`: OS version
  - `cpu`: sanitized CPU model
  - `total_ram_gb`: total system RAM
  - `gpu_names`: semicolon-separated GPU model names
  - `gpu_memory_gb`: semicolon-separated GPU memory (GB)

- **Model metadata**:
  - `model`: CLI load ID (e.g., `llama-3.3-70b-instruct/q4_k_m`)
  - `params_b`: parameter count (billions)
  - `size_gb`: disk size (GB)
  - `architecture`: model architecture
  - `format`: file format

- **Benchmark stats**:
  - `tokens_per_second`
  - `total_tokens`
  - `time_to_first_token`
  - `generation_time`

### `main2.py` Additional Columns:

All columns from `main.py` plus:

- **Context & Memory info**:
  - `context_tokens`: context length used for this test
  - `ram_loaded_gb`: RAM usage after model loading
  - `ram_after_query_gb`: RAM usage after inference
  - `gpu_loaded_gb`: GPU VRAM usage after model loading
  - `gpu_after_query_gb`: GPU VRAM usage after inference

---

## Customization

- Change the fixed `prompt` in `main.py` to benchmark different inputs.
- Adjust GPU offload by modifying `--gpu max` flag in the `load_model` function.

---

## Troubleshooting

- **No models found**: ensure installation with `lms install`.
- **Server startup issues**: verify `lms server start` manually.

---

Happy benchmarking!
