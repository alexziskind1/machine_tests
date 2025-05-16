# Benchmarking LLMs with LM Studio

Automate performance benchmarking of large language models (LLMs) via the LM Studio CLI. The `main.py` script will:

1. Detect machine specs (host, OS, CPU, RAM, GPU names & memory).
2. Start the LM Studio REST API server.
3. Enumerate installed LLMs and extract metadata (parameters, file size, architecture, format).
4. For each model:
   - Unload any loaded models.
   - Load with full GPU offload (`--gpu max`).
   - Send a fixed prompt and record timing stats.
   - Unload the model.
5. Save all results (including machine info) to a CSV file named by host, CPU, GPU details.

---

## Contents

- `main.py` — Benchmarking script with machine and model metadata.
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

### Example

Benchmark all models with auto-detected hardware info:
```bash
python main.py
```

Quick dev run on first model, manually specify GPU mem:
```bash
python main.py --first-model --gpu-memory "24.0"
```

---

## Output

The script generates a CSV named like:
```
benchmark_<host>_<cpu>_<gpu_names>_<gpu_memory>.csv
```

Columns:

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
