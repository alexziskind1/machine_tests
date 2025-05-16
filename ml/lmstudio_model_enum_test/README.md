# Benchmarking LLMs with LM Studio

This repository provides a Python script, `main.py`, to automate benchmarking of large language models (LLMs) using the LM Studio CLI. It loads each model, runs a sample query, records performance metrics, and outputs the results to a CSV file.

## Contents

- `main.py`: Main benchmarking script.
- `benchmark_results.csv`: Generated output containing performance metrics (created after running the script).
- `README.md`: This file.

## Requirements

- **Python 3.8+**
- **LM Studio CLI**: Installed and bootstrapped (`lms bootstrap`).
- **Python packages**:
  - `requests`

Install the Python dependency:

```bash
pip install requests
```

Ensure you have one or more LLMs installed in LM Studio:

```bash
lms install <model-identifier>
```

## Usage

1. **Start benchmarking**:
   ```bash
   python main.py
   ```
   This will:
   - Start the LM Studio REST API server on port 1234.
   - Enumerate all installed LLMs.
   - For each model:
     - Unload any loaded models.
     - Load the model with 100% GPU offload.
     - Send a fixed prompt and collect metrics.
     - Unload the model.
   - Write results to `benchmark_results.csv`.

2. **Development mode** (quick test on first model):
   ```bash
   python main.py --first-model
   ```
   Only the first discovered model is benchmarked, speeding up development iterations.

## Output

The script writes a CSV file, `benchmark_results.csv`, with the following columns:

- `model`: CLI model identifier (e.g., `llama-3.3-70b-instruct/q4_k_m`).
- `params_b`: Number of parameters (in billions).
- `size_gb`: Model file size on disk (in gigabytes).
- `architecture`: Model architecture (e.g., `llama`, `gemma3`).
- `format`: Model file format (e.g., `gguf`).
- `tokens_per_second`: Inference speed measured in tokens per second.
- `total_tokens`: Total tokens processed (prompt + completion).
- `time_to_first_token`: Time elapsed until the first token (in seconds).
- `generation_time`: Total generation time (in seconds).

## Customization

- Modify the `prompt` variable in `main.py` to benchmark with a different input.
- Adjust GPU offload by changing the `--gpu max` flag in the `load_model` function.

## Troubleshooting

- If no models are found, ensure you have installed them via `lms install`.
- Verify the LM Studio server is running if the script times out.

