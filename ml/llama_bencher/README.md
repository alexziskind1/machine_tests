# llama-bench Benchmarking Suite

A standalone, self-contained benchmarking automation and visualization suite for llama.cpp's `llama-bench` utility. This suite automates testing across multiple prompt sizes, stores results with comprehensive metadata, and generates interactive performance charts.

## Overview

This suite provides:
- **Automated benchmarking** with configurable prompt sizes and iterations
- **Organized result storage** in nested directory structure by model/quantization/hardware
- **Interactive visualizations** showing performance across prompt sizes and hardware
- **Statistical analysis** with mean, standard deviation, and confidence intervals
- **Flexible configuration** similar to the existing auto_prompter suite

## Directory Structure

```
llama_bench/
├── configs/                          # Configuration files
│   └── config_m4max_gemma3_4b.json  # Example config for M4 Max
├── results/                          # Results organized by model/quant/hardware
│   └── <model>/<quant>/<hardware>/  # Auto-generated nested structure
├── charts/                           # Generated visualization charts
├── config_loader.py                  # Configuration schema and loader
├── llama_bench_runner.py            # Main benchmark runner
├── plot_llama_bench_results.py      # Chart generation script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### Prerequisites

1. **llama.cpp built with llama-bench**
   - Clone and build llama.cpp: https://github.com/ggerganov/llama.cpp
   - The `llama-bench` binary will be in the `build/bin/` directory

2. **Python 3.8+** with pip

### Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Make scripts executable (optional)
chmod +x llama_bench_runner.py
chmod +x plot_llama_bench_results.py
```

## Configuration

Create a JSON configuration file in the `configs/` directory. The configuration follows the same schema pattern as the auto_prompter suite.

### Example Configuration

```json
{
  "schema_version": 1,
  
  "hardware": {
    "make": "Apple",
    "device_model": "MacBook Pro",
    "code": "mbp_m4max",
    "cpu": "M4 Max",
    "memory_gb": 128,
    "is_igpu": true,
    "gpu": "M4 Max",
    "vram_allocation": "dynamic",
    "vram_gb": 128,
    "notes": "Apple MacBook Pro with M4 Max (2024)"
  },
  
  "environment": {
    "os": "macOS Sequoia"
  },
  
  "target": {
    "llama_bench_path": "./llama-bench",
    "model_path": "~/.cache/lm-studio/models/lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf",
    "model_name": "gemma-3-4b-it",
    "quant": "Q4_K_M",
    "backend": "Metal",
    "threads": 12
  },
  
  "benchmark": {
    "prompt_sizes": [128, 256, 512, 1024, 4096, 16384, 32768],
    "output_tokens": 512,
    "iterations": 5,
    "warmup_iterations": 2
  }
}
```

### Configuration Fields

#### Hardware Section
- `make`: Hardware manufacturer (e.g., "Apple", "NVIDIA", "AMD")
- `device_model`: Device model name
- `code`: Short hardware identifier (used in folder structure)
- `cpu`: CPU model
- `memory_gb`: System RAM in GB
- `is_igpu`: Whether using integrated GPU
- `gpu`: GPU model
- `vram_allocation`: "dynamic" or "static"
- `vram_gb`: GPU VRAM in GB

#### Target Section
- `llama_bench_path`: Path to llama-bench executable (supports `~` expansion)
- `model_path`: Path to GGUF model file (supports `~` expansion)
- `model_name`: Display name for the model
- `quant`: Quantization level (e.g., "Q4_K_M", "fp16")
- `backend`: Backend used (e.g., "Metal", "CUDA", "BLAS")
- `threads`: Number of threads (optional, llama-bench default if omitted)

#### Benchmark Section
- `prompt_sizes`: Array of prompt sizes to test (in tokens)
- `output_tokens`: Number of tokens to generate per test
- `iterations`: Number of test iterations for statistical reliability
- `warmup_iterations`: Number of warmup runs before actual benchmarking

## Usage

### Running Benchmarks

```bash
# Run with default config
python llama_bench_runner.py

# Run with custom config
python llama_bench_runner.py --config configs/my_config.json
```

The script will:
1. Validate paths and configuration
2. Run warmup iterations
3. Execute benchmark iterations for each prompt size
4. Parse and store results
5. Print summary statistics

### Generating Charts

```bash
# Generate text generation (tg) charts (default)
python plot_llama_bench_results.py

# Generate prompt processing (pp) charts
python plot_llama_bench_results.py --test-type pp

# Generate both pp and tg charts
python plot_llama_bench_results.py --test-type both

# Filter by model
python plot_llama_bench_results.py --model gemma

# Filter by quantization
python plot_llama_bench_results.py --quant Q4_K_M

# Filter by hardware
python plot_llama_bench_results.py --hardware m4max

# Combine filters
python plot_llama_bench_results.py --test-type both --model gemma --quant Q4_K_M --hardware m4max

# Only show summary statistics
python plot_llama_bench_results.py --summary-only
```

### Chart Types Generated

The suite generates separate charts for two types of tests:

1. **Prompt Processing (pp) Charts** - Measures how fast the model processes input prompts
   - `llama_bench_prompt_size_pp.html` - Line chart showing pp performance vs prompt size
   - `llama_bench_hardware_comparison_pp.html` - Bar chart comparing hardware for pp tests

2. **Text Generation (tg) Charts** - Measures sustained token generation speed
   - `llama_bench_prompt_size_tg.html` - Line chart showing tg performance vs prompt size
   - `llama_bench_hardware_comparison_tg.html` - Bar chart comparing hardware for tg tests

**Performance Chart Features:**
   - Separate lines for each hardware+model+quant combination
   - Error bars showing standard deviation
   - Logarithmic x-axis for better visualization across large prompt size ranges
   - Interactive hover details

**Hardware Comparison Features:**
   - Grouped bars comparing different hardware configurations
   - Bars grouped by prompt size
   - Useful for comparing multiple hardware setups side-by-side

## Output

### Results Storage

Results are automatically organized in a nested directory structure with **separate files for pp and tg tests**:

```
results/
└── <model_name>/
    └── <quantization>/
        └── <hardware_code>/
            ├── <timestamp>__<model>__<quant>__<backend>__<hardware>__llama_bench_pp.csv
            └── <timestamp>__<model>__<quant>__<backend>__<hardware>__llama_bench_tg.csv
```

Example:
```
results/
└── gemma_3_4b_it/
    └── q4_k_m/
        └── mbp_m4max/
            ├── 20241117_143022__gemma_3_4b_it__Q4_K_M__Metal__mbp_m4max__llama_bench_pp.csv
            └── 20241117_143022__gemma_3_4b_it__Q4_K_M__Metal__mbp_m4max__llama_bench_tg.csv
```

### CSV Format

Each result CSV (both pp and tg files) contains:
- `timestamp`: ISO format timestamp
- `model_name`: Model name
- `model_size_gb`: Model size in GB
- `params_b`: Model parameters in billions
- `backend`: Inference backend
- `threads`: Number of threads used
- `test_type`: Test type (e.g., "pp128", "pp512", "tg512")
- `prompt_size`: Prompt size in tokens (for pp tests) or context size (for tg tests)
- `tokens_per_second`: Performance (tokens/second)
- `std_dev`: Standard deviation
- `quantization`: Quantization level
- `hardware_slug`: Hardware identifier
- `hardware_make`: Hardware manufacturer
- `hardware_model`: Hardware model
- `hardware_cpu`: CPU model
- `hardware_mem_gb`: System RAM
- `hardware_gpu`: GPU model
- `environment_os`: Operating system
- `model_path`: Full path to model file

**Note:** PP (prompt processing) tests measure input processing speed, while TG (text generation) tests measure sustained output generation speed. TG results are typically much lower but more representative of actual inference performance.

### Example Output

```
=== llama-bench Benchmark Runner ===
Model: gemma-3-4b-it
Quantization: Q4_K_M
Hardware: mbp_m4max
Prompt sizes: [128, 256, 512, 1024, 4096, 16384, 32768]
Output tokens: 512
Iterations: 5

🔥 Running 2 warmup iterations...
  Warmup 1/2
  Warmup 2/2
✓ Warmup complete

📊 Running 5 benchmark iterations...

================================================================================
Iteration 1/5
================================================================================
✓ Iteration 1 complete: 7 results

... (iterations 2-5)

================================================================================
Benchmark complete: 35 total results collected
================================================================================

=== Results Saved ===
File: results/gemma_3_4b_it/q4_k_m/mbp_m4max/20241117_143022__gemma_3_4b_it__Q4_K_M__Metal__mbp_m4max__llama_bench.csv
Organization: gemma_3_4b_it/q4_k_m/mbp_m4max
Total results: 35

================================================================================
📊 BENCHMARK SUMMARY
================================================================================

Prompt Size     Mean t/s        Min t/s         Max t/s         Samples   
--------------- --------------- --------------- --------------- ----------
128             1623.32         1610.45         1635.78         5         
256             1776.74         1770.12         1782.45         5         
512             1834.56         1833.21         1835.67         5         
...
```

## Example Workflow

1. **Configure your test**:
   ```bash
   cp configs/config_m4max_gemma3_4b.json configs/my_test.json
   # Edit my_test.json with your paths and settings
   ```

2. **Run the benchmark**:
   ```bash
   python llama_bench_runner.py --config configs/my_test.json
   ```

3. **Generate visualizations**:
   ```bash
   python plot_llama_bench_results.py
   ```

4. **View results**:
   - Open `charts/llama_bench_prompt_size.html` in your browser
   - Interactive charts with hover details and zoom capabilities

## Tips

- **Prompt Sizes**: Use powers of 2 or common context sizes (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
- **Iterations**: 5 iterations provide good statistical reliability; use more for production benchmarks
- **Warmup**: Always include warmup iterations to ensure the model is loaded and caches are warm
- **Multiple Runs**: Run tests at different times of day to account for thermal throttling
- **Comparison**: Use consistent prompt sizes across different hardware for fair comparisons

## Comparing Hardware

To compare different hardware configurations:

1. Run benchmarks on each system with the same model and prompt sizes
2. Use different config files with appropriate hardware codes
3. Generate comparison charts:
   ```bash
   python plot_llama_bench_results.py --model gemma
   ```

The hardware comparison chart will automatically group all configurations.

## Troubleshooting

### llama-bench not found
- Ensure `llama_bench_path` points to the correct executable
- Use absolute path or relative path from the llama_bench directory
- Check execute permissions: `chmod +x /path/to/llama-bench`

### Model not found
- Verify `model_path` is correct
- Ensure the path uses proper expansion (~ for home directory)
- Check file permissions

### No results generated
- Check llama-bench output for errors
- Verify model is compatible with your hardware
- Ensure sufficient RAM/VRAM for the model

### Charts not displaying
- Install required dependencies: `pip install -r requirements.txt`
- Check that result CSV files exist in the results directory
- Use `--summary-only` to verify data is loaded correctly

## Integration with auto_prompter Suite

This suite follows the same configuration schema and result organization as the auto_prompter suite, making it easy to:
- Compare llama-bench results with LLM inference results
- Use similar hardware mappings and configurations
- Maintain consistent metadata across different benchmarking tools

## License

This benchmarking suite is part of the machine_tests repository and follows the same license.

## Contributing

To add new features or fix issues:
1. Follow the existing code structure
2. Maintain compatibility with the configuration schema
3. Update this README with any new functionality
4. Test with multiple models and hardware configurations
