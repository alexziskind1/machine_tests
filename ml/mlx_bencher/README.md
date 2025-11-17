# MLX Benchmarking Suite

Automated benchmarking suite for Apple MLX models. Runs performance tests across different prompt sizes and generates interactive visualizations.

## Overview

This suite provides:
- Automated MLX model benchmarking with configurable prompt sizes
- Separate measurement of **prompt processing (pp)** and **text generation (tg)** performance
- Interactive Plotly charts for performance analysis
- CSV result storage for historical tracking
- Hardware comparison across different Apple Silicon configurations

## Directory Structure

```
mlx_bencher/
├── configs/                          # Configuration files
│   └── config_m4max_gemma2_2b.json  # Example config for M4 Max
├── results/                          # Results organized by model/quant/hardware
│   └── <model>/<quant>/<hardware>/  # Auto-generated nested structure
├── charts/                           # Generated visualization charts
├── config_loader.py                  # Configuration schema and loader
├── mlx_bench_runner.py              # Main benchmark runner
├── plot_mlx_bench_results.py        # Chart generation script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── README_EASY.md                    # Quick start guide
```

## Installation

### Prerequisites

1. **Apple Silicon Mac** (M1, M2, M3, M4, or later)
2. **Python 3.9+** with pip
3. **MLX and MLX-LM** installed

### Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install MLX if not already installed
pip install mlx mlx-lm

# Make scripts executable (optional)
chmod +x mlx_bench_runner.py
chmod +x plot_mlx_bench_results.py
```

## Configuration

Configuration files use JSON format (schema version 1). Create one config file per model/hardware combination.

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
    "model_path": "mlx-community/gemma-2-2b-it-4bit",
    "model_name": "gemma-2-2b-it",
    "quant": "4bit",
    "max_tokens": 100
  },
  
  "benchmark": {
    "prompt_sizes": [128, 256, 512, 1024, 2048, 4096, 8192],
    "output_tokens": 512,
    "iterations": 5,
    "warmup_iterations": 2,
    "temperature": 0.0,
    "repetition_penalty": 1.0
  }
}
```

### Configuration Fields

#### Hardware Section
- `make`: Hardware manufacturer (e.g., "Apple")
- `device_model`: Device type (e.g., "MacBook Pro", "Mac Studio")
- `code`: Short identifier (e.g., "mbp_m4max")
- `cpu`: CPU model (e.g., "M4 Max")
- `memory_gb`: System RAM in GB
- `is_igpu`: true for integrated GPU (always true for Apple Silicon)
- `gpu`: GPU identifier (same as CPU for Apple Silicon)
- `vram_allocation`: "dynamic" for unified memory
- `vram_gb`: Available VRAM (same as memory_gb for unified memory)
- `notes`: Additional hardware information

#### Environment Section
- `os`: Operating system (e.g., "macOS Sequoia")

#### Target Section
- `model_path`: MLX model path or HuggingFace repo (e.g., "mlx-community/gemma-2-2b-it-4bit")
- `model_name`: Display name for the model
- `quant`: Quantization level (e.g., "4bit", "8bit", "fp16")
- `max_tokens`: Maximum tokens for generation (optional)
- `tokenizer_config`: Path to custom tokenizer config (optional)

#### Benchmark Section
- `prompt_sizes`: Array of prompt sizes to test (in tokens)
- `output_tokens`: Number of tokens to generate in tg tests
- `iterations`: Number of iterations per prompt size (default: 5)
- `warmup_iterations`: Number of warmup runs (default: 2)
- `temperature`: Generation temperature (default: 0.0 for deterministic)
- `repetition_penalty`: Repetition penalty (default: 1.0)

## Usage

### Running Benchmarks

```bash
# Run with specific config
python mlx_bench_runner.py --config configs/config_m4max_gemma2_2b.json

# Run with default config
python mlx_bench_runner.py
```

### Generating Charts

```bash
# Generate text generation (tg) charts (default)
python plot_mlx_bench_results.py

# Generate prompt processing (pp) charts
python plot_mlx_bench_results.py --test-type pp

# Generate both pp and tg charts
python plot_mlx_bench_results.py --test-type both
```

### Test Types

- **pp (Prompt Processing)**: Measures how fast the model processes input tokens
  - Higher throughput, typically 1000+ tokens/sec
  - Important for understanding input processing overhead
  - File pattern: `*__mlx_bench_pp.csv`

- **tg (Text Generation)**: Measures sustained token generation speed
  - Lower throughput, typically 30-100 tokens/sec  
  - More representative of actual inference performance
  - File pattern: `*__mlx_bench_tg.csv`

## Output

### Results Storage

Results are automatically organized in a nested directory structure:

```
results/
└── <model_name>/
    └── <quantization>/
        └── <hardware_code>/
            ├── <timestamp>__<model>__<quant>__MLX__<hardware>__mlx_bench_pp.csv
            └── <timestamp>__<model>__<quant>__MLX__<hardware>__mlx_bench_tg.csv
```

Example:
```
results/
└── gemma_2_2b_it/
    └── 4bit/
        └── mbp_m4max/
            ├── 20241117_143022__gemma_2_2b_it__4bit__MLX__mbp_m4max__mlx_bench_pp.csv
            └── 20241117_143022__gemma_2_2b_it__4bit__MLX__mbp_m4max__mlx_bench_tg.csv
```

### CSV Format

Each result CSV (both pp and tg files) contains:
- `timestamp`: ISO format timestamp
- `model_name`: Model name
- `model_size_gb`: Model size in GB
- `params_b`: Model parameters in billions
- `test_type`: Test type (e.g., "pp128", "tg512")
- `prompt_size`: Prompt size in tokens
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
- `model_path`: Full path to model

**Note:** PP (prompt processing) tests measure input processing speed, while TG (text generation) tests measure sustained output generation speed. TG results are typically much lower but more representative of actual inference performance.

### Charts

Generated charts are saved to the `charts/` directory:

**With --test-type pp:**
- `mlx_bench_prompt_size_pp.html`: Performance vs prompt size (pp tests)
- `mlx_bench_hardware_comparison_pp.html`: Hardware comparison (pp tests)

**With --test-type tg (default):**
- `mlx_bench_prompt_size_tg.html`: Performance vs prompt size (tg tests)
- `mlx_bench_hardware_comparison_tg.html`: Hardware comparison (tg tests)

**With --test-type both:**
- All four charts above

#### Chart Types

1. **Prompt Size Performance Chart**
   - Line chart showing tokens/second vs prompt size
   - Log scale x-axis for better visualization
   - Error bars showing standard deviation
   - Separate lines for each hardware/model/quant combination

2. **Hardware Comparison Chart**
   - Grouped bar chart comparing different hardware configurations
   - Bars grouped by prompt size
   - Easy to see relative performance differences

### Example Output

```
================================================================================
MLX Benchmark Runner
================================================================================

Model: gemma-2-2b-it
Quantization: 4bit
Hardware: MacBook Pro
Prompt sizes: [128, 256, 512, 1024, 2048, 4096, 8192]
Output tokens: 512
Iterations per size: 5

================================================================================

📥 Loading model from mlx-community/gemma-2-2b-it-4bit...
✓ Model loaded successfully

🔥 Running 2 warmup iterations...
  Warmup 1/2
  Warmup 2/2
✓ Warmup complete

📊 Running benchmarks for 7 prompt sizes...

================================================================================
Prompt Size: 128 tokens
================================================================================
  Running pp test (5 iterations)...
  ✓ pp: 1623.45 ± 12.34 tokens/sec
  Running tg test (5 iterations)...
  ✓ tg: 67.89 ± 2.31 tokens/sec
...
```

## Chart Features

- **Interactive**: Hover for detailed stats, click legend to show/hide series
- **Responsive**: Zoom, pan, and download as PNG
- **Error bars**: Show standard deviation for each measurement
- **Multiple configurations**: Compare different hardware/models/quantizations
- **Test type filtering**: Separate charts for pp vs tg performance

## Tips

- **Warmup iterations**: Important for consistent results (model loading, GPU cache warming)
- **Multiple iterations**: Use 5+ iterations for reliable statistics
- **Prompt sizes**: Start with smaller sizes (128-1024) for quick tests
- **Memory**: Larger prompt sizes may require more unified memory
- **Temperature 0.0**: Recommended for benchmarking (deterministic output)
- **MLX models**: Use models from `mlx-community` on HuggingFace for best compatibility

## Troubleshooting

### Model Loading Errors
- Ensure model path is correct (HuggingFace repo or local path)
- Check that MLX model format is supported
- Verify sufficient memory for model size

### Performance Issues
- Close other applications to free memory
- Reduce prompt sizes or iterations
- Check system activity monitor for resource usage

### Chart Generation Issues
- Ensure results directory contains CSV files
- Check that pandas and plotly are installed
- Verify file permissions

## License

MIT License
