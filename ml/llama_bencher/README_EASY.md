# llama-bench Performance Testing - Quick Start Guide

## Run Benchmarks

### Run a benchmark with your config
```bash
python llama_bench_runner.py --config configs/config_m4max_gemma3_4b.json
```

## Generate Charts

### Generate text generation (tg) charts (default)
```bash
python plot_llama_bench_results.py
```

### Generate prompt processing (pp) charts
```bash
python plot_llama_bench_results.py --test-type pp
```

### Generate text generation (tg) charts explicitly
```bash
python plot_llama_bench_results.py --test-type tg
```

### Generate both pp and tg charts
```bash
python plot_llama_bench_results.py --test-type both
```

## Chart Types Generated

When you run the plotter, you'll get:
- **Prompt Size Performance**: How performance scales with different prompt sizes
- **Hardware Comparison**: Compare performance across different hardware configurations

## Favorite Analysis Commands

### Run benchmark and immediately plot tg results
```bash
python llama_bench_runner.py --config configs/config_m4max_gemma3_4b.json && python plot_llama_bench_results.py
```

### Run benchmark and plot both pp and tg results
```bash
python llama_bench_runner.py --config configs/config_m4max_gemma3_4b.json && python plot_llama_bench_results.py --test-type both
```

### Compare prompt processing speed across hardware
```bash
python plot_llama_bench_results.py --test-type pp
```

## Understanding Test Types

- **pp (Prompt Processing)**: Measures how fast the model processes input tokens
  - Higher is better
  - Tests different prompt sizes: 128, 256, 512, 1024, 2048 tokens
  - Results saved to: `*__llama_bench_pp.csv`

- **tg (Text Generation)**: Measures sustained token generation speed
  - Lower than pp, but more realistic for actual usage
  - Tests different context sizes with continuous generation
  - Results saved to: `*__llama_bench_tg.csv`

## Results Location

Results are stored in:
```
results/<model>/<quantization>/<hardware>/
├── <timestamp>__llama_bench_pp.csv
└── <timestamp>__llama_bench_tg.csv
```

Charts are saved in:
```
charts/
├── llama_bench_prompt_size_pp.html
├── llama_bench_hardware_comparison_pp.html
├── llama_bench_prompt_size_tg.html
└── llama_bench_hardware_comparison_tg.html
```

## Quick Tips

- **Default test type**: TG (text generation) - most representative of real-world performance
- **PP tests**: Useful for understanding input processing overhead
- **Interactive charts**: Open the HTML files in a browser, hover for details, zoom, pan
- **Multiple configs**: Create different JSON configs for different models/hardware combinations
