# LLM Performance Testing - Quick Start Guide

## Run Tests

### Run the full test suite
```bash
python statistical_llm_tester.py --config configs/config_lm_studio.json
```

## Individual Result Analysis

### Plot individual result scatter chart
```bash
python plot_results.py --csv results/qwen3_coder_30b/int4/m4pro/20250801_115217__qwen3_coder_30b__q4_k_m__mlx__lmstudio__sta_64gb__m4pro__detailed.csv --chart scatter
```

## Hardware Performance Comparison

### Compare specific model/quantization across all hardware
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int4
```

### Compare specific hardware for a model/quantization
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int4 --hardware "Apple MacBook Pro M4 Pro"
```

### Compare multiple hardware configurations
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int4 --hardware "Apple MacBook Pro M4 Pro" "GMKTec EVO X2" "Framework Desktop"
```

## Memory Configuration Analysis

### Compare all Framework Desktop memory configurations
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int8 --hardware "Framework Desktop"
```

### Compare dynamic vs static memory allocations
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int8 --hardware 'Framework Desktop (Dynamic 0.5GB)' 'Framework Desktop (96GB)'
```

### Test specific memory configuration
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int8 --hardware 'Framework Desktop (Dynamic 0.5GB)'
```

## Favorite Analysis Commands

### Cross-hardware comparison for best model/quantization combo
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int4
```

### Framework vs GMKTec head-to-head (same quantization)
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int8 --hardware "Framework Desktop" "GMKTec EVO X2"
```

### Apple Silicon comparison (different models, same quantization)
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int4 --hardware "Apple MacBook Pro M4 Pro" "Apple MacBook Pro M4 Max" "Apple Mac Studio M3 Ultra"
```

### RTX vs Apple Silicon performance showdown
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int4 --hardware "NVIDIA RTX 5090 + RTX 5060 Ti Desktop" "Apple MacBook Pro M4 Max"
```

### Memory optimization analysis (Framework)
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int8 --hardware 'Framework Desktop (8GB)' 'Framework Desktop (32GB)' 'Framework Desktop (96GB)' 'Framework Desktop (Dynamic 0.5GB)'
```

### Response time analysis (instead of tokens/sec)
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int4 --metric response_time_mean
```

### Coefficient of variation analysis (performance consistency)
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int4 --metric tokens_per_second_cv
```

### Prompt processing delay analysis
```bash
python prompt_hardware_comparison.py --model qwen3-coder-30b --quantization int4 --delay-chart
```

### List all available models and quantizations
```bash
python prompt_hardware_comparison.py --list-available
```

### Show detected hardware names (for troubleshooting)
```bash
python prompt_hardware_comparison.py --show-hardware-names
```

## Available Options

- **Models**: qwen3-coder-30b, qwen3-235b, llama-3-3-70b, gpt-oss-20b, gpt-oss-120b, deepseek-r1-distill-qwen-7b
- **Quantizations**: int4, int8, int3, fp4, fp8, fp16
- **Metrics**: tokens_per_second_mean (default), response_time_mean, tokens_per_second_cv
- **Hardware**: Use `--show-hardware-names` to see all detected configurations

