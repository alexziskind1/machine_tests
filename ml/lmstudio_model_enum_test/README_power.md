# LM Studio Benchmark with Power Monitoring

This is an enhanced version of the LM Studio benchmark script that adds GPU power monitoring capabilities during model inference.

## Features

- **Original functionality**: Benchmarks LLM models through LM Studio API
- **Power monitoring**: Tracks GPU power consumption during inference
- **Cross-platform support**: 
  - macOS: Uses `powermetrics` (requires sudo)
  - Windows/Linux with NVIDIA: Uses `nvidia-smi`
  - AMD/Intel GPUs: Placeholder implementations (to be completed)

## Files

- `main.py`: Original benchmark script
- `main_with_power.py`: Enhanced script with power monitoring
- `requirements.txt`: Python dependencies

## Usage

### Basic usage (with power monitoring):
```bash
python main_with_power.py
```

### Quick development (first model only):
```bash
python main_with_power.py --first-model
```

### Disable power monitoring:
```bash
python main_with_power.py --no-power
```

### Override GPU detection:
```bash
python main_with_power.py --gpu-names "RTX 4090" --gpu-memory "24"
```

## Power Monitoring Details

### macOS
- Uses `sudo powermetrics` to collect system power metrics
- Attempts to extract GPU power from the plist output
- **Note**: Requires sudo permissions for powermetrics

### NVIDIA GPUs (Windows/Linux)
- Uses `nvidia-smi --query-gpu=power.draw` to get real-time power consumption
- Supports multiple GPUs (sums total power)
- Samples power every second during inference

### AMD GPUs
- **Not yet implemented**: Placeholder for future AMD-specific tools
- Could use `rocm-smi`, `amdgpu-top`, or similar tools

### Intel GPUs
- **Not yet implemented**: Placeholder for future Intel-specific tools
- Could use `intel_gpu_top` or similar tools

## Output

The script generates CSV files with these additional power-related columns:
- `avg_gpu_power_watts`: Average GPU power during inference
- `max_gpu_power_watts`: Peak GPU power during inference
- `min_gpu_power_watts`: Minimum GPU power during inference
- `power_samples`: Number of power readings collected

Output files are named with the pattern:
```
benchmark_power_{hostname}_{cpu}_{gpu_info}.csv
```

## Requirements

- Python 3.7+
- LM Studio installed and configured
- Dependencies: `pip install -r requirements.txt`
- For macOS power monitoring: sudo access
- For NVIDIA power monitoring: nvidia-smi available in PATH

## Notes

- Power monitoring runs in a background thread during inference
- If power monitoring fails, the benchmark continues without power data
- The script automatically detects GPU vendor and chooses appropriate monitoring method
- Power readings are sampled approximately every second during inference
