# Configuration Files

This folder contains all configuration files for the LLM Performance Testing suite.

## Available Configurations

### 🦙 `config_ollama.json` (Default)
- **Purpose**: Testing models through Ollama API
- **URL**: `http://127.0.0.1:11434/v1/chat/completions`
- **Example Model**: `gemma3:4b-it-q4_K_M`
- **Use Case**: Local Ollama server testing

### 🎯 `config_lm_studio.json`
- **Purpose**: Testing models through LM Studio API  
- **URL**: `http://127.0.0.1:1234/v1/chat/completions`
- **Example Model**: `llama-3.3-70b-instruct@q8_0`
- **Use Case**: LM Studio local server testing

### ⚡ `config_mlc.json`
- **Purpose**: Testing models through MLC Chat API
- **URL**: `http://127.0.0.1:8000/v1/chat/completions`
- **Example Model**: `hi`
- **Use Case**: MLC-LLM framework testing

### 🌐 `config_openai_example.json`
- **Purpose**: Testing models through OpenAI API
- **URL**: `https://api.openai.com/v1/chat/completions`
- **Example Model**: `gpt-3.5-turbo`
- **Use Case**: Cloud API testing (requires API key)

### 📊 `config_statistical.json`
- **Purpose**: Template for statistical reliability testing
- **URL**: `http://127.0.0.1:11434/v1/chat/completions`
- **Example Model**: `your_model_name`
- **Use Case**: Multi-iteration statistical analysis

## Quick Start

### Single Test Run
```bash
# Use default Ollama config
python llm_performance_tester.py

# Or specify a different config
python llm_performance_tester.py --config configs/config_lm_studio.json
```

### Statistical Testing
```bash
# Use default Ollama config
python statistical_llm_tester.py

# Or specify a different config  
python statistical_llm_tester.py --config configs/config_lm_studio.json
```

## Configuration Structure

All config files include:

### Core Settings
- `llm_url`: API endpoint URL
- `model`: Model name/identifier
- `headers`: HTTP headers (including auth if needed)
- `request_timeout`: Request timeout in seconds
- `output_csv`: Base filename for results

### Results Organization
- `_comment_results_organization`: Explains nested folder structure `/results/<model>/<quantization>/<hardware>/`
- `_comment_hardware_override`: Explains `--hardware` command-line option

### Generation Parameters
- `max_tokens`: Maximum response length
- `temperature`: Response randomness (0.1 for consistent benchmarking)

### Statistical Parameters
- `iterations_per_prompt`: Number of runs per prompt
- `outlier_threshold`: Standard deviations for outlier detection
- `warmup_iterations`: Initial runs to discard
- `cooldown_between_iterations`: Delay between runs
- `cooldown_between_prompts`: Delay between different prompts

### Reliability Thresholds
- `max_acceptable_cv`: Maximum coefficient of variation (%)
- `min_success_rate`: Minimum success rate (%)

## Customization

1. **Copy a config file**: `cp configs/config_ollama.json configs/my_config.json`
2. **Edit your settings**: Update URL, model, and parameters
3. **Use your config**: `--config configs/my_config.json`

## Notes

- **Default Config**: `config_ollama.json` is used when no `--config` is specified
- **Hardware Detection**: Auto-detects Apple Silicon (M1/M2/M3/M4) or use `--hardware` override
- **Results Organization**: All configs automatically organize results in nested folders
- **JSON Validation**: All files are validated JSON - check syntax if editing manually
