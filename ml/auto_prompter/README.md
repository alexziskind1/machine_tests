# LLM Performance Tester

A Python script that tests LLM performance by sending prompts from text files and measuring tokens per second.

## Features

- ✅ Sends prompts to any LLM API via curl
- ✅ Measures response time and calculates tokens per second
- ✅ Configurable LLM URL and model settings
- ✅ Processes multiple prompt files automatically
- ✅ Saves results to CSV for analysis and graphing
- ✅ Handles timeouts and error cases gracefully
- ✅ Works with various LLM APIs (Ollama, OpenAI-compatible, etc.)
- ✅ Statistical testing framework with multiple iterations for reliable results
- ✅ Confidence intervals, error bars, and reliability metrics
- ✅ Interactive statistical visualizations with color-coded reliability indicators

## Setup

1. **Install Dependencies**
   ```bash
   # Ensure curl is installed (usually pre-installed on macOS/Linux)
   curl --version
   
   # Python 3.6+ required (uses only standard library)
   python3 --version
   ```

2. **Configure Your LLM**
   
   Edit `config.json` to match your LLM setup:
   
   ```json
   {
     "llm_url": "http://localhost:11434/api/generate",
     "model": "llama2",
     "headers": {
       "Content-Type": "application/json"
     },
     "request_timeout": 300,
     "output_csv": "llm_performance_results.csv"
   }
   ```

   **For different LLM providers:**
   
   - **Ollama** (default config): `http://localhost:11434/api/generate`
   - **OpenAI API**: `https://api.openai.com/v1/chat/completions` (requires API key in headers)
   - **Local servers**: Adjust URL and model name accordingly

3. **Add Your Prompts**
   
   Place your prompt text files in the `prompts/` directory. Each `.txt` file will be processed as a separate test case.

## Usage

### Basic Usage
```bash
python3 llm_performance_tester.py
```

### Custom Configuration
```bash
python3 llm_performance_tester.py --config my_config.json --prompts-dir my_prompts/
```

## Statistical Testing for Reliable Results

For reliable performance benchmarking, use the statistical testing framework that runs multiple iterations and provides confidence intervals:

### Statistical Performance Testing
```bash
# Run statistical tests with multiple iterations (default: 5 per prompt)
python3 statistical_llm_tester.py

# Custom number of iterations
python3 statistical_llm_tester.py --iterations 10

# Custom configuration
python3 statistical_llm_tester.py --config my_config.json --iterations 7
```

The statistical tester provides:
- **Multiple iterations** per prompt for reliable averages
- **Statistical metrics**: Mean, standard deviation, coefficient of variation (CV)
- **95% confidence intervals** for performance estimates
- **Outlier detection** and removal for cleaner results
- **Reliability classification**: Excellent (CV < 10%), Good (CV < 20%), Moderate (CV < 30%), Poor (CV ≥ 30%)

### Why Statistical Testing?

Single measurements can be unreliable due to:
- System load variations
- Memory allocation differences  
- Network latency fluctuations
- LLM internal state changes

Statistical testing with 5+ iterations provides:
- **Confidence intervals**: Know the true performance range
- **Reliability indicators**: Identify consistent vs variable performance
- **Error detection**: Spot outliers and system issues
- **Reproducible benchmarks**: Results you can trust and compare

### Statistical Output Format

Results are saved with detailed statistics:
- `*_statistics_*.csv`: Summary statistics (mean, std dev, CV, confidence intervals)
- `*_detailed_*.csv`: All individual test results for analysis

**Example Statistical Output:**
```
=== Statistical LLM Performance Testing ===
Testing each prompt 5 times for reliable statistics...

--- Testing prompt: short_simple_greeting_1t.txt (1/16) ---
✓ Iteration 1/5: 89.2 tokens/sec
✓ Iteration 2/5: 87.8 tokens/sec  
✓ Iteration 3/5: 91.1 tokens/sec
✓ Iteration 4/5: 88.5 tokens/sec
✓ Iteration 5/5: 90.3 tokens/sec
📊 Statistics: 89.4 ± 1.3 tokens/sec (CV: 1.5% - Excellent reliability)

=== Final Results ===
✅ Total prompts tested: 16
✅ Total iterations: 80  
✅ Overall success rate: 100.0%
✅ Average performance: 76.1 ± 2.1 tokens/sec
✅ Average reliability (CV): 3.9% (Excellent)
```

### Example Output
```
=== LLM Performance Tester ===
LLM URL: http://localhost:11434/api/generate
Model: llama2
Loading prompts from: prompts
Loaded prompt from 01_creative_writing.txt
Loaded prompt from 02_educational_explanation.txt
Found 6 prompts to test

--- Testing prompt from 01_creative_writing.txt ---
Prompt length: 156 characters
Prompt preview: Write a short story about a robot who discovers emotions for the first time...
Sending request to http://localhost:11434/api/generate...
✓ Success! Tokens: 245, Time: 12.34s, Tokens/sec: 19.87

=== Results saved to results/llm_performance_results_20250724_162845.csv ===
Total tests: 6
Successful tests: 6/6
Average tokens per second: 18.45
```

## Results Organization

All results and charts are automatically organized in the `results/` directory:

**Single Test Results:**
- **CSV files**: Single-run performance results (e.g., `llm_performance_results_20250724_162845.csv`)
- **HTML files**: Interactive charts for single results (e.g., `llm_performance_plots.html`)

**Statistical Test Results:**
- **Statistics CSV**: Summary statistics with means, std dev, CV (e.g., `ollama_statistics_20250725_092307.csv`)  
- **Detailed CSV**: All individual iteration results (e.g., `ollama_detailed_20250725_092307.csv`)
- **Statistical HTML**: Statistical dashboards with error bars (e.g., `statistical_performance_dashboard.html`)

### Working with Results

**For Statistical Results (Recommended):**
- **View statistical dashboard**: Use `python3 plot_statistical_results.py` to automatically find and visualize the latest statistical results
- **Analyze reliability**: Check coefficient of variation (CV) to identify consistent performance
- **Compare confidence intervals**: Use error bars to understand true performance ranges

**For Single Test Results:**
- **View latest results**: Use `python3 latest_results.py` to automatically find and visualize the most recent single-run results
- **List all results**: Use `python3 latest_results.py --list` to see all available result files
- **Manual analysis**: Use `python3 analyze_results.py --csv results/your_file.csv` for specific files

**Reliability Analysis:**
- **Check existing results**: Use `python3 reliability_analyzer.py` to analyze whether your current results are reliable
- **Compare methods**: Use `python3 comparison_analysis.py` to see differences between single vs statistical testing

## Output CSV Format

### Single Test Results

The single-run results are saved to a CSV file with the following columns:

- `timestamp`: When the test was run
- `filename`: Name of the prompt file
- `prompt_length`: Length of the prompt in characters
- `response_time`: Total response time in seconds
- `success`: Whether the request succeeded
- `token_count`: Number of tokens in the response
- `tokens_per_second`: Calculated performance metric
- `model`: LLM model used
- `llm_url`: API endpoint used
- `response_preview`: First 200 characters of the response

### Statistical Test Results

The statistical results include two CSV files:

**Statistics Summary (`*_statistics_*.csv`):**
- `filename`: Name of the prompt file
- `prompt_token_count`: Length of the prompt in tokens
- `iterations`: Number of test iterations performed
- `successful_runs`: Number of successful iterations
- `tokens_per_second_mean`: Average performance
- `tokens_per_second_std`: Standard deviation
- `tokens_per_second_cv`: Coefficient of variation (%)
- `tokens_per_second_95_ci_lower/upper`: 95% confidence interval bounds
- `response_time_mean/std`: Response time statistics
- `model`: LLM model used
- `llm_url`: API endpoint used

**Detailed Results (`*_detailed_*.csv`):**
- All individual iteration results
- Same columns as single test format
- Useful for deep analysis and outlier investigation

## Configuration Options

### `config.json` Parameters

- `llm_url`: API endpoint for your LLM
- `model`: Model name to use
- `headers`: HTTP headers for the request
- `request_timeout`: Maximum time to wait for a response (seconds)
- `output_csv`: Base name for the output CSV file (will be saved in `results/` with timestamp)

### Command Line Options

- `--config`: Path to configuration file (default: `config.json`)
- `--prompts-dir`: Directory containing prompt files (default: `prompts`)

## Understanding Statistical Metrics

### Key Reliability Indicators

- **Coefficient of Variation (CV)**: Measures consistency
  - **Excellent (CV < 10%)**: Very reliable, consistent performance
  - **Good (CV < 20%)**: Reliable for most use cases  
  - **Moderate (CV < 30%)**: Some variability, acceptable for rough estimates
  - **Poor (CV ≥ 30%)**: High variability, unreliable for benchmarking

- **95% Confidence Intervals**: Range where true performance likely falls
  - Narrow intervals = reliable measurements
  - Wide intervals = need more iterations or indicate system issues

- **Standard Deviation**: Spread of individual measurements
  - Lower values = more consistent performance
  - Higher values = more variable performance

### Example Interpretation

```
Prompt: long_programming_debug_complex_1366t.txt
Performance: 75.2 ± 1.8 tokens/sec (CV: 2.4% - Excellent)
95% CI: [73.1, 77.3] tokens/sec
```

This indicates:
- ✅ Very reliable measurement (CV < 10%)
- ✅ True performance is between 73.1-77.3 tokens/sec with 95% confidence
- ✅ Standard deviation of only 1.8 shows consistent performance
- ✅ This result can be trusted for benchmarking and comparison

## LLM Compatibility

This script works with any LLM that accepts JSON POST requests. The token counting logic attempts to detect tokens from various response formats:

- **Ollama**: Uses `eval_count` field
- **OpenAI-style APIs**: Uses `completion_tokens`
- **Other APIs**: Falls back to estimating tokens from response length

## Data Analysis

The CSV output can be easily imported into analysis tools:

### Statistical Visualizations

For statistical test results, use the statistical plotting script:

```bash
# Install required packages (one-time setup)
/usr/bin/python3 -m pip install pandas plotly --user

# Generate statistical dashboard (recommended)
/usr/bin/python3 plot_statistical_results.py --chart all

# Generate specific statistical charts
/usr/bin/python3 plot_statistical_results.py --chart tokens     # Performance with error bars
/usr/bin/python3 plot_statistical_results.py --chart reliability # CV and success rates  
/usr/bin/python3 plot_statistical_results.py --chart confidence  # 95% confidence intervals
/usr/bin/python3 plot_statistical_results.py --chart scatter     # Prompt length vs performance
/usr/bin/python3 plot_statistical_results.py --chart summary     # Statistics table

# Automatically finds latest statistical results
/usr/bin/python3 plot_statistical_results.py
```

**Statistical Chart Features:**
- **Color-coded reliability**: Green (reliable), orange (moderate), red (poor reliability)
- **Error bars**: Show standard deviation and confidence intervals
- **Interactive tooltips**: Detailed statistics on hover
- **Comprehensive legends**: Clear interpretation of color coding and metrics
- **Scatter plots**: Relationship between prompt length and performance with reliability indicators

### Interactive Plotly Visualizations (Single Results)

For single-run results, use the original plotting script:

Use the included plotting script to generate interactive charts:

```bash
# Install required packages (one-time setup)
/usr/bin/python3 -m pip install pandas plotly --user

# Generate interactive plots
/usr/bin/python3 plot_results.py

# View plots in browser
./view_plots.sh
```

**Available visualizations:**
- **Dashboard**: Combined overview with all key metrics
- **Summary Statistics**: Performance metrics table
- **Tokens per Second**: Bar chart by prompt type
- **Response Time**: Time analysis by prompt
- **Token Count**: Output volume comparison
- **Scatter Plot**: Performance vs prompt length relationship
- **Efficiency**: Tokens per second per input character

**Chart Options:**
```bash
# Generate specific chart types
/usr/bin/python3 plot_results.py --chart dashboard
/usr/bin/python3 plot_results.py --chart tokens
/usr/bin/python3 plot_results.py --chart scatter

# Custom output file
/usr/bin/python3 plot_results.py --output my_analysis.html
```

### Python with pandas
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('llm_performance_results.csv')

# Plot tokens per second by prompt
plt.figure(figsize=(10, 6))
plt.bar(df['filename'], df['tokens_per_second'])
plt.xticks(rotation=45)
plt.ylabel('Tokens per Second')
plt.title('LLM Performance by Prompt')
plt.tight_layout()
plt.show()
```

### Excel/Google Sheets
Simply open the CSV file to create charts and analyze the performance data.

## Troubleshooting

1. **Connection refused**: Make sure your LLM server is running
2. **Timeout errors**: Increase `request_timeout` in config
3. **No tokens detected**: Check if your LLM returns token counts in the response
4. **Permission denied**: Make sure the script has write permissions for the output directory

## Example Prompts

The `prompts/` directory includes example prompts of varying complexity:
- Creative writing tasks
- Educational explanations  
- Code generation
- Business analysis
- Simple questions
- Detailed instructions

Add your own `.txt` files to test with different prompt types and lengths.

## License

MIT License - feel free to modify and use for your projects!
