# Token Counting Implementation

## Overview

The LLM Performance Tester has been updated to use accurate token counting instead of character counting. This provides much more meaningful metrics for LLM performance analysis.

## What Changed

### 1. Input Prompt Measurement
- **Before**: `prompt_length` (character count)
- **After**: `prompt_token_count` (actual token count using tiktoken)

### 2. CSV Output Columns
- **Before**: `prompt_length`, `token_count`, `tokens_per_second`
- **After**: `prompt_token_count`, `response_token_count`, `tokens_per_second`

### 3. Efficiency Calculation
- **Before**: tokens per second per character
- **After**: tokens per second per input token

## Implementation Details

### Tokenizer Selection
The system automatically selects the appropriate tokenizer based on the model name:
- **GPT-4**: Uses `tiktoken` encoding for GPT-4
- **GPT-3.5**: Uses `tiktoken` encoding for GPT-3.5-turbo  
- **Other models**: Uses `cl100k_base` encoding as a reasonable fallback

### Fallback Behavior
If tokenization fails, the system falls back to character-based estimation (characters ÷ 4).

## Dependencies

Added to `requirements.txt`:
```
tiktoken>=0.5.0  # For accurate token counting
```

## Alternative Tokenization Options

### Option 1: tiktoken (Current Implementation)
```python
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode(text)
```

**Pros:**
- Very accurate for OpenAI models and many others
- Fast and reliable
- Wide compatibility

**Cons:**
- May not be 100% accurate for non-OpenAI models
- Additional dependency

### Option 2: Model-Specific Tokenizers
```python
# For Llama/Code Llama models
from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# For other HuggingFace models
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model-name")
```

**Pros:**
- 100% accurate for specific models
- Accounts for model-specific vocabulary

**Cons:**
- Requires downloading model files
- Different tokenizer per model
- Much larger dependencies

### Option 3: LLM API Token Counting
```python
# Some APIs provide token count in request
payload = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": prompt}],
    "return_token_count": True  # If supported
}
```

**Pros:**
- 100% accurate for the specific API
- No additional dependencies

**Cons:**
- Not all APIs support this
- Requires extra API calls
- May have rate limits

### Option 4: Approximate Token Counting
```python
# Simple approximation
def estimate_tokens(text):
    # Average of ~4 characters per token for English
    return len(text.split()) * 1.3  # More accurate than char/4
```

**Pros:**
- No dependencies
- Fast
- Works for any model

**Cons:**
- Inaccurate
- Varies significantly by language and content type

## Recommended Approach

The current implementation using `tiktoken` with `cl100k_base` encoding is recommended because:

1. **Good accuracy** for most modern LLMs (most use similar tokenization)
2. **Fast and lightweight** compared to model-specific tokenizers
3. **No model downloads** required
4. **Consistent** across different models

## Testing the Implementation

Run a test to see the difference:
```bash
python llm_performance_tester.py --config config_ollama.json --no-interactive
```

The output now shows token counts instead of character counts:
- Prompt length: **17,736 tokens** (vs. 85,271 characters before)
- More accurate performance metrics
- Better comparison between different prompt types

## Next Steps

If you need even more accuracy for specific models:

1. **Identify your most-used models**
2. **Install model-specific tokenizers** for those models
3. **Update the `_initialize_tokenizer()` method** to use model-specific tokenizers
4. **Benchmark the difference** to see if the extra accuracy is worth the complexity

The current implementation provides a good balance of accuracy and simplicity for most use cases.
