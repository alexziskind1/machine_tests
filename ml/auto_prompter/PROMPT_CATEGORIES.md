# LLM Prompt Testing Categories

This document organizes all prompts by length and type for systematic LLM performance comparison.

## Prompt Length Categories

### SHORT PROMPTS (1-10 words)
- `short_simple_greeting.txt` - Simple greeting
- `short_simple_math.txt` - Basic math question
- `short_programming_basic.txt` - Simple programming request
- `short_programming_debug.txt` - Basic bug fix
- `short_architecture_basic.txt` - Basic architecture request

### MEDIUM PROMPTS (50-100 words)
- `medium_non_programming_explanation.txt` - Technical explanation
- `medium_non_programming_creative.txt` - Creative writing task
- `medium_programming_algorithm.txt` - Algorithm implementation
- `medium_programming_debug.txt` - Debug with code sample
- `medium_architecture_web_app.txt` - Web application architecture
- `medium_architecture_social_media.txt` - Social media platform architecture

### LONG PROMPTS (1000 words)
- `long_non_programming_summarize.txt` - Long story summarization
- `long_non_programming_analysis.txt` - Comprehensive analysis task
- `long_programming_full_project.txt` - Complete project implementation
- `long_programming_debug_complex.txt` - Complex debugging task
- `long_architecture_enterprise.txt` - Enterprise architecture design

### EXTRA LONG PROMPTS (2000+ words, mostly code)
- `extra_long_programming_code_heavy.txt` - Code-heavy debugging and refactoring task

## Prompt Type Categories

### NON-PROGRAMMING PROMPTS
**Short:**
- `short_simple_greeting.txt`
- `short_simple_math.txt`

**Medium:**
- `medium_non_programming_explanation.txt`
- `medium_non_programming_creative.txt`

**Long:**
- `long_non_programming_summarize.txt`
- `long_non_programming_analysis.txt`

### PROGRAMMING PROMPTS
**Short:**
- `short_programming_basic.txt`
- `short_programming_debug.txt`

**Medium:**
- `medium_programming_algorithm.txt`
- `medium_programming_debug.txt`

**Long:**
- `long_programming_full_project.txt`
- `long_programming_debug_complex.txt`

**Extra Long:**
- `extra_long_programming_code_heavy.txt`

### SOFTWARE ARCHITECTURE PROMPTS
**Short:**
- `short_architecture_basic.txt`

**Medium:**
- `medium_architecture_web_app.txt`
- `medium_architecture_social_media.txt`

**Long:**
- `long_architecture_enterprise.txt`

## Legacy Prompts (from original set)
- `01_creative_writing.txt` - Creative writing (medium non-programming)
- `02_educational_explanation.txt` - Educational explanation (medium non-programming)
- `03_code_generation.txt` - Code generation (medium programming)
- `04_business_analysis.txt` - Business analysis
- `05_simple_question.txt` - Simple question
- `06_detailed_recipe.txt` - Detailed recipe

## Testing Methodology Suggestions

1. **Length Comparison**: Test same concept across different lengths
2. **Type Comparison**: Compare programming vs non-programming vs architecture
3. **Complexity Comparison**: Simple vs complex within same category
4. **Response Quality Metrics**: 
   - Accuracy
   - Completeness
   - Relevance
   - Clarity
   - Time to generate
   - Token usage

## Usage with LLM Performance Tester

These prompts are designed to work with the `llm_performance_tester.py` script in this workspace. Each file contains a single prompt that can be used to test and compare LLM performance across different dimensions.
