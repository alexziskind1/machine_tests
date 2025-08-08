run the full test
python statistical_llm_tester.py --config configs/config_lm_studio.json


plot individual result scatter
python plot_results.py --csv results/qwen3_coder_30b/4bit/m4pro64gb/m4pro_64GB_lm_studio_qwen3_coder_30b_q4km_detailed_20250801_115217.csv  --chart scatter


plot individual model/quant/hardware 
python prompt_hardware_comparison.py --model qwen3_coder_30b --quantization 4-bit --filter-hardware m4pro


plot combined model/quant
python prompt_hardware_comparison.py --model qwen3_coder_30b --quantization 4-bit

