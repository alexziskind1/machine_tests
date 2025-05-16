# LLM Benchmark Analysis Conclusions

This document summarizes key findings from the LLM benchmark tests conducted on various hardware configurations. The primary performance metric analyzed is `tokens_per_second` (t/s).

## 1. Impact of GPU Memory (VRAM) on Performance

*   **General Trend:** Increasing the allocated GPU VRAM generally improves `tokens_per_second`, particularly for larger models. This is because more model layers can be offloaded to the faster GPU memory, reducing reliance on slower system RAM. The benefit typically plateaus once the model (or a significant portion of it) fits comfortably in VRAM.

*   **AMD APUs (az_flow128, azflowz13 - Radeon 8060S):**
    *   For models like `gemma-3-27B-it-QAT-Q4_0` on `az_flow128`, performance scaled positively with VRAM, increasing from ~10.5 t/s at 8GB to ~11.3 t/s at 64GB/96GB.
    *   For the large `Llama-3.3-70B-Instruct-Q4_K_M` model on `az_flow128`, `tokens_per_second` increased from ~4.47 t/s (8GB VRAM) to ~4.75 t/s (32GB VRAM).
    *   An interesting observation for `Llama-3.3-70B` on `az_flow128`: the 96GB VRAM setting (from `benchmark_az_flow128_..._96.csv`) showed *lower* performance (~4.38 t/s) than the 32GB setting. This particular CSV also reported significantly less `total_ram_gb` (31.6GB) compared to other tests on the same host (e.g., 95.6GB for the 32GB VRAM test). This suggests that on this UMA system, allocating a very large portion of memory to the GPU might have starved system RAM, leading to a bottleneck.
    *   The **'auto' VRAM setting** on `az_flow128` often resulted in lower performance compared to manually configured higher VRAM settings, especially for larger models. For instance, `Llama-3.3-70B` achieved only ~3.10 t/s with 'auto' VRAM, significantly less than with manually set VRAM. This implies 'auto' might be too conservative in GPU layer offloading for optimal performance on this hardware.

## 2. Impact of Model Size (Parameters) on Performance

*   **Inverse Relationship:** As expected, `tokens_per_second` generally decreases as the model's parameter count (`params_b`) and overall size (`size_gb`) increase, assuming the same hardware and VRAM allocation.
    *   On the **Apple M4 Max (96GB unified memory):**
        *   `gemma-3-1B-it-QAT-Q4_0` (1B params): ~199.8 t/s
        *   `gemma-3-4B-it-QAT-Q4_0` (4B params): ~114.8 t/s
        *   `gemma-3-12B-it-QAT-Q4_0` (12B params): ~49.4 t/s
        *   `Llama-3.3-70B-Instruct-Q4_K_M` (70B params): ~10.1 t/s
    *   This trend is consistent across all tested hosts and configurations.

## 3. Host Hardware Performance Comparison

*   **Apple M4 Max (`Mac_fios_router_home`):**
    *   Demonstrates very strong `tokens_per_second`, leveraging its 96GB of high-bandwidth unified memory effectively.
    *   Achieved ~10.14 t/s for `Llama-3.3-70B-Instruct-Q4_K_M`.
    *   Showed excellent performance for smaller models, e.g., ~200 t/s for `gemma-3-1B-it-QAT-Q4_0`.

*   **AMD RYZEN AI MAX 395 (`az_flow128` - higher system RAM variant):**
    *   With 32GB VRAM allocated, achieved ~4.75 t/s for `Llama-3.3-70B-Instruct-Q4_K_M`.
    *   For `gemma-3-1B-it-QAT-Q4_0`, performance peaked around ~150 t/s with 64GB or 96GB VRAM allocation.

*   **AMD RYZEN AI MAX 395 (`azflowz13` - lower system RAM variant):**
    *   Performance for smaller models like `gemma-3-1B-it-QAT-Q4_0` reached ~151 t/s with 16GB VRAM, comparable to `az_flow128`'s best for that model. This machine was generally tested with lower VRAM ceilings (up to 24GB).

*   **Overall:** The Apple M4 Max generally shows a performance advantage, especially for larger models, likely due to its powerful GPU, high memory bandwidth, and large unified memory pool. The AMD APUs also provide respectable performance, which scales with the amount of memory allocated to the GPU.

## 4. Impact of Quantization and Model Format

*   **Quantization Benefits:** More aggressively quantized models (e.g., Q4_K_M) are smaller and generally faster than less quantized versions (e.g., Q8_0) of the same base model.
    *   On M4 Max, `Llama-3.3-70B-Instruct-Q4_K_M` (~40GB) ran at ~10.14 t/s, while the larger `Llama-3.3-70B-Instruct-Q8_0` (~70GB) ran at ~6.47 t/s.
*   **Model Format:** The data includes models in GGUF and Safetensors formats. While direct comparisons are limited by model differences, the M4 Max showed very high throughput for a `Meta-Llama-3.1-8B-Instruct-4bit` (Safetensors) model at ~99 t/s.

## 5. Time to First Token (TTFT)

*   **Model Size Dependency:** TTFT generally increases with model size.
    *   On `az_flow128` (8GB VRAM): `gemma-3-1B` (0.021s) vs. `Llama-3.3-70B` (1.939s).
*   **VRAM Impact on TTFT:** Increasing VRAM tends to reduce TTFT for larger models.
    *   `Llama-3.3-70B` on `az_flow128`: TTFT dropped from 1.939s (8GB VRAM) to ~1.36s (16GB/32GB VRAM). The 'auto' VRAM setting had a significantly higher TTFT of 4.227s.
*   **Host Impact on TTFT:** The M4 Max exhibited low TTFT even for large models (e.g., 0.829s for `Llama-3.3-70B-Instruct-Q4_K_M`).

## 6. Small Model Performance Considerations

*   For very small models (e.g., `gemma-3-1B-it-QAT-Q4_0`, ~0.67GB), which fit comfortably even in minimal VRAM allocations, performance scaling with VRAM was sometimes inconsistent on the AMD APUs. For instance, on `az_flow128`, the 8GB VRAM configuration occasionally outperformed 16GB or 32GB VRAM settings for this specific model. This could be due to measurement variability, fixed system overheads having a proportionally larger impact, or specific UMA/driver behaviors when VRAM utilization is very low.

## 7. UMA System Considerations (AMD APUs)

*   The `gpu_memory_gb` column for the AMD APU tests likely refers to the UMA (Unified Memory Architecture) aperture size configured, rather than fully discrete VRAM.
*   The observation with `Llama-3.3-70B` on `az_flow128` (96GB VRAM allocation test having lower `total_ram_gb` and lower performance than the 32GB VRAM test) underscores the importance of balancing GPU memory allocation with available system RAM in UMA systems. Over-allocating to the GPU can starve the CPU and lead to performance degradation if the overall workload still requires substantial system RAM.

These conclusions are based on the provided dataset and may vary with different models, software versions, or more extensive testing.
