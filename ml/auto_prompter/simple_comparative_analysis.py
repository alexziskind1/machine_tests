#!/usr/bin/env python3
"""
Simple Comparative Analysis for LLM Performance Results

Key comparative insights for benchmarking different models and configurations.
"""

import csv
import os
import glob
from collections import defaultdict


def analyze_comparative_performance():
    """Analyze and compare performance across all available results."""
    
    results_dir = "results"
    stats_files = glob.glob(os.path.join(results_dir, "**/*statistics*.csv"), recursive=True)
    
    if not stats_files:
        print("No statistics files found in results directory!")
        return
    
    print(f"\n🔍 COMPARATIVE ANALYSIS OF {len(stats_files)} TEST CONFIGURATIONS\n")
    print("="*80)
    
    # Data collection
    all_results = []
    model_performance = defaultdict(list)
    hardware_performance = defaultdict(list)
    prompt_performance = defaultdict(list)
    
    for file_path in stats_files:
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                
                # Extract metadata from filename
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                
                hardware_config = '_'.join(parts[:2]) if len(parts) > 2 else 'unknown'
                
                for row in reader:
                    if float(row['successful_runs']) > 0:  # Only successful tests
                        performance = float(row['tokens_per_second_mean'])
                        cv = float(row['tokens_per_second_cv'])
                        model = row['model']
                        prompt = row['filename']
                        
                        result = {
                            'hardware': hardware_config,
                            'model': model,
                            'prompt': prompt,
                            'performance': performance,
                            'cv': cv,
                            'response_time': float(row['response_time_mean']),
                            'prompt_tokens': int(row['prompt_token_count'])
                        }
                        
                        all_results.append(result)
                        model_performance[model].append(performance)
                        hardware_performance[hardware_config].append(performance)
                        prompt_performance[prompt].append(performance)
                        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_results:
        print("No successful test results found!")
        return
    
    print(f"📊 Analyzed {len(all_results)} successful test results")
    print(f"🤖 Models tested: {len(model_performance)}")
    print(f"💻 Hardware configs: {len(hardware_performance)}")
    print(f"📝 Prompt types: {len(prompt_performance)}")
    
    # Model Comparison
    print(f"\n🤖 MODEL PERFORMANCE COMPARISON")
    print("-" * 50)
    
    model_stats = []
    for model, performances in model_performance.items():
        avg_perf = sum(performances) / len(performances)
        min_perf = min(performances)
        max_perf = max(performances)
        model_stats.append((model, avg_perf, min_perf, max_perf, len(performances)))
    
    # Sort by average performance
    model_stats.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Model':<40} {'Avg':<8} {'Min':<8} {'Max':<8} {'Tests':<6}")
    print("-" * 75)
    for model, avg, min_p, max_p, count in model_stats:
        model_short = model[:39] if len(model) > 39 else model
        print(f"{model_short:<40} {avg:>7.1f} {min_p:>7.1f} {max_p:>7.1f} {count:>5}")
    
    # Hardware Performance Analysis
    print(f"\n💻 HARDWARE CONFIGURATION ANALYSIS")
    print("-" * 50)
    
    hardware_stats = []
    for hardware, performances in hardware_performance.items():
        avg_perf = sum(performances) / len(performances)
        hardware_stats.append((hardware, avg_perf, len(performances)))
    
    hardware_stats.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Hardware Config':<30} {'Avg Performance':<15} {'Tests':<6}")
    print("-" * 55)
    for hardware, avg, count in hardware_stats:
        print(f"{hardware:<30} {avg:>14.1f} {count:>5}")
    
    # Prompt Complexity Analysis
    print(f"\n📝 PROMPT COMPLEXITY IMPACT")
    print("-" * 50)
    
    # Categorize prompts by size
    prompt_categories = {
        'Short (< 100 tokens)': [],
        'Medium (100-1000 tokens)': [],
        'Long (1000-10000 tokens)': [],
        'Extra Long (> 10000 tokens)': []
    }
    
    for result in all_results:
        tokens = result['prompt_tokens']
        perf = result['performance']
        
        if tokens < 100:
            prompt_categories['Short (< 100 tokens)'].append(perf)
        elif tokens < 1000:
            prompt_categories['Medium (100-1000 tokens)'].append(perf)
        elif tokens < 10000:
            prompt_categories['Long (1000-10000 tokens)'].append(perf)
        else:
            prompt_categories['Extra Long (> 10000 tokens)'].append(perf)
    
    for category, performances in prompt_categories.items():
        if performances:
            avg_perf = sum(performances) / len(performances)
            print(f"{category:<25} {avg_perf:>10.1f} tokens/sec ({len(performances)} tests)")
    
    # Top Performers Analysis
    print(f"\n🏆 TOP PERFORMING CONFIGURATIONS")
    print("-" * 50)
    
    # Sort all results by performance
    sorted_results = sorted(all_results, key=lambda x: x['performance'], reverse=True)
    
    print(f"{'Rank':<4} {'Performance':<12} {'Model':<25} {'Hardware':<20}")
    print("-" * 65)
    
    for i, result in enumerate(sorted_results[:10], 1):
        model_short = result['model'][:24] if len(result['model']) > 24 else result['model']
        hardware_short = result['hardware'][:19] if len(result['hardware']) > 19 else result['hardware']
        print(f"{i:<4} {result['performance']:>11.1f} {model_short:<25} {hardware_short:<20}")
    
    # Reliability Analysis  
    print(f"\n🎯 RELIABILITY ANALYSIS (Coefficient of Variation)")
    print("-" * 50)
    
    reliable_configs = []
    for result in all_results:
        if result['cv'] < 20:  # CV < 20% is considered reliable
            reliable_configs.append(result)
    
    print(f"Configurations with CV < 20%: {len(reliable_configs)}/{len(all_results)} ({len(reliable_configs)/len(all_results)*100:.1f}%)")
    
    if reliable_configs:
        # Sort reliable configs by performance
        reliable_configs.sort(key=lambda x: x['performance'], reverse=True)
        
        print(f"\nTop 5 Reliable & Fast Configurations:")
        print(f"{'Performance':<12} {'CV%':<6} {'Model':<25} {'Hardware':<15}")
        print("-" * 65)
        
        for result in reliable_configs[:5]:
            model_short = result['model'][:24] if len(result['model']) > 24 else result['model']
            hardware_short = result['hardware'][:14] if len(result['hardware']) > 14 else result['hardware']
            print(f"{result['performance']:>11.1f} {result['cv']:>5.1f} {model_short:<25} {hardware_short:<15}")
    
    # Performance per Token Analysis
    print(f"\n📊 EFFICIENCY ANALYSIS")
    print("-" * 50)
    
    # Calculate performance normalized by prompt length
    efficiency_data = []
    for result in all_results:
        if result['prompt_tokens'] > 0:
            efficiency = result['performance'] / (result['prompt_tokens'] / 1000)  # Performance per 1K tokens
            efficiency_data.append((result['model'], result['hardware'], efficiency))
    
    # Group by model for efficiency comparison
    model_efficiency = defaultdict(list)
    for model, hardware, eff in efficiency_data:
        model_efficiency[model].append(eff)
    
    print("Model Efficiency (Performance per 1K prompt tokens):")
    print(f"{'Model':<40} {'Avg Efficiency':<15}")
    print("-" * 58)
    
    for model, efficiencies in model_efficiency.items():
        avg_eff = sum(efficiencies) / len(efficiencies)
        model_short = model[:39] if len(model) > 39 else model
        print(f"{model_short:<40} {avg_eff:>14.1f}")
    
    print(f"\n✅ Analysis complete! Consider running the full dashboard for visual charts.")
    print(f"💡 Tip: Run 'python comparative_analysis.py' for interactive visualizations")


if __name__ == "__main__":
    analyze_comparative_performance()
