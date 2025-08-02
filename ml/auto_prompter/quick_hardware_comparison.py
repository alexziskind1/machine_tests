#!/usr/bin/env python3
"""
Quick Hardware Comparison Tool

Get instant hardware performance rankings for specific model/quantization combinations.
"""

import csv
import os
import glob
from collections import defaultdict
import argparse


def quick_hardware_comparison(model_filter=None, quantization_filter=None):
    """Quick hardware performance comparison for specific model/quantization."""
    
    results_dir = "results"
    stats_files = glob.glob(os.path.join(results_dir, "**/*statistics*.csv"), recursive=True)
    
    if not stats_files:
        print("No statistics files found!")
        return
    
    # Collect data
    hardware_performance = defaultdict(list)
    all_configs = []
    
    for file_path in stats_files:
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                hardware_config = '_'.join(parts[:2]) if len(parts) > 2 else 'unknown'
                
                for row in reader:
                    if float(row['successful_runs']) > 0:
                        model = row['model']
                        
                        # Extract quantization from filename or model name
                        quantization = 'Unknown'
                        filename_lower = filename.lower()
                        model_lower = model.lower()
                        
                        # Check for 4-bit quantization (q4, 4bit, q4_k_m, etc.)
                        if any(x in filename_lower for x in ['q4', '4bit', '@4bit']) or \
                           any(x in model_lower for x in ['q4', '4bit', '@4bit']):
                            quantization = '4-bit'
                        # Check for 8-bit quantization (q8, 8bit, q8_0, etc.)  
                        elif any(x in filename_lower for x in ['q8', '8bit', '@8bit']) or \
                             any(x in model_lower for x in ['q8', '8bit', '@8bit']):
                            quantization = '8-bit'
                        # Check for 3-bit quantization
                        elif any(x in filename_lower for x in ['q3', '3bit', '@3bit']) or \
                             any(x in model_lower for x in ['q3', '3bit', '@3bit']):
                            quantization = '3-bit'
                        # Check for MLX optimization
                        elif 'mlx' in filename_lower or 'mlx' in model_lower:
                            quantization = 'MLX'
                        # Check for FP16/float16
                        elif any(x in filename_lower for x in ['fp16', 'float16', 'f16']) or \
                             any(x in model_lower for x in ['fp16', 'float16', 'f16']):
                            quantization = 'FP16'
                        
                        # Apply filters
                        if model_filter and model_filter.lower() not in model.lower():
                            continue
                        if quantization_filter and quantization_filter != quantization:
                            continue
                        
                        config = {
                            'hardware': hardware_config,
                            'model': model,
                            'quantization': quantization,
                            'performance': float(row['tokens_per_second_mean']),
                            'cv': float(row['tokens_per_second_cv']),
                            'response_time': float(row['response_time_mean']),
                            'tests': int(row['iterations'])
                        }
                        
                        all_configs.append(config)
                        hardware_performance[hardware_config].append(config['performance'])
                        
        except Exception as e:
            continue
    
    if not all_configs:
        print(f"❌ No matching configurations found for model='{model_filter}', quantization='{quantization_filter}'")
        return
    
    # Calculate hardware averages
    hardware_stats = {}
    for hw, performances in hardware_performance.items():
        hardware_stats[hw] = {
            'avg_performance': sum(performances) / len(performances),
            'min_performance': min(performances),
            'max_performance': max(performances),
            'test_count': len(performances)
        }
    
    # Sort by average performance
    sorted_hardware = sorted(hardware_stats.items(), 
                           key=lambda x: x[1]['avg_performance'], 
                           reverse=True)
    
    # Print results
    filter_text = []
    if model_filter:
        filter_text.append(f"Model: {model_filter}")
    if quantization_filter:
        filter_text.append(f"Quantization: {quantization_filter}")
    
    print(f"\n🏆 HARDWARE PERFORMANCE RANKING")
    if filter_text:
        print(f"🎯 Filters: {' | '.join(filter_text)}")
    print("=" * 70)
    
    print(f"{'Rank':<4} {'Hardware':<25} {'Avg Perf':<10} {'Range':<15} {'Tests':<6}")
    print("-" * 70)
    
    for i, (hardware, stats) in enumerate(sorted_hardware, 1):
        range_text = f"{stats['min_performance']:.1f}-{stats['max_performance']:.1f}"
        print(f"{i:<4} {hardware:<25} {stats['avg_performance']:>9.1f} {range_text:<15} {stats['test_count']:>5}")
    
    # Performance insights
    if len(sorted_hardware) > 1:
        best_hw = sorted_hardware[0]
        worst_hw = sorted_hardware[-1]
        improvement = ((best_hw[1]['avg_performance'] - worst_hw[1]['avg_performance']) / 
                      worst_hw[1]['avg_performance'] * 100)
        
        print(f"\n💡 INSIGHTS:")
        print(f"🚀 Best performer: {best_hw[0]} ({best_hw[1]['avg_performance']:.1f} tokens/sec)")
        print(f"📈 Performance range: {improvement:.0f}% improvement from worst to best")
        
        # Reliability insights
        reliable_configs = [c for c in all_configs if c['cv'] < 20]
        print(f"🎯 Reliable configs (CV<20%): {len(reliable_configs)}/{len(all_configs)} ({len(reliable_configs)/len(all_configs)*100:.1f}%)")
        
        if reliable_configs:
            best_reliable = max(reliable_configs, key=lambda x: x['performance'])
            print(f"🏅 Best reliable config: {best_reliable['hardware']} ({best_reliable['performance']:.1f} tokens/sec, CV: {best_reliable['cv']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Quick hardware performance comparison')
    parser.add_argument('--model', type=str, help='Filter by model name (partial match)')
    parser.add_argument('--quantization', type=str, 
                       choices=['4-bit', '8-bit', '3-bit', 'MLX', 'FP16', 'Unknown'], 
                       help='Filter by quantization level')
    args = parser.parse_args()
    
    quick_hardware_comparison(args.model, args.quantization)


if __name__ == "__main__":
    main()
