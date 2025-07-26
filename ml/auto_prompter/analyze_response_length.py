#!/usr/bin/env python3
"""
Response Length Analysis Script

Analyzes existing LLM performance results to understand the impact of 
response length inconsistency on tokens per second measurements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def analyze_response_length_impact(csv_file: str):
    """Analyze how response length affects TPS measurements."""
    
    # Load the data
    df = pd.read_csv(csv_file)
    
    print(f"=== Response Length Analysis for {csv_file} ===\n")
    
    # Basic statistics
    print("Response Token Count Statistics:")
    print(f"  Min: {df['response_token_count'].min()}")
    print(f"  Max: {df['response_token_count'].max()}")
    print(f"  Mean: {df['response_token_count'].mean():.1f}")
    print(f"  Median: {df['response_token_count'].median():.1f}")
    print(f"  Std Dev: {df['response_token_count'].std():.1f}")
    print(f"  Coefficient of Variation: {(df['response_token_count'].std() / df['response_token_count'].mean() * 100):.1f}%\n")
    
    # TPS statistics
    print("Tokens Per Second Statistics:")
    print(f"  Min: {df['tokens_per_second'].min():.2f}")
    print(f"  Max: {df['tokens_per_second'].max():.2f}")
    print(f"  Mean: {df['tokens_per_second'].mean():.2f}")
    print(f"  Median: {df['tokens_per_second'].median():.2f}")
    print(f"  Std Dev: {df['tokens_per_second'].std():.2f}")
    print(f"  Coefficient of Variation: {(df['tokens_per_second'].std() / df['tokens_per_second'].mean() * 100):.1f}%\n")
    
    # Correlation analysis
    correlation = df['response_token_count'].corr(df['tokens_per_second'])
    print(f"Correlation between response length and TPS: {correlation:.3f}")
    
    if abs(correlation) > 0.3:
        print("⚠️  Strong correlation detected - response length significantly affects TPS measurements!")
    elif abs(correlation) > 0.1:
        print("⚠️  Moderate correlation detected - response length somewhat affects TPS measurements")
    else:
        print("✅ Low correlation - response length has minimal impact on TPS measurements")
    
    print()
    
    # Analyze by prompt file
    print("Analysis by Prompt File:")
    by_prompt = df.groupby('filename').agg({
        'response_token_count': ['mean', 'std', 'min', 'max'],
        'tokens_per_second': ['mean', 'std'],
        'prompt_token_count': 'first'
    }).round(2)
    
    by_prompt.columns = ['_'.join(col).strip() for col in by_prompt.columns]
    by_prompt = by_prompt.sort_values('prompt_token_count_first')
    
    print(by_prompt.to_string())
    print()
    
    # Calculate alternative metrics
    print("Alternative Metrics (assuming TTFT = 0.5s):")
    df['assumed_ttft'] = 0.5  # Rough estimate
    df['generation_time'] = df['response_time'] - df['assumed_ttft']
    df['generation_tps'] = df['response_token_count'] / df['generation_time']
    df['time_per_input_token'] = df['assumed_ttft'] / df['prompt_token_count']
    
    print(f"  Mean Generation TPS: {df['generation_tps'].mean():.2f}")
    print(f"  Mean Time per Input Token: {df['time_per_input_token'].mean()*1000:.2f} ms")
    print()
    
    # Identify outliers
    q1 = df['response_token_count'].quantile(0.25)
    q3 = df['response_token_count'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df['response_token_count'] < lower_bound) | 
                  (df['response_token_count'] > upper_bound)]
    
    print(f"Response Length Outliers ({len(outliers)} out of {len(df)}):")
    if len(outliers) > 0:
        print(outliers[['filename', 'iteration', 'response_token_count', 'tokens_per_second']].to_string(index=False))
    else:
        print("  No outliers detected")
    print()
    
    # Recommendations
    print("=== RECOMMENDATIONS ===")
    
    response_cv = df['response_token_count'].std() / df['response_token_count'].mean() * 100
    tps_cv = df['tokens_per_second'].std() / df['tokens_per_second'].mean() * 100
    
    if response_cv > 50:
        print("🔴 HIGH response length variability detected!")
        print("   - Consider using max_tokens parameter to limit response length")
        print("   - Use temperature=0.1 for more consistent responses")
        print("   - Filter results to responses within a specific token range")
    elif response_cv > 25:
        print("🟡 MODERATE response length variability detected")
        print("   - Consider using max_tokens parameter for more consistent benchmarking")
        print("   - Analyze results by response length categories")
    else:
        print("🟢 Response length variability is acceptable")
    
    if tps_cv > 20:
        print("🔴 HIGH TPS variability detected!")
        print("   - Run more iterations to get stable averages")
        print("   - Consider removing outlier measurements")
    elif tps_cv > 10:
        print("🟡 MODERATE TPS variability detected")
        print("   - Consider increasing number of iterations")
    else:
        print("🟢 TPS measurements are reasonably stable")
    
    print(f"\nSuggested max_tokens setting based on your data: {int(df['response_token_count'].quantile(0.75))}")
    
    return df


def create_visualizations(df: pd.DataFrame, output_prefix: str):
    """Create visualizations for the analysis."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM Performance Analysis: Response Length Impact', fontsize=16)
    
    # 1. Response length distribution
    axes[0, 0].hist(df['response_token_count'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Response Token Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Response Lengths')
    axes[0, 0].axvline(df['response_token_count'].mean(), color='red', linestyle='--', label=f'Mean: {df["response_token_count"].mean():.0f}')
    axes[0, 0].legend()
    
    # 2. TPS vs Response Length scatter
    axes[0, 1].scatter(df['response_token_count'], df['tokens_per_second'], alpha=0.6)
    axes[0, 1].set_xlabel('Response Token Count')
    axes[0, 1].set_ylabel('Tokens Per Second')
    axes[0, 1].set_title('TPS vs Response Length')
    
    # Add trend line
    z = np.polyfit(df['response_token_count'], df['tokens_per_second'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df['response_token_count'], p(df['response_token_count']), "r--", alpha=0.8)
    
    # 3. TPS distribution
    axes[1, 0].hist(df['tokens_per_second'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Tokens Per Second')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of TPS Measurements')
    axes[1, 0].axvline(df['tokens_per_second'].mean(), color='red', linestyle='--', label=f'Mean: {df["tokens_per_second"].mean():.1f}')
    axes[1, 0].legend()
    
    # 4. Box plot by prompt file (top 8 files by frequency)
    top_prompts = df['filename'].value_counts().head(8).index
    df_subset = df[df['filename'].isin(top_prompts)]
    
    sns.boxplot(data=df_subset, x='filename', y='tokens_per_second', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Prompt File')
    axes[1, 1].set_ylabel('Tokens Per Second')
    axes[1, 1].set_title('TPS by Prompt File (Top 8)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Visualizations saved as {output_prefix}_analysis.png")
    
    # Create a separate correlation plot
    plt.figure(figsize=(10, 8))
    
    # Create correlation matrix
    corr_cols = ['prompt_token_count', 'response_token_count', 'response_time', 'tokens_per_second']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Performance Metrics')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_correlation.png', dpi=300, bbox_inches='tight')
    print(f"Correlation matrix saved as {output_prefix}_correlation.png")


def main():
    parser = argparse.ArgumentParser(description='Analyze LLM performance results for response length impact')
    parser.add_argument('csv_file', help='Path to the CSV results file')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    parser.add_argument('--output-prefix', default='llm_analysis', help='Prefix for output files')
    
    args = parser.parse_args()
    
    try:
        df = analyze_response_length_impact(args.csv_file)
        
        if args.plot:
            create_visualizations(df, args.output_prefix)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
