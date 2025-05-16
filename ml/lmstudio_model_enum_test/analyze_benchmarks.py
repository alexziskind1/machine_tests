import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

def load_and_preprocess_data(path_pattern):
    all_files = glob.glob(path_pattern)
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Standardize column name for model
            if 'model' in df.columns and 'model_name' not in df.columns:
                df.rename(columns={'model': 'model_name'}, inplace=True)
            
            # Extract gpu_memory_gb from filename if not in columns or if it needs override for clarity
            # This part is a bit heuristic and might need adjustment based on exact filename patterns
            # For now, we rely on the column in the CSV.
            # If 'gpu_memory_gb' column is 'auto', try to keep it as string or convert to a specific numeric marker if needed
            
            df_list.append(df)
        except Exception as e:
            print(f"Error reading or processing file {f}: {e}")
            
    if not df_list:
        print("No data loaded. Exiting.")
        return pd.DataFrame()
        
    combined_df = pd.concat(df_list, ignore_index=True)

    # Data type conversions
    numeric_cols = ['params_b', 'size_gb', 'tokens_per_second', 'total_tokens', 'time_to_first_token', 'generation_time']
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    # Convert gpu_memory_gb, keeping 'auto' as a special category if needed
    # For numerical plotting, 'auto' will be handled or filtered.
    if 'gpu_memory_gb' in combined_df.columns:
        combined_df['gpu_memory_gb_numeric'] = pd.to_numeric(combined_df['gpu_memory_gb'], errors='coerce')
        # Keep original gpu_memory_gb for categorical use
        combined_df['gpu_memory_gb'] = combined_df['gpu_memory_gb'].astype(str)


    # Clean model names (remove potential path-like structures if any)
    if 'model_name' in combined_df.columns:
        combined_df['model_name'] = combined_df['model_name'].apply(lambda x: x.split('/')[-1] if isinstance(x, str) else x)
        
    return combined_df

def plot_performance_by_model_gpu(df, output_dir='.'):
    if df.empty or 'host' not in df.columns or 'tokens_per_second' not in df.columns or 'model_name' not in df.columns or 'gpu_memory_gb' not in df.columns:
        print("Plot 1: Missing required columns (host, tokens_per_second, model_name, gpu_memory_gb). Skipping.")
        return

    for host, group_df in df.groupby('host'):
        plt.figure(figsize=(15, 10))
        sns.barplot(data=group_df, x='model_name', y='tokens_per_second', hue='gpu_memory_gb', dodge=True)
        plt.title(f'Tokens/sec by Model and GPU Memory on {host}')
        plt.xlabel('Model Name')
        plt.ylabel('Tokens per Second')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{host}_perf_by_model_gpu.png'))
        plt.close()
    print("Plot 1: Performance by Model and GPU Memory charts generated.")

def plot_performance_vs_model_size(df, output_dir='.'):
    if df.empty or 'host' not in df.columns or 'tokens_per_second' not in df.columns or 'params_b' not in df.columns:
        print("Plot 2: Missing required columns (host, tokens_per_second, params_b). Skipping.")
        return
        
    for host, group_df in df.groupby('host'):
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=group_df, x='params_b', y='tokens_per_second', hue='architecture', size='size_gb', style='format', alpha=0.7, sizes=(50, 500))
        plt.title(f'Tokens/sec vs. Model Size (params_b) on {host}')
        plt.xlabel('Model Parameters (Billions)')
        plt.ylabel('Tokens per Second')
        plt.xscale('log') # Model params often vary over orders of magnitude
        plt.yscale('log') # Tokens/sec can also vary widely
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{host}_perf_vs_model_size.png'))
        plt.close()
    print("Plot 2: Performance vs. Model Size charts generated.")

def plot_gpu_memory_impact(df, output_dir='.'):
    if df.empty or 'host' not in df.columns or 'tokens_per_second' not in df.columns or 'model_name' not in df.columns or 'gpu_memory_gb_numeric' not in df.columns:
        print("Plot 3: Missing required columns for GPU memory impact. Skipping.")
        return

    # Filter out NaN gpu_memory_gb_numeric values (which would include 'auto')
    plot_df = df.dropna(subset=['gpu_memory_gb_numeric'])
    if plot_df.empty:
        print("Plot 3: No numeric GPU memory data available after filtering. Skipping.")
        return

    for host, host_df in plot_df.groupby('host'):
        # Find models that have data for multiple GPU memory configurations
        model_counts = host_df.groupby('model_name')['gpu_memory_gb_numeric'].nunique()
        models_to_plot = model_counts[model_counts > 1].index
        
        if models_to_plot.empty:
            print(f"Plot 3: No models with multiple GPU memory points for host {host}. Skipping.")
            continue
            
        subset_df = host_df[host_df['model_name'].isin(models_to_plot)]
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=subset_df, x='gpu_memory_gb_numeric', y='tokens_per_second', hue='model_name', marker='o')
        plt.title(f'Impact of GPU Memory on Performance on {host}')
        plt.xlabel('GPU Memory (GB)')
        plt.ylabel('Tokens per Second')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{host}_gpu_memory_impact.png'))
        plt.close()
    print("Plot 3: GPU Memory Impact charts generated.")

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path_pattern = os.path.join(base_path, 'benchmark_*.csv')
    
    print(f"Looking for CSV files in: {data_path_pattern}")
    combined_data = load_and_preprocess_data(data_path_pattern)
    
    if combined_data.empty:
        return

    # Create an output directory for plots if it doesn't exist
    plot_output_dir = os.path.join(base_path, 'benchmark_plots')
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)
    print(f"Saving plots to: {plot_output_dir}")

    plot_performance_by_model_gpu(combined_data, plot_output_dir)
    plot_performance_vs_model_size(combined_data, plot_output_dir)
    plot_gpu_memory_impact(combined_data, plot_output_dir)
    
    print("Analysis complete. Plots saved.")

if __name__ == '__main__':
    main()
