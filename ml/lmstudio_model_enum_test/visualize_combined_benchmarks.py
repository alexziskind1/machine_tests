import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import numpy as np

def load_data(file_path):
    """Loads and preprocesses the combined benchmark data."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    # Standardize column name for model (should be done by combine script, but good to double check)
    if 'model' in df.columns and 'model_name' not in df.columns:
        df.rename(columns={'model': 'model_name'}, inplace=True)
    
    # Clean model names
    if 'model_name' in df.columns:
        df['model_name'] = df['model_name'].astype(str).apply(lambda x: x.split('/')[-1] if pd.notnull(x) else x)

    # Data type conversions
    numeric_cols = ['params_b', 'size_gb', 'tokens_per_second', 'total_tokens', 'time_to_first_token', 'generation_time']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle gpu_memory_gb
    if 'gpu_memory_gb' in df.columns:
        df['gpu_memory_gb_str'] = df['gpu_memory_gb'].astype(str) # Keep original for categorical
        df['gpu_memory_gb_numeric'] = pd.to_numeric(df['gpu_memory_gb'], errors='coerce') # 'auto' becomes NaN
    else:
        df['gpu_memory_gb_str'] = 'N/A'
        df['gpu_memory_gb_numeric'] = np.nan
        
    # Ensure key categorical columns are strings
    for col in ['host', 'architecture', 'format', 'system', 'model_name']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df

def plot_avg_performance_by_host_arch(df, output_dir):
    """Plots average tokens_per_second by host and architecture using Plotly."""
    if df.empty or not all(col in df.columns for col in ['host', 'architecture', 'tokens_per_second']):
        print("Plot 1: Missing required columns (host, architecture, tokens_per_second). Skipping.")
        return

    agg_df = df.dropna(subset=['tokens_per_second']).groupby(['host', 'architecture'], as_index=False)['tokens_per_second'].mean()
    
    fig = px.bar(agg_df, x='host', y='tokens_per_second', color='architecture', barmode='group',
                 title='Average Tokens/sec by Host and Model Architecture',
                 labels={'tokens_per_second': 'Average Tokens per Second', 'host': 'Host'})
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html(os.path.join(output_dir, 'avg_perf_by_host_arch.html'))
    print("Plot 1: Average Performance by Host and Architecture chart generated (HTML).")

def plot_detailed_model_perf_with_gpumem(df, models_to_compare, output_dir):
    """Plots tokens_per_second for specific models across hosts, detailed by gpu_memory_gb_str using Plotly."""
    if df.empty or not all(col in df.columns for col in ['host', 'model_name', 'tokens_per_second', 'gpu_memory_gb_str']):
        print("Plot 2 (Detailed Model Perf): Missing required columns. Skipping.")
        return

    # Filter for the models we want to compare
    subset_df = df[df['model_name'].isin(models_to_compare)].copy()
    subset_df.dropna(subset=['tokens_per_second', 'gpu_memory_gb_str'], inplace=True)

    if subset_df.empty:
        print(f"Plot 2 (Detailed Model Perf): No data found for specified models: {models_to_compare} with GPU memory info. Skipping.")
        return

    # Sort by gpu_memory_gb_numeric for consistent legend order, handling 'auto' by placing it perhaps first or last
    # Create a numeric version for sorting, 'auto' can be -1 or a large number depending on desired position
    subset_df['gpu_mem_sort_val'] = pd.to_numeric(subset_df['gpu_memory_gb_str'], errors='coerce').fillna(-1) # 'auto' becomes -1
    subset_df.sort_values(by=['host', 'model_name', 'gpu_mem_sort_val'], inplace=True)

    fig = px.bar(subset_df, 
                 x='model_name', 
                 y='tokens_per_second', 
                 color='gpu_memory_gb_str', 
                 barmode='group',
                 facet_col='host', 
                 facet_col_wrap=2, # Adjust as needed
                 title='Tokens/sec by Model, Host, and GPU Memory Configuration',
                 labels={'tokens_per_second': 'Tokens per Second', 
                         'model_name': 'Model Name', 
                         'gpu_memory_gb_str': 'GPU Memory (GB / Setting)'},
                 category_orders={'gpu_memory_gb_str': sorted(subset_df['gpu_memory_gb_str'].unique(), key=lambda x: float(x) if x.replace('.','',1).isdigit() else -1)}
                )
    fig.update_layout(xaxis_tickangle=-45)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean facet titles
    fig.write_html(os.path.join(output_dir, 'detailed_model_perf_with_gpumem.html'))
    print("Plot 2: Detailed Model Performance with GPU Memory chart generated (HTML).")

def plot_perf_vs_model_size_all_hosts(df, output_dir):
    """Plots tokens_per_second vs. params_b, colored by host, styled by architecture using Plotly."""
    if df.empty or not all(col in df.columns for col in ['params_b', 'tokens_per_second', 'host', 'architecture']):
        print("Plot 3: Missing required columns for performance vs model size. Skipping.")
        return
    
    plot_df = df.dropna(subset=['params_b', 'tokens_per_second']) 
    if plot_df.empty:
        print("Plot 3: No valid data after dropping NaNs for params_b or tokens_per_second. Skipping.")
        return

    fig = px.scatter(plot_df, x='params_b', y='tokens_per_second', color='host', symbol='architecture',
                     title='Tokens/sec vs. Model Size (params_b) - All Hosts',
                     labels={'params_b': 'Model Parameters (Billions)', 'tokens_per_second': 'Tokens per Second'},
                     log_x=True, log_y=True,
                     hover_data=['model_name', 'size_gb', 'gpu_memory_gb_str'])
    fig.update_layout(legend_title_text='Host / Architecture')
    fig.write_html(os.path.join(output_dir, 'perf_vs_model_size_all_hosts.html'))
    print("Plot 3: Performance vs. Model Size (All Hosts) chart generated (HTML).")

def plot_ttft_across_hosts(df, models_to_compare, output_dir):
    """Plots time_to_first_token for specific models across hosts using Plotly."""
    if df.empty or not all(col in df.columns for col in ['host', 'model_name', 'time_to_first_token', 'tokens_per_second']):
        print("Plot 4: Missing required columns for TTFT comparison. Skipping.")
        return

    best_perf_df = df.loc[df.dropna(subset=['tokens_per_second', 'time_to_first_token']).groupby(['host', 'model_name'])['tokens_per_second'].idxmax()]
    subset_df = best_perf_df[best_perf_df['model_name'].isin(models_to_compare)]

    if subset_df.empty:
        print(f"Plot 4: No data found for TTFT for specified models: {models_to_compare}. Skipping.")
        return

    fig = px.bar(subset_df, x='model_name', y='time_to_first_token', color='host', barmode='group',
                 title='Time to First Token (sec) for Selected Models Across Hosts (Best VRAM Setting)',
                 labels={'time_to_first_token': 'Time to First Token (seconds)', 'model_name': 'Model Name'})
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html(os.path.join(output_dir, 'ttft_across_hosts.html'))
    print("Plot 4: Time to First Token Across Hosts chart generated (HTML).")

def plot_tokens_vs_gpumem_faceted(df, output_dir, models_to_highlight=None):
    """Plots tokens_per_second vs. numeric GPU Memory, faceted by host, colored by model."""
    if df.empty or not all(col in df.columns for col in ['host', 'model_name', 'tokens_per_second', 'gpu_memory_gb_numeric', 'params_b']):
        print("Plot 5 (Tokens vs GPU Mem): Missing required columns. Skipping.")
        return

    plot_df = df.dropna(subset=['tokens_per_second', 'gpu_memory_gb_numeric'])
    if models_to_highlight:
        plot_df = plot_df[plot_df['model_name'].isin(models_to_highlight)]

    if plot_df.empty:
        print("Plot 5 (Tokens vs GPU Mem): No data for plotting after filtering. Skipping.")
        return
    
    # Sort by numeric GPU memory for potentially cleaner lines if using line+markers
    plot_df = plot_df.sort_values(by='gpu_memory_gb_numeric')

    fig = px.scatter(plot_df, 
                     x='gpu_memory_gb_numeric', 
                     y='tokens_per_second',
                     color='model_name', 
                     facet_col='host', 
                     facet_col_wrap=2, # Adjust as needed
                     size='params_b', # Optional: size points by model parameters
                     log_y=True,
                     title='Tokens/sec vs. GPU Memory Allocation by Host & Model',
                     labels={'gpu_memory_gb_numeric': 'GPU Memory Allocated (GB)',
                             'tokens_per_second': 'Tokens per Second (log scale)',
                             'model_name': 'Model'},
                     hover_data=['cpu', 'size_gb', 'architecture', 'format']
                    )
    fig.update_layout(xaxis_title='GPU Memory Allocated (GB)')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean facet titles
    fig.write_html(os.path.join(output_dir, 'tokens_vs_gpumem_faceted.html'))
    print("Plot 5: Tokens/sec vs. GPU Memory (Faceted) chart generated (HTML).")


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_path, 'combined_benchmark_results.csv')
    
    combined_data = load_data(input_csv)
    
    if combined_data.empty:
        print("Exiting due to data loading issues.")
        return

    plot_output_dir = os.path.join(base_path, 'combined_benchmark_plots')
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)
    print(f"Saving plots to: {plot_output_dir}")

    # Define some common models for detailed plots
    # These should be models present in your combined data across multiple hosts
    # You might need to inspect your 'combined_benchmark_results.csv' to choose these
    common_models = [
        'Llama-3.3-70B-Instruct-Q4_K_M',
        'gemma-3-12B-it-QAT-Q4_0',
        'gemma-3-4B-it-QAT-Q4_0',
        'gemma-3-1B-it-QAT-Q4_0',
        'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M' # Example, check if present
    ]

    plot_avg_performance_by_host_arch(combined_data, plot_output_dir)
    # plot_specific_model_performance(combined_data, common_models, plot_output_dir) # Old plot
    plot_detailed_model_perf_with_gpumem(combined_data, common_models, plot_output_dir) # New detailed plot
    plot_perf_vs_model_size_all_hosts(combined_data, plot_output_dir)
    plot_ttft_across_hosts(combined_data, common_models, plot_output_dir)
    plot_tokens_vs_gpumem_faceted(combined_data, plot_output_dir, models_to_highlight=common_models) # New plot
    
    print("Combined data visualization complete. Plots saved.")

if __name__ == '__main__':
    main()
