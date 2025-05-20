import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import plotly.io as pio
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
    numeric_cols = ['params_b', 'size_gb', 'tokens_per_second', 'total_tokens', 'time_to_first_token', 'generation_time'
                    ]
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
        fig = px.bar(group_df, x='model_name', y='tokens_per_second', color='gpu_memory_gb',
                     barmode='group',
                     title=f'Tokens/sec by Model and GPU Memory on {host}',
                     labels={'tokens_per_second': 'Tokens per Second', 'model_name': 'Model Name', 'gpu_memory_gb': 'GPU Memory (GB)'})
        fig.update_xaxes(categoryorder='total descending') # Sort by value
        fig.write_html(os.path.join(output_dir, f'{host}_perf_by_model_gpu.html'))
    print("Plot 1: Performance by Model and GPU Memory charts generated (HTML).")

def plot_performance_vs_model_size(df, output_dir='.'):
    if df.empty or 'host' not in df.columns or 'tokens_per_second' not in df.columns or 'params_b' not in df.columns or 'gpu_memory_gb_numeric' not in df.columns:
        print("Plot 2: Missing required columns (host, tokens_per_second, params_b, gpu_memory_gb_numeric). Skipping.")
        return
        
    for host, group_df in df.groupby('host'):
        # Drop rows where 'gpu_memory_gb_numeric' is NaN before plotting
        plot_df = group_df.dropna(subset=['gpu_memory_gb_numeric'])
        if plot_df.empty:
            print(f"Plot 2: No valid data after dropping NaN gpu_memory_gb_numeric for host {host}. Skipping.")
            continue

        fig = px.scatter(plot_df, x='params_b', y='tokens_per_second', 
                         color='model_name', size='gpu_memory_gb_numeric', symbol='format',
                         size_max=20, # Increased marker size further
                         hover_data=['model_name', 'gpu_memory_gb_numeric', 'size_gb', 'architecture', 'format'],
                         title=f'Tokens/sec vs. Model Size (params_b) on {host} (Color: Model, Size: GPU RAM)',
                         labels={'params_b': 'Model Parameters (Billions)', 'tokens_per_second': 'Tokens per Second', 'gpu_memory_gb_numeric': 'GPU Memory (GB)', 'model_name': 'Model Name'})
        fig.update_layout(xaxis_type="log", yaxis_type="log")
        fig.write_html(os.path.join(output_dir, f'{host}_perf_vs_model_size_details.html'))
    print("Plot 2: Performance vs. Model Size charts (with GPU memory and model details) generated (HTML).")

def plot_gpu_memory_impact(df, output_dir='.'):
    if df.empty or 'host' not in df.columns or 'tokens_per_second' not in df.columns or 'model_name' not in df.columns or 'gpu_memory_gb_numeric' not in df.columns:
        print("Plot 3: Missing required columns for GPU memory impact. Skipping.")
        return

    plot_df = df.dropna(subset=['gpu_memory_gb_numeric'])
    if plot_df.empty:
        print("Plot 3: No numeric GPU memory data available after filtering. Skipping.")
        return

    for host, host_df in plot_df.groupby('host'):
        model_counts = host_df.groupby('model_name')['gpu_memory_gb_numeric'].nunique()
        models_to_plot = model_counts[model_counts > 1].index
        
        if models_to_plot.empty:
            print(f"Plot 3: No models with multiple GPU memory points for host {host}. Skipping.")
            continue
            
        subset_df = host_df[host_df['model_name'].isin(models_to_plot)]
        # Sort by GPU memory for correct line plotting order
        subset_df = subset_df.sort_values(by=['model_name', 'gpu_memory_gb_numeric'])
        
        fig = px.line(subset_df, x='gpu_memory_gb_numeric', y='tokens_per_second', color='model_name', markers=True,
                      title=f'Impact of GPU Memory on Performance on {host}',
                      labels={'gpu_memory_gb_numeric': 'GPU Memory (GB)', 'tokens_per_second': 'Tokens per Second', 'model_name': 'Model Name'})
        
        fig.update_traces(marker=dict(size=20)) # Increase marker size for all traces

        if host == 'az_flow128':
            fig.update_traces(mode='markers') # For az_flow128, show only markers, no lines
            
        fig.write_html(os.path.join(output_dir, f'{host}_gpu_memory_impact.html'))
    print("Plot 3: GPU Memory Impact charts generated (HTML).")

def plot_3d_model_perf_gpu_mem_per_host(df, output_dir='.'):
    """Generates a 3D scatter plot per host: Model Size (params_b) vs GPU Memory vs Tokens/sec."""
    required_cols = ['host', 'params_b', 'gpu_memory_gb_numeric', 'tokens_per_second', 'model_name']
    if df.empty or not all(col in df.columns for col in required_cols):
        print(f"Plot 5 (3D Details): Missing one or more required columns from {required_cols}. Skipping.")
        return

    for host, group_df in df.groupby('host'):
        plot_df = group_df.dropna(subset=['params_b', 'gpu_memory_gb_numeric', 'tokens_per_second'])
        
        if plot_df.empty:
            print(f"Plot 5 (3D Details): No valid data for host {host} after dropping NaNs. Skipping.")
            continue

        # Determine if log scale can be used (values must be > 0)
        log_x_axis = (plot_df['params_b'].min() > 0) if not plot_df['params_b'].empty else False
        log_z_axis = (plot_df['tokens_per_second'].min() > 0) if not plot_df['tokens_per_second'].empty else False

        title = f'3D Perf. Details: {host} (X:Model Size, Y:GPU Mem, Z:TPS)'

        # Prepare hover data, ensuring columns exist
        hover_data_cols = ['params_b', 'gpu_memory_gb_numeric', 'tokens_per_second', 'gpu_memory_gb', 'size_gb', 'architecture']
        final_hover_data = [col for col in hover_data_cols if col in plot_df.columns]
        if 'format' in plot_df.columns and 'format' not in final_hover_data:
             final_hover_data.append('format')

        fig = px.scatter_3d(plot_df,
                            x='params_b',
                            y='gpu_memory_gb_numeric',
                            z='tokens_per_second',
                            color='model_name',
                            symbol='format' if 'format' in plot_df.columns else None,
                            log_x=log_x_axis,
                            log_z=log_z_axis,
                            hover_name='model_name', # Shows model name prominently on hover
                            hover_data=final_hover_data,
                            labels={ 
                                'params_b': 'Model Params (B)',
                                'gpu_memory_gb_numeric': 'GPU Mem (GB)', 
                                'tokens_per_second': 'Tokens/Sec',
                                'model_name': 'Model',
                                'format': 'Format',
                                'gpu_memory_gb': 'GPU Setting (Original)',
                                'size_gb': 'File Size (GB)',
                                'architecture': 'Architecture'
                            },
                            title=title)
        
        fig.update_layout(scene = dict(
                            xaxis_title='Model Params (B)' + (' (log)' if log_x_axis else ''),
                            yaxis_title='GPU Mem (GB)',
                            zaxis_title='Tokens/Sec' + (' (log)' if log_z_axis else '')),
                          margin=dict(l=0, r=0, b=0, t=50) # Adjust top margin for title
                         )

        output_filename = f'{host}_3d_perf_details.html'
        try:
            fig.write_html(os.path.join(output_dir, output_filename))
        except Exception as e:
            print(f"Plot 5 (3D Details): Error writing HTML for host {host}: {e}")
            
    print("Plot 5 (3D Details): Charts generated (HTML).")

def plot_all_models_performance_by_host_3d(df, output_dir='.'):
    """Generates a 3D scatter plot for all models: Model vs GPU Memory vs Tokens/sec, colored by Host."""
    required_cols = ['host', 'model_name', 'gpu_memory_gb_numeric', 'tokens_per_second']
    if df.empty or not all(col in df.columns for col in required_cols):
        print(f"Plot 6 (All Models 3D): Missing one or more required columns from {required_cols}. Skipping.")
        return

    # No specific model filtering, use all data after basic NaN drop
    plot_df = df.dropna(subset=['host', 'model_name', 'gpu_memory_gb_numeric', 'tokens_per_second'])
    if plot_df.empty:
        print("Plot 6 (All Models 3D): No valid data after dropping NaNs. Skipping.")
        return

    unique_models_found = sorted(plot_df['model_name'].unique().tolist())
    
    title = f'3D Perf: All Models - Model vs GPU Mem vs TPS (Color: Host)'

    # Determine if log scale can be used for Z-axis
    log_z_axis = (plot_df['tokens_per_second'].min() > 0) if not plot_df['tokens_per_second'].empty else False

    # Prepare hover data, ensuring columns exist
    hover_data_cols = ['model_name', 'host', 'gpu_memory_gb_numeric', 'tokens_per_second', 'gpu_memory_gb', 'params_b', 'size_gb', 'architecture']
    final_hover_data = [col for col in hover_data_cols if col in plot_df.columns]
    if 'format' in plot_df.columns and 'format' not in final_hover_data:
         final_hover_data.append('format')

    fig = px.scatter_3d(
        plot_df,
        x='model_name',
        y='gpu_memory_gb_numeric',
        z='tokens_per_second',
        color='host',
        symbol='format' if 'format' in plot_df.columns else None,
        log_z=log_z_axis,
        hover_name='host',
        hover_data=final_hover_data,
        labels={ 
            'model_name': 'Model',
            'gpu_memory_gb_numeric': 'GPU Mem (GB)', 
            'tokens_per_second': 'Tokens/Sec',
            'host': 'Host',
            'format': 'Format',
            'gpu_memory_gb': 'GPU Setting (Original)',
            'params_b': 'Params (B)',
            'size_gb': 'File Size (GB)',
            'architecture': 'Architecture'
        },
        title=title,
        category_orders={
            "model_name": unique_models_found, # Order for X-axis based on all unique models
            "host": sorted(plot_df['host'].unique().tolist()) # Order for color
        }
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='Model',
            yaxis_title='GPU Mem (GB)',
            zaxis_title='Tokens/Sec' + (' (log)' if log_z_axis else '')
        ),
        legend_title_text='Host',
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    output_filename = 'all_models_by_host_perf_3d.html'
    try:
        fig.write_html(os.path.join(output_dir, output_filename))
        print(f"Plot 6 (All Models 3D): Chart for all models generated as {output_filename}.")
    except Exception as e:
        print(f"Plot 6 (All Models 3D): Error writing HTML: {e}")

def plot_selected_models_across_hosts(df, output_dir='.'):
    if df.empty:
        print("Plot 4: Dataframe is empty. Skipping selected models comparison.")
        return

    all_unique_hosts = df['host'].nunique()
    if all_unique_hosts == 0:
        print("Plot 4: No hosts found in data. Skipping selected models comparison.")
        return

    model_host_counts = df.groupby('model_name')['host'].nunique()
    common_model_names = model_host_counts[model_host_counts == all_unique_hosts].index.tolist()

    if not common_model_names:
        print("Plot 4: No models found that are common to all hosts. Skipping this chart.")
        return

    # Get params_b for common models, drop duplicates and NaNs
    common_models_df = df[df['model_name'].isin(common_model_names)][['model_name', 'params_b']].drop_duplicates()
    common_models_df = common_models_df.dropna(subset=['params_b'])
    
    if common_models_df['model_name'].nunique() < 2:
        print(f"Plot 4: Need at least two distinct common models with valid parameter counts to compare. Found {common_models_df['model_name'].nunique()}. Skipping this chart.")
        return
        
    sorted_common_models = common_models_df.sort_values(by='params_b')
    
    small_model_name = sorted_common_models['model_name'].iloc[0]
    large_model_name = sorted_common_models['model_name'].iloc[-1]

    if small_model_name == large_model_name: # Should be caught by nunique < 2, but as a safeguard
        print(f"Plot 4: Smallest and largest common models are the same ({small_model_name}). Skipping this chart.")
        return

    print(f"Plot 4: Selected small model: {small_model_name}, Selected large model: {large_model_name} for host comparison.")

    # Filter for these two models
    models_to_plot_df = df[df['model_name'].isin([small_model_name, large_model_name])]

    # For each host and model, get the entry with max tokens_per_second
    # This handles multiple GPU memory settings by picking the best performance for this comparison
    idx = models_to_plot_df.groupby(['host', 'model_name'])['tokens_per_second'].idxmax()
    best_perf_df = models_to_plot_df.loc[idx]

    if best_perf_df.empty:
        print("Plot 4: No data available for the selected common models after filtering for best performance. Skipping.")
        return

    # Filter out rows where gpu_memory_gb_numeric is NaN, as it's used for size
    plot_df = best_perf_df.dropna(subset=['gpu_memory_gb_numeric'])
    if plot_df.shape[0] < best_perf_df.shape[0]:
        print(f"Plot 4: Note - Dropped {best_perf_df.shape[0] - plot_df.shape[0]} data points that had non-numeric 'gpu_memory_gb' values, as they cannot be used for sizing in the 3D plot.")
    
    if plot_df.empty:
        print("Plot 4: No data with numeric 'gpu_memory_gb' values available for the selected common models after filtering. Skipping 3D chart.")
        return

    # Ensure both selected models are still present after filtering
    if not (small_model_name in plot_df['model_name'].unique() and 
            large_model_name in plot_df['model_name'].unique()):
        print(f"Plot 4: After filtering for numeric GPU memory, one or both selected models ({small_model_name}, {large_model_name}) have no data. Skipping 3D chart.")
        return

    title = f'3D Perf: {small_model_name} vs {large_model_name} (Size: GPU Mem)'
    
    model_colors = {
        small_model_name: px.colors.qualitative.Plotly[0],
        large_model_name: px.colors.qualitative.Plotly[1]
    }

    # Ensure consistent ordering for axes and legend, based on the filtered plot_df
    host_order = sorted(plot_df['host'].unique().tolist())
    model_order = [small_model_name, large_model_name] # Retain original order for consistency

    fig = px.scatter_3d(plot_df,
                        x='host',
                        y='model_name',
                        z='tokens_per_second',
                        color='model_name',
                        size='gpu_memory_gb_numeric', # Use numeric GPU memory for size
                        size_max=60,                 # Adjust max marker size as needed
                        color_discrete_map=model_colors,
                        custom_data=['gpu_memory_gb'], # Original gpu_memory_gb (string) for hover
                        category_orders={"host": host_order, "model_name": model_order},
                        labels={
                            'tokens_per_second': 'TPS (Peak)', 
                            'host': 'Host', 
                            'model_name': 'Model',
                            'gpu_memory_gb_numeric': 'GPU Mem (GB)' # Label for size legend
                        },
                        title=title)

    fig.update_traces( # Removed marker=dict(size=8) as size is now data-driven
                      hovertemplate=(
                          "<b>Host:</b> %{x}<br>"
                          "<b>Model:</b> %{y}<br>"
                          "<b>TPS:</b> %{z:.2f}<br>"
                          "<b>GPU Memory:</b> %{customdata[0]}<br>" # Displays the string version from custom_data
                          "<extra></extra>"
                      ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Host',
            yaxis_title='Model Name',
            zaxis_title='Tokens per Second (Peak)',
            xaxis=dict(tickangle=-45, categoryorder='array', categoryarray=host_order),
            yaxis=dict(categoryorder='array', categoryarray=model_order)
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40) # Adjust margins to fit title
    )
    
    output_filename = 'all_hosts_selected_models_comparison_3d.html'
    fig.write_html(os.path.join(output_dir, output_filename))
    print(f"Plot 4: 3D Chart comparing hosts for models '{small_model_name}' and '{large_model_name}' generated as {output_filename}.")

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
    plot_selected_models_across_hosts(combined_data, plot_output_dir) 
    plot_3d_model_perf_gpu_mem_per_host(combined_data, plot_output_dir) # New plot function call
    plot_all_models_performance_by_host_3d(combined_data, plot_output_dir) # Updated function call
    
    print("Analysis complete. Plots saved.")

if __name__ == '__main__':
    main()
