#!/usr/bin/env python3
"""
Plot MLX Bench Results

Creates visualizations of MLX benchmark performance across different prompt sizes.
Generates line charts and bar charts comparing hardware configurations.
"""

import argparse
import csv
import os
from pathlib import Path
from collections import defaultdict
import statistics

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def load_results(results_dir: str = "results") -> pd.DataFrame:
    """Load all MLX bench result CSV files (both pp and tg)."""
    results_path = Path(results_dir)

    # Find all CSV files (both pp and tg)
    csv_files = []
    csv_files.extend(list(results_path.glob("**/*mlx_bench_pp.csv")))
    csv_files.extend(list(results_path.glob("**/*mlx_bench_tg.csv")))
    # Also look for old combined files for backward compatibility
    csv_files.extend(list(results_path.glob("**/*mlx_bench.csv")))

    if not csv_files:
        print(f"No results found in {results_dir}")
        return pd.DataFrame()

    # Load all CSV files
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["source_file"] = str(csv_file)
            all_data.append(df)
            print(f"Loaded: {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not all_data:
        print("No valid data loaded")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total: {len(combined_df)} results from {len(csv_files)} files\n")

    return combined_df


def create_prompt_size_performance_chart(
    df: pd.DataFrame,
    filter_model: str = None,
    filter_quant: str = None,
    filter_hardware: str = None,
    test_type_filter: str = None,
    show_chart: bool = True,
):
    """
    Create line chart showing tokens/second vs prompt size.

    Args:
        df: DataFrame with benchmark results
        filter_model: Filter by model name
        filter_quant: Filter by quantization
        filter_hardware: Filter by hardware
        test_type_filter: Filter by test type ('pp' or 'tg')
        show_chart: Whether to display the chart
    """
    if df.empty:
        print("No data to plot")
        return

    # Apply filters
    filtered_df = df.copy()

    if filter_model:
        filtered_df = filtered_df[
            filtered_df["model_name"].str.contains(filter_model, case=False, na=False)
        ]

    if filter_quant:
        filtered_df = filtered_df[
            filtered_df["quantization"].str.contains(filter_quant, case=False, na=False)
        ]

    if filter_hardware:
        filtered_df = filtered_df[
            filtered_df["hardware_slug"].str.contains(
                filter_hardware, case=False, na=False
            )
        ]

    # Filter by test type (pp or tg)
    if test_type_filter:
        filtered_df = filtered_df[
            filtered_df["test_type"].str.startswith(test_type_filter, na=False)
        ]

    if filtered_df.empty:
        print(f"No data after filtering (test_type={test_type_filter})")
        return

    # Create hardware + model + quant labels for grouping
    filtered_df["config_label"] = (
        filtered_df["hardware_cpu"]
        + " - "
        + filtered_df["model_name"]
        + " ("
        + filtered_df["quantization"]
        + ")"
    )

    # Group by prompt size and configuration, calculate statistics
    grouped = (
        filtered_df.groupby(["prompt_size", "config_label"])
        .agg({"tokens_per_second": ["mean", "std", "count"]})
        .reset_index()
    )

    grouped.columns = ["prompt_size", "config_label", "mean_tps", "std_tps", "count"]

    # Create figure
    fig = go.Figure()

    # Get unique configurations
    configs = sorted(grouped["config_label"].unique())

    # Generate colors
    colors = px.colors.qualitative.Plotly

    # Add line for each configuration
    for i, config in enumerate(configs):
        config_data = grouped[grouped["config_label"] == config].sort_values(
            "prompt_size"
        )

        color = colors[i % len(colors)]

        # Add line with error bars
        fig.add_trace(
            go.Scatter(
                x=config_data["prompt_size"],
                y=config_data["mean_tps"],
                mode="lines+markers",
                name=config,
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                error_y=dict(
                    type="data",
                    array=config_data["std_tps"],
                    visible=True,
                    color=color,
                    thickness=1.5,
                    width=4,
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    + "Prompt Size: %{x} tokens<br>"
                    + "Performance: %{y:.1f} t/s<br>"
                    + "Std Dev: %{error_y.array:.1f}<br>"
                    + "<extra></extra>"
                ),
            )
        )

    # Build title
    test_type_label = ""
    if test_type_filter == "pp":
        test_type_label = " (Prompt Processing)"
    elif test_type_filter == "tg":
        test_type_label = " (Text Generation)"

    title_parts = [
        f"MLX Bench Performance: Tokens/Second vs Prompt Size{test_type_label}"
    ]
    if filter_model:
        title_parts.append(f"Model: {filter_model}")
    if filter_quant:
        title_parts.append(f"Quantization: {filter_quant}")
    if filter_hardware:
        title_parts.append(f"Hardware: {filter_hardware}")

    title = "<br>".join(title_parts)

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1f77b4")),
        xaxis=dict(
            title="Prompt Size (tokens)",
            type="log",
            gridcolor="lightgray",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Tokens per Second",
            gridcolor="lightgray",
            showgrid=True,
            zeroline=False,
        ),
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#333",
            borderwidth=2,
            font=dict(size=11),
        ),
        plot_bgcolor="white",
        width=1200,
        height=700,
        font=dict(family="Arial, sans-serif", size=12),
    )

    # Save to file
    output_dir = Path("charts")
    output_dir.mkdir(exist_ok=True)

    filter_suffix = ""
    if filter_model:
        filter_suffix += f"_{filter_model.replace(' ', '_')}"
    if filter_quant:
        filter_suffix += f"_{filter_quant}"
    if filter_hardware:
        filter_suffix += f"_{filter_hardware}"
    if test_type_filter:
        filter_suffix += f"_{test_type_filter}"

    output_file = output_dir / f"mlx_bench_prompt_size{filter_suffix}.html"
    fig.write_html(str(output_file))
    print(f"✓ Chart saved: {output_file}")

    if show_chart:
        fig.show()


def create_hardware_comparison_chart(
    df: pd.DataFrame,
    filter_model: str = None,
    filter_quant: str = None,
    test_type_filter: str = None,
    show_chart: bool = True,
):
    """
    Create grouped bar chart comparing different hardware configurations.

    Args:
        df: DataFrame with benchmark results
        filter_model: Filter by model name
        filter_quant: Filter by quantization
        test_type_filter: Filter by test type ('pp' or 'tg')
        show_chart: Whether to display the chart
    """
    if df.empty:
        print("No data to plot")
        return

    # Apply filters
    filtered_df = df.copy()

    if filter_model:
        filtered_df = filtered_df[
            filtered_df["model_name"].str.contains(filter_model, case=False, na=False)
        ]

    if filter_quant:
        filtered_df = filtered_df[
            filtered_df["quantization"].str.contains(filter_quant, case=False, na=False)
        ]

    # Filter by test type (pp or tg)
    if test_type_filter:
        filtered_df = filtered_df[
            filtered_df["test_type"].str.startswith(test_type_filter, na=False)
        ]

    if filtered_df.empty:
        print(f"No data after filtering (test_type={test_type_filter})")
        return

    # Group by hardware and prompt size
    grouped = (
        filtered_df.groupby(["hardware_cpu", "prompt_size"])
        .agg({"tokens_per_second": ["mean", "std"]})
        .reset_index()
    )

    grouped.columns = ["hardware", "prompt_size", "mean_tps", "std_tps"]

    # Create figure
    fig = go.Figure()

    # Get unique hardware configs
    hardware_configs = sorted(grouped["hardware"].unique())
    colors = px.colors.qualitative.Set2

    # Add bars for each hardware
    for i, hw in enumerate(hardware_configs):
        hw_data = grouped[grouped["hardware"] == hw].sort_values("prompt_size")

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Bar(
                name=hw,
                x=hw_data["prompt_size"].astype(str),
                y=hw_data["mean_tps"],
                error_y=dict(type="data", array=hw_data["std_tps"]),
                marker_color=color,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    + "Prompt Size: %{x}<br>"
                    + "Performance: %{y:.1f} t/s<br>"
                    + "<extra></extra>"
                ),
            )
        )

    # Build title
    test_type_label = ""
    if test_type_filter == "pp":
        test_type_label = " (Prompt Processing)"
    elif test_type_filter == "tg":
        test_type_label = " (Text Generation)"

    title_parts = [f"MLX Bench: Hardware Comparison{test_type_label}"]
    if filter_model:
        title_parts.append(f"Model: {filter_model}")
    if filter_quant:
        title_parts.append(f"Quantization: {filter_quant}")

    title = "<br>".join(title_parts)

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1f77b4")),
        xaxis=dict(title="Prompt Size (tokens)", gridcolor="lightgray", showgrid=True),
        yaxis=dict(
            title="Tokens per Second",
            gridcolor="lightgray",
            showgrid=True,
            zeroline=False,
        ),
        barmode="group",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#333",
            borderwidth=2,
            font=dict(size=11),
        ),
        plot_bgcolor="white",
        width=1200,
        height=700,
        font=dict(family="Arial, sans-serif", size=12),
    )

    # Save to file
    output_dir = Path("charts")
    output_dir.mkdir(exist_ok=True)

    filter_suffix = ""
    if filter_model:
        filter_suffix += f"_{filter_model.replace(' ', '_')}"
    if filter_quant:
        filter_suffix += f"_{filter_quant}"
    if test_type_filter:
        filter_suffix += f"_{test_type_filter}"

    output_file = output_dir / f"mlx_bench_hardware_comparison{filter_suffix}.html"
    fig.write_html(str(output_file))
    print(f"✓ Chart saved: {output_file}")

    if show_chart:
        fig.show()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of the benchmark results."""
    if df.empty:
        print("No data to summarize")
        return

    print(f"\n{'='*80}")
    print("📊 SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    # Group by model, quantization, hardware, and prompt size
    grouped = (
        df.groupby(["model_name", "quantization", "hardware_cpu", "prompt_size"])
        .agg({"tokens_per_second": ["mean", "std", "min", "max", "count"]})
        .reset_index()
    )

    grouped.columns = [
        "model",
        "quant",
        "hardware",
        "prompt_size",
        "mean_tps",
        "std_tps",
        "min_tps",
        "max_tps",
        "count",
    ]

    for (model, quant, hardware), group in grouped.groupby(
        ["model", "quant", "hardware"]
    ):
        print(f"\n{model} ({quant}) on {hardware}")
        print(f"{'-'*80}")
        print(
            f"{'Prompt Size':<15} {'Mean t/s':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15} {'Samples':<10}"
        )
        print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")

        for _, row in group.sort_values("prompt_size").iterrows():
            print(
                f"{row['prompt_size']:<15} "
                f"{row['mean_tps']:<15.2f} "
                f"{row['std_tps']:<15.2f} "
                f"{row['min_tps']:<15.2f} "
                f"{row['max_tps']:<15.2f} "
                f"{int(row['count']):<10}"
            )

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot llama-bench results")
    parser.add_argument(
        "--results-dir", default="results", help="Directory containing result CSV files"
    )
    parser.add_argument(
        "--model", help="Filter by model name (case-insensitive substring match)"
    )
    parser.add_argument("--quant", help="Filter by quantization (e.g., Q4_K_M, fp16)")
    parser.add_argument("--hardware", help="Filter by hardware slug")
    parser.add_argument(
        "--test-type",
        choices=["pp", "tg", "both"],
        default="tg",
        help="Test type to plot: pp (prompt processing), tg (text generation), or both (default: tg)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display charts in browser (only save to files)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary statistics, do not generate charts",
    )

    args = parser.parse_args()

    # Load results
    df = load_results(args.results_dir)

    if df.empty:
        print("No results to plot")
        return 1

    # Print summary
    print_summary_statistics(df)

    # Generate charts
    if not args.summary_only:
        show_chart = not args.no_show

        print("\n📈 Generating charts...")

        charts_generated = []

        # Generate charts based on test type argument
        if args.test_type in ["pp", "both"]:
            print("  - Prompt Processing (pp) charts...")
            create_prompt_size_performance_chart(
                df,
                filter_model=args.model,
                filter_quant=args.quant,
                filter_hardware=args.hardware,
                test_type_filter="pp",
                show_chart=False,  # Don't auto-show, we'll list files instead
            )
            charts_generated.append("mlx_bench_prompt_size_pp.html")

            create_hardware_comparison_chart(
                df,
                filter_model=args.model,
                filter_quant=args.quant,
                test_type_filter="pp",
                show_chart=False,
            )
            charts_generated.append("mlx_bench_hardware_comparison_pp.html")

        if args.test_type in ["tg", "both"]:
            print("  - Text Generation (tg) charts...")
            create_prompt_size_performance_chart(
                df,
                filter_model=args.model,
                filter_quant=args.quant,
                filter_hardware=args.hardware,
                test_type_filter="tg",
                show_chart=False,
            )
            charts_generated.append("mlx_bench_prompt_size_tg.html")

            create_hardware_comparison_chart(
                df,
                filter_model=args.model,
                filter_quant=args.quant,
                test_type_filter="tg",
                show_chart=False,
            )
            charts_generated.append("mlx_bench_hardware_comparison_tg.html")

        print("\n✓ All charts generated")
        print(f"\n📁 Charts saved in charts/ directory:")
        for chart in charts_generated:
            print(f"   - charts/{chart}")

        if show_chart:
            print(f"\n💡 Tip: Open the chart files in your browser to view them")

    return 0


if __name__ == "__main__":
    exit(main())
