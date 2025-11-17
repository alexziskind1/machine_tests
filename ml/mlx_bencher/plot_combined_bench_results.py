#!/usr/bin/env python3
"""
Plot Combined Benchmark Results

Combines results from both llama_bencher and mlx_bencher for side-by-side comparison.
Creates visualizations showing performance across different implementations.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def load_results(results_dirs: list) -> pd.DataFrame:
    """Load all benchmark result CSV files from multiple directories."""
    all_data = []

    for results_dir in results_dirs:
        results_path = Path(results_dir)

        if not results_path.exists():
            print(f"Warning: Directory not found: {results_dir}")
            continue

        # Find all CSV files (both pp and tg)
        csv_files = []
        csv_files.extend(list(results_path.glob("**/*llama_bench_pp.csv")))
        csv_files.extend(list(results_path.glob("**/*llama_bench_tg.csv")))
        csv_files.extend(list(results_path.glob("**/*mlx_bench_pp.csv")))
        csv_files.extend(list(results_path.glob("**/*mlx_bench_tg.csv")))
        # Also look for old combined files for backward compatibility
        csv_files.extend(list(results_path.glob("**/*llama_bench.csv")))
        csv_files.extend(list(results_path.glob("**/*mlx_bench.csv")))

        # Load all CSV files
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df["source_file"] = str(csv_file)

                # Add backend identifier based on file path
                if "llama_bench" in csv_file.name or "llama_bencher" in str(csv_file):
                    df["bench_backend"] = "llama.cpp"
                elif "mlx_bench" in csv_file.name or "mlx_bencher" in str(csv_file):
                    df["bench_backend"] = "MLX"
                else:
                    df["bench_backend"] = "Unknown"

                all_data.append(df)
                print(
                    f"Loaded: {csv_file.name} ({len(df)} rows) - {df['bench_backend'].iloc[0]}"
                )
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

    if not all_data:
        print("No valid data loaded")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total: {len(combined_df)} results from {len(all_data)} files\n")

    return combined_df


def create_prompt_size_performance_chart(
    df: pd.DataFrame,
    filter_model: str = None,
    filter_quant: str = None,
    test_type_filter: str = None,
    show_chart: bool = True,
):
    """Create bar chart showing tokens/second vs prompt size for both backends."""
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

    # Get model names for title
    model_names = filtered_df["model_name"].unique()
    model_name_for_title = " / ".join(sorted(model_names))

    # Create hardware + backend + quant labels for grouping (without model name)
    filtered_df["config_label"] = (
        filtered_df["bench_backend"]
        + " - "
        + filtered_df["hardware_cpu"]
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

    # Add bar for each configuration
    for i, config in enumerate(configs):
        config_data = grouped[grouped["config_label"] == config].sort_values(
            "prompt_size"
        )

        color = colors[i % len(colors)]

        # Add bar with error bars
        fig.add_trace(
            go.Bar(
                x=config_data["prompt_size"],
                y=config_data["mean_tps"],
                name=config,
                marker_color=color,
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
        f"Combined Benchmark Performance: {model_name_for_title}{test_type_label}"
    ]
    if filter_quant:
        title_parts.append(f"Quantization: {filter_quant}")

    title = "<br>".join(title_parts)

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1f77b4")),
        xaxis=dict(
            title="Prompt Size (tokens)",
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
        width=1400,
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

    output_file = output_dir / f"combined_bench_prompt_size{filter_suffix}.html"
    fig.write_html(str(output_file))
    print(f"📊 Saved: {output_file}")

    if show_chart:
        fig.show()


def create_hardware_comparison_chart(
    df: pd.DataFrame,
    filter_model: str = None,
    filter_quant: str = None,
    test_type_filter: str = None,
    show_chart: bool = True,
):
    """Create bar chart comparing different hardware configurations."""
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

    # Get model names for title
    model_names = filtered_df["model_name"].unique()
    model_name_for_title = " / ".join(sorted(model_names))

    # Create backend + hardware labels
    filtered_df["hw_label"] = (
        filtered_df["bench_backend"] + " - " + filtered_df["hardware_cpu"]
    )

    # Group by prompt size and hardware, calculate statistics
    grouped = (
        filtered_df.groupby(["prompt_size", "hw_label"])
        .agg({"tokens_per_second": ["mean", "std", "count"]})
        .reset_index()
    )

    grouped.columns = ["prompt_size", "hw_label", "mean_tps", "std_tps", "count"]

    # Create figure
    fig = go.Figure()

    # Get unique hardware configurations
    hw_configs = sorted(grouped["hw_label"].unique())

    # Generate colors
    colors = px.colors.qualitative.Plotly

    # Add bar for each hardware configuration
    for i, hw in enumerate(hw_configs):
        hw_data = grouped[grouped["hw_label"] == hw].sort_values("prompt_size")

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Bar(
                x=hw_data["prompt_size"],
                y=hw_data["mean_tps"],
                name=hw,
                marker_color=color,
                error_y=dict(
                    type="data",
                    array=hw_data["std_tps"],
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

    title_parts = [f"Combined Benchmark: {model_name_for_title}{test_type_label}"]
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
        width=1400,
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

    output_file = output_dir / f"combined_bench_hardware_comparison{filter_suffix}.html"
    fig.write_html(str(output_file))
    print(f"📊 Saved: {output_file}")

    if show_chart:
        fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot combined benchmark results from llama_bencher and mlx_bencher"
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=[
            "../llama_bencher/results",
            "../mlx_bencher/results",
        ],
        help="Directories containing benchmark results",
    )
    parser.add_argument(
        "--test-type",
        choices=["pp", "tg", "both"],
        default="tg",
        help="Test type to plot (pp=prompt processing, tg=text generation, both=generate both)",
    )
    parser.add_argument(
        "--model",
        help="Filter by model name (partial match)",
    )
    parser.add_argument(
        "--quant",
        help="Filter by quantization",
    )

    args = parser.parse_args()

    # Load results
    df = load_results(args.results_dirs)

    if df.empty:
        print("No data loaded. Exiting.")
        return 1

    # Generate charts based on test type
    charts_generated = []

    if args.test_type in ["pp", "both"]:
        print("\n📊 Generating PP (Prompt Processing) charts...")
        create_prompt_size_performance_chart(
            df,
            filter_model=args.model,
            filter_quant=args.quant,
            test_type_filter="pp",
            show_chart=False,
        )
        charts_generated.append("combined_bench_prompt_size_pp.html")

        create_hardware_comparison_chart(
            df,
            filter_model=args.model,
            filter_quant=args.quant,
            test_type_filter="pp",
            show_chart=False,
        )
        charts_generated.append("combined_bench_hardware_comparison_pp.html")

    if args.test_type in ["tg", "both"]:
        print("\n📊 Generating TG (Text Generation) charts...")
        create_prompt_size_performance_chart(
            df,
            filter_model=args.model,
            filter_quant=args.quant,
            test_type_filter="tg",
            show_chart=False,
        )
        charts_generated.append("combined_bench_prompt_size_tg.html")

        create_hardware_comparison_chart(
            df,
            filter_model=args.model,
            filter_quant=args.quant,
            test_type_filter="tg",
            show_chart=False,
        )
        charts_generated.append("combined_bench_hardware_comparison_tg.html")

    print(f"\n✅ Generated {len(charts_generated)} charts:")
    for chart in charts_generated:
        print(f"   - charts/{chart}")

    return 0


if __name__ == "__main__":
    exit(main())
