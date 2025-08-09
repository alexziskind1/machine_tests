#!/usr/bin/env python3
"""
Prompt-specific Hardware Performance Comparison

Creates column charts showing average performance for each prompt type
with different colored bars for each hardware configuration.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os
import glob
from pathlib import Path
import argparse
import hashlib
import json
import re


def generate_hardware_colors(hardware_configs):
    """
    Generate a consistent color scheme for hardware configurations.

    This function dynamically creates colors for any hardware configuration,
    eliminating the need to hardcode hardware names and colors.
    """
    # Use plotly's qualitative color sequences for good visual distinction
    color_sequences = [
        px.colors.qualitative.Set1,
        px.colors.qualitative.Set2,
        px.colors.qualitative.Set3,
        px.colors.qualitative.Pastel1,
        px.colors.qualitative.Pastel2,
        px.colors.qualitative.Dark24,
    ]

    # Flatten all color sequences
    all_colors = []
    for seq in color_sequences:
        all_colors.extend(seq)

    # Sort hardware configs for consistent assignment
    sorted_configs = sorted(hardware_configs)

    color_map = {}
    for i, config in enumerate(sorted_configs):
        # Use modulo to cycle through colors if we have more hardware than colors
        color_idx = i % len(all_colors)
        color_map[config] = all_colors[color_idx]

    # Always assign gray to 'Unknown'
    if "Unknown" in color_map:
        color_map["Unknown"] = "#95A5A6"

    return color_map


def load_hardware_mappings():
    """Load hardware name mappings from JSON file."""
    mapping_file = Path("hardware_mapping.json")

    if not mapping_file.exists():
        print(f"⚠️  Hardware mapping file not found: {mapping_file}")
        print("   Using raw hardware names from file paths")
        return {
            "hardware_mappings": {},
            "fallback_patterns": {},
            "model_mappings": {},
            "quantization_mappings": {},
        }

    try:
        with open(mapping_file, "r") as f:
            mappings = json.load(f)
        print(f"✅ Loaded hardware mappings from {mapping_file}")
        return mappings
    except Exception as e:
        print(f"⚠️  Error loading hardware mappings: {e}")
        return {
            "hardware_mappings": {},
            "fallback_patterns": {},
            "model_mappings": {},
            "quantization_mappings": {},
        }


# Load mappings at module level
HARDWARE_MAPPINGS = load_hardware_mappings()


def load_statistical_results():
    """Load all statistical results CSV files."""
    results_dir = Path("results")

    # Look for statistics files in main directory and subdirectories
    csv_files = []
    csv_files.extend(list(results_dir.glob("*statistics*.csv")))
    csv_files.extend(list(results_dir.glob("**/*statistics*.csv")))

    if not csv_files:
        print("❌ No statistics CSV files found in results directory")
        print(f"Searched in: {results_dir.absolute()}")
        return None

    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add file source for tracking
            df["source_file"] = str(csv_file)
            all_data.append(df)
            print(f"✅ Loaded {len(df)} rows from {csv_file.name}")
        except Exception as e:
            print(f"⚠️  Error reading {csv_file}: {e}")

    if not all_data:
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Loaded {len(combined_df)} test results from {len(csv_files)} files")
    return combined_df


def load_detailed_results():
    """Load detailed results CSV files for prompt processing time calculation."""
    results_dir = Path("results")

    # Look for detailed files in main directory and subdirectories
    csv_files = []
    csv_files.extend(list(results_dir.glob("*detailed*.csv")))
    csv_files.extend(list(results_dir.glob("**/*detailed*.csv")))

    if not csv_files:
        print("❌ No detailed CSV files found in results directory")
        print(f"Searched in: {results_dir.absolute()}")
        return None

    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add file source for tracking
            df["source_file"] = str(csv_file)
            all_data.append(df)
            print(f"✅ Loaded {len(df)} detailed rows from {csv_file.name}")
        except Exception as e:
            print(f"⚠️  Error reading {csv_file}: {e}")

    if not all_data:
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    print(
        f"📊 Loaded {len(combined_df)} detailed test results from {len(csv_files)} files"
    )
    return combined_df


def clean_hardware_config(config):
    """Clean and standardize hardware configuration names using mapping file."""
    if pd.isna(config):
        return "Unknown"

    config_str = str(config).strip().lower()

    # Basic cleaning - remove extra spaces
    cleaned = " ".join(config_str.split())

    # Try exact mappings first
    hardware_mappings = HARDWARE_MAPPINGS.get("hardware_mappings", {})
    for key, clean_name in hardware_mappings.items():
        if key.lower() in cleaned:
            return clean_name

    # Try fallback patterns
    fallback_patterns = HARDWARE_MAPPINGS.get("fallback_patterns", {})
    for pattern, clean_name in fallback_patterns.items():
        if re.search(pattern.lower(), cleaned):
            return clean_name

    # If no mapping found, return cleaned version with basic formatting
    return cleaned.replace("_", " ").replace("-", " ").title()


def extract_prompt_info(filename):
    """Extract prompt category and complexity from filename."""
    if pd.isna(filename):
        return "Unknown"

    filename_lower = str(filename).lower()

    # Remove .txt extension if present
    if filename_lower.endswith(".txt"):
        filename_lower = filename_lower[:-4]

    # Extract prompt categories
    if "extra_long" in filename_lower:
        if "programming" in filename_lower or "code" in filename_lower:
            return "Extra Long Programming"
        elif "repo" in filename_lower:
            return "Extra Long Repo Analysis"
        else:
            return "Extra Long"
    elif "long" in filename_lower:
        if "programming" in filename_lower:
            if "debug" in filename_lower:
                return "Long Programming Debug"
            elif "project" in filename_lower:
                return "Long Programming Project"
            else:
                return "Long Programming"
        elif "architecture" in filename_lower:
            return "Long Architecture"
        elif "analysis" in filename_lower or "summarize" in filename_lower:
            return "Long Analysis/Summary"
        else:
            return "Long"
    elif "medium" in filename_lower:
        if "programming" in filename_lower:
            if "algorithm" in filename_lower:
                return "Medium Programming Algorithm"
            elif "debug" in filename_lower:
                return "Medium Programming Debug"
            else:
                return "Medium Programming"
        elif "architecture" in filename_lower:
            return "Medium Architecture"
        elif "explanation" in filename_lower:
            return "Medium Explanation"
        else:
            return "Medium"
    elif "short" in filename_lower:
        if "programming" in filename_lower or "debug" in filename_lower:
            return "Short Programming"
        elif "architecture" in filename_lower:
            return "Short Architecture"
        elif "simple" in filename_lower:
            if "greeting" in filename_lower:
                return "Short Simple Greeting"
            elif "math" in filename_lower:
                return "Short Simple Math"
            else:
                return "Short Simple"
        else:
            return "Short"
    else:
        # Fallback - use the filename itself cleaned up
        return filename_lower.replace("_", " ").title()


def extract_hardware_config(source_file_path):
    """
    Extract hardware configuration from source file path.

    This function dynamically extracts hardware names from file paths,
    allowing for new hardware configurations without code changes.
    It looks for hardware-specific terms in directory and file names.
    """
    file_path = str(source_file_path)
    filename = os.path.basename(file_path)

    # Look for hardware patterns at the beginning of the filename
    # Most files follow pattern: hardware_model_quantization_...
    filename_lower = filename.lower()

    # Try to extract hardware from beginning of filename
    # Look for known hardware patterns
    hardware_patterns = [
        "rtx5090wrtx5060ti",
        "rtx5090",
        "rtx_5090",
        "m4max",
        "m4_max",
        "m4pro",
        "m4_pro",
        "m3max",
        "m3_max",
        "m3pro",
        "m3_pro",
        "m3ultra",
        "m3_ultra",
        "m3mba",
        "m3_mba",
        "m2max",
        "m2_max",
        "m2mba",
        "m2_mba",
        "m1pro",
        "m1_pro",
        "m1mba",
        "m1_mba",
        "fw_ryzen_ai_395",
        "fw_ryzen",
        "framework",
        "xelite",
        "xplus",
        "surface_laptop7",
        "surface_laptop6",
        "galaxybook4edge",
        "vivobooks15",
        "xps13",
        "corei5",
        "corei9",
        "coreu7",
    ]

    # Sort by length (longest first) to match more specific patterns first
    hardware_patterns.sort(key=len, reverse=True)

    for pattern in hardware_patterns:
        if filename_lower.startswith(pattern.lower()):
            return pattern

    # Fallback: look for hardware terms in the first part of filename
    first_part = filename.split("_")[0].lower()
    if any(
        hw_term in first_part
        for hw_term in [
            "rtx",
            "m1",
            "m2",
            "m3",
            "m4",
            "intel",
            "amd",
            "ryzen",
            "framework",
            "surface",
            "xps",
            "galaxy",
        ]
    ):
        return filename.split("_")[0]

    return "Unknown"


def extract_quantization_level(source_file_path):
    """Extract quantization level from source file path using mapping file."""
    file_path = str(source_file_path).lower()

    # Try mappings first
    quant_mappings = HARDWARE_MAPPINGS.get("quantization_mappings", {})
    for key, clean_name in quant_mappings.items():
        if key.lower() in file_path:
            return clean_name

    # Fallback to consistent format matching mapping file
    if "q4" in file_path or "4bit" in file_path or "mlx4bit" in file_path:
        return "int4"
    elif "q8" in file_path or "8bit" in file_path:
        return "int8"
    elif "q3" in file_path or "3bit" in file_path:
        return "int3"
    elif "fp16" in file_path or "f16" in file_path:
        return "fp16"
    elif "mlx" in file_path and "bit" not in file_path:
        return "mlx"
    else:
        return "unknown"


def extract_model_info(source_file_path):
    """Extract model information from source file path using mapping file."""
    file_path = str(source_file_path).lower()

    # Try mappings first
    model_mappings = HARDWARE_MAPPINGS.get("model_mappings", {})
    for key, clean_name in model_mappings.items():
        if key.lower() in file_path:
            return clean_name

    # Fallback to original logic for unmapped models
    if "qwen3_coder_30b" in file_path:
        return "qwen3_coder_30b"
    elif "llama_3.3_70b" in file_path:
        return "llama_3.3_70b"
    elif "qwen3_235b" in file_path:
        return "qwen3_235b"
    elif "gemma" in file_path:
        return "gemma"
    else:
        return "unknown"


def resolve_filter_value(user_input, current_values, mapping_dict, mapping_key):
    """
    Resolve user filter input to actual values in the data.
    Supports both raw keys and clean names from mappings.
    """
    if not user_input:
        return None

    # Check if user input exactly matches any current value
    if user_input in current_values:
        return user_input

    # Special handling for quantization legacy formats (backward compatibility)
    if mapping_key == "quantization_mappings":
        legacy_quantization_map = {
            "4-bit": "int4",
            "8-bit": "int8",
            "3-bit": "int3",
            "FP16": "fp16",
            "MLX": "mlx",
        }
        if user_input in legacy_quantization_map:
            mapped_value = legacy_quantization_map[user_input]
            if mapped_value in current_values:
                return mapped_value

    # Check if user input is a raw key that maps to a clean name in current values
    mappings = mapping_dict.get(mapping_key, {})
    for raw_key, clean_name in mappings.items():
        if user_input.lower() == raw_key.lower() and clean_name in current_values:
            return clean_name
        if user_input.lower() == clean_name.lower() and clean_name in current_values:
            return clean_name

    # Check for partial matches (case insensitive)
    user_lower = user_input.lower()
    for value in current_values:
        if user_lower in value.lower() or value.lower() in user_lower:
            return value

    return None


def create_prompt_hardware_chart(
    df,
    metric="tokens_per_second_mean",
    show_chart=True,
    filter_model=None,
    filter_quantization=None,
):
    """Create column chart comparing hardware performance by prompt type."""

    # Clean the data and filter out failed tests
    df = df.copy()
    df = df[df["successful_runs"] > 0]  # Only include successful tests

    # Extract hardware config, prompt type, quantization, and model info
    df["hardware_config"] = df["source_file"].apply(extract_hardware_config)
    df["prompt_type"] = df["filename"].apply(extract_prompt_info)
    df["quantization"] = df["source_file"].apply(extract_quantization_level)
    df["model_name"] = df["source_file"].apply(extract_model_info)

    # Clean hardware names
    df["hardware_config"] = df["hardware_config"].apply(clean_hardware_config)

    # Filter out only Unknown hardware (keep all valid hardware configurations)
    df = df[df["hardware_config"] != "Unknown"]

    # Apply model filter if specified
    if filter_model:
        resolved_model = resolve_filter_value(
            filter_model, df["model_name"].unique(), HARDWARE_MAPPINGS, "model_mappings"
        )
        if resolved_model:
            df = df[df["model_name"] == resolved_model]
            print(f"🔍 Filtered to model: {filter_model} → {resolved_model}")
        else:
            print(
                f"❌ Model '{filter_model}' not found. Available: {list(df['model_name'].unique())}"
            )
            return None, None

    # Apply quantization filter if specified
    if filter_quantization:
        resolved_quant = resolve_filter_value(
            filter_quantization,
            df["quantization"].unique(),
            HARDWARE_MAPPINGS,
            "quantization_mappings",
        )
        if resolved_quant:
            df = df[df["quantization"] == resolved_quant]
            print(
                f"🔍 Filtered to quantization: {filter_quantization} → {resolved_quant}"
            )
        else:
            print(
                f"❌ Quantization '{filter_quantization}' not found. Available: {list(df['quantization'].unique())}"
            )
            return None, None

    if df.empty:
        print("❌ No data remaining after filtering")
        return None, None

    # Group by prompt type and hardware, calculate averages
    grouped = (
        df.groupby(["prompt_type", "hardware_config"])[metric]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    grouped.columns = [
        "prompt_type",
        "hardware_config",
        "avg_performance",
        "count",
        "std_dev",
    ]

    # Remove groups with too few samples
    grouped = grouped[grouped["count"] >= 1]

    # Create the chart
    fig = go.Figure()

    # Get unique prompt types and hardware configs
    prompt_types = sorted(grouped["prompt_type"].unique())
    hardware_configs = sorted(grouped["hardware_config"].unique())

    print(
        f"📊 Found {len(prompt_types)} prompt types and {len(hardware_configs)} hardware configs"
    )

    # Generate dynamic color scheme for all hardware configurations
    hardware_colors = generate_hardware_colors(hardware_configs)

    # Add bars for each hardware config
    for hardware in hardware_configs:
        hardware_data = grouped[grouped["hardware_config"] == hardware]

        x_values = []
        y_values = []
        error_values = []
        hover_text = []

        for prompt_type in prompt_types:
            prompt_data = hardware_data[hardware_data["prompt_type"] == prompt_type]
            if not prompt_data.empty:
                x_values.append(prompt_type)
                avg_perf = prompt_data["avg_performance"].iloc[0]
                y_values.append(avg_perf)
                std_dev = (
                    prompt_data["std_dev"].iloc[0]
                    if not pd.isna(prompt_data["std_dev"].iloc[0])
                    else 0
                )
                error_values.append(std_dev)
                hover_text.append(
                    f"<b>{hardware}</b><br>"
                    f"Prompt: {prompt_type}<br>"
                    f"Avg Performance: {avg_perf:.1f} tokens/sec<br>"
                    f"Tests: {prompt_data['count'].iloc[0]}<br>"
                    f"Std Dev: {std_dev:.1f}"
                )

        if x_values:  # Only add trace if there's data
            color = hardware_colors.get(hardware, "#95A5A6")

            fig.add_trace(
                go.Bar(
                    name=hardware,
                    x=x_values,
                    y=y_values,
                    error_y=dict(type="data", array=error_values, visible=True),
                    marker_color=color,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_text,
                    opacity=0.8,
                )
            )

    # Create title with filter information
    title_parts = ["Hardware Performance Comparison by Prompt Type"]
    if filter_model:
        title_parts.append(f"Model: {filter_model}")
    if filter_quantization:
        title_parts.append(f"Quantization: {filter_quantization}")

    title_text = "<br>".join(title_parts)
    if len(title_parts) > 1:
        title_text += f'<br><sub>Average {metric.replace("_", " ").title()}</sub>'
    else:
        title_text += f'<br><sub>Average {metric.replace("_", " ").title()}</sub>'

    # Update layout
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
            "y": 0.95,  # Position title lower
            "yanchor": "top",
        },
        xaxis_title="Prompt Type",
        yaxis_title=f'Average {metric.replace("_", " ").title()}',
        barmode="group",
        bargap=0.2,
        bargroupgap=0.1,
        height=800,  # Increased height
        width=1400,
        template="plotly_white",
        font={"size": 12},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,  # Position legend higher to avoid overlap
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            t=150,  # Top margin for title and legend
            b=120,  # Bottom margin for rotated labels
            l=80,  # Left margin
            r=80,  # Right margin
        ),
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)

    if show_chart:
        fig.show()

    return fig, grouped


def create_prompt_processing_chart(
    df, show_chart=True, filter_model=None, filter_quantization=None
):
    """Create column chart comparing estimated prompt processing time by prompt type."""

    # Clean the data and filter out failed tests
    df = df.copy()
    df = df[df["success"] == True]  # Only include successful tests

    # Calculate estimated prompt processing time
    # Formula: Prompt Processing Time = Total Response Time - (Response Tokens ÷ Tokens Per Second)
    df["generation_time"] = df["response_token_count"] / df["tokens_per_second"]
    df["estimated_prompt_processing_time"] = df["response_time"] - df["generation_time"]

    # Filter out negative values (unrealistic estimates)
    df = df[df["estimated_prompt_processing_time"] >= 0]

    # Extract hardware config, prompt type, quantization, and model info
    df["hardware_config"] = df["source_file"].apply(extract_hardware_config)
    df["prompt_type"] = df["filename"].apply(extract_prompt_info)
    df["quantization"] = df["source_file"].apply(extract_quantization_level)
    df["model_name"] = df["source_file"].apply(extract_model_info)

    # Clean hardware names
    df["hardware_config"] = df["hardware_config"].apply(clean_hardware_config)

    # Filter out only Unknown hardware (keep all valid hardware configurations)
    df = df[df["hardware_config"] != "Unknown"]

    # Apply model filter if specified
    if filter_model:
        resolved_model = resolve_filter_value(
            filter_model, df["model_name"].unique(), HARDWARE_MAPPINGS, "model_mappings"
        )
        if resolved_model:
            df = df[df["model_name"] == resolved_model]
            print(f"🔍 Filtered to model: {filter_model} → {resolved_model}")
        else:
            print(
                f"❌ Model '{filter_model}' not found. Available: {list(df['model_name'].unique())}"
            )
            return None, None

    # Apply quantization filter if specified
    if filter_quantization:
        resolved_quant = resolve_filter_value(
            filter_quantization,
            df["quantization"].unique(),
            HARDWARE_MAPPINGS,
            "quantization_mappings",
        )
        if resolved_quant:
            df = df[df["quantization"] == resolved_quant]
            print(
                f"🔍 Filtered to quantization: {filter_quantization} → {resolved_quant}"
            )
        else:
            print(
                f"❌ Quantization '{filter_quantization}' not found. Available: {list(df['quantization'].unique())}"
            )
            return None, None

    if df.empty:
        print("❌ No data remaining after filtering")
        return None, None

    # Group by prompt type and hardware, calculate averages for prompt processing time
    grouped = (
        df.groupby(["prompt_type", "hardware_config"])[
            "estimated_prompt_processing_time"
        ]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    grouped.columns = [
        "prompt_type",
        "hardware_config",
        "avg_prompt_processing_time",
        "count",
        "std_dev",
    ]

    # Remove groups with too few samples
    grouped = grouped[
        grouped["count"] >= 2
    ]  # Need at least 2 samples for meaningful statistics

    # Create the chart
    fig = go.Figure()

    # Get unique prompt types and hardware configs
    prompt_types = sorted(grouped["prompt_type"].unique())
    hardware_configs = sorted(grouped["hardware_config"].unique())

    print(
        f"📊 Found {len(prompt_types)} prompt types and {len(hardware_configs)} hardware configs"
    )

    # Generate dynamic color scheme for all hardware configurations
    hardware_colors = generate_hardware_colors(hardware_configs)

    # Add bars for each hardware config
    for hardware in hardware_configs:
        hardware_data = grouped[grouped["hardware_config"] == hardware]

        x_values = []
        y_values = []
        error_values = []
        hover_text = []

        for prompt_type in prompt_types:
            prompt_data = hardware_data[hardware_data["prompt_type"] == prompt_type]
            if not prompt_data.empty:
                x_values.append(prompt_type)
                avg_pp_time = prompt_data["avg_prompt_processing_time"].iloc[0]
                y_values.append(avg_pp_time)
                std_dev = (
                    prompt_data["std_dev"].iloc[0]
                    if not pd.isna(prompt_data["std_dev"].iloc[0])
                    else 0
                )
                error_values.append(std_dev)
                hover_text.append(
                    f"<b>{hardware}</b><br>"
                    f"Prompt: {prompt_type}<br>"
                    f"Avg Prompt Processing: {avg_pp_time:.2f} seconds<br>"
                    f"Tests: {prompt_data['count'].iloc[0]}<br>"
                    f"Std Dev: {std_dev:.2f}s"
                )

        if x_values:  # Only add trace if there's data
            color = hardware_colors.get(hardware, "#95A5A6")

            fig.add_trace(
                go.Bar(
                    name=hardware,
                    x=x_values,
                    y=y_values,
                    error_y=dict(type="data", array=error_values, visible=True),
                    marker_color=color,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_text,
                    opacity=0.8,
                )
            )

    # Create title with filter information
    title_parts = ["Estimated Prompt Processing Time Comparison by Hardware"]
    if filter_model:
        title_parts.append(f"Model: {filter_model}")
    if filter_quantization:
        title_parts.append(f"Quantization: {filter_quantization}")

    title_text = "<br>".join(title_parts)
    title_text += (
        "<br><sub>Calculated: Total Time - (Response Tokens ÷ Generation Speed)</sub>"
    )

    # Update layout
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
            "y": 0.95,
            "yanchor": "top",
        },
        xaxis_title="Prompt Type",
        yaxis_title="Average Prompt Processing Time (seconds)",
        barmode="group",
        bargap=0.2,
        bargroupgap=0.1,
        height=800,
        width=1400,
        template="plotly_white",
        font={"size": 12},
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(t=150, b=120, l=80, r=80),
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)

    # Fix y-axis formatting to show proper decimal numbers
    fig.update_yaxes(
        tickformat=".2f",  # Force 2 decimal places
        tickmode="linear",
        showticklabels=True,
    )

    if show_chart:
        fig.show()

    return fig, grouped


def create_prompt_processing_summary_stats(grouped_data):
    """Create summary statistics for the prompt processing time comparison."""
    print("\n📊 PROMPT PROCESSING TIME SUMMARY")
    print("=" * 60)

    # Best (lowest processing time) performing combinations
    best_combinations = grouped_data.nsmallest(10, "avg_prompt_processing_time")
    print("\n🏆 TOP 10 FASTEST PROMPT PROCESSING COMBINATIONS:")
    for i, (idx, row) in enumerate(best_combinations.iterrows(), 1):
        print(
            f"{i:2d}. {row['hardware_config']} - {row['prompt_type']}: "
            f"{row['avg_prompt_processing_time']:.2f}s ({row['count']} tests)"
        )

    # Worst (highest processing time) performing combinations
    worst_combinations = grouped_data.nlargest(10, "avg_prompt_processing_time")
    print("\n🐌 TOP 10 SLOWEST PROMPT PROCESSING COMBINATIONS:")
    for i, (idx, row) in enumerate(worst_combinations.iterrows(), 1):
        print(
            f"{i:2d}. {row['hardware_config']} - {row['prompt_type']}: "
            f"{row['avg_prompt_processing_time']:.2f}s ({row['count']} tests)"
        )

    # Best hardware by prompt type (lowest processing time)
    print("\n🎯 FASTEST PROMPT PROCESSING BY PROMPT TYPE:")
    for prompt_type in sorted(grouped_data["prompt_type"].unique()):
        prompt_data = grouped_data[grouped_data["prompt_type"] == prompt_type]
        best = prompt_data.loc[prompt_data["avg_prompt_processing_time"].idxmin()]
        print(
            f"   {prompt_type}: {best['hardware_config']} ({best['avg_prompt_processing_time']:.2f}s)"
        )

    # Hardware rankings (lower processing time is better)
    print("\n🏅 OVERALL HARDWARE RANKINGS (By Prompt Processing Speed):")
    hardware_avg = (
        grouped_data.groupby("hardware_config")["avg_prompt_processing_time"]
        .mean()
        .sort_values(ascending=True)
    )
    for i, (hardware, avg_time) in enumerate(hardware_avg.items(), 1):
        test_count = grouped_data[grouped_data["hardware_config"] == hardware][
            "count"
        ].sum()
        print(
            f"{i:2d}. {hardware}: {avg_time:.2f}s avg processing time ({test_count} total tests)"
        )

    print("\n💡 Note: Prompt processing times are estimated using the formula:")
    print(
        "   Prompt Processing Time = Total Response Time - (Response Tokens ÷ Generation Speed)"
    )


def create_prompt_processing_delay_chart(
    df, show_chart=True, filter_model=None, filter_quantization=None
):
    """Create column chart comparing prompt processing delay (response time) by prompt type."""

    # Clean the data and filter out failed tests
    df = df.copy()
    df = df[df["successful_runs"] > 0]  # Only include successful tests

    # Extract hardware config, prompt type, quantization, and model info
    df["hardware_config"] = df["source_file"].apply(extract_hardware_config)
    df["prompt_type"] = df["filename"].apply(extract_prompt_info)
    df["quantization"] = df["source_file"].apply(extract_quantization_level)
    df["model_name"] = df["source_file"].apply(extract_model_info)

    # Clean hardware names
    df["hardware_config"] = df["hardware_config"].apply(clean_hardware_config)

    # Filter out only Unknown hardware (keep all valid hardware configurations)
    df = df[df["hardware_config"] != "Unknown"]

    # Apply model filter if specified
    if filter_model:
        resolved_model = resolve_filter_value(
            filter_model, df["model_name"].unique(), HARDWARE_MAPPINGS, "model_mappings"
        )
        if resolved_model:
            df = df[df["model_name"] == resolved_model]
            print(f"🔍 Filtered to model: {filter_model} → {resolved_model}")
        else:
            print(
                f"❌ Model '{filter_model}' not found. Available: {list(df['model_name'].unique())}"
            )
            return None, None

    # Apply quantization filter if specified
    if filter_quantization:
        resolved_quant = resolve_filter_value(
            filter_quantization,
            df["quantization"].unique(),
            HARDWARE_MAPPINGS,
            "quantization_mappings",
        )
        if resolved_quant:
            df = df[df["quantization"] == resolved_quant]
            print(
                f"🔍 Filtered to quantization: {filter_quantization} → {resolved_quant}"
            )
        else:
            print(
                f"❌ Quantization '{filter_quantization}' not found. Available: {list(df['quantization'].unique())}"
            )
            return None, None

    if df.empty:
        print("❌ No data remaining after filtering")
        return None, None

    # Group by prompt type and hardware, calculate averages for response time
    grouped = (
        df.groupby(["prompt_type", "hardware_config"])["response_time_mean"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    grouped.columns = [
        "prompt_type",
        "hardware_config",
        "avg_delay",
        "count",
        "std_dev",
    ]

    # Remove groups with too few samples
    grouped = grouped[grouped["count"] >= 1]

    # Create the chart
    fig = go.Figure()

    # Get unique prompt types and hardware configs
    prompt_types = sorted(grouped["prompt_type"].unique())
    hardware_configs = sorted(grouped["hardware_config"].unique())

    print(
        f"📊 Found {len(prompt_types)} prompt types and {len(hardware_configs)} hardware configs"
    )

    # Generate dynamic color scheme for all hardware configurations
    hardware_colors = generate_hardware_colors(hardware_configs)

    # Add bars for each hardware config
    for hardware in hardware_configs:
        hardware_data = grouped[grouped["hardware_config"] == hardware]

        x_values = []
        y_values = []
        error_values = []
        hover_text = []

        for prompt_type in prompt_types:
            prompt_data = hardware_data[hardware_data["prompt_type"] == prompt_type]
            if not prompt_data.empty:
                x_values.append(prompt_type)
                avg_delay = prompt_data["avg_delay"].iloc[0]
                y_values.append(avg_delay)
                std_dev = (
                    prompt_data["std_dev"].iloc[0]
                    if not pd.isna(prompt_data["std_dev"].iloc[0])
                    else 0
                )
                error_values.append(std_dev)
                hover_text.append(
                    f"<b>{hardware}</b><br>"
                    f"Prompt: {prompt_type}<br>"
                    f"Avg Delay: {avg_delay:.2f} seconds<br>"
                    f"Tests: {prompt_data['count'].iloc[0]}<br>"
                    f"Std Dev: {std_dev:.2f}s"
                )

        if x_values:  # Only add trace if there's data
            color = hardware_colors.get(hardware, "#95A5A6")

            fig.add_trace(
                go.Bar(
                    name=hardware,
                    x=x_values,
                    y=y_values,
                    error_y=dict(type="data", array=error_values, visible=True),
                    marker_color=color,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_text,
                    opacity=0.8,
                )
            )

    # Create title with filter information
    title_parts = ["Prompt Processing Delay Comparison by Hardware"]
    if filter_model:
        title_parts.append(f"Model: {filter_model}")
    if filter_quantization:
        title_parts.append(f"Quantization: {filter_quantization}")

    title_text = "<br>".join(title_parts)
    title_text += "<br><sub>Average Response Time (Lower is Better)</sub>"

    # Update layout
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
            "y": 0.95,
            "yanchor": "top",
        },
        xaxis_title="Prompt Type",
        yaxis_title="Average Response Time (seconds)",
        barmode="group",
        bargap=0.2,
        bargroupgap=0.1,
        height=800,
        width=1400,
        template="plotly_white",
        font={"size": 12},
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(t=150, b=120, l=80, r=80),
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)

    # Reverse y-axis color scale conceptually (lower is better for delay)
    # We'll use a different color scheme to emphasize that lower is better

    if show_chart:
        fig.show()

    return fig, grouped


def create_delay_summary_stats(grouped_data):
    """Create summary statistics for the prompt processing delay comparison."""
    print("\n📊 PROMPT PROCESSING DELAY SUMMARY")
    print("=" * 60)

    # Best (lowest delay) performing combinations
    best_combinations = grouped_data.nsmallest(10, "avg_delay")
    print("\n🏆 TOP 10 FASTEST PROMPT-HARDWARE COMBINATIONS (Lowest Delay):")
    for i, (idx, row) in enumerate(best_combinations.iterrows(), 1):
        print(
            f"{i:2d}. {row['hardware_config']} - {row['prompt_type']}: "
            f"{row['avg_delay']:.2f}s ({row['count']} tests)"
        )

    # Worst (highest delay) performing combinations
    worst_combinations = grouped_data.nlargest(10, "avg_delay")
    print("\n🐌 TOP 10 SLOWEST PROMPT-HARDWARE COMBINATIONS (Highest Delay):")
    for i, (idx, row) in enumerate(worst_combinations.iterrows(), 1):
        print(
            f"{i:2d}. {row['hardware_config']} - {row['prompt_type']}: "
            f"{row['avg_delay']:.2f}s ({row['count']} tests)"
        )

    # Best hardware by prompt type (lowest delay)
    print("\n🎯 FASTEST HARDWARE BY PROMPT TYPE (Lowest Delay):")
    for prompt_type in sorted(grouped_data["prompt_type"].unique()):
        prompt_data = grouped_data[grouped_data["prompt_type"] == prompt_type]
        best = prompt_data.loc[prompt_data["avg_delay"].idxmin()]
        print(f"   {prompt_type}: {best['hardware_config']} ({best['avg_delay']:.2f}s)")

    # Hardware rankings (lower delay is better)
    print("\n🏅 OVERALL HARDWARE RANKINGS (By Speed - Lower Delay):")
    hardware_avg = (
        grouped_data.groupby("hardware_config")["avg_delay"]
        .mean()
        .sort_values(ascending=True)
    )
    for i, (hardware, avg_delay) in enumerate(hardware_avg.items(), 1):
        test_count = grouped_data[grouped_data["hardware_config"] == hardware][
            "count"
        ].sum()
        print(
            f"{i:2d}. {hardware}: {avg_delay:.2f}s avg delay ({test_count} total tests)"
        )


def create_summary_stats(grouped_data):
    """Create summary statistics for the prompt-hardware performance comparison."""
    print("\n📊 PROMPT-HARDWARE PERFORMANCE SUMMARY")
    print("=" * 60)

    # Overall best performing combinations
    top_combinations = grouped_data.nlargest(10, "avg_performance")
    print("\n🏆 TOP 10 PROMPT-HARDWARE COMBINATIONS:")
    for i, (idx, row) in enumerate(top_combinations.iterrows(), 1):
        print(
            f"{i:2d}. {row['hardware_config']} - {row['prompt_type']}: "
            f"{row['avg_performance']:.1f} tokens/sec ({row['count']} tests)"
        )

    # Best hardware by prompt type
    print("\n🎯 BEST HARDWARE BY PROMPT TYPE:")
    for prompt_type in sorted(grouped_data["prompt_type"].unique()):
        prompt_data = grouped_data[grouped_data["prompt_type"] == prompt_type]
        best = prompt_data.loc[prompt_data["avg_performance"].idxmax()]
        print(
            f"   {prompt_type}: {best['hardware_config']} ({best['avg_performance']:.1f} tokens/sec)"
        )

    # Hardware rankings
    print("\n🏅 OVERALL HARDWARE RANKINGS:")
    hardware_avg = (
        grouped_data.groupby("hardware_config")["avg_performance"]
        .mean()
        .sort_values(ascending=False)
    )
    for i, (hardware, avg_perf) in enumerate(hardware_avg.items(), 1):
        test_count = grouped_data[grouped_data["hardware_config"] == hardware][
            "count"
        ].sum()
        print(
            f"{i:2d}. {hardware}: {avg_perf:.1f} tokens/sec avg ({test_count} total tests)"
        )


def show_detected_hardware_names(df):
    """Show all detected hardware names to help with mapping file creation."""
    print("\n🔍 DETECTED HARDWARE NAMES:")
    print("=" * 50)

    # Get raw hardware names before cleaning
    df_temp = df.copy()
    df_temp["raw_hardware"] = df_temp["source_file"].apply(extract_hardware_config)
    df_temp["clean_hardware"] = df_temp["raw_hardware"].apply(clean_hardware_config)

    # Show mapping results
    hardware_mapping = df_temp[["raw_hardware", "clean_hardware"]].drop_duplicates()
    hardware_mapping = hardware_mapping[hardware_mapping["raw_hardware"] != "Unknown"]

    print("Raw Name → Clean Name:")
    for _, row in hardware_mapping.iterrows():
        mapped = (
            "✅"
            if row["raw_hardware"].lower() != row["clean_hardware"].lower()
            else "❌"
        )
        print(f"  {mapped} {row['raw_hardware']} → {row['clean_hardware']}")

    # Show unmapped names
    unmapped_mask = hardware_mapping.apply(
        lambda row: row["raw_hardware"].lower() == row["clean_hardware"].lower(), axis=1
    )
    unmapped = hardware_mapping[unmapped_mask]
    if not unmapped.empty:
        print(f"\n💡 Consider adding these to hardware_mapping.json:")
        for _, row in unmapped.iterrows():
            print(f'    "{row["raw_hardware"].lower()}": "Your Clean Name Here",')
    else:
        print(f"\n✅ All hardware names are properly mapped!")


def main():
    parser = argparse.ArgumentParser(
        description="Compare hardware performance by prompt type"
    )
    parser.add_argument(
        "--metric",
        default="tokens_per_second_mean",
        choices=["tokens_per_second_mean", "response_time", "cv"],
        help="Metric to compare (default: tokens_per_second_mean)",
    )
    parser.add_argument("--filter-prompt", help="Filter to specific prompt type")
    parser.add_argument("--filter-hardware", help="Filter to specific hardware config")
    parser.add_argument(
        "--model", help="Filter to specific model (e.g., qwen3_coder_30b)"
    )
    parser.add_argument(
        "--quantization",
        choices=["int4", "int8", "int3", "fp4", "fp8", "fp16", "mlx"],
        help="Filter to specific quantization level",
    )
    parser.add_argument(
        "--no-chart", action="store_true", help="Skip showing the chart"
    )
    parser.add_argument(
        "--save", help="Save chart as HTML file (will be saved in charts/ folder)"
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List available models and quantization levels",
    )
    parser.add_argument(
        "--delay-chart",
        action="store_true",
        help="Create prompt processing delay chart instead of performance chart",
    )
    parser.add_argument(
        "--pp",
        action="store_true",
        help="Create estimated prompt processing time chart (requires detailed data)",
    )
    parser.add_argument(
        "--show-hardware-names",
        action="store_true",
        help="Show detected hardware names to help with mapping file creation",
    )

    args = parser.parse_args()

    # Load data based on chart type
    if args.pp:
        # Load detailed data for prompt processing time calculation
        df = load_detailed_results()
        if df is None:
            print(
                "❌ No detailed data available for prompt processing time calculation"
            )
            print(
                "   The --pp flag requires detailed CSV files with response_token_count data"
            )
            return
    else:
        # Load statistical data for other charts
        df = load_statistical_results()
        if df is None:
            return

    # Add metadata columns
    df["hardware_config"] = df["source_file"].apply(extract_hardware_config)
    df["quantization"] = df["source_file"].apply(extract_quantization_level)
    df["model_name"] = df["source_file"].apply(extract_model_info)

    # Show detected hardware names if requested
    if args.show_hardware_names:
        show_detected_hardware_names(df)
        return

    # List available options if requested
    if args.list_available:
        print("\n📋 AVAILABLE OPTIONS:")
        print("=" * 40)

        print("\n🤖 Available Models:")
        models = sorted(df["model_name"].unique())
        for model in models:
            count = len(df[df["model_name"] == model])
            print(f"   {model} ({count} tests)")

        print("\n⚙️ Available Quantization Levels:")
        quants = sorted(df["quantization"].unique())
        for quant in quants:
            count = len(df[df["quantization"] == quant])
            print(f"   {quant} ({count} tests)")

        print("\n🖥️ Available Hardware Configs:")
        hardware = sorted(df["hardware_config"].unique())
        for hw in hardware:
            count = len(df[df["hardware_config"] == hw])
            print(f"   {hw} ({count} tests)")
        return

    # Apply legacy filters
    if args.filter_prompt:
        df = df[df["filename"].str.contains(args.filter_prompt, case=False, na=False)]
        print(f"🔍 Filtered to prompt type containing: {args.filter_prompt}")

    if args.filter_hardware:
        df = df[
            df["hardware_config"].str.contains(
                args.filter_hardware, case=False, na=False
            )
        ]
        print(f"🔍 Filtered to hardware containing: {args.filter_hardware}")

    if df.empty:
        print("❌ No data remaining after filtering")
        return

    # Create appropriate chart based on request
    if args.pp:
        # Create prompt processing time chart
        fig, grouped_data = create_prompt_processing_chart(
            df, not args.no_chart, args.model, args.quantization
        )

        if grouped_data is not None:
            create_prompt_processing_summary_stats(grouped_data)
            chart_type = "prompt_processing"
    elif args.delay_chart:
        # Create delay chart
        fig, grouped_data = create_prompt_processing_delay_chart(
            df, not args.no_chart, args.model, args.quantization
        )

        if grouped_data is not None:
            create_delay_summary_stats(grouped_data)
            chart_type = "delay"
    else:
        # Create performance chart
        fig, grouped_data = create_prompt_hardware_chart(
            df, args.metric, not args.no_chart, args.model, args.quantization
        )

        if grouped_data is not None:
            create_summary_stats(grouped_data)
            chart_type = "performance"

    if grouped_data is not None:
        # Save if requested
        if args.save:
            # Ensure charts directory exists
            charts_dir = Path("charts")
            charts_dir.mkdir(exist_ok=True)

            # Create full path in charts directory
            save_path = charts_dir / args.save
            fig.write_html(save_path)
            print(f"\n💾 Chart saved to: {save_path}")
        elif args.model or args.quantization or args.delay_chart or args.pp:
            # Auto-save with descriptive filename if model/quantization specified or special chart type
            charts_dir = Path("charts")
            charts_dir.mkdir(exist_ok=True)

            filename_parts = [chart_type + "_comparison"]
            if args.model:
                filename_parts.append(args.model)
            if args.quantization:
                filename_parts.append(args.quantization.replace("-", ""))
            if not args.delay_chart and not args.pp:
                filename_parts.append(args.metric)

            filename = "_".join(filename_parts) + ".html"
            save_path = charts_dir / filename
            fig.write_html(save_path)
            print(f"\n💾 Chart auto-saved to: {save_path}")


if __name__ == "__main__":
    main()
