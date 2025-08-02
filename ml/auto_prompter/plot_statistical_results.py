#!/usr/bin/env python3
"""
Statistical LLM Performance Visualization Script

Creates simple plots using Plotly to visualize statistical LLM performance data
from the CSV statistics output of statistical_llm_tester.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import sys
import os
import glob
from datetime import datetime


def find_latest_statistics_csv() -> str:
    """Find the most recent statistics CSV file in the results directory."""
    results_dir = "results"

    # Look for statistics CSV files in results directory
    csv_files = glob.glob(os.path.join(results_dir, "*_statistics_*.csv"))

    if not csv_files:
        # Fall back to any CSV with "statistics" in the name
        csv_files = glob.glob(os.path.join(results_dir, "*statistics*.csv"))

    if not csv_files:
        return "results/performance_results_statistics.csv"  # Default fallback

    # Sort by modification time, most recent first
    csv_files.sort(key=os.path.getmtime, reverse=True)
    return csv_files[0]


def load_statistical_data(csv_file: str) -> pd.DataFrame:
    """Load and prepare the statistical performance data."""
    try:
        df = pd.read_csv(csv_file)

        # Filter only successful tests (where we have data)
        df_success = df[df["successful_runs"] > 0].copy()

        if len(df_success) == 0:
            print("No successful tests found in the statistical data!")
            return None

        # Clean up filename for better display
        df_success["prompt_name"] = (
            df_success["filename"]
            .str.replace(".txt", "")
            .str.replace("_", " ")
            .str.title()
        )

        return df_success

    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        print("Make sure you've run the statistical performance tester first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_tokens_per_second_chart(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing average tokens per second with error bars."""
    # Get model and URL info from the data
    model = df["model"].iloc[0] if "model" in df.columns else "Unknown Model"
    llm_url = df["llm_url"].iloc[0] if "llm_url" in df.columns else "Unknown URL"

    # Create color mapping based on reliability (CV)
    colors = []
    reliability_labels = []
    for cv in df["tokens_per_second_cv"]:
        if cv < 10:
            colors.append("#28a745")  # Dark green - Excellent
            reliability_labels.append("Excellent (CV < 10%)")
        elif cv < 20:
            colors.append("#20c997")  # Light green - Good
            reliability_labels.append("Good (CV < 20%)")
        elif cv < 30:
            colors.append("#ffc107")  # Orange - Moderate
            reliability_labels.append("Moderate (CV < 30%)")
        else:
            colors.append("#dc3545")  # Red - Poor
            reliability_labels.append("Poor (CV ≥ 30%)")

    fig = go.Figure()

    # Add bars with error bars
    fig.add_trace(
        go.Bar(
            x=df["prompt_name"],
            y=df["tokens_per_second_mean"],
            error_y=dict(
                type="data",
                array=df["tokens_per_second_std"],
                visible=True,
                color="rgba(0,0,0,0.3)",
            ),
            marker_color=colors,
            text=[
                f"{mean:.1f}±{std:.1f}"
                for mean, std in zip(
                    df["tokens_per_second_mean"], df["tokens_per_second_std"]
                )
            ],
            textposition="outside",
            name="Tokens/sec",
            hovertemplate="<b>%{x}</b><br>"
            + "Tokens/sec: %{y:.1f}<br>"
            + "CV: %{customdata:.1f}%<br>"
            + "<extra></extra>",
            customdata=df["tokens_per_second_cv"],
        )
    )

    # Add invisible traces for legend
    legend_added = set()
    for i, (cv, reliability_label) in enumerate(
        zip(df["tokens_per_second_cv"], reliability_labels)
    ):
        if reliability_label not in legend_added:
            color = colors[i]
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=reliability_label,
                    showlegend=True,
                )
            )
            legend_added.add(reliability_label)

    # Add confidence interval annotation
    fig.update_layout(
        title=f"LLM Performance: Average Tokens per Second by Prompt<br><sub>Model: {model} | URL: {llm_url}<br>Error bars show ±1 standard deviation | Colors indicate reliability (CV)</sub>",
        xaxis_title="Prompt Type",
        yaxis_title="Tokens per Second",
        xaxis_tickangle=-45,
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title="Reliability (Coefficient of Variation)",
        ),
        font=dict(family="Arial, sans-serif"),
        hovermode="closest",
    )

    # Add headroom by setting y-axis range
    max_tokens = (df["tokens_per_second_mean"] + df["tokens_per_second_std"]).max()
    y_range_max = max_tokens * 1.2  # 20% headroom above the highest bar + error bar
    fig.update_yaxes(range=[0, y_range_max])

    return fig


def create_response_time_chart(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing average response times with error bars."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["prompt_name"],
            y=df["response_time_mean"],
            error_y=dict(
                type="data",
                array=df["response_time_std"],
                visible=True,
                color="rgba(0,0,0,0.3)",
            ),
            marker_color=px.colors.sequential.Plasma[: len(df)],
            text=[
                f"{mean:.1f}±{std:.1f}s"
                for mean, std in zip(df["response_time_mean"], df["response_time_std"])
            ],
            textposition="outside",
            name="Response Time",
        )
    )

    fig.update_layout(
        title="Average Response Time by Prompt<br><sub>Error bars show ±1 standard deviation</sub>",
        xaxis_title="Prompt Type",
        yaxis_title="Response Time (seconds)",
        xaxis_tickangle=-45,
        height=600,
        showlegend=False,
    )

    return fig


def create_reliability_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing reliability metrics (CV and success rate)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Coefficient of Variation (%)", "Success Rate (%)"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )

    # Create colors and track which legend items to add
    cv_colors = []
    cv_legend_items = {}
    for cv in df["tokens_per_second_cv"]:
        if cv < 20:
            color = "#28a745"  # Green
            label = "Good (CV < 20%)"
        elif cv < 30:
            color = "#ffc107"  # Orange
            label = "Moderate (20% ≤ CV < 30%)"
        else:
            color = "#dc3545"  # Red
            label = "Poor (CV ≥ 30%)"
        cv_colors.append(color)
        cv_legend_items[label] = color

    success_rates = df["successful_runs"] / df["iterations"] * 100
    sr_colors = []
    sr_legend_items = {}
    for sr in success_rates:
        if sr >= 90:
            color = "#28a745"  # Green
            label = "Excellent (≥90%)"
        elif sr >= 80:
            color = "#ffc107"  # Orange
            label = "Good (≥80%)"
        else:
            color = "#dc3545"  # Red
            label = "Poor (<80%)"
        sr_colors.append(color)
        sr_legend_items[label] = color

    # CV chart
    fig.add_trace(
        go.Bar(
            x=df["prompt_name"],
            y=df["tokens_per_second_cv"],
            marker_color=cv_colors,
            text=[f"{cv:.1f}%" for cv in df["tokens_per_second_cv"]],
            textposition="outside",
            name="CV %",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Success rate chart
    fig.add_trace(
        go.Bar(
            x=df["prompt_name"],
            y=success_rates,
            marker_color=sr_colors,
            text=[f"{sr:.0f}%" for sr in success_rates],
            textposition="outside",
            name="Success %",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Add legend items for CV
    for label, color in cv_legend_items.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=f"CV: {label}",
                showlegend=True,
                legendgroup="cv",
            )
        )

    # Add legend items for Success Rate
    for label, color in sr_legend_items.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=f"Success: {label}",
                showlegend=True,
                legendgroup="success",
            )
        )

    # Add horizontal reference lines
    fig.add_hline(
        y=20,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        row=1,
        col=1,
        annotation_text="CV=20% threshold",
        annotation_position="top right",
    )
    fig.add_hline(
        y=90,
        line_dash="dash",
        line_color="green",
        opacity=0.5,
        row=1,
        col=2,
        annotation_text="90% success threshold",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Reliability Metrics<br><sub>Lower CV = More reliable | Higher success rate = Better</sub>",
        height=700,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Color Legend",
        ),
    )

    # Update x-axis labels
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)

    return fig


def create_confidence_interval_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing 95% confidence intervals."""
    fig = go.Figure()

    # Add confidence interval bars
    for i, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["prompt_name"], row["prompt_name"]],
                y=[
                    row["tokens_per_second_95_ci_lower"],
                    row["tokens_per_second_95_ci_upper"],
                ],
                mode="lines+markers",
                line=dict(color="blue", width=3),
                marker=dict(size=8, color="blue"),
                name=f'{row["prompt_name"]} CI',
                showlegend=False,
            )
        )

        # Add mean point
        fig.add_trace(
            go.Scatter(
                x=[row["prompt_name"]],
                y=[row["tokens_per_second_mean"]],
                mode="markers",
                marker=dict(size=12, color="red", symbol="diamond"),
                name=f'{row["prompt_name"]} Mean',
                showlegend=False,
            )
        )

    fig.update_layout(
        title="95% Confidence Intervals for Tokens per Second<br><sub>Blue lines show confidence intervals, red diamonds show means</sub>",
        xaxis_title="Prompt Type",
        yaxis_title="Tokens per Second",
        xaxis_tickangle=-45,
        height=600,
    )

    return fig


def create_scatter_plot(df: pd.DataFrame) -> go.Figure:
    """Create a scatter plot showing relationship between prompt size and performance."""
    # Get model and URL info from the data
    model = df["model"].iloc[0] if "model" in df.columns else "Unknown Model"
    llm_url = df["llm_url"].iloc[0] if "llm_url" in df.columns else "Unknown URL"

    # Create color mapping based on reliability (CV)
    colors = []
    reliability_labels = []
    for cv in df["tokens_per_second_cv"]:
        if cv < 10:
            colors.append("#28a745")  # Dark green - Excellent
            reliability_labels.append("Excellent (CV < 10%)")
        elif cv < 20:
            colors.append("#20c997")  # Light green - Good
            reliability_labels.append("Good (CV < 20%)")
        elif cv < 30:
            colors.append("#ffc107")  # Orange - Moderate
            reliability_labels.append("Moderate (CV < 30%)")
        else:
            colors.append("#dc3545")  # Red - Poor
            reliability_labels.append("Poor (CV ≥ 30%)")

    fig = go.Figure()

    # Add scatter points with error bars
    fig.add_trace(
        go.Scatter(
            x=df["prompt_token_count"],
            y=df["tokens_per_second_mean"],
            error_y=dict(
                type="data",
                array=df["tokens_per_second_std"],
                visible=True,
                color="rgba(0,0,0,0.3)",
            ),
            mode="markers",
            marker=dict(
                size=df["response_time_mean"] * 2,  # Size based on response time
                color=colors,
                opacity=0.7,
                line=dict(width=1, color="black"),
                sizemode="diameter",
                sizemin=8,
            ),
            text=df["prompt_name"],
            hovertemplate="<b>%{text}</b><br>"
            + "Prompt Length: %{x} tokens<br>"
            + "Performance: %{y:.1f} ± %{error_y.array:.1f} tokens/sec<br>"
            + "CV: %{customdata[0]:.1f}%<br>"
            + "Response Time: %{customdata[1]:.1f}s<br>"
            + "Success Rate: %{customdata[2]:.0f}%<br>"
            + "<extra></extra>",
            customdata=list(
                zip(
                    df["tokens_per_second_cv"],
                    df["response_time_mean"],
                    df["successful_runs"] / df["iterations"] * 100,
                )
            ),
            name="Performance Data",
            showlegend=False,
        )
    )

    # Add invisible traces for legend
    legend_added = set()
    for i, (cv, reliability_label) in enumerate(
        zip(df["tokens_per_second_cv"], reliability_labels)
    ):
        if reliability_label not in legend_added:
            color = colors[i]
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=reliability_label,
                    showlegend=True,
                )
            )
            legend_added.add(reliability_label)

    fig.update_layout(
        title=f"Prompt Length vs Performance<br><sub>Model: {model} | URL: {llm_url}<br>Size = Response Time | Color = Reliability (CV) | Error bars = ±1 std dev</sub>",
        xaxis_title="Prompt Length (tokens)",
        yaxis_title="Tokens per Second",
        height=700,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Reliability (CV)",
        ),
        font=dict(family="Arial, sans-serif"),
        hovermode="closest",
    )

    return fig


def create_summary_stats_table(df: pd.DataFrame) -> go.Figure:
    """Create a summary statistics table."""
    overall_stats = {
        "Metric": [
            "Average Tokens/Second (Mean)",
            "Best Performance (Max Mean)",
            "Worst Performance (Min Mean)",
            "Average Coefficient of Variation",
            "Most Reliable Prompt (Lowest CV)",
            "Least Reliable Prompt (Highest CV)",
            "Overall Success Rate",
            "Total Iterations Performed",
            "Prompts with CV < 20% (Reliable)",
        ],
        "Value": [
            f"{df['tokens_per_second_mean'].mean():.2f}",
            f"{df['tokens_per_second_mean'].max():.2f} ({df.loc[df['tokens_per_second_mean'].idxmax(), 'prompt_name']})",
            f"{df['tokens_per_second_mean'].min():.2f} ({df.loc[df['tokens_per_second_mean'].idxmin(), 'prompt_name']})",
            f"{df['tokens_per_second_cv'].mean():.1f}%",
            f"{df.loc[df['tokens_per_second_cv'].idxmin(), 'prompt_name']} ({df['tokens_per_second_cv'].min():.1f}%)",
            f"{df.loc[df['tokens_per_second_cv'].idxmax(), 'prompt_name']} ({df['tokens_per_second_cv'].max():.1f}%)",
            f"{(df['successful_runs'].sum() / df['iterations'].sum() * 100):.1f}%",
            f"{df['iterations'].sum():,}",
            f"{(df['tokens_per_second_cv'] < 20).sum()}/{len(df)} ({(df['tokens_per_second_cv'] < 20).mean()*100:.0f}%)",
        ],
    }

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color="lightblue",
                    align="left",
                    font=dict(size=12, color="black"),
                ),
                cells=dict(
                    values=[overall_stats["Metric"], overall_stats["Value"]],
                    fill_color="white",
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )

    fig.update_layout(title="Statistical Performance Summary", height=400)
    return fig


def create_combined_dashboard(df: pd.DataFrame) -> str:
    """Create a combined HTML dashboard with all charts."""
    # Create individual charts
    tokens_chart = create_tokens_per_second_chart(df)
    response_chart = create_response_time_chart(df)
    reliability_chart = create_reliability_chart(df)
    ci_chart = create_confidence_interval_chart(df)
    summary_table = create_summary_stats_table(df)

    # Save individual charts as HTML divs
    tokens_html = tokens_chart.to_html(include_plotlyjs="inline", div_id="tokens-chart")
    response_html = response_chart.to_html(
        include_plotlyjs=False, div_id="response-chart"
    )
    reliability_html = reliability_chart.to_html(
        include_plotlyjs=False, div_id="reliability-chart"
    )
    ci_html = ci_chart.to_html(include_plotlyjs=False, div_id="ci-chart")
    summary_html = summary_table.to_html(include_plotlyjs=False, div_id="summary-table")

    # Generate combined HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Statistical LLM Performance Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .chart-container {{ margin: 30px 0; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; text-align: center; margin-bottom: 10px; }}
            h2 {{ color: #666; margin-top: 0; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .meta-info {{ text-align: center; color: #666; margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h1>📊 Statistical LLM Performance Dashboard</h1>
        <div class="meta-info">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Model: {df['model'].iloc[0] if 'model' in df.columns else 'Unknown'} |
            Prompts: {len(df)} | Average CV: {df['tokens_per_second_cv'].mean():.1f}%
        </div>
        
        <div class="chart-container">
            <h2>🚀 Performance: Tokens per Second</h2>
            {tokens_html.split('<body>')[1].split('</body>')[0]}
        </div>
        
        <div class="chart-container">
            <h2>⏱️ Response Times</h2>
            {response_html.split('<body>')[1].split('</body>')[0]}
        </div>
        
        <div class="chart-container">
            <h2>🎯 Reliability Metrics</h2>
            {reliability_html.split('<body>')[1].split('</body>')[0]}
        </div>
        
        <div class="chart-container">
            <h2>📈 95% Confidence Intervals</h2>
            {ci_html.split('<body>')[1].split('</body>')[0]}
        </div>
        
        <div class="chart-container">
            <h2>📋 Summary Statistics</h2>
            {summary_html.split('<body>')[1].split('</body>')[0]}
        </div>
    </body>
    </html>
    """

    # Ensure charts directory exists
    os.makedirs("charts", exist_ok=True)
    
    # Save to file
    output_file = "charts/statistical_performance_dashboard.html"
    with open(output_file, "w") as f:
        f.write(html_content)

    return output_file


def main():
    # Find the most recent statistics CSV file as the default
    default_csv = find_latest_statistics_csv()

    parser = argparse.ArgumentParser(
        description="Visualize statistical LLM performance data with Plotly"
    )
    parser.add_argument(
        "--csv",
        default=default_csv,
        help=f"CSV statistics file to plot (default: {default_csv})",
    )
    parser.add_argument(
        "--chart",
        choices=[
            "tokens",
            "response",
            "reliability",
            "confidence",
            "summary",
            "scatter",
            "all",
        ],
        default="all",
        help="Type of chart to generate (default: all)",
    )
    parser.add_argument(
        "--output",
        default="charts/statistical_llm_performance_plots.html",
        help="Output HTML file path",
    )

    args = parser.parse_args()

    # Load the data
    print(f"Loading statistical data from: {args.csv}")
    df = load_statistical_data(args.csv)

    if df is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)

    print(f"Loaded data for {len(df)} prompts")
    print(f"Average performance: {df['tokens_per_second_mean'].mean():.1f} tokens/sec")
    print(f"Average reliability (CV): {df['tokens_per_second_cv'].mean():.1f}%")

    # Generate the requested chart(s)
    if args.chart == "all":
        # Create combined dashboard
        output_file = create_combined_dashboard(df)
        print(f"\n📊 Statistical dashboard saved to: {output_file}")
    else:
        # Create individual chart
        if args.chart == "tokens":
            fig = create_tokens_per_second_chart(df)
        elif args.chart == "response":
            fig = create_response_time_chart(df)
        elif args.chart == "reliability":
            fig = create_reliability_chart(df)
        elif args.chart == "confidence":
            fig = create_confidence_interval_chart(df)
        elif args.chart == "summary":
            fig = create_summary_stats_table(df)
        elif args.chart == "scatter":
            fig = create_scatter_plot(df)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save individual chart
        fig.write_html(args.output)
        print(f"\n📊 Chart saved to: {args.output}")

    print(f"\nTo view the results, open the HTML file in your browser.")


if __name__ == "__main__":
    main()
