#!/usr/bin/env /usr/bin/python3
"""
LLM Performance Visualization Script

Creates interactive plots using Plotly to visualize LLM performance data
from the CSV output of llm_performance_tester.py
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


def find_latest_csv() -> str:
    """Find the most recent CSV file in the results directory."""
    results_dir = "results"

    # Look for CSV files in results directory
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

    if not csv_files:
        # Fall back to current directory if no files in results
        csv_files = glob.glob("*.csv")

    if not csv_files:
        return "results/llm_performance_results.csv"  # Default fallback

    # Sort by modification time, most recent first
    csv_files.sort(key=os.path.getmtime, reverse=True)
    return csv_files[0]


def load_data(csv_file: str) -> pd.DataFrame:
    """Load and prepare the performance data."""
    try:
        df = pd.read_csv(csv_file)

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter only successful tests
        df_success = df[df["success"] == True].copy()

        if len(df_success) == 0:
            print("No successful tests found in the data!")
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
        print("Make sure you've run the performance tester first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_tokens_per_second_chart(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing tokens per second by prompt."""
    # Get model and URL info from the data
    model = df["model"].iloc[0] if "model" in df.columns else "Unknown Model"
    url = df["llm_url"].iloc[0] if "llm_url" in df.columns else "Unknown URL"

    fig = px.bar(
        df,
        x="prompt_name",
        y="tokens_per_second",
        title=f"LLM Performance: Tokens per Second by Prompt<br><sub>Model: {model} | URL: {url}</sub>",
        labels={"prompt_name": "Prompt Type", "tokens_per_second": "Tokens per Second"},
        color="tokens_per_second",
        color_continuous_scale="viridis",
    )

    # Add value labels on bars
    fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")

    # Add headroom by setting y-axis range
    max_tokens = df["tokens_per_second"].max()
    y_range_max = max_tokens * 1.2  # 20% headroom above the highest bar

    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        showlegend=False,
        # Force text rendering instead of binary data
        font=dict(family="Arial, sans-serif"),
        hovermode="closest",
        yaxis=dict(range=[0, y_range_max]),
    )

    return fig


def create_response_time_chart(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing response times by prompt."""
    fig = px.bar(
        df,
        x="prompt_name",
        y="response_time",
        title="Response Time by Prompt",
        labels={
            "prompt_name": "Prompt Type",
            "response_time": "Response Time (seconds)",
        },
        color="response_time",
        color_continuous_scale="plasma",
    )

    # Add value labels on bars
    fig.update_traces(texttemplate="%{y:.1f}s", textposition="outside")

    fig.update_layout(xaxis_tickangle=-45, height=600, showlegend=False)

    return fig


def create_token_count_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing response token counts by prompt type."""
    fig = px.bar(
        df.sort_values("response_token_count", ascending=False),
        x="prompt_name",
        y="response_token_count",
        labels={
            "prompt_name": "Prompt Type",
            "response_token_count": "Response Tokens Generated",
        },
        color="response_token_count",
        color_continuous_scale="blues",
    )

    # Add value labels on bars
    fig.update_traces(texttemplate="%{y}", textposition="outside")

    fig.update_layout(xaxis_tickangle=-45, height=600, showlegend=False)

    return fig


def create_scatter_plot(df: pd.DataFrame) -> go.Figure:
    """Create a scatter plot showing relationship between prompt size and performance."""
    # Get model and URL info from the data
    model = df["model"].iloc[0] if "model" in df.columns else "Unknown Model"
    url = df["llm_url"].iloc[0] if "llm_url" in df.columns else "Unknown URL"

    fig = px.scatter(
        df,
        x="prompt_token_count",
        y="tokens_per_second",
        size="response_token_count",
        color="prompt_name",
        hover_data={
            "prompt_token_count": True,
            "response_time": True,
            "response_token_count": True,
            "filename": True,
        },
        labels={
            "prompt_token_count": "Prompt Length (tokens)",
            "tokens_per_second": "Tokens per Second",
            "response_token_count": "Response Tokens Generated",
        },
        title=f"Prompt Token Count vs. Performance<br><sub>Model: {model} | URL: {url}</sub>",
    )

    fig.update_layout(height=600)
    return fig


def create_efficiency_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing efficiency (tokens per second per input token)."""
    df_copy = df.copy()
    df_copy["efficiency"] = df_copy["tokens_per_second"] / df_copy["prompt_token_count"]

    fig = px.bar(
        df_copy,
        x="prompt_name",
        y="efficiency",
        title="Efficiency: Tokens per Second per Input Token",
        labels={
            "prompt_name": "Prompt Type",
            "efficiency": "Efficiency (tokens/sec/token)",
        },
        color="efficiency",
        color_continuous_scale="greens",
    )

    # Add value labels on bars
    fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")

    fig.update_layout(xaxis_tickangle=-45, height=600, showlegend=False)

    return fig


def create_combined_dashboard(df: pd.DataFrame) -> go.Figure:
    """Create a combined dashboard with multiple subplots."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Tokens per Second",
            "Response Time",
            "Token Count",
            "Performance vs Prompt Length",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Tokens per second
    fig.add_trace(
        go.Bar(
            x=df["prompt_name"],
            y=df["tokens_per_second"],
            name="Tokens/sec",
            marker_color="lightblue",
        ),
        row=1,
        col=1,
    )

    # Response time
    fig.add_trace(
        go.Bar(
            x=df["prompt_name"],
            y=df["response_time"],
            name="Response Time",
            marker_color="lightcoral",
        ),
        row=1,
        col=2,
    )

    # Token count
    fig.add_trace(
        go.Bar(
            x=df["prompt_name"],
            y=df["response_token_count"],
            name="Response Token Count",
            marker_color="lightgreen",
        ),
        row=2,
        col=1,
    )

    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=df["prompt_token_count"],
            y=df["tokens_per_second"],
            mode="markers",
            marker=dict(
                size=df["response_token_count"] / 50,
                color=df["response_time"],
                colorscale="viridis",
                showscale=True,
                colorbar=dict(title="Response Time (s)"),
            ),
            text=df["prompt_name"],
            name="Performance",
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        height=800, title_text="LLM Performance Dashboard", showlegend=False
    )

    # Update x-axis labels
    for i in range(1, 3):
        for j in range(1, 3):
            if i < 2 or j < 2:  # Don't update scatter plot axes
                fig.update_xaxes(tickangle=-45, row=i, col=j)

    return fig


def create_summary_stats_table(df: pd.DataFrame) -> go.Figure:
    """Create a summary statistics table."""
    stats = {
        "Metric": [
            "Average Tokens/Second",
            "Max Tokens/Second",
            "Min Tokens/Second",
            "Average Response Time (s)",
            "Max Response Time (s)",
            "Min Response Time (s)",
            "Average Response Token Count",
            "Total Response Tokens Generated",
            "Total Test Time (s)",
        ],
        "Value": [
            f"{df['tokens_per_second'].mean():.2f}",
            f"{df['tokens_per_second'].max():.2f}",
            f"{df['tokens_per_second'].min():.2f}",
            f"{df['response_time'].mean():.2f}",
            f"{df['response_time'].max():.2f}",
            f"{df['response_time'].min():.2f}",
            f"{df['response_token_count'].mean():.0f}",
            f"{df['response_token_count'].sum():,}",
            f"{df['response_time'].sum():.2f}",
        ],
    }

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Value"], fill_color="paleturquoise", align="left"
                ),
                cells=dict(
                    values=[stats["Metric"], stats["Value"]],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(title="Performance Summary Statistics", height=400)

    return fig


def main():
    # Find the most recent CSV file as the default
    default_csv = find_latest_csv()

    parser = argparse.ArgumentParser(
        description="Visualize LLM performance data with Plotly"
    )
    parser.add_argument(
        "--csv",
        default=default_csv,
        help=f"CSV file to visualize (default: {default_csv})",
    )
    parser.add_argument(
        "--output",
        default="results/llm_performance_plots.html",
        help="Output HTML file (default: results/llm_performance_plots.html)",
    )
    parser.add_argument(
        "--chart",
        choices=[
            "tokens",
            "time",
            "count",
            "scatter",
            "efficiency",
            "dashboard",
            "stats",
            "all",
        ],
        default="tokens",
        help="Type of chart to generate (default: tokens)",
    )

    # Check if no arguments provided
    if len(sys.argv) == 1:
        print("📊 LLM Performance Visualization Tool")
        print("=" * 40)
        print(
            "\nThis tool creates interactive charts from LLM performance test results."
        )
        print("\nUsage examples:")
        print("  python3 plot_results.py --csv results/latest_results.csv")
        print("  python3 plot_results.py --chart all")
        print("  python3 plot_results.py --chart dashboard --output my_charts.html")
        print("\nAvailable chart types:")
        print("  tokens     - Tokens per second performance")
        print("  time       - Response time analysis")
        print("  count      - Token count distribution")
        print("  scatter    - Scatter plot of performance metrics")
        print("  efficiency - Efficiency analysis")
        print("  dashboard  - Combined dashboard view")
        print("  stats      - Summary statistics table")
        print("  all        - Generate all chart types")
        print("\nFor detailed help:")
        print("  python3 plot_results.py --help")
        print(
            "\nTip: Use 'python3 latest_results.py' to automatically visualize the most recent results!"
        )
        return 0

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.csv}...")
    df = load_data(args.csv)

    if df is None:
        return 1

    print(f"Loaded {len(df)} successful test results")

    # Generate charts based on selection
    figures = []

    if args.chart == "all" or args.chart == "dashboard":
        print("Creating dashboard...")
        figures.append(create_combined_dashboard(df))

    if args.chart == "all" or args.chart == "stats":
        print("Creating summary statistics...")
        figures.append(create_summary_stats_table(df))

    if args.chart == "all" or args.chart == "tokens":
        print("Creating tokens per second chart...")
        figures.append(create_tokens_per_second_chart(df))

    if args.chart == "all" or args.chart == "time":
        print("Creating response time chart...")
        figures.append(create_response_time_chart(df))

    if args.chart == "all" or args.chart == "count":
        print("Creating token count chart...")
        figures.append(create_token_count_chart(df))

    if args.chart == "all" or args.chart == "scatter":
        print("Creating scatter plot...")
        figures.append(create_scatter_plot(df))

    if args.chart == "all" or args.chart == "efficiency":
        print("Creating efficiency chart...")
        figures.append(create_efficiency_chart(df))

    # Save all figures to HTML
    if figures:
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        print(f"Saving plots to {args.output}...")

        # For single chart, use Plotly's built-in HTML export with config
        if len(figures) == 1:
            config = {
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
            }
            figures[0].write_html(
                args.output, config=config, include_plotlyjs=True, div_id="chart"
            )
        else:
            # Create HTML content for multiple charts
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LLM Performance Analysis</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .chart-container {{ margin: 30px 0; }}
                    h1 {{ color: #333; text-align: center; }}
                    .info {{ background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>LLM Performance Analysis Report</h1>
                <div class="info">
                    <strong>Model:</strong> {df['model'].iloc[0]}<br>
                    <strong>Tests Run:</strong> {len(df)}<br>
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            """

            # Add each figure as a div
            for i, fig in enumerate(figures):
                div_id = f"chart_{i}"
                html_content += f'<div class="chart-container" id="{div_id}"></div>\n'

                # Add JavaScript to render the plot
                plot_json = fig.to_json()
                html_content += f"""
                <script>
                    var plotData_{i} = {plot_json};
                    Plotly.newPlot('{div_id}', plotData_{i}.data, plotData_{i}.layout);
                </script>
                """

            html_content += """
            </body>
            </html>
            """

            # Write to file
            with open(args.output, "w") as f:
                f.write(html_content)

        print(f"✓ Plots saved to {args.output}")
        print(f"✓ Open the file in your browser to view the interactive charts")

        # Also show individual plots if requested
        if args.chart != "all":
            figures[0].show()

    return 0


if __name__ == "__main__":
    exit(main())
