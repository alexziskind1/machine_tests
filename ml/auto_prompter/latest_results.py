#!/usr/bin/env python3
"""
Helper script to find the latest results and generate plots
"""

import os
import glob
import subprocess
import sys
from pathlib import Path


def find_latest_csv():
    """Find the most recent CSV file in the results directory."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        print("Run the performance tester first to generate results.")
        return None

    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{results_dir}' directory.")
        return None

    # Sort by modification time, most recent first
    csv_files.sort(key=os.path.getmtime, reverse=True)
    return csv_files[0]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        # List all available results
        results_dir = "results"
        if os.path.exists(results_dir):
            csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
            html_files = glob.glob(os.path.join(results_dir, "*.html"))

            print("=== Available Results ===")
            print(f"\nCSV files in {results_dir}/:")
            for csv_file in sorted(csv_files):
                size = os.path.getsize(csv_file)
                mtime = os.path.getmtime(csv_file)
                mtime_str = Path(csv_file).stat().st_mtime
                print(f"  {os.path.basename(csv_file)} ({size} bytes)")

            print(f"\nHTML files in {results_dir}/:")
            for html_file in sorted(html_files):
                print(f"  {os.path.basename(html_file)}")
        else:
            print(f"Results directory '{results_dir}' not found.")
        return

    # Find latest CSV and generate plots
    latest_csv = find_latest_csv()
    if not latest_csv:
        return

    print(f"Latest results file: {latest_csv}")

    # Generate plots from the latest results
    print("Generating plots from latest results...")
    try:
        result = subprocess.run(
            [sys.executable, "plot_results.py", "--csv", latest_csv],
            check=True,
            capture_output=True,
            text=True,
        )

        print("✓ Plots generated successfully!")
        print(f"✓ Charts saved to: results/llm_performance_plots.html")

        # Open the plots in browser
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", "results/llm_performance_plots.html"])
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", "results/llm_performance_plots.html"])
        elif sys.platform == "win32":
            subprocess.run(["start", "results/llm_performance_plots.html"], shell=True)

    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")


if __name__ == "__main__":
    main()
