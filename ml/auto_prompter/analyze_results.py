#!/usr/bin/env python3
"""
Simple analysis script for LLM performance results.
Creates basic visualizations from the CSV output.
"""

import csv
import argparse
import sys
from datetime import datetime


def analyze_results(csv_file: str) -> None:
    """Analyze and display basic statistics from the results CSV."""

    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            results = list(reader)
    except FileNotFoundError:
        print(f"Error: Could not find results file '{csv_file}'")
        print("Make sure you've run the performance tester first.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not results:
        print("No results found in the CSV file.")
        return

    print("=== LLM Performance Analysis ===\n")

    # Basic statistics
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get("success", "").lower() == "true")

    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%\n")

    if successful_tests == 0:
        print("No successful tests to analyze.")
        return

    # Performance metrics for successful tests
    successful_results = [r for r in results if r.get("success", "").lower() == "true"]

    # Tokens per second analysis
    tokens_per_sec = []
    response_times = []
    token_counts = []
    prompt_lengths = []

    for result in successful_results:
        try:
            tps = float(result.get("tokens_per_second", 0))
            rt = float(result.get("response_time", 0))
            tc = int(result.get("token_count", 0))
            pl = int(result.get("prompt_length", 0))

            tokens_per_sec.append(tps)
            response_times.append(rt)
            token_counts.append(tc)
            prompt_lengths.append(pl)
        except (ValueError, TypeError):
            continue

    if tokens_per_sec:
        print("Performance Metrics:")
        print(f"  Average tokens/second: {sum(tokens_per_sec)/len(tokens_per_sec):.2f}")
        print(f"  Max tokens/second: {max(tokens_per_sec):.2f}")
        print(f"  Min tokens/second: {min(tokens_per_sec):.2f}")
        print()

    if response_times:
        print("Response Times:")
        print(
            f"  Average response time: {sum(response_times)/len(response_times):.2f}s"
        )
        print(f"  Max response time: {max(response_times):.2f}s")
        print(f"  Min response time: {min(response_times):.2f}s")
        print()

    if token_counts:
        print("Token Counts:")
        print(f"  Average tokens generated: {sum(token_counts)/len(token_counts):.0f}")
        print(f"  Max tokens generated: {max(token_counts)}")
        print(f"  Min tokens generated: {min(token_counts)}")
        print()

    # Per-prompt breakdown
    print("Results by Prompt:")
    print("-" * 70)
    print(
        f"{'Filename':<25} {'Tokens/sec':<12} {'Time(s)':<10} {'Tokens':<8} {'Status'}"
    )
    print("-" * 70)

    for result in results:
        filename = result.get("filename", "Unknown")[:24]
        tps = result.get("tokens_per_second", "0")
        rt = result.get("response_time", "0")
        tc = result.get("token_count", "0")
        success = result.get("success", "false")

        try:
            tps_val = float(tps)
            rt_val = float(rt)
            tc_val = int(float(tc))
            status = "✓" if success.lower() == "true" else "✗"

            print(
                f"{filename:<25} {tps_val:<12.2f} {rt_val:<10.2f} {tc_val:<8} {status}"
            )
        except (ValueError, TypeError):
            print(f"{filename:<25} {'Error':<12} {'Error':<10} {'Error':<8} ✗")

    print("-" * 70)

    # Show model and configuration info
    if results:
        first_result = results[0]
        model = first_result.get("model", "Unknown")
        url = first_result.get("llm_url", "Unknown")
        print(f"\nModel: {model}")
        print(f"URL: {url}")

        # Show timestamp range
        timestamps = [r.get("timestamp", "") for r in results if r.get("timestamp")]
        if timestamps:
            print(f"Test period: {min(timestamps)} to {max(timestamps)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM performance test results")
    parser.add_argument(
        "--csv",
        default="llm_performance_results.csv",
        help="CSV file to analyze (default: llm_performance_results.csv)",
    )

    args = parser.parse_args()
    analyze_results(args.csv)


if __name__ == "__main__":
    main()
