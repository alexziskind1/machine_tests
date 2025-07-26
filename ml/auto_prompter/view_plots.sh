#!/bin/bash
# Simple script to view the LLM performance plots in the browser

PLOT_FILE="results/llm_performance_plots.html"

if [ -f "$PLOT_FILE" ]; then
    echo "Opening $PLOT_FILE in your default browser..."
    open "$PLOT_FILE"
else
    echo "Plot file not found. Please run the plotting script first:"
    echo "  /usr/bin/python3 plot_results.py"
fi
