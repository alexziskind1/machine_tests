#!/bin/bash
# View Statistical LLM Performance Plots
# Opens the statistical performance dashboard in the default browser

echo "📊 Opening Statistical LLM Performance Dashboard..."

# Check if the dashboard exists
if [ -f "results/statistical_performance_dashboard.html" ]; then
    echo "✅ Found statistical dashboard"
    open "results/statistical_performance_dashboard.html"
else
    echo "❌ Statistical dashboard not found"
    echo "💡 Run: python plot_statistical_results.py"
    echo "   to generate the dashboard first"
fi

# Also check for individual chart files
if [ -f "results/statistical_llm_performance_plots.html" ]; then
    echo "✅ Individual chart file also available"
    echo "🌐 You can also view: results/statistical_llm_performance_plots.html"
fi

echo "🔄 To regenerate dashboard: python plot_statistical_results.py"
echo "📈 To generate specific chart: python plot_statistical_results.py --chart [tokens|response|reliability|confidence|summary]"
