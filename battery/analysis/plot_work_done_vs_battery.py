import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_and_clean_data(filepath):
    """Loads the data from a CSV file and cleans it."""
    df = pd.read_csv(filepath)
    # Ensure 'Battery' is treated as a string and handle missing or malformed entries
    df["Battery"] = df["Battery"].astype(str).str.rstrip("%").replace("", np.nan)
    # Convert battery percentages from string to numeric, handling non-numeric entries
    df["Battery"] = pd.to_numeric(df["Battery"], errors="coerce") / 100.0

    # Convert necessary columns to numeric and handle errors
    columns_to_numeric = [
        "Memory",
        "Work Done",
        "Total Work Done",
        "CPU1",
        "CPU2",
    ]
    for column in columns_to_numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Drop rows where any of these specific columns have missing values
    df.dropna(subset=["Battery", "Work Done"], inplace=True)
    return df


def plot_battery_vs_work_done(df):
    """Plots Battery Drain vs. Work Done from the dataframe, with different colors for each model."""
    plt.figure(figsize=(10, 6))

    # Generate a color map based on the unique models
    models = df["Model"].unique()
    # Ensure only up to 20 unique colors are used
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(models)))

    # Plot each model with a different color
    for model, color in zip(models, colors):
        subset = df[df["Model"] == model]
        plt.scatter(
            subset["Work Done"],
            subset["Battery"],
            alpha=0.5,
            color=color,
            label=model,  # Use model name for label
            s=50,  # Increase scatter size for better visibility
        )

    plt.title("Battery Drain vs. Work Done by Model")
    plt.xlabel("Work Done (units)")
    plt.ylabel("Battery Drain (%)")
    plt.grid(True)

    # Place legend to the right of the plot
    plt.legend(title="Model", loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust layout to make room for the legend
    plt.tight_layout(
        rect=[0, 0, 0.85, 1]
    )  # Adjust the right margin to give more space for the legend

    plt.show()


# Usage
if __name__ == "__main__":
    filepath = "combined_data.csv"  # Change to your actual file path
    df = load_and_clean_data(filepath)
    if not df.empty:
        plot_battery_vs_work_done(df)
    else:
        print("No data available to plot.")
