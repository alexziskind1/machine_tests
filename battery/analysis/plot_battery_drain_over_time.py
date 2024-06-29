import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def read_data(filepath):
    """Reads the CSV data from the specified filepath."""
    data = pd.read_csv(filepath)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    # Assume Battery is already an integer, so no need for conversion
    return data


def process_data(data):
    """Processes data to compute the minutes from the start for each model and handle duplicates."""
    data["Minutes"] = data.groupby("Model")["Timestamp"].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 60
    )
    data["Minutes"] = (
        data["Minutes"].round().astype(int)
    )  # Round to nearest whole number

    # Average battery percentages for duplicate minute values per model
    data = data.groupby(["Model", "Minutes"]).agg({"Battery": "mean"}).reset_index()
    data["Battery"] = (
        data["Battery"].round().astype(int)
    )  # Ensure Battery is an integer after averaging
    return data


def plot_battery_drain(data):
    """Plots battery percentage over time (in minutes) for each model using customized colors."""
    fig, ax = plt.subplots(figsize=(12, 8))
    max_duration = data["Minutes"].max()
    x_ticks = range(0, max_duration + 1, 15)  # X ticks every 15 minutes

    # Get colors for each model
    num_models = data["Model"].nunique()
    colors = plt.get_cmap("tab20")(
        np.linspace(0, 1, min(num_models, 20))
    )  # Use only up to 20 unique colors

    for (model, group_data), color in zip(data.groupby("Model"), colors):
        X = group_data["Minutes"]
        Y = group_data["Battery"]
        X_smooth = np.linspace(
            X.min(), X.max(), 500
        )  # Increase number of points for smoother line
        spline = make_interp_spline(
            X, Y, k=3
        )  # You can try k=4 or k=5 for a higher-order spline
        Y_smooth = spline(X_smooth)

        # Clamping values to ensure they do not go below zero
        Y_smooth = np.clip(Y_smooth, 0, None)

        ax.plot(
            X_smooth, Y_smooth, label=model, linestyle="-", linewidth=2, color=color
        )

    plt.xticks(x_ticks)
    plt.xlabel("Minutes from Start")
    plt.ylabel("Battery Percentage")
    plt.title("Battery Drain Over Time by Model")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def main():
    filepath = "combined_data.csv"  # Update this to the path of your CSV file
    data = read_data(filepath)
    data = process_data(data)
    plot_battery_drain(data)


if __name__ == "__main__":
    main()
