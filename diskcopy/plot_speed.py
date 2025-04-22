import pandas as pd
import plotly.express as px
import numpy as np

# === Load CSV ===
df = pd.read_csv("dd_speed_log.csv")

# === Initial Data Prep ===
# Convert 'bytes' to numeric, coercing errors to NaN
df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce")

# Drop rows where 'bytes' is now NaN (handles the initial blank read entry)
df = df.dropna(subset=["bytes"])

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

# Ensure uniqueness: Keep only the last entry for each timestamp within each operation
df = df.groupby(["operation", "timestamp"]).last().reset_index()

df = df.sort_values(by=["operation", "timestamp"])

# Convert bytes to MB
df["mb"] = df["bytes"] / 1024 / 1024

# === Resample and Calculate Average Speed ===
resampled_data = []
for op, group in df.groupby("operation"):
    group = group.set_index("timestamp")

    # Find the actual start time for this operation
    start_time = group.index.min()

    # Resample to 1-second frequency and interpolate MB
    # 'ffill' handles potential gaps if no data point exists exactly at the second mark
    resampled_group = (
        group["mb"].resample("1s").ffill().interpolate()
    )  # Changed '1S' to '1s'

    # Calculate MB transferred per second (average MB/s for the interval)
    mbps = resampled_group.diff()

    # Create a DataFrame for the results
    result_df = pd.DataFrame({"mbps": mbps})
    result_df["operation"] = op

    # Calculate time elapsed in seconds from the start of the operation
    result_df["time_elapsed"] = (result_df.index - start_time).total_seconds()

    resampled_data.append(result_df)

# Combine resampled data
df_resampled = pd.concat(resampled_data).reset_index(drop=True)

# === Clean: remove initial NaN from diff and any invalid/zero/negative MB/s ===
df_resampled = df_resampled.dropna(subset=["mbps"])
df_resampled = df_resampled[df_resampled["mbps"] > 0]

# === Plot ===
fig = px.line(
    df_resampled,
    x="time_elapsed",
    y="mbps",
    color="operation",
    title="Thunderbolt Dock Speed Test (Average MB/s per Second)",  # Updated title
    labels={"time_elapsed": "Seconds", "mbps": "Average MB/s"},  # Updated label
)

fig.update_traces(mode="lines")
fig.write_html("speed_plot.html")
fig.write_image("speed_plot.png", width=1200, height=600)
fig.show()
