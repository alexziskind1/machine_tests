# The chart shows the average efficiency of different laptop models,
# where efficiency is calculated as the ratio of work done to power consumed over time,
# allowing you to quickly compare each model's performance based on how
# efficiently it uses its battery power.

import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file in the current directory
file_path = "combined_data.csv"
df = pd.read_csv(file_path)

# Convert 'Timestamp' to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Convert Battery Size to numeric (if it's not already)
df["Battery Size"] = pd.to_numeric(df["Battery Size"], errors="coerce")

# Calculate power consumed in mAh over each interval
df["Power Consumed (mAh)"] = (
    df["Battery Size"] * (df["Battery"].shift(1) - df["Battery"]) / 100
)

# Calculate average work done over each interval
df["Average Work Done"] = (df["Work Done"] + df["Work Done"].shift(1)) / 2

# Display rows with NaN values before dropping them to understand the filtering
print("Rows with NaN values:")
print(df[df.isna().any(axis=1)])

# Drop the first row since it has NaN values due to shifting
df = df.dropna()

# Ensure no rows have zero power consumption to avoid division by zero
df = df[df["Power Consumed (mAh)"] != 0]

# Calculate efficiency
df["Efficiency"] = df["Average Work Done"] / df["Power Consumed (mAh)"]

# Group by model and calculate the average efficiency for each group
average_efficiency = df.groupby("Model")["Efficiency"].mean().reset_index()

# Display the DataFrame showing the average efficiency for each model
print(average_efficiency)

# Save the results to a new CSV file in the current directory
average_efficiency.to_csv("average_laptop_efficiency_by_model.csv", index=False)

# Plotting the average efficiency for each model
plt.figure(figsize=(12, 8))
plt.barh(average_efficiency["Model"], average_efficiency["Efficiency"], color="skyblue")
plt.xlabel("Average Efficiency")
plt.title("Average Efficiency by Laptop Model")
plt.grid(axis="x")

# Save the plot as an image file
plt.savefig("average_laptop_efficiency_by_model.png")

# Show the plot
plt.show()
