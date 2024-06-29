import os
import pandas as pd


def clean_filename(file_name):
    """
    Cleans the filename by removing 'batlog_' prefix and '.txt' suffix.

    Parameters:
    file_name (str): The original filename.

    Returns:
    str: The cleaned filename.
    """
    return file_name.replace("batlog_", "").replace(".txt", "")


def clean_battery(battery_value):
    """
    Cleans the battery percentage value to remove unnecessary details.

    Parameters:
    battery_value (str): The original battery string.

    Returns:
    str: The cleaned battery percentage value.
    """
    return battery_value.split()[1]


def split_cpu(cpu_value):
    """
    Splits the CPU usage into two separate columns for each percentage.
    Returns 'Unknown' for either or both values if data is incomplete or malformed.

    Parameters:
    cpu_value (str): The original CPU string.

    Returns:
    tuple: The first and second CPU usage percentages, or 'Unknown' if data is malformed.
    """
    try:
        # Attempt to split the string and extract both CPU percentages
        percentages = cpu_value.split(":")[1].strip().split()
        # Ensure that two percentages are extracted
        if len(percentages) >= 2:
            return percentages[0], percentages[1]
        else:
            # Return 'Unknown' for missing percentages
            return "Unknown", "Unknown"
    except IndexError:
        # If splitting fails or parts are missing, return 'Unknown' for both
        return "Unknown", "Unknown"


def clean_memory(memory_value):
    """
    Cleans the memory usage to only keep the second value.

    Parameters:
    memory_value (str): The original memory string.

    Returns:
    str: The second memory value, or 'Unknown' if data is malformed.
    """
    try:
        # Attempt to split the string and extract the second memory value
        return memory_value.split(":")[1].strip().split()[1]
    except IndexError:
        # If splitting fails or parts are missing, return 'Unknown'
        return "Unknown"


def extract_iterations(iteration_string):
    """
    Extracts the integer number of iterations.

    Parameters:
    iteration_string (str): The original iteration string.

    Returns:
    int: The number of iterations.
    """
    return int(iteration_string.split(":")[1].strip().split()[0])


def read_data(file_path, columns, file_name):
    """
    Reads a single file, cleans necessary fields, and returns a DataFrame with specified columns.
    Adds and cleans the filename as the first column and cleans other specified fields.
    """
    data = pd.read_csv(file_path, names=columns, skiprows=1)
    data.insert(0, "Filename", clean_filename(file_name))
    data["Battery"] = data["Battery"].apply(clean_battery)
    data[["CPU1", "CPU2"]] = data.apply(
        lambda x: split_cpu(x["CPU"]), axis=1, result_type="expand"
    )
    data["Memory"] = data["Memory"].apply(clean_memory)
    data["Work Done"] = data["Work Done"].apply(extract_iterations)
    data["Total Work Done"] = data["Total Work Done"].apply(extract_iterations)
    data.drop(
        columns=["CPU"], inplace=True
    )  # Remove original 'CPU' column since it's no longer needed
    return data


def combine_data(directory, file_extension, columns):
    """
    Combines and cleans data from all files in a directory that have a specific file extension into a single DataFrame.

    Parameters:
    directory (str): The directory to search for files.
    file_extension (str): The file extension to look for.
    columns (list): The list of column names for the DataFrames, without 'Filename'.

    Returns:
    pd.DataFrame: The combined and cleaned DataFrame from all files.
    """
    data_frames = []
    for filename in os.listdir(directory):
        if filename.endswith(file_extension):
            file_path = os.path.join(directory, filename)
            data_frames.append(read_data(file_path, columns, filename))
    return pd.concat(data_frames, ignore_index=True)


def save_data(df, output_file):
    """
    Saves the DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    output_file (str): The path to the output file where the data should be saved.
    """
    df.to_csv(output_file, index=False)
    print(f"Data combined and saved to {output_file}")


def main():
    directory = "batloop_logs"
    output_file = "combined_data.csv"
    columns = ["Timestamp", "Battery", "CPU", "Memory", "Work Done", "Total Work Done"]
    combined_data = combine_data(directory, ".txt", columns)
    save_data(combined_data, output_file)


if __name__ == "__main__":
    main()
