import os
import pandas as pd


filename_to_model = {
    "m1pro14": "MacBook Pro 14 M1",
    "m2max16": "MacBook Pro 16 M2 Max",
    "corei9_mbp2019": "MacBook Pro 19 Core i9",
    "xelite_galaxybook4edge": "Samsung Galaxy Book 4 Edge X Elite",
    "coreU7_155H_xps13": "Dell XPS 13 Core Ultra 7 155H",
    "xelite_vivobooks15": "Asus VivoBook S15 X Elite",
    "m2mba13": "MacBook Air 13 M2",
    "corei5_2017mba": "MacBook Air 2017 Core i5",
    "xelite_surface_laptop7": "Microsoft Surface Laptop 7 X Elite",
    "m2max16_2": "MacBook Pro 16 M2 Max (2)",
    "m1mba": "MacBook Air M1",
    "m3mbp": "MacBook Pro M3",
    "xplus_surface_laptop7": "Microsoft Surface Laptop 7 X Plus",
    "xelite_xps13": "Dell XPS 13 X Elite",
    "m3mba": "MacBook Air 13 M3",
    "m3pro14": "MacBook Pro 14 M3 Pro",
    "m3max16": "MacBook Pro 16 M3 Max",
    "coreU7_165H_surface_laptop6": "Microsoft Surface Laptop 6 Core Ultra 7 165H",
}


def clean_filename(file_name, filename_to_model):
    cleaned_name = file_name.replace("batlog_", "").replace(".txt", "")
    return filename_to_model.get(
        cleaned_name, cleaned_name
    )  # Default to cleaned_name if not found


def clean_battery(battery_value):
    return battery_value.split()[1]


def split_cpu(cpu_value):
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
    try:
        # Attempt to split the string and extract the second memory value
        return memory_value.split(":")[1].strip().split()[1]
    except IndexError:
        # If splitting fails or parts are missing, return 'Unknown'
        return "Unknown"


def extract_iterations(iteration_string):
    return int(iteration_string.split(":")[1].strip().split()[0])


def read_data(file_path, columns, file_name):
    data = pd.read_csv(file_path, names=columns, skiprows=1)
    data.insert(
        0, "Model", clean_filename(file_name, filename_to_model)
    )  # Use 'Model' instead of 'Filename'
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
    data_frames = []
    for filename in os.listdir(directory):
        if filename.endswith(file_extension):
            file_path = os.path.join(directory, filename)
            data_frames.append(read_data(file_path, columns, filename))
    return pd.concat(data_frames, ignore_index=True)


def save_data(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Data combined and saved to {output_file}")


def main():
    directory = "../batloop_logs"
    output_file = "combined_data.csv"
    columns = ["Timestamp", "Battery", "CPU", "Memory", "Work Done", "Total Work Done"]
    combined_data = combine_data(directory, ".txt", columns)
    save_data(combined_data, output_file)


if __name__ == "__main__":
    main()
