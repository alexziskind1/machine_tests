import pandas as pd
import glob
import os

def combine_benchmark_csvs(path_pattern, output_filename):
    """
    Combines multiple benchmark CSV files into a single DataFrame and saves it.

    Args:
        path_pattern (str): Glob pattern to find the input CSV files.
        output_filename (str): Name of the CSV file to save the combined data.
    """
    all_files = glob.glob(path_pattern)
    
    if not all_files:
        print(f"No files found matching pattern: {path_pattern}")
        return

    df_list = []
    print(f"Found {len(all_files)} files to combine.")

    for f_idx, f_path in enumerate(all_files):
        try:
            df = pd.read_csv(f_path)
            print(f"Reading file {f_idx + 1}/{len(all_files)}: {os.path.basename(f_path)} - {df.shape[0]} rows")
            
            # Standardize column name for model
            if 'model' in df.columns and 'model_name' not in df.columns:
                df.rename(columns={'model': 'model_name'}, inplace=True)
            
            # Add a column for the source filename to trace data back if needed
            df['source_file'] = os.path.basename(f_path)
            
            df_list.append(df)
        except Exception as e:
            print(f"Error reading or processing file {f_path}: {e}")
            
    if not df_list:
        print("No dataframes were successfully loaded. Exiting.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Basic data cleaning: ensure model_name is a string and clean up potential path issues
    if 'model_name' in combined_df.columns:
        combined_df['model_name'] = combined_df['model_name'].astype(str).apply(lambda x: x.split('/')[-1])

    # Reorder columns to put 'source_file' early for easier inspection
    if 'source_file' in combined_df.columns:
        cols = ['source_file'] + [col for col in combined_df.columns if col != 'source_file']
        combined_df = combined_df[cols]

    try:
        combined_df.to_csv(output_filename, index=False)
        print(f"Successfully combined {len(df_list)} files into {output_filename}")
        print(f"Combined DataFrame shape: {combined_df.shape}")
    except Exception as e:
        print(f"Error writing combined data to {output_filename}: {e}")

def main():
    # Assuming the script is in the same directory as the CSVs
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Pattern to match benchmark CSV files, excluding the potential output file itself
    # This helps prevent reading the output file if the script is run multiple times.
    input_csv_pattern = os.path.join(base_path, 'benchmark_*.csv')
    output_csv_file = os.path.join(base_path, 'combined_benchmark_results.csv')
    
    print(f"Looking for input CSV files in: {base_path}")
    print(f"Output will be saved to: {output_csv_file}")
    
    combine_benchmark_csvs(input_csv_pattern, output_csv_file)

if __name__ == '__main__':
    main()
