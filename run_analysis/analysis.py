import pandas as pd
import os
import glob

def csv_to_dataframe(csv_file):
    """
    This function takes a CSV file as input and returns a pandas DataFrame.
    
    Parameters:
    csv_file (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None


def main():
    # Find all CSV files in the current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    # Dictionary to store all dataframes
    dataframes = {}
    
    # Convert CSV files to dataframes
    print("Converting CSV files to dataframes...")
    for csv_file in csv_files:
        df = csv_to_dataframe(csv_file)
        if df is not None:
            dataframes[csv_file] = df
            print(f"  Loaded: {csv_file} ({len(df)} rows)")
    
    all_means = {}
    
    for filename, df in dataframes.items():
        # Skip the all_metrics_means.csv file
        if filename == "all_metrics_means_gen_30.csv":
            continue
        
        # Each 30-th generation (by index position)
        generation_30 = df.loc[29]
        all_means[filename] = generation_30

    # Create a dataframe from all means and save to CSV
    all_means_df = pd.DataFrame(all_means).T

    all_means_df.drop(["file_path"], axis=1, inplace=True, errors='ignore')

    # Convert all columns to numeric (except non-numeric ones)
    for col in all_means_df.columns:
        all_means_df[col] = pd.to_numeric(all_means_df[col], errors='coerce')

    # Calculate mean of numeric columns
    mean = all_means_df.mean(numeric_only=True)

    all_means_df.loc["mean"] = mean

    # Calculate standard deviation of numeric columns
    std = all_means_df.iloc[:-1].std(numeric_only=True)

    all_means_df.loc["std"] = std

    all_means_df.to_csv("all_metrics_means_gen_30.csv")
    print(f"\n✓ Saved to all_metrics_means_gen_30.csv")

    # For pareto front analysis, save only specific columns
    pareto_column = pd.DataFrame(all_means_df[["pareto_front_area"]])

    pareto_column.to_csv("pareto_front_area_gen_30.csv")
    print(f"\n✓ Saved to pareto_front_area_gen_30.csv")

if __name__ == "__main__":
    main()
