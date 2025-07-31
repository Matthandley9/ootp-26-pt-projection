"""
Main data processing function for imported csv values in data directory.
"""
import os
import glob
import pandas as pd

def load_all_csvs(folder_path):
    """
    Finds all CSVs in a folder and combines them into one DataFrame.

    Args:
        folder_path: The folder path to the directory containing csv imports.

    Returns:
        combined_df: A single dataframe combining all imported csv files.

    Raises:
        OSError: The csv files are not found or fail to open correctly.
    """

    pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        raise OSError(f"No CSV files found in '{folder_path}'")

    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df

def calculate_player_averages(df, player_name, stat_header, position_header):
    """
    Groups by player name and calculates the average for their stat columns.

    Args:
        df: The combined dataframe of all csvs.
        player_name: The player name to calculate the average stats of.
        stat_columns: The columns that will be averaged.
        position_header: The position that'll be grouped by. 

    Returns:
        player_averages: Dataframe containing the average of each stat for each player.

    Raises:
        RuntimeError: Dataframe is empty or misconfigured
    """
    grouping_cols = [player_name, position_header]
    required_cols = [player_name]

    if df.empty or not all(col in df.columns for col in required_cols):
        raise RuntimeError("DataFrame is empty or required columns were not found.")

    player_averages = df.groupby(grouping_cols)[stat_header].mean()

    return player_averages
