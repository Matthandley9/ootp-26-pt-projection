"""
Calculates different values using imported data from csvs in data directory
"""
from data_processing import load_all_csvs, calculate_player_averages

def main():
    """Main function to run the data analysis workflow."""

    DATA_FOLDER = 'data'

    # --- Define your column names as strings here ---
    PLAYER_NAME_COLUMN = 'Name'
    POSITION_COLUMN = 'POS'
    STATS_TO_AVERAGE = ['AVG', 'OBP', 'SLG', 'wRC+', 'WAR']

    # 1. Load data
    print("Loading all CSV files...")
    combined_data = load_all_csvs(DATA_FOLDER)

    # 2. Calculate averages
    if not combined_data.empty:
        print("Calculating player data...")
        # --- Pass the variables into the function ---
        player_summary = calculate_player_averages(
            combined_data,
            PLAYER_NAME_COLUMN,
            STATS_TO_AVERAGE,
            POSITION_COLUMN,
        )

        if player_summary is not None:
            print("\n--- Player Summary (Stats and Position) ---")
            print(player_summary)
            print("-------------------------------------------")

if __name__ == "__main__":
    main()
