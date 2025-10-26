"""
Calculates different values using imported data from csvs in data directory
"""
from data_processing import (
    load_all_csvs,
    aggregate_rate_by_player_pos_vlvl,
    top_by_position,
)


def main():
    """Main function to run the data analysis workflow."""

    DATA_FOLDER = 'data'

    PLAYER_NAME_COLUMN = 'Title'
    POSITION_COLUMN = 'POS'
    VLVL_COLUMN = 'VLvl'

    MIN_PA = 600
    MIN_IP = 200
    TOP_N_BAT = 5
    TOP_N_PIT = 10

    print("Loading all CSV files...")
    combined_data = load_all_csvs(DATA_FOLDER)

    if combined_data.empty:
        print("No data loaded.")
        return

    print("Calculating batter data...")
    agg_batters = aggregate_rate_by_player_pos_vlvl(
        combined_data,
        player_col=PLAYER_NAME_COLUMN,
        pos_col=POSITION_COLUMN,
        vlvl_col=VLVL_COLUMN,
        numerator_col='WAR',
        denominator_col='PA',
        scale=600.0,
        numerator_out_name='WAR',
        denominator_out_name='PA',
        rate_out_name='WAR_per_600_PA',
    )

    top_batters = top_by_position(
        agg_batters,
        denom_col='PA',
        min_denom=MIN_PA,
        rate_col='WAR_per_600_PA',
        top_n=TOP_N_BAT,
        player_col=PLAYER_NAME_COLUMN,
        pos_col=POSITION_COLUMN,
        vlvl_col=VLVL_COLUMN,
    )

    print("Calculating pitcher data...")
    agg_pitchers = aggregate_rate_by_player_pos_vlvl(
        combined_data,
        player_col=PLAYER_NAME_COLUMN,
        pos_col=POSITION_COLUMN,
        vlvl_col=VLVL_COLUMN,
        numerator_col='rWAR',
        denominator_col='IP',
        scale=200.0,
        numerator_out_name='rWAR',
        denominator_out_name='IP',
        rate_out_name='rWAR_per_200_IP',
    )

    top_pitchers = top_by_position(
        agg_pitchers,
        denom_col='IP',
        min_denom=MIN_IP,
        rate_col='rWAR_per_200_IP',
        top_n=TOP_N_PIT,
        player_col=PLAYER_NAME_COLUMN,
        pos_col=POSITION_COLUMN,
        vlvl_col=VLVL_COLUMN,
    )

    if not top_batters:
        print(f"No batters with at least {MIN_PA} PA found.")
    else:
        print(f"\nTop {TOP_N_BAT} batters by WAR per 600 PA "
              f"(min {MIN_PA} PA) by position and VLvl:\n")
        for pos, pos_df in top_batters.items():
            print(f"Position: {pos}")
            print(pos_df.to_string(index=False, float_format='{:,.3f}'.format))
            print()

    if not top_pitchers:
        print(f"No pitchers with at least {MIN_IP} IP found.")
    else:
        print(f"\nTop {TOP_N_PIT} pitchers by rWAR per 200 IP "
              f"(min {MIN_IP} IP) by position and VLvl:\n")
        for pos, pos_df in top_pitchers.items():
            print(f"Position: {pos}")
            print(pos_df.to_string(index=False, float_format='{:,.3f}'.format))
            print()


if __name__ == "__main__":
    main()
