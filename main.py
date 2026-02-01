"""
Calculates different values using imported data from csvs in data directory
"""
from data_processing import (
    load_all_csvs,
    aggregate_rate_by_player_pos_vlvl,
    top_by_position,
    # predictor/plot helpers are available in data_processing if needed
    PlayerRateSpec,
    run_multiple_vlr_regressions,
)

DATA_FOLDER = 'data'

PLAYER_NAME_COLUMN = 'Title'
POSITION_COLUMN = 'POS'
VLVL_COLUMN = 'VLvl'

MIN_PA = 600
MIN_IP = 200
TOP_N_BAT = 5
TOP_N_PIT = 10

def find_top_players(combined_data, batter_spec):
    """Find and print top batters and pitchers by specified WAR rates."""
    print("Calculating batter data...")
    agg_batters = aggregate_rate_by_player_pos_vlvl(combined_data, batter_spec)

    top_batters = top_by_position(
        agg_batters,
        batter_spec,
        min_denom=MIN_PA,
        top_n=TOP_N_BAT,
    )

    pitcher_spec = PlayerRateSpec(
        player_col=PLAYER_NAME_COLUMN,
        pos_col=POSITION_COLUMN,
        vlvl_col=VLVL_COLUMN,
        numerator_col='rWAR',
        denominator_col='IP',
        scale=200.0,
    )

    print("Calculating pitcher data...")
    agg_pitchers = aggregate_rate_by_player_pos_vlvl(combined_data, pitcher_spec)

    top_pitchers = top_by_position(
        agg_pitchers,
        pitcher_spec,
        min_denom=MIN_IP,
        top_n=TOP_N_PIT,
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


def main():
    """Main function to run the data analysis workflow."""

    print("Loading all CSV files...")
    combined_data = load_all_csvs(DATA_FOLDER)

    if combined_data.empty:
        print("No data loaded.")
        return

    batter_spec = PlayerRateSpec(
        player_col=PLAYER_NAME_COLUMN,
        pos_col=POSITION_COLUMN,
        vlvl_col=VLVL_COLUMN,
        numerator_col='WAR',
        denominator_col='PA',
        scale=600.0,
    )

    find_top_players(combined_data, batter_spec)

    # Run generalized VL/VR -> metric regressions (BA, GAP, POW, K)
    specs = [
        {
            'name': 'BA_H_over_PA',
            'vl': 'BA vL',
            'vr': 'BA vR',
            'num': 'H',
            'denom': 'PA',
        },
        {
            'name': 'EYE_BB',
            'vl': 'EYE vL',
            'vr': 'EYE vR',
            'num': 'BB',
            'denom': 'PA',
        },
        {
            'name': 'GAP_EBH',
            'vl': 'GAP vL',
            'vr': 'GAP vR',
            'num': 'EBH',
            'denom': 'PA',
        },
        {
            'name': 'POW_HR_over_PA',
            'vl': 'POW vL',
            'vr': 'POW vR',
            'num': 'HR',
            'denom': 'PA',
        },
        {
            'name': 'K_SO',
            'vl': 'K vL',
            'vr': 'K vR',
            'num': 'SO',
            'denom': 'PA',
        },
    ]

    try:
        print('\nRunning VL/VR -> metric regressions for BA/GAP/POW/K...')
        # Skip specs whose numerator column is not present in the dataset
        runnable = []
        skipped = []
        for s in specs:
            if s.get('num') in combined_data.columns:
                runnable.append(s)
            else:
                skipped.append(s.get('name', s.get('num')))

        if skipped:
            print('Skipping specs with missing numerator columns:', skipped)

        if not runnable:
            print('No VL/VR regression specs runnable on this dataset.')
        else:
            results = run_multiple_vlr_regressions(
                combined_data, runnable, out_dir='output', plot=True
            )
            print('Completed VL/VR regressions. Results:')
            for spec, summary, model, pearson, plot_path in results:
                name = spec.get('name') or spec.get('num')
                print(f"- {name}: model={'present' if model else 'none'}, "
                      f"pearson={pearson}, plot={plot_path}"
                )
    except RuntimeError as exc:  # defensive: don't crash the main workflow on optional extras
        print('VL/VR regression runner failed:', exc)


if __name__ == "__main__":
    main()
