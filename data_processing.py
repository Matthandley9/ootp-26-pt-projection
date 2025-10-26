"""
Main data processing function for imported csv values in data directory.
"""
import os
import glob
from typing import Dict
import pandas as pd
import numpy as np

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


def aggregate_rate_by_player_pos_vlvl(
    df: pd.DataFrame,
    player_col: str = 'Title',
    pos_col: str = 'POS',
    vlvl_col: str = 'VLvl',
    numerator_col: str = 'WAR',
    denominator_col: str = 'PA',
    scale: float = 600.0,
    numerator_out_name: str | None = None,
    denominator_out_name: str | None = None,
    rate_out_name: str | None = None,
) -> pd.DataFrame:
    """
    Generic aggregator that sums numerator and denominator by player/position/VLvl
    and computes (numerator_sum / denominator_sum) * scale.

    Returns a DataFrame containing:
      player_col, pos_col, vlvl_col, <denominator_out>, <numerator_out>, <rate_out>
    """
    df = df.copy()

    df[denominator_col] = (
        pd.to_numeric(df.get(denominator_col, 0), errors='coerce').fillna(0)
    )
    df[numerator_col] = (
        pd.to_numeric(df.get(numerator_col, 0), errors='coerce').fillna(0)
    )

    agg = (
        df
        .groupby([player_col, pos_col, vlvl_col], as_index=False)
        .agg({denominator_col: 'sum', numerator_col: 'sum'})
    )

    denom_name = denominator_out_name or denominator_col
    num_name = numerator_out_name or numerator_col
    if rate_out_name:
        rate_name = rate_out_name
    else:
        rate_name = f"{numerator_col}_per_{int(scale)}_{denom_name}"

    agg = agg.rename(columns={denominator_col: denom_name, numerator_col: num_name})
    agg[rate_name] = np.where(
        agg[denom_name] > 0,
        agg[num_name] / agg[denom_name] * scale,
        np.nan)

    return agg


def top_by_position(
    agg_df: pd.DataFrame,
    denom_col: str,
    min_denom: float,
    rate_col: str,
    top_n: int = 5,
    player_col: str = 'Title',
    pos_col: str = 'POS',
    vlvl_col: str = 'VLvl',
) -> Dict[str, pd.DataFrame]:
    """
    Generic top-N selector by position.
    Returns a dict mapping position -> top-N DataFrame containing:
      player_col, vlvl_col, denom_col, rate_col (in that order).
    """
    eligible = agg_df[agg_df[denom_col] >= min_denom].copy()
    if eligible.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for pos in sorted(eligible[pos_col].unique()):
        pos_df = eligible[eligible[pos_col] == pos].sort_values(
            by=rate_col, ascending=False
            ).head(top_n)
        if not pos_df.empty:
            cols = [player_col, vlvl_col, denom_col, rate_col]
            cols = [c for c in cols if c in pos_df.columns]
            result[pos] = pos_df[cols].reset_index(drop=True)
    return result
