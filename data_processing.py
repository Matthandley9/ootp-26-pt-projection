"""
Main data processing function for imported csv values in data directory.
"""
import os
import glob
from dataclasses import dataclass
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


@dataclass
class PlayerRateSpec:
    """
    Specification for a player rate aggregation.

    Example for batters (WAR per 600 PA):
      PlayerRateSpec(numerator_col='WAR', denominator_col='PA', scale=600.0)
    """
    player_col: str = 'Title'
    pos_col: str = 'POS'
    vlvl_col: str = 'VLvl'
    numerator_col: str = 'WAR'
    denominator_col: str = 'PA'
    scale: float = 600.0

    def denom_name(self) -> str: # pylint: disable=C0116
        return self.denominator_col

    def num_name(self) -> str: # pylint: disable=C0116
        return self.numerator_col

    def rate_name(self) -> str: # pylint: disable=C0116
        return f"{self.numerator_col}_per_{int(self.scale)}_{self.denominator_col}"


def aggregate_rate_by_player_pos_vlvl(
    df: pd.DataFrame,
    spec: PlayerRateSpec,
) -> pd.DataFrame:
    """
    Generic aggregator that sums numerator and denominator by player/position/VLvl
    according to the provided PlayerRateSpec and computes 
    (numerator_sum / denominator_sum) * scale.

    Returns a DataFrame containing:
      spec.player_col, spec.pos_col, spec.vlvl_col,
      <denom_name>, <num_name>, <rate_name>
    """
    df = df.copy()

    df[spec.denominator_col] = (
        pd.to_numeric(df.get(spec.denominator_col, 0), errors='coerce').fillna(0)
    )
    df[spec.numerator_col] = (
        pd.to_numeric(df.get(spec.numerator_col, 0), errors='coerce').fillna(0)
    )

    agg = (
        df
        .groupby([spec.player_col, spec.pos_col, spec.vlvl_col], as_index=False)
        .agg({spec.denominator_col: 'sum', spec.numerator_col: 'sum'})
    )

    denom_out = spec.denom_name()
    num_out = spec.num_name()
    rate_out = spec.rate_name()

    agg = agg.rename(
        columns={spec.denominator_col: denom_out, spec.numerator_col: num_out}
    )
    agg[rate_out] = np.where(
        agg[denom_out] > 0,
        agg[num_out] / agg[denom_out] * spec.scale,
        np.nan
    )

    return agg


def top_by_position(
    agg_df: pd.DataFrame,
    spec: PlayerRateSpec,
    min_denom: float,
    top_n: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Generic top-N selector by position driven by a PlayerRateSpec.

    Returns a dict mapping position -> top-N DataFrame containing:
      spec.player_col, spec.vlvl_col, <denom_out>,
      <rate_out> (in that order, when present).
    """
    denom_col = spec.denom_name()
    rate_col = spec.rate_name()
    player_col = spec.player_col
    pos_col = spec.pos_col
    vlvl_col = spec.vlvl_col

    if denom_col not in agg_df.columns or rate_col not in agg_df.columns:
        return {}

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
