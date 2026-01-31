"""
Main data processing function for imported csv values in data directory.
"""
# pylint: disable=R0902
import os
import glob
from dataclasses import dataclass
from typing import Dict
from typing import Tuple, Optional, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
class PlayerRateSpec:  # pylint: disable=R0902
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


@dataclass
class EyePipelineConfig:
    """Configuration for the weighted-eye -> BB/PA pipeline.

    This bundles many small parameters.
    """
    eye_vl_col: str = 'EYE vL'
    eye_vr_col: str = 'EYE vR'
    bb_col: str = 'BB'
    pa_col: str = 'PA'
    player_col: str = 'Title'
    pos_col: str = 'POS'
    pitcher_pos_markers: Sequence[str] = ('SP', 'RP', 'CL')
    ip_col: str = 'IP'
    hand_col: str = 'T'


def _prepare_numeric_columns(
    frame: pd.DataFrame,
    cfg: EyePipelineConfig,
) -> pd.DataFrame:
    """Coerce important columns to numeric types on a DataFrame copy.

    Only touch columns that exist to avoid creating new columns.
    """
    f = frame
    # Determine which columns exist and coerce only those
    have_pa = cfg.pa_col in f.columns
    have_bb = cfg.bb_col in f.columns
    have_vl = cfg.eye_vl_col in f.columns
    have_vr = cfg.eye_vr_col in f.columns

    if have_pa:
        f[cfg.pa_col] = pd.to_numeric(f.get(cfg.pa_col, 0), errors='coerce').fillna(0)
    if have_bb:
        f[cfg.bb_col] = pd.to_numeric(f.get(cfg.bb_col, 0), errors='coerce').fillna(0)
    if have_vl:
        f[cfg.eye_vl_col] = pd.to_numeric(f.get(cfg.eye_vl_col, 0), errors='coerce').fillna(np.nan)
    if have_vr:
        f[cfg.eye_vr_col] = pd.to_numeric(f.get(cfg.eye_vr_col, 0), errors='coerce').fillna(np.nan)
    return f


def _aggregate_player_metrics(
    frame: pd.DataFrame,
    cfg: EyePipelineConfig,
) -> pd.DataFrame:
    """Return per-player aggregated metrics (PA, BB and per-side EYE).

    If PA exists, per-side eye values are PA-weighted means. Otherwise simple means.
    """
    group_cols = [cfg.player_col]
    agg_spec = {}

    have_pa = cfg.pa_col in frame.columns
    have_bb = cfg.bb_col in frame.columns
    have_vl = cfg.eye_vl_col in frame.columns
    have_vr = cfg.eye_vr_col in frame.columns

    if have_pa:
        agg_spec[cfg.pa_col] = 'sum'
    if have_bb:
        agg_spec[cfg.bb_col] = 'sum'

    if have_vl:
        if have_pa:
            frame['_vl_pa'] = frame[cfg.eye_vl_col] * frame[cfg.pa_col]
            agg_spec['_vl_pa'] = 'sum'
        else:
            agg_spec[cfg.eye_vl_col] = 'mean'
    if have_vr:
        if have_pa:
            frame['_vr_pa'] = frame[cfg.eye_vr_col] * frame[cfg.pa_col]
            agg_spec['_vr_pa'] = 'sum'
        else:
            agg_spec[cfg.eye_vr_col] = 'mean'

    per_player = frame.groupby(group_cols, as_index=False).agg(agg_spec)

    if '_vl_pa' in per_player.columns and cfg.pa_col in per_player.columns:
        per_player[cfg.eye_vl_col] = (
            per_player['_vl_pa'] / per_player.get(cfg.pa_col, 0).replace({0: np.nan})
        )
        per_player.drop(columns=['_vl_pa'], inplace=True, errors='ignore')
    if '_vr_pa' in per_player.columns and cfg.pa_col in per_player.columns:
        per_player[cfg.eye_vr_col] = (
            per_player['_vr_pa'] / per_player.get(cfg.pa_col, 0).replace({0: np.nan})
        )
        per_player.drop(columns=['_vr_pa'], inplace=True, errors='ignore')

    return per_player


def _compute_weighted_eye(
    per_player: pd.DataFrame,
    cfg: EyePipelineConfig,
    w_l: float,
    w_r: float,
) -> pd.Series:
    """Compute weighted_eye Series for a per-player aggregated DataFrame."""
    have_vl = cfg.eye_vl_col in per_player.columns
    have_vr = cfg.eye_vr_col in per_player.columns

    if have_vl:
        vl_series = per_player[cfg.eye_vl_col]
    else:
        vl_series = pd.Series(np.nan, index=per_player.index)
    if have_vr:
        vr_series = per_player[cfg.eye_vr_col]
    else:
        vr_series = pd.Series(np.nan, index=per_player.index)

    return vl_series.fillna(0.0) * w_l + vr_series.fillna(0.0) * w_r


def _compute_bb_per_pa(per_player: pd.DataFrame, cfg: EyePipelineConfig) -> pd.Series:
    have_bb = cfg.bb_col in per_player.columns
    have_pa = cfg.pa_col in per_player.columns
    if have_bb and have_pa:
        return per_player[cfg.bb_col] / per_player[cfg.pa_col].replace({0: np.nan})
    return pd.Series(np.nan, index=per_player.index)


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


def compute_pitcher_hand_ip_shares(
    df: pd.DataFrame,
    pos_col: str = 'POS',
    pitcher_pos_markers: Sequence[str] = ('SP', 'RP', 'CL'),
    ip_col: str = 'IP',
    hand_col: str = 'T',
) -> Tuple[float, float]:
    """
    Compute the global share of pitcher IP faced that was against left-handed
    vs right-handed pitchers. Returns (wL, wR).

    Logic and fallbacks:
    - If `hand_col` exists and there are rows with `pos_col==pitcher_pos_marker` and
      numeric `ip_col`, sum IP by hand and compute shares.
    - If the computed total IP is zero or the columns are missing, returns (0.5, 0.5).
    """
    if hand_col in df.columns and pos_col in df.columns and ip_col in df.columns:
        # Support multiple pitcher position labels (SP, RP, CL) or a single marker.
        try:
            pitchers = df[df[pos_col].isin(pitcher_pos_markers)].copy()
        except RuntimeError:
            # Fallback for single-value marker
            pitchers = df[df[pos_col] == pitcher_pos_markers].copy()

        if pitchers.empty:
            return 0.5, 0.5

        pitchers[ip_col] = pd.to_numeric(pitchers.get(ip_col, 0), errors='coerce').fillna(0)
        ip_by_hand = pitchers.groupby(pitchers[hand_col].fillna(''))[ip_col].sum()
        total_ip = ip_by_hand.sum()
        if total_ip <= 0:
            return 0.5, 0.5

        # Typical handedness labels: 'L' and 'R'. Accept upper/lower-case by normalizing.
        hand_index = {str(k).upper(): v for k, v in ip_by_hand.items()}
        w_l = float(hand_index.get('L', 0.0)) / float(total_ip)
        w_r = float(hand_index.get('R', 0.0)) / float(total_ip)
        # If other or missing hand labels exist, normalize L/R only and
        # leave remainder distributed proportionally. If both 0 -> fallback.
        if w_l + w_r == 0:
            return 0.5, 0.5
        norm = w_l + w_r
        return w_l / norm, w_r / norm
    # missing columns -> fallback to even split
    return 0.5, 0.5


def weighted_eye_vs_bb_rate(
    df: pd.DataFrame,
    config: Optional[EyePipelineConfig] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Compute handedness-weighted EYE per player and correlate it with BB/PA.

    Returns a tuple (summary_df, correlation) where summary_df has columns:
      player_col, 'weighted_eye', 'bb_per_pa', pa_col

    Approach and assumptions (explicit):
    - We compute global pitcher-hand shares (wL, wR) using `compute_pitcher_hand_ip_shares`.
      This follows the assumption that a batter's plate appearance split
      mirrors the IP share of left/right pitchers.
    - For each batter we compute a per-player vL and vR value. If `PA` is present,
      we compute a PA-weighted average of eye values across rows; otherwise we use a simple mean.
    - BB/PA is computed as sum(BB)/sum(PA) per player; players with PA <= 0 get NaN.
    - Correlation is Pearson correlation between `weighted_eye` and `bb_per_pa` across players
      with non-null values.

    The function is defensive about missing columns and falls back to reasonable defaults.
    """
    # Allow callers to pass a config object, rely on defaults, or use
    # legacy keyword-arguments. Accepting **kwargs keeps the function
    # backward-compatible with older tests/callers.
    cfg = config or EyePipelineConfig()
    if kwargs:
        # only keep known config keys
        valid_keys = {
            'eye_vl_col', 'eye_vr_col', 'bb_col', 'pa_col', 'player_col',
            'pos_col', 'pitcher_pos_markers', 'ip_col', 'hand_col',
        }
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        if filtered:
            # create a new config overriding defaults from filtered kwargs
            cfg = EyePipelineConfig(**{**cfg.__dict__, **filtered})

    # compute weights for L/R pitchers
    w_l, w_r = compute_pitcher_hand_ip_shares(
        df,
        pos_col=cfg.pos_col,
        pitcher_pos_markers=cfg.pitcher_pos_markers,
        ip_col=cfg.ip_col,
        hand_col=cfg.hand_col,
    )

    if not (cfg.eye_vl_col in df.columns or cfg.eye_vr_col in df.columns):
        return pd.DataFrame(), None

    # Work on a shallow copy; helpers will coerce numeric columns as needed.
    work = df.copy()

    # Delegate aggregation and per-player math to small helpers.
    work = _prepare_numeric_columns(work, cfg)
    player_agg = _aggregate_player_metrics(work, cfg)

    weighted_eye_series = _compute_weighted_eye(player_agg, cfg, w_l, w_r)
    bb_per_pa_series = _compute_bb_per_pa(player_agg, cfg)

    out = pd.DataFrame({
        cfg.player_col: player_agg[cfg.player_col],
        'weighted_eye': weighted_eye_series,
        'bb_per_pa': bb_per_pa_series,
    })
    if cfg.pa_col in df.columns:
        out[cfg.pa_col] = player_agg.get(cfg.pa_col, 0)

    valid = out.dropna(subset=['weighted_eye', 'bb_per_pa'])
    corr = None
    if not valid.empty:
        corr = valid['weighted_eye'].corr(valid['bb_per_pa'])

    return out, corr


def weighted_eye_regression(
    summary_df: pd.DataFrame,
    eye_col: str = 'weighted_eye',
    target_col: str = 'bb_per_pa',
    weight_col: str = 'PA',
) -> Optional[dict]:
    """
    Fit a single linear regression of target_col ~ eye_col using weights from weight_col.

    The summary_df is expected to contain one row per player with columns:
      - eye_col: predictor (weighted_eye)
      - target_col: dependent variable (bb_per_pa)
      - weight_col: weight (PA)

    Returns a dict with keys: slope, intercept, r_squared, n_obs, sum_weights
    or None if the regression could not be fit (e.g., insufficient data).
    """
    if summary_df is None or summary_df.empty:
        return None

    if eye_col not in summary_df.columns or target_col not in summary_df.columns:
        return None

    # select valid rows and convert to numeric numpy arrays
    df = summary_df[[eye_col, target_col, weight_col]].dropna()
    if df.empty:
        return None

    x = pd.to_numeric(df[eye_col], errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(df[target_col], errors='coerce').to_numpy(dtype=float)
    w = pd.to_numeric(df[weight_col], errors='coerce').to_numpy(dtype=float)

    # Keep only rows with finite values and positive weights
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return None

    return _fit_weighted_wls(x[mask], y[mask], w[mask])


def _fit_weighted_wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Optional[dict]:
    """Fit a weighted least-squares line y ~ a + b*x and return stats.

    Kept as a small helper to reduce local-variable counts in the public
    wrapper function for linting.
    """
    # Design matrix with intercept
    design_matrix = np.vstack([np.ones_like(x), x]).T

    # Weighted least squares: beta = (X^T W X)^{-1} X^T W y
    xtw_x = design_matrix.T @ (w[:, None] * design_matrix)
    try:
        inv = np.linalg.inv(xtw_x)
    except np.linalg.LinAlgError:
        return None

    beta = inv @ (design_matrix.T @ (w * y))
    intercept, slope = float(beta[0]), float(beta[1])

    # Predictions and weighted sums for R^2
    y_hat = intercept + slope * x
    y_bar = (w * y).sum() / w.sum()
    ss_res = np.sum(w * (y - y_hat) ** 2)
    ss_tot = np.sum(w * (y - y_bar) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'n_obs': int(w.size),
        'sum_weights': float(w.sum()),
    }


def _prepare_plot_data(summary_df: pd.DataFrame, weight_col: str):
    """Prepare plotting series (x, y, w, sizes) from the summary DataFrame.

    Returns a tuple (x, y, w, sizes) or None if there's no valid data.
    """
    df = summary_df.dropna(subset=['weighted_eye', 'bb_per_pa']).copy()
    if df.empty:
        return None

    x = df['weighted_eye'].astype(float)
    y = df['bb_per_pa'].astype(float)
    w = df.get(weight_col, pd.Series(1, index=df.index)).astype(float)

    max_w = w.max() if not w.empty else 1.0
    sizes = (w / max_w) * 200 + 10
    return x, y, w, sizes


def _plot_regression_and_r2(ax, model: dict, x_series: pd.Series) -> None:
    """Plot the regression line and annotate R^2 on the provided Axes."""
    xs = np.linspace(x_series.min(), x_series.max(), 100)
    ys = model['intercept'] + model['slope'] * xs
    ax.plot(xs, ys, color='red', linewidth=2, label='WLS fit')
    ax.legend()

    if 'r_squared' in model and model['r_squared'] is not None:
        try:
            r2_val = float(model['r_squared'])
            txt = f"$R^2$ = {r2_val:.3f}"
            ax.text(
                0.02,
                0.98,
                txt,
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=10,
                bbox={"boxstyle": 'round', "facecolor": 'white', "alpha": 0.7, "edgecolor": 'none'},
            )
        except (ValueError, TypeError):
            # Non-fatal if formatting fails
            pass


def _ensure_output_dir(out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def plot_weighted_eye_vs_bb(
    summary_df: pd.DataFrame,
    model: Optional[dict] = None,
    weight_col: str = 'PA',
    out_path: str = 'output/eye_vs_bb.png',
    show: bool = True,
) -> Optional[str]:
    """
    Create a scatter plot of weighted_eye vs bb_per_pa.

    - Points are sized by `weight_col` (PA).
    - If `model` is provided (dict with slope/intercept), plot the regression line.
    - Saves PNG to `out_path`. If `show` is True and running on Windows, opens the image.

    Returns the path to the saved image, or None if plotting couldn't run (missing lib or data).
    """
    if summary_df is None or summary_df.empty:
        print('No summary data to plot.')
        return None
    prepared = _prepare_plot_data(summary_df, weight_col)
    if prepared is None:
        print('No valid points to plot (need weighted_eye and bb_per_pa).')
        return None
    x, y, _, sizes = prepared

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=sizes, alpha=0.6, edgecolors='k')
    plt.xlabel('weighted_eye')
    plt.ylabel('BB per PA')
    plt.title('weighted_eye vs BB/PA (point size ~ PA)')

    if model is not None and 'slope' in model and 'intercept' in model:
        ax = plt.gca()
        _plot_regression_and_r2(ax, model, x)

    _ensure_output_dir(out_path)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    if show:
        try:
            # Only attempt to open the file if it actually exists. On some
            # environments plt.savefig may not write the file or the path may
            # be invalid; avoid raising an unhandled FileNotFoundError.
            if os.path.exists(out_path):
                os.startfile(out_path)
        except OSError:
            # Any issue opening the image is non-fatal for the pipeline.
            pass

    return out_path
