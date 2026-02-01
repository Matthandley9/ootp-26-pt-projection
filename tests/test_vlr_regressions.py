"""Tests for VL/VR weighted regression functions in data_processing.py."""
import pandas as pd

from data_processing import (
    weighted_vlr_regression,
    run_multiple_vlr_regressions,
)


def make_synthetic_df():
    """Create a synthetic dataset for testing VL/VR weighted regression."""
    # Two pitchers (one left, one right) to make weights 0.5/0.5
    rows = [
        {"Title": "P_L", "POS": "SP", "T": "L", "IP": 100},
        {"Title": "P_R", "POS": "SP", "T": "R", "IP": 100},
        # Batter 1
        {
            "Title": "B1",
            "POS": "1B",
            "PA": 1000,
            "EYE vL": 0.10,
            "EYE vR": 0.20,
            "BB": 100,
            "H": 200,
            "XBH": 30,
            "HR": 20,
            "SO": 80,
        },
        # Batter 2
        {
            "Title": "B2",
            "POS": "2B",
            "PA": 500,
            "EYE vL": 0.30,
            "EYE vR": 0.40,
            "BB": 150,
            "H": 150,
            "XBH": 40,
            "HR": 30,
            "SO": 90,
        },
    ]
    return pd.DataFrame(rows)


def test_weighted_vlr_regression_happy_path():
    """Test weighted VL/VR regression with synthetic data."""
    df = make_synthetic_df()

    summary, model, pearson = weighted_vlr_regression(
        df,
        vl_col='EYE vL',
        vr_col='EYE vR',
        target_num_col='BB',
        target_denom_col='PA',
    )

    # Basic structural assertions
    assert summary is not None
    assert 'predictor' in summary.columns
    assert 'target' in summary.columns

    # Model should be present and slope should be positive
    # (higher EYE -> higher BB/PA in our synthetic data)
    assert model is not None
    assert isinstance(model.get('slope'), float)
    assert model.get('slope') > 0
    assert pearson is not None


def test_run_multiple_vlr_regressions(tmp_path):
    """Test running multiple VL/VR regressions with synthetic data."""
    df = make_synthetic_df()
    specs = [
        {'name': 'BB_rate', 'vl': 'EYE vL', 'vr': 'EYE vR', 'num': 'BB', 'denom': 'PA'},
        {'name': 'H_rate', 'vl': 'EYE vL', 'vr': 'EYE vR', 'num': 'H', 'denom': 'PA'},
    ]

    results = run_multiple_vlr_regressions(df, specs, out_dir=str(tmp_path))
    assert isinstance(results, list)
    assert len(results) == 2
    for item in results:
        spec, summary, model, pearson, plot_path = item
        assert isinstance(spec, dict)
        assert 'vl' in spec and 'vr' in spec and 'num' in spec
        # summary is a DataFrame
        assert hasattr(summary, 'columns')
        # model may be None or dict
        assert (model is None) or isinstance(model, dict)
        # pearson may be None or float
        assert (pearson is None) or isinstance(pearson, float)
