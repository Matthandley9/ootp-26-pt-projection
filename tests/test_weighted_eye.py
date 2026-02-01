"""Unit tests for weighted eye vs. BB rate calculations and regression."""
import math
import numpy as np
import pandas as pd

from data_processing import (
    compute_pitcher_hand_ip_shares,
    compute_weighted_predictor_vs_rate,
    fit_weighted_regression
)


def test_weighted_eye_and_correlation():
    """Test correlation equation using synthetic data."""
    # Construct synthetic dataset
    rows = []
    # Pitchers: SP L with 20 IP, RP R with 80 IP -> wL=0.2, wR=0.8
    rows.append({'Title': 'P1', 'POS': 'SP', 'T': 'L', 'IP': 20})
    rows.append({'Title': 'P2', 'POS': 'RP', 'T': 'R', 'IP': 80})

    # Batter Alice: two rows to test PA-weighted averaging
    rows.append({'Title': 'Alice', 'POS': '1B', 'EYE vL': 8, 'EYE vR': 22, 'PA': 40, 'BB': 4})
    rows.append({'Title': 'Alice', 'POS': '1B', 'EYE vL': 12, 'EYE vR': 18, 'PA': 60, 'BB': 6})

    # Batter Bob: single row
    rows.append({'Title': 'Bob', 'POS': '2B', 'EYE vL': 5, 'EYE vR': 15, 'PA': 50, 'BB': 5})

    # Batter Carol: single row with higher BB rate
    rows.append({'Title': 'Carol', 'POS': 'OF', 'EYE vL': 30, 'EYE vR': 40, 'PA': 100, 'BB': 30})

    df = pd.DataFrame(rows)

    # Verify pitcher hand shares
    wL, wR = compute_pitcher_hand_ip_shares( #pylint: disable=C0103
        df, pos_col='POS', pitcher_pos_markers=('SP', 'RP'), ip_col='IP', hand_col='T'
    )
    assert math.isclose(wL, 0.2, rel_tol=1e-9)
    assert math.isclose(wR, 0.8, rel_tol=1e-9)

    summary, corr = compute_weighted_predictor_vs_rate(df,
                                           eye_vl_col='EYE vL',
                                           eye_vr_col='EYE vR',
                                           bb_col='BB',
                                           pa_col='PA',
                                           player_col='Title',
                                           pos_col='POS',
                                           pitcher_pos_markers=('SP', 'RP'),
                                           ip_col='IP',
                                           hand_col='T')

    # Check per-player weighted_eye values (PA-weighted for Alice)
    alice = summary[summary['Title'] == 'Alice'].iloc[0]
    bob = summary[summary['Title'] == 'Bob'].iloc[0]
    carol = summary[summary['Title'] == 'Carol'].iloc[0]

    # Expected per-player vL/vR for Alice (PA-weighted): vL = 10.4, vR = 19.6
    expected_alice_weighted = 10.4 * 0.2 + 19.6 * 0.8  # = 17.76
    expected_bob_weighted = 5 * 0.2 + 15 * 0.8        # = 13.0
    expected_carol_weighted = 30 * 0.2 + 40 * 0.8     # = 38.0

    assert np.isclose(alice['weighted_eye'], expected_alice_weighted, atol=1e-6)
    assert np.isclose(bob['weighted_eye'], expected_bob_weighted, atol=1e-6)
    assert np.isclose(carol['weighted_eye'], expected_carol_weighted, atol=1e-6)

    # Check bb_per_pa
    assert np.isclose(alice['bb_per_pa'], 10 / 100)
    assert np.isclose(bob['bb_per_pa'], 5 / 50)
    assert np.isclose(carol['bb_per_pa'], 30 / 100)

    # Compute expected correlation using numpy on the matching valid rows
    valid = summary.dropna(subset=['weighted_eye', 'bb_per_pa'])
    exp_corr = np.corrcoef(valid['weighted_eye'], valid['bb_per_pa'])[0, 1]
    assert np.isclose(corr, exp_corr, atol=1e-12) or (np.isnan(corr) and np.isnan(exp_corr))


def test_weighted_regression():
    """Test weighted regression using synthetic data."""
    # Create a synthetic summary dataframe where bb_per_pa = 0.01 * weighted_eye
    players = ['A', 'B', 'C', 'D']
    weighted_eye = np.array([10.0, 20.0, 30.0, 40.0])
    pa = np.array([100.0, 200.0, 50.0, 150.0])
    bb_per_pa = 0.01 * weighted_eye

    df = pd.DataFrame({
        'Title': players,
        'weighted_eye': weighted_eye,
        'bb_per_pa': bb_per_pa,
        'PA': pa,
    })

    model = fit_weighted_regression(df, eye_col='weighted_eye',
                                    target_col='bb_per_pa', weight_col='PA'
            )
    assert model is not None
    # slope should be approximately 0.01
    assert np.isclose(model['slope'], 0.01, atol=1e-12)
    assert np.isclose(model['intercept'], 0.0, atol=1e-12)
