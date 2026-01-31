# OOTP 26 Perfect Team Projection

Small analysis tool for OOTP Perfect Team CSV exports. The project concatenates CSVs from `data/`, computes player-level aggregates, derives a handedness-weighted EYE metric, fits a PA-weighted regression (weighted_eye -> BB/PA), and saves a scatter plot to `output/eye_vs_bb.png`.

Eventually this tool will correlate *all* ratings with stats, and allow projection of stats based on input ratings.

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run the main script from the repository root. CSV files must be placed under the `data/` directory.

```powershell
python main.py
```

What it does:
- Combines all `data/*.csv` into a single DataFrame.
- Prints top players by position (WAR per 600 PA, rWAR per 200 IP) using `aggregate_rate_by_player_pos_vlvl` and `top_by_position`.
- If the inputs contain `EYE vL`, `EYE vR`, `PA`, and `BB`, computes per-player `weighted_eye` and `bb_per_pa`, fits a PA-weighted regression, and saves a scatter plot at `output/eye_vs_bb.png`.

## Tests

Unit tests are under `tests/` and use `pytest`. Run them with:

```powershell
pytest -q
```

## Contributing

Pull requests welcome. For large changes, open an issue first. If you change column names or data conventions, update `PlayerRateSpec` and the plotting/aggregation helpers accordingly.

## License

[MIT](https://choosealicense.com/licenses/mit/)