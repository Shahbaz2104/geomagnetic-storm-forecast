"""Test-only helpers for building on-disk mini datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geomag_forecast.config import EXPECTED_SOLAR


def write_mini_dataset(data_dir: Path) -> None:
    """Write a tiny but schema-valid pair of CSVs that load_and_validate accepts."""
    data_dir.mkdir(parents=True, exist_ok=True)

    geomag = pd.DataFrame(
        [
            {
                "Timestamp": "2020-01-02T06:00:00Z",
                "Middle Latitude A": 4,
                "High Latitude A": 5,
                "Estimated A": 30,
                "Middle Latitude K": 3,
                "High Latitude K": 4,
                "Estimated K": 3,
            },
            {
                "Timestamp": "2020-01-01T00:00:00Z",
                "Middle Latitude A": 6,
                "High Latitude A": 7,
                "Estimated A": "*",
                "Middle Latitude K": 5,
                "High Latitude K": 6,
                "Estimated K": "*",
            },
        ]
    )
    solar_row: dict[str, object] = {"Date": "2020-01-01"}
    for col in EXPECTED_SOLAR - {"Date"}:
        solar_row[col] = "*" if col == "Sunspot Number" else 1
    solar = pd.DataFrame(
        [solar_row, {**solar_row, "Date": "2020-01-02", "Sunspot Number": 42}]
    )

    geomag.to_csv(data_dir / "daily_geomagnetic_data.csv", index=False)
    solar.to_csv(data_dir / "daily_solar_data.csv", index=False)
