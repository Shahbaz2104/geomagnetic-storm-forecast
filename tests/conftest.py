"""Shared fixtures: small synthetic frames for feature/model/chart tests."""

from __future__ import annotations

import pandas as pd
import pytest

from geomag_forecast.config import SOLAR_FEATURES


@pytest.fixture
def geomag_df() -> pd.DataFrame:
    """Two 3-hour observations per day over 5 days.

    Daily max K by day: 4, 6, 2, 7, 2 → next-day storm labels: 1, 0, 1, 0.
    """
    k_by_day = {0: (3, 4), 1: (6, 5), 2: (2, 2), 3: (7, 4), 4: (1, 2)}
    base = pd.Timestamp("2020-01-01", tz="UTC")
    rows = []
    for day, (k1, k2) in k_by_day.items():
        for hours, k in ((0, k1), (6, k2)):
            rows.append(
                {
                    "Timestamp": base + pd.Timedelta(days=day, hours=hours),
                    "Middle Latitude A": k + 1,
                    "High Latitude A": k + 2,
                    "Estimated A": 10 * k,
                    "Middle Latitude K": k,
                    "High Latitude K": k,
                    "Estimated K": k,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def solar_df() -> pd.DataFrame:
    """Daily solar rows matching the 5 days of ``geomag_df`` (UTC-normalized)."""
    base = pd.Timestamp("2020-01-01", tz="UTC")
    rows = []
    for day in range(5):
        row = {"Date": (base + pd.Timedelta(days=day)).normalize()}
        for i, col in enumerate(SOLAR_FEATURES):
            row[col] = float(10 * day + i)
        rows.append(row)
    return pd.DataFrame(rows)
