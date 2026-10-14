"""Dataset discovery, schema validation, and sentinel cleaning."""

from __future__ import annotations

import pandas as pd
import pytest
from helpers import write_mini_dataset

from geomag_forecast.config import EXPECTED_GEOMAG, EXPECTED_SOLAR
from geomag_forecast.data import find_csvs, load_and_validate


def test_find_csvs_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing"):
        find_csvs(tmp_path)


def test_find_csvs_discovers_nested_files(tmp_path):
    nested = tmp_path / "nested" / "deeper"
    write_mini_dataset(nested)
    found = find_csvs(tmp_path)
    assert set(found) == {"daily_geomagnetic_data.csv", "daily_solar_data.csv"}
    assert found["daily_solar_data.csv"] == nested / "daily_solar_data.csv"


def test_load_and_validate_rejects_bad_geomag_schema(tmp_path):
    bad = pd.DataFrame({col: [] for col in EXPECTED_GEOMAG - {"Estimated K"}})
    bad.to_csv(tmp_path / "daily_geomagnetic_data.csv", index=False)
    pd.DataFrame({col: [] for col in EXPECTED_SOLAR}).to_csv(
        tmp_path / "daily_solar_data.csv", index=False
    )
    with pytest.raises(ValueError, match="Unexpected geomagnetic schema"):
        load_and_validate(tmp_path)


def test_load_and_validate_cleans_sentinels_and_sorts(tmp_path):
    write_mini_dataset(tmp_path)
    geomag, solar = load_and_validate(tmp_path)

    # '*' sentinels became missing values in the numeric columns.
    assert geomag["Estimated K"].isna().sum() == 1
    assert geomag["Estimated A"].isna().sum() == 1
    assert solar["Sunspot Number"].isna().sum() == 1
    # Unparseable sentinel values are numeric dtype after cleaning.
    assert pd.api.types.is_numeric_dtype(geomag["Estimated K"])
    # Input rows were out of order; output is chronological.
    assert geomag["Timestamp"].is_monotonic_increasing
    assert solar["Date"].is_monotonic_increasing
    assert geomag["Timestamp"].dt.tz is not None
    assert solar["Date"].dt.tz is not None
