"""Dataset discovery, validation, and cleaning."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_FILES,
    EXPECTED_GEOMAG,
    EXPECTED_SOLAR,
    KAGGLE_INPUT_ROOT,
)


def find_csvs(data_dir: Path | None) -> dict[str, Path]:
    """Recursively discover the expected CSVs.

    Searches ``data_dir`` first, then falls back to ``/kaggle/input`` when it
    exists so the same code still runs on a Kaggle notebook runtime. Fails
    loudly if the attachment is missing or ambiguous.
    """
    roots = [r for r in (data_dir, KAGGLE_INPUT_ROOT) if r is not None and r.exists()]
    csvs: dict[str, Path] = {}
    for root in roots:
        for p in root.rglob("*.csv"):
            csvs.setdefault(p.name, p)
        if EXPECTED_FILES.issubset(csvs):
            break

    missing = EXPECTED_FILES - set(csvs)
    if missing:
        searched = [str(r) for r in roots] or ["<no existing data directory>"]
        raise FileNotFoundError(
            f"Missing {sorted(missing)}. Run scripts/download_data.sh to fetch the "
            f"Kaggle dataset 'erevear/space-weather-solar-geomagnetic-indices'. "
            f"Searched: {searched}. CSV files found: {sorted(csvs)}"
        )
    return csvs


def load_and_validate(data_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    csvs = find_csvs(data_dir)
    gmag_path = csvs["daily_geomagnetic_data.csv"]
    solar_path = csvs["daily_solar_data.csv"]
    geomag_raw = pd.read_csv(gmag_path)
    solar_raw = pd.read_csv(solar_path)

    print(f"Geomagnetic file: {gmag_path} | shape={geomag_raw.shape}")
    print(f"Solar file:       {solar_path} | shape={solar_raw.shape}")
    print("Geomagnetic columns:", list(geomag_raw.columns))
    print("Solar columns:", list(solar_raw.columns))

    assert EXPECTED_GEOMAG.issubset(geomag_raw.columns), "Unexpected geomagnetic schema"
    assert EXPECTED_SOLAR.issubset(solar_raw.columns), "Unexpected solar schema"

    geomag = geomag_raw.copy()
    solar = solar_raw.copy()
    geomag["Timestamp"] = pd.to_datetime(geomag["Timestamp"], errors="coerce", utc=True)
    solar["Date"] = pd.to_datetime(solar["Date"], errors="coerce", utc=True).dt.normalize()
    assert geomag["Timestamp"].notna().all() and solar["Date"].notna().all()

    # The source uses '*' and -1 as sentinel values. Convert numeric-looking
    # columns explicitly; "Stanford Background X-Ray Flux" is kept as-is.
    geomag_num = [c for c in geomag.columns if c != "Timestamp"]
    solar_num = [c for c in solar.columns if c not in {"Date", "Stanford Background X-Ray Flux"}]
    for c in geomag_num:
        geomag[c] = pd.to_numeric(geomag[c].replace("*", np.nan), errors="coerce")
    for c in solar_num:
        solar[c] = pd.to_numeric(solar[c].replace("*", np.nan), errors="coerce")

    geomag = geomag.sort_values("Timestamp").reset_index(drop=True)
    solar = solar.sort_values("Date").reset_index(drop=True)

    print("Duplicate geomagnetic timestamps:", int(geomag["Timestamp"].duplicated().sum()))
    print("Duplicate solar dates:", int(solar["Date"].duplicated().sum()))
    print(
        "Geomagnetic date range:",
        geomag["Timestamp"].min().date(),
        "to",
        geomag["Timestamp"].max().date(),
    )
    print("Solar date range:", solar["Date"].min().date(), "to", solar["Date"].max().date())
    print(
        "Converted missing values:",
        int(geomag.isna().sum().sum() + solar.isna().sum().sum()),
    )
    assert geomag["Timestamp"].is_monotonic_increasing, "Geomagnetic chronology is not sorted"

    return geomag, solar
