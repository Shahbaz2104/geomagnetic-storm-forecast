"""Shared configuration for the next-day geomagnetic storm benchmark."""

from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")

EXPECTED_FILES = {"daily_geomagnetic_data.csv", "daily_solar_data.csv"}

EXPECTED_GEOMAG = {
    "Timestamp",
    "Middle Latitude A",
    "High Latitude A",
    "Estimated A",
    "Middle Latitude K",
    "High Latitude K",
    "Estimated K",
}

EXPECTED_SOLAR = {
    "Date",
    "Radio Flux 10.7cm",
    "Sunspot Number",
    "Sunspot Area (10^6 Hemis.)",
    "New Regions",
    "Stanford Mean Solar Field (GOES15)",
    "Stanford Background X-Ray Flux",
    "Flares: C",
    "Flares: M",
    "Flares: X",
    "Flares: S",
    "Flares: 1",
    "Flares: 2",
    "Flares: 3",
}

SOLAR_FEATURES = [
    "Radio Flux 10.7cm",
    "Sunspot Number",
    "Sunspot Area (10^6 Hemis.)",
    "New Regions",
    "Flares: C",
    "Flares: M",
    "Flares: X",
    "Flares: S",
    "Flares: 1",
    "Flares: 2",
    "Flares: 3",
]

STORM_K_THRESHOLD = 5
TRAIN_FRACTION = 0.80
