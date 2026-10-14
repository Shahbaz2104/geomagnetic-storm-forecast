"""End-to-end smoke test on the committed dataset (no model training)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomag_forecast.config import DEFAULT_DATA_DIR, STORM_K_THRESHOLD
from geomag_forecast.data import load_and_validate
from geomag_forecast.features import build_daily_table, leakage_audit
from geomag_forecast.models import chronological_split


def test_full_pipeline_on_committed_data():
    geomag, solar = load_and_validate(DEFAULT_DATA_DIR)
    daily = build_daily_table(geomag, solar)
    feature_cols = leakage_audit(daily)
    train, test = chronological_split(daily)

    # Coverage: 1997 → 2024, ~9k modeled days.
    assert daily["date"].min().year == 1997
    assert daily["date"].max().year == 2024
    assert 8_800 <= len(daily) <= 9_200
    assert daily["date"].is_monotonic_increasing

    # Rare but not degenerate event rate.
    rate = daily["next_day_storm"].mean()
    assert 0.10 <= rate <= 0.16
    assert set(daily["next_day_storm"].unique()) == {0, 1}

    # Split is chronological, ~80/20, with a clean handoff point.
    assert abs(len(train) / len(daily) - 0.80) < 0.01
    assert train["date"].max() < test["date"].min()

    # Leakage guarantees hold on the real feature list.
    assert "next_day_storm" not in feature_cols
    assert "next_day_max_k" not in feature_cols
    assert "date" not in feature_cols
    assert "gmag_k_max" in feature_cols

    # Every surviving row has a true calendar-day t+1 that exists in both
    # sources, and the label is taken from THAT day (gaps are dropped, never
    # bridged to a later row).
    gmag_daily = geomag.assign(date=geomag["Timestamp"].dt.normalize()).groupby("date")[
        "Estimated K"
    ].max()
    solar_dates = pd.DatetimeIndex(solar["Date"].unique())
    next_date = daily["date"] + pd.Timedelta(days=1)
    assert next_date.isin(gmag_daily.index).all()
    assert next_date.isin(solar_dates).all()
    expected_k = gmag_daily.reindex(next_date).to_numpy()
    np.testing.assert_allclose(daily["next_day_max_k"].to_numpy(), expected_k)

    # Storm labels agree with the threshold definition.
    observed = (daily["next_day_max_k"] >= STORM_K_THRESHOLD).astype(int)
    pd.testing.assert_series_equal(
        observed.astype(int),
        daily["next_day_storm"].astype(int),
        check_names=False,
    )
