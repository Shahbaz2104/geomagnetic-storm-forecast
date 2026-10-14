"""Next-day target construction and the leakage audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geomag_forecast.features import build_daily_table, leakage_audit


def test_build_daily_table_next_day_target(geomag_df, solar_df):
    daily = build_daily_table(geomag_df, solar_df)

    # 5 observed days; the final day has no next-day label and is dropped.
    assert len(daily) == 4
    assert daily["date"].is_monotonic_increasing

    # Daily max K: 4, 6, 2, 7, 2 → each row's target is the FOLLOWING day's max.
    assert daily["gmag_k_max"].tolist() == [4, 6, 2, 7]
    assert daily["next_day_max_k"].tolist() == [6, 2, 7, 2]
    assert daily["next_day_storm"].tolist() == [1, 0, 1, 0]

    # Target column is a clean integer label with no missing values.
    assert daily["next_day_storm"].dtype.kind == "i"
    assert daily["next_day_storm"].notna().all()

    # Same-day solar columns are joined in.
    assert "Sunspot Number" in daily.columns
    assert daily["Sunspot Number"].notna().all()


def test_build_daily_table_drops_days_without_solar_data(geomag_df, solar_df):
    # Keep only Jan 1–3 solar rows: the inner join then limits the table,
    # and Jan 3 loses its label (no Jan 4 row remains) and is dropped too.
    solar_subset = solar_df[solar_df["Date"] <= pd.Timestamp("2020-01-03", tz="UTC")]
    daily = build_daily_table(geomag_df, solar_subset)
    assert daily["date"].tolist() == [
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-01-02", tz="UTC"),
    ]


def test_build_daily_table_never_labels_across_a_gap(geomag_df, solar_df):
    """A missing middle day must not be bridged: Jan 2's label would come from
    Jan 4 (t+3) if gaps were skipped — that row must be dropped instead."""
    solar_subset = solar_df[solar_df["Date"] != pd.Timestamp("2020-01-03", tz="UTC")]
    daily = build_daily_table(geomag_df, solar_subset)

    # Jan 2's successor is Jan 4 (gap) → dropped; Jan 5 is last → dropped.
    assert daily["date"].tolist() == [
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-01-04", tz="UTC"),
    ]
    # Jan 1 keeps its true t+1 label (Jan 2's max K = 6 → storm).
    jan1 = daily[daily["date"] == pd.Timestamp("2020-01-01", tz="UTC")].iloc[0]
    assert jan1["next_day_max_k"] == 6
    assert jan1["next_day_storm"] == 1
    # Jan 4's true t+1 is Jan 5 (max K = 2 → calm).
    jan4 = daily[daily["date"] == pd.Timestamp("2020-01-04", tz="UTC")].iloc[0]
    assert jan4["next_day_max_k"] == 2
    assert jan4["next_day_storm"] == 0


def test_leakage_audit_excludes_future_columns(geomag_df, solar_df):
    daily = build_daily_table(geomag_df, solar_df)
    feature_cols = leakage_audit(daily)

    for leaked in ("date", "next_day_max_k", "next_day_storm"):
        assert leaked not in feature_cols
    assert "gmag_k_max" in feature_cols
    assert "Sunspot Number" in feature_cols
    # Every other data column is a legal feature.
    assert set(feature_cols) == set(daily.columns) - {"date", "next_day_max_k", "next_day_storm"}


def test_leakage_audit_rejects_missing_targets():
    bad = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, tz="UTC"),
            "gmag_k_max": [1.0, 2.0, 3.0],
            "next_day_max_k": [2.0, 3.0, np.nan],
            "next_day_storm": [0, 1, np.nan],
        }
    )
    with pytest.raises(ValueError, match="Leakage audit failed"):
        leakage_audit(bad)


def test_build_daily_table_threshold_boundary(geomag_df, solar_df):
    """A next-day max K of exactly 5 counts as a storm (NOAA threshold)."""
    geomag = geomag_df.copy()
    # Lower Jan 4's peak from 7 to 5 so Jan 3's target sits on the threshold.
    peak = geomag["Timestamp"] == pd.Timestamp("2020-01-04 00:00", tz="UTC")
    assert peak.sum() == 1
    geomag.loc[peak, "Estimated K"] = 5

    daily = build_daily_table(geomag, solar_df)
    row = daily[daily["date"] == pd.Timestamp("2020-01-03", tz="UTC")]
    assert row["next_day_max_k"].iloc[0] == 5
    assert row["next_day_storm"].iloc[0] == 1
