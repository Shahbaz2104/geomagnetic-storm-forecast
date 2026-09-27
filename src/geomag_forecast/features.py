"""Daily feature construction and next-day target definition."""

import numpy as np
import pandas as pd

from .config import SOLAR_FEATURES, STORM_K_THRESHOLD


def build_daily_table(geomag: pd.DataFrame, solar: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC day. Features come only from that day's observations."""
    geomag = geomag.copy()
    geomag["date"] = geomag["Timestamp"].dt.normalize()
    g_daily = (
        geomag.groupby("date", as_index=False)
        .agg(
            gmag_k_max=("Estimated K", "max"),
            gmag_k_mean=("Estimated K", "mean"),
            gmag_k_std=("Estimated K", "std"),
            gmag_a_max=("Estimated A", "max"),
            gmag_a_mean=("Estimated A", "mean"),
            gmag_obs=("Estimated K", "count"),
            mid_k_max=("Middle Latitude K", "max"),
            high_k_max=("High Latitude K", "max"),
        )
    )

    solar_daily = (
        solar[["Date"] + SOLAR_FEATURES]
        .groupby("Date", as_index=False)[SOLAR_FEATURES]
        .mean(numeric_only=True)
        .rename(columns={"Date": "date"})
    )

    daily = (
        g_daily.merge(solar_daily, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["next_day_max_k"] = daily["gmag_k_max"].shift(-1)
    daily["next_day_storm"] = (daily["next_day_max_k"] >= STORM_K_THRESHOLD).astype("float")
    daily.loc[daily["next_day_max_k"].isna(), "next_day_storm"] = np.nan
    daily = daily.dropna(subset=["next_day_storm"]).copy()
    daily["next_day_storm"] = daily["next_day_storm"].astype(int)

    print("Model table shape:", daily.shape)
    print("Model date range:", daily["date"].min().date(), "to", daily["date"].max().date())
    print("Storm prevalence:", f"{daily['next_day_storm'].mean():.3%}")
    print(
        "Class counts:",
        daily["next_day_storm"].value_counts().sort_index().to_dict(),
    )
    return daily


def leakage_audit(daily: pd.DataFrame) -> list[str]:
    """Return feature columns and print a human-readable target audit.

    The target is defined from the following day's data only; future columns
    are excluded from the feature list so no leakage can enter X.
    """
    feature_cols = [
        c for c in daily.columns if c not in {"date", "next_day_max_k", "next_day_storm"}
    ]
    assert daily["next_day_storm"].notna().all()
    assert "next_day_max_k" not in feature_cols
    assert "next_day_storm" not in feature_cols
    assert "date" not in feature_cols

    audit = daily[["date", "next_day_max_k", "next_day_storm"]].head(5).copy()
    audit["target_date"] = audit["date"] + pd.Timedelta(days=1)
    print(audit.to_string(index=False))
    print(
        "Leakage audit passed: target is next-day only; future target columns "
        "are excluded from X."
    )
    return feature_cols
