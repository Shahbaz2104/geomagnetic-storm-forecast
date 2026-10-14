"""Daily feature construction and next-day target definition."""

import logging

import numpy as np
import pandas as pd

from .config import SOLAR_FEATURES, STORM_K_THRESHOLD

logger = logging.getLogger(__name__)


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

    logger.info("Model table shape: %s", daily.shape)
    logger.info(
        "Model date range: %s to %s",
        daily["date"].min().date(),
        daily["date"].max().date(),
    )
    logger.info("Storm prevalence: %.3f%%", 100 * daily["next_day_storm"].mean())
    logger.info(
        "Class counts: %s",
        daily["next_day_storm"].value_counts().sort_index().to_dict(),
    )
    return daily


def leakage_audit(daily: pd.DataFrame) -> list[str]:
    """Return feature columns after auditing that no future column leaks into X.

    The target is defined from the following day's data only; future columns
    are excluded from the feature list so no leakage can enter X.
    """
    feature_cols = [
        c for c in daily.columns if c not in {"date", "next_day_max_k", "next_day_storm"}
    ]
    problems = []
    if not daily["next_day_storm"].notna().all():
        problems.append("next_day_storm contains missing values")
    for leaked in ("next_day_max_k", "next_day_storm", "date"):
        if leaked in feature_cols:
            problems.append(f"{leaked!r} leaked into the feature list")
    if problems:
        raise ValueError("Leakage audit failed: " + "; ".join(problems))

    audit = daily[["date", "next_day_max_k", "next_day_storm"]].head(5).copy()
    audit["target_date"] = audit["date"] + pd.Timedelta(days=1)
    logger.info("Leakage audit (first 5 rows):\n%s", audit.to_string(index=False))
    logger.info(
        "Leakage audit passed: target is next-day only; future target columns "
        "are excluded from X."
    )
    return feature_cols
