"""Chronological split and model evaluation on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomag_forecast.models import build_pipelines, chronological_split, evaluate


def _synthetic_daily(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, tz="UTC"),
            "gmag_k_max": np.linspace(0, 8, n),
            "feature_a": np.linspace(1.0, -1.0, n),
            "next_day_storm": np.arange(n) % 2,
        }
    )


def test_chronological_split_is_ordered_and_80_20():
    daily = _synthetic_daily(10)
    train, test = chronological_split(daily)

    assert len(train) == 8
    assert len(test) == 2
    # Strictly chronological: train ends before test begins, no overlap.
    assert train["date"].max() < test["date"].min()
    assert train["date"].is_monotonic_increasing
    assert test["date"].is_monotonic_increasing
    assert set(train["date"]).isdisjoint(set(test["date"]))


def test_chronological_split_preserves_row_order():
    daily = _synthetic_daily(7)
    train, test = chronological_split(daily)
    reassembled = pd.concat([train, test])
    pd.testing.assert_frame_equal(reassembled.reset_index(drop=True), daily.reset_index(drop=True))


def test_evaluate_scores_all_models_sorted_by_average_precision():
    n = 60
    rng = np.random.default_rng(42)
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, tz="UTC"),
            "gmag_k_max": np.clip(rng.normal(3, 2, n), 0, 9),
            "signal": rng.normal(size=n),
            "next_day_storm": np.arange(n) % 2,
        }
    )
    train, test = daily.iloc[:48], daily.iloc[48:]
    feature_cols = ["gmag_k_max", "signal"]

    results, probs, preds = evaluate(
        train[feature_cols],
        train["next_day_storm"],
        test[feature_cols],
        test["next_day_storm"],
        build_pipelines(),
    )

    expected = {"Logistic regression", "Histogram gradient boosting", "Persistence baseline"}
    assert set(results["model"]) == expected
    assert set(probs) == expected
    assert set(preds) == expected
    # Ranked best-first by the primary metric.
    assert results["average_precision"].is_monotonic_decreasing
    # Probabilities are proper scores in [0, 1].
    for prob in probs.values():
        assert np.all((0 <= prob) & (prob <= 1))
    # Metrics are finite.
    assert results["average_precision"].notna().all()


def test_persistence_baseline_uses_storm_threshold():
    n = 60
    rng = np.random.default_rng(0)
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, tz="UTC"),
            "gmag_k_max": np.concatenate(
                [np.full(n // 2, 7.0), np.full(n - n // 2, 1.0)]
            ),
            "noise": rng.normal(size=n),
            "next_day_storm": np.arange(n) % 2,
        }
    )
    train, test = daily.iloc[:48], daily.iloc[48:]
    _results, probs, preds = evaluate(
        train[["gmag_k_max", "noise"]],
        train["next_day_storm"],
        test[["gmag_k_max", "noise"]],
        test["next_day_storm"],
        build_pipelines(),
    )

    # Persistence predicts storm iff today's max K >= 5.
    expected_pred = (test["gmag_k_max"] >= 5).astype(int).to_numpy()
    np.testing.assert_array_equal(preds["Persistence baseline"], expected_pred)
    np.testing.assert_array_equal(
        probs["Persistence baseline"], expected_pred.astype(float)
    )
