"""Plotly chart builders: structure, storm marking, and threshold handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomag_forecast import charts as ch
from geomag_forecast.config import STORM_K_THRESHOLD


def test_model_comparison_has_one_bar_per_model_and_best_title():
    results = pd.DataFrame(
        {
            "model": ["Logistic regression", "Histogram gradient boosting", "Persistence baseline"],
            "average_precision": [0.27, 0.23, 0.16],
        }
    )
    fig = ch.model_comparison(results)
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert len(bar.x) == 3
    assert "Logistic regression" in fig.layout.title.text
    assert "0.270" in fig.layout.title.text


def test_storm_timeline_marks_threshold_crossings():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, tz="UTC"),
            "gmag_k_max": [2, 6, 3, 9, 4, 5, 1, 3, 7, 2],
        }
    )
    fig = ch.storm_timeline(daily)

    # Line trace + storm-marker trace.
    assert len(fig.data) == 2
    markers = fig.data[1]
    expected_storms = int((daily["gmag_k_max"] >= STORM_K_THRESHOLD).sum())
    assert len(markers.x) == expected_storms == 4

    # Threshold annotation reflects the configured value.
    annotations = [a.text for a in fig.layout.annotations]
    assert any(f"K = {STORM_K_THRESHOLD}" in t for t in annotations)


def test_threshold_curve_returns_one_trace_per_metric():
    y = pd.Series(np.array([0, 1, 0, 1, 1, 0] * 5))
    prob = np.linspace(0.05, 0.95, len(y))
    fig = ch.threshold_curve(y, prob, selected=0.4)
    assert [t.name for t in fig.data] == ["Precision", "Recall", "F1"]
    for trace in fig.data:
        assert len(trace.x) == 97


def test_importance_bars_title_uses_top_feature():
    importance = pd.Series(
        {"gmag_k_std": 0.08, "gmag_k_mean": 0.03, "Radio Flux 10.7cm": 0.01}
    )
    fig = ch.importance_bars(importance)
    assert "gmag_k_std" in fig.layout.title.text


def test_probability_gauge_scales_to_percent():
    fig = ch.probability_gauge(0.42)
    indicator = fig.data[0]
    assert indicator.value == 42
