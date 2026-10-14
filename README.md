# Forecasting next-day geomagnetic storm risk from solar activity

[![CI](https://github.com/Shahbaz2104/geomagnetic-storm-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/Shahbaz2104/geomagnetic-storm-forecast/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Interactive%20dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://share.streamlit.io/Shahbaz2104/geomagnetic-storm-forecast/main/dashboard.py)

**[Live demo](https://share.streamlit.io/Shahbaz2104/geomagnetic-storm-forecast/main/dashboard.py)** · A practical, leakage-safe benchmark that asks: **can information available by the end of day _t_ distinguish whether day _t+1_ will contain a geomagnetic storm** (maximum estimated 3-hour K-index ≥ 5)?

This repository is a faithful, runnable port of the Kaggle notebook exported in [`docs/prompt.md`](docs/prompt.md). The emphasis is a transparent early-warning baseline, not a claim of operational forecasting skill.

## Dataset

[Kaggle: Space Weather: Solar + Geomagnetic Indices](https://www.kaggle.com/datasets/erevear/space-weather-solar-geomagnetic-indices) (`erevear/space-weather-solar-geomagnetic-indices`)

| File | Contents | Shape |
| --- | --- | --- |
| `daily_geomagnetic_data.csv` | 3-hour A and K indices (middle/high latitude, estimated) | 78,672 × 7 |
| `daily_solar_data.csv` | Daily radio flux, sunspot numbers, flares, X-ray flux | 10,330 × 14 |

Coverage: 1997-01-01 → 2024-04-12. The source uses `*` (and `-1` in documentation) as missing-value sentinels; these are converted explicitly during loading.

## Methodology

1. **Discover & validate** — recursive CSV discovery (works under `/kaggle/input` on Kaggle too), schema assertions, date parsing, duplicate-timestamp and chronology checks, sentinel conversion.
2. **Aggregate** — one row per UTC day: max/mean/std of estimated K, A-index stats, observation counts, mid/high-latitude K maxima, joined with same-day solar indicators.
3. **Target** — `next_day_storm = 1` when the following day's maximum estimated K is ≥ 5 (NOAA: K = 5 or higher indicates storm conditions). The final day is dropped because its future label is unavailable.
4. **Leakage audit** — `next_day_max_k` / `next_day_storm` are excluded from the feature list; an audit table prints feature dates vs. target dates.
5. **Chronological split** — earliest 80% of dates for training, latest 20% as an untouched test set (no shuffling: nearby days belong to the same space-weather regime).
6. **Models** — a persistence baseline (today's K ≥ 5), logistic regression, and histogram gradient boosting, all with median imputation inside the pipeline and `class_weight="balanced"` for the rare-event setting.
7. **Evaluation** — average precision (primary), ROC AUC, balanced accuracy, precision/recall/F1, confusion matrix, precision-recall curves, and held-out permutation importance.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). The dataset CSVs are committed, so a clone runs out of the box:

```bash
uv sync
uv run streamlit run dashboard.py   # interactive dashboard
uv run forecast                     # full pipeline: metrics + figures -> outputs/
```

Figures and `results.csv` are written to `outputs/`. Useful flags:

```bash
uv run forecast --skip-plots           # metrics only
uv run forecast --data-dir /kaggle/input   # run against a Kaggle attachment
```

To refresh the dataset from Kaggle (optional; needs API credentials, see script for details):

```bash
./scripts/download_data.sh
```

## Interactive dashboard

A Streamlit + Plotly explorer for the data, the models, and single-day predictions:

```bash
uv run streamlit run dashboard.py
```

![Dashboard overview](docs/dashboard_overview.png)

**Views**

| View | What it shows |
| --- | --- |
| Overview | Dynamic headline, KPIs, model comparison (AP vs baseline) |
| Storm history | 1997–2024 daily max-K timeline with storm markers and a year-range filter |
| Solar cycle | Sunspots + radio flux on stacked axes, flare composition, storm correlations |
| Storm calendar | Year × month heatmap of storm days |
| Model lab | PR/ROC curves, results table, and a live decision-threshold slider (confusion matrix updates as you drag) |
| Feature lab | Permutation importance, feature distributions by class, 2-feature scatter |
| Predict a day | Date picker → probability gauge, persistence comparison, actual outcome |

![Storm history view](docs/dashboard_storm_history.png)

## Project structure

```
.
├── dashboard.py                   # streamlit + plotly dashboard entrypoint
├── .streamlit/config.toml         # pinned dark theme (matches the screenshots)
├── .github/workflows/ci.yml       # lint + test on every push/PR
├── LICENSE                        # MIT
├── requirements.txt               # pinned deps (Streamlit Community Cloud deploy)
├── docs/prompt.md                 # original Kaggle notebook export (source writeup)
├── docs/dashboard_*.png           # dashboard screenshots
├── scripts/download_data.sh       # Kaggle CLI download via uvx
├── src/geomag_forecast/
│   ├── config.py                  # thresholds, expected schemas, paths
│   ├── data.py                    # discovery, validation, sentinel cleaning
│   ├── features.py                # daily aggregation, next-day target, leakage audit
│   ├── models.py                  # split, pipelines, metric computation
│   ├── evaluation.py              # figures, classification report, permutation importance
│   ├── charts.py                  # plotly figure builders for the dashboard
│   └── cli.py                     # `forecast` entrypoint
├── tests/                         # pytest suite (synthetic units + real-data smoke test)
├── data/raw/                      # committed dataset CSVs
└── outputs/                       # generated figures + metrics (gitignored)
```

## Tests

```bash
uv sync --group dev
uv run pytest          # 19 tests: target construction, leakage, split, charts, smoke
uv run ruff check .
```

The suite covers next-day target construction and the NOAA threshold boundary, the leakage audit (future columns must never enter X), the chronological split, schema/sentinel validation, chart structure, plus an end-to-end smoke test against the committed dataset. CI runs both commands on every push.

## Reference results

Reported by the original notebook run (data through 2024-04-11; test window 2018-09-27 → 2024-04-11, storm rate 8.23%):

| Model | Avg. precision | ROC AUC | Balanced acc. | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | **0.270** | 0.734 | 0.679 | 0.212 | 0.536 | 0.304 |
| Histogram gradient boosting | 0.229 | 0.708 | 0.648 | 0.207 | 0.451 | 0.284 |
| Persistence baseline | 0.159 | 0.630 | 0.630 | 0.320 | 0.320 | 0.320 |

Top permutation importances (logistic regression): `gmag_k_mean` (0.089), `gmag_k_std` (0.028), `Radio Flux 10.7cm` (0.015), `mid_k_max` (0.012).

Storm prevalence in the model table: 13.09% (8,076 calm / 1,216 storm days).

> Logistic regression, the persistence baseline, the classification report, and the permutation importances reproduce exactly. Histogram gradient boosting can vary slightly across scikit-learn versions: the locked environment in `uv.lock` (scikit-learn 1.7.2) yields AP 0.236 versus 0.229 in the table above; `uv run forecast` regenerates `outputs/results.csv` for whatever environment you run.

## Limitations

- Retrospective benchmark, not an operational warning system: the dataset ends in 2024 Q2, contains aggregated indices rather than raw solar-wind vectors, and includes no real-time latency simulation.
- Permutation importance is not a causal effect and is unstable under correlated predictors.
- Suggested extensions: solar-wind variables, probabilistic calibration, and rolling-origin evaluation.

## Sources

1. [NOAA Space Weather Prediction Center — Planetary K-index](https://www.spaceweather.gov/products/planetary-k-index)
2. [Kervalishvili et al. (2025), A Novel Model for Forecasting Geomagnetic Indices Using Machine Learning](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025GL114848)
3. [Wang et al. (2023), A machine learning-based model for the next 3-day geomagnetic index (Kp) forecast](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2023.1082737/full)

## License

Released under the [MIT License](LICENSE).
