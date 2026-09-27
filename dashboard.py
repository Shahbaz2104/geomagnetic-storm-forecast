"""Interactive Streamlit dashboard for the next-day geomagnetic storm benchmark.

Run with:  uv run streamlit run dashboard.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from geomag_forecast import charts as ch
from geomag_forecast.config import DEFAULT_DATA_DIR, STORM_K_THRESHOLD
from geomag_forecast.data import load_and_validate
from geomag_forecast.evaluation import compute_importance
from geomag_forecast.features import build_daily_table, leakage_audit
from geomag_forecast.models import build_pipelines, chronological_split, evaluate

st.set_page_config(
    page_title="Geomag Storm Forecast",
    page_icon=":cyclone:",
    layout="wide",
)


# --------------------------------------------------------------------------
# Data + model pipeline (cached: trains once per server session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and training models…")
def run_pipeline() -> dict:
    geomag, solar = load_and_validate(DEFAULT_DATA_DIR)
    daily = build_daily_table(geomag, solar)
    feature_cols = leakage_audit(daily)
    train, test = chronological_split(daily)

    X_train, y_train = train[feature_cols], train["next_day_storm"]
    X_test, y_test = test[feature_cols], test["next_day_storm"]

    models = build_pipelines()
    results, probs, preds = evaluate(X_train, y_train, X_test, y_test, models)
    _, importance = compute_importance(models, results, X_test, y_test, feature_cols)

    solar_daily = solar.sort_values("Date").copy()
    return {
        "geomag": geomag,
        "solar": solar_daily,
        "daily": daily,
        "feature_cols": feature_cols,
        "train": train,
        "test": test,
        "X_test": X_test,
        "y_test": y_test,
        "models": models,
        "results": results,
        "probs": probs,
        "preds": preds,
        "importance": importance,
    }


RUN = run_pipeline()
daily: pd.DataFrame = RUN["daily"]
results: pd.DataFrame = RUN["results"]
y_test: pd.Series = RUN["y_test"]
probs: dict[str, np.ndarray] = RUN["probs"]
importance: pd.Series = RUN["importance"]
feature_cols: list[str] = RUN["feature_cols"]

best_name = results.iloc[0]["model"]
best_row = results.iloc[0]
persist_row = results[results["model"] == "Persistence baseline"].iloc[0]
lift = best_row["average_precision"] / persist_row["average_precision"] - 1


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
st.sidebar.title("Geomag Storm Forecast")
st.sidebar.caption(
    f"Space Weather indices, {daily['date'].min().date()} → {daily['date'].max().date()}\n\n"
    "Features from day *t* predict a storm on day *t+1*."
)
section = st.sidebar.radio(
    "Views",
    [
        "Overview",
        "Storm history",
        "Solar cycle",
        "Storm calendar",
        "Model lab",
        "Feature lab",
        "Predict a day",
    ],
)
st.sidebar.divider()
st.sidebar.caption(
    "Chronological 80/20 split · test window 2018-09-27 → 2024-04-11 · "
    "storm threshold K ≥ 5 (NOAA)"
)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def overview() -> None:
    st.title("Can today's space weather predict tomorrow's storm?")
    st.markdown(
        f"**Headline:** {best_name} lifts next-day storm ranking "
        f"**{lift:.0%}** above the persistence baseline "
        f"(AP {best_row['average_precision']:.3f} vs {persist_row['average_precision']:.3f}). "
        "Storms are rare — only about 8% of test days — so precision at the default "
        "threshold stays modest while recall stays useful for early warning."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days modeled", f"{len(daily):,}")
    c2.metric("Storm-day rate", f"{daily['next_day_storm'].mean():.1%}")
    c3.metric(f"Test AP — {best_name}", f"{best_row['average_precision']:.3f}")
    c4.metric("Test recall @0.5", f"{best_row['recall']:.3f}")

    st.plotly_chart(ch.model_comparison(results), width="stretch")

    st.markdown(
        """
**How to read this dashboard**

1. **Storm history / calendar** — when storms actually happened, and how rare they are.
2. **Solar cycle** — the 11-year driver behind storm seasons.
3. **Model lab** — how much better the models are than "tomorrow repeats today",
   including a live decision-threshold trade-off.
4. **Feature lab** — which measurements carry the early signal.
5. **Predict a day** — pick any historical date and see the model's estimate.
"""
    )


def storm_history() -> None:
    st.title("Storm history")
    dmin = daily["date"].min().date()
    dmax = daily["date"].max().date()
    y0, y1 = st.slider(
        "Year range",
        min_value=dmin.year,
        max_value=dmax.year,
        value=(dmin.year, dmax.year),
        key="history_years",
    )
    mask = (daily["date"].dt.year >= y0) & (daily["date"].dt.year <= y1)
    view = daily[mask]

    storms = view[view["gmag_k_max"] >= STORM_K_THRESHOLD]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Days in view", f"{len(view):,}")
    s2.metric("Storm days", f"{len(storms):,}")
    s3.metric("Share of days", f"{len(storms) / max(len(view), 1):.1%}")
    if len(view):
        worst = view.loc[view["gmag_k_max"].idxmax()]
        s4.metric(
            "Worst day",
            f"{worst['date'].date()}",
            delta=f"max K {worst['gmag_k_max']:.0f}",
        )

    st.plotly_chart(ch.storm_timeline(view), width="stretch")
    st.caption(
        "Orange triangles are days whose maximum estimated K reached 5 or higher — "
        "these are the events the benchmark tries to signal one day ahead."
    )


def solar_cycle() -> None:
    st.title("Solar cycle")
    st.plotly_chart(ch.solar_cycle(RUN["solar"]), width="stretch")

    merged = daily[["Sunspot Number", "Radio Flux 10.7cm", "next_day_storm"]]
    corr_sun = merged["Sunspot Number"].corr(merged["next_day_storm"], method="spearman")
    corr_flux = merged["Radio Flux 10.7cm"].corr(merged["next_day_storm"], method="spearman")
    c1, c2 = st.columns(2)
    c1.metric("Spearman corr: sunspots → storm next day", f"{corr_sun:+.3f}")
    c2.metric("Spearman corr: radio flux → storm next day", f"{corr_flux:+.3f}")

    st.plotly_chart(ch.flare_activity(RUN["solar"]), width="stretch")
    st.caption(
        "Solar maximum years stack the most X-class flares — and the most storm days. "
        "Correlation is positive but modest: tomorrow's storm still depends on "
        "short-term geomagnetic conditions, not just the solar cycle."
    )


def storm_calendar() -> None:
    st.title("Storm calendar")
    st.plotly_chart(ch.storm_calendar(daily), width="stretch")

    d = daily.assign(
        year=daily["date"].dt.year,
        month=daily["date"].dt.month,
        is_storm=(daily["gmag_k_max"] >= STORM_K_THRESHOLD).astype(int),
    )
    by_year = d.groupby("year")["is_storm"].sum().sort_values(ascending=False)
    by_month = d.groupby("month")["is_storm"].sum()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    c1, c2 = st.columns(2)
    c1.metric(
        "Most active year",
        int(by_year.index[0]),
        delta=f"{int(by_year.iloc[0])} storm days",
    )
    c2.metric(
        "Most active month (all years)",
        month_names[int(by_month.idxmax()) - 1],
        delta=f"{int(by_month.max())} storm days",
    )


def model_lab() -> None:
    st.title("Model lab")
    st.dataframe(
        results.style.format(
            {
                "average_precision": "{:.3f}",
                "roc_auc": "{:.3f}",
                "balanced_accuracy": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    col_pr, col_roc = st.columns(2)
    with col_pr:
        st.plotly_chart(ch.pr_curves(y_test, probs), width="stretch")
    with col_roc:
        st.plotly_chart(ch.roc_curves(y_test, probs), width="stretch")

    st.subheader("Decision threshold")
    learned = [n for n in probs if n != "Persistence baseline"]
    model_name = st.selectbox("Model", learned)
    threshold = st.slider(
        "Classify as storm when P ≥", 0.05, 0.95, 0.50, 0.05, key="threshold"
    )

    prob = probs[model_name]
    pred = (prob >= threshold).astype(int)
    cm_col, metric_col = st.columns([3, 2])
    with cm_col:
        st.plotly_chart(
            ch.confusion_matrix_fig(y_test, pred, f"Confusion matrix — {model_name}"),
            width="stretch",
        )
    with metric_col:
        st.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
        st.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
        st.metric("F1", f"{f1_score(y_test, pred, zero_division=0):.3f}")
        st.metric(
            "Balanced accuracy",
            f"{balanced_accuracy_score(y_test, pred):.3f}",
        )
        flagged = int(pred.sum())
        st.caption(
            f"{flagged:,} of {len(pred):,} test days flagged · "
            f"{int(((pred == 1) & (y_test == 1)).sum()):,} were real storms"
        )

    st.plotly_chart(ch.threshold_curve(y_test, prob, threshold), width="stretch")
    st.caption(
        "Lowering the threshold trades precision for recall: catch more storms, "
        "at the cost of more false alarms. Choose the operating point that suits "
        "the cost of a missed storm."
    )


def feature_lab() -> None:
    st.title("Feature lab")
    st.plotly_chart(ch.importance_bars(importance), width="stretch")
    st.caption(
        "Permutation importance on the held-out test set: how much average precision "
        "drops when one column is shuffled. It is not a causal effect, and correlated "
        "predictors split credit."
    )

    left, right = st.columns(2)
    with left:
        feat = st.selectbox("Distribution of feature", feature_cols, key="violin_feat")
        st.plotly_chart(ch.feature_violin(daily, feat), width="stretch")
    with right:
        x_col = st.selectbox("X axis", feature_cols, index=feature_cols.index("gmag_k_mean"), key="scatter_x")
        y_col = st.selectbox(
            "Y axis",
            feature_cols,
            index=feature_cols.index("Radio Flux 10.7cm"),
            key="scatter_y",
        )
        st.plotly_chart(ch.feature_scatter(daily, x_col, y_col), width="stretch")


def predict_day() -> None:
    st.title("Predict a day")
    dmin = daily["date"].min().date()
    dmax = daily["date"].max().date()
    picked = st.date_input(
        "Feature day (model predicts the next UTC day)",
        value=default_date(daily),
        min_value=dmin,
        max_value=dmax,
    )
    day_ts = pd.Timestamp(picked)
    if day_ts.tzinfo is None:
        day_ts = day_ts.tz_localize("UTC")

    match = daily[daily["date"] == day_ts]
    if match.empty:
        st.warning("No geomagnetic + solar data for that date — pick another day.")
        return

    row = match.iloc[0]
    logit = RUN["models"]["Logistic regression"]
    hgb = RUN["models"]["Histogram gradient boosting"]
    X_row = row[feature_cols].astype(float).to_frame().T
    p_logit = float(logit.predict_proba(X_row)[0, 1])
    p_hgb = float(hgb.predict_proba(X_row)[0, 1])

    g1, g2 = st.columns([2, 3])
    with g1:
        st.plotly_chart(ch.probability_gauge(p_logit), width="stretch")
        st.caption(f"Histogram gradient boosting agrees: {p_hgb:.0%}")
    with g2:
        persistence_flag = row["gmag_k_max"] >= STORM_K_THRESHOLD
        actual_storm = int(row["next_day_storm"])
        st.metric(
            "Actual next day",
            "Storm" if actual_storm else "Calm",
            delta=f"max K {row['next_day_max_k']:.0f}",
            delta_color="off",
        )
        st.write(
            f"**Persistence rule:** {'storm' if persistence_flag else 'calm'} "
            f"(today's max K = {row['gmag_k_max']:.0f})"
        )
        st.write(
            f"**Model verdict:** "
            f"{'storm' if p_logit >= 0.5 else 'calm'} at the 0.50 threshold"
        )
        correct = (p_logit >= 0.5) == bool(actual_storm)
        st.write(f"**Outcome:** {'model was right' if correct else 'model missed it'}")

    with st.expander("Features used for this prediction"):
        st.dataframe(
            row[feature_cols].to_frame(name="value").T,
            width="stretch",
        )


def default_date(d: pd.DataFrame) -> dt.date:
    """A mid-range date that exists in the table."""
    return d["date"].iloc[len(d) // 2].date()


SECTIONS = {
    "Overview": overview,
    "Storm history": storm_history,
    "Solar cycle": solar_cycle,
    "Storm calendar": storm_calendar,
    "Model lab": model_lab,
    "Feature lab": feature_lab,
    "Predict a day": predict_day,
}

SECTIONS[section]()
