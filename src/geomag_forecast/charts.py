"""Plotly figure builders for the dashboard.

Design follows the data-visualization skill: colorblind-safe palette, insight-
driven titles, bars that start at zero with direct value labels, no pie/donut
charts, and line styles/symbols so meaning never depends on color alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Colorblind-safe palette (data-visualization skill).
BLUE = "#4C72B0"      # calm / primary series
ORANGE = "#DD8452"    # storm / highlighted series
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B3"
GREY = "#999999"
MUTED = "#C7C7C7"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

PALETTE_MODELS = {
    "Logistic regression": BLUE,
    "Histogram gradient boosting": ORANGE,
    "Persistence baseline": GREEN,
}


def _finalize(fig: go.Figure, x_title: str = "", y_title: str = "", height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=55, b=40),
        font=dict(size=12),
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="",
        hovermode="x unified",
    )
    return fig


def model_comparison(results: pd.DataFrame) -> go.Figure:
    """Horizontal bars of average precision; best model highlighted."""
    df = results.sort_values("average_precision")
    colors = [ORANGE if i == len(df) - 1 else MUTED for i in range(len(df))]
    fig = go.Figure(
        go.Bar(
            x=df["average_precision"],
            y=df["model"],
            orientation="h",
            marker_color=colors,
            text=df["average_precision"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
            hovertemplate="%{y}<br>AP: %{x:.3f}<extra></extra>",
        )
    )
    best = df.iloc[-1]
    fig.update_layout(
        title=dict(
            text=f"{best['model']} ranks storms best (AP {best['average_precision']:.3f})",
            x=0,
        )
    )
    return _finalize(
        fig,
        x_title="Average precision on the 2018–2024 test set (higher is better)",
        height=320,
    )


def storm_timeline(daily: pd.DataFrame) -> go.Figure:
    """Daily maximum estimated K with the storm threshold and storm days marked."""
    d = daily.assign(storm=daily["gmag_k_max"] >= 5)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["gmag_k_max"],
            mode="lines",
            name="Daily max K",
            line=dict(color=BLUE, width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>max K: %{y:.1f}<extra></extra>",
        )
    )
    storms = d[d["storm"]]
    fig.add_trace(
        go.Scatter(
            x=storms["date"],
            y=storms["gmag_k_max"],
            mode="markers",
            name="Storm days (K ≥ 5)",
            marker=dict(color=ORANGE, size=4, symbol="triangle-up"),
            hovertemplate="%{x|%Y-%m-%d}<br>max K: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=5,
        line_dash="dash",
        line_color=RED,
        annotation_text="Storm threshold K = 5",
        annotation_position="bottom right",
    )
    fig.update_layout(title=dict(text="Storm days appear as spikes above the K = 5 threshold", x=0))
    return _finalize(fig, x_title="Date", y_title="Maximum estimated K")


def solar_cycle(solar: pd.DataFrame) -> go.Figure:
    """Sunspots and radio flux stacked (no dual axis) with a shared range slider."""
    s = solar.sort_values("Date")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Daily sunspot number", "Radio flux 10.7 cm (sfu)"),
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=s["Date"], y=s["Sunspot Number"], mode="lines",
            line=dict(color=BLUE, width=1), name="Sunspots",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}<extra>sunspots</extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=s["Date"], y=s["Radio Flux 10.7cm"], mode="lines",
            line=dict(color=ORANGE, width=1), name="Radio flux",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f} sfu<extra>flux</extra>",
        ),
        row=2, col=1,
    )
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
    fig.update_layout(
        title=dict(text="Solar activity rises and falls on an ~11-year cycle", x=0),
        showlegend=True,
    )
    return _finalize(fig, y_title="", height=520)


def flare_activity(solar: pd.DataFrame) -> go.Figure:
    """Composition over time: C/M/X flare counts as a stacked area chart."""
    s = solar.sort_values("Date")
    fig = go.Figure()
    for flare, color in zip(["Flares: C", "Flares: M", "Flares: X"], [MUTED, ORANGE, RED]):
        fig.add_trace(
            go.Scatter(
                x=s["Date"], y=s[flare], mode="lines", name=flare.replace("Flares: ", ""),
                stackgroup="flares", line=dict(color=color, width=0.7),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}<extra>" + flare + "</extra>",
            )
        )
    fig.update_layout(title=dict(text="X-class flares cluster at solar maximum", x=0))
    return _finalize(fig, x_title="Date", y_title="Flare count")


def storm_calendar(daily: pd.DataFrame) -> go.Figure:
    """Year × month heatmap of observed storm days."""
    d = daily.assign(
        year=daily["date"].dt.year,
        month=daily["date"].dt.month,
        is_storm=(daily["gmag_k_max"] >= 5).astype(int),
    )
    pivot = d.pivot_table(index="year", columns="month", values="is_storm", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=range(1, 13), fill_value=0)
    fig = px.imshow(
        pivot.values,
        x=MONTH_LABELS,
        y=pivot.index,
        color_continuous_scale="YlOrRd",
        text_auto=True,
        aspect="auto",
        labels=dict(color="Storm days"),
    )
    fig.update_traces(
        hovertemplate="Year %{y}<br>%{x}: %{z} storm days<extra></extra>"
    )
    fig.update_layout(
        title=dict(text="Storm days per month — active years stand out", x=0),
        yaxis_title="Year",
    )
    return _finalize(fig, height=520)


def pr_curves(y_test: pd.Series, probs: dict[str, np.ndarray]) -> go.Figure:
    prevalence = float(np.mean(y_test))
    fig = go.Figure()
    for name, prob in probs.items():
        p, r, _ = precision_recall_curve(y_test, prob)
        fig.add_trace(
            go.Scatter(
                x=r, y=p, mode="lines", name=name,
                line=dict(color=PALETTE_MODELS.get(name, BLUE), width=2),
                hovertemplate="Recall %{x:.2f}<br>Precision %{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.add_hline(
        y=prevalence, line_dash="dot", line_color=GREY,
        annotation_text=f"No-skill rate ({prevalence:.1%})", annotation_position="top right",
    )
    fig.update_layout(title=dict(text="Precision–recall: learned models beat the baseline", x=0))
    return _finalize(fig, x_title="Recall", y_title="Precision")


def roc_curves(y_test: pd.Series, probs: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name=name,
                line=dict(color=PALETTE_MODELS.get(name, BLUE), width=2),
                hovertemplate="FPR %{x:.2f}<br>TPR %{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance",
            line=dict(color=GREY, width=1, dash="dot"), showlegend=False,
        )
    )
    fig.update_layout(title=dict(text="ROC curves on the chronological test set", x=0))
    return _finalize(fig, x_title="False positive rate", y_title="True positive rate")


def confusion_matrix_fig(y_true: pd.Series, pred: np.ndarray, title: str) -> go.Figure:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig = px.imshow(
        cm,
        x=["Predicted calm", "Predicted storm"],
        y=["Observed calm", "Observed storm"],
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(color="Days"),
    )
    fig.update_traces(hovertemplate="%{y} / %{x}<br>%{z} days<extra></extra>")
    fig.update_layout(title=dict(text=title, x=0))
    return _finalize(fig, height=360)


def threshold_curve(y_test: pd.Series, prob: np.ndarray, selected: float) -> go.Figure:
    thresholds = np.linspace(0.02, 0.98, 97)
    from sklearn.metrics import f1_score, precision_score, recall_score

    precision, recall, f1 = [], [], []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        precision.append(precision_score(y_test, pred, zero_division=0))
        recall.append(recall_score(y_test, pred, zero_division=0))
        f1.append(f1_score(y_test, pred, zero_division=0))
    fig = go.Figure()
    for values, name, color, dash in [
        (precision, "Precision", BLUE, "solid"),
        (recall, "Recall", ORANGE, "dash"),
        (f1, "F1", PURPLE, "dot"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=thresholds, y=values, mode="lines", name=name,
                line=dict(color=color, width=2, dash=dash),
                hovertemplate="%{x:.2f}: %{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.add_vline(x=selected, line_color=RED, line_dash="dash")
    fig.update_layout(
        title=dict(
            text=f"Precision / recall trade-off — marker at {selected:.2f}", x=0
        )
    )
    return _finalize(fig, x_title="Classification threshold", y_title="Score")


def importance_bars(importance: pd.Series) -> go.Figure:
    top = importance.head(10)
    df = top.reset_index()
    df.columns = ["feature", "value"]
    df = df.sort_values("value")
    colors = [ORANGE if i == len(df) - 1 else BLUE for i in range(len(df))]
    fig = go.Figure(
        go.Bar(
            x=df["value"], y=df["feature"], orientation="h",
            marker_color=colors,
            text=df["value"].map(lambda v: f"{v:.4f}"),
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Today's K-index variability is the strongest early signal", x=0)
    )
    return _finalize(
        fig,
        x_title="Decrease in average precision after shuffling",
        height=430,
    )


def feature_violin(daily: pd.DataFrame, feature: str) -> go.Figure:
    d = daily[[feature, "next_day_storm"]].dropna().copy()
    d["Next day"] = d["next_day_storm"].map({0: "Calm", 1: "Storm"})
    fig = px.violin(
        d, y=feature, x="Next day", color="Next day",
        color_discrete_map={"Calm": BLUE, "Storm": ORANGE},
        box=True, points=False,
        category_orders={"Next day": ["Calm", "Storm"]},
        title=f"Distribution of {feature} by next-day outcome",
    )
    fig.update_layout(title_x=0, showlegend=False)
    return _finalize(fig, y_title=feature, height=400)


def feature_scatter(daily: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    d = daily[[x_col, y_col, "next_day_storm"]].dropna().copy()
    d["Next day"] = d["next_day_storm"].map({0: "Calm", 1: "Storm"})
    fig = px.scatter(
        d, x=x_col, y=y_col, color="Next day", symbol="Next day",
        color_discrete_map={"Calm": BLUE, "Storm": ORANGE},
        symbol_map={"Calm": "circle", "Storm": "triangle-up"},
        opacity=0.55,
        category_orders={"Next day": ["Calm", "Storm"]},
        title=f"{y_col} vs {x_col} — storm days (triangles) sit higher",
    )
    fig.update_layout(title_x=0)
    return _finalize(fig, x_title=x_col, y_title=y_col)


def probability_gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Model-estimated chance of a storm tomorrow", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": BLUE},
                "steps": [
                    {"range": [0, 50], "color": "#EAF0F9"},
                    {"range": [50, 100], "color": "#FDEEDD"},
                ],
                "threshold": {
                    "line": {"color": RED, "width": 3},
                    "thickness": 0.85,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig
