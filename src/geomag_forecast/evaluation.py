"""Figures and report printing (mirrors the original notebook sections)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import PrecisionRecallDisplay, classification_report, confusion_matrix

from .config import RANDOM_STATE, STORM_K_THRESHOLD

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def plot_raw_signals(geomag: pd.DataFrame, solar: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    geomag.set_index("Timestamp")["Estimated K"].resample("D").max().plot(
        ax=axes[0], color="#1f77b4"
    )
    axes[0].axhline(
        STORM_K_THRESHOLD, color="#d62728", linestyle="--", label=f"Storm threshold K={STORM_K_THRESHOLD}"
    )
    axes[0].set_title("Daily maximum estimated K index")
    axes[0].set_ylabel("Maximum K")
    axes[0].legend()
    solar.set_index("Date")["Sunspot Number"].plot(ax=axes[1], color="#2ca02c")
    axes[1].set_title("Daily sunspot number")
    axes[1].set_ylabel("Sunspot number")
    _save(fig, output_dir, "01_daily_k_and_sunspots.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(geomag["Estimated K"].dropna(), discrete=True, ax=ax, color="#9467bd")
    ax.set_title("Distribution of observed 3-hour estimated K values")
    ax.set_xlabel("Estimated K")
    _save(fig, output_dir, "02_estimated_k_distribution.png")


def plot_model_diagnostics(
    y_test: pd.Series,
    preds: dict[str, pd.Series],
    probs: dict[str, pd.Series],
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for name, pred in preds.items():
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
        axes[0].set_title(f"Confusion matrix: {name}")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Observed")
        break

    for name, prob in probs.items():
        PrecisionRecallDisplay.from_predictions(y_test, prob, name=name, ax=axes[1])
    axes[1].set_title("Precision-recall curves on chronological test set")
    _save(fig, output_dir, "03_confusion_matrix_and_pr_curves.png")

    best_name = results.iloc[0]["model"]
    print("Best by average precision on this executed split:", best_name)
    print("\nClassification report for the best learned model or baseline:")
    print(classification_report(y_test, preds[best_name], digits=3, zero_division=0))


def permutation_importance_report(
    models: dict[str, object],
    results: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    output_dir: Path,
) -> pd.Series:
    learned_names = list(models)
    best_learned = max(
        learned_names,
        key=lambda n: results.loc[results["model"] == n, "average_precision"].iloc[0],
    )
    perm = permutation_importance(
        models[best_learned],
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.Series(perm.importances_mean, index=feature_cols).sort_values(
        ascending=False
    )
    print("Permutation importance for:", best_learned)
    print(importance.head(10).round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    importance.head(10).sort_values().plot(kind="barh", ax=ax, color="#17becf")
    ax.set_title("Top held-out permutation importances")
    ax.set_xlabel("Decrease in average precision after shuffling")
    _save(fig, output_dir, "04_permutation_importance.png")
    return importance
