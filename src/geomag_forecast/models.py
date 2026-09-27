"""Chronological split, model pipelines, and metric computation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, STORM_K_THRESHOLD, TRAIN_FRACTION


def chronological_split(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(daily) * TRAIN_FRACTION)
    train = daily.iloc[:cut].copy()
    test = daily.iloc[cut:].copy()

    print(
        "Train:",
        train["date"].min().date(),
        "to",
        train["date"].max().date(),
        train.shape,
    )
    print(
        "Test: ",
        test["date"].min().date(),
        "to",
        test["date"].max().date(),
        test.shape,
    )
    print("Train storm rate:", f"{train['next_day_storm'].mean():.3%}")
    print("Test storm rate:", f"{test['next_day_storm'].mean():.3%}")
    return train, test


def build_pipelines() -> dict[str, Pipeline]:
    logit = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    hgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {
        "Logistic regression": logit,
        "Histogram gradient boosting": hgb,
    }


def score_row(name: str, y_true: pd.Series, pred: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "model": name,
        "average_precision": average_precision_score(y_true, prob),
        "roc_auc": roc_auc_score(y_true, prob) if y_true.nunique() == 2 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, Pipeline],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit the learned models and score everything against the test set."""
    for model in models.values():
        model.fit(X_train, y_train)

    persistence_pred = (X_test["gmag_k_max"] >= STORM_K_THRESHOLD).astype(int)
    probs: dict[str, np.ndarray] = {
        "Persistence baseline": persistence_pred.astype(float)
    }
    preds: dict[str, np.ndarray] = {"Persistence baseline": persistence_pred}
    rows = [
        score_row(
            "Persistence baseline", y_test, persistence_pred, persistence_pred.astype(float)
        )
    ]

    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        probs[name], preds[name] = prob, pred
        rows.append(score_row(name, y_test, pred, prob))

    results = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    print(results.round(3).to_string(index=False))
    return results, probs, preds
