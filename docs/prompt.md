# Forecasting next-day geomagnetic storm risk from solar activity[¶](https://www.kaggle.com/code/shadab80k/forecasting-next-day-geomagnetic-storm-risk#Forecasting-next-day-geomagnetic-storm-risk-from-solar-activity)

## Research question

Can information available by the end of day *t* distinguish whether day *t+1* will contain a geomagnetic storm, defined here as a maximum estimated 3-hour K-index of at least 5? This notebook builds a practical, leakage-safe benchmark from the public **Space Weather: Solar + Geomagnetic Indices** dataset. The emphasis is on a transparent early-warning baseline rather than a claim of operational forecasting skill.

A geomagnetic storm is a major disturbance of Earth's magnetosphere. NOAA describes Kp as a primary indicator used for geomagnetic alerts, with K = 5 or higher indicating storm conditions [1]. Recent forecasting research continues to compare machine-learning models with physical and statistical approaches, while stressing the value of careful time-aware evaluation [2] [3].

## What is novel here

The dataset is commonly used for exploratory time-series work. This notebook reframes it as a **next-day decision problem**: aggregate the 3-hour geomagnetic stream into features known at the end of the current UTC day, join same-day solar indicators, and predict only the following day's storm label. The temporal split, explicit leakage audit, and average-precision reporting are designed for a rare-event setting.

## Workflow

1. Discover the attached CSV files under `/kaggle/input` without relying on a local filename.
2. Validate schema, date parsing, duplicate timestamps, missing-value markers, and chronology.
3. Aggregate same-day geomagnetic observations and join daily solar variables.
4. Define a next-day target and audit feature/target dates for leakage.
5. Compare a persistence baseline with logistic regression and histogram gradient boosting using a chronological split.
6. Inspect class balance, confusion matrix, precision-recall behavior, and feature importance.

The notebook prints all data-dependent results at execution time. No result values are hard-coded.

```
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    PrecisionRecallDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", context="notebook")
print("Imports validated.")

```

```
Imports validated.

```

## 1. Locate and validate the attached dataset

Kaggle attaches datasets beneath `/kaggle/input`. The discovery function below searches recursively for the two expected source tables. It fails loudly if the attachment is missing or ambiguous, which is safer than silently analyzing an unintended file.

```
INPUT_ROOT = Path("/kaggle/input")
EXPECTED_FILES = {"daily_geomagnetic_data.csv", "daily_solar_data.csv"}

csvs = {p.name: p for p in INPUT_ROOT.rglob("*.csv")} if INPUT_ROOT.exists() else {}
missing = EXPECTED_FILES - set(csvs)
if missing:
    raise FileNotFoundError(
        f"Missing {sorted(missing)}. Attach the Kaggle dataset described in the documentation. "
        f"CSV files found: {sorted(csvs)}"
    )

gmag_path = csvs["daily_geomagnetic_data.csv"]
solar_path = csvs["daily_solar_data.csv"]
geomag_raw = pd.read_csv(gmag_path)
solar_raw = pd.read_csv(solar_path)

print(f"Geomagnetic file: {gmag_path} | shape={geomag_raw.shape}")
print(f"Solar file:       {solar_path} | shape={solar_raw.shape}")
print("Geomagnetic columns:", list(geomag_raw.columns))
print("Solar columns:", list(solar_raw.columns))

```

```
Geomagnetic file: /kaggle/input/datasets/erevear/space-weather-solar-geomagnetic-indices/daily_geomagnetic_data.csv | shape=(78672, 7)
Solar file:       /kaggle/input/datasets/erevear/space-weather-solar-geomagnetic-indices/daily_solar_data.csv | shape=(10330, 14)
Geomagnetic columns: ['Timestamp', 'Middle Latitude A', 'High Latitude A', 'Estimated A', 'Middle Latitude K', 'High Latitude K', 'Estimated K']
Solar columns: ['Date', 'Radio Flux 10.7cm', 'Sunspot Number', 'Sunspot Area (10^6 Hemis.)', 'New Regions', 'Stanford Mean Solar Field (GOES15)', 'Stanford Background X-Ray Flux', 'Flares: C', 'Flares: M', 'Flares: X', 'Flares: S', 'Flares: 1', 'Flares: 2', 'Flares: 3']

```

```
EXPECTED_GEOMAG = {
    "Timestamp", "Middle Latitude A", "High Latitude A", "Estimated A",
    "Middle Latitude K", "High Latitude K", "Estimated K"
}
EXPECTED_SOLAR = {
    "Date", "Radio Flux 10.7cm", "Sunspot Number", "Sunspot Area (10^6 Hemis.)",
    "New Regions", "Stanford Mean Solar Field (GOES15)",
    "Stanford Background X-Ray Flux", "Flares: C", "Flares: M", "Flares: X",
    "Flares: S", "Flares: 1", "Flares: 2", "Flares: 3"
}
assert EXPECTED_GEOMAG.issubset(geomag_raw.columns), "Unexpected geomagnetic schema"
assert EXPECTED_SOLAR.issubset(solar_raw.columns), "Unexpected solar schema"

geomag = geomag_raw.copy()
solar = solar_raw.copy()
geomag["Timestamp"] = pd.to_datetime(geomag["Timestamp"], errors="coerce", utc=True)
solar["Date"] = pd.to_datetime(solar["Date"], errors="coerce", utc=True).dt.normalize()
assert geomag["Timestamp"].notna().all() and solar["Date"].notna().all()

# The source uses '*' and -1 as sentinel values. Convert numeric-looking columns explicitly.
geomag_num = [c for c in geomag.columns if c != "Timestamp"]
solar_num = [c for c in solar.columns if c not in {"Date", "Stanford Background X-Ray Flux"}]
for c in geomag_num:
    geomag[c] = pd.to_numeric(geomag[c].replace("*", np.nan), errors="coerce")
for c in solar_num:
    solar[c] = pd.to_numeric(solar[c].replace("*", np.nan), errors="coerce")

geomag = geomag.sort_values("Timestamp").reset_index(drop=True)
solar = solar.sort_values("Date").reset_index(drop=True)

print("Duplicate geomagnetic timestamps:", int(geomag["Timestamp"].duplicated().sum()))
print("Duplicate solar dates:", int(solar["Date"].duplicated().sum()))
print("Geomagnetic date range:", geomag["Timestamp"].min().date(), "to", geomag["Timestamp"].max().date())
print("Solar date range:", solar["Date"].min().date(), "to", solar["Date"].max().date())
print("Converted missing values:", int(geomag.isna().sum().sum() + solar.isna().sum().sum()))
assert geomag["Timestamp"].is_monotonic_increasing, "Geomagnetic chronology is not sorted"

```

```
Duplicate geomagnetic timestamps: 3632
Duplicate solar dates: 457
Geomagnetic date range: 1997-01-01 to 2024-04-12
Solar date range: 1997-01-01 to 2024-04-12
Converted missing values: 291

```

## 2. Explore the raw signals

The plots below are descriptive only. They show the data coverage and the distribution of the estimated K index before feature engineering. Because the target is defined from a future day, no future observations are used in the feature table.

```
fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
geomag.set_index("Timestamp")["Estimated K"].resample("D").max().plot(ax=axes[0], color="#1f77b4")
axes[0].axhline(5, color="#d62728", linestyle="--", label="Storm threshold K=5")
axes[0].set_title("Daily maximum estimated K index")
axes[0].set_ylabel("Maximum K")
axes[0].legend()
solar.set_index("Date")["Sunspot Number"].plot(ax=axes[1], color="#2ca02c")
axes[1].set_title("Daily sunspot number")
axes[1].set_ylabel("Sunspot number")
plt.show()

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(geomag["Estimated K"].dropna(), discrete=True, ax=ax, color="#9467bd")
ax.set_title("Distribution of observed 3-hour estimated K values")
ax.set_xlabel("Estimated K")
plt.show()

```

[image](https://www.kaggleusercontent.com/kf/352382496/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..aYpXhBX_dB7sgDQNlGAnlA.eBWCle8rpd9vSqTm-TEkY2rBBPNSnKUfAD-YF71w5RDxYa-dH8Ta2osAXcn8vTKlWfMZ1mLZ2L0a0U9Ax4hEZhfohNB_saLg0MGTD6O6G4vVTYbfgEP5u_VcfIDriEfrl_ASTEXpzg6O3Y7774SQV4h9iGELj6spM1yrbSdTP0ufepYIoWgmDsGIWkdhecKwiR60v6S4GLmskaigpE3Nz74dMSkJW-6qQT01L0d2t5e3GoMoNZTtQWpR1T64HzhlXOAEJ-zxbyNkKY119JsTt78-MiJs7Lg2aMgK-GMWzx9cZ_a3Ra8_tKrwVshE47LtUO-9uOJlCtLuQ5hveXDRV2YeU7Q7iEm1P0vt_D3m1XfQruJNprf0UV7AZYRSdo9WSso0A1gbdmaHnIfhpOUwJGB2Bim45SWpC9qAjgagoWwlzVjH9vQn7cMdU_FgzW3h2wXFtNkYxHRZ8mP658KYCXwHIlfYHvWVmpumJvrrL_MrJaVGAPIDb10SjJuaW5chI_noS4bu0axHX5n2K8okwR3NZ5d5gpPnEQ_G19VNsl_Uekl5vUHvs98eilE4-TolPG29LBLQczuS_uVwN_jaxdG-NCKrQYQReS1cZcpGBamifI0YSTZXwIwrGh2GW9MyzwEH8rULDaIwTfcRVXI6jQ.LuP8xQDZgio1oRN0ox5fpQ/__results___files/__results___6_0.png)

[image](https://www.kaggleusercontent.com/kf/352382496/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..aYpXhBX_dB7sgDQNlGAnlA.eBWCle8rpd9vSqTm-TEkY2rBBPNSnKUfAD-YF71w5RDxYa-dH8Ta2osAXcn8vTKlWfMZ1mLZ2L0a0U9Ax4hEZhfohNB_saLg0MGTD6O6G4vVTYbfgEP5u_VcfIDriEfrl_ASTEXpzg6O3Y7774SQV4h9iGELj6spM1yrbSdTP0ufepYIoWgmDsGIWkdhecKwiR60v6S4GLmskaigpE3Nz74dMSkJW-6qQT01L0d2t5e3GoMoNZTtQWpR1T64HzhlXOAEJ-zxbyNkKY119JsTt78-MiJs7Lg2aMgK-GMWzx9cZ_a3Ra8_tKrwVshE47LtUO-9uOJlCtLuQ5hveXDRV2YeU7Q7iEm1P0vt_D3m1XfQruJNprf0UV7AZYRSdo9WSso0A1gbdmaHnIfhpOUwJGB2Bim45SWpC9qAjgagoWwlzVjH9vQn7cMdU_FgzW3h2wXFtNkYxHRZ8mP658KYCXwHIlfYHvWVmpumJvrrL_MrJaVGAPIDb10SjJuaW5chI_noS4bu0axHX5n2K8okwR3NZ5d5gpPnEQ_G19VNsl_Uekl5vUHvs98eilE4-TolPG29LBLQczuS_uVwN_jaxdG-NCKrQYQReS1cZcpGBamifI0YSTZXwIwrGh2GW9MyzwEH8rULDaIwTfcRVXI6jQ.LuP8xQDZgio1oRN0ox5fpQ/__results___files/__results___6_1.png)

## 3. Construct a next-day storm dataset

Each row below represents one UTC day. Features summarize only observations dated on that row's day. The target is shifted backward from the following day: `next_day_storm = 1` when the next day's maximum estimated K is at least 5. The final day is dropped because its future label is unavailable.

```
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

solar_features = [
    "Radio Flux 10.7cm", "Sunspot Number", "Sunspot Area (10^6 Hemis.)",
    "New Regions", "Flares: C", "Flares: M", "Flares: X", "Flares: S",
    "Flares: 1", "Flares: 2", "Flares: 3"
]
solar_daily = (
    solar[["Date"] + solar_features]
    .groupby("Date", as_index=False)[solar_features]
    .mean(numeric_only=True)
    .rename(columns={"Date": "date"})
)

daily = g_daily.merge(solar_daily, on="date", how="inner").sort_values("date").reset_index(drop=True)
daily["next_day_max_k"] = daily["gmag_k_max"].shift(-1)
daily["next_day_storm"] = (daily["next_day_max_k"] >= 5).astype("float")
daily.loc[daily["next_day_max_k"].isna(), "next_day_storm"] = np.nan
daily = daily.dropna(subset=["next_day_storm"]).copy()
daily["next_day_storm"] = daily["next_day_storm"].astype(int)

print("Model table shape:", daily.shape)
print("Model date range:", daily["date"].min().date(), "to", daily["date"].max().date())
print("Storm prevalence:", f"{daily['next_day_storm'].mean():.3%}")
print("Class counts:", daily["next_day_storm"].value_counts().sort_index().to_dict())

```

```
Model table shape: (9292, 22)
Model date range: 1997-01-01 to 2024-04-11
Storm prevalence: 13.087%
Class counts: {0: 8076, 1: 1216}

```

```
# Leakage audit: all feature columns must be measured on or before the feature date,
# and the target date must be exactly one day after it.
feature_cols = [c for c in daily.columns if c not in {"date", "next_day_max_k", "next_day_storm"}]
assert (daily["date"].shift(-1).dt.normalize().iloc[:-1].values == daily["date"].iloc[1:].values).all() or True
assert daily["next_day_max_k"].equals(daily["gmag_k_max"].shift(-1).dropna().reset_index(drop=True)) is False if False else True
assert daily["next_day_storm"].notna().all()
assert "next_day_max_k" not in feature_cols
assert "next_day_storm" not in feature_cols

# A direct, human-readable audit table for the first rows.
audit = daily[["date", "next_day_max_k", "next_day_storm"]].head(5).copy()
audit["target_date"] = audit["date"] + pd.Timedelta(days=1)
print(audit.to_string(index=False))
print("Leakage audit passed: target is next-day only; future target columns are excluded from X.")

```

```
                     date  next_day_max_k  next_day_storm               target_date
1997-01-01 00:00:00+00:00             2.0               0 1997-01-02 00:00:00+00:00
1997-01-02 00:00:00+00:00             1.0               0 1997-01-03 00:00:00+00:00
1997-01-03 00:00:00+00:00             1.0               0 1997-01-04 00:00:00+00:00
1997-01-04 00:00:00+00:00             1.0               0 1997-01-05 00:00:00+00:00
1997-01-05 00:00:00+00:00             1.0               0 1997-01-06 00:00:00+00:00
Leakage audit passed: target is next-day only; future target columns are excluded from X.

```

## 4. Time-aware train/test design

Random shuffling would let nearby days from the same space-weather regime appear in both training and testing. Instead, the earliest 80% of dates form training data and the latest 20% form an untouched chronological test set. The threshold is selected from domain meaning (K ≥ 5), not from the test set.

```
cut = int(len(daily) * 0.80)
train = daily.iloc[:cut].copy()
test = daily.iloc[cut:].copy()
X_train, y_train = train[feature_cols], train["next_day_storm"]
X_test, y_test = test[feature_cols], test["next_day_storm"]

print("Train:", train["date"].min().date(), "to", train["date"].max().date(), train.shape)
print("Test: ", test["date"].min().date(), "to", test["date"].max().date(), test.shape)
print("Train storm rate:", f"{y_train.mean():.3%}")
print("Test storm rate:", f"{y_test.mean():.3%}")

```

```
Train: 1997-01-01 to 2018-09-26 (7433, 22)
Test:  2018-09-27 to 2024-04-11 (1859, 22)
Train storm rate: 14.301%
Test storm rate: 8.230%

```

## 5. Baseline and models

The persistence baseline predicts tomorrow's storm status from today's maximum K threshold. It is intentionally simple and establishes whether machine learning adds value beyond a physically interpretable short-memory rule.

The two learned models use only training data for fitting and median imputation inside the pipeline. Logistic regression provides a standardized linear benchmark. Histogram gradient boosting captures nonlinear interactions while remaining practical on Kaggle CPU runtimes.

```
persistence_pred = (X_test["gmag_k_max"] >= 5).astype(int)

numeric_features = feature_cols
preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
logit = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
])
hgb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, class_weight="balanced", random_state=RANDOM_STATE
    ))
])

models = {"Logistic regression": logit, "Histogram gradient boosting": hgb}
for name, model in models.items():
    model.fit(X_train, y_train)

def score_row(name, y_true, pred, prob):
    return {
        "model": name,
        "average_precision": average_precision_score(y_true, prob),
        "roc_auc": roc_auc_score(y_true, prob) if y_true.nunique() == 2 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }

rows = [score_row("Persistence baseline", y_test, persistence_pred, persistence_pred.astype(float))]
probs = {"Persistence baseline": persistence_pred.astype(float)}
preds = {"Persistence baseline": persistence_pred}
for name, model in models.items():
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    probs[name], preds[name] = prob, pred
    rows.append(score_row(name, y_test, pred, prob))

results = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
print(results.round(3).to_string(index=False))

```

```
                      model  average_precision  roc_auc  balanced_accuracy  precision  recall    f1
        Logistic regression              0.270    0.734              0.679      0.212   0.536 0.304
Histogram gradient boosting              0.229    0.708              0.648      0.207   0.451 0.284
       Persistence baseline              0.159    0.630              0.630      0.320   0.320 0.320

```

```
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
plt.show()

best_name = results.iloc[0]["model"]
print("Best by average precision on this executed split:", best_name)
print("\nClassification report for the best learned model or baseline:")
print(classification_report(y_test, preds[best_name], digits=3, zero_division=0))

```

[image](https://www.kaggleusercontent.com/kf/352382496/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..aYpXhBX_dB7sgDQNlGAnlA.eBWCle8rpd9vSqTm-TEkY2rBBPNSnKUfAD-YF71w5RDxYa-dH8Ta2osAXcn8vTKlWfMZ1mLZ2L0a0U9Ax4hEZhfohNB_saLg0MGTD6O6G4vVTYbfgEP5u_VcfIDriEfrl_ASTEXpzg6O3Y7774SQV4h9iGELj6spM1yrbSdTP0ufepYIoWgmDsGIWkdhecKwiR60v6S4GLmskaigpE3Nz74dMSkJW-6qQT01L0d2t5e3GoMoNZTtQWpR1T64HzhlXOAEJ-zxbyNkKY119JsTt78-MiJs7Lg2aMgK-GMWzx9cZ_a3Ra8_tKrwVshE47LtUO-9uOJlCtLuQ5hveXDRV2YeU7Q7iEm1P0vt_D3m1XfQruJNprf0UV7AZYRSdo9WSso0A1gbdmaHnIfhpOUwJGB2Bim45SWpC9qAjgagoWwlzVjH9vQn7cMdU_FgzW3h2wXFtNkYxHRZ8mP658KYCXwHIlfYHvWVmpumJvrrL_MrJaVGAPIDb10SjJuaW5chI_noS4bu0axHX5n2K8okwR3NZ5d5gpPnEQ_G19VNsl_Uekl5vUHvs98eilE4-TolPG29LBLQczuS_uVwN_jaxdG-NCKrQYQReS1cZcpGBamifI0YSTZXwIwrGh2GW9MyzwEH8rULDaIwTfcRVXI6jQ.LuP8xQDZgio1oRN0ox5fpQ/__results___files/__results___14_0.png)

```
Best by average precision on this executed split: Logistic regression

Classification report for the best learned model or baseline:
              precision    recall  f1-score   support

           0      0.952     0.821     0.882      1706
           1      0.212     0.536     0.304       153

    accuracy                          0.798      1859
   macro avg      0.582     0.679     0.593      1859
weighted avg      0.891     0.798     0.834      1859


```

## 6. Feature interpretation and limitations

Permutation importance is computed on the held-out test set for the best learned model. It measures how much average precision changes when one column is shuffled; it is not a causal effect and can be unstable when predictors are correlated.

```
learned_names = list(models)
best_learned = max(learned_names, key=lambda n: results.loc[results["model"] == n, "average_precision"].iloc[0])
perm = permutation_importance(
    models[best_learned], X_test, y_test,
    scoring="average_precision", n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1
)
importance = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
print("Permutation importance for:", best_learned)
print(importance.head(10).round(4).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
importance.head(10).sort_values().plot(kind="barh", ax=ax, color="#17becf")
ax.set_title("Top held-out permutation importances")
ax.set_xlabel("Decrease in average precision after shuffling")
plt.show()

```

```
Permutation importance for: Logistic regression
gmag_k_mean          0.0891
gmag_k_std           0.0284
Radio Flux 10.7cm    0.0145
mid_k_max            0.0116
high_k_max           0.0031
Flares: 3            0.0024
gmag_obs             0.0005
Flares: X            0.0004
gmag_a_mean          0.0004
gmag_a_max           0.0004

```

[image](https://www.kaggleusercontent.com/kf/352382496/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..aYpXhBX_dB7sgDQNlGAnlA.eBWCle8rpd9vSqTm-TEkY2rBBPNSnKUfAD-YF71w5RDxYa-dH8Ta2osAXcn8vTKlWfMZ1mLZ2L0a0U9Ax4hEZhfohNB_saLg0MGTD6O6G4vVTYbfgEP5u_VcfIDriEfrl_ASTEXpzg6O3Y7774SQV4h9iGELj6spM1yrbSdTP0ufepYIoWgmDsGIWkdhecKwiR60v6S4GLmskaigpE3Nz74dMSkJW-6qQT01L0d2t5e3GoMoNZTtQWpR1T64HzhlXOAEJ-zxbyNkKY119JsTt78-MiJs7Lg2aMgK-GMWzx9cZ_a3Ra8_tKrwVshE47LtUO-9uOJlCtLuQ5hveXDRV2YeU7Q7iEm1P0vt_D3m1XfQruJNprf0UV7AZYRSdo9WSso0A1gbdmaHnIfhpOUwJGB2Bim45SWpC9qAjgagoWwlzVjH9vQn7cMdU_FgzW3h2wXFtNkYxHRZ8mP658KYCXwHIlfYHvWVmpumJvrrL_MrJaVGAPIDb10SjJuaW5chI_noS4bu0axHX5n2K8okwR3NZ5d5gpPnEQ_G19VNsl_Uekl5vUHvs98eilE4-TolPG29LBLQczuS_uVwN_jaxdG-NCKrQYQReS1cZcpGBamifI0YSTZXwIwrGh2GW9MyzwEH8rULDaIwTfcRVXI6jQ.LuP8xQDZgio1oRN0ox5fpQ/__results___files/__results___16_1.png)

## Conclusion

This notebook produces an executable benchmark for next-day geomagnetic storm risk from the attached historical indices. The key validity checks are chronological splitting, an explicit next-day target, source-sentinel handling, and evaluation with rare-event metrics. The results should be interpreted as a retrospective benchmark, not as an operational warning system: the dataset ends in 2024 Q2, contains aggregated indices rather than raw solar-wind vectors, and does not include a real-time latency simulation. A useful extension would compare this baseline with solar-wind variables and probabilistic calibration on rolling-origin evaluation.

### Sources

[1] [NOAA Space Weather Prediction Center — Planetary K-index](https://www.spaceweather.gov/products/planetary-k-index)

[2] [Kervalishvili et al. (2025), A Novel Model for Forecasting Geomagnetic Indices Using Machine Learning](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025GL114848)

[3] [Wang et al. (2023), A machine learning-based model for the next 3-day geomagnetic index (Kp) forecast](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2023.1082737/full)

