"""End-to-end CLI: data -> features -> chronological split -> models -> reports."""

import argparse
from pathlib import Path

from .config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, RANDOM_STATE
from .data import load_and_validate
from .evaluation import (
    permutation_importance_report,
    plot_model_diagnostics,
    plot_raw_signals,
)
from .features import build_daily_table, leakage_audit
from .models import build_pipelines, chronological_split, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="forecast",
        description="Next-day geomagnetic storm risk benchmark.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory searched recursively for the CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where figures and metrics are written (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Compute metrics without rendering figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Random state: {RANDOM_STATE}")

    # 1. Locate and validate the dataset.
    geomag, solar = load_and_validate(args.data_dir)

    if not args.skip_plots:
        plot_raw_signals(geomag, solar, args.output_dir)

    # 2. Build the next-day table and audit it for leakage.
    daily = build_daily_table(geomag, solar)
    feature_cols = leakage_audit(daily)

    # 3. Chronological split.
    train, test = chronological_split(daily)
    X_train, y_train = train[feature_cols], train["next_day_storm"]
    X_test, y_test = test[feature_cols], test["next_day_storm"]

    # 4. Persistence baseline vs learned models.
    models = build_pipelines()
    results, probs, preds = evaluate(X_train, y_train, X_test, y_test, models)

    if not args.skip_plots:
        plot_model_diagnostics(y_test, preds, probs, results, args.output_dir)
        permutation_importance_report(
            models, results, X_test, y_test, feature_cols, args.output_dir
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved metrics: {results_path}")


if __name__ == "__main__":
    main()
