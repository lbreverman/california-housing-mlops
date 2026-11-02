from pathlib import Path
import json

import mlflow
import pandas as pd


def compare_experiments():
    # Connect to the existing MLflow database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Retrieve all completed experiment runs
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string="attributes.status = 'FINISHED'"
    )

    if runs.empty:
        raise ValueError("No completed MLflow runs found.")

    # Require the metric used for model selection
    runs = runs.dropna(subset=["metrics.r2"])

    if runs.empty:
        raise ValueError("No runs with R2 metrics found.")

    # Sort by R2, highest first
    runs = runs.sort_values(
        by="metrics.r2",
        ascending=False
    )

    columns = [
        "run_id",
        "params.model_type",
        "params.n_estimators",
        "params.max_depth",
        "metrics.r2",
        "metrics.mae",
        "metrics.rmse"
    ]

    available_columns = [
        col for col in columns if col in runs.columns
    ]

    results = runs[available_columns].copy()

    print("\nMLflow Experiment Comparison:")
    print(results.head(10).to_string(index=False))

    best_run = runs.iloc[0]

    print("\nBest Performing MLflow Run:")
    print(f"Run ID: {best_run['run_id']}")
    print(f"R2: {best_run['metrics.r2']:.6f}")

    if pd.notna(best_run.get("metrics.mae")):
        print(f"MAE: {best_run['metrics.mae']:.6f}")

    if pd.notna(best_run.get("metrics.rmse")):
        print(f"RMSE: {best_run['metrics.rmse']:.6f}")

    # Save comparison results
    output_path = Path("reports/model_comparison.json")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results = results.where(pd.notna(results), None)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results.to_dict(orient="records"),
            file,
            indent=4,
            default=str
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    compare_experiments()