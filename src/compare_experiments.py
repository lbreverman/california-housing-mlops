import json
from pathlib import Path

import pandas as pd


def compare_experiments():
    input_path = Path("reports/experiments_manifest.json")

    if not input_path.exists():
        raise FileNotFoundError(
            "Experiment results not found. Run the experiment suite first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        experiments = json.load(file)

    if not experiments:
        raise ValueError("No experiment results found.")

    results_df = pd.DataFrame(experiments)
    results_df = results_df.sort_values(
        by="R2",
        ascending=False
    )

    print("\nExperiment Comparison:")
    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]

    print("\nBest Performing Experiment:")
    print(f"Model: {best_model['Model']}")
    print(f"R2 Score: {best_model['R2']:.4f}")
    print(f"MAE: {best_model['MAE']:.4f}")
    print(f"RMSE: {best_model['RMSE']:.4f}")

    output_path = Path("reports/model_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results_df.to_dict(orient="records"), file, indent=4)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    compare_experiments()