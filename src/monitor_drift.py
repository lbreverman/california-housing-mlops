from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from evidently import Report
from evidently.presets import DataDriftPreset



def monitor_drift():
    # Step 1: Load the prepared dataset
    data_path = Path("data/california_housing.csv")
    df = pd.read_csv(data_path)

    # Step 2: Separate features from the target
    X = df.drop(columns=["MedHouseValue"])

    # Step 3: Create reference and new data samples
    # These simulate historical and incoming data.
    reference_data = X.sample(frac=0.7, random_state=42)
    new_data = X.drop(index=reference_data.index).copy()

    # Step 4: Simulate a change in incoming data
    # This demonstrates whether monitoring detects drift.
    new_data["MedInc"] = new_data["MedInc"] * 1.20

    # Step 5: Select numerical features
    numerical_features = X.select_dtypes(
        include=[np.number]
    ).columns

    # Step 6: Detect drift using the KS test
    drift_results = {}
    alpha = 0.05

    print("California Housing Data Drift Monitoring")
    print("-" * 45)

    for feature in numerical_features:
        reference_values = reference_data[feature].dropna()
        new_values = new_data[feature].dropna()

        statistic, p_value = ks_2samp(
            reference_values,
            new_values
        )

        drift_detected = bool(p_value < alpha)

        drift_results[feature] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": float(p_value),
            "drift_detected": drift_detected
        }

        status = "DRIFT DETECTED" if drift_detected else "No drift"

        print(f"{feature}: {status} (p={p_value:.6f})")

    # Step 7: Calculate overall drift summary
    total_features = len(drift_results)

    drifted_features = sum(
        result["drift_detected"]
        for result in drift_results.values()
    )

    drift_percentage = (
        drifted_features / total_features * 100
        if total_features > 0 else 0
    )

    print("\nDrift Monitoring Summary")
    print("-" * 45)
    print(f"Total numerical features: {total_features}")
    print(f"Features with drift: {drifted_features}")
    print(f"Drift percentage: {drift_percentage:.2f}%")

    # Step 8: Save monitoring results
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "drift_report.json"

    report = {
        "monitoring_type": "simulated_data_drift",
        "reference_rows": len(reference_data),
        "new_data_rows": len(new_data),
        "significance_level": alpha,
        "total_features": total_features,
        "drifted_features": drifted_features,
        "drift_percentage": round(drift_percentage, 2),
        "feature_results": drift_results
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nDrift report saved to: {report_path}")

    # Step 9: Generate an Evidently HTML drift report
    print("\nGenerating Evidently HTML drift report...")

    evidently_report = Report(
        metrics=[DataDriftPreset()]
    )

    evidently_result = evidently_report.run(
        reference_data=reference_data,
        current_data=new_data
    )

    html_path = report_dir / "drift_report.html"
    evidently_result.save_html(str(html_path))

    print(f"Evidently HTML report saved to: {html_path}")


if __name__ == "__main__":
    monitor_drift()