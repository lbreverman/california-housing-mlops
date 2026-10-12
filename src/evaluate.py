
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


def evaluate_model():
    # Step 1: Load the prepared dataset
    data_path = Path("data/california_housing.csv")
    df = pd.read_csv(data_path)

    # Step 2: Separate features and target
    X = df.drop(columns=["MedHouseValue"])
    y = df["MedHouseValue"]

    # Step 3: Recreate the same train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Step 4: Load the trained model
    model_path = Path("models/random_forest_model.joblib")
    model = joblib.load(model_path)

    # Step 5: Make predictions
    predictions = model.predict(X_test)

    # Step 6: Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    # Step 7: Display results
    print("\nModel Evaluation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Step 8: Save evaluation metrics
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    output_path = reports_dir / "metrics.json"

    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nEvaluation results saved to: {output_path}")


if __name__ == "__main__":
    evaluate_model()