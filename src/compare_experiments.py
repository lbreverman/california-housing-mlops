
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compare_experiments():
    # Step 1: Load dataset
    df = pd.read_csv("data/california_housing.csv")

    X = df.drop(columns=["MedHouseValue"])
    y = df["MedHouseValue"]

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Identify feature types
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Step 4: Create preprocessing pipeline
    numerical_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Step 5: Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    # Step 6: Train and evaluate each model
    for name, estimator in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(
            y_test, predictions
        ) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4)
        })

    # Step 7: Compare results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="R2", ascending=False
    )

    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False))

    # Step 8: Save comparison results
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "model_comparison.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results_df.to_dict(orient="records"),
            file,
            indent=4
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    compare_experiments()