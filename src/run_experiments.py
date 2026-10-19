from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Five configurations to compare
EXPERIMENT_CONFIGS = [
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 150, "max_depth": None},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 200, "max_depth": 20},
]


def run_experiments():
    # Load the project configuration
    with Path("configs/config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Connect to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("California Housing Price Prediction")

    # Load the dataset
    df = pd.read_csv(config["data"]["input_path"])

    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )
    
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    # Build preprocessing pipelines
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    experiment_results = []

    # Train and log all five configurations
    for experiment in EXPERIMENT_CONFIGS:
        n_estimators = experiment["n_estimators"]
        max_depth = experiment["max_depth"]

        run_name = f"RandomForest_{n_estimators}_trees_depth_{max_depth}"

        print(f"\nTraining {run_name}...")

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=config["model"]["random_state"],
                n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        experiment_results.append({
            "Model": run_name,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        })

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", str(max_depth))
            mlflow.log_param("random_state", config["model"]["random_state"])
            mlflow.log_param("test_size", config["split"]["test_size"])

            mlflow.log_metric("mae", float(mae))
            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("r2", float(r2))

            
            # Save this experiment's trained model
            artifact_dir = Path("models/experiment_artifacts")
            artifact_dir.mkdir(parents=True, exist_ok=True)

            artifact_path = artifact_dir / f"{run_name}.joblib"
            joblib.dump(model, artifact_path)

            mlflow.log_artifact(str(artifact_path), artifact_path="model")

            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2: {r2:.4f}")

    # Save experiment results for DVC
    import json

    output_path = Path("reports/experiments_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_results.sort(
        key=lambda result: result["R2"],
        reverse=True
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(experiment_results, file, indent=4)

    print(f"\nExperiment results saved to: {output_path}")

    print("\nAll five experiments completed successfully!")


if __name__ == "__main__":
    run_experiments()