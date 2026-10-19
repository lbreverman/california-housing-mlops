
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_model():
    # Step 1: Load configuration
    config_path = Path("configs/config.yaml")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Step 2: Load the prepared dataset
    data_path = Path(config["data"]["input_path"])
    df = pd.read_csv(data_path)

    # Step 3: Separate features and target
    target_column = config["data"]["target_column"]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 4: Identify feature types
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Step 5: Build preprocessing pipelines
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

    # Step 6: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"]
    )

    # Step 7: Create the machine learning pipeline
    model_config = config["model"]

    if model_config["type"] != "RandomForestRegressor":
        raise ValueError("Unsupported model type in configuration")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            random_state=model_config["random_state"],
            n_jobs=-1
        ))
    ])

    # Step 8: Train the model
    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    # Step 9: Evaluate the model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Step 10: Save the trained model
    model_path = Path(config["paths"]["model_output"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    train_model()