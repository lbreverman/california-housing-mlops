
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_model():
    # Step 1: Load the prepared dataset
    data_path = Path("data/california_housing.csv")
    df = pd.read_csv(data_path)

    # Step 2: Separate features and target
    X = df.drop(columns=["MedHouseValue"])
    y = df["MedHouseValue"]

    # Step 3: Identify feature types
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Step 4: Build preprocessing pipelines
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

    # Step 5: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Step 6: Create the machine learning pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Step 7: Train the model
    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    # Step 8: Evaluate the model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Step 9: Save the trained model
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "random_forest_model.joblib"
    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    train_model()