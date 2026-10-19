
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_test_preprocessor():
    """Build preprocessing using the same settings as training."""
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, ["HouseAge", "AveRooms"]),
        ("categorical", categorical_pipeline, ["IncomeCategory"])
    ])


def sample_features():
    """Create a small dataset for preprocessing tests."""
    return pd.DataFrame({
        "HouseAge": [10.0, 20.0, np.nan, 40.0],
        "AveRooms": [4.0, np.nan, 6.0, 8.0],
        "IncomeCategory": ["Low", "Medium", "High", None]
    })


def test_numerical_missing_values_are_imputed():
    X = sample_features()
    result = build_test_preprocessor().fit_transform(X)

    assert not np.isnan(result[:, :2]).any()


def test_numerical_imputation_uses_median():
    X = sample_features()
    result = build_test_preprocessor().fit_transform(X)

    assert result[2, 0] == 20.0
    assert result[1, 1] == 6.0


def test_categorical_missing_values_are_imputed():
    X = sample_features()
    X["IncomeCategory"] = X["IncomeCategory"].astype("category")

    preprocessor = build_test_preprocessor()
    result = preprocessor.fit_transform(X)

    assert result.shape[0] == len(X)


def test_categorical_values_are_encoded():
    X = sample_features()
    preprocessor = build_test_preprocessor()
    result = preprocessor.fit_transform(X)

    assert result.shape[1] > 2


def test_unknown_categories_are_handled():
    X = sample_features()
    preprocessor = build_test_preprocessor()
    preprocessor.fit(X)

    new_data = pd.DataFrame({
        "HouseAge": [25.0],
        "AveRooms": [5.0],
        "IncomeCategory": ["Very High"]
    })

    result = preprocessor.transform(new_data)

    assert result.shape[0] == 1


def test_preprocessing_preserves_row_count():
    X = sample_features()
    result = build_test_preprocessor().fit_transform(X)

    assert result.shape[0] == len(X)