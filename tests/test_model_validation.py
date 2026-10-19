
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="module")
def model_and_data():
    """Load the actual trained model and prepared dataset."""
    with open("configs/config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_path = Path(config["paths"]["model_output"])
    data_path = Path(config["data"]["input_path"])

    assert model_path.exists(), f"Model not found: {model_path}"
    assert data_path.exists(), f"Dataset not found: {data_path}"

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    features = df.drop(columns=[config["data"]["target_column"]])

    return model, features


def test_saved_model_structure(model_and_data):
    """Verify the saved model contains the expected pipeline."""
    model, _ = model_and_data

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "regressor" in model.named_steps
    assert isinstance(
        model.named_steps["regressor"],
        RandomForestRegressor
    )


def test_model_produces_valid_predictions(model_and_data):
    """Verify the trained model produces finite predictions."""
    model, features = model_and_data

    sample = features.head(10)
    predictions = model.predict(sample)

    assert len(predictions) == len(sample)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()