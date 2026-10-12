
from sklearn.datasets import fetch_california_housing


def test_california_housing_dataset():
    """Verify the California Housing dataset loads correctly."""
    housing = fetch_california_housing()

    assert housing.data.shape[0] == 20640
    assert housing.data.shape[1] == 8
    assert len(housing.target) == 20640
    assert not any(housing.feature_names is None for _ in [0])