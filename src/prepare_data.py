
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def prepare_dataset():
    # Step 1: Download the California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    # Step 2: Rename the target column
    df = df.rename(
        columns={"MedHouseVal": "MedHouseValue"}
    )

    # Step 3: Create a categorical feature
    df["IncomeCategory"] = pd.cut(
        df["MedInc"],
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["Low", "Medium", "High", "Very High"]
    )

    # Step 4: Introduce reproducible missing values
    rng = np.random.default_rng(42)

    columns_with_missing = [
        "HouseAge",
        "AveRooms",
        "AveOccup"
    ]

    for column in columns_with_missing:
        missing_indices = rng.choice(
            df.index,
            size=int(len(df) * 0.05),
            replace=False
        )

        df.loc[missing_indices, column] = np.nan

    # Step 5: Create the data directory
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 6: Save the dataset
    output_path = output_dir / "california_housing.csv"
    df.to_csv(output_path, index=False)

    # Step 7: Display dataset information
    print("Dataset prepared successfully!")
    print(f"\nDataset shape: {df.shape}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst five rows:")
    print(df.head())

    print(f"\nDataset saved to: {output_path}")



if __name__ == "__main__":
    try:
        prepare_dataset()
    except Exception as error:
        backup_path = Path("ci_data/california_housing.csv")
        output_path = Path("data/california_housing.csv")

        if not backup_path.exists():
            raise

        print(f"Dataset preparation failed: {error}")
        print("Restoring verified backup dataset...")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, output_path)

        print(f"Dataset restored successfully: {output_path}")