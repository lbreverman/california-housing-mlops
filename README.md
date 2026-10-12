# California Housing Price Prediction — MLOps Pipeline
[![Python Tests](https://github.com/lbreverman/california-housing-mlops/actions/workflows/python-tests.yml/badge.svg?branch=clean-main)](https://github.com/lbreverman/california-housing-mlops/actions/workflows/python-tests.yml)
## Project Overview

This project demonstrates an end-to-end machine learning workflow for predicting California housing prices using Python, scikit-learn, and Data Version Control (DVC).

The project includes data preprocessing, model training, model evaluation, experiment comparison, and data drift monitoring.

The goal is to build a reproducible machine learning pipeline that supports reliable experimentation and model monitoring.

## Technologies Used

- Python
- pandas and NumPy
- scikit-learn
- DVC (Data Version Control)
- Git and GitHub
- Joblib
- SciPy

## Dataset

The project uses the California Housing dataset available through scikit-learn.

The dataset contains 20,640 observations with features such as median income, housing age, average rooms, population, and geographic location.

The target variable is median house value.

## Machine Learning Models

Three regression models were evaluated:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Random Forest** | **0.3313** | **0.5105** | **0.8012** |
| Decision Tree | 0.4558 | 0.7152 | 0.6097 |
| Linear Regression | 0.5344 | 0.7413 | 0.5806 |

Random Forest achieved the strongest performance, explaining approximately 80.12% of the variance in housing prices on the evaluation dataset.

## Reproducible DVC Pipeline

The project uses DVC to manage five pipeline stages:

1. **Data Preparation:** Load, clean, and prepare the California Housing dataset.
2. **Model Training:** Train a Random Forest regression model.
3. **Model Evaluation:** Calculate MAE, RMSE, and R².
4. **Model Comparison:** Compare Random Forest, Decision Tree, and Linear Regression.
5. **Data Drift Monitoring:** Detect statistical changes in numerical feature distributions.

The pipeline can be reproduced using:

```bash
dvc repro
```

Its dependency graph can be viewed using:

```bash
dvc dag
```

## Data Drift Monitoring

The project includes a statistical drift monitoring script using the Kolmogorov-Smirnov test.

In a simulated drift scenario:

- Total numerical features monitored: 8
- Features with detected drift: 1
- Drift percentage: 12.5%
- Feature with detected drift: `MedInc`

The monitoring results are saved to `reports/drift_report.json`.

These results demonstrate how changes in input feature distributions can be identified as part of an MLOps workflow.

## Project Structure

```text
california-housing-mlops/
├── configs/
├── data/
├── models/
├── reports/
├── src/
│   ├── prepare_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_experiments.py
│   └── monitor_drift.py
├── tests/
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/lbreverman/california-housing-mlops.git
cd california-housing-mlops
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Reproduce the machine learning pipeline:

```bash
dvc repro
```

## Key Findings

- Random Forest outperformed Decision Tree and Linear Regression.
- The best model achieved an R² score of 0.8012.
- DVC successfully tracked the machine learning pipeline and its dependencies.
- Statistical drift monitoring detected a distribution change in median income.
- The project demonstrates reproducibility, model comparison, and monitoring within a structured machine learning workflow.

## Future Improvements

- Add automated unit tests and continuous integration.
- Expand experiment tracking and hyperparameter tuning.
- Implement a prediction API.
- Introduce scheduled drift monitoring and model retraining.

## Author

Leslie Breverman

Data Analytics and Machine Learning Portfolio

[GitHub Profile](https://github.com/lbreverman)