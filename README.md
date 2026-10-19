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
- MLflow (Experiment Tracking)
- Evidently AI (Data Drift Monitoring)

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


## MLflow Experiment Tracking

The project uses MLflow to track and compare Random Forest regression experiments.

Five Random Forest configurations were evaluated using different hyperparameter settings.

MLflow records model parameters, evaluation metrics, and trained model artifacts to support reproducible experimentation.

### Best Experiment Results

| Metric | Result |
|---|---|
| Model | Random Forest |
| Number of Trees | 200 |
| Maximum Depth | None |
| MAE | 0.3296 |
| RMSE | 0.5085 |
| R² | 0.8026 |

The best experiment achieved an R² of approximately 0.803 on the evaluation dataset.

The original model comparison results are retained in the Machine Learning Models section above.

To run the experiment tracking pipeline:

```powershell
dvc repro run_experiments
```

To open the MLflow tracking interface:

```powershell
mlflow ui --port 5001
```

Then open http://127.0.0.1:5001 in your browser.

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

Clone the repository using the `clean-main` branch:

```bash
git clone --branch clean-main https://github.com/lbreverman/california-housing-mlops.git
cd california-housing-mlops
```

Create a virtual environment using Python 3.11:

```powershell
py -3.11 -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

Reproduce the complete machine learning pipeline:

```powershell
dvc repro
```

Verify that all pipeline stages are up to date:

```powershell
dvc status
```

Run the automated tests:

```powershell
python -m pytest
```

The pipeline generates the prepared dataset, trained model, evaluation metrics, model comparison, and drift monitoring report.

## Key Findings

- Random Forest outperformed Decision Tree and Linear Regression.
- The best model achieved an R² score of 0.8012.
- DVC successfully tracked the machine learning pipeline and its dependencies.
- Statistical drift monitoring detected a distribution change in median income.
- The project demonstrates reproducibility, model comparison, and monitoring within a structured machine learning workflow.

## Testing and Continuous Integration

The project includes automated testing with pytest and a GitHub Actions workflow.

GitHub Actions runs the Python tests to help identify issues when changes are pushed to the repository.

The project was also successfully reproduced in a fresh GitHub clone using Python 3.11 and DVC, confirming that the documented setup can rebuild the machine learning pipeline.

## Data Drift Monitoring

This project includes automated data drift monitoring using
statistical hypothesis testing and Evidently AI.
### Monitoring Approach

The monitoring pipeline compares two datasets:

- **Reference data:** 70% of the prepared California Housing dataset.
- **Current data:** The remaining 30%, representing simulated incoming observations.

To demonstrate drift detection, median income (`MedInc`) in the
current dataset is increased by 20%.

The monitoring system uses:

- **Kolmogorov-Smirnov (KS) tests:** Detect changes in numerical
  feature distributions using a significance level of 0.05.
- **Evidently AI:** Generates an HTML report showing feature
  distributions and drift detection results.

### Drift Detection Results

The KS-based monitoring detected drift in 1 of 8 numerical
features (12.50%).

Evidently detected drift in 1 of 9 features (11.11%),
including the categorical `IncomeCategory` feature.

The feature with detected drift was:

- **MedInc (Median Income)**

The remaining features did not show detected drift under
their respective monitoring tests.

### Why Median Income Drift Matters

Median income is an important predictor of California housing
prices.

A change in its distribution may indicate that incoming
housing data differs from the data used to train the model.

If the model continues making predictions on substantially
different data, its prediction accuracy could deteriorate.

However, detected feature drift does not automatically mean
the model's performance has declined.

### Recommended Production Response

If drift is detected in production:

1. Verify that the data change is real and not caused by
   collection errors or preprocessing problems.
2. Investigate which features have changed and why.
3. Monitor prediction quality using recent labeled data
   when available.
4. Evaluate whether the existing model still meets
   performance requirements.
5. Retrain and validate the model on representative,
   updated data if performance has deteriorated.
6. Deploy an updated model only after successful validation.

## Future Improvements

- Expand experiment tracking and hyperparameter tuning.
- Implement a prediction API using FastAPI.
- Introduce scheduled data drift monitoring and automated model retraining.
- Add model performance visualizations and an interactive dashboard.

## Author

Leslie Breverman

Data Analytics and Machine Learning Portfolio

[GitHub Profile](https://github.com/lbreverman)


### Generated Monitoring Reports

The pipeline produces two reports:

- `reports/drift_report.json` — KS-test results and drift summary.
- `reports/drift_report.html` — Evidently visual data drift report.

Both reports are generated automatically by the DVC
`monitor_drift` stage.

Run the monitoring pipeline with:

```bash
dvc repro monitor_drift
```