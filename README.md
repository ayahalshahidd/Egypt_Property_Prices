# Egypt Property Prices

This project builds a Spark-based predictive pricing pipeline for Egyptian real estate listings. Reusable cleaning, feature engineering, training, prediction, and reporting code lives in a simple pip-based project structure.

## Project Structure

```text
data/
  raw/          Original datasets
  interim/      Intermediate cleaned outputs
  processed/    Final feature-engineered datasets
  external/     External datasets
src/            Reusable pipeline source code
tests/          Unit and integration tests
models/         Trained model artifacts
reports/        Generated figures and result reports
configs/        JSON model parameters
scripts/        Standalone pipeline scripts
```

## Setup

```bash
pip install -r requirements.txt
```

## Run The Pipeline

```bash
python scripts/train_model.py
```

This runs the Spark ML pipeline. It loads `data/raw/propertyfinder.csv`, writes cleaned data to `data/interim/`, writes feature-engineered data to `data/processed/`, trains Spark ML models, saves the best Spark model to `models/`, and generates reports under `reports/`.

## Current Modeling Scope

The training pipeline is implemented end-to-end with PySpark and Spark ML:

- Spark Linear Regression
- Spark Decision Tree
- Spark Random Forest
- Spark Gradient-Boosted Trees

Each run writes `reports/results/spark_model_experimentation.md`, `reports/results/spark_model_comparison.csv`, and `reports/results/spark_holdout_metrics.json`. Spark figures are written under `reports/figures/`, including `spark_model_comparison.png`, `spark_feature_importance.png`, and `spark_residuals.png`.

## Generate Predictions

```bash
python scripts/generate_predictions.py --input data/processed/prediction_input.csv
```

The script uses the latest Spark model directory in `models/` and writes Spark predictions to `reports/results/spark_predictions_csv/`.

## Tests

```bash
pytest tests
```
