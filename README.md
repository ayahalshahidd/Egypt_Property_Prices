# Egypt Property Prices

This project builds a data-driven predictive pricing pipeline for Egyptian real estate listings. It keeps notebooks for exploration while moving reusable cleaning, feature engineering, training, prediction, and reporting code into a simple pip-based project structure.

## Project Structure

```text
data/
  raw/          Original datasets
  interim/      Intermediate cleaned outputs
  processed/    Final feature-engineered datasets
  external/     External datasets
notebooks/      Jupyter notebooks for exploration
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

This loads `data/raw/propertyfinder.csv`, writes cleaned data to `data/interim/`, writes feature-engineered data to `data/processed/`, saves the best model to `models/`, and generates reports under `reports/`.

## Current Modeling Scope

The training pipeline starts with fast baseline models and now includes a tuned Random Forest ensemble:

- Ridge Regression
- Lasso Regression
- Elastic Net
- Shallow Decision Tree
- Random Forest with randomized hyperparameter search
- XGBoost with randomized hyperparameter search

Longer-running ensembles and boosting models, such as extra trees, LightGBM, CatBoost, and stacking, are deferred to later experimentation. Each run writes `reports/results/model_experimentation.md`, `reports/results/model_comparison.csv`, and `reports/results/holdout_metrics.json` so experiments can be documented before moving to heavier models.

## Generate Predictions

```bash
python scripts/generate_predictions.py --input data/processed/prediction_input.csv
```

By default, the script uses the latest `.pkl` model in `models/`.

## Tests

```bash
pytest tests
```
