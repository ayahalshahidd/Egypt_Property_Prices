# Spark Model Experimentation Log

## Goal

Train and evaluate the price-per-sqm model end-to-end with Spark ML.
The target is modeled as `log1p(price_per_sqm)` to reduce skew.

## Experiments

| Model | Purpose | Distributed Training Path |
| --- | --- | --- |
| Spark Linear Regression | Distributed linear baseline. | Spark ML Pipeline with VectorAssembler. |
| Spark Decision Tree | Distributed non-linear baseline. | Spark ML Pipeline with tree estimator. |
| Spark Random Forest | Distributed bagged tree ensemble. | Spark ML tuning over RandomForestRegressor params. |
| Spark Gradient-Boosted Trees | Distributed boosted tree ensemble. | Spark ML tuning over GBTRegressor params. |

## Holdout Results

| Model | R2 | RMSE log | MAE log | RMSE EGP/sqm |
| --- | ---: | ---: | ---: | ---: |
| Spark Random Forest | 0.5432 | 0.4393 | 0.3246 | 59,584 |
| Spark Decision Tree | 0.4707 | 0.4729 | 0.3506 | 60,799 |
| Spark Linear Regression | 0.4540 | 0.4803 | 0.3621 | 64,537 |

## Selected Model

`Spark Random Forest` was selected by highest holdout R2.

## Training Scope

All production model training in this project uses Spark ML.
