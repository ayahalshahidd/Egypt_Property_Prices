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
| Spark Random Forest | 0.5394 | 0.4411 | 0.3264 | 59,856 |
| Spark Decision Tree | 0.4589 | 0.4781 | 0.3555 | 61,225 |
| Spark Linear Regression | 0.4534 | 0.4805 | 0.3627 | 64,707 |

## Selected Model

`Spark Random Forest` was selected by highest holdout R2.

## Train, Validation, and Test Metrics

| Model | Train R2 | Validation R2 | Test R2 | Test RMSE EGP/sqm |
| --- | ---: | ---: | ---: | ---: |
| Spark Random Forest | 0.5565 | 0.5278 | 0.5394 | 59,856 |
| Spark Decision Tree | 0.4775 | 0.4535 | 0.4589 | 61,225 |
| Spark Linear Regression | 0.4254 | 0.4202 | 0.4534 | 64,707 |

## Training Scope

All production model training in this project uses Spark ML.
