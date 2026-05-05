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
| Spark Random Forest | 0.4784 | 0.4377 | 0.3291 | 35,593 |
| Spark Decision Tree | 0.4567 | 0.4467 | 0.3322 | 35,133 |
| Spark Linear Regression | 0.4020 | 0.4686 | 0.3512 | 38,293 |

## Selected Model

`Spark Random Forest` was selected by highest holdout R2.

## Train, Validation, and Test Metrics

| Model | Train R2 | Validation R2 | Test R2 | Test RMSE EGP/sqm |
| --- | ---: | ---: | ---: | ---: |
| Spark Random Forest | 0.5081 | 0.4964 | 0.4784 | 35,593 |
| Spark Decision Tree | 0.4849 | 0.4766 | 0.4567 | 35,133 |
| Spark Linear Regression | 0.4168 | 0.4174 | 0.4020 | 38,293 |

## Training Scope

All production model training in this project uses Spark ML.
