# Model Experimentation Log

## Goal

Predict `price_per_sqm` for buy listings using fast, interpretable baseline models.
The target is modeled with `log1p(price_per_sqm)` to reduce the effect of very expensive listings.

## Experiments

| Model | Purpose | Interpretability |
| --- | --- | --- |
| Ridge Regression | Linear baseline with L2 regularization. | Coefficients show feature direction and relative strength after scaling. |
| Lasso Regression | Sparse linear model for feature selection. | Non-zero coefficients identify retained predictors. |
| Elastic Net | Hybrid L1/L2 regularized linear model. | Balances sparse selection with correlated-feature stability. |
| Shallow Decision Tree | Captures simple non-linear rules. | Tree depth is capped, so split rules can be inspected. |
| Random Forest | Tuned ensemble of decision trees using randomized grid search. | Feature importances summarize the strongest predictive signals. |
| XGBoost | Tuned gradient-boosted tree ensemble using randomized grid search. | Feature importances help rank non-linear predictors. |

## Holdout Results

| Model | R2 | RMSE log | MAE log | RMSE EGP/sqm |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.7349 | 0.3329 | 0.2254 | 75,711 |
| Random Forest | 0.7123 | 0.3468 | 0.2334 | 78,336 |
| Shallow Decision Tree | 0.4787 | 0.4669 | 0.3489 | 83,927 |
| Elastic Net | 0.4407 | 0.4836 | 0.3685 | 74,894 |
| Lasso Regression | 0.4406 | 0.4836 | 0.3684 | 74,844 |
| Ridge Regression | 0.4406 | 0.4837 | 0.3686 | 74,917 |

## Selected Model

`XGBoost` was selected by highest holdout R2.

## Deferred Experiments

Heavier models such as extra trees, LightGBM, CatBoost, and stacking were intentionally deferred.
They are useful next-step experiments after the baseline pipeline, reporting, and interpretation are stable.
