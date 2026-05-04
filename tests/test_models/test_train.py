from __future__ import annotations

import pandas as pd

from src.models.train import FEATURES, prepare_model_data, split_and_scale, train_models


def test_prepare_model_data_filters_buy_and_logs_target():
    rows = []
    for idx in range(8):
        row = {feature: float(idx + 1) for feature in FEATURES}
        row["category"] = "buy" if idx < 6 else "rent"
        row["price_per_sqm"] = float(10000 + idx)
        rows.append(row)

    X, y = prepare_model_data(pd.DataFrame(rows))

    assert len(X) == 6
    assert list(X.columns) == FEATURES
    assert len(y) == 6


def test_split_and_scale_is_reproducible():
    X = pd.DataFrame({feature: range(20) for feature in FEATURES})
    y = pd.Series(range(20), dtype=float)

    first = split_and_scale(X, y)
    second = split_and_scale(X, y)

    assert first["X_train"].index.tolist() == second["X_train"].index.tolist()
    assert first["X_test"].index.tolist() == second["X_test"].index.tolist()


def test_train_models_includes_tree_ensembles_with_randomized_search():
    X = pd.DataFrame({feature: range(30) for feature in FEATURES})
    y = pd.Series(range(30), dtype=float)
    splits = split_and_scale(X, y)

    results_df, _ = train_models(
        splits,
        {
            "random_forest": {
                "estimator": {"random_state": 42, "n_jobs": 1},
                "randomized_search": {
                    "n_iter": 1,
                    "cv": 2,
                    "random_state": 42,
                    "n_jobs": 1,
                    "param_distributions": {
                        "n_estimators": [5],
                        "max_depth": [2],
                        "min_samples_split": [2],
                        "min_samples_leaf": [1],
                        "max_features": ["sqrt"],
                        "bootstrap": [True],
                    },
                },
            },
            "xgboost": {
                "estimator": {
                    "objective": "reg:squarederror",
                    "random_state": 42,
                    "n_jobs": 1,
                    "verbosity": 0,
                },
                "randomized_search": {
                    "n_iter": 1,
                    "cv": 2,
                    "random_state": 42,
                    "n_jobs": 1,
                    "param_distributions": {
                        "n_estimators": [5],
                        "max_depth": [2],
                        "learning_rate": [0.1],
                        "subsample": [1.0],
                        "colsample_bytree": [1.0],
                        "min_child_weight": [1],
                        "reg_lambda": [1.0],
                    },
                },
            }
        },
    )

    assert "Random Forest" in results_df["Model"].tolist()
    assert "XGBoost" in results_df["Model"].tolist()
