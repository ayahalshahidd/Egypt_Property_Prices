from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import RANDOM_STATE, TARGET, TEST_SIZE


FEATURES = [
    "area_value", "log_area", "bedrooms", "bathrooms",
    "bed_bath_ratio", "total_rooms", "area_per_room", "area_x_beds",
    "lat", "lon",
    "city_enc", "town_enc", "district_enc", "property_type_enc",
    "completion_score", "is_furnished", "has_installments",
    "amenity_count", "has_pool", "has_gym", "has_security",
    "has_parking", "has_garden", "has_balcony",
    "has_private_pool", "has_spa", "has_ac",
    "days_listed",
    "is_premium", "is_verified", "is_new_construction",
    "is_direct_from_dev", "images_count",
]


def prepare_model_data(
    df: Any,
    features: list[str] | None = None,
    target: str = TARGET,
    category: str = "buy",
) -> tuple[pd.DataFrame, pd.Series]:
    features = features or FEATURES
    model_cols = features + [target]

    if hasattr(df, "filter") and hasattr(df, "toPandas"):
        from pyspark.sql import functions as F

        model_df = (
            df.filter(F.col("category") == category)
            .select(model_cols)
            .dropna()
            .toPandas()
        )
    else:
        model_df = df.loc[df["category"] == category, model_cols].dropna().copy()

    boolean_like = {"true": 1, "false": 0, "yes": 1, "no": 0}
    for column_name in model_cols:
        if model_df[column_name].dtype == object:
            normalized = model_df[column_name].astype(str).str.strip().str.lower()
            model_df[column_name] = normalized.map(boolean_like).fillna(normalized)
        model_df[column_name] = pd.to_numeric(model_df[column_name], errors="coerce")

    model_df = model_df.dropna()
    if model_df.empty:
        raise ValueError(
            "No rows remain after preparing model data. Check feature nulls and numeric conversions "
            f"for columns: {model_cols}"
        )
    return model_df[features], np.log1p(model_df[target])


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_sc": X_train_sc,
        "X_test_sc": X_test_sc,
        "scaler": scaler,
    }


def evaluate_model(name: str, model: Any, X_tr: Any, X_te: Any, y_tr: Any, y_te: Any) -> dict[str, Any]:
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return _build_evaluation_result(name, model, y_te, preds)


def evaluate_random_search(
    name: str,
    estimator: Any,
    search_params: dict[str, Any],
    X_tr: Any,
    X_te: Any,
    y_tr: Any,
    y_te: Any,
) -> dict[str, Any]:
    param_distributions = search_params.get("param_distributions", {})
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=search_params.get("n_iter", 20),
        scoring=search_params.get("scoring", "r2"),
        cv=search_params.get("cv", 3),
        random_state=search_params.get("random_state", RANDOM_STATE),
        n_jobs=search_params.get("n_jobs", -1),
        verbose=search_params.get("verbose", 0),
        refit=True,
    )
    search.fit(X_tr, y_tr)
    preds = search.best_estimator_.predict(X_te)
    result = _build_evaluation_result(name, search.best_estimator_, y_te, preds)
    result["Best_Params"] = search.best_params_
    result["CV_Best_R2"] = float(search.best_score_)
    return result


def _build_evaluation_result(name: str, model: Any, y_te: Any, preds: Any) -> dict[str, Any]:
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    mae = float(mean_absolute_error(y_te, preds))
    r2 = float(r2_score(y_te, preds))
    rmse_orig = float(np.sqrt(mean_squared_error(np.expm1(y_te), np.expm1(preds))))
    return {
        "Model": name,
        "R2": r2,
        "RMSE_log": rmse,
        "MAE_log": mae,
        "RMSE_EGP": rmse_orig,
        "fitted": model,
        "predictions_log": preds,
    }


def train_models(splits: dict[str, Any], model_params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = model_params or {}
    results: list[dict[str, Any]] = []

    ridge = Ridge(**params.get("ridge", {"alpha": 10}))
    results.append(evaluate_model(
        "Ridge Regression", ridge,
        splits["X_train_sc"], splits["X_test_sc"], splits["y_train"], splits["y_test"],
    ))

    lasso = Lasso(**params.get("lasso", {"alpha": 0.001, "max_iter": 5000, "random_state": 42}))
    results.append(evaluate_model(
        "Lasso Regression", lasso,
        splits["X_train_sc"], splits["X_test_sc"], splits["y_train"], splits["y_test"],
    ))

    elastic_net = ElasticNet(**params.get(
        "elastic_net",
        {"alpha": 0.001, "l1_ratio": 0.3, "max_iter": 5000, "random_state": 42},
    ))
    results.append(evaluate_model(
        "Elastic Net", elastic_net,
        splits["X_train_sc"], splits["X_test_sc"], splits["y_train"], splits["y_test"],
    ))

    decision_tree = DecisionTreeRegressor(**params.get(
        "decision_tree",
        {"max_depth": 6, "min_samples_leaf": 50, "random_state": 42},
    ))
    results.append(evaluate_model(
        "Shallow Decision Tree", decision_tree,
        splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"],
    ))

    random_forest_params = params.get("random_forest", {})
    random_forest = RandomForestRegressor(**random_forest_params.get(
        "estimator",
        {"random_state": RANDOM_STATE, "n_jobs": -1},
    ))
    results.append(evaluate_random_search(
        "Random Forest",
        random_forest,
        random_forest_params.get("randomized_search", _DEFAULT_RANDOM_FOREST_SEARCH),
        splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"],
    ))

    xgboost_params = params.get("xgboost", {})
    xgboost = XGBRegressor(**xgboost_params.get(
        "estimator",
        {
            "objective": "reg:squarederror",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        },
    ))
    results.append(evaluate_random_search(
        "XGBoost",
        xgboost,
        xgboost_params.get("randomized_search", _DEFAULT_XGBOOST_SEARCH),
        splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"],
    ))

    results_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k not in {"fitted", "predictions_log"}} for result in results]
    ).sort_values("R2", ascending=False).reset_index(drop=True)

    best_name = results_df.iloc[0]["Model"]
    best_result = next(result for result in results if result["Model"] == best_name)
    best_bundle = {
        "model": best_result["fitted"],
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "scaler": splits["scaler"] if best_name in _SCALED_MODEL_NAMES else None,
        "metrics": {k: best_result[k] for k in ["R2", "RMSE_log", "MAE_log", "RMSE_EGP"]},
    }
    return results_df, best_bundle


def save_model_bundle(bundle: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = bundle["model_name"].lower().replace(" ", "_")
    output_path = output_dir / f"{model_name}_buy_price_per_sqm_{timestamp}.pkl"
    joblib.dump(bundle, output_path)
    return output_path


def write_model_reports(results_df: pd.DataFrame, bundle: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "holdout_metrics.json").write_text(
        json.dumps({"best_model": bundle["model_name"], **bundle["metrics"]}, indent=2),
        encoding="utf-8",
    )
    write_experiment_log(results_df, bundle, output_dir / "model_experimentation.md")


def write_experiment_log(results_df: pd.DataFrame, bundle: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Model Experimentation Log",
        "",
        "## Goal",
        "",
        "Predict `price_per_sqm` for buy listings using fast, interpretable baseline models.",
        "The target is modeled with `log1p(price_per_sqm)` to reduce the effect of very expensive listings.",
        "",
        "## Experiments",
        "",
        "| Model | Purpose | Interpretability |",
        "| --- | --- | --- |",
        "| Ridge Regression | Linear baseline with L2 regularization. | Coefficients show feature direction and relative strength after scaling. |",
        "| Lasso Regression | Sparse linear model for feature selection. | Non-zero coefficients identify retained predictors. |",
        "| Elastic Net | Hybrid L1/L2 regularized linear model. | Balances sparse selection with correlated-feature stability. |",
        "| Shallow Decision Tree | Captures simple non-linear rules. | Tree depth is capped, so split rules can be inspected. |",
        "| Random Forest | Tuned ensemble of decision trees using randomized grid search. | Feature importances summarize the strongest predictive signals. |",
        "| XGBoost | Tuned gradient-boosted tree ensemble using randomized grid search. | Feature importances help rank non-linear predictors. |",
        "",
        "## Holdout Results",
        "",
        "| Model | R2 | RMSE log | MAE log | RMSE EGP/sqm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in results_df.itertuples(index=False):
        lines.append(
            f"| {row.Model} | {row.R2:.4f} | {row.RMSE_log:.4f} | "
            f"{row.MAE_log:.4f} | {row.RMSE_EGP:,.0f} |"
        )

    lines.extend([
        "",
        "## Selected Model",
        "",
        f"`{bundle['model_name']}` was selected by highest holdout R2.",
        "",
        "## Deferred Experiments",
        "",
        "Heavier models such as extra trees, LightGBM, CatBoost, and stacking were intentionally deferred.",
        "They are useful next-step experiments after the baseline pipeline, reporting, and interpretation are stable.",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model_params(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_SCALED_MODEL_NAMES = {"Ridge Regression", "Lasso Regression", "Elastic Net"}

_DEFAULT_RANDOM_FOREST_SEARCH = {
    "n_iter": 20,
    "cv": 3,
    "scoring": "r2",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "param_distributions": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 8, 12, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", 0.5, 0.8],
        "bootstrap": [True],
    },
}

_DEFAULT_XGBOOST_SEARCH = {
    "n_iter": 20,
    "cv": 3,
    "scoring": "r2",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "param_distributions": {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
}
