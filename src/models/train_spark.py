from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def prepare_spark_model_data(
    df: DataFrame,
    features: list[str] | None = None,
    target: str = TARGET,
    category: str = "buy",
) -> DataFrame:
    features = features or FEATURES
    model_cols = features + [target]

    model_df = df.filter(F.col("category") == category).select(
        *[F.col(column_name).cast("double").alias(column_name) for column_name in model_cols]
    )
    model_df = model_df.dropna(subset=model_cols)
    model_df = model_df.withColumn("label", F.log1p(F.col(target)))

    if model_df.limit(1).count() == 0:
        raise ValueError(
            "No rows remain after preparing Spark model data. Check feature nulls and numeric conversions "
            f"for columns: {model_cols}"
        )
    return model_df.select(*features, target, "label")


def split_spark_data(
    model_df: DataFrame,
    test_size: float = TEST_SIZE,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, DataFrame]:
    train_size = 1.0 - test_size - validation_size
    if train_size <= 0:
        raise ValueError("Train split must be positive. Reduce test_size or validation_size.")
    train_df, validation_df, test_df = model_df.randomSplit(
        [train_size, validation_size, test_size], seed=random_state
    )
    return {"train": train_df.cache(), "validation": validation_df.cache(), "test": test_df.cache()}


def train_spark_models(
    splits: dict[str, DataFrame],
    model_params: dict[str, Any] | None = None,
    features: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = model_params or {}
    features = features or FEATURES
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    results: list[dict[str, Any]] = []

    for model_name, estimator, grid in _spark_model_specs(params):
        tuned_model = _fit_tuned_model(model_name, estimator, grid, evaluator, splits["train"], params)
        fitted_model = tuned_model.bestModel if hasattr(tuned_model, "bestModel") else tuned_model
        train_predictions = fitted_model.transform(splits["train"])
        validation_predictions = fitted_model.transform(splits["validation"])
        test_predictions = fitted_model.transform(splits["test"])
        results.append(
            _evaluate_spark_predictions(
                model_name,
                fitted_model,
                tuned_model,
                train_predictions,
                validation_predictions,
                test_predictions,
            )
        )

    results = sorted(results, key=lambda item: item["R2"], reverse=True)
    best_result = results[0]
    best_bundle = {
        "model": best_result["fitted"],
        "model_name": best_result["Model"],
        "features": features,
        "target": TARGET,
        "metrics": {k: best_result[k] for k in ["R2", "RMSE_log", "MAE_log", "RMSE_EGP"]},
    }
    return results, best_bundle


def save_spark_model_bundle(bundle: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = bundle["model_name"].lower().replace(" ", "_")
    output_path = output_dir / f"{model_name}_spark_buy_price_per_sqm_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "backend": "spark",
        "model_name": bundle["model_name"],
        "features": bundle["features"],
        "target": bundle["target"],
        "metrics": bundle["metrics"],
        "model_persisted": True,
        "model_path": "model",
    }
    try:
        bundle["model"].write().overwrite().save(str(output_path / "model"))
    except Exception as exc:
        metadata["model_persisted"] = False
        metadata["model_path"] = None
        metadata["save_error"] = str(exc)
        metadata["save_note"] = (
            "Spark ML native model persistence failed. On Windows this usually means Hadoop winutils.exe "
            "is missing and HADOOP_HOME/hadoop.home.dir are unset. Reports and metrics were still written."
        )
        print("WARNING: Spark model metrics were produced, but native Spark model persistence failed.")
        print("WARNING: Configure HADOOP_HOME with winutils.exe to save reloadable Spark ML models on Windows.")
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def write_spark_model_reports(results: list[dict[str, Any]], bundle: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_results = [_serializable_result(result) for result in results]
    _write_results_csv(serializable_results, output_dir / "spark_model_comparison.csv")
    (output_dir / "spark_holdout_metrics.json").write_text(
        json.dumps({"best_model": bundle["model_name"], **bundle["metrics"]}, indent=2),
        encoding="utf-8",
    )
    write_spark_experiment_log(serializable_results, bundle, output_dir / "spark_model_experimentation.md")


def write_spark_experiment_log(results: list[dict[str, Any]], bundle: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spark Model Experimentation Log",
        "",
        "## Goal",
        "",
        "Train and evaluate the price-per-sqm model end-to-end with Spark ML.",
        "The target is modeled as `log1p(price_per_sqm)` to reduce skew.",
        "",
        "## Experiments",
        "",
        "| Model | Purpose | Distributed Training Path |",
        "| --- | --- | --- |",
        "| Spark Linear Regression | Distributed linear baseline. | Spark ML Pipeline with VectorAssembler. |",
        "| Spark Decision Tree | Distributed non-linear baseline. | Spark ML Pipeline with tree estimator. |",
        "| Spark Random Forest | Distributed bagged tree ensemble. | Spark ML tuning over RandomForestRegressor params. |",
        "| Spark Gradient-Boosted Trees | Distributed boosted tree ensemble. | Spark ML tuning over GBTRegressor params. |",
        "",
        "## Holdout Results",
        "",
        "| Model | R2 | RMSE log | MAE log | RMSE EGP/sqm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['Model']} | {row['R2']:.4f} | {row['RMSE_log']:.4f} | "
            f"{row['MAE_log']:.4f} | {row['RMSE_EGP']:,.0f} |"
        )
    lines.extend([
        "",
        "## Selected Model",
        "",
        f"`{bundle['model_name']}` was selected by highest holdout R2.",
        "",
        "## Train, Validation, and Test Metrics",
        "",
        "| Model | Train R2 | Validation R2 | Test R2 | Test RMSE EGP/sqm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in results:
        validation_r2 = row.get("Validation_R2")
        validation_text = f"{validation_r2:.4f}" if validation_r2 is not None else "N/A"
        lines.append(
            f"| {row['Model']} | {row['Train_R2']:.4f} | {validation_text} | "
            f"{row['R2']:.4f} | {row['RMSE_EGP']:,.0f} |"
        )
    lines.extend([
        "",
        "## Training Scope",
        "",
        "All production model training in this project uses Spark ML.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_spark_model_params(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_tuned_model(
    model_name: str,
    estimator: Any,
    param_grid: list[Any],
    evaluator: RegressionEvaluator,
    train_df: DataFrame,
    params: dict[str, Any],
) -> CrossValidatorModel | TrainValidationSplitModel | PipelineModel:
    pipeline = Pipeline(stages=[VectorAssembler(inputCols=FEATURES, outputCol="features"), estimator])
    if not param_grid:
        return pipeline.fit(train_df)

    tuning = params.get("tuning", {})
    method = tuning.get("method", "train_validation_split")
    if method == "cross_validator":
        tuner = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=tuning.get("num_folds", 3),
            seed=tuning.get("random_state", RANDOM_STATE),
            parallelism=tuning.get("parallelism", 2),
        )
    else:
        tuner = TrainValidationSplit(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            trainRatio=tuning.get("train_ratio", 0.8),
            seed=tuning.get("random_state", RANDOM_STATE),
            parallelism=tuning.get("parallelism", 2),
        )
    print(f"Training {model_name} with {len(param_grid)} Spark param set(s)")
    return tuner.fit(train_df)


def _evaluate_spark_predictions(
    model_name: str,
    fitted_model: PipelineModel,
    tuned_model: CrossValidatorModel | TrainValidationSplitModel | PipelineModel,
    train_predictions: DataFrame,
    validation_predictions: DataFrame,
    test_predictions: DataFrame,
) -> dict[str, Any]:
    train_metrics = _spark_regression_metrics(train_predictions)
    validation_metrics = _spark_regression_metrics(validation_predictions)
    test_metrics = _spark_regression_metrics(test_predictions)
    return {
        "Model": model_name,
        "Train_R2": train_metrics["R2"],
        "Train_RMSE_log": train_metrics["RMSE_log"],
        "Train_MAE_log": train_metrics["MAE_log"],
        "Train_RMSE_EGP": train_metrics["RMSE_EGP"],
        "Tuning_Validation_R2": _best_validation_score(tuned_model),
        "Validation_R2": validation_metrics["R2"],
        "Validation_RMSE_log": validation_metrics["RMSE_log"],
        "Validation_MAE_log": validation_metrics["MAE_log"],
        "Validation_RMSE_EGP": validation_metrics["RMSE_EGP"],
        "R2": test_metrics["R2"],
        "RMSE_log": test_metrics["RMSE_log"],
        "MAE_log": test_metrics["MAE_log"],
        "RMSE_EGP": test_metrics["RMSE_EGP"],
        "Best_Params": _best_params(tuned_model),
        "fitted": fitted_model,
    }


def _spark_regression_metrics(predictions: DataFrame) -> dict[str, float]:
    metrics = predictions.select(
        F.sqrt(F.avg(F.pow(F.col("label") - F.col("prediction"), 2))).alias("RMSE_log"),
        F.avg(F.abs(F.col("label") - F.col("prediction"))).alias("MAE_log"),
        F.sqrt(F.avg(F.pow(F.expm1(F.col("label")) - F.expm1(F.col("prediction")), 2))).alias("RMSE_EGP"),
    ).first()
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)
    return {
        "R2": float(r2),
        "RMSE_log": float(metrics["RMSE_log"]),
        "MAE_log": float(metrics["MAE_log"]),
        "RMSE_EGP": float(metrics["RMSE_EGP"]),
    }


def _spark_model_specs(params: dict[str, Any]) -> list[tuple[str, Any, list[Any]]]:
    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")

    specs = []

    linear_params = params.get("linear_regression", {})
    if linear_params.get("enabled", True):
        linear = LinearRegression(featuresCol="features", labelCol="label", **linear_params.get("estimator", {}))
        specs.append(("Spark Linear Regression", linear, _build_param_grid(assembler, linear, linear_params)))

    tree_params = params.get("decision_tree", {})
    if tree_params.get("enabled", True):
        tree = DecisionTreeRegressor(
            featuresCol="features",
            labelCol="label",
            seed=RANDOM_STATE,
            **tree_params.get("estimator", {}),
        )
        specs.append(("Spark Decision Tree", tree, _build_param_grid(assembler, tree, tree_params)))

    forest_params = params.get("random_forest", {})
    if forest_params.get("enabled", True):
        forest = RandomForestRegressor(
            featuresCol="features",
            labelCol="label",
            seed=RANDOM_STATE,
            **forest_params.get("estimator", {}),
        )
        specs.append(("Spark Random Forest", forest, _build_param_grid(assembler, forest, forest_params)))

    gbt_params = params.get("gbt", {})
    if gbt_params.get("enabled", True):
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="label",
            seed=RANDOM_STATE,
            **gbt_params.get("estimator", {}),
        )
        specs.append(("Spark Gradient-Boosted Trees", gbt, _build_param_grid(assembler, gbt, gbt_params)))

    return specs


def _build_param_grid(assembler: VectorAssembler, estimator: Any, model_params: dict[str, Any]) -> list[Any]:
    del assembler
    grid_config = model_params.get("param_grid", {})
    builder = ParamGridBuilder()
    for param_name, values in grid_config.items():
        builder = builder.addGrid(getattr(estimator, param_name), values)
    return builder.build()


def _best_params(tuned_model: CrossValidatorModel | TrainValidationSplitModel | PipelineModel) -> dict[str, Any]:
    if not hasattr(tuned_model, "bestModel"):
        return {}
    best_model = tuned_model.bestModel
    if not hasattr(tuned_model, "getEstimatorParamMaps"):
        return {}
    param_maps = tuned_model.getEstimatorParamMaps()
    metrics = getattr(tuned_model, "validationMetrics", None) or getattr(tuned_model, "avgMetrics", None)
    if not param_maps or not metrics:
        return {}
    best_index = max(range(len(metrics)), key=lambda idx: metrics[idx])
    best_params = {}
    for param, value in param_maps[best_index].items():
        best_params[param.name] = value
    # Keep the fitted pipeline referenced so callers save the actual best model.
    del best_model
    return best_params


def _best_validation_score(tuned_model: CrossValidatorModel | TrainValidationSplitModel | PipelineModel) -> float | None:
    metrics = getattr(tuned_model, "validationMetrics", None) or getattr(tuned_model, "avgMetrics", None)
    if not metrics:
        return None
    return float(max(metrics))


def _serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "fitted"}


def _write_results_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Model",
        "Train_R2",
        "Train_RMSE_log",
        "Train_MAE_log",
        "Train_RMSE_EGP",
        "Tuning_Validation_R2",
        "Validation_R2",
        "Validation_RMSE_log",
        "Validation_MAE_log",
        "Validation_RMSE_EGP",
        "R2",
        "RMSE_log",
        "MAE_log",
        "RMSE_EGP",
        "Best_Params",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
