from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.features.build_features import add_listing_features, apply_target_encoding_model, deserialize_target_encoding_model


def load_spark_model_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("model_persisted", True):
        raise RuntimeError(
            "This Spark model artifact contains metrics metadata but no reloadable Spark ML model. "
            "Configure HADOOP_HOME with winutils.exe on Windows, then rerun training."
        )
    return {
        **metadata,
        "model": PipelineModel.load(str(path / "model")),
        "target_encoding_model": (
            deserialize_target_encoding_model(metadata["target_encoding"])
            if "target_encoding" in metadata
            else None
        ),
        "path": path,
    }


def predict_price_per_sqm_spark(bundle: dict[str, Any], data: DataFrame) -> DataFrame:
    features = bundle.get("features", [])
    if any(feature not in data.columns for feature in features):
        data = add_listing_features(data)
        encoding_model = bundle.get("target_encoding_model")
        if encoding_model is not None:
            data = apply_target_encoding_model(data, encoding_model)
    predictions = bundle["model"].transform(data)
    return predictions.withColumn("predicted_price_per_sqm", F.expm1(F.col("prediction")))


def read_prediction_input(spark: SparkSession, path: str | Path) -> DataFrame:
    return spark.read.csv(str(path), header=True, inferSchema=True)
