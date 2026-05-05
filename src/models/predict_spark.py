from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


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
        "path": path,
    }


def predict_price_per_sqm_spark(bundle: dict[str, Any], data: DataFrame) -> DataFrame:
    predictions = bundle["model"].transform(data)
    return predictions.withColumn("predicted_price_per_sqm", F.expm1(F.col("prediction")))


def read_prediction_input(spark: SparkSession, path: str | Path) -> DataFrame:
    return spark.read.csv(str(path), header=True, inferSchema=True)
