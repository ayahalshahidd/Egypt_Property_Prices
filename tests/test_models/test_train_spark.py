from __future__ import annotations

import json

from src.config import CONFIGS_DIR
from src.models.train_spark import load_spark_model_params


def test_spark_model_params_define_distributed_models():
    params = load_spark_model_params(CONFIGS_DIR / "spark_model_params.json")

    assert params["tuning"]["method"] in {"train_validation_split", "cross_validator"}
    assert {"linear_regression", "decision_tree", "random_forest", "gbt"}.issubset(params)
    assert params["gbt"]["enabled"] is False


def test_spark_model_params_json_is_valid():
    with (CONFIGS_DIR / "spark_model_params.json").open(encoding="utf-8") as config_file:
        params = json.load(config_file)

    assert params["random_forest"]["param_grid"]["numTrees"]
