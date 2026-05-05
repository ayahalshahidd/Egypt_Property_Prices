from __future__ import annotations

import json

from src.config import CONFIGS_DIR
from src.models.train_spark import load_spark_model_params, prepare_spark_model_data


def test_spark_model_params_define_distributed_models():
    params = load_spark_model_params(CONFIGS_DIR / "spark_model_params.json")

    assert params["tuning"]["method"] in {"train_validation_split", "cross_validator"}
    assert {"linear_regression", "decision_tree", "random_forest", "gbt"}.issubset(params)
    assert params["gbt"]["enabled"] is False


def test_spark_model_params_json_is_valid():
    with (CONFIGS_DIR / "spark_model_params.json").open(encoding="utf-8") as config_file:
        params = json.load(config_file)

    assert params["random_forest"]["param_grid"]["numTrees"]


def test_prepare_spark_model_data_creates_log_label(spark):
    df = spark.sql(
        """
        SELECT
            'buy' AS category,
            100.0 AS area_value,
            4.615 AS log_area,
            2.0 AS bedrooms,
            1.0 AS bathrooms,
            2.0 AS bed_bath_ratio,
            3.0 AS total_rooms,
            33.3 AS area_per_room,
            200.0 AS area_x_beds,
            30.0 AS lat,
            31.0 AS lon,
            10000.0 AS city_enc,
            10000.0 AS town_enc,
            10000.0 AS district_enc,
            10000.0 AS property_type_enc,
            2.0 AS completion_score,
            0.0 AS is_furnished,
            0.0 AS has_installments,
            1.0 AS amenity_count,
            0.0 AS has_pool,
            0.0 AS has_gym,
            1.0 AS has_security,
            1.0 AS has_parking,
            0.0 AS has_garden,
            1.0 AS has_balcony,
            0.0 AS has_private_pool,
            0.0 AS has_spa,
            1.0 AS has_ac,
            2.0 AS days_listed,
            0.0 AS is_premium,
            1.0 AS is_verified,
            0.0 AS is_new_construction,
            0.0 AS is_direct_from_dev,
            5.0 AS images_count,
            10000.0 AS price_per_sqm
        """
    )

    prepared = prepare_spark_model_data(df)

    assert prepared.count() == 1
    assert prepared.first()["label"] > 0
