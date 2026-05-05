from __future__ import annotations

import pytest

from src.features.build_features import (
    AMENITY_FEATURES,
    apply_target_encoding_model,
    fit_target_encoding_model,
)


def test_amenity_features_have_unique_output_names():
    output_names = [column_name for _keyword, column_name in AMENITY_FEATURES]

    assert len(output_names) == len(set(output_names))
    assert "has_pool" in output_names
    assert "has_ac" in output_names


def test_target_encoding_model_uses_training_categories_only(spark):
    train_df = spark.sql(
        """
        SELECT 'Cairo' AS city, 10000.0 AS price_per_sqm
        UNION ALL SELECT 'Cairo', 20000.0
        UNION ALL SELECT 'Giza', 40000.0
        """
    )
    validation_df = spark.sql(
        "SELECT 'Unseen City' AS city, 999999.0 AS price_per_sqm"
    )

    model = fit_target_encoding_model(train_df, categorical_cols=("city",), smoothing=0.0)
    encoded = apply_target_encoding_model(validation_df, model)

    row = encoded.first()
    assert row["city_enc"] == pytest.approx(70_000.0 / 3.0)
    assert "Unseen City" not in model.mappings["city"]
