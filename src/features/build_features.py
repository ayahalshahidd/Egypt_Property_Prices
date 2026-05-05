from __future__ import annotations

import csv
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


DEFAULT_REFERENCE_DATE = "2026-03-05"
DEFAULT_TARGET_COL = "price_per_sqm"
DEFAULT_CATEGORICAL_TARGET_COLUMNS = ("city", "town", "district", "property_type")

AMENITY_FEATURES = (
    ("Pool", "has_pool"),
    ("Gym", "has_gym"),
    ("Security", "has_security"),
    ("Parking", "has_parking"),
    ("Garden", "has_garden"),
    ("Balcony", "has_balcony"),
    ("Private Pool", "has_private_pool"),
    ("Spa", "has_spa"),
    ("A/C", "has_ac"),
)

BOOLEAN_FLAG_COLUMNS = (
    "is_premium",
    "is_verified",
    "is_new_construction",
    "is_direct_from_dev",
)


@dataclass
class TargetEncodingModel:
    global_mean: float
    categorical_cols: tuple[str, ...]
    mappings: dict[str, dict[Any, float]]


def _safe_denominator(column_name: str) -> F.Column:
    return F.when(F.col(column_name) == 0, F.lit(1.0)).otherwise(F.col(column_name))


def _boolean_to_int(column_name: str) -> F.Column:
    normalized = F.lower(F.trim(F.coalesce(F.col(column_name).cast("string"), F.lit(""))))
    return (
        F.when(normalized.isin("true", "1", "yes"), F.lit(1))
        .when(normalized.isin("false", "0", "no"), F.lit(0))
        .otherwise(F.lit(0))
    )


def add_listing_features(df: DataFrame, reference_date: str = DEFAULT_REFERENCE_DATE) -> DataFrame:
    amenities = F.coalesce(F.col("amenities"), F.lit(""))

    featured_df = df.withColumn(
        "amenity_count",
        F.when(
            F.col("amenities").isNotNull() & (F.trim(F.col("amenities")) != ""),
            F.size(F.split(F.col("amenities"), r"\|")),
        ).otherwise(F.lit(0)),
    )

    for keyword, col_name in AMENITY_FEATURES:
        featured_df = featured_df.withColumn(
            col_name,
            (F.instr(F.lower(amenities), keyword.lower()) > 0).cast("int"),
        )

    total_rooms = F.col("bedrooms") + F.col("bathrooms")
    safe_total_rooms = F.when(total_rooms == 0, F.lit(1.0)).otherwise(total_rooms)
    payment_method = F.lower(F.trim(F.coalesce(F.col("payment_method"), F.lit(""))))

    featured_df = (
        featured_df.withColumn("listed_timestamp", F.to_timestamp("listed_date"))
        .withColumn(
            "days_listed",
            F.coalesce(
                F.datediff(F.to_date(F.lit(reference_date)), F.to_date("listed_timestamp")),
                F.lit(-1),
            ),
        )
        .drop("listed_timestamp")
        .withColumn("bed_bath_ratio", F.round(F.col("bedrooms") / _safe_denominator("bathrooms"), 2))
        .withColumn("total_rooms", total_rooms)
        .withColumn("area_per_room", F.col("area_value") / safe_total_rooms)
        .withColumn("area_x_beds", F.col("area_value") * F.col("bedrooms"))
        .withColumn("log_area", F.log1p("area_value"))
        .withColumn("is_furnished", (F.upper(F.trim(F.col("furnished"))) == F.lit("YES")).cast("int"))
        .withColumn(
            "completion_score",
            F.when(F.lower(F.trim(F.col("completion_status"))).isin("completed", "completed_primary"), F.lit(2))
            .when(F.lower(F.trim(F.col("completion_status"))).isin("off_plan", "off_plan_primary"), F.lit(0))
            .otherwise(F.lit(1)),
        )
        .withColumn(
            "has_installments",
            (payment_method.contains("installments") | payment_method.isin("both")).cast("int"),
        )
    )

    for column_name in BOOLEAN_FLAG_COLUMNS:
        if column_name in featured_df.columns:
            featured_df = featured_df.withColumn(column_name, _boolean_to_int(column_name))

    return featured_df


def add_target_encodings(
    df: DataFrame,
    categorical_cols: Iterable[str] = DEFAULT_CATEGORICAL_TARGET_COLUMNS,
    target_col: str = DEFAULT_TARGET_COL,
    smoothing: float = 10.0,
) -> DataFrame:
    global_mean = df.agg(F.mean(target_col).alias("global_mean")).first()["global_mean"]
    if global_mean is None:
        raise ValueError(f"Cannot target encode because {target_col!r} has no non-null values.")

    encoded_df = df
    for column_name in categorical_cols:
        mean_col = f"__{column_name}_mean"
        count_col = f"__{column_name}_count"
        enc_col = f"{column_name}_enc"

        stats = (
            df.groupBy(column_name)
            .agg(F.mean(target_col).alias(mean_col), F.count(target_col).alias(count_col))
            .withColumn(
                enc_col,
                ((F.col(count_col) * F.col(mean_col)) + (F.lit(smoothing) * F.lit(global_mean)))
                / (F.col(count_col) + F.lit(smoothing)),
            )
            .select(column_name, enc_col)
        )

        encoded_df = (
            encoded_df.join(stats, on=column_name, how="left")
            .withColumn(enc_col, F.coalesce(F.col(enc_col), F.lit(global_mean)))
        )

    return encoded_df


def fit_target_encoding_model(
    df: DataFrame,
    categorical_cols: Iterable[str] = DEFAULT_CATEGORICAL_TARGET_COLUMNS,
    target_col: str = DEFAULT_TARGET_COL,
    smoothing: float = 10.0,
) -> TargetEncodingModel:
    categorical_cols = tuple(categorical_cols)
    global_mean = df.agg(F.mean(target_col).alias("global_mean")).first()["global_mean"]
    if global_mean is None:
        raise ValueError(f"Cannot target encode because {target_col!r} has no non-null values.")

    mappings: dict[str, dict[Any, float]] = {}
    for column_name in categorical_cols:
        rows = (
            df.groupBy(column_name)
            .agg(F.mean(target_col).alias("mean"), F.count(target_col).alias("count"))
            .withColumn(
                "encoded",
                ((F.col("count") * F.col("mean")) + (F.lit(smoothing) * F.lit(global_mean)))
                / (F.col("count") + F.lit(smoothing)),
            )
            .select(column_name, "encoded")
            .collect()
        )
        mappings[column_name] = {row[column_name]: float(row["encoded"]) for row in rows}

    return TargetEncodingModel(float(global_mean), categorical_cols, mappings)


def apply_target_encoding_model(df: DataFrame, model: TargetEncodingModel) -> DataFrame:
    encoded_df = df
    for column_name in model.categorical_cols:
        enc_col = f"{column_name}_enc"
        mapping = model.mappings.get(column_name, {})
        if mapping:
            mapping_entries = []
            for key, value in mapping.items():
                mapping_entries.extend([F.lit(str(key)), F.lit(float(value))])
            mapping_expr = F.create_map(*mapping_entries)
            encoded_df = encoded_df.withColumn(
                enc_col,
                F.coalesce(F.element_at(mapping_expr, F.col(column_name).cast("string")), F.lit(model.global_mean)),
            )
        else:
            encoded_df = encoded_df.withColumn(enc_col, F.lit(model.global_mean))
    return encoded_df


def serialize_target_encoding_model(model: TargetEncodingModel) -> dict[str, Any]:
    return {
        "global_mean": model.global_mean,
        "categorical_cols": list(model.categorical_cols),
        "mappings": {
            column_name: {str(key): value for key, value in values.items()}
            for column_name, values in model.mappings.items()
        },
    }


def deserialize_target_encoding_model(payload: dict[str, Any]) -> TargetEncodingModel:
    return TargetEncodingModel(
        global_mean=float(payload["global_mean"]),
        categorical_cols=tuple(payload["categorical_cols"]),
        mappings={
            column_name: dict(values)
            for column_name, values in payload.get("mappings", {}).items()
        },
    )


def build_model_features(
    df: DataFrame,
    reference_date: str = DEFAULT_REFERENCE_DATE,
    target_col: str = DEFAULT_TARGET_COL,
    smoothing: float = 10.0,
) -> DataFrame:
    return add_target_encodings(
        add_listing_features(df, reference_date=reference_date),
        categorical_cols=DEFAULT_CATEGORICAL_TARGET_COLUMNS,
        target_col=target_col,
        smoothing=smoothing,
    )


def feature_correlations(df: DataFrame, feature_cols: Iterable[str], target_col: str) -> DataFrame:
    spark = df.sparkSession
    rows = [(feature_col, df.stat.corr(feature_col, target_col)) for feature_col in feature_cols]
    return spark.createDataFrame(rows, ["feature", "pearson_r"]).orderBy("pearson_r")


def write_feature_summary(df: DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["feature", "dtype"])
        writer.writerows(df.dtypes)
