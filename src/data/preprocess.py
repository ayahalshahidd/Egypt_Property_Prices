from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


DROP_COLUMNS = [
    "internal_id", "detail_url", "listing_type", "reference",
    "price_currency", "area_unit", "offering_type",
    "agent_id", "agent_name", "agent_email", "agent_is_super", "agent_languages",
    "broker_id", "broker_name", "broker_email", "broker_phone",
    "contact_phone", "contact_whatsapp", "contact_email",
    "scraped_at", "listed_date_parsed", "title", "description", "video_url", "rera",
]


@dataclass
class CleaningReport:
    initial_rows: int
    final_rows: int
    dropped_columns: int
    duplicates_removed: int
    critical_nulls_removed: int
    payment_rows_standardized_unknown: int
    outliers_removed: int
    bedroom_median: float
    bathroom_median: float
    min_price_per_sqm: float
    max_price_per_sqm: float


def drop_irrelevant_columns(df: DataFrame, drop_columns: list[str] | None = None) -> tuple[DataFrame, int]:
    columns = drop_columns or DROP_COLUMNS
    cols_to_drop = [column_name for column_name in columns if column_name in df.columns]
    return df.drop(*cols_to_drop), len(cols_to_drop)


def clean_numeric_columns(df: DataFrame) -> tuple[DataFrame, float, float, int]:
    is_numeric_regex = r"^[0-9]+(\.[0-9]*)?$"

    df = df.withColumn(
        "bedrooms",
        F.when(F.lower(F.trim(F.col("bedrooms"))) == "studio", F.lit(1.0))
        .when(F.trim(F.col("bedrooms")).rlike(r"^7\+$"), F.lit(7.0))
        .when(F.trim(F.col("bedrooms")).rlike(is_numeric_regex), F.col("bedrooms").cast(DoubleType()))
        .otherwise(None),
    )
    df = df.withColumn(
        "bathrooms",
        F.when(F.trim(F.col("bathrooms")).rlike(r"^7\+$"), F.lit(7.0))
        .when(F.trim(F.col("bathrooms")).rlike(is_numeric_regex), F.col("bathrooms").cast(DoubleType()))
        .otherwise(None),
    )

    for column_name in ["price_egp", "area_value", "images_count"]:
        cleaned_expr = F.regexp_replace(F.col(column_name), "[^0-9.]", "")
        df = df.withColumn(
            column_name,
            F.when(cleaned_expr.rlike(is_numeric_regex), cleaned_expr.cast(DoubleType())).otherwise(None),
        )

    df_numeric = df.filter(F.col("bedrooms").isNotNull() & F.col("bathrooms").isNotNull())
    bedroom_median = _median_or_default(df_numeric, "bedrooms", 2.0)
    bathroom_median = _median_or_default(df_numeric, "bathrooms", 1.0)

    rows_modified = (
        df.filter(F.col("bedrooms").isNull()).count()
        + df.filter(F.col("bathrooms").isNull()).count()
        + df.filter(F.col("furnished").isNull()).count()
    )

    df = (
        df.withColumn("bedrooms", F.coalesce(F.col("bedrooms"), F.lit(bedroom_median)))
        .withColumn("bathrooms", F.coalesce(F.col("bathrooms"), F.lit(bathroom_median)))
        .withColumn("images_count", F.coalesce(F.col("images_count"), F.lit(0.0)))
        .withColumn("furnished", F.coalesce(F.col("furnished"), F.lit("NO")))
        .withColumn("completion_status", F.coalesce(F.col("completion_status"), F.lit("unknown")))
        .withColumn("payment_method", F.coalesce(F.col("payment_method"), F.lit("unknown")))
    )

    return df, bedroom_median, bathroom_median, rows_modified


def standardize_payment_method(df: DataFrame) -> tuple[DataFrame, int]:
    normalized = F.lower(F.trim(F.coalesce(F.col("payment_method"), F.lit(""))))
    df = df.withColumn(
        "payment_method",
        F.when(normalized == "cash", "cash")
        .when(normalized == "installments", "installments")
        .when(normalized.isin("cash | installments", "installments | cash"), "both")
        .when(normalized.contains("cash") & normalized.contains("install"), "both")
        .when(normalized.contains("install"), "installments")
        .when(normalized.contains("cash"), "cash")
        .otherwise("unknown"),
    )

    return df, df.filter(F.col("payment_method") == "unknown").count()


def add_target_and_remove_outliers(
    df: DataFrame,
    price_quantiles: tuple[float, float] = (0.01, 0.99),
    area_bounds: tuple[float, float] = (20.0, 1500.0),
    price_per_sqm_bounds: tuple[float, float] = (1_000.0, 300_000.0),
) -> tuple[DataFrame, int, float, float]:
    df = df.withColumn("price_per_sqm", F.col("price_egp") / F.col("area_value"))
    lo_price, hi_price = df.approxQuantile("price_per_sqm", list(price_quantiles), 0.01)
    lo_area, hi_area = area_bounds
    min_price_per_sqm, max_price_per_sqm = price_per_sqm_bounds
    lo_price = max(float(lo_price), min_price_per_sqm)
    hi_price = min(float(hi_price), max_price_per_sqm)

    before_count = df.count()
    df = df.filter(
        (F.col("price_per_sqm") >= lo_price)
        & (F.col("price_per_sqm") <= hi_price)
        & (F.col("area_value") >= lo_area)
        & (F.col("area_value") <= hi_area)
    )
    after_count = df.count()
    stats = df.select(
        F.min("price_per_sqm").alias("min_sqm"),
        F.max("price_per_sqm").alias("max_sqm"),
    ).collect()[0]
    return df, before_count - after_count, float(stats["min_sqm"]), float(stats["max_sqm"])


def clean_property_data(df_raw: DataFrame) -> tuple[DataFrame, CleaningReport]:
    initial_rows = df_raw.count()
    df, dropped_columns = drop_irrelevant_columns(df_raw)

    before_dedup = df.count()
    df = df.dropDuplicates(subset=["listing_id"])
    duplicates_removed = before_dedup - df.count()

    df, bedroom_median, bathroom_median, _rows_modified = clean_numeric_columns(df)

    before_null_drop = df.count()
    df = df.dropna(subset=["price_egp", "area_value", "lat", "lon"])
    df = df.filter((F.col("price_egp") > 0) & (F.col("area_value") > 0))
    critical_nulls_removed = before_null_drop - df.count()

    df, payment_rows_standardized_unknown = standardize_payment_method(df)
    df, outliers_removed, min_price, max_price = add_target_and_remove_outliers(df)

    report = CleaningReport(
        initial_rows=initial_rows,
        final_rows=df.count(),
        dropped_columns=dropped_columns,
        duplicates_removed=duplicates_removed,
        critical_nulls_removed=critical_nulls_removed,
        payment_rows_standardized_unknown=payment_rows_standardized_unknown,
        outliers_removed=outliers_removed,
        bedroom_median=bedroom_median,
        bathroom_median=bathroom_median,
        min_price_per_sqm=min_price,
        max_price_per_sqm=max_price,
    )
    return df, report


def missing_values(df: DataFrame) -> DataFrame:
    expressions = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    row = df.select(expressions).collect()[0].asDict()
    rows = [(column_name, int(missing_count)) for column_name, missing_count in row.items()]
    return df.sparkSession.createDataFrame(rows, ["column", "missing_count"])


def write_missing_values(df: DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expressions = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    row = df.select(expressions).collect()[0].asDict()

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["column", "missing_count"])
        for column_name, missing_count in row.items():
            writer.writerow([column_name, int(missing_count)])


def write_data_validation_report(report: CleaningReport, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Validation Report", ""]
    for key, value in asdict(report).items():
        lines.append(f"- **{key}**: {value}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _median_or_default(df: DataFrame, column_name: str, default: float) -> float:
    values = df.approxQuantile(column_name, [0.5], 0.01)
    return float(values[0]) if values else default
