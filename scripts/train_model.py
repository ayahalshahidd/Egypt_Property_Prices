from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CONFIGS_DIR,
    FIGURES_DIR,
    INTERIM_CLEANED_PATH,
    MODELS_DIR,
    PROCESSED_FEATURES_PATH,
    RAW_DATA_PATH,
    RESULTS_DIR,
)
from src.data.load_data import configure_local_spark, get_spark, load_raw_data, write_csv  # noqa: E402
from src.data.preprocess import clean_property_data, write_data_validation_report, write_missing_values  # noqa: E402
from src.features.build_features import write_feature_summary  # noqa: E402
from src.features.build_features import (  # noqa: E402
    add_listing_features,
    apply_target_encoding_model,
    fit_target_encoding_model,
    serialize_target_encoding_model,
)
from src.models.train_spark import (  # noqa: E402
    load_spark_model_params,
    prepare_spark_model_data,
    save_spark_model_bundle,
    split_spark_data,
    train_spark_models,
    write_spark_model_reports,
)
from src.visualization.visualize import (  # noqa: E402
    plot_model_comparison,
    plot_residuals,
    plot_spark_feature_importance,
    plot_target_distribution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Spark ML Egypt property price pipeline.")
    parser.parse_args()

    configure_local_spark()
    spark = get_spark("EgyptPropertyPricesTraining")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from {RAW_DATA_PATH}")
    raw_df = load_raw_data(spark, RAW_DATA_PATH)

    print("Cleaning data")
    clean_df, cleaning_report = clean_property_data(raw_df)
    write_csv(clean_df, INTERIM_CLEANED_PATH)
    write_data_validation_report(cleaning_report, RESULTS_DIR / "data_validation_report.md")
    write_missing_values(clean_df, RESULTS_DIR / "missing_values.csv")

    print("Building listing features")
    listing_feature_df = add_listing_features(clean_df)
    raw_splits = split_spark_data(listing_feature_df)
    target_encoding = fit_target_encoding_model(raw_splits["train"].filter("category = 'buy'"))
    encoded_splits = {
        split_name: apply_target_encoding_model(split_df, target_encoding)
        for split_name, split_df in raw_splits.items()
    }
    feature_df = (
        encoded_splits["train"]
        .unionByName(encoded_splits["validation"])
        .unionByName(encoded_splits["test"])
    )
    write_csv(feature_df, PROCESSED_FEATURES_PATH)
    write_feature_summary(feature_df, RESULTS_DIR / "feature_summary.csv")

    print("Preparing buy-listing Spark model data")
    splits = {
        split_name: prepare_spark_model_data(split_df)
        for split_name, split_df in encoded_splits.items()
    }
    model_df = splits["train"].unionByName(splits["validation"]).unionByName(splits["test"])
    params = load_spark_model_params(CONFIGS_DIR / "spark_model_params.json")

    print(
        f"Training Spark ML models on {splits['train'].count():,} rows; "
        f"validating on {splits['validation'].count():,}; testing on {splits['test'].count():,}"
    )
    spark_results, best_bundle = train_spark_models(splits, params)
    best_bundle["target_encoding"] = serialize_target_encoding_model(target_encoding)
    write_spark_model_reports(spark_results, best_bundle, RESULTS_DIR)
    model_path = save_spark_model_bundle(best_bundle, MODELS_DIR)

    y = pd.Series(np.expm1(_collect_numeric_column(model_df, "label")))
    plot_target_distribution(y, FIGURES_DIR / "target_distribution.png")
    spark_results_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k != "fitted"} for result in spark_results]
    )
    plot_model_comparison(spark_results_df, FIGURES_DIR / "spark_model_comparison.png")
    plot_spark_feature_importance(
        best_bundle["model"],
        best_bundle["features"],
        FIGURES_DIR / "spark_feature_importance.png",
    )
    spark_predictions = best_bundle["model"].transform(splits["test"]).select("label", "prediction")
    spark_predictions_df = pd.DataFrame(
        {
            "label": _collect_numeric_column(spark_predictions, "label"),
            "prediction": _collect_numeric_column(spark_predictions, "prediction"),
        }
    )
    plot_residuals(
        spark_predictions_df["label"],
        spark_predictions_df["prediction"].to_numpy(),
        FIGURES_DIR / "spark_residuals.png",
    )
    print("Spark backend writes comparison metrics and figures under reports/")

    print(f"Best model: {best_bundle['model_name']}")
    print(f"Saved model: {model_path}")
    print(f"Reports: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


def _collect_numeric_column(df, column_name: str, limit: int = 50_000) -> list[float]:
    return [float(row[column_name]) for row in df.select(column_name).limit(limit).toLocalIterator()]


if __name__ == "__main__":
    main()
