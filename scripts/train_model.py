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
from src.features.build_features import build_model_features, write_feature_summary  # noqa: E402
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

    print("Building features")
    feature_df = build_model_features(clean_df)
    write_csv(feature_df, PROCESSED_FEATURES_PATH)
    write_feature_summary(feature_df, RESULTS_DIR / "feature_summary.csv")

    print("Preparing buy-listing Spark model data")
    model_df = prepare_spark_model_data(feature_df)
    splits = split_spark_data(model_df)
    params = load_spark_model_params(CONFIGS_DIR / "spark_model_params.json")

    print(
        f"Training Spark ML models on {splits['train'].count():,} rows; "
        f"validating on {splits['validation'].count():,}; testing on {splits['test'].count():,}"
    )
    spark_results, best_bundle = train_spark_models(splits, params)
    write_spark_model_reports(spark_results, best_bundle, RESULTS_DIR)
    model_path = save_spark_model_bundle(best_bundle, MODELS_DIR)

    y = model_df.select("label").toPandas()["label"]
    plot_target_distribution(y.map(lambda value: float(np.expm1(value))), FIGURES_DIR / "target_distribution.png")
    spark_results_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k != "fitted"} for result in spark_results]
    )
    plot_model_comparison(spark_results_df, FIGURES_DIR / "spark_model_comparison.png")
    plot_spark_feature_importance(
        best_bundle["model"],
        best_bundle["features"],
        FIGURES_DIR / "spark_feature_importance.png",
    )
    spark_predictions = best_bundle["model"].transform(splits["test"]).select("label", "prediction").toPandas()
    plot_residuals(
        spark_predictions["label"],
        spark_predictions["prediction"].to_numpy(),
        FIGURES_DIR / "spark_residuals.png",
    )
    print("Spark backend writes comparison metrics and figures under reports/")

    print(f"Best model: {best_bundle['model_name']}")
    print(f"Saved model: {model_path}")
    print(f"Reports: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
