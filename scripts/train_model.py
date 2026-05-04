from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


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
    TARGET,
)
from src.data.load_data import configure_local_spark, get_spark, load_raw_data, write_csv  # noqa: E402
from src.data.preprocess import clean_property_data, write_data_validation_report, write_missing_values  # noqa: E402
from src.features.build_features import build_model_features, write_feature_summary  # noqa: E402
from src.models.train import (  # noqa: E402
    load_model_params,
    prepare_model_data,
    save_model_bundle,
    split_and_scale,
    train_models,
    write_model_reports,
)
from src.visualization.visualize import (  # noqa: E402
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    plot_target_distribution,
)


def main() -> None:
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

    print("Preparing buy-listing model data")
    X, y = prepare_model_data(feature_df)
    splits = split_and_scale(X, y)
    params = load_model_params(CONFIGS_DIR / "model_params.json")

    print(f"Training models on {len(splits['X_train']):,} rows; testing on {len(splits['X_test']):,}")
    results_df, best_bundle = train_models(splits, params)
    model_path = save_model_bundle(best_bundle, MODELS_DIR)
    write_model_reports(results_df, best_bundle, RESULTS_DIR)

    plot_target_distribution(y.map(lambda value: float(np.expm1(value))), FIGURES_DIR / "target_distribution.png")
    plot_model_comparison(results_df, FIGURES_DIR / "model_comparison.png")
    plot_feature_importance(best_bundle["model"], best_bundle["features"], FIGURES_DIR / "feature_importance.png")

    y_pred = best_bundle["model"].predict(
        best_bundle["scaler"].transform(splits["X_test"])
        if best_bundle.get("scaler") is not None
        else splits["X_test"]
    )
    plot_residuals(splits["y_test"], y_pred, FIGURES_DIR / "residuals.png")

    print(f"Best model: {best_bundle['model_name']}")
    print(f"Saved model: {model_path}")
    print(f"Reports: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
