from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data.load_data import configure_local_spark, get_spark, write_csv  # noqa: E402
from src.models.predict_spark import load_spark_model_bundle, predict_price_per_sqm_spark, read_prediction_input  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate price-per-sqm predictions from a saved model.")
    parser.add_argument("--model", type=Path, default=None, help="Path to a saved Spark model bundle.")
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "prediction_input.csv")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or RESULTS_DIR / "spark_predictions_csv"
    configure_local_spark()
    spark = get_spark("EgyptPropertyPricesPrediction")
    model_path = args.model or latest_spark_model_path()
    bundle = load_spark_model_bundle(model_path)
    data = read_prediction_input(spark, args.input)
    predictions = predict_price_per_sqm_spark(bundle, data)
    write_csv(predictions, output_path)
    print(f"Predictions written to {output_path}")


def latest_spark_model_path() -> Path:
    candidates = sorted(
        [path for path in MODELS_DIR.glob("*_spark_buy_price_per_sqm_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No Spark model artifacts found in {MODELS_DIR}")
    return candidates[0]


if __name__ == "__main__":
    main()
