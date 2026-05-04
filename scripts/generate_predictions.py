from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from src.models.predict import load_model_bundle, predict_price_per_sqm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate price-per-sqm predictions from a saved model.")
    parser.add_argument("--model", type=Path, default=None, help="Path to a saved .pkl model bundle.")
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "prediction_input.csv")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "predictions.csv")
    args = parser.parse_args()

    model_path = args.model or latest_model_path()
    bundle = load_model_bundle(model_path)
    data = pd.read_csv(args.input)
    predictions = predict_price_per_sqm(bundle, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")


def latest_model_path() -> Path:
    candidates = sorted(MODELS_DIR.glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pkl model artifacts found in {MODELS_DIR}")
    return candidates[0]


if __name__ == "__main__":
    main()
