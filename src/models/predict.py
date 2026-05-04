from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_price_per_sqm(bundle: dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    features = bundle["features"]
    X = data[features].copy()
    for column_name in features:
        X[column_name] = pd.to_numeric(X[column_name], errors="coerce")

    X_model = bundle["scaler"].transform(X) if bundle.get("scaler") is not None else X
    predictions_log = bundle["model"].predict(X_model)
    output = data.copy()
    output["predicted_price_per_sqm"] = np.expm1(predictions_log)
    return output
