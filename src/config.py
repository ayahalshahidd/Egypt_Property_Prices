from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

RAW_DATA_PATH = RAW_DATA_DIR / "propertyfinder.csv"
INTERIM_CLEANED_PATH = INTERIM_DATA_DIR / "propertyFinder_cleaned_csv"
PROCESSED_FEATURES_PATH = PROCESSED_DATA_DIR / "propertyFinder_features_csv"

TARGET = "price_per_sqm"
RANDOM_STATE = 42
TEST_SIZE = 0.2
