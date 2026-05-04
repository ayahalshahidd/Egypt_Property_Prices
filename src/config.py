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

FEATURES = [
    "area_value", "log_area", "bedrooms", "bathrooms",
    "bed_bath_ratio", "total_rooms", "area_per_room", "area_x_beds",
    "lat", "lon",
    "city_enc", "town_enc", "district_enc", "property_type_enc",
    "completion_score", "is_furnished", "has_installments",
    "amenity_count", "has_pool", "has_gym", "has_security",
    "has_parking", "has_garden", "has_balcony",
    "has_private_pool", "has_spa", "has_ac",
    "days_listed",
    "is_premium", "is_verified", "is_new_construction",
    "is_direct_from_dev", "images_count",
]
