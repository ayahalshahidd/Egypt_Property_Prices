from __future__ import annotations

from src.features.build_features import AMENITY_FEATURES


def test_amenity_features_have_unique_output_names():
    output_names = [column_name for _keyword, column_name in AMENITY_FEATURES]

    assert len(output_names) == len(set(output_names))
    assert "has_pool" in output_names
    assert "has_ac" in output_names
