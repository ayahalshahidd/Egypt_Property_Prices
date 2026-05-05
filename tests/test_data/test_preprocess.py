from __future__ import annotations

from src.data.preprocess import DROP_COLUMNS, add_target_and_remove_outliers, standardize_payment_method


def test_drop_columns_contains_personal_contact_fields():
    assert "contact_phone" in DROP_COLUMNS
    assert "agent_email" in DROP_COLUMNS
    assert "broker_phone" in DROP_COLUMNS


def test_payment_method_keeps_unknown_rows(spark):
    df = spark.sql(
        """
        SELECT 'cash' AS payment_method
        UNION ALL SELECT 'installments'
        UNION ALL SELECT 'cash | installments'
        UNION ALL SELECT 'mortgage'
        UNION ALL SELECT NULL
        """
    )

    cleaned, unknown_count = standardize_payment_method(df)

    assert cleaned.count() == 5
    assert unknown_count == 2
    assert {row["payment_method"] for row in cleaned.collect()} == {"cash", "installments", "both", "unknown"}


def test_outlier_filter_applies_business_sanity_bounds(spark):
    df = spark.sql(
        """
        SELECT 1000000.0 AS price_egp, 100.0 AS area_value
        UNION ALL SELECT 100.0, 100.0
        UNION ALL SELECT 500000000.0, 100.0
        """
    )

    cleaned, removed, min_price, max_price = add_target_and_remove_outliers(df)

    assert cleaned.count() == 1
    assert removed == 2
    assert min_price == 10_000.0
    assert max_price == 10_000.0
