from __future__ import annotations

from src.data.preprocess import DROP_COLUMNS


def test_drop_columns_contains_personal_contact_fields():
    assert "contact_phone" in DROP_COLUMNS
    assert "agent_email" in DROP_COLUMNS
    assert "broker_phone" in DROP_COLUMNS
