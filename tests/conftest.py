from __future__ import annotations

import pytest

from src.data.load_data import configure_local_spark, get_spark


@pytest.fixture(scope="session")
def spark():
    configure_local_spark()
    session = get_spark("EgyptPropertyPricesTests", master="local[1]", allow_fallback=False)
    yield session
    session.stop()
