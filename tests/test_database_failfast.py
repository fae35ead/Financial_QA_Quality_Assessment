from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from app.core import database


def test_init_db_fail_fast_when_fallback_disabled(monkeypatch):
    error = OperationalError("CREATE TABLE", {}, RuntimeError("db down"))

    monkeypatch.setattr(
        database,
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/qa_review",
        raising=False,
    )
    monkeypatch.setattr(database, "_as_bool", lambda name, default, **kwargs: False)
    monkeypatch.setattr(database.Base.metadata, "create_all", lambda bind: (_ for _ in ()).throw(error))

    with pytest.raises(OperationalError):
        database.init_db()
