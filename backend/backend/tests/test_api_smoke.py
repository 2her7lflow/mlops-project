import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
    os.environ.setdefault("CREATE_TABLES", "true")
    os.environ.setdefault("DISABLE_RAG", "true")

    from app.main import app  # imported after env set

    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
