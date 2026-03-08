import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client(tmp_path_factory: pytest.TempPathFactory):
    test_db = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{test_db}"
    os.environ["CREATE_TABLES"] = "true"
    os.environ["DISABLE_RAG"] = "true"

    from app.main import app  # imported after env set

    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_feedback_roundtrip(client: TestClient):
    signup = client.post("/api/auth/signup", json={"email": "feedback@example.com", "password": "secret123"})
    assert signup.status_code == 200
    token = signup.json()["token"]
    headers = {"X-Session-Token": token}

    pet = client.post(
        "/api/pets",
        headers=headers,
        json={
            "name": "Milo",
            "species": "dog",
            "breed": "Pug",
            "age_years": 2,
            "weight_kg": 8,
            "is_neutered": True,
            "activity_level": "moderate",
            "health_conditions": None,
            "allergies": None,
        },
    )
    assert pet.status_code == 200
    pet_id = pet.json()["id"]

    create_feedback = client.post(
        "/api/feedback",
        headers=headers,
        json={
            "pet_id": pet_id,
            "page": "feedback",
            "category": "idea",
            "rating": 5,
            "message": "Please add export support for meal plans.",
        },
    )
    assert create_feedback.status_code == 200
    body = create_feedback.json()
    assert body["pet_id"] == pet_id
    assert body["category"] == "idea"
    assert body["rating"] == 5

    listing = client.get("/api/feedback", headers=headers)
    assert listing.status_code == 200
    rows = listing.json()
    assert len(rows) >= 1
    assert rows[0]["message"] == "Please add export support for meal plans."
