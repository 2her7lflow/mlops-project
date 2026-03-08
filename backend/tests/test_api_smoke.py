import os
from uuid import uuid4

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


def test_feedback_roundtrip(client: TestClient):
    email = f"feedback-{uuid4().hex[:8]}@example.com"
    signup = client.post("/api/auth/signup", json={"email": email, "password": "secret123"})
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


def test_chat_logging_and_feedback_summary(client: TestClient):
    email = f"chatlogs-{uuid4().hex[:8]}@example.com"
    signup = client.post("/api/auth/signup", json={"email": email, "password": "secret123"})
    assert signup.status_code == 200
    token = signup.json()["token"]
    headers = {"X-Session-Token": token}

    pet = client.post(
        "/api/pets",
        headers=headers,
        json={
            "name": "Nori",
            "species": "cat",
            "breed": "Domestic Shorthair",
            "age_years": 3,
            "weight_kg": 5,
            "is_neutered": True,
            "activity_level": "moderate",
            "health_conditions": None,
            "allergies": None,
        },
    )
    assert pet.status_code == 200
    pet_id = pet.json()["id"]

    chat = client.post(
        "/api/nutrition/chat",
        headers=headers,
        json={"pet_id": pet_id, "question": "What portion size is right for my cat?"},
    )
    assert chat.status_code == 200
    chat_body = chat.json()
    assert chat_body["answer"]
    assert chat_body["chat_log_id"]

    logs = client.get("/api/chat/logs", headers=headers)
    assert logs.status_code == 200
    log_rows = logs.json()
    assert len(log_rows) >= 1
    assert log_rows[0]["question"] == "What portion size is right for my cat?"

    vote = client.post(
        "/api/chat/feedback",
        headers=headers,
        json={"chat_log_id": chat_body["chat_log_id"], "rating": -1, "reason": "incorrect", "comment": "Need a clearer portion estimate."},
    )
    assert vote.status_code == 200
    assert vote.json()["rating"] == -1

    summary = client.get("/api/chat/summary", headers=headers)
    assert summary.status_code == 200
    summary_body = summary.json()
    assert summary_body["total_chats"] >= 1
    assert summary_body["avg_latency_ms"] >= 0
    assert summary_body["negative_feedback_rate"] >= 0
