from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ..auth import require_user
from ..db import get_db
from ..models import Pet, User
from ..schemas import ChatRequest, ChatResponse, CalorieCalculation
from ..rag_engine import get_rag
from ..nutrition_calculator import NutritionCalculator
from ..services.mlflow_tracker import mlflow_run, log_params, log_metrics, now_ms

router = APIRouter(prefix="/api/nutrition", tags=["nutrition"])


def _get_pet_owned(db: Session, pet_id: int, user: User) -> Pet:
    pet = db.query(Pet).filter(Pet.id == pet_id, Pet.user_email == user.email).first()
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")
    return pet


@router.post("/chat", response_model=ChatResponse)
def nutrition_chat(request: ChatRequest, user: User = Depends(require_user), db: Session = Depends(get_db)):
    pet_context = None
    if request.pet_id:
        pet = _get_pet_owned(db, request.pet_id, user)
        pet_context = {
            "name": pet.name,
            "breed": pet.breed,
            "age_years": pet.age_years,
            "weight_kg": pet.weight_kg,
            "health_conditions": pet.health_conditions,
            "allergies": pet.allergies,
        }

    try:
        rag = get_rag()
        t0 = now_ms()
        result = rag.ask(request.question, pet_context)
        latency_ms = now_ms() - t0

        meta = result.get("_meta", {}) or {}

        # Optional: log to MLflow for production-like observability / monitoring
        if os.getenv("ENABLE_MLFLOW_CHAT_LOGGING", "false").lower() in {"1", "true", "yes"}:
            tags = {"component": "api", "endpoint": "/api/nutrition/chat"}
            with mlflow_run(run_name="chat", tags=tags):
                log_params(
                    {
                        "retrieval_strategy": meta.get("retrieval_strategy", getattr(rag, "retriever_mode", "unknown")),
                        "prompt_version": getattr(rag, "prompt_version", "unknown"),
                        "llm_provider": meta.get("llm_provider", getattr(rag, "llm_provider", "unknown")),
                        "llm_model": meta.get("llm_model", getattr(rag, "llm_model_name", "unknown")),
                        "k_vector": getattr(rag, "k_vector", None),
                        "k_bm25": getattr(rag, "k_bm25", None),
                        "k_pages": getattr(rag, "k_pages", None),
                        "k_per_page": getattr(rag, "k_per_page", None),
                        "k_final": getattr(rag, "k_final", None),
                        "min_relevance": getattr(rag, "min_relevance", None),
                    }
                )
                log_metrics(
                    {
                        "latency_total_ms": latency_ms,
                        "latency_retrieval_ms": meta.get("retrieval_ms", None),
                        "latency_llm_ms": meta.get("llm_ms", None),
                        "best_relevance": meta.get("best_relevance", None),
                        "contexts_used": meta.get("num_contexts", None),
                        "unique_sources": meta.get("unique_sources", None),
                        "page_index_hit": meta.get("page_index_hit", None),
                        "guardrail_no_context": meta.get("guardrail_no_context", 0.0),
                    }
                )

        sources = []
        for c in result.get("contexts", []) or []:
            sources.append(
                {
                    "source": c.get("source", "unknown"),
                    "page": c.get("page", None),
                    "snippet": c.get("snippet", None),
                }
            )

        return ChatResponse(answer=result.get("answer", ""), sources=sources)

    except Exception as e:
        # Provide a friendly fallback if RAG isn't configured/installed.
        # (common on Windows if vector store deps are missing)
        fallback = (
            "ระบบตอบคำถาม (RAG) ยังไม่พร้อมใช้งานใน environment นี้ "
            "— ตรวจสอบการตั้งค่า .env (เช่น GOOGLE_API_KEY/OPENROUTER_API_KEY) "
            "หรือใช้ DISABLE_RAG=true เพื่อทดสอบ flow ของ UI/DB ก่อน"
        )
        return ChatResponse(answer=fallback + f"\n\n(รายละเอียด: {type(e).__name__})", sources=[])


@router.get("/calculate/{pet_id}", response_model=CalorieCalculation)
def calculate_daily_calories(
    pet_id: int,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    pet = _get_pet_owned(db, pet_id, user)

    calc = NutritionCalculator()
    daily_calories = calc.calculate_der(
        weight_kg=pet.weight_kg,
        activity_level=pet.activity_level,
        age_years=pet.age_years,
        is_neutered=pet.is_neutered,
    )

    meal_plan = calc.calculate_food_amount(daily_calories, "kibble", age_years=pet.age_years)

    recommendations = calc.get_recommendations(pet.species, pet.age_years, pet.health_conditions)

    return CalorieCalculation(
        daily_calories=daily_calories,
        meal_plan=meal_plan,
        recommendations=recommendations,
    )
