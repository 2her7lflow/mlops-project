from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import require_user
from ..db import get_db
from ..metrics import record_rag
from ..models import ChatLog, Pet, User
from ..nutrition_calculator import NutritionCalculator
from ..rag_engine import get_rag
from ..schemas import CalorieCalculation, ChatRequest, ChatResponse
from ..services.mlflow_tracker import log_metrics, log_params, mlflow_run, now_ms
from ..services.nutrition_service import extract_pet_facts, is_calculation_question

router = APIRouter(prefix="/api/nutrition", tags=["nutrition"])


def _get_pet_owned(db: Session, pet_id: int, user: User) -> Pet:
    pet = db.query(Pet).filter(Pet.id == pet_id, Pet.user_email == user.email).first()
    if not pet:
        raise HTTPException(status_code=404, detail="Pet not found")
    return pet


def _has_thai(text: str) -> bool:
    return any("\u0E00" <= ch <= "\u0E7F" for ch in (text or ""))


def _create_chat_log(
    db: Session,
    *,
    user_email: str,
    pet_id: Optional[int],
    question: str,
    answer: str,
    route_type: str,
    status: str,
    latency_ms: float,
    retrieved_docs_count: int = 0,
    model_name: Optional[str] = None,
    source: str = "web_app",
    error_message: Optional[str] = None,
) -> int:
    row = ChatLog(
        user_email=user_email,
        pet_id=pet_id,
        question=question,
        answer=answer,
        route_type=route_type,
        status=status,
        latency_ms=float(latency_ms or 0.0),
        retrieved_docs_count=int(retrieved_docs_count or 0),
        model_name=model_name,
        source=source,
        error_message=error_message,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.id


def _build_calculation_answer(
    question: str,
    *,
    weight_kg: Optional[float],
    species: Optional[str],
    age_years: Optional[float],
    is_neutered: Optional[bool],
    activity_level: str,
    food_type: str,
    daily_calories: Optional[float] = None,
    meal_plan: Optional[dict] = None,
    recommendations: Optional[str] = None,
    missing_weight: bool = False,
    used_pet_profile: bool = False,
) -> str:
    thai = _has_thai(question)

    if missing_weight:
        if thai:
            return (
                "ฉันคำนวณปริมาณอาหารให้ได้ แต่ต้องรู้น้ำหนักของสัตว์เลี้ยงก่อน\n\n"
                "กรุณาระบุอย่างน้อย:\n"
                "- ชนิดสัตว์ (สุนัข/แมว)\n"
                "- น้ำหนัก เช่น 5 kg\n"
                "- อายุ ถ้ามี\n"
                "- ประเภทอาหาร เช่น อาหารเม็ด/อาหารเปียก\n\n"
                "ตัวอย่าง: แมวหนัก 5 kg ควรกินอาหารเม็ดวันละเท่าไร"
            )
        return (
            "I can calculate an estimated portion size, but I need the pet's weight first.\n\n"
            "Please include at least:\n"
            "- species (dog or cat)\n"
            "- weight, for example 5 kg\n"
            "- age, if known\n"
            "- food type, for example kibble or wet food\n\n"
            "Example: What portion size is right for a 5 kg cat on kibble?"
        )

    species_label_en = {"dog": "dog", "cat": "cat"}.get(species or "", "pet")
    species_label_th = {"dog": "สุนัข", "cat": "แมว"}.get(species or "", "สัตว์เลี้ยง")
    food_label_en = {
        "kibble": "dry food / kibble",
        "wet": "wet food",
        "raw": "raw food",
        "mixed": "mixed feeding",
    }.get(food_type, food_type)
    food_label_th = {
        "kibble": "อาหารเม็ด",
        "wet": "อาหารเปียก",
        "raw": "อาหารดิบ",
        "mixed": "อาหารแบบผสม",
    }.get(food_type, food_type)

    assumption_bits_en = []
    assumption_bits_th = []
    if age_years is None:
        assumption_bits_en.append("adult")
        assumption_bits_th.append("อายุโตเต็มวัย")
    if is_neutered is None:
        assumption_bits_en.append("neuter status not specified")
        assumption_bits_th.append("ไม่ได้ระบุสถานะการทำหมัน")
    activity_level = activity_level or "moderate"
    assumption_bits_en.append(f"{activity_level.replace('_', ' ')} activity")
    assumption_bits_th.append(f"กิจกรรมระดับ {activity_level.replace('_', ' ')}")
    assumption_bits_en.append(food_label_en)
    assumption_bits_th.append(food_label_th)

    intro_en = "using the saved pet profile" if used_pet_profile else "using the details in your question"
    intro_th = "โดยใช้ข้อมูลโปรไฟล์สัตว์เลี้ยงที่บันทึกไว้" if used_pet_profile else "โดยใช้รายละเอียดจากคำถามของคุณ"

    if thai:
        lines = [
            f"ปริมาณอาหารเริ่มต้นที่เหมาะสมสำหรับ{species_label_th}น้ำหนัก {weight_kg:.1f} กก. คือประมาณ:",
            f"- พลังงานต่อวัน: {daily_calories:.0f} kcal/วัน",
            f"- อาหารรวมต่อวัน: {meal_plan['total_grams_per_day']:.1f} กรัม/วัน",
            f"- แบ่งเป็น {meal_plan['meal_frequency']} มื้อ",
            f"- ต่อมื้อ: {meal_plan['grams_per_meal']:.1f} กรัม/มื้อ",
            f"- เทียบเป็นถ้วย: {meal_plan['cups_per_day']:.2f} ถ้วย/วัน",
            "",
            f"สมมติฐานที่ใช้ ({intro_th}): {', '.join(assumption_bits_th)}",
            "ตัวเลขนี้เป็นค่าเริ่มต้นเท่านั้น เพราะพลังงานจริงของอาหารแต่ละยี่ห้อไม่เท่ากัน ควรปรับตาม kcal บนฉลากอาหารและน้ำหนักตัวจริงของสัตว์เลี้ยง",
        ]
        if recommendations:
            lines.extend(["", f"คำแนะนำเพิ่มเติม: {recommendations}"])
        return "\n".join(lines)

    lines = [
        f"A reasonable starting portion for a {weight_kg:.1f} kg {species_label_en} is:",
        f"- Daily energy: {daily_calories:.0f} kcal/day",
        f"- Total food: {meal_plan['total_grams_per_day']:.1f} g/day",
        f"- Split into {meal_plan['meal_frequency']} meals",
        f"- Per meal: {meal_plan['grams_per_meal']:.1f} g/meal",
        f"- Approximate cups: {meal_plan['cups_per_day']:.2f} cups/day",
        "",
        f"Assumptions used ({intro_en}): {', '.join(assumption_bits_en)}.",
        "This is only a starting estimate because actual calorie density varies by brand and formula. The most accurate adjustment is to check the food label kcal and monitor body weight over time.",
    ]
    if recommendations:
        lines.extend(["", f"Additional note: {recommendations}"])
    return "\n".join(lines)


@router.post("/chat", response_model=ChatResponse)
def nutrition_chat(request: ChatRequest, user: User = Depends(require_user), db: Session = Depends(get_db)):
    pet_context = None
    pet = None
    if request.pet_id:
        pet = _get_pet_owned(db, request.pet_id, user)
        pet_context = {
            "name": pet.name,
            "species": pet.species,
            "breed": pet.breed,
            "age_years": pet.age_years,
            "weight_kg": pet.weight_kg,
            "is_neutered": pet.is_neutered,
            "activity_level": pet.activity_level,
            "health_conditions": pet.health_conditions,
            "allergies": pet.allergies,
        }

    question = (request.question or "").strip()
    try:
        if is_calculation_question(question):
            t0 = now_ms()
            calc = NutritionCalculator()
            extracted = extract_pet_facts(question)

            species = pet.species if pet else extracted.get("species")
            weight_kg = pet.weight_kg if pet else extracted.get("weight_kg")
            age_years = pet.age_years if pet else extracted.get("age_years")
            is_neutered = pet.is_neutered if pet else extracted.get("is_neutered")
            activity_level = pet.activity_level if pet else extracted.get("activity_level") or "moderate"
            food_type = extracted.get("food_type") or "kibble"
            recommendations = calc.get_recommendations(species or "dog", age_years, pet.health_conditions if pet else None)

            if weight_kg is None:
                answer = _build_calculation_answer(
                    question,
                    weight_kg=None,
                    species=species,
                    age_years=age_years,
                    is_neutered=is_neutered,
                    activity_level=activity_level,
                    food_type=food_type,
                    missing_weight=True,
                    used_pet_profile=bool(pet),
                )
                latency_ms = now_ms() - t0
                chat_log_id = _create_chat_log(
                    db,
                    user_email=user.email,
                    pet_id=pet.id if pet else None,
                    question=question,
                    answer=answer,
                    route_type="calculator",
                    status="success",
                    latency_ms=latency_ms,
                    model_name="nutrition_calculator",
                )

                if os.getenv("ENABLE_MLFLOW_CHAT_LOGGING", "false").lower() in {"1", "true", "yes"}:
                    with mlflow_run(run_name="chat", tags={"component": "api", "endpoint": "/api/nutrition/chat"}):
                        log_params({"route_type": "calculator_missing_input"})
                        log_metrics({"latency_total_ms": latency_ms, "calc_routed": 1.0})

                record_rag(route_type="calculator", latency_ms=latency_ms)
                return ChatResponse(answer=answer, sources=[], chat_log_id=chat_log_id)

            daily_calories = calc.calculate_der(
                weight_kg=weight_kg,
                activity_level=activity_level,
                age_years=age_years,
                is_neutered=bool(is_neutered),
            )
            meal_plan = calc.calculate_food_amount(
                daily_calories=daily_calories,
                food_type=food_type,
                age_years=age_years,
            )
            answer = _build_calculation_answer(
                question,
                weight_kg=weight_kg,
                species=species,
                age_years=age_years,
                is_neutered=is_neutered,
                activity_level=activity_level,
                food_type=food_type,
                daily_calories=daily_calories,
                meal_plan=meal_plan,
                recommendations=recommendations,
                used_pet_profile=bool(pet),
            )
            latency_ms = now_ms() - t0
            chat_log_id = _create_chat_log(
                db,
                user_email=user.email,
                pet_id=pet.id if pet else None,
                question=question,
                answer=answer,
                route_type="calculator",
                status="success",
                latency_ms=latency_ms,
                model_name="nutrition_calculator",
            )

            if os.getenv("ENABLE_MLFLOW_CHAT_LOGGING", "false").lower() in {"1", "true", "yes"}:
                with mlflow_run(run_name="chat", tags={"component": "api", "endpoint": "/api/nutrition/chat"}):
                    log_params(
                        {
                            "route_type": "calculator",
                            "calc_food_type": food_type,
                            "calc_activity_level": activity_level,
                        }
                    )
                    log_metrics(
                        {
                            "latency_total_ms": latency_ms,
                            "latency_retrieval_ms": 0.0,
                            "latency_llm_ms": 0.0,
                            "calc_routed": 1.0,
                            "daily_calories": daily_calories,
                        }
                    )

            record_rag(route_type="calculator", latency_ms=latency_ms)
            return ChatResponse(answer=answer, sources=[], chat_log_id=chat_log_id)

        rag = get_rag()
        t0 = now_ms()
        result = rag.ask(question, pet_context)
        latency_ms = now_ms() - t0

        meta = result.get("_meta", {}) or {}

        if os.getenv("ENABLE_MLFLOW_CHAT_LOGGING", "false").lower() in {"1", "true", "yes"}:
            tags = {"component": "api", "endpoint": "/api/nutrition/chat"}
            with mlflow_run(run_name="chat", tags=tags):
                log_params(
                    {
                        "route_type": "rag",
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
                        "latency_llm_ms": (meta.get("draft_llm_ms", 0.0) or 0.0) + (meta.get("safety_llm_ms", 0.0) or 0.0),
                        "latency_draft_llm_ms": meta.get("draft_llm_ms", None),
                        "latency_safety_llm_ms": meta.get("safety_llm_ms", None),
                        "latency_triage_ms": meta.get("triage_ms", None),
                        "best_relevance": meta.get("best_relevance", None),
                        "contexts_used": meta.get("num_contexts", None),
                        "unique_sources": meta.get("unique_sources", None),
                        "page_index_hit": meta.get("page_index_hit", None),
                        "guardrail_no_context": meta.get("guardrail_no_context", 0.0),
                        "safety_review_run": 1.0 if meta.get("safety_review_run") else 0.0,
                        "calc_routed": 0.0,
                    }
                )

        sources = []
        for c in result.get("sources", []) or []:
            sources.append(
                {
                    "source": c.get("source", "unknown"),
                    "page": c.get("page", None),
                    "snippet": c.get("snippet", None),
                }
            )

        record_rag(route_type="rag", latency_ms=latency_ms, meta=meta, sources_count=len(sources))
        chat_log_id = _create_chat_log(
            db,
            user_email=user.email,
            pet_id=pet.id if pet else None,
            question=question,
            answer=result.get("answer", ""),
            route_type="rag",
            status="success",
            latency_ms=latency_ms,
            retrieved_docs_count=len(sources),
            model_name=meta.get("llm_model") or getattr(rag, "llm_model_name", None),
        )
        return ChatResponse(answer=result.get("answer", ""), sources=sources, chat_log_id=chat_log_id)

    except Exception as e:
        fallback = (
            "ระบบตอบคำถาม (RAG) ยังไม่พร้อมใช้งานใน environment นี้ "
            "— ตรวจสอบการตั้งค่า .env (เช่น GOOGLE_API_KEY/OPENROUTER_API_KEY) "
            "หรือใช้ DISABLE_RAG=true เพื่อทดสอบ flow ของ UI/DB ก่อน"
        )
        answer = fallback + f"\n\n(รายละเอียด: {type(e).__name__})"
        chat_log_id = _create_chat_log(
            db,
            user_email=user.email,
            pet_id=pet.id if pet else None,
            question=question,
            answer=answer,
            route_type="rag",
            status="error",
            latency_ms=0.0,
            model_name=None,
            error_message=type(e).__name__,
        )
        return ChatResponse(answer=answer, sources=[], chat_log_id=chat_log_id)


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
