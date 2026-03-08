from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..auth import require_user
from ..db import get_db
from ..models import ChatFeedback, ChatLog, Feedback, User
from ..schemas import ChatFeedbackCreate, ChatFeedbackResponse, ChatLogResponse, ChatSummaryResponse

router = APIRouter(prefix="/api/chat", tags=["chat_logs"])

_VALID_REASONS = {"incorrect", "irrelevant", "unsafe", "too_vague", "other"}


def _to_log_response(row: ChatLog) -> ChatLogResponse:
    return ChatLogResponse(
        id=row.id,
        pet_id=row.pet_id,
        question=row.question,
        answer=row.answer,
        route_type=row.route_type,
        status=row.status,
        latency_ms=float(row.latency_ms or 0.0),
        retrieved_docs_count=int(row.retrieved_docs_count or 0),
        model_name=row.model_name,
        source=row.source,
        error_message=row.error_message,
        created_at=row.created_at.isoformat(),
    )


@router.get('/logs', response_model=list[ChatLogResponse])
def list_chat_logs(
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(ChatLog)
        .filter(ChatLog.user_email == user.email)
        .order_by(ChatLog.created_at.desc(), ChatLog.id.desc())
        .limit(limit)
        .all()
    )
    return [_to_log_response(r) for r in rows]


@router.get('/summary', response_model=ChatSummaryResponse)
def get_chat_summary(user: User = Depends(require_user), db: Session = Depends(get_db)):
    logs = db.query(ChatLog).filter(ChatLog.user_email == user.email).all()
    total = len(logs)
    if not total:
        return ChatSummaryResponse(total_chats=0, avg_latency_ms=0.0, negative_feedback_rate=0.0, error_rate=0.0)

    avg_latency = sum(float(r.latency_ms or 0.0) for r in logs) / total
    error_rate = sum(1 for r in logs if (r.status or '').lower() != 'success') / total

    rated_ids = [r.id for r in logs]
    feedback_rows = db.query(ChatFeedback).filter(ChatFeedback.user_email == user.email, ChatFeedback.chat_log_id.in_(rated_ids)).all() if rated_ids else []
    negative_count = sum(1 for r in feedback_rows if r.rating < 0)
    negative_rate = (negative_count / len(feedback_rows)) if feedback_rows else 0.0

    return ChatSummaryResponse(
        total_chats=total,
        avg_latency_ms=round(avg_latency, 2),
        negative_feedback_rate=round(negative_rate, 3),
        error_rate=round(error_rate, 3),
    )


@router.post('/feedback', response_model=ChatFeedbackResponse)
def create_chat_feedback(payload: ChatFeedbackCreate, user: User = Depends(require_user), db: Session = Depends(get_db)):
    if payload.rating not in {-1, 1}:
        raise HTTPException(status_code=400, detail='Rating must be -1 or 1')

    row = db.query(ChatLog).filter(ChatLog.id == payload.chat_log_id, ChatLog.user_email == user.email).first()
    if not row:
        raise HTTPException(status_code=404, detail='Chat log not found')

    reason = (payload.reason or '').strip().lower() or None
    if reason and reason not in _VALID_REASONS:
        raise HTTPException(status_code=400, detail=f'Reason must be one of: {sorted(_VALID_REASONS)}')

    existing = (
        db.query(ChatFeedback)
        .filter(ChatFeedback.chat_log_id == row.id, ChatFeedback.user_email == user.email)
        .first()
    )
    if existing:
        existing.rating = payload.rating
        existing.reason = reason
        existing.comment = (payload.comment or '').strip() or None
        db.commit()
        db.refresh(existing)
        feedback_row = existing
    else:
        feedback_row = ChatFeedback(
            chat_log_id=row.id,
            user_email=user.email,
            rating=payload.rating,
            reason=reason,
            comment=(payload.comment or '').strip() or None,
        )
        db.add(feedback_row)
        db.commit()
        db.refresh(feedback_row)

    # Also mirror a compact row into generic feedback so existing history view stays useful.
    mirrored = Feedback(
        user_email=user.email,
        pet_id=row.pet_id,
        page='advisor',
        category='accuracy',
        rating=payload.rating,
        message='chat_vote',
        question=row.question,
        answer=row.answer,
        corrected_answer=(payload.comment or '').strip() or None,
    )
    db.add(mirrored)
    db.commit()

    return ChatFeedbackResponse(
        id=feedback_row.id,
        chat_log_id=feedback_row.chat_log_id,
        user_email=feedback_row.user_email,
        rating=feedback_row.rating,
        reason=feedback_row.reason,
        comment=feedback_row.comment,
        created_at=feedback_row.created_at.isoformat(),
    )
