from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from ..auth import create_session, hash_password, require_user, verify_password
from ..db import get_db
from ..models import User, AuthSession
from ..schemas import UserSignup, UserLogin, AuthResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse)
def signup(payload: UserSignup, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    try:
        pw_hash = hash_password(payload.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user = User(email=email, password_hash=pw_hash)
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_session(db, user)
    return AuthResponse(token=token, email=user.email)


@router.post("/login", response_model=AuthResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_session(db, user)
    return AuthResponse(token=token, email=user.email)


@router.post("/logout")
def logout(
    user: User = Depends(require_user),
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
    db: Session = Depends(get_db),
):
    if x_session_token:
        sess = db.query(AuthSession).filter(AuthSession.token == x_session_token).first()
        if sess:
            db.delete(sess)
            db.commit()
    return {"status": "ok"}


@router.get("/me")
def me(user: User = Depends(require_user)):
    return {"email": user.email}
