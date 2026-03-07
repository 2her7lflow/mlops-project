from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from .db import get_db
from .models import User, AuthSession


_PBKDF2_ITERS = 200_000
_SESSION_TTL_HOURS = 24 * 7  # 7 days


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def _b64d(s: str) -> bytes:
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def hash_password(password: str) -> str:
    if not isinstance(password, str) or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return f"pbkdf2_sha256${_PBKDF2_ITERS}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, dk_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters_i = int(iters)
        salt = _b64d(salt_b64)
        dk_expected = _b64d(dk_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters_i)
        return hmac.compare_digest(dk, dk_expected)
    except Exception:
        return False


def create_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    sess = AuthSession(
        token=token,
        user_id=user.id,
        created_at=now,
        expires_at=now + timedelta(hours=_SESSION_TTL_HOURS),
    )
    db.add(sess)
    db.commit()
    return token


def get_user_by_token(db: Session, token: str) -> Optional[User]:
    if not token:
        return None
    sess = db.query(AuthSession).filter(AuthSession.token == token).first()
    if not sess:
        return None
    if sess.expires_at and sess.expires_at < datetime.utcnow():
        db.delete(sess)
        db.commit()
        return None
    return db.query(User).filter(User.id == sess.user_id).first()


def require_user(
    x_session_token: Optional[str] = Header(default=None, alias="X-Session-Token"),
    db: Session = Depends(get_db),
) -> User:
    user = get_user_by_token(db, x_session_token or "")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def optional_user(
    x_session_token: Optional[str] = Header(default=None, alias="X-Session-Token"),
    db: Session = Depends(get_db),
) -> Optional[User]:
    return get_user_by_token(db, x_session_token or "")
