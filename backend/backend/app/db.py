import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

DEFAULT_SQLITE_URL = "sqlite:///./dev.db"


def make_engine():
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        # School-project friendly default.
        database_url = DEFAULT_SQLITE_URL

    if database_url.startswith("sqlite"):
        return create_engine(database_url, connect_args={"check_same_thread": False})

    return create_engine(database_url, pool_pre_ping=True)


engine = make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create tables for demo usage. In production use Alembic migrations."""
    if os.getenv("CREATE_TABLES", "true").lower() in {"1", "true", "yes"}:
        Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
