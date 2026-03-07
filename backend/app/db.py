import os

# Load .env automatically so running `uvicorn app.main:app --reload` works
# without needing `--env-file` or manually exporting variables.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

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

    # Supabase Postgres requires SSL. psycopg2 honors sslmode.
    connect_args = {}
    sslmode = os.getenv("DATABASE_SSLMODE", "").strip().lower()
    if sslmode in {"require", "verify-ca", "verify-full"}:
        connect_args["sslmode"] = sslmode

    return create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)


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
