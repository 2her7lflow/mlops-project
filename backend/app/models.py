# models.py
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Pet(Base):
    __tablename__ = "pets"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True, nullable=False)

    name = Column(String, nullable=False)
    species = Column(String, nullable=False)  # "dog" or "cat"
    breed = Column(String, nullable=False)

    age_years = Column(Float, nullable=False)
    weight_kg = Column(Float, nullable=False)

    is_neutered = Column(Boolean, default=False, nullable=False)
    activity_level = Column(String, nullable=False)  # sedentary/moderate/active/very_active

    health_conditions = Column(Text, nullable=True)
    allergies = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    pet_id = Column(Integer, ForeignKey("pets.id", ondelete="CASCADE"), index=True, nullable=False)

    # Store as DATE so “adjust by day” comparisons work reliably
    activity_date = Column(Date, index=True, nullable=False)

    steps = Column(Integer, default=0, nullable=False)
    active_minutes = Column(Integer, default=0, nullable=False)
    calories_burned = Column(Float, default=0.0, nullable=False)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    token = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True, nullable=False)
    pet_id = Column(Integer, ForeignKey("pets.id", ondelete="SET NULL"), index=True, nullable=True)

    page = Column(String, nullable=False, default="general")
    category = Column(String, nullable=False, default="general")
    rating = Column(Integer, nullable=True)
    message = Column(Text, nullable=False)

    # Optional: chat vote fields (for 👍/👎)
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    corrected_answer = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
