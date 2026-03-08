from typing import List, Optional
from pydantic import BaseModel, ConfigDict, EmailStr


class PetCreate(BaseModel):
    user_email: EmailStr
    name: str
    species: str  # "dog" or "cat"
    breed: str
    age_years: float
    weight_kg: float
    is_neutered: bool = False
    activity_level: str = "moderate"
    health_conditions: Optional[str] = None
    allergies: Optional[str] = None


class PetResponse(BaseModel):
    id: int
    user_email: EmailStr
    name: str
    species: str
    breed: str
    age_years: float
    weight_kg: float
    is_neutered: bool
    activity_level: str
    health_conditions: Optional[str] = None
    allergies: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    pet_id: Optional[int] = None
    question: str


class ChatSource(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource] = []
    chat_log_id: Optional[int] = None


class CalorieCalculation(BaseModel):
    daily_calories: float
    meal_plan: dict
    recommendations: str


class ActivitySync(BaseModel):
    pet_id: int
    date: str  # YYYY-MM-DD
    steps: int
    active_minutes: int


class UserSignup(BaseModel):
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    email: EmailStr


class PetCreateAuth(BaseModel):
    name: str
    species: str  # "dog" or "cat"
    breed: str
    age_years: float
    weight_kg: float
    is_neutered: bool = False
    activity_level: str = "moderate"
    health_conditions: Optional[str] = None
    allergies: Optional[str] = None


class PetUpdate(BaseModel):
    name: Optional[str] = None
    species: Optional[str] = None
    breed: Optional[str] = None
    age_years: Optional[float] = None
    weight_kg: Optional[float] = None
    is_neutered: Optional[bool] = None
    activity_level: Optional[str] = None
    health_conditions: Optional[str] = None
    allergies: Optional[str] = None


class ActivityCreate(BaseModel):
    pet_id: int
    date: str  # YYYY-MM-DD
    steps: int = 0
    active_minutes: int = 0


class ActivityResponse(BaseModel):
    id: int
    pet_id: int
    activity_date: str
    steps: int
    active_minutes: int
    calories_burned: float

    model_config = ConfigDict(from_attributes=True)


class FeedbackCreate(BaseModel):
    pet_id: Optional[int] = None
    page: str = "general"
    category: str = "other"
    rating: Optional[int] = None

    # General feedback text (optional if `question`/`answer` are provided)
    message: Optional[str] = None

    # Optional: chat vote fields (for 👍/👎 after an answer)
    question: Optional[str] = None
    answer: Optional[str] = None
    corrected_answer: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    user_email: EmailStr
    pet_id: Optional[int] = None
    page: str
    category: str
    rating: Optional[int] = None
    message: str
    question: Optional[str] = None
    answer: Optional[str] = None
    corrected_answer: Optional[str] = None
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class ChatLogResponse(BaseModel):
    id: int
    pet_id: Optional[int] = None
    question: str
    answer: str
    route_type: str
    status: str
    latency_ms: float
    retrieved_docs_count: int
    model_name: Optional[str] = None
    source: str
    error_message: Optional[str] = None
    created_at: str


class ChatFeedbackCreate(BaseModel):
    chat_log_id: int
    rating: int
    reason: Optional[str] = None
    comment: Optional[str] = None


class ChatFeedbackResponse(BaseModel):
    id: int
    chat_log_id: int
    user_email: EmailStr
    rating: int
    reason: Optional[str] = None
    comment: Optional[str] = None
    created_at: str


class ChatSummaryResponse(BaseModel):
    total_chats: int
    avg_latency_ms: float
    negative_feedback_rate: float
    error_rate: float
