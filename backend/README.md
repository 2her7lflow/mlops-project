# backend/ — Pet Nutrition AI Backend

FastAPI backend for a DSBA pet-care project: a **personalized diet planner** that adapts feeding advice based on a pet profile, activity logs, and a RAG-based nutrition assistant.

## What this backend does

- user signup, login, logout, and session-based authentication
- pet profile CRUD
- daily calorie calculation from pet profile
- activity logging and activity-adjusted feeding plan
- nutrition Q&A through a RAG pipeline grounded in your knowledge base
- simple health, metrics, and DB diagnostics endpoints
- user feedback collection for bugs, ideas, UX issues, and model quality notes

## Current architecture

```
Frontend (Gradio)
    -> FastAPI backend
        -> SQLAlchemy
            -> SQLite (local dev) or Supabase Postgres
        -> RAG service / Gemini or configured LLM provider
        -> knowledge_base/* documents
```

## Project structure

```
backend/
├── main.py                # backward-compatible entrypoint for `uvicorn main:app`
├── app/
│   ├── main.py            # real FastAPI app
│   ├── db.py              # DB engine/session/init
│   ├── models.py          # SQLAlchemy models
│   ├── schemas.py         # Pydantic request/response models
│   ├── auth.py            # password hashing + session auth helpers
│   ├── nutrition_calculator.py
│   ├── rag_engine.py
│   ├── routers/
│   │   ├── auth.py
│   │   ├── pets.py
│   │   ├── activity.py
│   │   ├── nutrition.py
│   │   ├── admin.py
│   │   └── system.py
│   └── services/
├── tests/
├── requirements.txt
└── Dockerfile
```

## Database tables

This version keeps only the tables that are currently used by the API:

- `users`
- `auth_sessions`
- `pets`
- `activity_logs`
- `feedback`

`nutrition_plans` was removed because the app calculates meal plans on demand from the pet profile and latest activity instead of reading a persisted plan table.

## Main API routes

### System
- `GET /`
- `GET /health`
- `GET /metrics`
- `GET /db`

### Auth
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`

### Pets
- `POST /api/pets`
- `GET /api/pets`
- `GET /api/pets/{pet_id}`
- `PUT /api/pets/{pet_id}`
- `DELETE /api/pets/{pet_id}`

Legacy compatibility:
- `POST /api/pets/register`
- `GET /api/pets/user/{email}`

### Activity
- `POST /api/activity/logs`
- `GET /api/activity/logs?pet_id=...&limit=...`
- `GET /api/activity/adjust/{pet_id}?activity_date=YYYY-MM-DD`

Legacy compatibility:
- `POST /api/activity/sync`

### Nutrition
- `POST /api/nutrition/chat`
- `GET /api/nutrition/calculate/{pet_id}`

### Feedback
- `POST /api/feedback`
- `GET /api/feedback?limit=20`

### Admin
- `POST /admin/setup-rag`

## Auth model

Protected routes use the header below:

```
X-Session-Token: <token>
```

The Gradio frontend already sends this header after login.

## Environment variables

Create `backend/.env` from `../.env.example`, or use the repo-root `.env`. The most important values are:

```env
DATABASE_URL=sqlite:///./dev.db
CREATE_TABLES=true
GOOGLE_API_KEY=your_key_here
KNOWLEDGE_BASE_DIR=../knowledge_base
DISABLE_RAG=false

# optional RAG tuning
PROMPT_VERSION=v2
RAG_K_VECTOR=8
RAG_K_BM25=8
RAG_K_PAGES=6
RAG_K_PER_PAGE=3
RAG_K_FINAL=6
RAG_MIN_RELEVANCE=0.15

# optional MLflow
ENABLE_MLFLOW_CHAT_LOGGING=false
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=pet-nutrition-api
```

Notes:
- local development can use SQLite
- deployment can use Supabase Postgres via `DATABASE_URL`
- set `DISABLE_RAG=true` if you want to test API and frontend flow without the LLM stack

## Run locally

### PowerShell
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item ..\.env.example .env
uvicorn main:app --reload --reload-include .\.env --port 8000
```

### Bash
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env
uvicorn main:app --reload --port 8000
```

Open Swagger UI at:

```
http://localhost:8000/docs
```

## Docker

```bash
docker build -t pet-backend:latest .
docker run --rm -p 8000:8000 --env-file .env pet-backend:latest
```

Or:

```bash
docker compose up --build
```

## Build or rebuild the RAG index

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/admin/setup-rag
```

## Minimal manual test flow

1. Sign up
2. Create a pet profile
3. Log activity for that pet
4. Generate an adjusted daily plan
5. Ask a nutrition question in `/api/nutrition/chat`
6. Submit UX or answer-quality feedback in `/api/feedback`

## Run tests

```powershell
cd backend
pytest -q
```

## Suggested next improvements

- add Alembic migrations instead of relying on `create_all()`
- add endpoint tests for auth, pets, activity, and nutrition
- persist chat/evaluation logs separately for monitoring
- add wearable-device ingestion once FitBark or similar data is available


## Supabase notes (schema updates)

This project uses SQLAlchemy `create_all` for school/demo usage. On Supabase Postgres, **existing tables are not auto-altered**.
If you already created the `feedback` table before enabling chat thumbs (👍/👎), run:

```sql
ALTER TABLE feedback
  ADD COLUMN IF NOT EXISTS question TEXT,
  ADD COLUMN IF NOT EXISTS answer TEXT,
  ADD COLUMN IF NOT EXISTS corrected_answer TEXT;
```

