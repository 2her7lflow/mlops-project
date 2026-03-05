# backend/ — FastAPI API (DB + Nutrition + RAG)

This folder is the **server**. It exposes HTTP endpoints (FastAPI) for:
- pets (CRUD-ish)
- nutrition chat (RAG answers + nutrition calculator)
- admin actions (build the RAG index)

## Key entrypoints
- `main.py` — tiny entrypoint that imports `app.main:app` (so `uvicorn main:app` works)
- `app/main.py` — real FastAPI app + router wiring
- `rag_engine.py` — the real RAG implementation (PageIndex-inspired page/row-first + hybrid + guardrails)
- `app/rag_engine.py` — thin wrapper so `app.*` imports use the same engine
- `app/services/` — business logic (RAG service, nutrition service, MLflow logging)

## Environment variables (most important)
These can be in your shell env or in `backend/.env`:

- `DATABASE_URL` — Postgres/Supabase connection string (required)
- `GOOGLE_API_KEY` — Gemini API key (required for LLM answers / RAGAS)
- `KNOWLEDGE_BASE_DIR` — optional override for KB location

RAG knobs:
- `PROMPT_VERSION` = v1 or v2 (v2 is stricter guardrails)
- `RAG_K_VECTOR`, `RAG_K_BM25`, `RAG_K_PAGES`, `RAG_K_PER_PAGE`, `RAG_K_FINAL`
- `RAG_MIN_RELEVANCE` (guardrail threshold)
- `RAG_PAGE_TEXT_MAX_CHARS` (truncate page/row text before indexing)

MLflow:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`

## Run locally
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit backend/.env
uvicorn main:app --reload --port 8000
```

Then open:
- `http://localhost:8000/docs`

## Run (Docker)
### Dockerfile
```bash
docker build -t pet-backend:latest .
docker run --rm -p 8000:8000 --env-file .env pet-backend:latest
```

### docker-compose
```bash
docker compose up --build
```

## Build the KB index
```powershell
Invoke-RestMethod -Method Post http://localhost:8000/admin/setup-rag
```

## Run tests
```powershell
cd backend
pytest -q
```


## Repo hygiene (cleaned)
- Removed generated folders: `__pycache__/`, `*.pyc`, and `mlruns/` (these are created at runtime).
- Removed committed `.env` (use `.env.example` and create your own `.env`).
- `models.py` and `nutrition_calculator.py` at repo root are now **thin re-exports** of `app/*` to avoid duplicated logic.



## Auth

This backend uses a simple session-token auth (header `X-Session-Token`).

Endpoints:
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`

Most `/api/pets`, `/api/activity`, `/api/nutrition` endpoints require auth.
