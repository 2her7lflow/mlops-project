# backend/app/ — Application package (FastAPI wiring)

Think of `backend/app/` as “the FastAPI application code”:
- It defines routes (`routers/`)
- It defines data contracts (`schemas.py`)
- It defines services (`services/`)
- It initializes DB + RAG warmup (`main.py` lifespan)

## Important files
- `main.py` — creates `FastAPI()` and attaches routers
- `db.py` — SQLAlchemy setup and DB init helper
- `schemas.py` — request/response models (Pydantic)
- `metrics.py` — simple request latency tracking
- `rag_engine.py` — imports the shared RAG from `backend/rag_engine.py`

## Folder map
- `routers/` — HTTP endpoints (what the frontend calls)
- `services/` — “business logic” that routers use
- `routers/feedback.py` — stores user feedback from the frontend

If you are new: start by reading `routers/nutrition.py` (the chat endpoint) and `services/rag_service.py`.
