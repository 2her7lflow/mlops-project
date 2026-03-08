# Setup Guide

This guide matches the current repository structure and the backend CI workflow.

## 1. Prerequisites

- Python 3.11 recommended for parity with GitHub Actions
- pip
- Docker Desktop or Docker Engine + Docker Compose plugin (optional)
- A Google Gemini API key if you want to enable RAG
- A Supabase Postgres connection string if you do not want to use local SQLite

## 2. Environment files

### Backend-only local run

```bash
cd backend
cp .env.example .env
```

Key variables in `backend/.env`:

```env
DATABASE_URL=sqlite:///./dev.db
CREATE_TABLES=true
GOOGLE_API_KEY=YOUR_KEY_HERE
DISABLE_RAG=false
KNOWLEDGE_BASE_DIR=../knowledge_base
MLFLOW_TRACKING_URI=http://localhost:5000
ENABLE_MLFLOW_CHAT_LOGGING=false
```

### Docker Compose run from repo root

```bash
cp .env.example .env
```

The root `.env` is used by `docker-compose.yml` and `docker-compose.prod.yml`.

## 3. Local backend setup

### Bash

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
pytest -q
uvicorn main:app --reload --port 8000
```

### PowerShell

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
pytest -q
uvicorn main:app --reload --reload-include .\.env --port 8000
```

Swagger UI will be available at `http://localhost:8000/docs`.

## 4. Local frontend setup

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export API_BASE_URL=http://localhost:8000
python app.py
```

On Windows PowerShell:

```powershell
$env:API_BASE_URL="http://localhost:8000"
python app.py
```

The frontend also accepts `BACKEND_URL`, but `API_BASE_URL` is the primary environment variable used in Docker Compose.

## 5. Docker Compose

From the repository root:

```bash
docker compose up --build
```

Services:
- backend: `http://localhost:8000`
- frontend: `http://localhost:7860`
- mlflow: `http://localhost:5000`

Notes:
- The backend mounts `./knowledge_base` into `/knowledge_base`.
- The backend and scheduler containers read environment values from the root `.env`.
- The scheduler is optional but enabled in the default compose file.

## 6. RAG setup

If you want the nutrition chat to use the knowledge base:

1. Put source documents under `knowledge_base/raw/`
2. Set `GOOGLE_API_KEY`
3. Start the backend
4. Build the index:

```bash
curl -X POST http://localhost:8000/admin/setup-rag
```

PowerShell:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/admin/setup-rag
```

For UI and database flow testing without the LLM stack, set:

```env
DISABLE_RAG=true
```

## 7. CI parity checks

These are the same checks used by GitHub Actions:

```bash
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
ruff check backend
pytest -q backend/tests
```

## 8. Chat logging features

The current backend stores:
- `chat_logs`
- `chat_feedback`
- mirrored compact vote rows in `feedback`

Useful routes:
- `GET /api/chat/logs`
- `GET /api/chat/summary`
- `POST /api/chat/feedback`

## 9. Common issues

### Tests fail because of missing DB config
Use SQLite for local testing:

```env
DATABASE_URL=sqlite:///./dev.db
CREATE_TABLES=true
```

### RAG returns fallback answers
Check:
- `GOOGLE_API_KEY`
- `KNOWLEDGE_BASE_DIR`
- whether `/admin/setup-rag` has been run
- whether `knowledge_base/processed/` was generated

### Docker backend cannot find the knowledge base
Use the root compose file and keep:

```env
KNOWLEDGE_BASE_DIR=/knowledge_base
```
