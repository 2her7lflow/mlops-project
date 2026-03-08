# Setup Guide

This guide covers the fastest ways to run the project locally and the minimum checks needed before pushing to GitHub.

## 1. Requirements

Recommended:

- Python **3.11** for backend parity with GitHub Actions
- Docker Desktop or Docker Engine + Compose plugin
- a valid `GOOGLE_API_KEY` if you want live RAG chat/evaluation
- a SQLite database for local development or a Supabase Postgres connection string for deployment-like testing

## 2. Environment files

At repo root, copy the example file:

### PowerShell

```powershell
Copy-Item .env.example .env
```

### Bash

```bash
cp .env.example .env
```

Minimum values to review in `.env`:

```env
DATABASE_URL=sqlite:///./dev.db
CREATE_TABLES=true
GOOGLE_API_KEY=YOUR_KEY_HERE
KNOWLEDGE_BASE_DIR=../knowledge_base
HTTP_TIMEOUT_S=180
MLFLOW_TRACKING_URI=file:./mlruns
```

For Supabase, replace `DATABASE_URL` with the project Postgres connection string and keep SSL enabled.

## 3. Fastest way to run everything

From the repository root:

```bash
docker compose up --build
```

This starts:

- `mlflow`
- `backend`
- `frontend`
- `scheduler`

Open:

- Frontend: `http://localhost:7860`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

## 4. Run backend locally without Docker

### PowerShell

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
Copy-Item ..\.env.example .env
$env:DATABASE_URL = "sqlite:///./dev.db"
$env:CREATE_TABLES = "true"
$env:DISABLE_RAG = "true"
uvicorn main:app --reload --port 8000
```

### Bash

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp ../.env.example .env
export DATABASE_URL="sqlite:///./dev.db"
export CREATE_TABLES="true"
export DISABLE_RAG="true"
uvicorn main:app --reload --port 8000
```

Use `DISABLE_RAG=false` only after the RAG dependencies and key are configured.

The canonical example env file lives at the repository root as `.env.example`.

## 5. Run frontend locally without Docker

Start the backend first, then in a second terminal:

### PowerShell

```powershell
cd frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:BACKEND_URL = "http://localhost:8000"
python app.py
```

### Bash

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BACKEND_URL="http://localhost:8000"
python app.py
```

## 6. Frontend breed selector behavior

On the **Pet Profiles** page:

- selecting **Dog** shows only dog breeds
- selecting **Cat** shows only cat breeds
- editing an existing pet refreshes the breed dropdown to match that pet's species
- custom breed entry remains enabled

Breed options are sourced from:

- `knowledge_base/raw/breed_info/dog_breeds.csv`
- `knowledge_base/raw/breed_info/cat_breeds.csv`

## 7. Local pre-push checks

From the repository root, run the same backend checks as CI:

```bash
ruff check backend
pytest -q backend/tests
```

## 8. What GitHub Actions currently checks

The workflow file is `.github/workflows/backend-ci.yml`.

It currently:

1. installs backend runtime and dev dependencies
2. runs `ruff check backend`
3. runs `pytest -q backend/tests`
4. optionally runs RAG evaluation when `GOOGLE_API_KEY` is configured as a secret

Important:

- frontend code is not currently part of the GitHub Actions workflow
- a frontend regression can still slip through unless you test it locally or add a frontend job

## 9. Troubleshooting

### `ModuleNotFoundError` during backend startup

Make sure you installed both:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### RAG warmup fails on startup

This can be expected in local development if:

- `GOOGLE_API_KEY` is missing
- the vector store has not been built yet
- you intentionally set `DISABLE_RAG=true`

### Frontend cannot reach backend

Check:

- backend is running on port `8000`
- `BACKEND_URL` or `API_BASE_URL` points to the correct host
- Docker-local URLs are not mixed with host-local URLs

## 10. Suggested next CI improvement

Add a lightweight frontend validation job, for example:

- `python -m py_compile frontend/app.py`
- or a small smoke test around the breed dropdown logic
