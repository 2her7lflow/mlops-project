# Pet Nutrition AI MLOps Project

End-to-end pet nutrition assistant with:
- **FastAPI backend** for auth, pet profiles, activity logging, nutrition chat, feedback, and chat monitoring
- **Gradio frontend** for the user interface
- **RAG pipeline** over the local knowledge base
- **MLflow + optional scheduler** for experiment tracking and recurring rebuild/evaluation flows

## Repository layout

```text
.
├── backend/                # FastAPI app, tests, Dockerfile
├── frontend/               # Gradio app
├── knowledge_base/         # raw docs + generated processed artifacts
├── .github/workflows/      # GitHub Actions CI
├── docker-compose.yml      # local dev stack
├── docker-compose.prod.yml # production-oriented stack
├── dvc.yaml                # optional DVC pipeline
└── SETUP_GUIDE.md          # step-by-step setup instructions
```

## What the backend exposes

- `GET /health`
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `POST /api/pets`
- `POST /api/activity/logs`
- `POST /api/nutrition/chat`
- `POST /api/feedback`
- `GET /api/chat/logs`
- `GET /api/chat/summary`
- `POST /api/chat/feedback`

## CI

GitHub Actions runs the backend checks from `.github/workflows/backend-ci.yml`:

```bash
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt
ruff check backend
pytest -q backend/tests
```

## Fast start

For the full local setup, environment variables, and Docker instructions, see [SETUP_GUIDE.md](./SETUP_GUIDE.md).

Quick local backend test:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env       # Windows PowerShell: Copy-Item .env.example .env
pytest -q
uvicorn main:app --reload --port 8000
```
