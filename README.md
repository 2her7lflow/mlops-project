# Pet Nutrition AI

Pet Nutrition AI is a full-stack pet care project with:

- a **Gradio frontend** for pet profile management, nutrition Q&A, activity logging, and feedback
- a **FastAPI backend** with session-based auth, pet CRUD, activity-based meal adjustments, and feedback APIs
- a **RAG pipeline** over the local knowledge base for nutrition guidance
- optional **MLflow**, **DVC**, and scheduler support for MLOps workflows

## Repository layout

```text
.
├── backend/                # FastAPI app, tests, scheduler, scripts
├── frontend/               # Gradio app
├── knowledge_base/         # raw and processed knowledge sources
├── .github/workflows/      # GitHub Actions CI
├── docker-compose.yml      # local development stack
├── docker-compose.prod.yml # production-style compose stack
└── Dockerfile.mlflow       # MLflow server image
```

## Main features

- user signup, login, logout, and session authentication
- pet profile CRUD
- species-specific breed selection in the frontend (`dog` breeds vs `cat` breeds)
- daily activity logging and nutrition adjustment
- chat-style nutrition assistant backed by the project knowledge base
- feedback collection for bugs, UX issues, ideas, and answer quality

## Quick start

### Docker Compose

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:7860`
- Backend API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

### Local development

See [SETUP.md](SETUP.md) for full local setup instructions.

## CI status

The current GitHub workflow is in `.github/workflows/backend-ci.yml` and checks the backend only:

```bash
ruff check backend
pytest -q backend/tests
```

Notes:

- the workflow uses **Python 3.11**
- the frontend is **not** currently covered by GitHub Actions
- RAG evaluation runs only when `GOOGLE_API_KEY` exists in repository secrets

## Key files to read first

- `SETUP.md`
- `backend/README.md`
- `frontend/README.md`
- `.github/workflows/backend-ci.yml`

## Current behavior worth noting

- the frontend breed dropdown now switches between **dog breeds** and **cat breeds** based on the selected species
- breed options are loaded from `knowledge_base/raw/breed_info/*.csv` with built-in fallback lists
- custom breed values are still allowed from the frontend
