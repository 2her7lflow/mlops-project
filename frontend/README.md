# Gradio Frontend

This replaces the previous React/Vite UI.

## Run with Docker Compose (recommended)

From repo root:

```bash
docker compose up --build
```

Open: http://localhost:7860

## Local run (no docker)

```bash
cd frontend
python -m venv .venv
# activate venv...
pip install -r requirements.txt
set BACKEND_URL=http://localhost:8000   # PowerShell: $env:BACKEND_URL="http://localhost:8000"
python app.py
```
