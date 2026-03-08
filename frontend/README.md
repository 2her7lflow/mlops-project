# frontend/ — Gradio UI

This directory contains the Gradio frontend for the Pet Nutrition AI project.

## What the frontend does

- login and signup
- pet profile CRUD
- species-specific breed dropdowns for dogs and cats
- activity logging
- nutrition plan generation
- chat-based nutrition assistant
- feedback submission

## Run with Docker Compose

From the repository root:

```bash
docker compose up --build
```

Open the UI at `http://localhost:7860`.

## Run locally

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

## Breed dropdown behavior

The Pet Profiles page uses separate breed lists:

- `Dog` -> dog breeds
- `Cat` -> cat breeds

Breed data is loaded from the CSV files in `knowledge_base/raw/breed_info/` and falls back to a small built-in list if the CSV files are unavailable.
