# eval/ — Lightweight evaluation runners (quick checks)

This folder contains smaller scripts than `backend/evaluation/`.
Use it when you want a fast sanity check without the “full before/after” report.

## What’s inside
- `datasets/` — small JSON datasets (usually `.json`)
- `runners/` — python scripts you run from the repo root
- `configs/` — optional YAML config (future extension)

## Prerequisites
- You must have indexed the KB at least once:
  - `POST /admin/setup-rag`
  - `mlflow ui --port 5000`

## Run: RAG eval (hit@k + latency; optional RAGAS)
```powershell
python eval/runners/evaluate_rag.py --eval eval/datasets/rag_eval_set.json --k 4 --out eval/reports/rag_results.json
```

If you want it logged to MLflow, set:
```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME = "pet-nutrition-rag"
```

## Run: Prompt-only eval (no retrieval)
This tests prompt versions in isolation and should trigger guardrails because context is empty.
```powershell
$env:PROMPT_VERSION="v1"
python eval/runners/evaluate_prompt_only.py --eval eval/datasets/rag_eval_set.json

$env:PROMPT_VERSION="v2"
python eval/runners/evaluate_prompt_only.py --eval eval/datasets/rag_eval_set.json
```
