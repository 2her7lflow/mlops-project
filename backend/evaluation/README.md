# backend/evaluation/ — RAG evaluation + MLflow (single retrieval strategy)

This folder is the **main evaluation pipeline** meant to show “engineering evidence”:
- You have a **test set**
- You run **repeatable experiments**
- You log everything to **MLflow**
- You evaluate the **same retrieval strategy as the API**
- You report **RAGAS** scores (when Gemini key is set)

## What this evaluation evaluates
One mode (no comparisons):
- **Page/row-first narrowing (PageIndex-inspired)** + **Hybrid fusion** (dense chunks + BM25)

## Files
- `run_rag_eval.py` — the runner (logs to MLflow)
- `testset.json` / `testset.jsonl` — your evaluation questions

## 0) Prerequisites
1) Backend dependencies installed (`backend/requirements.txt`)
2) KB already indexed (run once):
   - `POST /admin/setup-rag`

PowerShell:
```powershell
Invoke-RestMethod -Method Post http://localhost:8000/admin/setup-rag
```

## 1) Start MLflow UI
```powershell
cd backend
mlflow ui --host 0.0.0.0 --port 5000
```
Open: `http://localhost:5000`

Make sure env vars are set (optional but recommended):
- `MLFLOW_TRACKING_URI=http://localhost:5000`
- `MLFLOW_EXPERIMENT_NAME=pet-nutrition-rag`

## 2) Edit the test set

Recommended: `testset.json` (easier to edit, no JSONL escaping headaches).

Example:
```json
[
  {"id":"tf01","mode":"consult","question":"สุนัขกินช็อกโกแลตได้ไหม?","ground_truth":""},
  {"id":"pf01","mode":"profile","question":"Ben อายุเท่าไหร่?","pet_context":{"name":"Ben","age_years":1},"expected_answer_contains":["1","ปี"]}
]
```

Notes:
- `ground_truth` is optional.
- If you later fill `ground_truth`, you can enable **context_recall**.

## 3) Run the evaluation

Simplest:
```powershell
cd backend
python -m evaluation
```

Optional flags:
```powershell
python -m evaluation --max_n 12
python -m evaluation --testset .\evaluation\testset.json --report .\evaluation\reports\my_report.json
```

### What gets logged
- Params: prompt version, k values, retrieval strategy, min relevance threshold
- Retrieval stats: avg relevance, %no-docs, %weak-retrieval
- RAGAS (if `GOOGLE_API_KEY` exists): faithfulness, answer_relevancy, context_precision  
  Optional: context_recall (needs ground_truth)

## Interpreting results (simple)
- If Hybrid improves **hit/precision** without hurting faithfulness → retrieval got better.
- If faithfulness drops → model is hallucinating more (often caused by weak/irrelevant context).
- If many “weak retrieval” cases happen → raise `RAG_MIN_RELEVANCE` or improve KB/chunking.
