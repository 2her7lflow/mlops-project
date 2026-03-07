# backend/app/services/ — Business logic

Routers should be “thin”. The real logic lives here.

## Files
- `rag_service.py` — loads/warms RAG and provides helper functions
- `nutrition_service.py` — combines RAG answer + nutrition calculator output
- `mlflow_tracker.py` — small wrapper around MLflow logging (params/metrics/artifacts)
- `types.py` — shared type hints / small data structures

If something feels “magical”, check `rag_service.py` first — it controls how the RAG engine is used.
