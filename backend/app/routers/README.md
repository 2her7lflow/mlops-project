# backend/app/routers/ — API endpoints

Each file here defines a group of HTTP routes.

## What each router does
- `system.py` — health/status endpoints
- `pets.py` — pet CRUD endpoints
- `nutrition.py` — nutrition chat endpoint (calls RAG + nutrition calculator)
- `activity.py` — activity logging endpoints (if used)
- `admin.py` — admin actions (most important: `/admin/setup-rag` builds the KB index)

## Most important endpoint
### POST `/admin/setup-rag`
Indexes the knowledge base and writes:
- `processed/vectorstore/` (Chroma)
- `processed/chunks.jsonl` (used for hybrid/BM25)

You should run this once after you change KB files.
