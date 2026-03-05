# knowledge_base/ — Your documents & tables for RAG

This is the **data** that the RAG system searches.

Important concept:
- Raw files live under `knowledge_base/knowledge_base/...`
- When you run `/admin/setup-rag`, the system generates a `processed/` folder
  that contains the vector DB and chunk cache.

## Structure
- `knowledge_base/knowledge_base/breed_info/` — CSVs (dog/cat breed info)
- `knowledge_base/knowledge_base/nutrition_guidelines/` — PDFs (WSAVA/FEDIAF)
- `knowledge_base/knowledge_base/toxic_foods/` — CSVs (toxic foods list)

## Generated outputs (after indexing)
- `knowledge_base/**/processed/vectorstore/` — Chroma database files
- `knowledge_base/**/processed/chunks.jsonl` — chunk text used for hybrid/BM25

## Index (build) command
With backend running:
```powershell
Invoke-RestMethod -Method Post http://localhost:8000/admin/setup-rag
```

## Tip: using an external KB folder
Set env var:
- `KNOWLEDGE_BASE_DIR=C:\path\to\your\knowledge_base`
