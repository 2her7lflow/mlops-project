# eval/runners/ — Scripts you execute

- `evaluate_rag.py`  
  Runs retrieval+generation using the shared RAG engine, then scores:
  - hit@k (simple heuristics using expected keywords/sources)
  - latency (avg + p95)
  - optional RAGAS (if installed + `GOOGLE_API_KEY`)

- `evaluate_prompt_only.py`  
  Runs **no retrieval**. Sends empty context to test:
  - guardrail behavior (refuse to guess / ask follow-up)
  - prompt differences (v1 vs v2)
