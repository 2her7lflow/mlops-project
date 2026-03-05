"""Convenience entrypoint.

So you can run evaluation with a single command:

  python -m evaluation

This calls the main RAG eval runner (evaluation/run_rag_eval.py).
"""

from __future__ import annotations

from .run_rag_eval import main


if __name__ == "__main__":
    main()
