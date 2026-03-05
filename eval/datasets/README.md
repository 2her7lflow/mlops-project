# eval/datasets/ — Evaluation datasets

## rag_eval_set.json
This is a list of test cases. Each item can include:
- `id` (string)
- `question` (string)
- `expected_source_contains` (list[str], optional)
- `expected_keywords` (list[str], optional)

Example:
```json
{
  "id": "tox01",
  "question": "สุนัขกินช็อกโกแลตได้ไหม?",
  "expected_source_contains": ["toxic_foods"],
  "expected_keywords": ["ช็อกโกแลต", "พิษ"]
}
```

These fields are used only for the simple hit@k scoring.
