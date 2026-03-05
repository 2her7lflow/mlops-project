# backend/tests/ — Automated tests

These are small smoke tests to catch obvious breakages.

## Run
```powershell
cd backend
pytest -q
```

If tests fail due to missing DB/keys, set environment variables first (especially `DATABASE_URL`).
