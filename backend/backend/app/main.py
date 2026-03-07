from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .metrics import record
from .routers import pets, nutrition, activity, admin, system, auth
from .services.rag_service import warmup_rag


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pet-nutrition-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init DB (demo) + warm RAG (best effort)
    init_db()
    warmup_rag()
    yield


app = FastAPI(title="Pet Nutrition AI API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000.0

    # group by route template if available
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    record(path=path, method=request.method, latency_ms=latency_ms)

    response.headers["X-Response-Time-ms"] = f"{latency_ms:.2f}"
    return response


# Routers
app.include_router(system.router)
app.include_router(pets.router)
app.include_router(nutrition.router)
app.include_router(activity.router)
app.include_router(auth.router)
app.include_router(admin.router)
