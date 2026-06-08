"""
api/main.py
QSE Market Copilot — FastAPI application entry point.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints import anomaly, features, feedback, health, models_status, query, regime, similarity

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

app = FastAPI(
    title="QSE Market Copilot",
    description="Natural-language analytics copilot for the Qatar Stock Exchange.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: allow the Vite dev server and same-origin production builds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(query.router,         tags=["query"])
app.include_router(regime.router,        tags=["regime"])
app.include_router(features.router,      tags=["features"])
app.include_router(similarity.router,    tags=["similarity"])
app.include_router(anomaly.router,       tags=["anomaly"])
app.include_router(feedback.router,      tags=["feedback"])
app.include_router(models_status.router, tags=["models"])
app.include_router(health.router,        tags=["health"])
