"""
Phase 5: FastAPI Backend
Provides REST APIs for the process discovery system:
  - /generate-data: Run Phase 1 data generation
  - /search-vector: Query Vector DB for similar tasks
  - /discover-process: Upload event log -> GNN inference -> Petri net
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import generate_data, search_vector, discover_process

app = FastAPI(
    title="GNN Process Discovery API",
    description="Automated Process Discovery using Graph Neural Networks",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_data.router, prefix="/api", tags=["Data Generation"])
app.include_router(search_vector.router, prefix="/api", tags=["Vector Search"])
app.include_router(discover_process.router, prefix="/api", tags=["Process Discovery"])


@app.get("/")
async def root():
    return {"message": "GNN Process Discovery API", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}
