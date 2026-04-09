"""API route: /search-vector - Query Vector DB for similar tasks."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/search-vector")
async def search_vector(
    activity: Optional[str] = Query(None, description="Activity name to search"),
    graph_id: Optional[str] = Query(None, description="Filter by graph ID"),
    min_cost: Optional[float] = Query(None, description="Minimum cost threshold"),
    top_k: int = Query(5, description="Number of results to return"),
):
    """Search Vector DB for similar tasks/activities."""
    try:
        import numpy as np
        import pandas as pd

        from src.phase2_vector_db import FeatureExtractor, VectorStore

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        csv_path = project_root / "src" / "data" / "generated" / "event_log_all.csv"

        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Event log not found. Run /generate-data first.",
            )

        df = pd.read_csv(csv_path)

        extractor = FeatureExtractor()
        extractor.fit(df)
        event_vectors = extractor.extract_event_vectors(df)

        store = VectorStore()
        vec_size = len(event_vectors[0]["vector"]) if event_vectors else 10
        store.create_collection("events", vec_size)

        points = [
            {
                "id": i,
                "vector": ev["vector"],
                "payload": {
                    "activity": ev["activity"],
                    "graph_id": ev["graph_id"],
                    **ev["metadata"],
                },
            }
            for i, ev in enumerate(event_vectors)
        ]
        store.upsert("events", points)

        if activity:
            matching = [p for p in points if p["payload"]["activity"] == activity]
            if matching:
                query_vec = matching[0]["vector"]
            else:
                query_vec = np.zeros(vec_size, dtype=np.float32)
        else:
            query_vec = np.ones(vec_size, dtype=np.float32) / vec_size

        results = store.search("events", query_vec, top_k=top_k)

        if min_cost is not None:
            results = [r for r in results if r["payload"].get("cost", 0) >= min_cost]
        if graph_id:
            results = [r for r in results if r["payload"].get("graph_id") == graph_id]

        return JSONResponse(
            content={
                "status": "success",
                "query": {"activity": activity, "graph_id": graph_id, "min_cost": min_cost},
                "results": results[:top_k],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
