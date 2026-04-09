"""API route: /discover-process - Upload event log -> GNN inference -> Petri net."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/discover-process")
async def discover_process(
    file: Optional[UploadFile] = File(None),
    graph_id: Optional[str] = Query(None, description="Graph ID to discover from generated data"),
    beam_width: int = Query(10, description="Beam search width"),
):
    """
    Upload an event log CSV or specify a graph_id to run GNN inference.
    Returns the discovered Petri net structure and cost predictions.
    """
    try:
        import pandas as pd
        import torch

        from src.phase3_gnn import (
            DiscoveryGraphEncoder,
            ProcessDiscoveryModel,
            BeamSearchInference,
        )
        from src.phase4_regression import RegressionHead

        project_root = Path(__file__).resolve().parent.parent.parent.parent

        if file is not None:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
        elif graph_id:
            csv_path = project_root / "src" / "data" / "generated" / "event_log_all.csv"
            if not csv_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail="Event log not found. Run /generate-data first.",
                )
            df = pd.read_csv(csv_path)
            df = df[df["graph_id"] == graph_id]
            if df.empty:
                raise HTTPException(
                    status_code=404, detail=f"No data for graph_id={graph_id}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a CSV file or a graph_id parameter.",
            )

        encoder = DiscoveryGraphEncoder(k_max=3, max_place_size=2)
        encoded = encoder.encode(df, graph_id=graph_id)

        feature_dim = encoded["feature_dim"]
        model = ProcessDiscoveryModel(feature_dim=feature_dim, hidden_dim=16)

        model_path = project_root / "src" / "data" / "generated" / "discovery_model.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))

        inference_engine = BeamSearchInference(
            model=model, beam_width=beam_width, device="cpu"
        )
        results = inference_engine.discover(encoded)

        response = {
            "status": "success",
            "input_info": {
                "graph_id": graph_id,
                "num_traces": len(encoded["traces"]),
                "num_activities": encoded["num_transitions"],
                "num_candidate_places": encoded["num_places"],
                "activities": encoded["transition_nodes"],
            },
            "discovered_nets": [],
        }

        for i, result in enumerate(results[:5]):
            places_json = []
            for in_set, out_set in result["places"]:
                places_json.append(
                    {"inputs": sorted(list(in_set)), "outputs": sorted(list(out_set))}
                )

            net_info = {
                "rank": i + 1,
                "log_probability": result["log_prob"],
                "num_places": result["num_places_selected"],
                "transitions": result["transitions"],
                "places": places_json,
            }
            response["discovered_nets"].append(net_info)

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
