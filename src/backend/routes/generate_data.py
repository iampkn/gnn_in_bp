"""API route: /generate-data - Run Phase 1 data generation pipeline."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/generate-data")
async def generate_data(
    num_variants: int = 20,
    traces_per_graph: int = 10,
    seed: int = 42,
):
    """Run Phase 1: generate graph variants and simulate event logs."""
    try:
        from src.phase1_data_generation import (
            GraphReader,
            GraphGenerator,
            EventLogSimulator,
        )

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        csv_dir = project_root / "src" / "data" / "csv"
        output_dir = project_root / "src" / "data" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)

        reader = GraphReader(csv_dir)
        base_graphs = reader.read_all()

        generator = GraphGenerator(seed=seed)
        variant_graphs = generator.generate_variants(base_graphs, num_variants=num_variants)
        all_graphs = {**base_graphs, **variant_graphs}

        simulator = EventLogSimulator(seed=seed)
        event_log_df = simulator.simulate(all_graphs, traces_per_graph=traces_per_graph)

        csv_output = output_dir / "event_log_all.csv"
        simulator.save_csv(event_log_df, str(csv_output))

        summary = {
            "status": "success",
            "base_graphs": list(base_graphs.keys()),
            "variant_graphs": list(variant_graphs.keys()),
            "total_graphs": len(all_graphs),
            "total_events": len(event_log_df),
            "total_cases": int(event_log_df["case:concept:name"].nunique()),
            "unique_activities": int(event_log_df["concept:name"].nunique()),
            "output_file": str(csv_output),
        }

        per_graph = []
        for gid in sorted(all_graphs.keys()):
            gid_df = event_log_df[event_log_df["graph_id"] == gid]
            if not gid_df.empty:
                per_graph.append(
                    {
                        "graph_id": gid,
                        "cases": int(gid_df["case:concept:name"].nunique()),
                        "events": len(gid_df),
                        "total_cost": float(gid_df["cost"].sum()),
                    }
                )
        summary["per_graph"] = per_graph

        return JSONResponse(content=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
