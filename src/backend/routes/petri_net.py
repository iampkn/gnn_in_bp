"""API route: /petri-net - Return process template graph structure for visualization."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter()

LAYER_GAP_X = 220
LAYER_GAP_Y = 100


def _compute_layered_layout(
    nodes: Dict[str, dict],
    edges: List[Tuple[str, str]],
) -> Dict[str, Tuple[float, float]]:
    """BFS-layered layout starting from the StartEvent node."""
    adj: Dict[str, List[str]] = {nid: [] for nid in nodes}
    for src, tgt in edges:
        if src in adj:
            adj[src].append(tgt)

    start = next(
        (nid for nid, n in nodes.items() if n["type"] == "StartEvent"), None
    )
    if start is None:
        start = next(iter(nodes))

    layers: Dict[str, int] = {}
    queue = deque([start])
    layers[start] = 0

    while queue:
        nid = queue.popleft()
        for tgt in adj.get(nid, []):
            if tgt not in layers:
                layers[tgt] = layers[nid] + 1
                queue.append(tgt)

    for nid in nodes:
        if nid not in layers:
            layers[nid] = max(layers.values(), default=0) + 1

    layer_groups: Dict[int, List[str]] = {}
    for nid, layer in layers.items():
        layer_groups.setdefault(layer, []).append(nid)

    positions: Dict[str, Tuple[float, float]] = {}
    for layer_idx, members in sorted(layer_groups.items()):
        n = len(members)
        total_height = (n - 1) * LAYER_GAP_Y
        start_y = -total_height / 2
        for i, nid in enumerate(members):
            positions[nid] = (layer_idx * LAYER_GAP_X, start_y + i * LAYER_GAP_Y)

    return positions


@router.get("/petri-net")
async def get_petri_net(
    graph_id: str = Query(..., description="Graph ID (e.g. G1, G2)"),
):
    """Return process template graph structure with layout positions."""
    try:
        from src.phase1_data_generation.graph_reader import GraphReader
        from src.phase1_data_generation.graph_generator import GraphGenerator

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = project_root / "src" / "data" / "csv"

        if not data_dir.exists():
            raise HTTPException(status_code=404, detail="Data directory not found")

        reader = GraphReader(data_dir)
        base_graphs = reader.read_all()

        generator = GraphGenerator(seed=42)
        template_graphs = generator.generate_variants(base_graphs, num_variants=20)
        graphs = {**base_graphs, **template_graphs}

        if graph_id not in graphs:
            available = sorted(graphs.keys(), key=lambda g: int(g[1:]))
            raise HTTPException(
                status_code=404,
                detail=f"Graph '{graph_id}' not found. Available: {available}",
            )

        pg = graphs[graph_id]

        nodes_dict: Dict[str, dict] = {}
        for nid, node in pg.nodes.items():
            nodes_dict[nid] = {
                "id": nid,
                "label": node.label,
                "type": node.node_type,
                "cost": node.cost,
                "human_res": node.human_res,
            }

        edge_tuples = [(e.source, e.target) for e in pg.edges]
        positions = _compute_layered_layout(nodes_dict, edge_tuples)

        nodes_out = []
        for nid, info in nodes_dict.items():
            x, y = positions.get(nid, (0, 0))
            nodes_out.append({**info, "x": x, "y": y})

        edges_out = [
            {"source": src, "target": tgt, "frequency": 1}
            for src, tgt in edge_tuples
        ]

        return JSONResponse(
            content={
                "graph_id": graph_id,
                "total_cost": pg.total_cost(),
                "case_count": 1,
                "nodes": nodes_out,
                "edges": edges_out,
                "available_graphs": sorted(graphs.keys(), key=lambda g: int(g[1:])),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
