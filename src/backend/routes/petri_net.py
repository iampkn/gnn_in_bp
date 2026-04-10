"""API route: /petri-net - Mine a Directly-Follows Graph from event logs."""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter()

LAYER_GAP_X = 220
LAYER_GAP_Y = 100


def _mine_dfg(df: pd.DataFrame) -> Tuple[
    Dict[str, dict],
    List[dict],
    float,
]:
    """Mine a Directly-Follows Graph from an event log DataFrame.

    Returns (nodes_dict, edges_list, total_cost).
    """
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    activity_stats: Dict[str, dict] = defaultdict(
        lambda: {"count": 0, "total_cost": 0.0, "node_type": "Task"}
    )
    dfg_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for _case_id, grp in df.groupby("case:concept:name"):
        acts = list(grp["concept:name"])
        costs = list(grp["cost"])
        node_types = list(grp["node_type"])

        for act, cost, ntype in zip(acts, costs, node_types):
            s = activity_stats[act]
            s["count"] += 1
            s["total_cost"] += cost
            s["node_type"] = ntype

        for a, b in zip(acts, acts[1:]):
            dfg_counts[(a, b)] += 1

    nodes: Dict[str, dict] = {}
    for i, (act, stats) in enumerate(activity_stats.items()):
        nodes[act] = {
            "id": act,
            "label": act,
            "type": stats["node_type"],
            "cost": round(stats["total_cost"] / max(stats["count"], 1), 2),
            "human_res": stats["count"],
            "occurrence": stats["count"],
        }

    edges = []
    for (src, tgt), freq in dfg_counts.items():
        edges.append({"source": src, "target": tgt, "frequency": freq})

    total_cost = sum(s["total_cost"] for s in activity_stats.values())
    return nodes, edges, round(total_cost, 2)


def _compute_layered_layout(
    nodes: Dict[str, dict],
    edges: List[dict],
) -> Dict[str, Tuple[float, float]]:
    """BFS-layered layout from the StartEvent node."""
    adj: Dict[str, List[str]] = {nid: [] for nid in nodes}
    for e in edges:
        if e["source"] in adj:
            adj[e["source"]].append(e["target"])

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
    """Mine a Directly-Follows Graph from event logs for the given graph."""
    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        csv_path = project_root / "src" / "data" / "generated" / "event_log_all.csv"

        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Event log not found. Run /api/generate-data first.",
            )

        full_df = pd.read_csv(csv_path)
        available = sorted(
            full_df["graph_id"].unique().tolist(),
            key=lambda g: int(g[1:]),
        )

        if graph_id not in available:
            raise HTTPException(
                status_code=404,
                detail=f"Graph '{graph_id}' not found. Available: {available}",
            )

        gdf = full_df[full_df["graph_id"] == graph_id]
        nodes_dict, edges_list, total_cost = _mine_dfg(gdf)
        positions = _compute_layered_layout(nodes_dict, edges_list)

        nodes_out = []
        for nid, info in nodes_dict.items():
            x, y = positions.get(nid, (0, 0))
            nodes_out.append({**info, "x": x, "y": y})

        case_count = int(gdf["case:concept:name"].nunique())

        return JSONResponse(
            content={
                "graph_id": graph_id,
                "total_cost": total_cost,
                "case_count": case_count,
                "nodes": nodes_out,
                "edges": edges_list,
                "available_graphs": available,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
