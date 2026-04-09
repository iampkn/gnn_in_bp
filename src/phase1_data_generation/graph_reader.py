"""
Read base graph structures from CSV files (node.csv, edge.csv, human.csv, device.csv)
and build directed graph objects per Graph ID.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class NodeInfo:
    node_id: str
    label: str
    node_type: str  # StartEvent | Task | ExclusiveGateway | EndEvent
    graph: str
    cost: float = 0.0
    human_res: int = 0
    humans: List[Dict[str, str]] = field(default_factory=list)
    devices: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class EdgeInfo:
    source: str
    target: str
    graph: str


@dataclass
class ProcessGraph:
    graph_id: str
    nodes: Dict[str, NodeInfo] = field(default_factory=dict)
    edges: List[EdgeInfo] = field(default_factory=list)

    @property
    def start_node(self) -> Optional[str]:
        for nid, node in self.nodes.items():
            if node.node_type == "StartEvent":
                return nid
        return None

    @property
    def end_node(self) -> Optional[str]:
        for nid, node in self.nodes.items():
            if node.node_type == "EndEvent":
                return nid
        return None

    def get_successors(self, node_id: str) -> List[str]:
        return [e.target for e in self.edges if e.source == node_id]

    def get_predecessors(self, node_id: str) -> List[str]:
        return [e.source for e in self.edges if e.target == node_id]

    def get_task_nodes(self) -> List[str]:
        return [nid for nid, n in self.nodes.items() if n.node_type == "Task"]

    def get_gateway_nodes(self) -> List[str]:
        return [
            nid
            for nid, n in self.nodes.items()
            if n.node_type == "ExclusiveGateway"
        ]

    def total_cost(self) -> float:
        return sum(n.cost for n in self.nodes.values())


class GraphReader:
    """Read CSV files and construct ProcessGraph objects."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def _read_csv(self, filename: str) -> List[dict]:
        filepath = self.data_dir / filename
        with open(filepath, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def read_all(self) -> Dict[str, ProcessGraph]:
        """Read all CSVs and return dict of graph_id -> ProcessGraph."""
        graphs: Dict[str, ProcessGraph] = {}

        for row in self._read_csv("node.csv"):
            gid = row["Graph"]
            if gid not in graphs:
                graphs[gid] = ProcessGraph(graph_id=gid)
            node = NodeInfo(
                node_id=row["NodeId"],
                label=row["NodeLabel"],
                node_type=row["NodeType"],
                graph=gid,
                cost=float(row["Cost"]),
                human_res=int(row["HumanRes"]),
            )
            graphs[gid].nodes[row["NodeId"]] = node

        for row in self._read_csv("edge.csv"):
            gid = row["Graph"]
            if gid in graphs:
                graphs[gid].edges.append(
                    EdgeInfo(source=row["Source"], target=row["Target"], graph=gid)
                )

        for row in self._read_csv("human.csv"):
            gid = row["Graph"]
            nid = row["NodeId"]
            if gid in graphs and nid in graphs[gid].nodes:
                graphs[gid].nodes[nid].humans.append(
                    {
                        "human_id": row["HumanId"],
                        "name": row["Name"],
                        "role": row["Role"],
                    }
                )

        for row in self._read_csv("device.csv"):
            gid = row["Graph"]
            nid = row["NodeId"]
            if gid in graphs and nid in graphs[gid].nodes:
                graphs[gid].nodes[nid].devices.append(
                    {
                        "device_id": row["DeviceId"],
                        "device_name": row["DeviceName"],
                        "device_type": row["DeviceType"],
                    }
                )

        return graphs

    def summary(self, graphs: Dict[str, ProcessGraph]) -> str:
        lines = []
        for gid, g in sorted(graphs.items()):
            lines.append(f"Graph {gid}: {len(g.nodes)} nodes, {len(g.edges)} edges")
            lines.append(f"  Start: {g.start_node}, End: {g.end_node}")
            lines.append(f"  Tasks: {g.get_task_nodes()}")
            lines.append(f"  Gateways: {g.get_gateway_nodes()}")
            lines.append(f"  Total Cost: {g.total_cost()}")
        return "\n".join(lines)
