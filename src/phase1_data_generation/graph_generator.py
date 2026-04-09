"""
Generate ~20 IT process graph variants from base graph templates.
Varies Cost, HumanRes, Role assignments, and DeviceType while
preserving the graph topology (node types, edges, gateways).
"""
from __future__ import annotations

import copy
import random
from typing import Dict, List

from .graph_reader import ProcessGraph, NodeInfo, EdgeInfo

ROLES = [
    "PM", "Techlead", "Developer", "Senior Developer", "Tester",
    "QA Lead", "BA", "DevOps", "SRE", "DBA", "Kỹ sư Hệ thống",
    "CSKH", "Backend Developer", "Frontend Developer", "Trưởng ca IT",
    "Security Engineer",
]

DEVICE_TYPES = [
    "Laptop", "PC", "Server", "Monitor", "Smartphone", "Tablet",
    "Projector", "Phần mềm", "Hệ thống", "Cloud VM", "NAS",
]

DEVICE_NAMES_BY_TYPE = {
    "Laptop": ["Laptop-Dev", "Laptop-PM", "Laptop-QA", "MacBook-Pro", "ThinkPad"],
    "PC": ["PC-Dev-01", "PC-Dev-02", "PC-QA-01", "Workstation-01"],
    "Server": ["Prod-Server", "UAT-Server", "Staging-Server", "Dev-Server", "Log-Server-ELK"],
    "Monitor": ["Monitor-Screen-01", "Monitor-4K", "Dual-Monitor-Setup"],
    "Smartphone": ["Mobile-Testing", "iPhone-Test", "Android-Test"],
    "Tablet": ["iPad-Testing", "Tab-Demo"],
    "Projector": ["Projector-Room1", "Projector-Room2"],
    "Phần mềm": ["Alert-System", "Chat-Bot", "Slack-Bot", "Jira", "Confluence"],
    "Hệ thống": ["CI-CD-Pipeline", "Confluence-Wiki", "K8s-Cluster", "Docker-Swarm"],
    "Cloud VM": ["AWS-EC2-01", "GCP-VM-01", "Azure-VM-01"],
    "NAS": ["NAS-Backup-01", "NAS-Storage-01"],
}

HUMAN_NAMES = [
    "Nguyễn Văn A", "Nguyễn Văn B", "Nguyễn Văn C", "Nguyễn Văn D",
    "Nguyễn Văn E", "Nguyễn Văn F", "Nguyễn Văn G", "Nguyễn Văn H",
    "Nguyễn Văn I", "Nguyễn Văn K", "Nguyễn Văn L", "Nguyễn Văn M",
    "Trần Thị N", "Lê Văn O", "Phạm Thị P", "Hoàng Văn Q",
    "Đặng Văn R", "Vũ Thị S", "Bùi Văn T", "Đỗ Văn U",
]


class GraphGenerator:
    """Generate process graph variants from base templates."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_variants(
        self,
        base_graphs: Dict[str, ProcessGraph],
        num_variants: int = 20,
    ) -> Dict[str, ProcessGraph]:
        """
        Generate `num_variants` process graph variants by cycling through
        base templates and varying node attributes.
        """
        templates = list(base_graphs.values())
        variants: Dict[str, ProcessGraph] = {}
        next_graph_idx = max(
            (int(gid[1:]) for gid in base_graphs if gid[0] == "G"), default=0
        ) + 1

        for i in range(num_variants):
            template = templates[i % len(templates)]
            new_gid = f"G{next_graph_idx + i}"
            variant = self._create_variant(template, new_gid)
            variants[new_gid] = variant

        return variants

    def _create_variant(self, template: ProcessGraph, new_gid: str) -> ProcessGraph:
        variant = ProcessGraph(graph_id=new_gid)

        for nid, orig_node in template.nodes.items():
            new_node = NodeInfo(
                node_id=nid,
                label=orig_node.label,
                node_type=orig_node.node_type,
                graph=new_gid,
                cost=self._vary_cost(orig_node.cost),
                human_res=self._vary_human_res(orig_node.human_res),
                humans=self._vary_humans(orig_node.humans, orig_node.node_type),
                devices=self._vary_devices(orig_node.devices, orig_node.node_type),
            )
            variant.nodes[nid] = new_node

        for edge in template.edges:
            variant.edges.append(
                EdgeInfo(source=edge.source, target=edge.target, graph=new_gid)
            )

        return variant

    def _vary_cost(self, base_cost: float) -> float:
        if base_cost == 0:
            return 0.0
        factor = self.rng.uniform(0.6, 1.5)
        return round(base_cost * factor, 2)

    def _vary_human_res(self, base_hr: int) -> int:
        if base_hr == 0:
            return 0
        delta = self.rng.randint(-1, 2)
        return max(1, base_hr + delta)

    def _vary_humans(
        self, orig_humans: List[Dict[str, str]], node_type: str
    ) -> List[Dict[str, str]]:
        if node_type in ("StartEvent", "EndEvent", "ExclusiveGateway"):
            return []
        if not orig_humans:
            return []

        count = max(1, len(orig_humans) + self.rng.randint(-1, 1))
        new_humans = []
        used_ids = set()
        for _ in range(count):
            hid = f"H{self.rng.randint(1, 20)}"
            while hid in used_ids:
                hid = f"H{self.rng.randint(1, 20)}"
            used_ids.add(hid)
            new_humans.append(
                {
                    "human_id": hid,
                    "name": self.rng.choice(HUMAN_NAMES),
                    "role": self.rng.choice(ROLES),
                }
            )
        return new_humans

    def _vary_devices(
        self, orig_devices: List[Dict[str, str]], node_type: str
    ) -> List[Dict[str, str]]:
        if node_type in ("StartEvent", "EndEvent", "ExclusiveGateway"):
            return []
        if not orig_devices:
            if self.rng.random() < 0.3:
                return [self._random_device()]
            return []

        count = max(1, len(orig_devices) + self.rng.randint(-1, 1))
        return [self._random_device() for _ in range(count)]

    def _random_device(self) -> Dict[str, str]:
        dtype = self.rng.choice(DEVICE_TYPES)
        dname = self.rng.choice(DEVICE_NAMES_BY_TYPE.get(dtype, [dtype]))
        did = f"D{self.rng.randint(1, 50)}"
        return {"device_id": did, "device_name": dname, "device_type": dtype}
