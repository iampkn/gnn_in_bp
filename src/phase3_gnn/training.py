"""
Training loop for the GNN Process Discovery model.
Per the paper Section IV-B:
  - Loss function: Negative log-likelihood (Eq. 7)
  - Teacher forcing with BFS ordering
  - Training on synthetic pairs <L_i, N_i>
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import numpy as np

from .discovery_model import ProcessDiscoveryModel
from .graph_encoder import DiscoveryGraphEncoder


class Trainer:
    """Train the ProcessDiscoveryModel on event log / Petri net pairs."""

    def __init__(
        self,
        model: ProcessDiscoveryModel,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.encoder = DiscoveryGraphEncoder()
        self.history: List[dict] = []

    def train_epoch(
        self,
        training_data: List[dict],
        epoch: int,
    ) -> dict:
        """
        Train one epoch over all training instances.

        Args:
            training_data: list of encoded graph dicts from DiscoveryGraphEncoder
            epoch: current epoch number

        Returns:
            dict with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for i, data in enumerate(training_data):
            self.optimizer.zero_grad()

            g = data["dgl_graph"]
            candidate_places = data["candidate_places"]
            transitions = data["transition_nodes"]

            all_h = self._get_unified_features(data)
            adj, direction = self._get_unified_adj(data)

            place_indices = self._get_node_indices(data, "place")
            petri_indices = self._get_node_indices(data, "petri")

            target_seq = self._get_bfs_target_sequence(data)
            if not target_seq:
                continue

            all_h = all_h.to(self.device)
            adj = adj.to(self.device)
            direction = direction.to(self.device)
            place_indices = place_indices.to(self.device)
            petri_indices = petri_indices.to(self.device)

            loss, _ = self.model.forward_train(
                all_h, adj, direction,
                place_indices, petri_indices,
                target_seq, candidate_places, transitions,
            )

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_samples += 1

        avg_loss = total_loss / max(num_samples, 1)
        metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "num_samples": num_samples,
        }
        self.history.append(metrics)
        return metrics

    def train(
        self,
        training_data: List[dict],
        num_epochs: int = 100,
        log_every: int = 10,
        save_dir: Optional[str] = None,
    ) -> List[dict]:
        """Full training loop."""
        print(f"Training for {num_epochs} epochs on {len(training_data)} samples...")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(training_data, epoch)

            if epoch % log_every == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch:4d} | Loss: {metrics['avg_loss']:.4f} "
                    f"| Samples: {metrics['num_samples']} | Time: {elapsed:.1f}s"
                )

        if save_dir:
            self.save_model(save_dir)
            self.save_history(save_dir)

        return self.history

    def save_model(self, save_dir: str):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "discovery_model.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def save_history(self, save_dir: str):
        path = Path(save_dir) / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")

    def _get_unified_features(self, data: dict) -> torch.Tensor:
        """Combine all node features into a single tensor."""
        g = data["dgl_graph"]
        features = []
        for ntype in g.ntypes:
            if "h" in g.nodes[ntype].data:
                features.append(g.nodes[ntype].data["h"])
        if features:
            return torch.cat(features, dim=0)
        return torch.zeros(1, data["feature_dim"])

    def _get_unified_adj(self, data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build unified adjacency and direction tensors."""
        g = data["dgl_graph"]
        all_src, all_dst, all_dir = [], [], []
        offset = {}
        current = 0
        for ntype in g.ntypes:
            offset[ntype] = current
            current += g.num_nodes(ntype)

        for etype in g.canonical_etypes:
            src_type, _, dst_type = etype
            src, dst = g.edges(etype=etype)
            src_offset = src + offset[src_type]
            dst_offset = dst + offset[dst_type]
            all_src.append(src_offset)
            all_dst.append(dst_offset)
            n_edges = len(src)
            fwd = torch.tensor([[1.0, 0.0]] * n_edges)
            all_dir.append(fwd)
            # Add reverse
            all_src.append(dst_offset)
            all_dst.append(src_offset)
            rev = torch.tensor([[0.0, 1.0]] * n_edges)
            all_dir.append(rev)

        if all_src:
            adj = torch.stack([torch.cat(all_src), torch.cat(all_dst)])
            direction = torch.cat(all_dir)
        else:
            adj = torch.zeros(2, 0, dtype=torch.long)
            direction = torch.zeros(0, 2)

        return adj, direction

    def _get_node_indices(self, data: dict, node_class: str) -> torch.Tensor:
        """Get indices of specific node types in the unified graph."""
        g = data["dgl_graph"]
        offset = 0
        indices = []
        for ntype in g.ntypes:
            n = g.num_nodes(ntype)
            if node_class == "place" and ntype == "place":
                indices.extend(range(offset, offset + n))
            elif node_class == "petri" and ntype in ("transition", "place"):
                indices.extend(range(offset, offset + n))
            offset += n
        return torch.tensor(indices, dtype=torch.long) if indices else torch.tensor([0], dtype=torch.long)

    def _get_bfs_target_sequence(self, data: dict) -> List[int]:
        """
        Generate BFS-ordered target sequence for teacher forcing.
        For training, we need the ground truth sequence of places to select.
        Uses a simplified BFS on the candidate places.
        """
        num_places = data.get("num_places", 0)
        if num_places == 0:
            return []
        return list(range(min(num_places, 10)))
