"""
Inference engine using Beam Search.
Per the paper Section IV-B:
  - Beam search with width b to find highest joint probability Petri net
  - At each step, expand b candidates, keep top b by joint probability
  - Return top b Petri nets ranked by joint probability
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import numpy as np

from .discovery_model import ProcessDiscoveryModel
from .graph_encoder import DiscoveryGraphEncoder


@dataclass
class BeamState:
    """State of one beam during search."""
    selected_indices: List[int] = field(default_factory=list)
    selected_mask: Optional[torch.Tensor] = None
    log_prob: float = 0.0
    embeddings: Optional[torch.Tensor] = None
    is_finished: bool = False


class BeamSearchInference:
    """
    Perform inference using beam search to discover the most likely Petri net.
    """

    def __init__(
        self,
        model: ProcessDiscoveryModel,
        beam_width: int = 10,
        max_places: int = 50,
        stop_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.beam_width = beam_width
        self.max_places = max_places
        self.stop_threshold = stop_threshold
        self.device = device

    @torch.no_grad()
    def discover(self, encoded_data: dict) -> List[dict]:
        """
        Run beam search inference on an encoded event log.

        Args:
            encoded_data: dict from DiscoveryGraphEncoder.encode()

        Returns:
            List of discovered Petri net candidates, sorted by probability.
            Each contains:
              - 'places': list of (in_set, out_set) tuples
              - 'transitions': list of activity labels
              - 'log_prob': joint log probability
              - 'selected_indices': list of selected place indices
        """
        g = encoded_data["dgl_graph"]
        candidate_places = encoded_data["candidate_places"]
        transitions = encoded_data["transition_nodes"]
        num_places = encoded_data["num_places"]

        all_h = self._get_unified_features(encoded_data).to(self.device)
        adj, direction = self._get_unified_adj(encoded_data)
        adj = adj.to(self.device)
        direction = direction.to(self.device)

        place_indices = self._get_place_indices(encoded_data).to(self.device)
        petri_indices = self._get_petri_indices(encoded_data).to(self.device)

        h = self.model.pn1(all_h, adj, direction)

        initial_state = BeamState(
            selected_indices=[],
            selected_mask=torch.zeros(num_places, dtype=torch.bool, device=self.device),
            log_prob=0.0,
            embeddings=h.clone(),
            is_finished=False,
        )
        beams = [initial_state]

        for step in range(self.max_places):
            all_candidates = []

            for beam in beams:
                if beam.is_finished:
                    all_candidates.append(beam)
                    continue

                place_h = beam.embeddings[place_indices]
                probs, _ = self.model.scn(place_h, beam.selected_mask)

                top_k = min(self.beam_width, (probs > 0).sum().item())
                if top_k == 0:
                    beam.is_finished = True
                    all_candidates.append(beam)
                    continue

                top_probs, top_indices = torch.topk(probs, k=max(1, int(top_k)))

                for prob, idx in zip(top_probs, top_indices):
                    idx_val = idx.item()
                    if prob.item() <= 0:
                        continue

                    new_mask = beam.selected_mask.clone()
                    new_mask[idx_val] = True
                    new_selected = beam.selected_indices + [idx_val]
                    new_log_prob = beam.log_prob + torch.log(prob + 1e-10).item()

                    petri_h = beam.embeddings[petri_indices]
                    is_wf = self.model.s_checker.is_workflow_net(
                        [candidate_places[i] for i in new_selected],
                        transitions,
                    )
                    should_stop, _ = self.model.sn.should_stop(
                        petri_h, self.stop_threshold, is_wf
                    )

                    if should_stop:
                        new_state = BeamState(
                            selected_indices=new_selected,
                            selected_mask=new_mask,
                            log_prob=new_log_prob,
                            embeddings=beam.embeddings,
                            is_finished=True,
                        )
                    else:
                        selected_feature = torch.zeros(
                            beam.embeddings.size(0), 1, device=self.device
                        )
                        for si in range(len(place_indices)):
                            if new_mask[si]:
                                selected_feature[place_indices[si]] = 1.0

                        h_ext = torch.cat([beam.embeddings, selected_feature], dim=-1)
                        new_h = self.model.pn2(h_ext, adj, direction)

                        new_state = BeamState(
                            selected_indices=new_selected,
                            selected_mask=new_mask,
                            log_prob=new_log_prob,
                            embeddings=new_h,
                            is_finished=False,
                        )

                    all_candidates.append(new_state)

            all_candidates.sort(key=lambda s: s.log_prob, reverse=True)
            beams = all_candidates[: self.beam_width]

            if all(b.is_finished for b in beams):
                break

        # Reduce beam width as we progress (per paper)
        results = []
        for beam in sorted(beams, key=lambda b: b.log_prob, reverse=True):
            places = [candidate_places[i] for i in beam.selected_indices]
            results.append(
                {
                    "places": places,
                    "transitions": transitions,
                    "log_prob": beam.log_prob,
                    "selected_indices": beam.selected_indices,
                    "num_places_selected": len(beam.selected_indices),
                }
            )

        return results

    def _get_unified_features(self, data: dict) -> torch.Tensor:
        g = data["dgl_graph"]
        features = []
        for ntype in g.ntypes:
            if "h" in g.nodes[ntype].data:
                features.append(g.nodes[ntype].data["h"])
        if features:
            return torch.cat(features, dim=0)
        return torch.zeros(1, data["feature_dim"])

    def _get_unified_adj(self, data: dict):
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
            all_src.append(src + offset[src_type])
            all_dst.append(dst + offset[dst_type])
            n = len(src)
            all_dir.append(torch.tensor([[1.0, 0.0]] * n))
            all_src.append(dst + offset[dst_type])
            all_dst.append(src + offset[src_type])
            all_dir.append(torch.tensor([[0.0, 1.0]] * n))

        if all_src:
            adj = torch.stack([torch.cat(all_src), torch.cat(all_dst)])
            direction = torch.cat(all_dir)
        else:
            adj = torch.zeros(2, 0, dtype=torch.long)
            direction = torch.zeros(0, 2)
        return adj, direction

    def _get_place_indices(self, data: dict) -> torch.Tensor:
        g = data["dgl_graph"]
        offset = 0
        indices = []
        for ntype in g.ntypes:
            n = g.num_nodes(ntype)
            if ntype == "place":
                indices.extend(range(offset, offset + n))
            offset += n
        return torch.tensor(indices, dtype=torch.long) if indices else torch.tensor([0], dtype=torch.long)

    def _get_petri_indices(self, data: dict) -> torch.Tensor:
        g = data["dgl_graph"]
        offset = 0
        indices = []
        for ntype in g.ntypes:
            n = g.num_nodes(ntype)
            if ntype in ("transition", "place"):
                indices.extend(range(offset, offset + n))
            offset += n
        return torch.tensor(indices, dtype=torch.long) if indices else torch.tensor([0], dtype=torch.long)
