"""
Full GNN-based Process Discovery Model.
Combines PN1, SCN, SN, PN2 into the iterative discovery algorithm
described in the paper (Section IV, Figure 3).

The generation loop:
  1. PN1: Propagate behavioral info from trace graph to places
  2. SCN: Select the most probable candidate place
  3. S-coverability check: reject if unsound
  4. SN: Decide whether to stop
  5. PN2: Propagate the decision info through the graph
  6. Repeat from step 2
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple

from .propagation_net import PropagationNetwork, build_pn1, build_pn2
from .select_candidate import SelectCandidateNetwork
from .stop_network import StopNetwork
from .s_coverability import SCoverabilityChecker


class ProcessDiscoveryModel(nn.Module):
    """
    The complete d = (f, PN1, SCN, SN, PN2) model.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 16,
        num_heads: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.pn1 = build_pn1(feature_dim, output_dim=hidden_dim, num_heads=num_heads)
        self.scn = SelectCandidateNetwork(hidden_dim)
        self.sn = StopNetwork(hidden_dim)
        self.pn2 = build_pn2(hidden_dim + 1, output_dim=hidden_dim, num_heads=num_heads)

        self.s_checker = SCoverabilityChecker()

    def forward_train(
        self,
        all_embeddings: torch.Tensor,
        adj: torch.Tensor,
        direction: torch.Tensor,
        place_indices: torch.Tensor,
        petri_indices: torch.Tensor,
        target_sequence: List[int],
        candidate_places: list,
        transitions: list,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Training forward pass with teacher forcing.

        Args:
            all_embeddings: [N, feature_dim] initial node features
            adj: [2, E] adjacency
            direction: [E, 2] direction vectors
            place_indices: indices of place nodes in the full graph
            petri_indices: indices of petri net nodes (transitions + places)
            target_sequence: BFS-ordered list of place indices to select
            candidate_places: list of (in_set, out_set) tuples
            transitions: list of transition labels

        Returns:
            total_loss: negative log-likelihood loss
            step_probs: list of probability distributions at each step
        """
        h = self.pn1(all_embeddings, adj, direction)

        selected_mask = torch.zeros(len(place_indices), dtype=torch.bool, device=h.device)
        total_loss = torch.tensor(0.0, device=h.device)
        step_probs = []

        for target_idx in target_sequence:
            place_h = h[place_indices]
            probs, _ = self.scn(place_h, selected_mask)
            step_probs.append(probs)

            local_target = self._global_to_local(target_idx, place_indices)
            if local_target >= 0 and probs[local_target] > 0:
                total_loss -= torch.log(probs[local_target] + 1e-10)

            # Teacher forcing: mark the correct target as selected
            if local_target >= 0:
                selected_mask[local_target] = True

            selected_feature = torch.zeros(h.size(0), 1, device=h.device)
            for si in range(len(place_indices)):
                if selected_mask[si]:
                    selected_feature[place_indices[si]] = 1.0

            h_extended = torch.cat([h, selected_feature], dim=-1)
            h = self.pn2(h_extended, adj, direction)

        return total_loss, step_probs

    def forward_inference(
        self,
        all_embeddings: torch.Tensor,
        adj: torch.Tensor,
        direction: torch.Tensor,
        place_indices: torch.Tensor,
        petri_indices: torch.Tensor,
        candidate_places: list,
        transitions: list,
        max_places: int = 50,
        stop_threshold: float = 0.5,
    ) -> Tuple[List[int], float]:
        """
        Inference: iteratively select places until SN says stop.

        Returns:
            selected_place_indices: list of selected place indices
            joint_probability: product of selection probabilities
        """
        h = self.pn1(all_embeddings, adj, direction)

        selected_mask = torch.zeros(len(place_indices), dtype=torch.bool, device=h.device)
        selected_places: Set[int] = set()
        selected_sequence: List[int] = []
        log_joint_prob = 0.0

        for step in range(max_places):
            place_h = h[place_indices]

            # SCN with S-coverability check
            idx, probs = self.scn.select_best(
                place_h,
                selected_mask,
                s_coverability_checker=self.s_checker,
                selected_places=selected_places,
                candidate_places=candidate_places,
                transitions=transitions,
            )

            if idx < 0:
                break

            selected_mask[idx] = True
            selected_places.add(idx)
            selected_sequence.append(idx)
            log_joint_prob += torch.log(probs[idx] + 1e-10).item()

            # SN: check if we should stop
            petri_h = h[petri_indices]
            is_wf = self.s_checker.is_workflow_net(
                [candidate_places[i] for i in selected_places],
                transitions,
            )
            should_stop, p_add = self.sn.should_stop(
                petri_h, threshold=stop_threshold, is_workflow_net=is_wf
            )

            if should_stop:
                break

            # PN2: propagate the decision
            selected_feature = torch.zeros(h.size(0), 1, device=h.device)
            for si in range(len(place_indices)):
                if selected_mask[si]:
                    selected_feature[place_indices[si]] = 1.0

            h_extended = torch.cat([h, selected_feature], dim=-1)
            h = self.pn2(h_extended, adj, direction)

        return selected_sequence, log_joint_prob

    @staticmethod
    def _global_to_local(global_idx: int, place_indices: torch.Tensor) -> int:
        matches = (place_indices == global_idx).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            return matches[0].item()
        return -1
