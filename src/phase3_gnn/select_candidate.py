"""
Select Candidate Network (SCN) - Paper Section IV-A2.
Single fully-connected layer that scores each candidate place
and uses SOFTMAX to produce a probability distribution.

Implements Equations (3) and (4):
  s_v = h_v * W           (Eq. 3)
  p_v = softmax(s_v)      (Eq. 4)

Integrates S-coverability check: if a selected candidate causes
loss of S-coverability, it is rejected and the next most probable
candidate is chosen.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Set, Tuple


class SelectCandidateNetwork(nn.Module):
    """
    SCN: Maps node embedding h_v to a score s_v via a single FC layer,
    then applies softmax over all unselected candidate places.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.W = nn.Linear(embedding_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(
        self,
        place_embeddings: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            place_embeddings: [num_places, embedding_dim]
            selected_mask: [num_places] boolean, True = already selected

        Returns:
            probabilities: [num_places] softmax probabilities (0 for selected)
            scores: [num_places] raw scores
        """
        scores = self.W(place_embeddings).squeeze(-1)

        masked_scores = scores.clone()
        masked_scores[selected_mask] = float("-inf")

        probabilities = F.softmax(masked_scores, dim=0)
        probabilities[selected_mask] = 0.0

        return probabilities, scores

    def select_best(
        self,
        place_embeddings: torch.Tensor,
        selected_mask: torch.Tensor,
        s_coverability_checker=None,
        selected_places: Optional[Set[int]] = None,
        candidate_places: Optional[list] = None,
        transitions: Optional[list] = None,
    ) -> Tuple[int, torch.Tensor]:
        """
        Select the best candidate place, respecting S-coverability.

        Returns:
            (selected_index, probabilities)
        """
        probs, scores = self.forward(place_embeddings, selected_mask)

        sorted_indices = torch.argsort(probs, descending=True)

        for idx in sorted_indices:
            idx_val = idx.item()
            if selected_mask[idx_val]:
                continue
            if probs[idx_val] <= 0:
                break

            if s_coverability_checker is not None and candidate_places is not None:
                test_places = (selected_places or set()) | {idx_val}
                place_tuples = [candidate_places[i] for i in test_places]
                if not s_coverability_checker.check(place_tuples, transitions):
                    continue

            return idx_val, probs

        return -1, probs
