"""
Stop Network (SN) - Paper Section IV-A3.
Two-layer network that decides when to stop adding candidate places.

Layer 1 (Eq. 5): Graph embedding via gating function
  h_G = Σ_v sigmoid(h_v * W_a) ⊙ (h_v * W_g)

Layer 2 (Eq. 6): Stop probability
  p_add = sigmoid(h_G * W_d)

Additional constraint: if the Petri net is not yet a connected
workflow net, the stop decision is overridden.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StopNetwork(nn.Module):
    """
    SN: Aggregates Petri net node embeddings into a graph embedding,
    then outputs a probability for whether to continue adding places.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.W_a = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_g = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_d = nn.Linear(embedding_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.xavier_uniform_(self.W_d.weight)

    def forward(self, petri_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            petri_embeddings: [N_petri, embedding_dim] embeddings of all
                              Petri net nodes (transitions + selected places)

        Returns:
            p_add: scalar probability that another place should be added
        """
        gate = torch.sigmoid(self.W_a(petri_embeddings))
        value = self.W_g(petri_embeddings)
        h_G = (gate * value).sum(dim=0)

        p_add = torch.sigmoid(self.W_d(h_G))
        return p_add.squeeze()

    def should_stop(
        self,
        petri_embeddings: torch.Tensor,
        threshold: float = 0.5,
        is_workflow_net: bool = True,
    ) -> tuple[bool, float]:
        """
        Decide whether to stop, with workflow net override.

        Returns:
            (should_stop, p_add_value)
        """
        p_add = self.forward(petri_embeddings)
        p_add_val = p_add.item()

        wants_to_stop = p_add_val < threshold

        if wants_to_stop and not is_workflow_net:
            return False, p_add_val

        return wants_to_stop, p_add_val
