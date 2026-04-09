"""
Phase 4: Regression Head for cost/resource prediction.
Added after the Petri net structure is established.
Takes graph embedding and predicts:
  - Total Cost for a process branch
  - HumanRes count
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Linear regression layer on top of graph embeddings
    to predict cost/time/resource metrics for process branches.
    """

    def __init__(self, embedding_dim: int, num_targets: int = 2):
        """
        Args:
            embedding_dim: size of the graph/node embedding input
            num_targets: number of regression targets (default 2: cost, human_res)
        """
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.output = nn.Linear(embedding_dim, num_targets)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_embedding: [batch, embedding_dim] or [embedding_dim]

        Returns:
            predictions: [batch, num_targets] or [num_targets]
        """
        x = self.relu(self.fc1(graph_embedding))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.output(x)

    def predict_cost(self, graph_embedding: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            preds = self.forward(graph_embedding)
        if preds.dim() > 1:
            return preds[0, 0].item()
        return preds[0].item()

    def predict_human_res(self, graph_embedding: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            preds = self.forward(graph_embedding)
        if preds.dim() > 1:
            return preds[0, 1].item()
        return preds[1].item()
