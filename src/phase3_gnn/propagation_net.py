"""
Propagation Networks (PN1, PN2) using GCN with K-headed attention.
Per the paper Section IV-A1 and IV-A4:
  - PN1: >= 3 layers, propagates behavioral info from trace graph to places
  - PN2: >= 2 layers, propagates decision info after candidate selection

Implements Equations (1) and (2) from the paper with:
  - Bi-directional edges with direction vectors d_ij
  - Self-loops for information retention
  - Multi-head attention mechanism (GAT-style)
  - RELU activation on the final layer
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PropagationLayer(nn.Module):
    """
    Single propagation layer with K-headed attention.
    Implements the update function from Eq. (1) with direction-aware processing.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_heads: int,
        is_last_layer: bool = False,
    ):
        super().__init__()
        self.is_last_layer = is_last_layer
        self.num_heads = num_heads
        self.out_feats = out_feats

        self.W_forward = nn.Linear(in_feats, out_feats, bias=False)
        self.W_reverse = nn.Linear(in_feats, out_feats, bias=False)

        self.attn_forward = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_reverse = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W_forward.weight)
        nn.init.xavier_uniform_(self.W_reverse.weight)
        nn.init.xavier_uniform_(self.attn_forward)
        nn.init.xavier_uniform_(self.attn_reverse)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: node embeddings [N, in_feats]
            adj: adjacency pairs [2, E] (src, dst)
            direction: direction vectors [E, 2] where [1,0]=forward, [0,1]=reverse
        Returns:
            Updated embeddings [N, num_heads * out_feats] or [N, out_feats] for last layer
        """
        N = h.size(0)
        h_forward = self.W_forward(h)
        h_reverse = self.W_reverse(h)

        if adj.size(1) == 0:
            if self.is_last_layer:
                return F.relu(h_forward)
            return h_forward.unsqueeze(1).expand(-1, self.num_heads, -1).reshape(N, -1)

        src, dst = adj[0], adj[1]
        d_fwd = direction[:, 0:1]
        d_rev = direction[:, 1:2]

        msg_fwd = h_forward[src] * d_fwd
        msg_rev = h_reverse[src] * d_rev
        messages = msg_fwd + msg_rev

        # Multi-head attention scores
        messages_heads = messages.unsqueeze(1).expand(-1, self.num_heads, -1)

        attn_scores = (messages_heads * self.attn_forward).sum(dim=-1)
        attn_scores = self.leaky_relu(attn_scores)

        # Scatter with softmax normalization per destination
        attn_weights = torch.zeros(N, self.num_heads, device=h.device)
        output = torch.zeros(N, self.num_heads, self.out_feats, device=h.device)

        # Softmax per destination node
        for head in range(self.num_heads):
            head_scores = attn_scores[:, head]
            head_messages = messages_heads[:, head, :]

            exp_scores = torch.exp(head_scores - head_scores.max())
            sum_exp = torch.zeros(N, device=h.device)
            sum_exp.scatter_add_(0, dst, exp_scores)
            norm_scores = exp_scores / (sum_exp[dst] + 1e-10)

            weighted_msg = head_messages * norm_scores.unsqueeze(-1)
            output[:, head, :].scatter_add_(
                0, dst.unsqueeze(-1).expand(-1, self.out_feats), weighted_msg
            )

        if self.is_last_layer:
            result = output.mean(dim=1)
            return F.relu(result)
        else:
            return output.reshape(N, self.num_heads * self.out_feats)


class PropagationNetwork(nn.Module):
    """
    Multi-layer propagation network (PN1 or PN2).

    PN1 config (paper): 4 layers [20, 32, 64, 32] -> output 16
    PN2 config (paper): 2 layers [17, 32] -> output 16
    """

    def __init__(
        self,
        layer_dims: list[int],
        output_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layer_dims)):
            in_dim = layer_dims[i]
            if i < len(layer_dims) - 1:
                out_dim = layer_dims[i + 1] // num_heads
                is_last = False
            else:
                out_dim = output_dim
                is_last = True

            self.layers.append(
                PropagationLayer(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    num_heads=1 if is_last else num_heads,
                    is_last_layer=is_last,
                )
            )

    def forward(self, h: torch.Tensor, adj: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, adj, direction)
        return h


def build_pn1(feature_dim: int, output_dim: int = 16, num_heads: int = 4) -> PropagationNetwork:
    """Build PN1 with paper's hyperparameters: 4 layers."""
    return PropagationNetwork(
        layer_dims=[feature_dim, 32, 64, 32],
        output_dim=output_dim,
        num_heads=num_heads,
    )


def build_pn2(feature_dim: int, output_dim: int = 16, num_heads: int = 4) -> PropagationNetwork:
    """Build PN2 with paper's hyperparameters: 2 layers."""
    return PropagationNetwork(
        layer_dims=[feature_dim, 32],
        output_dim=output_dim,
        num_heads=num_heads,
    )
