"""old.py — v1 (unweighted) tertile sweep, all 3 approaches × 2 edge cols.

Mirrors the logic of `notebooks/old/gnn_*.ipynb`:
- WeightedSAGE wraps SAGEConv (its forward signature accepts edge_weight but
  the v1 notebooks don't actually consume it — kept identical here).
- WeightedGAT wraps GATConv (multi-head, no edge_attr).

Outputs land under: notebooks/slumrun/results/old/<approach>/...

Resumable + atomic-write via _common.run_all. Set FGNN_SMOKE=1 for a quick
smoke test (5 epochs, 1 quarter per approach × edge_col).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

import SocialNetwork.src.Korkevados.storage.slumrun._common as _c
from SocialNetwork.src.Korkevados.storage.slumrun._common import NUM_CLASSES


class WeightedSAGE(nn.Module):
    """v1 SAGE — accepts edge_weight in signature but does not consume it."""

    def __init__(self, in_dim, hidden_dim, num_classes=NUM_CLASSES,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class WeightedGAT(nn.Module):
    """v1-style GAT — multi-head attention, no edge_attr.

    Plain ``GATConv`` (matches the original cluster notebooks). GATv2Conv was
    tried earlier and triggered a cuda/cpu device-mismatch on the bipartite
    (CIK↔CUSIP) graphs even with ``add_self_loops=False``. GATConv has no such
    issue and gives the v1 baseline the original notebooks shipped with.
    """

    def __init__(self, in_dim, hidden_dim, num_classes=NUM_CLASSES,
                 num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim,
                                       heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1,
                                   concat=False, dropout=dropout))
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)  # no edge_attr — keeps the "unweighted" v1 semantics
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


def forward_fn(model, data):
    """v1 forward: only (x, edge_index, edge_weight). edge_weight is ignored
    by the v1 model classes but still accepted positionally."""
    return model(data.x, data.edge_index, getattr(data, "edge_weight", None))


def main():
    _c.run_all(
        version="old",
        sage_cls=WeightedSAGE,
        gat_cls=WeightedGAT,
        weighted=False,
        forward_fn=forward_fn,
    )


if __name__ == "__main__":
    main()
