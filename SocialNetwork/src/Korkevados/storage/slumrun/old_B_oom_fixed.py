"""old_B_oom_fixed.py — multiq_B with NeighborLoader mini-batch training.

Same scope as old_B.py (v1 unweighted sweep, multiq_B only) but trains via
torch_geometric.loader.NeighborLoader instead of full-graph forward passes.
multiq_B is the worst offender for GAT OOM — the temporal graph shares
fund/CUSIP nodes across K=3 quarter slots and concatenates per-slot
edge_index, so attention coefficients explode. Mini-batch sampling caps
memory per step at O(batch_size * fanout^L).

Tunables (env vars):
  FGNN_FANOUT     fanout per layer, e.g. "20,10" (default — matches num_layers=2)
  FGNN_BS_TRAIN   training batch size in seed nodes (default 512)
  FGNN_BS_EVAL    inference batch size (default 2048)

Outputs land under: notebooks/slumrun/results/old/multiq_B/...
"""
from __future__ import annotations

import SocialNetwork.src.Korkevados.storage.slumrun._common as _c
import SocialNetwork.src.Korkevados.storage.slumrun._minibatch as _mb
from SocialNetwork.src.Korkevados.storage.slumrun.old import WeightedSAGE, WeightedGAT, forward_fn


def main():
    _mb.patch_common()
    _c.run_all(
        version="old",
        sage_cls=WeightedSAGE,
        gat_cls=WeightedGAT,
        weighted=False,
        forward_fn=forward_fn,
        approaches=["multiq_B"],
    )


if __name__ == "__main__":
    main()
