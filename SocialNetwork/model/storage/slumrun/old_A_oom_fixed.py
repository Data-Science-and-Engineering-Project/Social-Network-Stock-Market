"""old_A_oom_fixed.py — multiq_A with NeighborLoader mini-batch training.

Same scope as old_A.py (v1 unweighted sweep, multiq_A only) but trains via
torch_geometric.loader.NeighborLoader instead of full-graph forward passes.
Cuts GAT peak GPU memory from ~18GB to 2-4GB on the block-diagonal K=3 union
graph, fixing the cluster OOM.

Tunables (env vars):
  FGNN_FANOUT     fanout per layer, e.g. "20,10" (default — matches num_layers=2)
  FGNN_BS_TRAIN   training batch size in seed nodes (default 512)
  FGNN_BS_EVAL    inference batch size (default 2048)

Outputs land under: notebooks/slumrun/results/old/multiq_A/...
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
        approaches=["multiq_A"],
    )


if __name__ == "__main__":
    main()
