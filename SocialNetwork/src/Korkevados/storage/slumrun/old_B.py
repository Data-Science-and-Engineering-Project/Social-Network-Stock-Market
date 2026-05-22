"""old_B.py — multiq_B only (temporal, K=3 shared-node slots).

Same v1 (unweighted) sweep as old.py, scoped to a single approach so it can
run as its own SLURM job in parallel with old_A.py. Reuses the exact model
classes and forward_fn from old.py — no duplication.

Outputs land under: notebooks/slumrun/results/old/multiq_B/...
"""
from __future__ import annotations

import SocialNetwork.src.Korkevados.storage.slumrun._common as _c
from SocialNetwork.src.Korkevados.storage.slumrun.old import WeightedSAGE, WeightedGAT, forward_fn


def main():
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
