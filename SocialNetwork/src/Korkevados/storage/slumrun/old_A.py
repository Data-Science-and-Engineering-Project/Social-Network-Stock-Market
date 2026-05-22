"""old_A.py — multiq_A only (block-diagonal union, K=3).

Same v1 (unweighted) sweep as old.py, scoped to a single approach so it can
run as its own SLURM job in parallel with old_B.py. Reuses the exact model
classes and forward_fn from old.py — no duplication.

Outputs land under: notebooks/slumrun/results/old/multiq_A/...
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
        approaches=["multiq_A"],
    )


if __name__ == "__main__":
    main()
