"""baseline_ml_B.py — multiq_B only baseline runner.

Runs LogReg + HGB + MLP on the multiq_B feature window (per-CUSIP K=3
quarter feature stack, 30 features) so it can SLURM-fan out in parallel
with baseline_ml_A.py.

Outputs land at: slumrun/results/baseline/multiq_B/
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from SocialNetwork.src.Korkevados.storage.slumrun.baseline.baseline_ml import run_all


def main():
    run_all(approaches=["multiq_B"])


if __name__ == "__main__":
    main()
