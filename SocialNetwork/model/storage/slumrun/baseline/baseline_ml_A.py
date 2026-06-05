"""baseline_ml_A.py — multiq_A only baseline runner.

Runs LogReg + HGB + MLP on the multiq_A train window (concat of K=3 past
quarters' stock rows) so it can SLURM-fan out in parallel with
baseline_ml_B.py, mirroring the old_A.py / old_B.py pattern.

Outputs land at: slumrun/results/baseline/multiq_A/
"""
from __future__ import annotations

import sys
from pathlib import Path

# Both slumrun/ and slumrun/baseline/ on path so we can import _common
# (parent) and baseline_ml (sibling) without an __init__.py.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from SocialNetwork.src.Korkevados.storage.slumrun.baseline.baseline_ml import run_all


def main():
    run_all(approaches=["multiq_A"])


if __name__ == "__main__":
    main()
