#!/bin/bash
# run_sweep.sbatch — Slurm submission script for sweep_features.py
#
# Submit:  sbatch run_sweep.sbatch
# With args: sbatch run_sweep.sbatch --edges-col change_in_adjusted_weight
# Monitor: squeue -u $USER ; tail -f logs/slurm_<JOBID>.log
# Cancel:  scancel <JOBID>
#
# After submission, the job runs on a compute node — you can log out and close
# your laptop. Slurm keeps the job running until it finishes, fails, or hits
# the --time limit.
#
# >>> ADJUST the partition / account / time / mem / gres lines to match your
# >>> cluster's policies before submitting. <<<

#SBATCH --job-name=lightgcn_sweep
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.log
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT      # uncomment + set if your cluster requires accounting
##SBATCH --mail-type=END,FAIL        # uncomment to get email on completion / failure
##SBATCH --mail-user=you@example.com

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# --- Activate your Python environment (ADJUST as needed) ---
# Common patterns:
#   module load python/3.11 cuda/12.1
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate fgnn
#   source ~/venvs/fgnn/bin/activate

echo "=== Slurm job $SLURM_JOB_ID ==="
echo "node:    $(hostname)"
echo "started: $(date)"
echo "python:  $(which python)"
echo "torch:   $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo

python -u sweep_features.py "$@"

echo
echo "finished: $(date)"
