#!/bin/bash

######################################################################
# SBATCH DIRECTIVES - SLURM Resource Configuration for BGU Server
# Baseline Link Prediction Model - Extended Runtime (3+ days)
######################################################################

#SBATCH --job-name=baseline_link_prediction
#SBATCH --output=slurm_baseline_%j.out
#SBATCH --error=slurm_baseline_%j.err

# Partition and QoS settings
#SBATCH --partition=main
#SBATCH --qos=normal

# Resource allocation (CPU-only, memory-intensive)
# Baseline models are computationally light but data-intensive
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00

# Email notifications (update with your email)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=

######################################################################
# EXECUTION STEPS
######################################################################

echo "================================================================================"
echo "BASELINE LINK PREDICTION - SLURM JOB STARTED"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================================================================"

# 1. Change to working directory (adjust path if needed)
cd "/home/zenoua/Social-Network-Stock-Market/SocialNetwork/src" || { 
    echo "ERROR: Working directory not found!"; 
    exit 1; 
}

echo "[*] Working directory: $(pwd)"

# 2. Environment setup (prevent output buffering for real-time logs)
export PYTHONUNBUFFERED=1

# 3. Optimize CPU usage for mathematical libraries
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. Optional: Add Python package path if using custom installations
# export PYTHONPATH="/home/zenoua/miniconda3/envs/your_env/lib/python3.x/site-packages:$PYTHONPATH"

echo "[*] Python unbuffered mode: ENABLED"
echo "[*] OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# 5. Run the baseline script with extensive logging
echo "[*] Starting baseline link prediction model execution..."
echo "[*] This will process sliding windows over ALL YEARS (2018-2024)"
echo "[*] Estimated runtime: 3+ days"
echo "================================================================================"

python run_baselines_with_logs.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "JOB EXECUTION COMPLETED"
echo "================================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================================"

# 6. Optional: Save results summary
if [ $EXIT_CODE -eq 0 ]; then
    echo "[✓] SUCCESS: Baseline models completed successfully"
    echo "[*] Results saved to: results/baselines_scores_detailed_report.csv"
else
    echo "[!] ERROR: Script exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
