#!/bin/bash

######################################################################
# SBATCH DIRECTIVES - SLURM Resource Configuration
# Adamic-Adar Link Prediction - Extended Runtime (3+ days)
######################################################################

#SBATCH --job-name=aa_link_prediction
#SBATCH --output=slurm_aa_%j.out
#SBATCH --error=slurm_aa_%j.err
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=

######################################################################
# EXECUTION STEPS
######################################################################

echo "================================================================================"
echo "ADAMIC-ADAR LINK PREDICTION - SLURM JOB STARTED"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================================================================"

cd "/home/zenoua/Social-Network-Stock-Market/SocialNetwork/src/Asaf/adamicAdar" || { 
    echo "ERROR: Working directory not found!"; 
    exit 1; 
}

echo "[*] Working directory: $(pwd)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[*] Python unbuffered mode: ENABLED"
echo "[*] OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

echo "[*] Starting Adamic-Adar model execution..."
echo "================================================================================"

python run_adamicadar_with_logs.py

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "JOB EXECUTION COMPLETED"
echo "================================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[✓] SUCCESS: Adamic-Adar model completed successfully"
else
    echo "[!] ERROR: Script exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
