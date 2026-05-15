#!/bin/bash

######################################################################
# SBATCH DIRECTIVES - SLURM Resource Configuration
# Preferential Attachment Link Prediction - Extended Runtime (3+ days)
######################################################################

#SBATCH --job-name=pa_link_prediction
#SBATCH --output=slurm_pa_%j.out
#SBATCH --error=slurm_pa_%j.err
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
echo "PREFERENTIAL ATTACHMENT LINK PREDICTION - SLURM JOB STARTED"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "================================================================================"

cd "/home/zenoua/Social-Network-Stock-Market/SocialNetwork/src/Asaf/preferentialAttachment" || { 
    echo "ERROR: Working directory not found!"; 
    exit 1; 
}

echo "[*] Working directory: $(pwd)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[*] Python unbuffered mode: ENABLED"
echo "[*] OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

echo "[*] Starting Preferential Attachment model execution..."
echo "================================================================================"

python run_preferentialattachment_with_logs.py

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "JOB EXECUTION COMPLETED"
echo "================================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[✓] SUCCESS: Preferential Attachment model completed successfully"
else
    echo "[!] ERROR: Script exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
