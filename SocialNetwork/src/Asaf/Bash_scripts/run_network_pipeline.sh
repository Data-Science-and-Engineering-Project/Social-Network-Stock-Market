#!/bin/bash

######################################################################
# SBATCH DIRECTIVES - הגדרות משאבים לשרת BGU
######################################################################
#SBATCH --job-name=stock_prediction
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# הגדרות תור ומשאבים
#SBATCH --partition=main
#SBATCH --qos=normal

# בקשת GPU מינימלי לכניסה מהירה לתור
#SBATCH --gpus=gtx_1080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# הגדרות אימייל
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=

######################################################################
# EXECUTION STEPS - שלבי ההרצה
######################################################################

echo "Job started on $(hostname) at $(date)"

# 1. מעבר לתיקיית הקוד (הנתיב המעודכן ללא רווח)
cd "/home/zenoua/Social-Network-Stock-Market/SocialNetwork/src" || { echo "Directory not found!"; exit 1; }

# 2. הגדרות סביבה (מניעת באפר בלוגים)
export PYTHONUNBUFFERED=1

# 3. הרצת קוד ה-Python
# וודא שבתוך הקוד שינית את ה-root ל: Social-Network-Stock-Market/SocialNetwork/parquet_files
echo "Running network-pipeline.py..."
python network-pipeline.py

echo "Job finished at $(date)"