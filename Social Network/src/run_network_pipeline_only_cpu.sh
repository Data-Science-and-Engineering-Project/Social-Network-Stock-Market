#!/bin/bash

######################################################################
# SBATCH DIRECTIVES - הגדרות משאבים לשרת BGU
######################################################################
#SBATCH --job-name=stock_prediction
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# שינוי ל-Partition של CPU בלבד (מגדיל סיכוי להיכנס מהר ובלי איומי Idle)
#SBATCH --partition=main
#SBATCH --qos=normal

# ביטלנו את ה-GPU כדי למנוע את הודעות ה-Idle שקיבלת
# במקום זה, ביקשנו יותר ליבות (CPU) כדי להאיץ את חישובי המרכזיות (Step 2)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# הגדרות אימייל
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=

######################################################################
# EXECUTION STEPS - שלבי ההרצה
######################################################################

echo "Job started on $(hostname) at $(date)"

# 1. מעבר לתיקיית הקוד
cd "/home/zenoua/Social-Network-Stock-Market/SocialNetwork/src" || { echo "Directory not found!"; exit 1; }

# 2. הגדרות סביבה (מניעת באפר בלוגים)
export PYTHONUNBUFFERED=1

# 3. הגדרת כמות התהליכונים עבור ספריות מתמטיות (מאיץ את ה-CPU)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. הרצת קוד ה-Python
# הקוד שלך יזהה אוטומטית שאין CUDA ויעבור ל-CPU Device
echo "Running network-pipeline.py on CPU only..."
python network-pipeline.py

echo "Job finished at $(date)"