#!/bin/bash
#SBATCH --job-name=stock_graph_job
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

# הגדרות שליחת מייל בסיום או כשל
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zenoua@post.bgu.ac.il

# משאבים: משימה אחת עם 8 ליבות (טוב ל-LightGBM/NetworkX) ו-32GB ראם
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# הגבלת זמן ל-5 ימים (פורמט: ימים-שעות:דקות:שניות)
#SBATCH --time=5-00:00:00

# הגדרה מפורשת של ה-partition ל-CPU כדי למנוע אזהרות
#SBATCH --partition=cpu

echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# טעינת סביבת עבודה
source ~/.bashrc

# אם אתה משתמש ב-Conda, בטל את ה-comment בשורה הבאה ושנה לשם הסביבה שלך
# conda activate your_env

# מעבר לתיקיית המקור (השתמשתי בנתיב מהצילום מסך שלך)
cd "/home/zenoua/Social-Network-Stock-Market/Social Network/src"

# יצירת תיקיית לוגים אם היא לא קיימת
mkdir -p logs

# הרצת הסקריפט
python network-pipeline.py

echo "Job finished at $(date)"