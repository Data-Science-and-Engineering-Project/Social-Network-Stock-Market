#!/bin/bash
#SBATCH --job-name=stock_graph_job
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# זמן בלתי מוגבל (תלוי קלאסטר – לפעמים צריך partition מתאים)
#SBATCH --time=UNLIMITED

# אם יש GPU (אופציונלי)
#SBATCH --gres=gpu:1

echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# הפעלת סביבה (אם יש)
source ~/.bashrc

# אם אתה משתמש ב-conda:
# conda activate your_env

# מעבר לתיקייה של הפרויקט
cd /path/to/your/project

# יצירת תיקיית לוגים אם לא קיימת
mkdir -p logs

# הרצת הקוד
python network-pipeline.py

echo "Job finished"