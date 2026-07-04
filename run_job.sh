#!/bin/bash
#SBATCH --job-name=apxkit_exp
#SBATCH --partition=gpu-v100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# 1. Exakt den gleichen Zeitstempel wie in deinem alten Skript generieren
CUR_DATE_TIME=$(date +"%d_%m_%Y_%H_%M_%S")
RESULT_DIR="results/$CUR_DATE_TIME"

# 2. Ordner erstellen
mkdir -p "$RESULT_DIR"

# 3. SLURM anweisen, die Log-Dateien genau dorthin zu schreiben
# --output und --error können innerhalb des Skripts über scontrol/srun umgeleitet werden,
# aber der sauberste Weg für die exakt gleiche Struktur ist die Angabe beim sbatch-Aufruf.
# Daher nutzen wir diese Variablen für den Python-Aufruf unten.

# 4. Cluster-Module laden
module load nvidia/cuda/12.8
module load python/312

# 5. Deine virtuelle Umgebung aktivieren
source .venv/bin/activate

# 6. Job-ID (entspricht der PID auf Clustern) in die pid.txt schreiben
echo $SLURM_JOB_ID > "$RESULT_DIR/pid.txt"

# 7. Python ausführen und die Terminal-Ausgabe exakt wie bei dir umleiten
python main.py --folder_name "$CUR_DATE_TIME" > "$RESULT_DIR/console_out.txt" 2>&1