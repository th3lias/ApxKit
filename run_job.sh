#!/bin/bash
#SBATCH --job-name=ApxKit
#SBATCH --partition=gpu-v100s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# timestamp
CUR_DATE_TIME=$(date +"%d_%m_%Y_%H_%M_%S")
RESULT_DIR="results/$CUR_DATE_TIME"

# folder
mkdir -p "$RESULT_DIR"

# load modules
module load nvidia/cuda/12.8
module load python/312

# activate venv
source .venv/bin/activate

# write job_id to a file in the results folder
echo $SLURM_JOB_ID > "$RESULT_DIR/pid.txt"

# start main.py and redirect stdout and stderr to console_out.txt in the results folder
python main.py --folder_name "$CUR_DATE_TIME" > "$RESULT_DIR/console_out.txt" 2>&1