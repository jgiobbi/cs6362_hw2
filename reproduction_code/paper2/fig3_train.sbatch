#!/bin/bash
# Array of 6 training jobs: 0..5 -> (F1,C1,C2) x (SB,LB). Fill in the <...> placeholders for your ACCRE account/partition.
#SBATCH --job-name=fig3_train
#SBATCH --array=0-5
#SBATCH --account=<account>
#SBATCH --partition=<gpu-partition-with-A6000>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
mkdir -p logs outputs
NETS=(F1 F1 C1 C1 C2 C2); REGS=(SB LB SB LB SB LB)
# <activate your env / load modules here>
python fig3.py train --net ${NETS[$SLURM_ARRAY_TASK_ID]} --reg ${REGS[$SLURM_ARRAY_TASK_ID]} --data-dir data --out outputs
# If a job hits the walltime, just resubmit that array index: it resumes from its last checkpoint (every 5 epochs).
