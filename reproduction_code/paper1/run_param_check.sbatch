#!/bin/bash
#SBATCH --account=l2_jan_lab_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --job-name=param_check
#SBATCH --output=/home/giobbij/cs6362_aml/paper1/logs/param_check_%j.log
#SBATCH --error=/home/giobbij/cs6362_aml/paper1/logs/param_check_%j.err

# Resumable: if the job hits the time limit, just `sbatch` it again.
# Finished runs are cached in results.json and skipped.

cd "/home/giobbij/cs6362_aml/paper1"

source /panfs/accrepfs.vampire/home/giobbij/miniforge3/etc/profile.d/conda.sh
conda activate fig1

python3 param_check.py
