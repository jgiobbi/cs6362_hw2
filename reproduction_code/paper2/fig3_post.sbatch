#!/bin/bash
#SBATCH --account=l2_jan_lab_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=fig3_post
#SBATCH --output=/home/giobbij/cs6362_aml/paper2/logs/paper2_%j.log
#SBATCH --error=/home/giobbij/cs6362_aml/paper2/logs/paper2_%j.err

cd "/home/giobbij/cs6362_aml/paper2"

source /panfs/accrepfs.vampire/home/giobbij/miniforge3/etc/profile.d/conda.sh
conda activate fig1

python3 -u fig3.py interp --out outputs --data-dir data
python3 -u fig3.py plot --out outputs

