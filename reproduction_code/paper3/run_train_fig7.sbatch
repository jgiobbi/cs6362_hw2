#!/bin/bash
#SBATCH --account=l2_jan_lab_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=fig7_bvae
#SBATCH --output=/home/giobbij/cs6362_aml/paper3/logs/paper3_%j.log
#SBATCH --error=/home/giobbij/cs6362_aml/paper3/logs/paper3_%j.err

# Resumable: if the job hits the time limit, just `sbatch` it again.
# Finished runs are cached in results.json and skipped.

cd "/home/giobbij/cs6362_aml/paper3"

source /panfs/accrepfs.vampire/home/giobbij/miniforge3/etc/profile.d/conda.sh
conda activate fig1

DSPRITES=/home/giobbij/cs6362_aml/paper3/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
ls -lh "$DSPRITES"                                                       # fails early if wrong
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python3 -u train_fig7.py \
    --dsprites-path "$DSPRITES" \
    --out-dir fig7_outputs --epochs 30 --batch-size 64 --seeds 0