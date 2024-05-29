#!/bin/bash
#SBATCH -J train_echomorph
# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=logs/slurm-%x.%j.out
# The job requires 1 compute node
#SBATCH -N 1
# The job requires 1 task per node
#SBATCH --ntasks-per-node=1
# The maximum walltime of the job
#SBATCH -t 50:10:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=semjon.00@gmail.com
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon2

module load any/python/3.8.3-conda
# git pull --force
git log -n 1 --pretty=format:"Commit: %H %s%n"
# conda deactivate
# conda create -n hypatia
conda activate hypatia
# conda install "pytorch::pytorch-cuda" "pytorch::torchaudio" "conda-forge::einops" "conda-forge::torchinfo" "conda-forge::ffmpeg<7" -c nvidia
export PYTHONUNBUFFERED=TRUE
python training.py --total_epochs=2 --batch_size=32 --learning_rate=0.0001 --save_time=3600 --no_random_degradation

# Reminder:
#sbatch job_train.sh
#squeue -u semjon00
#squeue -j <job number>
#scancel
