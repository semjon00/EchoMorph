#!/bin/bash
#SBATCH -J train_echomorph
# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=logs/slurm-%x.%j.out
# The job requires 1 compute node
#SBATCH -N 1
# The job requires 1 task per node
#SBATCH --ntasks-per-node=1
# The maximum walltime of the job
#SBATCH -t 10:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=semjon.00@gmail.com
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon2

module load any/python/3.8.3-conda
#module load ffmpeg/4.4.1
#module load sox/14.4.2
conda activate transformers-course
#conda install conda-forge::ffmpeg
#conda install esri::einops

export PYTHONUNBUFFERED=TRUE
python training.py --save_time=1800 --batch_size=32 --fp16

# Reminder:
#sbatch train_job.sh
#squeue -u semjon00
#squeue -j <job number>
#scancel