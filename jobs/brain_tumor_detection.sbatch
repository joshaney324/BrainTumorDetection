#!/bin/bash

#SBATCH --account=priority-csci-494-594-hpc
#SBATCH --partition=gpupriority
#SBATCH --job-name=brain_tumor_detection
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128000
#SBATCH --time=0-72:00:00
#SBATCH --output=brain_tumor_detection-%j.out
#SBATCH --error=brain_tumor_detection-%j.err
#SBATCH --mail-user=josh.aney@icloud.com
#SBATCH --mail-type=ALL

module purge
module load Mamba/4.14.0-0

source ~/.bashrc

cd /home/x89h835/BrainTumorDetection
/home/x89h835/BrainTumorDetection/.venv/bin/python src/main.py
