#!/bin/bash
#SBATCH --job-name=test_hello_batch
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=hello_batch_%j.out
#SBATCH --error=hello_batch_%j.err

# Optional: activate a conda environment to use for this job
# eval "$(conda shell.bash hook)"
# conda activate cs336_basics

python3 testing_tokenizer.py
