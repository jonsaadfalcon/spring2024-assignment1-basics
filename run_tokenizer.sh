#!/bin/bash
#SBATCH --job-name=test_hello_batch
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=30G
#SBATCH --output=hello_batch_v2_%j.out
#SBATCH --error=hello_batch_v2_%j.err

# Optional: activate a conda environment to use for this job
# eval "$(conda shell.bash hook)"
# conda activate cs336_basics

nohup python3 testing_tokenizer.py &> logs/owl_training.log &
