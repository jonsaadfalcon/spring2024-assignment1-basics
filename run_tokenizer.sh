#!/bin/bash
#SBATCH --job-name=test_hello_batch
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=hello_batch_v2.1_%j.out
#SBATCH --error=hello_batch_v2.1_%j.err

# Optional: activate a conda environment to use for this job
# eval "$(conda shell.bash hook)"
# conda activate cs336_basics

nohup python3 testing_tokenizer.py &> logs/tiny-stories_training.log &
