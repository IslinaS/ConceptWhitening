#!/usr/bin/env bash
#SBATCH --job-name=cw_CUB             # Job name
#SBATCH --output=logs/log_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=160gb                   # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:0 

python3 data/make_labels.py