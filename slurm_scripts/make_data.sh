#!/usr/bin/env bash
#SBATCH --job-name=cw_CUB             # Job name
#SBATCH --output=logs/log_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=160gb                   # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:0 

source slurm_scripts/secrets.txt
source $VENV_PATH/bin/activate
# 15: Lazuli Bunting (colorful, normal shape), 107: Common Raven (bizarre shape, no color), 
# 186: Cedar Waxwing (tiny concepts like eyes I suspect)
python3 data/make_labels.py 15 107 186 --write_json