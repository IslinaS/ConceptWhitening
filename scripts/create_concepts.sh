#!/usr/bin/env bash
#SBATCH --job-name=create_cub_concepts # Job name
#SBATCH --output=logs/create_cub_concepts_log_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=160gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:2

source secrets.txt
source /home/users/$NET_ID/.venv/bin/activate
python3 ../extract_CUB_concepts.py \
    --cub-path /usr/xtmp/$NET_ID/CUB_200_2011 \
    --concept-path /usr/xtmp/$NET_ID/CUB_200_2011/concepts
