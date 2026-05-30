#!/bin/bash
#SBATCH --job-name=pluraltree
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load python/3.11 gcc cuda/13.2 arrow/24.0.0
source ~/envs/pluraltree/bin/activate

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
python scripts/train.py --device cuda --n_epochs 300 --d_hidden 128 --warmup1 400 --warmup2 1600 --embed_model all-mpnet-base-v2
