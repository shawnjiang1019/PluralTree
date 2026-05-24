#!/bin/bash
#SBATCH --job-name=pluraltree
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=def-enaskt
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load python/3.11 gcc cuda/12.2
source ~/envs/pluraltree/bin/activate

cd /home/shawnj/projects/def-enaskt/shawnj/PluralTree
python scripts/train.py --device cuda --n_epochs 100 --d_hidden 128
