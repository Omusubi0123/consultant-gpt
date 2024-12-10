#!/bin/sh
#SBATCH -p a
#SBATCH --gres=gpu:2
#SBATCH --mem=64G

conda activate consultant-gpt

python3 gemma_ft/fine_tuning.py
