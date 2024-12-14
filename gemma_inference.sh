#!/bin/bash
#SBATCH --partition=p
#SBATCH --gres=gpu:1

poetry run python gemma_ft/inference.py --model_ft_date=20241212
