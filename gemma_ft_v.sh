#!/bin/bash
#SBATCH --partition=v
#SBATCH --gres=gpu:2
#SBATCH --mem=32G

poetry run python gemma_ft/fine_tuning.py
