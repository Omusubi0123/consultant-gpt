#!/bin/bash
#SBATCH --partition=v
#SBATCH --gres=gpu:2
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

poetry run python gemma_ft/fine_tuning.py
