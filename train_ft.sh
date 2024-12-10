#!/bin/bash
#SBATCH --partition=a
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

# python gemma_ft/fine_tuning.py
poetery run python gemma_ft/fine_tuning.py
