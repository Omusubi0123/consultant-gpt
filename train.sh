#!/bin/bash
#SBATCH --partition=p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH


which nvcc

echo $LD_LIBRARY_PATH
ls /usr/local/
nvcc -v
nvidia-smi


ls /usr/local/lib/
ls /usr/local/lib/

pyenv version
pyenv install 3.10.6

poetry install

poetry show torch
poetry show transformers
poetry show trl

poetry run python gemma_ft/fine_tuning.py