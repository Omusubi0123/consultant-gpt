#!/bin/sh
#SBATCH -p a
#SBATCH --gres=gpu:2
#SBATCH --mem=64G

conda activate consultant-gpt

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH

python3 -m pip list

python3 gemma_ft/fine_tuning.py
