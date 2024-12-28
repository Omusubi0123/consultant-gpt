#!/bin/bash
#SBATCH -p a
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source ~/.bashrc
. ~/miniconda3/etc/profile.d/conda.sh
conda activate consultant-gpt

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH

pip install -e .
pip list

python gemma_ft/chkp_to_hf.py
