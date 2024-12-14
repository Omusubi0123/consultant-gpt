#!/bin/bash
#SBATCH --partition=a
#SBATCH --gres=gpu:1

# ローカルユーザーディレクトリにライブラリをインストール
mkdir -p ~/local/lib
wget https://github.com/libffi/libffi/releases/download/v3.4.4/libffi-3.4.4.tar.gz
tar -xzvf libffi-3.4.4.tar.gz
cd libffi-3.4.4
./configure --prefix=$HOME/local
make
make install

# インストールしたライブラリのパスを追加
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/local/lib64:$LD_LIBRARY_PATH

# ctypes用の設定
export PYTHONPATH=$HOME/local/lib/python3.10/site-packages:$PYTHONPATH


# CUDA が見つからない場合の対処
# CUDA のダウンロードとローカルインストール
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.47.03_linux.run
chmod +x cuda_11.6.0_510.47.03_linux.run
./cuda_11.6.0_510.47.03_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-11.6

# パスの設定
export CUDA_HOME=$HOME/cuda-11.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


# # 仮想環境の再作成
# poetry env remove --all
# poetry env use python3.10

# # 依存関係の再インストール
# poetry install

# # 必要に応じて、以下の環境変数も追加
# poetry run pip install --upgrade pip
# poetry run pip install torch --extra-index-url https://download.pytorch.org/whl/cu116

# ファインチューニングの実行
poetry run python gemma_ft/fine_tuning.py