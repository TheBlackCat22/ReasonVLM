#!/bin/bash

#SBATCH --job-name=InstallVLLM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-4:00:00
#SBATCH --output=logs/%j.log

ml purge
ml WebProxy
ml CUDA/12.9.0
ml GCC/14.3.0

conda create -n vllm_env python=3.12 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm_env

mkdir -p ./tmp
export TMPDIR=$(pwd)/tmp
export TEMP=$(pwd)/tmp
export TMP=$(pwd)/tmp

MAX_JOBS=4 pip install vllm==0.19.1
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
pip install -r requirements.txt

hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B

mkdir logs
rm -rf ./tmp