#!/bin/bash

#SBATCH --job-name=vllm-serve-qwen2b
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=logs/vllm_serve_qwen2b_%j.log

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."

ml purge
ml WebProxy
ml CUDA/12.9.0
ml GCC/14.3.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm_env

MODEL_ID="models/Qwen3.5-2B"
NUM_GPUS_PER_NODE=2 

echo "Starting vLLM Server on Head Node..."
srun --overlap --nodes=1 --ntasks=1 \
    vllm serve "$MODEL_ID" \
    --served-model-name qwen3.5 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$NUM_GPUS_PER_NODE" \
    --max-model-len 262144 \
    --reasoning-parser qwen3 &

wait