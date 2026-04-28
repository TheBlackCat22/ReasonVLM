#!/bin/bash

#SBATCH --job-name=vllm-serve-qwen4b
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=logs/vllm_serve_qwen4b_%j.log

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."

ml purge
ml WebProxy
ml CUDA/12.9.0
ml GCC/14.3.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm_env

MODEL_ID="models/Qwen3.5-4B"

echo "Starting vLLM Server..."
vllm serve "$MODEL_ID" \
    --served-model-name qwen3.5 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --reasoning-parser qwen3 &
VLLM_PID=$!

export NO_PROXY='localhost,127.0.0.1,0.0.0.0'
while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
    if [ "$STATUS" -eq 200 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') vLLM server is online!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died unexpectedly! Exiting job..."
        exit 1
    fi
    sleep 10
done

python src/eval.py --method=$1

exit 0