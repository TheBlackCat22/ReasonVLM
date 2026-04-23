#!/bin/bash

#SBATCH --job-name=vllm-serve-qwen9b
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=logs/vllm_serve_qwen9b_%j.log

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."

ml purge
ml WebProxy
ml CUDA/12.9.0
ml GCC/14.3.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm_env

cleanup() {
    echo "Caught termination signal. Cleaning up Ray cluster..."
    srun --overlap --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES ray stop -f > /dev/null 2>&1
    echo "Cleanup complete."
}
trap cleanup EXIT SIGTERM SIGINT

MODEL_ID="models/Qwen3.5-9B"
NUM_GPUS_PER_NODE=2
NUM_NODES=$SLURM_JOB_NUM_NODES
RAY_PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

head_node_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
if [[ "$head_node_ip" == "" ]]; then
    head_node_ip=$head_node
fi

cleanup() {
    echo "Caught termination signal. Cleaning up Ray cluster..."
    srun --overlap --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES ray stop -f > /dev/null 2>&1
    echo "Cleanup complete."
}
trap cleanup EXIT SIGTERM SIGINT

export ip_head=$head_node_ip:$RAY_PORT
export RAY_ADDRESS="ray://${ip_head}"

echo "Starting Ray Head Node on $head_node ($ip_head)"
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
    --node-ip-address="$head_node_ip" \
    --port=$RAY_PORT \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus="${NUM_GPUS_PER_NODE}" \
    --block &

sleep 30

worker_num=$((NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray Worker Node on $node_i"
    srun --overlap --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus="${NUM_GPUS_PER_NODE}" \
        --block &
done

echo "Waiting for Ray workers to connect..."
sleep 60

ray status

echo "Starting vLLM Server on Head Node..."
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    vllm serve "$MODEL_ID" \
    --served-model-name qwen3.5 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$NUM_GPUS_PER_NODE" \
    --pipeline-parallel-size "$NUM_NODES" \
    --max-model-len 262144 \
    --reasoning-parser qwen3 &

wait