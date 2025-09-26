#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 5-00:00
#SBATCH --gres=gpu:2
#SBATCH -C "NVIDIAA10080GBPCIe|NVIDIAA100-SXM4-80GB|NVIDIAH100NVL"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH -w ice193

echo "launching LLM Server"

hostname

module load cuda/12.2.2
module load uv

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1
export MKL_THREADING_LAYER=GNU

SCRIPT_DIR=$(dirname "$0")
pushd $SCRIPT_DIR > /dev/null # go to script dir
source ../.venv/bin/activate  # activate uv environment
popd > /dev/null  # return to original working directory

export SERVER_HOSTNAME=$(hostname)

# get hostname file
pushd $SCRIPT_DIR/.. > /dev/null # go to script dir
HOSTNAME_FILE=$(pwd)"/hostname.log"
popd > /dev/null  # return to original working directory

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting LLM server on host: $SERVER_HOSTNAME"

uvicorn server:app --host $SERVER_HOSTNAME --port 8000 --workers 1
