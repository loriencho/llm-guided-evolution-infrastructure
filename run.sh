#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 0-08:00   # Runtime in D-HH:MM
# request a GPU on the following list
#SBATCH -C "A100-40GB|A100-80GB|H100|H200|V100-16GB|V100-32GB|RTX6000|A40|L40S|NVIDIAA10080GBPCIe"

echo "launching LLM Guided Evolution"
hostname

module load uv

export SERVER_HOSTNAME=$(hostname)

echo "INFO: Setting up the environment..."

save_dir=${0:-titanic_test}

uv run python run_improved.py ${save_dir)
