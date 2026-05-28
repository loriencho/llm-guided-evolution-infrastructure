#!/bin/bash
#SBATCH --job-name=test_coco2017
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:2
#SBATCH -G 2
#SBATCH -C "H100|H200" # For yolo training
#SBATCH --mem 224G	# prevent OOM errors for loading COCO2017
#SBATCH -c 16 # 8 workers per GPU

echo "launching test_coco2017"
hostname

module load cuda/12.6.1

conda deactivate
source .venv/bin/activate

echo "--- DEBUGGING PYTHON ENVIRONMENT ---"
which python
echo "--- END DEBUGGING ---"

uv run tests/sota/test_coco2017.py