#!/bin/bash
#SBATCH --job-name=IslandsController
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=16:00:00
#SBATCH --output=run_job_outputs/islands/Report_islands-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel

# This is the first argument passed to the script, which is the number of iterations
# Usage: sbatch pace_ice_island_controller.sbatch <number_of_iterations>
# Example: sbatch pace_ice_island_controller.sbatch 10
# This is a workaround for pace_ice time limits.
# Restart loop is synchronized with server.sh:
# - server.sh submits this controller for the same COUNT
# - this controller submits the next server.sh iteration
COUNT=$1
PREV_SERVER_JOB_ID=${2:-}

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"
echo "$COUNT iterations remaining"

module load cuda/12

# LLM_Storage on PACE-ICE
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Change to the repository root
cd /home/hice1/hyu462/scratch/llm-guided-evolution-infrastructure

# Starts running Island Migration with 3 islands (same LLM, three prompt groups)
uv run python islands_wrapper.py titanic_islands_run \
    --num_islands 3 \
    --llms llama3 \
    --prompt_groups "titanic/focused,titanic/general,titanic/roleplay"

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    if [[ -n "$PREV_SERVER_JOB_ID" ]]; then
        echo "Stopping previous server job: $PREV_SERVER_JOB_ID"
        scancel "$PREV_SERVER_JOB_ID"
    fi
    echo "Controller completed; launching next server iteration with count: $NEXT_COUNT"
    sbatch server.sh "$NEXT_COUNT"
fi