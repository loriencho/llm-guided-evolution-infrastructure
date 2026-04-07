#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/slurm-%j.out

COUNT=$1
PREV_SERVER_JOB_ID=${2:-}

echo "launching LLM Guided Evolution"
hostname
module load uv

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    if [[ -n "$PREV_SERVER_JOB_ID" ]]; then
        echo "Stopping previous server job: $PREV_SERVER_JOB_ID"
        scancel "$PREV_SERVER_JOB_ID"
    fi
    echo "Run completed; launching next server iteration with count: $NEXT_COUNT"
    sbatch server.sh "$NEXT_COUNT"
fi

export SERVER_HOSTNAME=$(hostname)
uv run python run_improved.py titanic_test
