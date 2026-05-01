#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 16:00:00                 # Maximum walltime allowed for CPU jobs
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --exclusive                     # Request all available memory on the node (as per your documentation)
#SBATCH -c 24                       # Request 48 threads (all 24 cores x 2 CPUs with hyperthreading for Gold 6226)

export SCRATCH_DIR="/home/hice1/user/scratch/your_directory_here"

# Force uv to use scratch for everything
export UV_CACHE_DIR="$SCRATCH_DIR/your_uv_cache_dir"

# All of the above settings are configured for running on the PACE-ICE HPC

echo "launching LLM-Guided-Evolution"
hostname
module load cuda/12.6.1
# module load anaconda3 # <--- COMMENT THIS OUT OR REMOVE IT if it's adding a competing Python to PATH

# Verify the path and environment immediately before running your script
echo "--- DEBUGGING PYTHON ENVIRONMENT ---"
which python
echo "--- END DEBUGGING ---"

# Set LD_LIBRARY_PATH if needed (double-check the path, it looks slightly off for a site-packages lib)
# Ensure this path exists and is correct for your 'llm_guided_env' (note: your current active env is 'llm_env')
# export LD_LIBRARY_PATH="/home/hice1/yzhang3942/.conda/envs/llm_env/lib:$LD_LIBRARY_PATH" # Corrected path for typical lib

# Now, execute your script using the explicit Python executable
source .venv/bin/activate
uv run run_improved.py first_test