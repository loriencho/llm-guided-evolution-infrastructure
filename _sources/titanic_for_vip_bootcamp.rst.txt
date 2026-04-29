LLM-Guided Evolution on Titanic Dataset with PACE-ICE
=====================================================

Overview
--------

So far in this bootcamp, you’ve implemented baseline **Scikit-learn (or other)** models and a **MOGP approach** for the Titanic dataset.  
Now, you’re in the **final step** — working with **LLM-Guided Evolution (LLM-GE)** on **PACE-ICE**, Georgia Tech’s high-performance computing cluster.  

This stage focuses on running your evolutionary LLM experiments efficiently, monitoring results, and analyzing your Pareto frontiers comparing FPR/FNR (or FP/FN) tradeoffs against your previous methods.

---

Understanding PACE-ICE and the Cluster Environment
--------------------------------------------------

**PACE-ICE** is Georgia Tech’s interactive computing **cluster** — a group of powerful connected machines (called **nodes**) that share resources like GPUs and memory.  
You’ll use it to run experiments that would be too slow or heavy for your laptop.

Basic structure:
- **Login node:** the entry point — for setup only.  
- **Compute node:** where your code actually runs.  
- **SLURM scheduler:** manages job queues and assigns resources across users.

In short: you connect → request compute → submit jobs → SLURM runs them when resources are available.

> If you’ve never used Georgia Tech VPN before, download the client `here <https://vpn.gatech.edu/global-protect/getsoftwarepage.esp>`_.  
> Once installed, set the **portal address** to `vpn.gatech.edu`, log in with Duo 2FA — and you’re connected.  
> This step is **essential** to access PACE-ICE.

For step-by-step PACE-ICE login and onboarding instructions (accounts, Duo, connecting to nodes) see Georgia Tech's KB article: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042100

---

Step 1 — Connect to PACE-ICE
----------------------------

In your terminal:

.. code-block:: bash

    ssh your_gtid@login-ice.pace.gatech.edu
    srun -G 1 --pty bash

`ssh` connects you to the cluster’s login node.

`srun` moves you to a compute node (where the real work happens).

The `-G 1` flag requests one GPU for your session.

You’ll know you’re on a compute node if your terminal prompt looks like `atl1-1-03-013-8-0.pace.gatech.edu`.

Step 2 — Repository Setup
-------------------------

Your scratch directory already exists on PACE. It’s your large-quota workspace for all experiments.

Clone the repository (HTTPS is simplest):

.. code-block:: bash

    cd ~/scratch
    git clone https://github.com/jasonzutty/llm-guided-evolution-fork.git
    cd llm-guided-evolution-fork
    git fetch --all
    git checkout MosesTheRedSea-main
    git pull

Always work inside `~/scratch` — your home directory has limited space.

Step 3 — Setting Up the Python Environment
------------------------------------------

The project uses `uv`, a fast tool for managing Python environments.

To install it:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

Then, inside your repo folder, sync dependencies (this installs all needed packages):

.. code-block:: bash

    uv sync --cache-dir ~/scratch/.uv

This builds an isolated Python environment stored under your scratch space.

Step 4 — Setting Up Kaggle Access
---------------------------------

The Titanic dataset is downloaded via the Kaggle API.
You’ll need your Kaggle credentials (`kaggle.json`).

From your laptop:

.. code-block:: bash

    scp /path/to/kaggle.json your_gtid@login-ice.pace.gatech.edu:~/.config/kaggle/kaggle.json

On the cluster:

.. code-block:: bash

    mkdir -p ~/.config/kaggle
    chmod 600 ~/.config/kaggle/kaggle.json

This ensures your credentials are securely stored.

Step 5 — Configure Before Running
---------------------------------

Before you run anything, make these first changes inside `~/scratch/llm-guided-evolution-fork/src/cfg/constants_titanic.py`:

- Edit `preprocess.py` in `sota/Titanic/` so that the data matches how your team prepared it for previous parts.
- Your preprocessing should produce the same features and target columns as before.

- Choose a unique `PORT` number and set it in `constants_titanic.py`.
  - This prevents multiple teams or users from overwriting each other’s runs.

- If your team decides to, adjust `num_generations`, `population_size`, and similar parameters to fit your team’s plan.

Once configured, prepare your data and check the pipeline:

.. code-block:: bash

    cd sota/Titanic
    uv run ./pull_data.sh
    uv run preprocess.py
    uv run eval.py

Confirm that `eval.py` prints the expected output (e.g., FPR/FNR or counts).

Step 6 — Generate Job Scripts
-----------------------------

From the project root:

.. code-block:: bash

    cd ~/scratch/llm-guided-evolution-fork
    uv run slurm.py

This creates:

- `server.sh` — starts the LLM inference server
- `run.sh` — runs the LLM-GE evolution process

Step 7 — Submit and Monitor Jobs
--------------------------------

Submit:

.. code-block:: bash

    sbatch server.sh
    sbatch run.sh

Monitor:

.. code-block:: bash

    squeue -u $USER
    tail -f server_*.out
    tail -f run_*.out

Cancel by job ID or class name:

.. code-block:: bash

    scancel <jobid>
    scancel -n llm_oper     # Example: cancel specific class of jobs
    scancel -n llm_opt

Job states:

- `PD` → Pending (waiting)
- `R` → Running
- `CG` → Completing

Step 8 — Chaining Jobs for Continuous Runs
------------------------------------------

Each SLURM job runs for about 8 hours. To automate longer runs, you can chain jobs sequentially, one after another:

.. code-block:: bash

    sbatch server.sh
    # Suppose this prints: Submitted batch job 3376242
    sbatch -d afterany:3376242 server.sh
    sbatch -d afterany:3376243 server.sh

You should always chain each job off the one before it, not all off the same starting job.
This ensures they run one at a time in order without overlap.

You can apply the same idea to your `run.sh` jobs.

Note on dependency flags: `-d after:<jobid>` makes the next job start only if the specified job finishes successfully; `-d afterany:<jobid>` starts the next job regardless of the previous job's exit status. Use `afterany:` if you want resilience to failures or timeouts.

You can also chain your run jobs so they automatically start once your server goes up — that way you don’t have to babysit it. For example:

.. code-block:: bash

    sbatch server.sh
    # Suppose this prints: Submitted batch job 3376242
    sbatch -d after:3376242 run.sh

Step 9 — Understanding and Analyzing Outputs
--------------------------------------------

After jobs complete, you’ll see several output files:

+-----------------------+------------------------------------------+------------------------------------------------------------------+
| File                  | Location                                 | Description                                                      |
+=======================+==========================================+==================================================================+
| `server_*.out`        | Project root                             | Logs from the inference server (node, port, errors)              |
+-----------------------+------------------------------------------+------------------------------------------------------------------+
| `run_*.out`           | Project root                             | Logs showing LLM prompts, responses, and model generations       |
+-----------------------+------------------------------------------+------------------------------------------------------------------+
| `*_results.txt`       | `/results`                               | Each contains model metrics — FPR/FNR or counts depending on your|
|                       |                                          | `eval.py`                                                        |
+-----------------------+------------------------------------------+------------------------------------------------------------------+
| `checkpoints/`        | `/titanic_test/checkpoints/`             | Pickled (.pkl) population snapshots per generation               |
+-----------------------+------------------------------------------+------------------------------------------------------------------+
| `model_*.py`          | Run folders                              | Generated model code from LLM-GE                                 |
+-----------------------+------------------------------------------+------------------------------------------------------------------+
| `hostname.log`        | Root                                     | Node name and connection log for debugging                       |
+-----------------------+------------------------------------------+------------------------------------------------------------------+

Make sure your `eval.py` outputs are in the same format (rates or counts) as your earlier Scikit/MOGP work so you can make clean comparisons.

### What to Do with These Outputs

Your job now is to make sense of these results.
Develop your own scripts and approaches to:

- Aggregate and summarize results across generations
- Visualize Pareto frontiers (FPR vs FNR)
- Show changes and improvements over time
- Automate progress updates while runs are still ongoing

You’re encouraged to design creative visualization and aggregation methods — this analysis is a key part of your engineering insight.

Step 10 — Deliverables
----------------------

At a minimum, your presentation and report should:

- Summarize and compare results across Scikit, MOGP, and LLM-GE
- Plot Pareto frontiers for each approach
- Discuss tradeoffs and observed trends

However, as emerging engineers, you’re expected to go deeper —
analyze why certain tradeoffs appeared, what drove model improvements, and what the LLM’s behavior revealed about guided evolution.

Present both your findings and your approach in your midterm presentation.

Notes and Best Practices
------------------------

- Always run on a compute node, not the login node.
- Choose unique ports to avoid server collisions.
- Be patient — PACE job queues can take time.
- Focus on understanding outputs more than re-running setups.

Good luck!
For any issues or debugging questions, the Discord server is the best resource for real-time help and collaboration.
