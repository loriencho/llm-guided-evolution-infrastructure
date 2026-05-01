Object Detection Adaptation
=======================================

Overview
--------

The LLM-guided evolution framework can be used to advance object detection,
specifically by improving upon the YOLO object detection models. YOLO model
variants are generated as individuals as LLMs edit YOLO architecture files,
which are then trained and evaluated on the COCO 2017 dataset.

The setup steps below must be followed in order. Steps 1–3 prepare the code
and environment, Steps 4–6 prepare the dataset, Steps 7–10 configure the
run parameters, and Steps 11–13 verify and launch.

----

Step 1 — Add the Ultralytics Submodule
---------------------------------------

The object detection adaptation depends on a forked copy of the
`Ultralytics <https://github.com/jasonzutty/ultralytics>`_ repository,
included as a git submodule under ``sota/ultralytics``. The submodule is
pinned to the ``evol_yolo`` branch, which extends Ultralytics with the hooks
LLM-GE needs to load mutated YOLO architecture YAMLs as evolutionary
individuals.

From the repository root:

.. code-block:: bash

   cd sota
   git submodule add https://github.com/jasonzutty/ultralytics.git
   cd ultralytics
   git checkout evol_yolo
   cd ../..

Confirm the submodule is on the correct branch:

.. code-block:: bash

   cd sota/ultralytics && git branch

The output should show ``* evol_yolo`` as the active branch.

----

Step 2 — Place the Seed Architecture YAML
------------------------------------------

The ``evol_yolo`` branch does not ship a seed architecture file. The seed
YAML, ``network_v3.yaml``, is distributed through the project Discord and
must be placed manually inside the submodule.

Create the destination directory:

.. code-block:: bash

   mkdir -p sota/ultralytics/ultralytics/cfg/models/llm

From your local machine, copy the YAML into that directory on the cluster:

.. code-block:: bash

   scp ~/Downloads/network_v3.yaml \
       <USER>@login-ice.pace.gatech.edu:~/scratch/llm-guided-evolution/sota/ultralytics/ultralytics/cfg/models/llm/

Verify on the cluster:

.. code-block:: bash

   ls sota/ultralytics/ultralytics/cfg/models/llm/

The output should list ``network_v3.yaml``.

----

Step 3 — Register the uv Workspace and Bootstrap the Environment
-----------------------------------------------------------------

**Register the submodule with uv**

Append the following block to the bottom of ``pyproject.toml`` at the
repository root:

.. code-block:: toml

   [tool.uv]
   workspace = { members = ["sota/ultralytics"] }
   environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]

The ``environments`` line restricts dependency resolution to x86_64 Linux.
This is required because Ultralytics pins conflicting NumPy versions across
macOS and aarch64 platforms.

**Redirect the uv cache to scratch space**

On PACE-ICE the home directory quota is small. Open ``run.sh`` and confirm
the following lines appear near the top, before any ``uv`` commands:

.. code-block:: bash

   export SCRATCH_DIR="/storage/ice1/<PARTITION>/<USER>/scratch.cache"
   export UV_CACHE_DIR="$SCRATCH_DIR/uv"
   export UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/uv_python"

Set ``SCRATCH_DIR`` to your actual scratch path. Without this redirect,
repeated ``uv`` syncs can exhaust the home quota and fail silently.

**Create the virtual environment**

.. code-block:: bash

   uv sync
   uv pip install -e sota/ultralytics

``uv sync`` creates ``.venv/`` at the repository root and installs all
dependencies declared in ``pyproject.toml``. The second command installs the
submodule in editable mode so that DDP training workers can import
``ultralytics``. All SLURM job templates activate this environment via
``source .venv/bin/activate``.

----

Step 4 — Download the COCO 2017 Dataset
-----------------------------------------

Run the download script from the repository root *before* launching any
training or evolutionary run:

.. code-block:: bash

   bash coco2017prep.sh

The script downloads four archives (annotation labels, train, val, and test
images) using ``curl -fL``, which fails loudly on HTTP errors and follows
redirects. Expect roughly **25 GB** of downloads and several minutes of
extraction time. The final layout is:

.. code-block:: text

   datasets/coco2017/
   ├── images/
   │   ├── train2017/       # ~118k images
   │   ├── val2017/         # 5k images
   │   └── test2017/        # 41k images
   ├── labels/              # YOLO-format .txt labels
   ├── train2017.txt
   ├── val2017.txt
   └── test-dev2017.txt

----

Step 5 — (Optional) Create a Dataset Subset
----------------------------------------------

Training on the full 118k-image training set is expensive. The script
``dataset_subset.sh`` can downsize it to a smaller subset while maintaining
class balance.

**Location:**

.. code-block:: bash

   sota/ultralytics/ultralytics/data/scripts/dataset_subset.sh

**Parameters:**

- ``--n <int>`` — number of images to include (default: ``50000``)
- ``--seed <int>`` — random seed for reproducibility (default: ``0``)
- ``--coco_root <path>`` — path to the original COCO dataset
- ``--coco_out_root <path>`` — output directory for the subset

**Output:**

Creates a new dataset directory containing:

- ``images/`` — symlinked subset of images
- ``labels/`` — corresponding labels
- ``train.txt`` — list of training images
- ``coco2017_downsized.yaml`` — dataset config for Ultralytics

Files are **symlinked, not copied**, for efficiency. The greedy
class-balancing algorithm ensures the subset remains representative of all
80 COCO classes.

If you use a subset, update ``DATA_PATH`` in ``constants.py`` and the
``path`` field in ``coco.yaml`` (Step 6) to point to the subset directory
instead of the full dataset.

----

Step 6 — Configure the Dataset YAML
--------------------------------------

Open ``sota/ultralytics/ultralytics/cfg/datasets/coco.yaml`` and set the
``path`` field to the absolute path of your COCO 2017 directory:

.. code-block:: yaml

   path: /home/hice1/<USER>/scratch/llm-guided-evolution/datasets/coco2017
   train: train2017.txt
   val: val2017.txt
   test: test-dev2017.txt

All four entries are required. Relative paths are not supported here because
Ultralytics resolves them against its own working directory rather than the
LLM-GE root.

----

Step 7 — Set Path Variables in constants.py
---------------------------------------------

Open ``src/cfg/constants.py`` and update the path variables at the top of the
file to point at your scratch directory:

- ``ROOT_DIR`` — absolute path to the LLM-GE repository root,
  e.g. ``/home/hice1/<USER>/scratch/llm-guided-evolution``
- ``DATA_PATH`` — absolute path to the COCO 2017 dataset directory,
  e.g. ``<ROOT_DIR>/datasets/coco2017/``
- ``SOTA_ROOT`` — derived from ``ROOT_DIR``; no change needed if ``ROOT_DIR``
  is correct
- ``RESULT_DIR`` — absolute path where evaluation results are written,
  e.g. ``<ROOT_DIR>/sota/ultralytics/results``
- ``SEED_NETWORK`` — derived from ``SOTA_ROOT``; no change needed
- ``LLM_GPU`` — SLURM GPU feature constraint string for LLM inference jobs.
  Set to ``H100|H200`` to match currently available PACE-ICE GPU features

----

Step 8 — Review Training Hyperparameters
------------------------------------------

Open ``sota/ultralytics/ultralytics/cfg/default.yaml``. The fields below are
**overridden directly** by ``test.py`` at runtime, so edits to them in this
file have no effect:

.. code-block:: yaml

   epochs: 25       # overridden by test.py
   batch: 128       # requires 2× H100/H200 with 224 GB host memory
   imgsz: 640
   workers: 8
   device: "0,1"    # two-GPU DDP

All other augmentation and optimiser defaults (learning rate schedule,
mosaic, HSV augmentation, etc.) are inherited from this file and can be
tuned here without touching ``test.py``.

----

Step 9 — Tune Evolution Parameters
-------------------------------------

The following variables in ``constants.py`` control the evolutionary
algorithm. Adjust them to trade off exploration breadth against wall-clock
budget:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Effect
   * - ``start_population_size``
     - 32
     - Individuals created in generation 0 from the seed. Larger values give
       the initial Pareto front more diversity but require more evaluation jobs.
   * - ``population_size``
     - 4
     - Population size in every generation after generation 0.
   * - ``num_generations``
     - 15
     - Total number of generations to run.
   * - ``crossover_probability``
     - 0.2
     - Per-individual probability of pairing with another individual for
       crossover.
   * - ``mutation_probability``
     - 0.9
     - Per-individual probability of LLM-guided mutation.
   * - ``num_elites``
     - 44
     - Number of elite individuals carried forward into the next generation.
   * - ``hof_size``
     - 100
     - Maximum size of the Hall of Fame (best individuals seen across all
       generations).
   * - ``PROB_EOT``
     - 0.25
     - Probability of applying the Evolution-of-Thought (EoT) operator, which
       conditions the LLM on top-performing individuals.

----

Step 10 — Review SLURM Job Templates
---------------------------------------

``constants.py`` contains two embedded SLURM script templates that are
rendered and submitted at runtime. These are **not** scripts you edit
directly; they are Python string templates. Verify their constraints match
what is available on PACE-ICE before launching.

**PYTHON_BASH_SCRIPT_TEMPLATE** — submitted once per candidate architecture
to run the fitness evaluation:

.. code-block:: bash

   #SBATCH --gres=gpu:2
   #SBATCH -C "H100|H200"
   #SBATCH --mem 224G
   #SBATCH -c 16

**LLM_BASH_SCRIPT_TEMPLATE** — submitted for each LLM mutation or crossover
operation:

.. code-block:: bash

   #SBATCH --gres=gpu:2
   #SBATCH -C "{}"    # filled from LLM_GPU at runtime
   #SBATCH --mem 128G
   #SBATCH -c 4
   #SBATCH --exclude=atl1-1-03-014-16-0,atl1-1-01-009-33-0,atl1-1-02-003-31-0

The ``--exclude`` list blacklists specific PACE-ICE nodes that have been
observed to cause LLM inference failures. Extend it if additional nodes
become problematic.

Both templates load ``gcc/13.3.0`` and activate ``.venv``. Conda is
deactivated with ``conda deactivate 2>/dev/null || true`` to avoid PATH
conflicts.

----

Step 11 — Verify the Installation
-----------------------------------

Before launching a full evolutionary run, confirm the evaluation pipeline
works end-to-end with the seed architecture:

.. code-block:: bash

   sbatch run_test_coco2017.sh

This job requests 2× H100 GPUs, 224 GB memory, and 16 CPU cores — matching
the evaluation template in ``constants.py``. It runs
``tests/sota/test_coco2017.py``, which calls ``sota/ultralytics/test.py``
with ``network_v3.yaml`` and prints training progress and mAP metrics to the
SLURM output file.

A successful run ends with ``job done`` in the output. Monitor progress:

.. code-block:: bash

   tail -f slurm-<JOBID>.out

----

Step 12 — Launch an Evolutionary Run
--------------------------------------

From the repository root, submit the orchestrator job:

.. code-block:: bash

   sbatch run.sh

``run.sh`` requests a **CPU-only** node — it does not need GPUs. The
orchestrator dispatches evaluation and LLM jobs dynamically via ``sbatch``
calls inside ``run_improved.py`` as the evolution proceeds. The run is named
by the positional argument passed to ``run_improved.py``; the default in
``run.sh`` is ``first_test``, which creates a checkpoint directory
``first_test/`` under the repository root.

Monitor the orchestrator output:

.. code-block:: bash

   tail -f slurm-<JOBID>.out

----

Step 13 — Resume from a Checkpoint
-------------------------------------

``run_improved.py`` writes a checkpoint after each generation. If the job is
cancelled or times out, resume it by resubmitting with the same checkpoint
name:

.. code-block:: bash

   sbatch run.sh    # run.sh passes 'first_test' as the checkpoint name

On startup, ``run_improved.py`` detects an existing checkpoint in
``first_test/`` and restores ``GLOBAL_DATA``, ``GLOBAL_DATA_ANCESTRY``, the
population, and the Hall of Fame, then continues from the last completed
generation.

The ancestry dictionary is pre-seeded with the root individual so the
lineage tree is always rooted at ``network_v3.yaml``, even on a fresh run:

.. code-block:: python

   GLOBAL_DATA_ANCESTRY = {'v3': {'GENES': ['v3'], 'MUTATE_TYPE': ['SEED']}}
