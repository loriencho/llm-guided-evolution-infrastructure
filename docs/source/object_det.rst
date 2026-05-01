Object Detection Adaptation
=======================================

Overview
--------
The LLM-guided evolution framework can be used to advance object detection, specifically by improving upon the YOLO object detection models. YOLO model variants can be generated as individuals as LLMs edit YOLO architecture files, which are then trained and evaluated on the COCO2017 dataset.

Setup
-----
 
Ultralytics Submodule
~~~~~~~~~~~~~~~~~~~~~
 
    The object detection adaptation depends on a forked copy of the `Ultralytics <https://github.com/jasonzutty/ultralytics>`_ repository, included in the project as a git submodule under ``sota/ultralytics``. The submodule is pinned to the ``evol_yolo`` branch, which extends Ultralytics with the hooks LLM-GE needs to load mutated YOLO architecture YAMLs as evolutionary individuals.
 
**Adding the Submodule**
 
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
 
**Seed Architecture YAML**
 
    The ``evol_yolo`` branch does not ship a seed architecture file. The seed YAML, ``network_v3.yaml``, is distributed through the project Discord and must be placed manually inside the submodule.
 
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
 
**uv Workspace Registration**
 
    The submodule must be registered as a member of the project's ``uv`` workspace so its dependencies resolve alongside the rest of the project. Append the following block to the bottom of ``pyproject.toml`` at the repository root:
 
    .. code-block:: toml
 
       [tool.uv]
       workspace = { members = ["sota/ultralytics"] }
       environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
 
    The ``environments`` line restricts dependency resolution to x86_64 Linux. This restriction is required because Ultralytics pins conflicting NumPy versions across macOS and aarch64 platforms.
 
    Workspace membership alone does not expose ``ultralytics`` as an importable package to distributed training workers. Install the submodule in editable mode so that DDP processes can import it:
 
    .. code-block:: bash
 
       uv pip install -e sota/ultralytics
 
Constants and Path Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
    Several files in the repository reference absolute paths and SLURM constraints that depend on the user and the current state of the PACE-ICE cluster. These must be updated before launching any object detection run.
 
**constants.py**
 
    Located at ``src/cfg/constants.py``. The following variables must be set to point at your scratch directory and the submodule paths above:
 
    * ``ROOT_DIR`` — absolute path to the LLM-GE repository root, e.g. ``/home/hice1/<USER>/scratch/llm-guided-evolution``
    * ``DATA_PATH`` — absolute path to the COCO 2017 dataset directory, e.g. ``<ROOT_DIR>/datasets/coco2017/``
    * ``SOTA_ROOT`` — absolute path to the LLM models directory inside the submodule, ``<ROOT_DIR>/sota/ultralytics/ultralytics/cfg/models/llm``
    * ``RESULT_DIR`` — absolute path where evaluation results are written, e.g. ``<ROOT_DIR>/sota/ultralytics/results``
    * ``SEED_NETWORK`` — full path to the seed architecture YAML, ``<SOTA_ROOT>/network_v3.yaml``
    * ``LLM_GPU`` — SLURM GPU feature constraint string. Set to ``nvidia-gpu|H100|H200`` to match currently available PACE-ICE GPU features.
 
    The SLURM job templates defined further down in ``constants.py`` (``LLM_BASH_SCRIPT_TEMPLATE`` and the evaluation template) must use a ``-C "H100|H200"`` constraint and a ``--gres=gpu:2`` resource request to match available PACE-ICE node configurations.
 
**run.sh**
 
    The shell launcher's SLURM directives must match the constraints set in ``constants.py``:
 
    .. code-block:: bash
 
       #SBATCH -C "H100|H200"
       #SBATCH --gres=gpu:2

dataset_subset.sh
~~~~~~~~~~~~~~~~~

   **Location:**

   .. code-block:: bash

      sota/ultralytics/ultralytics/data/scripts/dataset_subset.sh

   This script downsizes the COCO 2017 training dataset (100,000+ images) to a smaller subset of size ``n``, while maintaining class balance so the data remains representative of the original.

   **Parameters:**

   - ``--n <int>``  
     Number of images to include in the subset (default: ``50000``)

   - ``--seed <int>``  
     Random seed for reproducibility (default: ``0``)

   - ``--coco_root <path>``  
     Path to the original COCO dataset (images + labels in YOLO format)

   - ``--coco_out_root <path>``  
     Output directory where the subset will be created

   **Output:**

   Creates a new dataset directory containing:

   - ``images/`` — symlinked subset of images  
   - ``labels/`` — corresponding labels  
   - ``train.txt`` — list of training images  
   - ``coco2017_downsized.yaml`` — dataset config for Ultralytics  

   **Notes:**

   - Uses a **greedy class-balancing algorithm** to preserve class diversity  
   - Files are **symlinked (not copied)** for efficiency    
 
**coco.yaml**
 
    Located at ``sota/ultralytics/ultralytics/cfg/datasets/coco.yaml`` (or wherever the dataset config is referenced from ``constants.py``). The ``path`` field must point to the absolute COCO 2017 directory in your scratch space:
 
    .. code-block:: yaml
 
       path: /home/hice1/<USER>/scratch/llm-guided-evolution/datasets/coco2017
       train: train2017.txt
       val: val2017.txt
       test: test-dev2017.txt
 
    All four entries are required; relative paths are not supported here because Ultralytics resolves them against its own working directory rather than the LLM-GE root.
