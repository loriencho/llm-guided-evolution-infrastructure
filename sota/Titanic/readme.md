# README For VIP Bootcamp Students
## Working on PACE-ICE
LLM-GE is resource intensive when running with local LLMS. There are ways to configure LLM-GE to run with API calls, but for this readme, we are going to use local models.
You will want to begin by logging on to PACE-ICE. You can do this by following the instructions [Here](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042100)

Once logged into PACE-ICE, you will land on a login node. 
> [!IMPORTANT]
> You should not be doing any work on a login node, as soon as you land on one you should put yourself on a remote session on a compute node. 
> For this guide, we recommend requesting a GPU for this session so that you can ensure you set up a proper environment. You can do so with `srun -G 1 --pty bash`

We recommend installing a tool called uv for Python package management. Instructions for the install can be found [Here](https://docs.astral.sh/uv/getting-started/installation/). But on this system, you can simply type `curl -LsSf https://astral.sh/uv/install.sh | sh`

Once uv is installed, you can set up your Python environment from within your LLM-GE directory by running
`uv sync --cache-dir ~/scratch/.uv`

> [!NOTE]
> When running on PACE-ICE you have a limited quote in your home directory, which is why we specify using the cache folder on the scratch partition, where your quota is much larger

## Preparing Titanic Dataset
We don't want to check in data into our repository, so instead we check in scripts to pull the data. We have prepared a script to download the titanic dataset here, but you will have to first set up an API key with kaggle to utilize it. If you don't want to go through this, you can simply drop the train.csv file into sota/Titanic/data. Otherwise, please follow the instructions [Here](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials)

After following the instructions, from the sota/Titanic directory, run the command `uv run ./pull_data.sh`

