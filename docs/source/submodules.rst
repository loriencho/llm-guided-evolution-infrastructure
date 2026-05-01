Using GitHub Submodules
###########################

Submodules allow you to keep a Git repository as a subdirectory of another Git repository. This lets you clone another repository into your project and keep your commits separate.
For LLM-GE, submodules can be utilized with git sparse checkout in order to easily include or switch out models within the sota folder. Git sparse checkout reduces your working tree to a subset of tracked files, and in this case, we use it to only track files within a repository's root directory and sota folder. 

For ease of use, submodule add and remove scripts have been written so they can be run by any user.

submodule_add.sh
--------
To use submodule_add.sh, we have three variables. SUBMODULE_PATH is the path where our submodule will be stored within the repository. As stated before, our suggested use case is to place models within the sota folder. SPARSE_PATH is the path within your repository that you will be using as a submodule that you want to extract files from. Note that with sparse checkout, files from the root directory will also be included in your submodule. SUBMODULE_REPO is the repository URL to add as a GitHub submodule.


submodule_remove.sh
--------
The submodule_remove script removes a submodule from your repostiory. Simply update the SUBMODULE_PATH variable, which is the path relative to your root directory from your parent repository to the submodule repository.

Using with uv
--------
If using uv, we recommend adding this to your pyproject.toml file in your parent directory. This will ensure that your submodule dependencies can be properly synced with your parent repository and that your submodule can be treated as a package. The name of your package in tool.uv.sources must match the name of your project in for the pyproject.toml file inside your GitHub submodule.

   .. code-block:: bash

        [tool.uv]
        workspace = { members = ["sota/submodule_folder"] }

        [tool.uv.sources]
        submodule_project_name = { workspace = true}
