set -eu 

SUBMODULE_PATH="sota/ultralytics"

git rm -f "$SUBMODULE_PATH" --sparse
rm -rf ".git/modules/$SUBMODULE_PATH"