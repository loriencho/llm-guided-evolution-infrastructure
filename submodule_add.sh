SUBMODULE_PATH="sota/ultralytics"
SPARSE_PATH="ultralytics/cfg"
SUBMODULE_REPO="https://github.com/jasonzutty/ultralytics.git"

git submodule add "$SUBMODULE_REPO" "$SUBMODULE_PATH"
cd "$SUBMODULE_PATH"
git submodule update --init --filter=blob:none
git sparse-checkout init cone
git sparse-checkout set "$SPARSE_PATH"
