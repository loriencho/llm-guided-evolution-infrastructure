SUBMODULE_PATH="sota/ultralytics"
SPARSE_PATH="ultralytics/cfg"

git submodule add https://github.com/jasonzutty/ultralytics.git  "$SUBMODULE_PATH"
cd "$SUBMODULE_PATH"
git submodule update --init --filter=blob:none
git sparse-checkout init cone
git sparse-checkout set "$SPARSE_PATH"