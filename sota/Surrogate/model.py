# --OPTION--
DEFAULT_SURROGATE_CONFIG = {
    # shared
    "name": "xgboost",  # mutate this to switch model family: "xgboost" or "mlp"
    "corpus_path": "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_pytorch_corrected.csv",
    "embedding_col": "codellama_python_7b_pytorch_code_exclude_helper_embedding",
    "use_pca": False,
    "pca_components": 128,

    # xgboost params
    "ss_type": "nasbench201",
    "hparams_from_file": False,
    "nthread": 4,
    "device": "cuda",
    "tree_method": "hist",

    # mlp params
    "num_layers": 3,
    "layer_width": 128,
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 200,
    "loss": "mse",
}
