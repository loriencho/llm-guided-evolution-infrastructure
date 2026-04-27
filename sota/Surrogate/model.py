# --OPTION--
import time

from naslib.predictors.mlp import MLPPredictor
from naslib.predictors.trees.xgb import XGBoost
from naslib.predictors.trees.lgb import LGBoost
from naslib.predictors.trees.ngb import NGBoost
from naslib.predictors.trees.random_forest import RandomForestPredictor


DEFAULT_SURROGATE_CONFIG = {
    "name": "xgboost",
    "embedding_col": "codellama_python_7b_pytorch_code_exclude_helper_embedding",
    "use_pca": False,
    "pca_components": 128,
    "ss_type": "nasbench201",
    "hparams_from_file": False,
    # xgboost-specific
    "nthread": 4,
    "device": "cuda",
    "tree_method": "hist",
    # mlp-specific
    "num_layers": 3,
    "layer_width": 128,
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 200,
    "loss": "mse",
    # lgboost-specific
    "lgb_num_leaves": 31,
    "lgb_learning_rate": 0.05,
    "lgb_feature_fraction": 0.9,
    "lgb_min_data_in_leaf": 5,
    # ngboost-specific
    "ngb_n_estimators": 505,
    "ngb_learning_rate": 0.08,
    "ngb_max_depth": 6,
    "ngb_max_features": 0.79,
    # random_forest-specific
    "rf_n_estimators": 116,
    "rf_max_features": 0.17,
    "rf_min_samples_leaf": 2,
    "rf_min_samples_split": 2,
    "rf_bootstrap": False,
}


def get_surrogate_config(base_cfg=None, **runtime_constants):
    cfg = DEFAULT_SURROGATE_CONFIG.copy() if base_cfg is None else base_cfg.copy()
    cfg.update({key: value for key, value in runtime_constants.items() if value is not None})
    return cfg


# --OPTION--
class CustomXGBoost(XGBoost):
    def __init__(self, **kwargs):
        base_valid_args = [
            "encoding_type",
            "ss_type",
            "zc",
            "zc_only",
            "hpo_wrapper",
            "hparams_from_file",
        ]
        base_args = {k: v for k, v in kwargs.items() if k in base_valid_args}
        self.custom_hyperparams = {k: v for k, v in kwargs.items() if k not in base_valid_args}

        super().__init__(**base_args)

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        print(f"[CustomXGBoost] Hyperparams set: {self.hyperparams}")

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        start = time.time()
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        result = super().fit(xtrain, ytrain, train_info, params, **kwargs)

        print(f"[CustomXGBoost] Training completed in {time.time() - start:.2f} seconds.")
        return result


# --OPTION--
class CustomMLP(MLPPredictor):
    def __init__(self, **kwargs):
        base_valid_args = [
            "encoding_type",
            "ss_type",
            "zc",
            "zc_only",
            "hpo_wrapper",
            "hparams_from_file",
            "config",
        ]
        base_args = {k: v for k, v in kwargs.items() if k in base_valid_args}
        self.custom_hyperparams = {k: v for k, v in kwargs.items() if k not in base_valid_args}

        super().__init__(**base_args)

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        print(f"[CustomMLP] Hyperparams set: {self.hyperparams}")

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        start = time.time()
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        result = super().fit(
            xtrain,
            ytrain,
            train_info=train_info,
            epochs=self.hyperparams["epochs"],
            loss=self.hyperparams["loss"],
            **kwargs,
        )
        print(f"[CustomMLP] Training completed in {time.time() - start:.2f} seconds.")
        return result


# --OPTION--
class CustomLGBoost(LGBoost):
    def __init__(self, **kwargs):
        base_valid_args = [
            "encoding_type",
            "ss_type",
            "zc",
            "zc_only",
            "hpo_wrapper",
            "hparams_from_file",
        ]
        base_args = {k: v for k, v in kwargs.items() if k in base_valid_args}
        self.custom_hyperparams = {k: v for k, v in kwargs.items() if k not in base_valid_args}

        super().__init__(**base_args)

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        print(f"[CustomLGBoost] Hyperparams set: {self.hyperparams}")

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        start = time.time()
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        result = super().fit(xtrain, ytrain, train_info, params, **kwargs)

        print(f"[CustomLGBoost] Training completed in {time.time() - start:.2f} seconds.")
        return result


# --OPTION--
_NGBoost_KEY_MAP = {
    "n_estimators": "param:n_estimators",
    "learning_rate": "param:learning_rate",
    "max_depth": "base:max_depth",
    "max_features": "base:max_features",
}


class CustomNGBoost(NGBoost):
    def __init__(self, **kwargs):
        base_valid_args = [
            "encoding_type",
            "ss_type",
            "zc",
            "zc_only",
            "hpo_wrapper",
            "hparams_from_file",
        ]
        base_args = {k: v for k, v in kwargs.items() if k in base_valid_args}
        raw_custom = {k: v for k, v in kwargs.items() if k not in base_valid_args}
        # Remap clean names to NGBoost's internal param:/base: key format
        self.custom_hyperparams = {_NGBoost_KEY_MAP.get(k, k): v for k, v in raw_custom.items()}

        super().__init__(**base_args)

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        print(f"[CustomNGBoost] Hyperparams set: {self.hyperparams}")

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        start = time.time()
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        result = super().fit(xtrain, ytrain, train_info, params, **kwargs)

        print(f"[CustomNGBoost] Training completed in {time.time() - start:.2f} seconds.")
        return result


# --OPTION--
class CustomRandomForest(RandomForestPredictor):
    def __init__(self, **kwargs):
        base_valid_args = [
            "encoding_type",
            "ss_type",
            "zc",
            "zc_only",
            "hpo_wrapper",
            "hparams_from_file",
        ]
        base_args = {k: v for k, v in kwargs.items() if k in base_valid_args}
        self.custom_hyperparams = {k: v for k, v in kwargs.items() if k not in base_valid_args}

        super().__init__(**base_args)

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        print(f"[CustomRandomForest] Hyperparams set: {self.hyperparams}")

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        start = time.time()
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        result = super().fit(xtrain, ytrain, train_info, params, **kwargs)

        print(f"[CustomRandomForest] Training completed in {time.time() - start:.2f} seconds.")
        return result


# --OPTION--
SURROGATE_REGISTRY = {
    "xgboost": CustomXGBoost,
    "mlp": CustomMLP,
    "lgboost": CustomLGBoost,
    "ngboost": CustomNGBoost,
    "random_forest": CustomRandomForest,
}


def build_predictor_kwargs(cfg):
    name = cfg["name"]
    if name not in SURROGATE_REGISTRY:
        raise ValueError(f"Unknown surrogate: {name}")
    if "corpus_path" not in cfg:
        raise ValueError("Missing required runtime config: corpus_path")

    predictor_kwargs = {
        "base_predictor_cls": SURROGATE_REGISTRY[name],
        "corpus_path": cfg["corpus_path"],
        "embedding_col": cfg["embedding_col"],
        "use_pca": cfg["use_pca"],
        "pca_components": cfg["pca_components"],
    }

    if name == "xgboost":
        predictor_kwargs.update(
            {
                "ss_type": cfg["ss_type"],
                "hparams_from_file": cfg["hparams_from_file"],
                "nthread": cfg["nthread"],
                "device": cfg["device"],
                "tree_method": cfg["tree_method"],
            }
        )
    elif name == "mlp":
        predictor_kwargs.update(
            {
                "num_layers": cfg["num_layers"],
                "layer_width": cfg["layer_width"],
                "batch_size": cfg["batch_size"],
                "lr": cfg["lr"],
                "epochs": cfg["epochs"],
                "loss": cfg["loss"],
            }
        )
    elif name == "lgboost":
        predictor_kwargs.update(
            {
                "ss_type": cfg["ss_type"],
                "hparams_from_file": cfg["hparams_from_file"],
                "num_leaves": cfg["lgb_num_leaves"],
                "learning_rate": cfg["lgb_learning_rate"],
                "feature_fraction": cfg["lgb_feature_fraction"],
                "min_data_in_leaf": cfg["lgb_min_data_in_leaf"],
            }
        )
    elif name == "ngboost":
        predictor_kwargs.update(
            {
                "ss_type": cfg["ss_type"],
                "hparams_from_file": cfg["hparams_from_file"],
                "n_estimators": cfg["ngb_n_estimators"],
                "learning_rate": cfg["ngb_learning_rate"],
                "max_depth": cfg["ngb_max_depth"],
                "max_features": cfg["ngb_max_features"],
            }
        )
    elif name == "random_forest":
        predictor_kwargs.update(
            {
                "ss_type": cfg["ss_type"],
                "hparams_from_file": cfg["hparams_from_file"],
                "n_estimators": cfg["rf_n_estimators"],
                "max_features": cfg["rf_max_features"],
                "min_samples_leaf": cfg["rf_min_samples_leaf"],
                "min_samples_split": cfg["rf_min_samples_split"],
                "bootstrap": cfg["rf_bootstrap"],
            }
        )

    return predictor_kwargs
