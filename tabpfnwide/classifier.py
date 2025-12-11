import torch
import os
from tabpfn import TabPFNClassifier
from tabpfn.model.loading import load_model_criterion_config
from tabpfnwide.patches import fit as patched_fit
from tabpfn.base import determine_precision
from tabpfn.utils import infer_random_state, infer_devices, update_encoder_params


class TabPFNWideClassifier(TabPFNClassifier):
    def __init__(
        self,
        model_name="v2.5-Wide-1.5k",
        model_path="",
        device="cuda",
        features_per_group=1,
        n_estimators=1,
        **kwargs,
    ):

        # Check arguments
        if (model_name and model_path) or (not model_name and not model_path):
            raise ValueError("Either model_name or model_path must be specified, but not both.")

        if model_name:
            valid_models = ["v2.5-Wide-1.5k", "v2.5-Wide-5k", "v2.5-Wide-8k", "v2.5"]
            if model_name not in valid_models:
                raise ValueError(
                    f"Model name {model_name} not recognized. Choose from {valid_models}"
                )
            if model_name != "v2.5":
                # TODO FIX LOCAL PATH
                model_path = os.path.join(f"TODO FIX LOCAL PATH{model_name}.pt")

        if model_name != "v2.5" and not os.path.isfile(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")

        self.model_path = model_path
        self.model_name = model_name
        self.features_per_group = features_per_group
        self.n_estimators = n_estimators
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize parent TabPFNClassifier
        # We pass ignore_pretraining_limits=True by default as in the example, but allow override
        if "ignore_pretraining_limits" not in kwargs:
            kwargs["ignore_pretraining_limits"] = True

        kwargs["n_estimators"] = n_estimators
        # kwargs["features_per_group"] = features_per_group

        super().__init__(device=device, **kwargs)
        self.wide_model = self._load_wide_model()

    def _initialize_model_variables(self):
        # We already loaded the model in __init__
        self.models_ = [self.wide_model]

        static_seed, rng = infer_random_state(self.random_state)

        self.devices_ = infer_devices(self.device)

        (
            self.use_autocast_,
            self.forced_inference_dtype_,
            byte_size,
        ) = determine_precision(self.inference_precision, self.devices_)

        # Handle inference config override
        if hasattr(self, "inference_config_"):
            self.inference_config_ = self.inference_config_.override_with_user_input(
                user_config=self.inference_config
            )

            outlier_removal_std = self.inference_config_.OUTLIER_REMOVAL_STD
            if outlier_removal_std == "auto":
                outlier_removal_std = (
                    self.inference_config_._CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD
                )
        else:
            # Fallback if inference_config_ was not set (should not happen with updated _load_wide_model)
            outlier_removal_std = None

        update_encoder_params(
            models=self.models_,
            remove_outliers_std=outlier_removal_std,
            seed=static_seed,
            differentiable_input=self.differentiable_input,
        )

        return byte_size, rng

    def _load_wide_model(self):
        # Load the base model structure
        models, _, configs, inference_config = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2.5",
            download_if_not_exists=self.model_name == "v2.5",
        )
        model = models[0]

        self.configs_ = configs
        self.inference_config_ = inference_config

        if self.model_name != "v2.5":
            model.features_per_group = self.features_per_group
            model.n_estimators = self.n_estimators
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Handle DDP-wrapped checkpoints
            # TODO fix this during training / saving
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Unwrap DDP prefix if present
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)

        return model

    def fit(self, X, y):
        # Use the patched fit function, passing the loaded model
        return patched_fit(self, X, y, model=self.wide_model)
