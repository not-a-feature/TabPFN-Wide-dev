import torch
import os
import dataclasses
from tabpfn import TabPFNClassifier
from tabpfn.model.loading import load_model_criterion_config
from tabpfnwide.patches import fit as patched_fit
from tabpfn.base import determine_precision
from tabpfn.utils import infer_random_state, infer_devices, update_encoder_params


class TabPFNWideClassifier(TabPFNClassifier):
    def __init__(
        self,
        model_name="",
        model_path="",
        device="cuda",
        n_estimators=1,
        features_per_group=3,
        save_attention_maps=False,
        **kwargs,
    ):

        # Check arguments
        if (model_name and model_path) or (not model_name and not model_path):
            raise ValueError("Either model_name or model_path must be specified, but not both.")

        if model_name:
            valid_models = ["v2", "wide-v2-1.5k-nocat", "wide-v2-5k-nocat", "wide-v2-8k-nocat"]
            if model_name not in valid_models:
                raise ValueError(
                    f"Model name {model_name} not recognized. Choose from {valid_models}"
                )
            if model_name != "v2":
                model_path = os.path.join(f"models/tabpfn-{model_name}.pt")

        if model_name != "v2" and not os.path.isfile(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")

        if save_attention_maps and (n_estimators != 1 or features_per_group != 1):
            raise ValueError(
                "save_attention_maps can only be True when n_estimators=1 and features_per_group=1"
            )

        # Initialize parent TabPFNClassifier
        # We pass ignore_pretraining_limits=True by default, but allow override
        if "ignore_pretraining_limits" not in kwargs:
            kwargs["ignore_pretraining_limits"] = True

        super().__init__(
            device=device,
            n_estimators=n_estimators,
            model_path=model_path,
            **kwargs,
        )

        # Restore model_path after super().__init__ overwrites it with default "auto"
        self.model_path = model_path
        self.model_name = model_name
        self.features_per_group = features_per_group
        self.n_estimators = n_estimators
        self.save_attention_maps = save_attention_maps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model()

    def _initialize_model_variables(self):
        # We already loaded the model in __init__
        self.models_ = [self.model]

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

    def _load_model(self):
        # Load the base model structure
        models, _, configs, inference_config = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2",
            download_if_not_exists=True,
        )
        model = models[0]

        self.configs_ = configs
        self.inference_config_ = inference_config

        if self.model_name != "v2":
            # Manually override features_per_group to match the training behavior
            # This is crucial because we are loading weights trained with grouping=1
            # into a base model initialized with grouping=2
            if hasattr(self, "features_per_group"):
                 model.features_per_group = self.features_per_group
                 self.configs_[0].features_per_group = self.features_per_group

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Handle DDP-wrapped checkpoints
            # TODO fix this during training / saving
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict)

        return model

    def fit(self, X, y):
        if self.save_attention_maps:
            for layer in self.model.transformer_encoder.layers:
                if hasattr(layer, "self_attn_between_features"):
                    layer.self_attn_between_features.save_att_map = True
                    layer.self_attn_between_features.number_of_samples = X.shape[0]
                    layer.self_attn_between_features.attention_map = None

        # Use the patched fit function, passing the loaded model
        return patched_fit(self, X, y, model=self.model)

    def get_attention_maps(self):
        maps = []
        for layer in self.model.transformer_encoder.layers:
            if hasattr(layer, "self_attn_between_features"):
                attn = getattr(layer.self_attn_between_features, "attention_map", None)
                if attn is not None:
                    maps.append(attn.numpy())
        return maps
