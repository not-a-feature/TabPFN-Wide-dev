import torch
import os
from tabpfn import TabPFNClassifier
from tabpfn.model.loading import load_model_criterion_config
from tabpfnwide.patches import fit as patched_fit


class TabPFNWideClassifier(TabPFNClassifier):
    def __init__(self, model_name="TabPFN-Wide-8k", model_path="./models", device=None, **kwargs):
        # Initialize parent TabPFNClassifier
        # We pass ignore_pretraining_limits=True by default as in the example, but allow override
        if "ignore_pretraining_limits" not in kwargs:
            kwargs["ignore_pretraining_limits"] = True

        super().__init__(device=device, **kwargs)

        self.model_name = model_name
        self.model_path = model_path
        # Ensure device is set (TabPFNClassifier might set it, but we need it for loading)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        valid_models = ["TabPFN-Wide-1.5k", "TabPFN-Wide-5k", "TabPFN-Wide-8k", "TabPFNv2.5"]
        assert (
            model_name in valid_models
        ), f"Model name {model_name} not recognized. Choose from {valid_models}"

        self.wide_model = self._load_wide_model()

    def _load_wide_model(self):
        # Load the base model structure
        model, _, _ = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2.5",
            download=True,
        )

        if self.model_name != "TabPFNv2.5":
            model.features_per_group = 1
            checkpoint_path = os.path.join(self.model_path, f"{self.model_name}_submission.pt")

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found at {checkpoint_path}. Please ensure the model is downloaded."
                )

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint)

        return model

    def fit(self, X, y):
        # Use the patched fit function, passing the loaded model
        return patched_fit(self, X, y, model=self.wide_model)
