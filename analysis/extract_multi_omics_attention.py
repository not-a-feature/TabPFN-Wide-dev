import numpy as np
import warnings
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from analysis.load_mm_data import load_multiomics
from tabpfnwide.patches import fit
from tabpfn.model_loading import load_model_criterion_config
from tabpfn import TabPFNClassifier

setattr(TabPFNClassifier, "fit", fit)
import warnings

warnings.filterwarnings("ignore")
import argparse


def main(
    dataset_name, output_file, checkpoint_path, device="cuda:0", omic="mrna", config_path=None
):
    """
    Extracts and saves attention maps from a trained transformer-based model on multi-omics data.
    Parameters:
        dataset_name (str): Name of the multi-omics dataset to use.
        output_file (str): Path to save the extracted attention maps (as a torch file).
        checkpoint_path (str): Path to the model checkpoint to load weights from.
        device (str, optional): Device to run the model on (default: "cuda:0").
        omic (str, optional): Omics data type to use from the dataset (default: "mrna").
        config_path (str, optional): Path to the config.json file. Defaults to None.
    Description:
        - Loads multi-omics data and encodes labels.
        - Loads a TabPFN-Wide model.
        - Configures the model to save attention maps during inference.
        - Runs inference to obtain predictions and attention maps.
        - Saves the extracted attention maps to the specified output file.
    """

    ds_dict, labels = load_multiomics(dataset_name)
    mrna = ds_dict[omic]
    X, y = mrna.values, labels
    y = LabelEncoder().fit_transform(y)
    print(X.shape)

    models, _, _, _ = load_model_criterion_config(
        model_path=None,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2.5",
        download_if_not_exists=True,
    )
    model = models[0]

    if config_path and os.path.exists(config_path):
        import json

        with open(config_path, "r") as f:
            config = json.load(f)
        if "model_config" in config:
            model.features_per_group = config["model_config"].get("features_per_group", 1)
    else:
        model.features_per_group = 1

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle DDP-wrapped checkpoints
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Unwrap DDP prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configure model for attention extraction
    for layer in model.transformer_encoder.layers:
        layer.self_attn_between_features.save_att_map = True
        layer.self_attn_between_features.number_of_samples = X_train.shape[0]

    clf = TabPFNClassifier(device=device, n_estimators=1, ignore_pretraining_limits=True)
    clf.fit(X_train, y_train, model=model)
    clf.predict_proba(X_test)

    atts = [
        getattr(layer.get_submodule("self_attn_between_features"), "attention_map")
        for layer in model.transformer_encoder.layers
    ]
    atts = torch.stack(atts, dim=0)
    torch.save(atts, f"{output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        help="Path to data directory (unused but kept for compatibility)",
    )
    parser.add_argument("output_file", type=str, help="Path to save the attention maps")
    parser.add_argument(
        "--dataset", dest="dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--omic", type=str, default="mrna", help="Omic type to use (default: mRNA)")

    args = parser.parse_args()
    main(
        args.dataset_name,
        args.output_file,
        args.checkpoint_path,
        args.device,
        args.omic,
        config_path=args.config_path,
    )
