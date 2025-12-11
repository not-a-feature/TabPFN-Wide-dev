import numpy as np
import torch
from analysis.utils import get_new_features
from tabpfnwide.patches import fit
from tabpfn.model_loading import load_model_criterion_config
from tabpfn import TabPFNClassifier

setattr(TabPFNClassifier, "fit", fit)
import matplotlib.pyplot as plt
import openml
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import argparse
import pickle


def main(device, openml_id, checkpoint_path, output, config_path=None):
    """
    Runs an analysis pipeline to evaluate attention patterns in a transformer-based model on tabular data with added noise and sparsity.

    Parameters:
        device (str): The device to run computations on (e.g., 'cpu' or 'cuda').
        openml_id (int): The OpenML dataset ID to load.
        checkpoint_path (str): Path to the model checkpoint file.
        output (str): Path to save the output pickle file.
        config_path (str, optional): Path to the config.json file. Defaults to None.
    Description:
        - Loads the specified OpenML dataset and preprocesses it (encoding, shuffling).
        - Adds new features using either feature smearing or needle-in-a-haystack approach.
        - Saves attention maps from the model during inference to an output pickle file.
    """
    dataset = openml.datasets.get_dataset(openml_id)
    X, y, categorical_indicator, _ = dataset.get_data(target=dataset.default_target_attribute)
    X, y = shuffle(X, y, random_state=42)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = X[X.columns[np.array(categorical_indicator) == False]]
    X = torch.tensor(X.values, dtype=torch.float32)
    X_new_noise = get_new_features(
        X, features_to_be_added=2000 - X.shape[-1], sparsity=0, noise_std=1, include_original=False
    )
    X_new_sparse = get_new_features(
        X, features_to_be_added=2000, sparsity=0.02, noise_std=1, include_original=False
    )

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

    permutation = None
    attentions_to_last_column = {}

    for X_new, name in zip([X_new_noise, X_new_sparse], ["noise", "sparse"]):
        if name == "noise":
            X_new = torch.cat((X, X_new), dim=1)
            permuted_indices = np.random.permutation(X_new.shape[1])
            permutation = permuted_indices
            X_new = X_new[:, permuted_indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y, test_size=0.2, random_state=42
        )

        # Convert to numpy for TabPFNClassifier
        X_train_np = X_train.cpu().numpy()
        y_train_np = y_train
        X_test_np = X_test.cpu().numpy()

        for layer in model.transformer_encoder.layers:
            layer.self_attn_between_features.attention_map = None
            layer.self_attn_between_features.save_att_map = True
            layer.self_attn_between_features.number_of_samples = X_train.shape[0]

        clf = TabPFNClassifier(device=device, n_estimators=1, ignore_pretraining_limits=True)
        clf.fit(X_train_np, y_train_np, model=model)
        pred_probs = clf.predict_proba(X_test_np)

        try:
            print("Accuracy:", (pred_probs.argmax(axis=-1) == y_test).mean())
        except Exception as e:
            print(e)

        atts = [
            getattr(layer.get_submodule("self_attn_between_features"), "attention_map")
            for layer in model.transformer_encoder.layers
        ]
        atts = torch.stack(atts, dim=0)
        att_to_last_column = atts.mean(dim=0)[-1, :-1]

        attentions_to_last_column[name] = att_to_last_column.cpu().numpy()

    with open(output, "wb") as f:
        pickle.dump([attentions_to_last_column, permutation, X_new_noise, X_new_sparse], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Output pickle file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for torch")
    parser.add_argument("--openml_id", type=int, default=1494, help="OpenML dataset ID")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()
    main(
        args.device, args.openml_id, args.checkpoint_path, args.output, config_path=args.config_path
    )
