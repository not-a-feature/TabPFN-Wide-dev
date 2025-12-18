import pandas as pd
import numpy as np
import warnings
import wandb
import os
import torch
import matplotlib.pyplot as plt
from analysis.utils import PredictionResults
from analysis.data import get_wide_validation_datasets
from tabpfnwide.classifier import TabPFNWideClassifier
import warnings

warnings.filterwarnings("ignore")
import argparse


def main(
    dataset_name, checkpoint_paths, output_file, device="cuda:0", omics_list=None, config_path=None
):
    """
    Runs feature reduction experiments on multi-omics datasets using a specified model and checkpoints.
    Parameters:
        dataset_name (str): Name of the dataset to evaluate.
        checkpoint_paths (list of str): List of paths to model checkpoints. Use "default" for the base model.
        output_file (str): Path to the CSV file where results will be saved.
        device (str, optional): Device to run the model on (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        omics_list (list of str, optional): List of omics types to include. If None, defaults to "mRNA".
        config_path (str, optional): Path to the config.json file. Defaults to None.
    For each checkpoint and a range of feature counts, the function:
    - Loads the model and checkpoint.
    - Iterates over validation splits of the dataset with reduced features.
    - Evaluates the model, collects predictions, and computes accuracy and weighted F1 score.
    - Appends results to a CSV file, skipping experiments that have already been run.
    """

    api = wandb.Api()
    results = pd.DataFrame(
        columns=[
            "Dataset",
            "Omic",
            "Checkpoint",
            "n_features",
            "Fold",
            "Model",
            "Max_finetune",
            "Accuracy",
            "f1_weighted",
            "prediction_probas",
            "ground_truth",
        ]
    )
    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    for checkpoint_path in checkpoint_paths:
        features_per_group = 1
        n_estimators = 1

        if checkpoint_path == "default_n1g1":
            clf = TabPFNWideClassifier(
                model_name="v2.5",
                device=device,
                n_estimators=1,
                features_per_group=1,
                ignore_pretraining_limits=True,
            )
            name = "default_n1g1"
        elif checkpoint_path == "default_n8g3":
            clf = TabPFNWideClassifier(
                model_name="v2.5",
                device=device,
                n_estimators=8,
                features_per_group=3,
                ignore_pretraining_limits=True,
            )
            name = "default_n8g3"
        else:
            if config_path and os.path.exists(config_path):
                import json

                with open(config_path, "r") as f:
                    config = json.load(f)
                if "model_config" in config:
                    features_per_group = config["model_config"].get("features_per_group", 1)
                if "n_estimators" in config:
                    n_estimators = config["n_estimators"]
            else:
                try:
                    config_file = os.path.join(os.path.dirname(checkpoint_path), "config.json")
                    with open(config_file, "r") as f:
                        config = json.load(f)
                        features_per_group = config["model_config"]
                        n_estimators = config["n_estimators"]
                except:
                    pass

            clf = TabPFNWideClassifier(
                model_path=checkpoint_path,
                device=device,
                n_estimators=n_estimators,
                features_per_group=features_per_group,
                ignore_pretraining_limits=True,
            )
            name = checkpoint_path.split("/")[-1]

        for n_features in [
            200,
            500,
            2000,
            5000,
            7500,
            10000,
            15000,
            20000,
            25000,
            30000,
            40000,
            50000,
            60000,
            0,
        ]:
            exists = (
                (results["n_features"] == n_features)
                & (results["Checkpoint"] == checkpoint_path.split("/")[-1])
                & (results["Dataset"] == dataset_name)
            ).any()
            if exists:
                print(f"Skipping {dataset_name} with {n_features} features, already exists")
                continue
            print(f"Validating {dataset_name} with {n_features} features")
            clf.model.to(device)
            clf.model.eval()
            for i, dataset in enumerate(
                get_wide_validation_datasets(
                    device,
                    dataset_name=dataset_name,
                    n_splits=5,
                    n_repeats=1,
                    reduced_features=n_features,
                    omics=omics_list,
                )
            ):
                X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = dataset
                exists = (
                    (results["n_features"] == X_train_tensor.shape[-1])
                    & (results["Checkpoint"] == checkpoint_path.split("/")[-1])
                    & (results["Dataset"] == dataset_name)
                ).any()
                if exists and i == 0:
                    print(
                        f"Skipping {dataset_name} with {X_train_tensor.shape[-1]} features, already exists"
                    )
                    break

                # Convert tensors to numpy for TabPFNClassifier
                X_train = X_train_tensor.cpu().numpy()
                y_train = y_train_tensor.cpu().numpy().flatten()
                X_test = X_test_tensor.cpu().numpy()
                y_test = y_test_tensor.cpu().numpy().flatten()

                clf.fit(X_train, y_train)
                pred_probs = clf.predict_proba(X_test)

                pred_res = PredictionResults(y_test, pred_probs)
                accuracy = pred_res.get_classification_report(print_report=False)["accuracy"]
                f1_weighted = pred_res.get_f1_score(average="weighted")
                results = pd.concat(
                    [
                        results,
                        pd.DataFrame(
                            {
                                "Dataset": dataset_name,
                                "Omic": "+".join(omics_list) if omics_list else "mRNA",
                                "Checkpoint": checkpoint_path.split("/")[-1],
                                "n_features": X_train_tensor.shape[-1],
                                "Fold": i,
                                "Model": name,
                                "Accuracy": accuracy,
                                "f1_weighted": f1_weighted,
                                "prediction_probas": [
                                    " ".join(map(str, pred_res.prediction_probas))
                                ],
                                "ground_truth": [" ".join(map(str, pred_res.ground_truth))],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            clf.model.to("cpu")
            results.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        help="Path to data directory (unused but kept for compatibility)",
    )
    parser.add_argument("output_file", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--omics", dest="omics_list", type=str, nargs="+", default=["mRNA"])
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    dataset_name = args.dataset
    output_file = args.output_file

    checkpoint_paths = []
    if args.checkpoint_path:
        checkpoint_paths.append(args.checkpoint_path)
    elif args.checkpoints_dir:
        checkpoint_paths = [
            os.path.join(args.checkpoints_dir, cp)
            for cp in os.listdir(args.checkpoints_dir)
            if cp.endswith(".pt")
        ]

    if not checkpoint_paths:
        checkpoint_paths = ["default"]

    omics_list = args.omics_list
    device = args.device

    main(
        dataset_name,
        checkpoint_paths,
        output_file,
        device,
        omics_list,
        config_path=args.config_path,
    )
