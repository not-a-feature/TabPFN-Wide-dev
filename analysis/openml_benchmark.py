import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import torch
import openml
from openml import tasks
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from analysis.utils import PredictionResults
from tabpfnwide.classifier import TabPFNWideClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
import argparse
import json


def main(
    suite_id,
    output_file,
    max_features=500,
    min_features=0,
    max_instances=10000,
    checkpoints=[],
    config_path=None,
    device="cuda:0",
):
    """
    Runs classification experiments on a suite of OpenML tasks, evaluating different model checkpoints and the default model (TabPFNv2.5).
    Parameters:
        suite_id (int): The OpenML suite ID specifying the collection of tasks to evaluate.
        output_file (str): Path to the CSV file where results will be saved or appended.
        max_features (int, optional): Maximum number of features allowed in a dataset. Defaults to 500.
        min_features (int, optional): Minimum number of features required in a dataset. Defaults to 0.
        max_instances (int, optional): Maximum number of instances allowed in a dataset. Defaults to 10,000.
        checkpoints (list, optional): List of checkpoint file paths to evaluate. Use "default" for the base model. Defaults to [].
        config_path (str, optional): Path to the config.json file. Defaults to None.
        device (str, optional): Device identifier for model computation (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
    Description:
        - Filters OpenML tasks based on classification type, feature/instance/class count
        - Evaluates TabPFNClassifier on each task using repeated stratified k-fold cross-validation.
        - Results are saved to the specified output CSV file
    """
    suite = openml.study.get_suite(suite_id=suite_id)
    openml_df = tasks.list_tasks(output_format="dataframe", task_id=suite.tasks)

    if "task_type" in openml_df.columns:
        openml_df = openml_df[openml_df["task_type"] == "Supervised Classification"]

    openml_df = openml_df[openml_df["NumberOfFeatures"] >= min_features]
    openml_df = openml_df[openml_df["NumberOfFeatures"] <= max_features]
    openml_df = openml_df[openml_df["NumberOfInstances"] <= max_instances]
    # TabPFN is not suitable for datasets with more than 10 classes
    openml_df = openml_df[openml_df["NumberOfClasses"] < 10]
    openml_df = openml_df[openml_df["NumberOfClasses"] > 1]

    for checkpoint_path in checkpoints:
        print(f"Initializing model from {checkpoint_path}")
        if checkpoint_path == "stock":
            clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
            clf.device = device
        elif checkpoint_path == "stock_2.5":
            clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True, model_path="v2")
        else:
            config_file = (
                config_path
                if config_path
                else os.path.join(os.path.dirname(checkpoint_path), "config.json")
            )
            with open(config_file, "r") as f:
                config = json.load(f)
                features_per_group = config["model_config"]["features_per_group"]
                n_estimators = config["train_config"]["n_estimators"]

            clf = TabPFNWideClassifier(
                model_path=checkpoint_path,
                device=device,
                n_estimators=n_estimators,
                features_per_group=features_per_group,
                ignore_pretraining_limits=True,
                save_attention_maps=False,
                save_attention_maps=False,
            )


        res_df = pd.DataFrame(
            columns=[
                "task_id",
                "fold",
                "checkpoint",
                "accuracy",
                "roc_auc_score",
                "prediction_probas",
                "ground_truth",
            ]
        )
        if os.path.exists(output_file):
            res_df = pd.read_csv(output_file)

        for task_id in openml_df["tid"].values:

            if (
                (res_df["task_id"] == task_id)
                & (res_df["checkpoint"] == checkpoint_path.split("/")[-1])
            ).any():
                print(f"Skipping task ID {task_id}, already processed")
                continue
            try:
                task = openml.tasks.get_task(int(task_id))
                dataset = task.get_dataset()
                X, y, _, _ = dataset.get_data(target=task.target_name)
                X, y = shuffle(X, y, random_state=42)
                X = X.values
                le = LabelEncoder()
                y = le.fit_transform(y)

                skf = RepeatedStratifiedKFold(
                    n_splits=3, n_repeats=10 if X.shape[0] < 2500 else 3, random_state=42
                )

                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    clf.fit(X_train, y_train)
                    pred_probs = clf.predict_proba(X_test)

                    pred_res = PredictionResults(y_test, pred_probs)
                    accuracy = pred_res.get_classification_report(print_report=False)["accuracy"]
                    if pred_probs.shape[-1] == 2:
                        roc_auc = roc_auc_score(
                            pred_res.ground_truth, pred_res.prediction_probas[:, 1]
                        )
                    else:
                        roc_auc = roc_auc_score(
                            pred_res.ground_truth,
                            pred_res.prediction_probas,
                            multi_class="ovr",
                            average="macro",
                            labels=np.arange(pred_probs.shape[-1]),
                        )
                    res_df = pd.concat(
                        [
                            res_df,
                            pd.DataFrame(
                                {
                                    "task_id": task_id,
                                    "fold": fold,
                                    "checkpoint": checkpoint_path.split("/")[-1],
                                    "accuracy": accuracy,
                                    "roc_auc_score": roc_auc,
                                    "prediction_probas": [
                                        " ".join(map(str, pred_res.prediction_probas))
                                    ],
                                    "ground_truth": [" ".join(map(str, pred_res.ground_truth))],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                print(f"Task ID {task_id} processed successfully.")
                res_df.to_csv(output_file, index=False)

            except Exception as e:
                print(f"Error with ID {task_id}: {e.with_traceback()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str)
    parser.add_argument("--suite_id", type=int, required=True, help="OpenML suite ID to process")
    parser.add_argument(
        "--max_features", type=int, default=50000, help="Maximum number of features to consider"
    )
    parser.add_argument(
        "--min_features", type=int, default=500, help="Minimum number of features to consider"
    )
    parser.add_argument(
        "--max_instances", type=int, default=10000, help="Maximum number of instances to consider"
    )
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")


    args = parser.parse_args()

    checkpoints = []
    if args.checkpoint_path:
        checkpoints.append(args.checkpoint_path)
    elif args.checkpoint_dir:
        checkpoints = [
            os.path.join(args.checkpoint_dir, f)
            for f in os.listdir(args.checkpoint_dir)
            if f.endswith(".pt")
        ]

    if not checkpoints:
        checkpoints = ["v2"]

    main(
        suite_id=args.suite_id,
        output_file=args.output_file,
        max_features=args.max_features,
        min_features=args.min_features,
        max_instances=args.max_instances,
        checkpoints=checkpoints,
        config_path=args.config_path,
        device=args.device,
    )
