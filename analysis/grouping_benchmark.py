import argparse
import os
import sys
import warnings

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from openml import tasks
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from tabpfn import TabPFNClassifier
from tabpfn.model_loading import load_model_criterion_config
from tabpfnwide.patches import fit
from tabpfnwide.utils import PredictionResults

# Apply patch
setattr(TabPFNClassifier, "fit", fit)

def plot_results(results_file, output_plot):
    """Generate comparison plot of performance across grouping settings."""
    assert os.path.exists(results_file), f"Results file not found: {results_file}"
    df = pd.read_csv(results_file)
    
    if df.empty:
        print(f"No results to plot in {results_file}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["accuracy", "f1_weighted", "roc_auc_score"]
    
    for idx, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
            
        # Aggregate across folds for each task
        agg_df = df.groupby(["task_id", "features_per_group"])[metric].mean().reset_index()
        
        sns.boxplot(data=agg_df, x="features_per_group", y=metric, ax=axes[idx])
        axes[idx].set_title(f"{metric.replace('_', ' ').title()}")
        axes[idx].set_xlabel("Features Per Group")
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_plot}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if "features_per_group" in df.columns:
        summary = df.groupby("features_per_group")[metrics].agg(["mean", "std"])
        print(summary)


def evaluate_task(
    task_id, 
    grouping, 
    model, 
    device, 
    openml_task, 
    res_df, 
    duplicate_features=1
):
    """
    Evaluates a single task with specific grouping and feature duplication settings.
    """
    dataset = openml_task.get_dataset()
    X, y, _, _ = dataset.get_data(target=openml_task.target_name)
    
    # Ensure X is numpy array for consistent manipulation
    if hasattr(X, "values"):
        X = X.values
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    X = np.asarray(X)
    
    # Handle feature duplication
    # e.g. [f1, f1, f2, f2] corresponds to groups [f1, f1], [f2, f2].
    if duplicate_features > 1:
        X = np.repeat(X, duplicate_features, axis=1)
    
    # Assertions for validity
    assert len(X) == len(y), "X and y must have same number of instances"
    assert duplicate_features >= 1, "Duplicate features must be >= 1"

    num_features = X.shape[1]
    num_instances = X.shape[0]
    num_classes = len(np.unique(y))
    dataset_name = dataset.name

    print(f"\nTask: {task_id} (Dataset: {dataset_name}) - Grouping: {grouping}, Dup: {duplicate_features}")
    print(f"  Features: {num_features}, Instances: {num_instances}, Classes: {num_classes}")

    # Shuffle
    X, y = shuffle(X, y, random_state=42)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    clf = TabPFNClassifier(
        device=device, n_estimators=1, ignore_pretraining_limits=True
    )
    
    skf = RepeatedStratifiedKFold(
        n_splits=3, n_repeats=3, random_state=42 
    )

    model.features_per_group = grouping

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf.fit(X_train, y_train, model=model)
        pred_probs = clf.predict_proba(X_test)
        
        # Verify predictions shape
        assert pred_probs.shape[0] == len(y_test), f"Predictions shape mismatch: {pred_probs.shape} vs {len(y_test)}"

        pred_res = PredictionResults(y_test, pred_probs)
        report = pred_res.get_classification_report(print_report=False)
        accuracy = report["accuracy"]
        f1_weighted = pred_res.get_f1_score(average="weighted")
        
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
        
        new_row = pd.DataFrame(
            {
                "task_id": [task_id],
                "task_name": [dataset_name],
                "num_features": [num_features],
                "num_instances": [num_instances],
                "num_classes": [num_classes],
                "fold": [fold],
                "features_per_group": [grouping],
                "duplicate_factor": [duplicate_features],
                "accuracy": [accuracy],
                "f1_weighted": [f1_weighted],
                "roc_auc_score": [roc_auc],
            }
        )
        res_df = pd.concat([res_df, new_row], ignore_index=True)
        
    return res_df


def main(
    suite_id,
    output_file,
    max_features=100,
    min_features=0,
    max_instances=2000,
    grouping_values=[1, 2, 3],
    device="cuda:0",
    generate_plot=True,
    duplication_output_file=None,
):
    """
    Benchmark TabPFN base model.
    """
    # Fetch OpenML suite
    suite = openml.study.get_suite(suite_id=suite_id)
    openml_df = tasks.list_tasks(output_format="dataframe", task_id=suite.tasks)
    
    assert not openml_df.empty, f"No tasks found for suite {suite_id}"

    if "task_type" in openml_df.columns:
        openml_df = openml_df[openml_df["task_type"] == "Supervised Classification"]

    openml_df = openml_df[
        (openml_df["NumberOfFeatures"] >= min_features) &
        (openml_df["NumberOfFeatures"] <= max_features) &
        (openml_df["NumberOfInstances"] <= max_instances) &
        (openml_df["NumberOfClasses"] < 10) &
        (openml_df["NumberOfClasses"] > 1)
    ]

    print(f"Found {len(openml_df)} tasks to process in suite {suite_id}")
    
    if len(openml_df) == 0:
        print("No tasks meet the criteria. Exiting.")
        return

    # Load model once
    models, _, _, _ = load_model_criterion_config(
        model_path=None,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2.5",
        download_if_not_exists=True,
    )
    model = models[0]

    # --- Feature Duplication Benchmark ---
    print(f"Running Feature Duplication Benchmark")
    
    dup_output = duplication_output_file or output_file.replace(".csv", "_duplication.csv")
    
    dup_columns = [
        "task_id", "task_name", "num_features", "num_instances", "num_classes",
        "fold", "features_per_group", "duplicate_factor", 
        "accuracy", "f1_weighted", "roc_auc_score"
    ]
    
    if os.path.exists(dup_output):
        dup_df = pd.read_csv(dup_output)
    else:
        dup_df = pd.DataFrame(columns=dup_columns)

    duplication_factors = [1, 2, 3] 
    
    for dup in duplication_factors:
        grouping = dup
        print(f"\nTraining with Duplication={dup}, Grouping={grouping}")
        
        for task_id in openml_df["tid"].values:
            task_id = int(task_id)
            if not dup_df.empty and ((dup_df["task_id"] == task_id) & (dup_df["duplicate_factor"] == dup)).any():
               continue

            # Scientific code: Crash early on error. Do not try/except broadly.
            t = openml.tasks.get_task(task_id)
            dup_df = evaluate_task(
                task_id, grouping, model, device, t, dup_df, duplicate_features=dup
            )
            
            os.makedirs(os.path.dirname(os.path.abspath(dup_output)) or ".", exist_ok=True)
            dup_df.to_csv(dup_output, index=False)
    
    print(f"Duplication results saved to {dup_output}")

    # --- Standard Grouping Benchmark ---
    print(f"\nRunning Standard Grouping Benchmark")
    print(f"Testing features_per_group values: {grouping_values}")
    
    res_columns = [
        "task_id", "task_name", "num_features", "num_instances", "num_classes",
        "fold", "features_per_group", "duplicate_factor",
        "accuracy", "f1_weighted", "roc_auc_score"
    ]
    
    if os.path.exists(output_file):
        res_df = pd.read_csv(output_file)
        if "duplicate_factor" not in res_df.columns:
            res_df["duplicate_factor"] = 1
    else:
        res_df = pd.DataFrame(columns=res_columns)

    for grouping in grouping_values:
        print(f"\nProcessing Grouping: {grouping}")
        for task_id in openml_df["tid"].values:
            task_id = int(task_id)
            if not res_df.empty and ((res_df["task_id"] == task_id) & (res_df["features_per_group"] == grouping)).any():
                continue
            
            t = openml.tasks.get_task(task_id)
            res_df = evaluate_task(
                task_id, grouping, model, device, t, res_df, duplicate_features=1
            )
            
            os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
            res_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")
    if generate_plot and not res_df.empty:
        plot_output = output_file.replace(".csv", "_plot.png")
        plot_results(output_file, plot_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_id", type=int, help="OpenML suite ID")
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--min_features", type=int, default=0)
    parser.add_argument("--max_instances", type=int, default=10000)
    parser.add_argument("--output_file", type=str, default="analysis_results/grouping_benchmark_results.csv")
    parser.add_argument("--grouping_values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--duplication_output_file", type=str, default="analysis_results/duplication_benchmark_results.csv")
    
    args = parser.parse_args()
    
    main(
        suite_id=args.suite_id,
        output_file=args.output_file,
        max_features=args.max_features,
        min_features=args.min_features,
        max_instances=args.max_instances,
        grouping_values=args.grouping_values,
        device=args.device,
        generate_plot=not args.no_plot,
        duplication_output_file=args.duplication_output_file
    )
