import argparse
import os
import pickle
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

from tabpfn import TabPFNClassifier
from tabpfn.model_loading import load_model_criterion_config
from tabpfnwide.patches import fit
from tabpfnwide.utils import PredictionResults

# Apply patch
setattr(TabPFNClassifier, "fit", fit)

# Filter warnings
warnings.filterwarnings("ignore")

# Column definitions
RESULT_COLUMNS = [
    "task_id",
    "task_name",
    "num_features",
    "num_instances",
    "num_classes",
    "fold",
    "features_per_group",
    "duplicate_factor",
    "accuracy",
    "f1_weighted",
    "roc_auc_score",
    "mask_injected",
    "analysis_type",
]


def shuffle_arrays(*arrays, random_state=None):
    """Shuffle arrays in unison using a deterministic RNG."""
    if not arrays:
        return arrays

    length = arrays[0].shape[0]
    for arr in arrays:
        assert len(arr) == length, "All arrays must have the same length"

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(length)
    return tuple(arr[perm] for arr in arrays)


def print_all_results(df):
    """Print the consolidated benchmarks and a grouped summary.

    Args:
        df: Concatenated results from all enabled analyses.
    """

    if df.empty:
        print("No benchmark results to display.")
        return

    sort_cols = [
        col
        for col in [
            "analysis_type",
            "task_id",
            "features_per_group",
            "duplicate_factor",
            "mask_injected",
        ]
        if col in df.columns
    ]
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    print("\n=== Complete Benchmark Results ===")
    print(df_sorted.to_string(index=False))

    metrics = ["accuracy", "f1_weighted", "roc_auc_score"]
    available_metrics = [m for m in metrics if m in df.columns]
    if available_metrics:
        summary = (
            df_sorted.groupby(
                ["analysis_type", "features_per_group", "duplicate_factor", "mask_injected"]
            )[available_metrics]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        print("\n=== Aggregated Summary by Combination ===")
        print(summary.to_string(index=False))


def get_combo_label(row):
    analysis = row.get("analysis_type", "analysis").capitalize()
    mask = "mask" if row.get("mask_injected", False) else "no-mask"
    return f"{analysis} | grp={row['features_per_group']} | dup={row['duplicate_factor']} | {mask}"


def plot_combined_results(df, output_plot):
    """Create a box plot that compares each combination across all metrics."""

    if df.empty:
        print("No results available for plotting.")
        return

    plot_df = df.copy()
    plot_df["duplicate_factor"] = plot_df.get("duplicate_factor", 1).fillna(1).astype(int)
    plot_df["mask_injected"] = plot_df.get("mask_injected", False).astype(bool)
    plot_df["features_per_group"] = plot_df.get("features_per_group", 1).fillna(1).astype(int)

    plot_df["combo_label"] = plot_df.apply(get_combo_label, axis=1)

    # Determine order
    unique_labels = sorted(plot_df["combo_label"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(unique_labels) * 0.45), 6))
    sns.boxplot(
        data=plot_df, x="combo_label", y="accuracy", ax=ax, palette="Set2", order=unique_labels
    )
    ax.set_title("Accuracy Distribution per Combination")
    ax.set_xlabel("Combination")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined plot saved to {output_plot}")


def register_embedding_hook(model, embeddings_list):
    """
    Register a forward hook on the input encoder to capture embeddings.

    Args:
        model: The TabPFN model
        embeddings_list: List to store captured embeddings

    Returns:
        Hook handle (for removal later)
    """

    def hook_fn(module, input, output):
        embeddings_list.append(output.detach().cpu().clone())

    # Hook into the encoder output
    # The encoder is model.encoder, which is a SequentialEncoder
    # We want to capture the output after the LinearInputEncoderStep
    assert hasattr(model, "encoder"), "Model must have encoder attribute"

    handle = model.encoder.register_forward_hook(hook_fn)
    return handle


def extract_embeddings_with_model(model, X_train, y_train, X_test, clf, device):
    """
    Extract embeddings from the model using a forward hook.

    Returns:
        embeddings: numpy array of embeddings from test set
        predictions: prediction probabilities
    """
    embeddings_list = []
    hook_handle = register_embedding_hook(model, embeddings_list)

    try:
        # Run forward pass
        clf.fit(X_train, y_train, model=model)
        pred_probs = clf.predict_proba(X_test)

        # Extract embeddings
        # The hook captures embeddings in the format from encoder output
        # We need the test portion
        if len(embeddings_list) > 0:
            # Get the last captured embeddings
            emb = embeddings_list[-1]
            # emb shape: (seq_len, batch*features, emsize)
            # We need to extract test portion and reshape appropriately
            embeddings = emb.numpy()
        else:
            embeddings = None

    finally:
        hook_handle.remove()

    return embeddings, pred_probs


def analyze_embeddings(embeddings_dict, output_dir, all_labels_order=None):
    """
    Analyze and visualize embeddings from different methods.

    Args:
        embeddings_dict: Dict mapping method name -> embeddings array
        output_dir: Directory to save visualizations
        all_labels_order: Optional list of all possible labels to ensure consistent coloring
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for UMAP
    all_embeddings = []
    all_labels = []

    for method_name, emb_array in embeddings_dict.items():
        if emb_array is None or len(emb_array) == 0:
            continue

        # Flatten to 2D if needed
        if len(emb_array.shape) > 2:
            # Take mean over sequence dimension
            emb_flat = emb_array.mean(axis=0)
        else:
            emb_flat = emb_array

        # Flatten further if needed
        if len(emb_flat.shape) > 2:
            emb_flat = emb_flat.reshape(emb_flat.shape[0], -1)

        all_embeddings.append(emb_flat)
        all_labels.extend([method_name] * len(emb_flat))

    if len(all_embeddings) == 0:
        print("No embeddings to analyze")
        return

    all_embeddings = np.vstack(all_embeddings)

    # Standardize
    scaler = StandardScaler()
    all_embeddings_scaled = scaler.fit_transform(all_embeddings)

    # UMAP projection
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(all_embeddings_scaled)

    # Plot UMAP
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_methods = sorted(list(embeddings_dict.keys()))

    if all_labels_order:
        # Use consistent palette based on provided order
        palette = sns.color_palette("Set2", len(all_labels_order))
        color_map = dict(zip(all_labels_order, palette))
    else:
        # Fallback to local palette
        palette = sns.color_palette("Set2", len(unique_methods))
        color_map = dict(zip(unique_methods, palette))

    for i, method in enumerate(unique_methods):
        mask = np.array(all_labels) == method
        color = color_map.get(method, (0.5, 0.5, 0.5))  # Default to gray if not found
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[color],
            label=method,
            alpha=0.6,
            s=50,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Projection of Embeddings by Method")
    ax.legend()
    ax.grid(True, alpha=0.3)

    umap_path = os.path.join(output_dir, "umap_embeddings.png")
    plt.savefig(umap_path, dpi=300, bbox_inches="tight")
    print(f"UMAP plot saved to {umap_path}")
    plt.close()

    # Distribution plots
    fig, axes = plt.subplots(1, len(unique_methods), figsize=(5 * len(unique_methods), 4))
    if len(unique_methods) == 1:
        axes = [axes]

    for i, method in enumerate(unique_methods):
        emb = embeddings_dict[method]
        if emb is not None and len(emb) > 0:
            emb_flat = emb.flatten()
            axes[i].hist(emb_flat, bins=50, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"{method}\nMean: {emb_flat.mean():.3f}, Std: {emb_flat.std():.3f}")
            axes[i].set_xlabel("Embedding Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(output_dir, "embedding_distributions.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    print(f"Distribution plot saved to {dist_path}")
    plt.close()

    # Statistics table
    stats_data = []
    for method in unique_methods:
        emb = embeddings_dict[method]
        if emb is not None and len(emb) > 0:
            emb_flat = emb.flatten()
            stats_data.append(
                {
                    "method": method,
                    "mean": emb_flat.mean(),
                    "std": emb_flat.std(),
                    "min": emb_flat.min(),
                    "max": emb_flat.max(),
                    "shape": str(emb.shape),
                }
            )

    stats_df = pd.DataFrame(stats_data)
    stats_path = os.path.join(output_dir, "embedding_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Statistics saved to {stats_path}")
    print("\n=== Embedding Statistics ===")
    print(stats_df.to_string(index=False))


def evaluate_task(
    task_id,
    grouping,
    model,
    device,
    openml_task,
    res_df,
    duplicate_features=1,
    inject_masks=False,
    extract_embeddings=False,
    analysis_type="grouping",
):
    """
    Evaluates a single task with specific grouping and feature duplication settings.

    Returns:
        res_df: Updated results dataframe
        embeddings: Extracted embeddings (if extract_embeddings=True), else None
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
        assert not inject_masks, "Cannot duplicate features and inject masks simultaneously"
        X = np.repeat(X, duplicate_features, axis=1)

    # Handle mask injection
    # e.g. grouping=2 -> [f1, mask], [f2, mask]
    # e.g. grouping=3 -> [f1, mask, mask], [f2, mask, mask]
    if inject_masks:
        assert duplicate_features == 1, "Cannot inject masks with feature duplication"
        if grouping > 1:
            n_samples, n_features = X.shape
            X_new = np.full((n_samples, n_features * grouping), np.nan, dtype=X.dtype)
            X_new[:, 0::grouping] = X
            X = X_new

    # Assertions for validity
    assert len(X) == len(y), "X and y must have same number of instances"
    assert duplicate_features >= 1, "Duplicate features must be >= 1"

    num_features = X.shape[1]
    num_instances = X.shape[0]
    num_classes = len(np.unique(y))
    dataset_name = dataset.name

    print(
        f"\nTask: {task_id} (Dataset: {dataset_name}) - Grouping: {grouping}, Dup: {duplicate_features}, Masking: {inject_masks}"
    )
    print(f"  Features: {num_features}, Instances: {num_instances}, Classes: {num_classes}")

    # Shuffle
    X, y = shuffle_arrays(X, y, random_state=42)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    clf = TabPFNClassifier(device=device, n_estimators=1, ignore_pretraining_limits=True)

    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)

    model.features_per_group = grouping

    embeddings_collected = [] if extract_embeddings else None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            if extract_embeddings and fold == 0:  # Only extract for first fold
                embeddings, pred_probs = extract_embeddings_with_model(
                    model, X_train, y_train, X_test, clf, device
                )
                if embeddings is not None:
                    embeddings_collected.append(embeddings)
            else:
                clf.fit(X_train, y_train, model=model)
                pred_probs = clf.predict_proba(X_test)

            # Verify predictions shape
            assert pred_probs.shape[0] == len(
                y_test
            ), f"Predictions shape mismatch: {pred_probs.shape} vs {len(y_test)}"

            pred_res = PredictionResults(y_test, pred_probs)
            report = pred_res.get_classification_report(print_report=False)
            accuracy = report["accuracy"]
            f1_weighted = pred_res.get_f1_score(average="weighted")

            if pred_probs.shape[-1] == 2:
                roc_auc = roc_auc_score(pred_res.ground_truth, pred_res.prediction_probas[:, 1])
            else:
                roc_auc = roc_auc_score(
                    pred_res.ground_truth,
                    pred_res.prediction_probas,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(pred_probs.shape[-1]),
                )

            row_data = {
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
                "mask_injected": [inject_masks],
                "analysis_type": [analysis_type],
            }

            new_row = pd.DataFrame(row_data)
            res_df = pd.concat([res_df, new_row], ignore_index=True)
        except Exception as e:
            print(f"Skipping fold {fold} for task {task_id} due to error: {e}")
            continue

    final_embeddings = embeddings_collected[0] if embeddings_collected else None
    return res_df, final_embeddings


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
    masking_output_file=None,
    extract_embeddings=False,
    tasks_limit=None,
):
    """
    Benchmark TabPFN base model.
    """
    # Fetch OpenML suite
    suite = openml.study.get_suite(suite_id=suite_id)
    openml_df = tasks.list_tasks(output_format="dataframe", task_id=suite.tasks)
    print(f"Loaded tasks dataframe shape: {openml_df.shape}")

    assert not openml_df.empty, f"No tasks found for suite {suite_id}"

    if "task_type" in openml_df.columns:
        openml_df = openml_df[openml_df["task_type"] == "Supervised Classification"]

    openml_df = openml_df[
        (openml_df["NumberOfFeatures"] >= min_features)
        & (openml_df["NumberOfFeatures"] <= max_features)
        & (openml_df["NumberOfInstances"] <= max_instances)
        & (openml_df["NumberOfClasses"] < 10)
        & (openml_df["NumberOfClasses"] > 1)
    ]

    if tasks_limit is not None:
        print(f"Limiting to {tasks_limit} tasks")
        openml_df = openml_df.head(tasks_limit)

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

    all_results_dfs = []

    # --- Masking Benchmark ---
    print(f"\nRunning Masking Benchmark")

    mask_output = masking_output_file or output_file.replace(".csv", "_masking.csv")

    if os.path.exists(mask_output):
        mask_df = pd.read_csv(mask_output)
    else:
        mask_df = pd.DataFrame(columns=RESULT_COLUMNS)

    masking_grouping_values = [1, 2, 3]

    for grouping in masking_grouping_values:
        print(f"\nTraining with Masking, Grouping={grouping}")

        for task_id in openml_df["tid"].values:
            task_id = int(task_id)
            if (
                not mask_df.empty
                and (
                    (mask_df["task_id"] == task_id)
                    & (mask_df["features_per_group"] == grouping)
                    & (mask_df.get("mask_injected", True) == True)
                ).any()
            ):
                continue

            t = openml.tasks.get_task(task_id)
            mask_df, _ = evaluate_task(
                task_id,
                grouping,
                model,
                device,
                t,
                mask_df,
                duplicate_features=1,
                inject_masks=True,
                analysis_type="masking",
            )

            os.makedirs(os.path.dirname(os.path.abspath(mask_output)) or ".", exist_ok=True)
            mask_df.to_csv(mask_output, index=False)

    print(f"Masking results saved to {mask_output}")
    all_results_dfs.append(mask_df)

    # --- Standard Grouping Benchmark ---
    print(f"\nRunning Standard Grouping Benchmark")
    print(f"Testing features_per_group values: {grouping_values}")

    if os.path.exists(output_file):
        res_df = pd.read_csv(output_file)
        if "duplicate_factor" not in res_df.columns:
            res_df["duplicate_factor"] = 1
        if "mask_injected" not in res_df.columns:
            res_df["mask_injected"] = False
    else:
        res_df = pd.DataFrame(columns=RESULT_COLUMNS)

    for grouping in grouping_values:
        print(f"\nProcessing Grouping: {grouping}")
        for task_id in openml_df["tid"].values:
            task_id = int(task_id)

            exists = False
            if not res_df.empty:
                cond = (res_df["task_id"] == task_id) & (res_df["features_per_group"] == grouping)
                if "duplicate_factor" in res_df.columns:
                    cond &= res_df["duplicate_factor"] == 1
                if "mask_injected" in res_df.columns:
                    cond &= res_df["mask_injected"] == False
                if cond.any():
                    exists = True

            if exists:
                continue

            t = openml.tasks.get_task(task_id)
            res_df, _ = evaluate_task(
                task_id,
                grouping,
                model,
                device,
                t,
                res_df,
                duplicate_features=1,
                inject_masks=False,
                analysis_type="grouping",
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
            res_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")
    all_results_dfs.append(res_df)

    # Combine all results for plotting and printing
    all_labels_order = None
    if all_results_dfs:
        final_df = pd.concat(all_results_dfs, ignore_index=True)
        print_all_results(final_df)

        if generate_plot:
            plot_output = output_file.replace(".csv", "_combined_plot.png")
            plot_combined_results(final_df, plot_output)

        # Get all unique labels for consistent coloring
        if "combo_label" not in final_df.columns:
            final_df["combo_label"] = final_df.apply(get_combo_label, axis=1)
        all_labels_order = sorted(final_df["combo_label"].unique())

    # --- Embedding Extraction and Analysis ---
    if extract_embeddings:
        print(f"\n=== Extracting Embeddings for Analysis ===")

        embedding_output_dir = os.path.join(
            os.path.dirname(os.path.abspath(output_file)), "embeddings"
        )
        os.makedirs(embedding_output_dir, exist_ok=True)

        # Select subset of tasks for embedding analysis
        embedding_tasks = openml_df["tid"].values
        print(f"Analyzing embeddings for {len(embedding_tasks)} tasks")

        grouping_for_comparison = 2  # Use grouping=2 for all methods

        embeddings_dict = {}

        for task_id in embedding_tasks:
            task_id = int(task_id)
            t = openml.tasks.get_task(task_id)

            print(f"\n--- Extracting embeddings for task {task_id} ---")

            # Direct method (no duplication, no masking)
            print("  Method: Direct")
            _, emb_direct = evaluate_task(
                task_id,
                grouping_for_comparison,
                model,
                device,
                t,
                pd.DataFrame(columns=RESULT_COLUMNS),
                duplicate_features=1,
                inject_masks=False,
                extract_embeddings=True,
            )
            if emb_direct is not None:
                key = f"t{task_id}_direct"
                embeddings_dict[key] = emb_direct

            # Masking method
            print("  Method: Masking")
            _, emb_mask = evaluate_task(
                task_id,
                grouping_for_comparison,
                model,
                device,
                t,
                pd.DataFrame(columns=RESULT_COLUMNS),
                duplicate_features=1,
                inject_masks=True,
                extract_embeddings=True,
            )
            if emb_mask is not None:
                key = f"t{task_id}_masking"
                embeddings_dict[key] = emb_mask

        # Analyze
        if len(embeddings_dict) > 0:
            # Group by method type for visualization
            # Use labels consistent with boxplot

            # Construct labels for the methods we extracted
            # Direct: Grouping | grp=2 | dup=1 | no-mask
            direct_label = get_combo_label(
                {
                    "analysis_type": "grouping",
                    "features_per_group": grouping_for_comparison,
                    "duplicate_factor": 1,
                    "mask_injected": False,
                }
            )

            # Masking: Masking | grp=2 | dup=1 | mask
            masking_label = get_combo_label(
                {
                    "analysis_type": "masking",
                    "features_per_group": grouping_for_comparison,
                    "duplicate_factor": 1,
                    "mask_injected": True,
                }
            )

            method_embeddings = {direct_label: [], masking_label: []}

            for key, emb in embeddings_dict.items():
                if "direct" in key:
                    method_embeddings[direct_label].append(emb)
                elif "masking" in key:
                    method_embeddings[masking_label].append(emb)

            # Concatenate embeddings for each method
            method_embeddings_concat = {}
            for method, emb_list in method_embeddings.items():
                if len(emb_list) > 0:
                    # Handle different shapes
                    try:
                        method_embeddings_concat[method] = np.concatenate(emb_list, axis=0)
                    except:
                        # If shapes differ, take the first one
                        method_embeddings_concat[method] = emb_list[0]

            analyze_embeddings(method_embeddings_concat, embedding_output_dir, all_labels_order)
        else:
            print("No embeddings were extracted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_id", type=int, help="OpenML suite ID")
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--min_features", type=int, default=0)
    parser.add_argument("--max_instances", type=int, default=10000)
    parser.add_argument(
        "--output_file", type=str, default="analysis_results/grouping_benchmark_results.csv"
    )
    parser.add_argument("--grouping_values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument(
        "--duplication_output_file",
        type=str,
        default="analysis_results/duplication_benchmark_results.csv",
    )
    parser.add_argument(
        "--masking_output_file", type=str, default="analysis_results/masking_benchmark_results.csv"
    )
    parser.add_argument(
        "--extract_embeddings", action="store_true", help="Extract and analyze embeddings"
    )
    parser.add_argument(
        "--tasks_limit",
        type=int,
        default=20,
        help="Number of tasks to analyze",
    )

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
        duplication_output_file=args.duplication_output_file,
        masking_output_file=args.masking_output_file,
        extract_embeddings=args.extract_embeddings,
        tasks_limit=args.tasks_limit,
    )
