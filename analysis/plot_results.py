import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

import argparse

# Set style for scientific plotting
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def save_plots(fig, output_dir, filename_prefix):
    """Save figure as PDF and PNG."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{filename_prefix}.pdf")
    png_path = os.path.join(output_dir, f"{filename_prefix}.png")

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved plots:\n  {pdf_path}\n  {png_path}")
    plt.close(fig)


def plot_metric_vs_categorical(
    df, x_col, hue_col, metric, output_dir, basename, xlabel=None, ylabel=None, title=None
):
    """Generic boxplot for Metric vs Categorical (e.g. Dataset) grouped by Hue (e.g. Checkpoint)."""
    plt.figure(figsize=(10, 6))

    # Check number of categories
    unique_x = df[x_col].nunique()
    if unique_x > 40:
        # If too many items, fall back to aggregated plot grouped by hue
        print(f"Comparison has {unique_x} categories for {x_col}, switching to aggregated view.")

        # Sort hue_col by median metric
        hue_order = df.groupby(hue_col)[metric].median().sort_values(ascending=False).index

        sns.boxplot(data=df, x=hue_col, y=metric, order=hue_order, palette="tab10")
        plt.xlabel(hue_col.replace("_", " ").title() if not xlabel else xlabel)
        plt.ylabel(ylabel if ylabel else metric.replace("_", " ").title())
        plt.title(f"{title if title else basename} - Aggregated {metric}")
        plt.ylim(0, 1.05)
        save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_aggregated_by_{hue_col}")
        return

    # Standard detailed plot
    # Sort x_col by median metric
    order = df.groupby(x_col)[metric].median().sort_values(ascending=False).index

    sns.boxplot(data=df, x=x_col, y=metric, hue=hue_col, order=order, palette="tab10")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.xlabel(xlabel if xlabel else x_col.replace("_", " ").title())
    plt.ylabel(ylabel if ylabel else metric.replace("_", " ").title())
    plt.title(title if title else f"{basename} - {metric} per {x_col}")

    # Adjust legend
    if df[hue_col].nunique() > 1:
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", title=hue_col.replace("_", " ").title()
        )

    save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_per_{x_col}")


def plot_hdlss(df, output_dir, basename):
    """Plotting logic for HDLSS benchmarks."""
    output_dir = os.path.join(output_dir, "hdlss")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]

    # Clean up checkpoint names if needed
    if "checkpoint" in df.columns:
        df["checkpoint"] = df["checkpoint"].apply(
            lambda x: str(x).split("/")[-1].replace(".pt", "")
        )

    for metric in metrics:
        # Plot 1: Per Dataset Comparison (Grouped Boxplot)
        plot_metric_vs_categorical(
            df,
            x_col="dataset_name",
            hue_col="checkpoint",
            metric=metric,
            output_dir=output_dir,
            basename=basename,
            xlabel="Dataset",
            ylabel=metric.replace("_", " ").title(),
        )

        # Plot 2: Aggregated Summary (if more than 1 dataset)
        if df["dataset_name"].nunique() > 1 and df["checkpoint"].nunique() > 1:
            plt.figure(figsize=(8, 6))
            order = df.groupby("checkpoint")[metric].mean().sort_values(ascending=False).index
            sns.barplot(data=df, x="checkpoint", y=metric, errorbar="sd", capsize=0.1, order=order, palette="tab10")
            plt.ylim(0, 1.05)
            plt.title(f"Aggregated {metric.replace('_', ' ').title()} - {basename}")
            plt.xlabel("Checkpoint")
            plt.ylabel(metric.replace("_", " ").title())
            plt.xticks(rotation=45, ha="right")
            save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_overall_bar")

        # Plot 3: One plot per dataset (Comparison of checkpoints)
        # Only do this if we have multiple checkpoints to compare
        if df["checkpoint"].nunique() > 1:
            per_dataset_dir = os.path.join(output_dir, "per_dataset_plots")
            os.makedirs(per_dataset_dir, exist_ok=True)

            datasets = df["dataset_name"].unique()
            for ds in datasets:
                ds_df = df[df["dataset_name"] == ds]
                if ds_df.empty:
                    continue

                plt.figure(figsize=(8, 6))
                order = sorted(ds_df["checkpoint"].unique())
                sns.barplot(data=ds_df, x="checkpoint", y=metric, order=order, palette="tab10")
                plt.ylim(0, 1.05)
                plt.title(f"{ds} - {metric.replace('_', ' ').title()}")
                plt.xlabel("Checkpoint")
                plt.ylabel(metric.replace("_", " ").title())
                plt.xticks(rotation=45, ha="right")
                save_plots(plt.gcf(), per_dataset_dir, f"{basename}_{ds}_{metric}_comparison")


def plot_openml(df, output_dir, basename):
    """Plotting logic for OpenML benchmarks."""
    output_dir = os.path.join(output_dir, "openml")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]

    if "checkpoint" in df.columns:
        df["checkpoint"] = df["checkpoint"].apply(
            lambda x: str(x).split("/")[-1].replace(".pt", "")
        )

    # Use task_id as categorical
    df["Task"] = "Task " + df["task_id"].astype(str)

    for metric in metrics:
        plot_metric_vs_categorical(
            df,
            x_col="Task",
            hue_col="checkpoint",
            metric=metric,
            output_dir=output_dir,
            basename=basename,
        )

        # Aggregated
        if df["task_id"].nunique() > 1 and df["checkpoint"].nunique() > 1:
            plt.figure(figsize=(8, 6))
            order = df.groupby("checkpoint")[metric].mean().sort_values(ascending=False).index
            sns.barplot(data=df, x="checkpoint", y=metric, errorbar="sd", capsize=0.1, order=order, palette="tab10")
            plt.ylim(0, 1.05)
            plt.title(f"Aggregated {metric.replace('_', ' ').title()} - {basename}")
            plt.xlabel("Checkpoint")
            plt.ylabel(metric.replace("_", " ").title())
            plt.xticks(rotation=45, ha="right")
            save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_overall_bar")

        # Plot 3: One plot per task
        if df["checkpoint"].nunique() > 1:
            per_task_dir = os.path.join(output_dir, "per_task_plots")
            os.makedirs(per_task_dir, exist_ok=True)

            tasks = df["task_id"].unique()
            for task in tasks:
                task_df = df[df["task_id"] == task]
                if task_df.empty:
                    continue

                plt.figure(figsize=(8, 6))
                order = sorted(task_df["checkpoint"].unique())
                sns.barplot(data=task_df, x="checkpoint", y=metric, order=order, palette="tab10")
                plt.ylim(0, 1.05)
                plt.title(f"Task {task} - {metric.replace('_', ' ').title()}")
                plt.xlabel("Checkpoint")
                plt.ylabel(metric.replace("_", " ").title())
                plt.xticks(rotation=45, ha="right")
                save_plots(plt.gcf(), per_task_dir, f"{basename}_task_{task}_{metric}_comparison")


def plot_grouping(df, output_dir, basename):
    """Plotting logic for Grouping benchmarks."""
    output_dir = os.path.join(output_dir, "grouping")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        # Group by features_per_group
        sns.boxplot(data=df, x="features_per_group", y=metric)
        plt.ylim(0, 1.05)
        plt.title(f"Impact of Grouping on {metric.replace('_', ' ').title()}")
        plt.xlabel("Features Per Group")
        plt.ylabel(metric.replace("_", " ").title())
        save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_grouping")


def plot_multiomics(df, output_dir, basename):
    """Plotting logic for Multiomics Feature Reduction."""
    output_dir = os.path.join(output_dir, "multiomics")
    # Line plot: x=n_features, y=metric, hue=checkpoint
    metrics = [c for c in ["accuracy"] if c in df.columns]

    if "checkpoint" in df.columns:
        df["checkpoint"] = df["checkpoint"].apply(
            lambda x: str(x).split("/")[-1].replace(".pt", "")
        )

    for metric in metrics:
        # Separate plot per Dataset if multiple
        datasets = df["dataset_name"].unique()
        for ds in datasets:
            ds_df = df[df["dataset_name"] == ds]
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=ds_df,
                x="n_features",
                y=metric,
                style="checkpoint",
                markers=True,
                palette="tab10",
                err_kws={"alpha": 0.1},
            )
            plt.ylim(0, 1.05)
            plt.title(f"{ds} - {metric.replace('_', ' ').title()} vs Feature Count")
            plt.xlabel("Number of Features")
            plt.ylabel(metric.replace("_", " ").title())
            feats = sorted(ds_df["n_features"].unique())
            if 0 in feats:
                # Move 0 to the end as "All"
                pass

            save_plots(plt.gcf(), output_dir, f"{basename}_{ds}_{metric}_feature_curve")


def plot_widening(df, output_dir, basename):
    """Plotting logic for OpenML Widening."""
    output_dir = os.path.join(output_dir, "widening")
    # Line plot: x=features_added, y=metric, hue=checkpoint
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]

    if "checkpoint" in df.columns:
        df["checkpoint"] = df["checkpoint"].apply(
            lambda x: str(x).split("/")[-1].replace(".pt", "")
        )

    for metric in metrics:
        # Separate plot per Dataset ID if multiple
        if "dataset_id" in df.columns:
            datasets = df["dataset_id"].unique()
            for ds in datasets:
                ds_df = df[df["dataset_id"] == ds]
                plt.figure(figsize=(10, 6))
                sns.lineplot(
                    data=ds_df,
                    x="features_added",
                    y=metric,
                    hue="checkpoint",
                    style="checkpoint",
                    markers=True,
                    palette="tab10",
                    err_kws={"alpha": 0.1},
                )
                plt.ylim(0, 1.05)
                plt.title(f"Dataset {ds} - {metric} vs Features Added")
                plt.xlabel("Features Added")
                plt.ylabel(metric.replace("_", " ").title())
                save_plots(plt.gcf(), output_dir, f"{basename}_{ds}_{metric}_widening_curve")
        else:
            # Fallback if no dataset_id column
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df,
                x="features_added",
                y=metric,
                hue="checkpoint",
                style="checkpoint",
                markers=True,
                palette="tab10",
                err_kws={"alpha": 0.1},
            )
            plt.ylim(0, 1.05)
            plt.title(f"{basename} - {metric} vs Features Added")
            plt.xlabel("Features Added")
            plt.ylabel(metric.replace("_", " ").title())
            save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_widening_curve")


def plot_tabarena(df, output_dir, basename):
    """
    Scatter plot comparing two models (e.g., TabPFN-v2 vs TabPFN-Wide) 
    using 'roc_auc_score' from OpenML benchmark results.
    Ref: snp_analysis/tabarena_vis.ipynb
    """
    output_dir = os.path.join(output_dir, "tabarena")
    os.makedirs(output_dir, exist_ok=True)
    
    # We expect 'task_id', 'checkpoint', 'roc_auc_score'
    if not all(col in df.columns for col in ["task_id", "checkpoint", "roc_auc_score"]):
        print("   [TabArena] Missing columns for TabArena plot. Skipping.")
        return

    # Aggregate if multiple entries per task/checkpoint (e.g. folds)
    # Ensure task_id is numeric to handle int/str mismatch
    df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce")
    df = df.dropna(subset=["task_id"])
    
    df_agg = (
        df.groupby(["task_id", "checkpoint"])["roc_auc_score"]
        .mean()
        .reset_index()
    )
    
    # Pivot to have checkpoints as columns
    df_pivot = df_agg.pivot(index='task_id', columns='checkpoint', values='roc_auc_score')
    
    # Needs at least 2 columns to compare
    if df_pivot.shape[1] < 2:
        print("   [TabArena] Need at least 2 checkpoints to compare. Skipping.")
        return
        
    cols = df_pivot.columns
    # Simple pairwise comparison: Compare the last column vs the first column 
    # (assuming sorted order, often baseline vs new model)
    # OR compare all against the first one.
    
    baseline = cols[0]
    others = cols[1:]
    
    import scipy.stats as stats
    
    for other in others:
        df_xy = df_pivot[[baseline, other]].dropna()
        if df_xy.empty:
            continue
            
        x_vals = df_xy[baseline]
        y_vals = df_xy[other]
        
        rho, _ = stats.spearmanr(x_vals, y_vals)
        
        plt.figure(figsize=(6, 6))
        plt.scatter(x_vals, y_vals, marker="x", s=100, linewidths=1.5, color="mediumblue")
        
        # Plot diagonal
        # lims = [min(x_vals.min(), y_vals.min()), max(x_vals.max(), y_vals.max())]
        lims = [0.5, 1.0] # Fixed range often better for AUC
        plt.plot(lims, lims, "--", color="gray", linewidth=1.5)
        
        plt.xlim(lims)
        plt.ylim(lims)
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.xlabel(f"{baseline}", fontsize=12)
        plt.ylabel(f"{other}", fontsize=12)
        plt.title(f"TabArena Style Comparison (Spearman rho={rho:.4f})")
        
        save_plots(plt.gcf(), output_dir, f"tabarena_scatter_{baseline}_vs_{other}")


def plot_snp(df, output_dir, basename):
    """Plotting logic for SNP benchmarks."""
    output_dir = os.path.join(output_dir, "snp")
    # Make sure n_features is numeric
    if "n_features" in df.columns:
        df["n_features"] = pd.to_numeric(df["n_features"])
    
    # Use checkpoint column for model name if available (aggregated), else 'model'
    hue_col = "checkpoint" if "checkpoint" in df.columns else "model"
    
    # Clean up checkpoint names
    if hue_col in df.columns:
        df[hue_col] = df[hue_col].apply(
            lambda x: str(x).split("/")[-1].replace(".pt", "")
        )

    # Metrics to plot
    metrics = [c for c in ["roc_auc", "accuracy"] if c in df.columns]
    
    for metric in metrics:
        # 1. Faceted Plot by Polygenicity
        if "Polygenicity" in df.columns:
            plt.figure(figsize=(10, 6))
            g = sns.FacetGrid(df, col="Polygenicity", sharey=False, height=5, aspect=1.2)
            g.map_dataframe(
                sns.lineplot, 
                x="n_features", 
                y=metric, 
                hue=hue_col, 
                style=hue_col, 
                markers=True,
                palette="tab10",
                err_kws={"alpha": 0.1}
            )
            g.add_legend()
            g.set_axis_labels("Number of Features", metric.replace("_", " ").title())
            g.set_titles(col_template="Polygenicity: {col_name}")
            save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_by_polygenicity")
        else:
            # Fallback simple plot
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df,
                x="n_features",
                y=metric,
                hue=hue_col,
                style=hue_col, 
                markers=True,
                palette="tab10",
            )
            plt.title(f"{basename} - {metric.replace('_', ' ').title()}")
            plt.xlabel("Number of Features")
            plt.ylabel(metric.replace("_", " ").title())
            save_plots(plt.gcf(), output_dir, f"{basename}_{metric}")


def clean_basename(basename):
    for suffix in ["_benchmark_results", "_results", "_benchmark"]:
        basename = basename.replace(suffix, "")
    return basename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default=None, help="Directory containing results CSVs"
    )
    parser.add_argument(
        "--compare_mode",
        action="store_true",
        help="Recursively find and aggregate results for comparison",
    )
    args = parser.parse_args()

    cwd = os.getcwd()
    base_results_dir = args.input_dir if args.input_dir else os.path.join(cwd, "analysis_results")

    if not os.path.exists(base_results_dir):
        print(f"Directory '{base_results_dir}' not found. Skipping plotting.")
        return

    if args.compare_mode:
        print(f"Running in COMPARISON MODE. Scanning {base_results_dir} recursively...")
        # Find all CSVs recursively
        all_csvs = glob.glob(os.path.join(base_results_dir, "**/*.csv"), recursive=True)

        # Group by filename (e.g. hdlss_benchmark_results.csv)
        grouped_csvs = {}
        for csv_file in all_csvs:
            filename = os.path.basename(csv_file)
            if filename not in grouped_csvs:
                grouped_csvs[filename] = []
            grouped_csvs[filename].append(csv_file)

        # Process each group
        for filename, file_list in grouped_csvs.items():
            print(f"Aggregating {len(file_list)} files for {filename}...")
            dfs = []
            for f in file_list:
                temp_df = pd.read_csv(f)
                if temp_df.empty:
                    continue

                # Use parent folder name as checkpoint label
                parent_folder = os.path.dirname(f)
                folder_name = os.path.basename(parent_folder)
                # If the folder name is "openml_widening", go up one level
                if folder_name == "openml_widening":
                    folder_name = os.path.basename(os.path.dirname(parent_folder))

                if "checkpoint" in temp_df.columns:
                    temp_df["checkpoint"] = folder_name
                dfs.append(temp_df)

            if not dfs:
                continue

            combined_df = pd.concat(dfs, ignore_index=True)

            # Plotting
            basename = os.path.splitext(filename)[0]
            basename = clean_basename(basename)
            output_dir = os.path.join(base_results_dir, "comparison_plots")

            # Special handling for OpenML Widening which has numeric filenames
            if filename.replace(".csv", "").isdigit():
                # This is likely an OpenML widening file
                basename = "openml_widening"
                plot_widening(
                    combined_df, output_dir, f"widening_dataset_{filename.replace('.csv', '')}"
                )
            else:
                if "multiomics" in basename.lower():
                    plot_multiomics(combined_df, output_dir, basename)
                elif "grouping" in basename.lower():
                    plot_grouping(combined_df, output_dir, basename)
                elif "hdlss" in basename.lower():
                    plot_hdlss(combined_df, output_dir, basename)
                elif "openml" in basename.lower() and "widening" not in basename.lower():
                    plot_openml(combined_df, output_dir, basename)
                    # Also run TabArena plot for OpenML results in comparison mode
                    plot_tabarena(combined_df, output_dir, basename)
                elif "snp" in basename.lower():
                    plot_snp(combined_df, output_dir, basename)
                else:
                    print(f"Skipping {basename}: filename pattern not recognized.")

    else:
        # Single directory mode (original behavior + recursive search for widening)
        csv_files = glob.glob(os.path.join(base_results_dir, "*.csv"))
        # Also look for widening files in subdirectories
        widening_files = glob.glob(os.path.join(base_results_dir, "**/*.csv"), recursive=True)
        # Filter out the ones already in csv_files
        widening_files = [f for f in widening_files if f not in csv_files]

        all_files = csv_files + widening_files

        if not all_files:
            print(f"No CSV files found in {base_results_dir}")
            return

        print(f"Found {len(all_files)} CSV files in {base_results_dir}...")

        for csv_file in all_files:
            basename = os.path.splitext(os.path.basename(csv_file))[0]
            basename = clean_basename(basename)
            print(f"-- Processing {basename} --")

            df = pd.read_csv(csv_file)
            if df.empty:
                print("   File is empty, skipping.")
                continue

            # Dispatch based on filename
            output_dir = os.path.dirname(csv_file)
            if "multiomics" in basename.lower():
                print("   Detected Multiomics Feature Reduction format.")
                plot_multiomics(df, output_dir, basename)
            elif "grouping" in basename.lower():
                print("   Detected Grouping Benchmark format.")
                plot_grouping(df, output_dir, basename)
            elif "hdlss" in basename.lower():
                print("   Detected HDLSS Benchmark format.")
                plot_hdlss(df, output_dir, basename)
            elif "openml" in basename.lower() and "widening" not in basename.lower():
                print("   Detected OpenML Benchmark format.")
                plot_openml(df, output_dir, basename)
                # Also run TabArena plot for OpenML results
                print("   Running TabArena plot...")
                plot_tabarena(df, output_dir, basename)
            elif "snp" in basename.lower():
                print("   Detected SNP Benchmark format.")
                plot_snp(df, output_dir, basename)
            elif basename.isdigit() or "widening" in os.path.dirname(csv_file).lower():
                print("   Detected OpenML Widening format.")
                # For widening, we might want to save plots in the same folder as the CSV
                plot_widening(df, os.path.dirname(csv_file), basename)
            else:
                print(f"   Skipping {basename}: filename pattern not recognized.")


if __name__ == "__main__":
    main()
